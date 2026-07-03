"""
Training loop for the dual-branch model with C1 auxiliary supervision,
cell-to-fragment gating, and C2 evidence fusion.

This release keeps the method-facing training structure and evaluation hooks
while omitting private data, checkpoints, and production automation.
"""
import os
import json
import time
import argparse
import csv
from pathlib import Path
from typing import Any, List

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from config import DUAL_CKPT as DUAL_CKPT_STR
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss, FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou, calc_dice


class EarlyStopping:
    """Minimal save-best early stopping utility for the release copy."""

    def __init__(self, save_dir: str, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_score_save = None
        self.best_score_patience = None
        self.best_score = None
        self.score_max = float("-inf")
        self.checkpoint_path = Path(str(_cfg_get(cfg, ["model", "dual_checkpoint"], Path(save_dir) / "sam_dual.pth")))
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[EarlyStopping] checkpoint: {self.checkpoint_path}")
            print(f"[EarlyStopping] patience={self.patience}, min_delta={self.min_delta}")

    def __call__(self, score: float, model: torch.nn.Module, fabric, epoch: int):
        if self.best_score_save is None:
            self.best_score_save = score
            self.best_score_patience = score
            self.best_score = score
            self.score_max = score
            self.save_checkpoint(score, model, fabric, epoch)
            return
        if score > self.best_score_save:
            self.best_score_save = score
            self.best_score = score
            self.score_max = score
            self.save_checkpoint(score, model, fabric, epoch)
        improvement = score - self.best_score_patience
        if improvement > self.min_delta:
            self.best_score_patience = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score: float, model: torch.nn.Module, fabric, epoch: int):
        if self.verbose:
            print(f"  -> save epoch={epoch}, score={score:.4f} to {self.checkpoint_path}")
        fabric.save(self.checkpoint_path, {"model": model, "epoch": epoch, "score": score})

from medpy import metric
from cbim_aux_modules import dice_loss_2channel, bce_dice_loss, compute_region_dice

torch.set_float32_matmul_precision("high")


def _cfg_get(cfg: Any, keys: List[str], default=None):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict):
            if k not in cur:
                return default
            cur = cur[k]
        else:
            if not hasattr(cur, k):
                return default
            cur = getattr(cur, k)
    return cur


METRIC_CSV_FIELDS = [
    "time", "phase", "epoch", "step",
    "train_focal_loss", "train_dice_loss_ins", "train_sem_loss_2ch_dice",
    "train_boundary_loss", "train_overlap_loss", "train_cooccur_loss",
    "train_cc_overlap_loss", "train_iou_loss", "train_total_loss",
    "lr_sam", "lr_sem", "lr_aux_heads", "lr_apam_fusion", "lr_gate",
    "val_ins_iou", "val_ins_f1", "val_ins_dice", "val_ins_precision", "val_ins_recall",
    "val_sem_dice", "val_sem_precision", "val_sem_recall", "val_sem_f1",
    "val_sem_hd", "val_sem_hd95",
    "val_boundary_bce", "val_overlap_bce", "val_cooccur_loss", "val_cc_overlap_loss",
    "gate_penalty_raw", "gate_penalty_softplus", "gate_penalty_grad",
    "fusion_weight_abs_mean", "fusion_bias_abs", "fusion_scale", "fusion_raw_scale",
    "fusion_delta_abs_mean", "fusion_delta_abs_p95", "fusion_delta_cf_only",
    "fusion_delta_cf_and_cc", "fusion_delta_bg", "fusion_changed_px",
    "region_cf_only", "region_cc_only", "region_cf_and_cc", "region_bg",
]


def _safe_float(value):
    if value is None:
        return ""
    try:
        f = float(value)
    except Exception:
        return ""
    if np.isnan(f) or np.isinf(f):
        return ""
    return f


def _tensor_diag(name: str, tensor: torch.Tensor) -> str:
    try:
        with torch.no_grad():
            t = tensor.detach()
            finite = torch.isfinite(t)
            finite_count = int(finite.sum().item())
            total_count = int(t.numel())
            if finite_count == 0:
                return f"{name}: shape={tuple(t.shape)} finite=0/{total_count}"
            ft = t[finite]
            return (
                f"{name}: shape={tuple(t.shape)} finite={finite_count}/{total_count} "
                f"min={float(ft.min().item()):+.4e} max={float(ft.max().item()):+.4e} "
                f"mean={float(ft.mean().item()):+.4e}"
            )
    except Exception as exc:
        return f"{name}: diag_failed={exc}"


def _append_metrics_csv(path_value, row: dict) -> None:
    if not path_value:
        return
    path = Path(str(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    clean_row = {k: row.get(k, "") for k in METRIC_CSV_FIELDS}
    clean_row["time"] = clean_row.get("time") or time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)


def _resolve_dual_ckpt_path(cfg: Box) -> Path:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None and hasattr(model_cfg, "dual_checkpoint"):
        return Path(str(model_cfg.dual_checkpoint))
    return Path(str(DUAL_CKPT_STR))


def _binary_dice_hd_hd95(pred_b: np.ndarray, gt_b: np.ndarray):
    """Return Dice, Hausdorff distance, and HD95 for binary masks."""
    pred_b = pred_b.astype(np.uint8)
    gt_b = gt_b.astype(np.uint8)
    if pred_b.sum() > 0 and gt_b.sum() > 0:
        dice = metric.binary.dc(pred_b, gt_b)
        hd_full = metric.binary.hd(pred_b, gt_b)
        hd95 = metric.binary.hd95(pred_b, gt_b)
        return float(dice), float(hd_full), float(hd95)
    elif pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0, 0.0, 0.0
    else:
        return 0.0, 500.0, 500.0


def _binary_dice_hd95(pred_b: np.ndarray, gt_b: np.ndarray):
    """Compatibility wrapper around _binary_dice_hd_hd95; returns only dice and hd95."""
    dice, _, hd95 = _binary_dice_hd_hd95(pred_b, gt_b)
    return dice, hd95


def _load_apam_pos_weights(cfg, fabric):
    """
    Prefer pos_weight values from aux_gt/train/apam_gt_stats.json.
    Fall back to config values when the stats file is missing or invalid.
    """
    train_aux_dir = _cfg_get(cfg, ["dataset", "train", "aux_gt_dir"], None)
    fallback_co = float(_cfg_get(cfg, ["model", "apam", "pos_weight_cooccur_fallback"], 20.0))
    fallback_cc = float(_cfg_get(cfg, ["model", "apam", "pos_weight_cc_overlap_fallback"], 5.0))

    if train_aux_dir is None:
        return fallback_co, fallback_cc

    stats_path = os.path.join(train_aux_dir, "apam_gt_stats.json")
    if not os.path.isfile(stats_path):
        fabric.print(
            f"[APAM] {stats_path} missing, using fallback "
            f"pos_weight: cooccur={fallback_co}, cc_overlap={fallback_cc}"
        )
        return fallback_co, fallback_cc

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        pw_co = float(stats.get("pos_weight_cooccurrence", fallback_co))
        pw_cc = float(stats.get("pos_weight_cc_overlap", fallback_cc))
        fabric.print(
            f"[APAM] loaded pos_weight from {stats_path}: "
            f"cooccur={pw_co:.2f}, cc_overlap={pw_cc:.2f}"
        )
        return pw_co, pw_cc
    except Exception as e:
        fabric.print(f"[APAM][WARN] failed to read stats, using fallback: {e}")
        return fallback_co, fallback_cc


def _load_init_checkpoint(model: torch.nn.Module, ckpt_path: str, fabric: L.Fabric):
    """Load an optional compatible initialization checkpoint."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        fabric.print("[INIT] no init checkpoint provided, using SAM weights + random init")
        return

    fabric.print(f"[INIT] loading init checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in ckpt:
                ckpt = ckpt[k]
                break

    # Strip DDP prefixes.
    stripped_sd = {}
    for k, v in ckpt.items():
        name = k
        for prefix in ("module.", "_forward_module."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        stripped_sd[name] = v

    # Filter shape mismatches defensively.
    model_sd = model.state_dict()
    filtered_sd = {}
    shape_mismatch = []
    for k, v in stripped_sd.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            shape_mismatch.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}")
            continue
        filtered_sd[k] = v

    msg = model.load_state_dict(filtered_sd, strict=False)

    fabric.print(f"[INIT] loaded {len(filtered_sd)}/{len(stripped_sd)} tensors")
    if shape_mismatch:
        fabric.print(f"[INIT] shape mismatch entries: {len(shape_mismatch)}")
        for line in shape_mismatch[:10]:
            fabric.print(f"       {line}")
    fabric.print(f"[INIT] missing keys (new modules keep random init): {len(msg.missing_keys)}")
    if msg.missing_keys:
        fabric.print(f"[INIT] missing keys sample: {msg.missing_keys[:10]}")
    fabric.print(f"[INIT] unexpected keys (dropped old modules): {len(msg.unexpected_keys)}")
    if msg.unexpected_keys:
        fabric.print(f"[INIT] unexpected keys sample: {msg.unexpected_keys[:10]}")


# =========================
# Validation
# =========================
def validate(
    fabric: L.Fabric,
    model: Model,
    val_dataloader: DataLoader,
    early_stop: EarlyStopping = None,
    epoch: int = 0,
    pos_weight_cooccur: float = 20.0,
    pos_weight_cc_overlap: float = 5.0,
    metrics_csv: str = "",
):
    model.eval()

    ious = AverageMeter()
    f1_scores = AverageMeter()
    recalls = AverageMeter()
    precisions = AverageMeter()
    dices = AverageMeter()

    sem_dices = AverageMeter()
    sem_f1_scores = AverageMeter()
    sem_recalls = AverageMeter()
    sem_precisions = AverageMeter()
    sem_hds = AverageMeter()
    sem_hd95s = AverageMeter()

    boundary_bces = AverageMeter()
    overlap_bces = AverageMeter()
    cooccur_losses = AverageMeter()
    cc_overlap_losses = AverageMeter()

    region_dices = {
        "cf_only": AverageMeter(),
        "cc_only": AverageMeter(),
        "cf_and_cc": AverageMeter(),
        "bg": AverageMeter(),
    }
    last_aux = None
    last_cooccur_gt = None
    last_cc_gt = None

    with torch.no_grad():
        for _, data in enumerate(val_dataloader):
            if len(data) == 9:
                (images, bboxs, ins_masks, sem_mask, _,
                 boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt) = data
            elif len(data) == 7:
                images, bboxs, ins_masks, sem_mask, _, boundary_gt, overlap_gt = data
                cooccur_gt = None
                cc_overlap_gt = None
            elif len(data) == 5:
                images, bboxs, ins_masks, sem_mask, _ = data
                boundary_gt = overlap_gt = cooccur_gt = cc_overlap_gt = None
            else:
                images, bboxs, ins_masks, sem_mask = data
                boundary_gt = overlap_gt = cooccur_gt = cc_overlap_gt = None

            num_images = images.size(0)
            sem_logits_refined, pred_masks, _, aux_dict = model(images, bboxs, return_aux=True)
            aux_for_diag = dict(aux_dict)
            aux_for_diag["sem_logits_refined"] = sem_logits_refined.detach()

            if boundary_gt is not None:
                bce_b = F.binary_cross_entropy_with_logits(
                    aux_dict["boundary_logits"].squeeze(1), boundary_gt.float(),
                )
                boundary_bces.update(float(bce_b.item()), num_images)
            if overlap_gt is not None:
                bce_o = F.binary_cross_entropy_with_logits(
                    aux_dict["overlap_logits"].squeeze(1), overlap_gt.float(),
                )
                overlap_bces.update(float(bce_o.item()), num_images)

            if cooccur_gt is not None and "cooccur_logits" in aux_dict:
                l_co = bce_dice_loss(
                    aux_dict["cooccur_logits"], cooccur_gt,
                    pos_weight=pos_weight_cooccur,
                )
                cooccur_losses.update(float(l_co.item()), num_images)
            if cc_overlap_gt is not None and "cc_overlap_logits" in aux_dict:
                l_cc = bce_dice_loss(
                    aux_dict["cc_overlap_logits"], cc_overlap_gt,
                    pos_weight=pos_weight_cc_overlap,
                )
                cc_overlap_losses.update(float(l_cc.item()), num_images)

            sem_pred_prob = torch.softmax(sem_logits_refined, dim=1)
            sem_pred_mask = torch.argmax(sem_pred_prob, dim=1)

            sp = sem_pred_mask.cpu().numpy().astype(np.uint8)
            gt = sem_mask.cpu().numpy().astype(np.uint8)
            if sp.ndim == 4:
                sp = sp[:, 0]
            if gt.ndim == 4:
                gt = gt[:, 0]

            for b in range(sp.shape[0]):
                pred_b = sp[b] > 0
                gt_b = gt[b] > 0
                dice_i, hd_i, hd95_i = _binary_dice_hd_hd95(pred_b, gt_b)
                sem_dices.update(dice_i, 1)
                sem_hds.update(hd_i, 1)
                sem_hd95s.update(hd95_i, 1)

                tp = np.logical_and(pred_b, gt_b).sum()
                fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
                fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()
                p = tp / max(tp + fp, 1e-12)
                r = tp / max(tp + fn, 1e-12)
                f1 = 2.0 * p * r / max(p + r, 1e-12)

                sem_precisions.update(p, 1)
                sem_recalls.update(r, 1)
                sem_f1_scores.update(f1, 1)

            cached_aux = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in aux_for_diag.items()
            }
            if cooccur_gt is not None and cc_overlap_gt is not None:
                frag_pred_mask_b = sem_pred_mask.bool()
                frag_gt_b = sem_mask.bool()
                co_gt_b = cooccur_gt.bool()
                cc_gt_b = cc_overlap_gt.bool()
                rd = compute_region_dice(frag_pred_mask_b, frag_gt_b, co_gt_b, cc_gt_b)
                for name, val in rd.items():
                    if val >= 0:
                        region_dices[name].update(val, 1)
                last_aux = cached_aux
                last_cooccur_gt = cooccur_gt.detach()
                last_cc_gt = cc_overlap_gt.detach()
            elif last_aux is None:
                last_aux = cached_aux

            for pred_mask, gt_mask in zip(pred_masks, ins_masks):
                if pred_mask.numel() == 0 and gt_mask.numel() == 0:
                    continue
                if pred_mask.numel() == 0 and gt_mask.numel() > 0:
                    pred_mask = torch.zeros_like(gt_mask, device=images.device)
                batch_stats = smp.metrics.get_stats(
                    pred_mask, gt_mask.int(), mode="binary", threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                batch_recall = smp.metrics.recall(*batch_stats, reduction="micro-imagewise")
                batch_precision = smp.metrics.precision(*batch_stats, reduction="micro-imagewise")
                batch_dice = calc_dice(*batch_stats)
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
                recalls.update(batch_recall, num_images)
                precisions.update(batch_precision, num_images)
                dices.update(batch_dice, num_images)

    pen_raw = -1.0
    pen_soft = -1.0
    pen_grad = None
    try:
        p_param = model.cell_to_fragment_gate.penalty
        pen_raw = float(p_param.item())
        pen_soft = float(F.softplus(p_param).item())
        if p_param.grad is not None:
            pen_grad = float(p_param.grad.item())
    except Exception:
        pass
    pen_grad_str = f"{pen_grad:+.2e}" if pen_grad is not None else "None"

    fusion_info = ""
    fusion_diag = ""
    w_abs_mean = None
    b_abs = None
    scale_val = None
    raw_scale_val = None
    d_mean = None
    d_p95 = None
    d_cf_only = None
    d_cf_cc = None
    d_bg = None
    changed = None
    try:
        if hasattr(model, "apam_fusion") and model.apam_fusion.enable:
            fuse_last = model.apam_fusion.fuse[-1]
            w_abs_mean = float(fuse_last.weight.detach().abs().mean().item())
            b_abs = float(fuse_last.bias.detach().abs().item()) if fuse_last.bias is not None else 0.0
            scale_val = float(model.apam_fusion.gamma.item())
            raw_scale_val = float(model.apam_fusion.raw_scale.detach().item())
            fusion_info = (
                f" | FusionW(abs_mean={w_abs_mean:.4f}, b_abs={b_abs:.4f}, "
                f"scale={scale_val:.4f}, raw_scale={raw_scale_val:+.4f})"
            )

            if last_aux is not None and last_cooccur_gt is not None and last_cc_gt is not None:
                frag_ref = last_aux["frag_logits_refined"]
                frag_gat = last_aux["frag_logits_gated"]
                delta = (frag_ref - frag_gat).abs()
                delta_flat = delta.flatten()
                d_mean = float(delta_flat.mean().item())
                d_p95 = float(torch.quantile(delta_flat, 0.95).item())

                co_gt_b = last_cooccur_gt.bool()
                cc_gt_b = last_cc_gt.bool()
                cf_only = co_gt_b & (~cc_gt_b)
                cf_and_cc = co_gt_b & cc_gt_b
                bg = (~co_gt_b) & (~cc_gt_b)

                def _region_mean(d_chw, mask_bhw):
                    d_bhw = d_chw.squeeze(1)
                    mask = mask_bhw.to(d_bhw.device)
                    if mask.sum().item() == 0:
                        return 0.0
                    return float(d_bhw[mask].mean().item())

                d_cf_only = _region_mean(delta, cf_only)
                d_cf_cc = _region_mean(delta, cf_and_cc)
                d_bg = _region_mean(delta, bg)

                pred_old = torch.argmax(torch.softmax(last_aux["sem_logits_gated"], dim=1), dim=1)
                pred_new = torch.argmax(torch.softmax(last_aux["sem_logits_refined"], dim=1), dim=1)
                changed = float((pred_old != pred_new).float().mean().item())
                fusion_diag = (
                    f"[FusionScaleDiag Ep{epoch}] scale={scale_val:.4f} (raw={raw_scale_val:+.4f}) | "
                    f"|delta|mean={d_mean:.4f} p95={d_p95:.4f} | "
                    f"cf_only={d_cf_only:.4f} cf_and_cc={d_cf_cc:.4f} bg={d_bg:.4f} | "
                    f"ChangedPx={changed * 100:.2f}%"
                )
            else:
                fusion_diag = (
                    f"[FusionScaleDiag Ep{epoch}] scale={scale_val:.4f} (raw={raw_scale_val:+.4f}) "
                    f"(no APAM region batch cached)"
                )
    except Exception:
        pass

    fabric.print(
        f"Validation [{epoch}]: "
        f"InsIoU:[{ious.avg:.4f}] InsF1:[{f1_scores.avg:.4f}] InsDice:[{dices.avg:.4f}]  "
        f"SemDice:[{sem_dices.avg:.4f}] SemP:[{sem_precisions.avg:.4f}] SemR:[{sem_recalls.avg:.4f}] "
        f"SemF1:[{sem_f1_scores.avg:.4f}] SemHD:[{sem_hds.avg:.4f}] SemHD95:[{sem_hd95s.avg:.4f}] "
        f"BoundaryBCE:[{boundary_bces.avg:.4f}] OverlapBCE:[{overlap_bces.avg:.4f}] "
        f"CoOccurBCE+Dice:[{cooccur_losses.avg:.4f}] CCOverlapBCE+Dice:[{cc_overlap_losses.avg:.4f}] "
        f"GatePen(raw={pen_raw:.3f}, softplus={pen_soft:.3f}, grad={pen_grad_str})"
        f"{fusion_info}"
    )

    if any(rd.count > 0 for rd in region_dices.values()):
        fabric.print(
            f"  [RegionDice Ep{epoch}] "
            f"cf_only={region_dices['cf_only'].avg:.4f} | "
            f"cc_only={region_dices['cc_only'].avg:.4f} | "
            f"cf_and_cc={region_dices['cf_and_cc'].avg:.4f} | "
            f"bg={region_dices['bg'].avg:.4f}"
        )
    if fusion_diag:
        fabric.print(f"  {fusion_diag}")

    tb_metrics = {
        "val/InsIoU": _safe_float(ious.avg),
        "val/InsF1": _safe_float(f1_scores.avg),
        "val/InsDice": _safe_float(dices.avg),
        "val/InsPrecision": _safe_float(precisions.avg),
        "val/InsRecall": _safe_float(recalls.avg),
        "val/SemDice": _safe_float(sem_dices.avg),
        "val/SemPrecision": _safe_float(sem_precisions.avg),
        "val/SemRecall": _safe_float(sem_recalls.avg),
        "val/SemF1": _safe_float(sem_f1_scores.avg),
        "val/SemHD": _safe_float(sem_hds.avg),
        "val/SemHD95": _safe_float(sem_hd95s.avg),
        "val/BoundaryBCE": _safe_float(boundary_bces.avg),
        "val/OverlapBCE": _safe_float(overlap_bces.avg),
        "val/CoOccurBCE_Dice": _safe_float(cooccur_losses.avg),
        "val/CCOverlapBCE_Dice": _safe_float(cc_overlap_losses.avg),
        "gate/PenaltyRaw": _safe_float(pen_raw),
        "gate/PenaltySoftplus": _safe_float(pen_soft),
        "fusion/WeightAbsMean": _safe_float(w_abs_mean),
        "fusion/BiasAbs": _safe_float(b_abs),
        "fusion/Scale": _safe_float(scale_val),
        "fusion/RawScale": _safe_float(raw_scale_val),
        "fusion/DeltaAbsMean": _safe_float(d_mean),
        "fusion/DeltaAbsP95": _safe_float(d_p95),
        "fusion/DeltaCfOnly": _safe_float(d_cf_only),
        "fusion/DeltaCfAndCc": _safe_float(d_cf_cc),
        "fusion/DeltaBg": _safe_float(d_bg),
        "fusion/ChangedPx": _safe_float(changed),
    }
    if pen_grad is not None:
        tb_metrics["gate/PenaltyGrad"] = _safe_float(pen_grad)
    if any(rd.count > 0 for rd in region_dices.values()):
        tb_metrics.update({
            "region/cf_only": _safe_float(region_dices["cf_only"].avg),
            "region/cc_only": _safe_float(region_dices["cc_only"].avg),
            "region/cf_and_cc": _safe_float(region_dices["cf_and_cc"].avg),
            "region/bg": _safe_float(region_dices["bg"].avg),
        })
    tb_metrics = {k: v for k, v in tb_metrics.items() if v != ""}
    if tb_metrics:
        fabric.log_dict(tb_metrics, step=epoch)

    if getattr(fabric, "global_rank", 0) == 0:
        _append_metrics_csv(metrics_csv, {
            "phase": "val",
            "epoch": epoch,
            "step": epoch,
            "val_ins_iou": _safe_float(ious.avg),
            "val_ins_f1": _safe_float(f1_scores.avg),
            "val_ins_dice": _safe_float(dices.avg),
            "val_ins_precision": _safe_float(precisions.avg),
            "val_ins_recall": _safe_float(recalls.avg),
            "val_sem_dice": _safe_float(sem_dices.avg),
            "val_sem_precision": _safe_float(sem_precisions.avg),
            "val_sem_recall": _safe_float(sem_recalls.avg),
            "val_sem_f1": _safe_float(sem_f1_scores.avg),
            "val_sem_hd": _safe_float(sem_hds.avg),
            "val_sem_hd95": _safe_float(sem_hd95s.avg),
            "val_boundary_bce": _safe_float(boundary_bces.avg),
            "val_overlap_bce": _safe_float(overlap_bces.avg),
            "val_cooccur_loss": _safe_float(cooccur_losses.avg),
            "val_cc_overlap_loss": _safe_float(cc_overlap_losses.avg),
            "gate_penalty_raw": _safe_float(pen_raw),
            "gate_penalty_softplus": _safe_float(pen_soft),
            "gate_penalty_grad": _safe_float(pen_grad),
            "fusion_weight_abs_mean": _safe_float(w_abs_mean),
            "fusion_bias_abs": _safe_float(b_abs),
            "fusion_scale": _safe_float(scale_val),
            "fusion_raw_scale": _safe_float(raw_scale_val),
            "fusion_delta_abs_mean": _safe_float(d_mean),
            "fusion_delta_abs_p95": _safe_float(d_p95),
            "fusion_delta_cf_only": _safe_float(d_cf_only),
            "fusion_delta_cf_and_cc": _safe_float(d_cf_cc),
            "fusion_delta_bg": _safe_float(d_bg),
            "fusion_changed_px": _safe_float(changed),
            "region_cf_only": _safe_float(region_dices["cf_only"].avg if region_dices["cf_only"].count > 0 else None),
            "region_cc_only": _safe_float(region_dices["cc_only"].avg if region_dices["cc_only"].count > 0 else None),
            "region_cf_and_cc": _safe_float(region_dices["cf_and_cc"].avg if region_dices["cf_and_cc"].count > 0 else None),
            "region_bg": _safe_float(region_dices["bg"].avg if region_dices["bg"].count > 0 else None),
        })

    model.to("cuda")
    score = sem_dices.avg
    if early_stop is not None:
        early_stop(score, model, fabric, epoch)
    model.train()
# =========================
# Training
# =========================
def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()  # instance branch only

    lambda_boundary = float(_cfg_get(cfg, ["model", "loss_weights", "boundary"], 0.3))
    lambda_overlap = float(_cfg_get(cfg, ["model", "loss_weights", "overlap"], 0.3))
    lambda_cooccur = float(_cfg_get(cfg, ["model", "loss_weights", "cooccur"], 0.3))
    lambda_cc_overlap = float(_cfg_get(cfg, ["model", "loss_weights", "cc_overlap"], 0.3))
    w_sem_dice_2ch = float(_cfg_get(cfg, ["model", "loss_weights", "sem_dice_2ch"], 1.0))
    metrics_csv = str(_cfg_get(cfg, ["metrics_csv"], "") or "")
    tb_log_interval = int(_cfg_get(cfg, ["tb_log_interval"], 200) or 0)

    cooccur_enable = bool(_cfg_get(cfg, ["model", "apam", "cooccur_enable"], True))
    cc_overlap_enable = bool(_cfg_get(cfg, ["model", "apam", "cc_overlap_enable"], True))

    pw_cooccur, pw_cc_overlap = _load_apam_pos_weights(cfg, fabric)

    es_min_delta = float(_cfg_get(cfg, ["opt", "early_stop_min_delta"], 0.0))
    lr_fusion = float(_cfg_get(cfg, ["opt", "learning_rate_fusion"], _cfg_get(cfg, ["opt", "learning_rate_aux"], 1e-4)))
    fusion_scale_init = float(_cfg_get(cfg, ["model", "apam", "fusion_scale_init"], 1.0))

    early_stop = EarlyStopping(
        cfg.out_dir,
        patience=cfg.patience,
        min_delta=es_min_delta,
    )

    fabric.print(
        f"[Config] CBIM: op1_alpha={_cfg_get(cfg, ['model', 'cbim', 'op1_alpha'], 0.0)}, "
        f"op3_enable={_cfg_get(cfg, ['model', 'cbim', 'op3_enable'], True)}, "
        f"detach_gate={_cfg_get(cfg, ['model', 'cbim', 'detach_gate'], True)}"
    )
    fabric.print(
        f"[Config] APAM: cooccur_enable={cooccur_enable}, "
        f"cc_overlap_enable={cc_overlap_enable}, "
        f"fusion_enable={_cfg_get(cfg, ['model', 'apam', 'fusion_enable'], True)}, "
        f"fusion_scale_init={fusion_scale_init}, lr_fusion={lr_fusion}, "
        f"lambda_cooccur={lambda_cooccur}, lambda_cc_overlap={lambda_cc_overlap}, "
        f"pw_cooccur={pw_cooccur:.2f}, pw_cc_overlap={pw_cc_overlap:.2f}"
    )
    fabric.print(f"[Config] DataLoader: batch_size={cfg.batch_size}, num_workers={cfg.num_workers}, tb_log_interval={tb_log_interval}")
    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        sem_losses = AverageMeter()
        iou_losses = AverageMeter()
        boundary_losses = AverageMeter()
        overlap_losses = AverageMeter()
        cooccur_losses_m = AverageMeter()
        cc_overlap_losses_m = AverageMeter()
        total_losses = AverageMeter()

        end = time.time()

        if epoch > 1 and epoch % cfg.eval_interval == 0:
            validate(
                fabric, model, val_dataloader,
                early_stop, epoch,
                pos_weight_cooccur=pw_cooccur,
                pos_weight_cc_overlap=pw_cc_overlap,
                metrics_csv=metrics_csv,
            )
        if early_stop.early_stop:
            fabric.print("--EarlyStop--")
            break

        for it, data in enumerate(train_dataloader):
            data_time.update(time.time() - end)

            if len(data) == 9:
                (images, bboxs, ins_masks, sem_mask, _,
                 boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt) = data
            elif len(data) == 7:
                images, bboxs, ins_masks, sem_mask, _, boundary_gt, overlap_gt = data
                cooccur_gt = None
                cc_overlap_gt = None
            elif len(data) == 5:
                images, bboxs, ins_masks, sem_mask, _ = data
                boundary_gt = overlap_gt = cooccur_gt = cc_overlap_gt = None
            else:
                images, bboxs, ins_masks, sem_mask = data
                boundary_gt = overlap_gt = cooccur_gt = cc_overlap_gt = None

            batch_size = images.size(0)
            sem_target = sem_mask.long()  # [B, H, W] {0, 1}

            sem_logits_refined, pred_masks, iou_predictions, aux_dict = model(
                images, bboxs, return_aux=True,
            )
            num_masks = sum(int(pm.shape[0]) for pm in pred_masks)

            # -------- Instance branch loss --------
            loss_focal = torch.tensor(0.0, device=fabric.device)
            loss_dice_ins = torch.tensor(0.0, device=fabric.device)
            loss_iou = torch.tensor(0.0, device=fabric.device)

            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, ins_masks, iou_predictions):
                if pred_mask.numel() == 0:
                    continue
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask)
                loss_dice_ins += dice_loss(pred_mask, gt_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction="sum")

            if num_masks > 0:
                loss_focal = loss_focal / num_masks
                loss_dice_ins = loss_dice_ins / num_masks
                loss_iou = loss_iou / num_masks

            loss_sem = dice_loss_2channel(sem_logits_refined, sem_target)

            # -------- Boundary / Overlap BCE --------
            loss_boundary = torch.tensor(0.0, device=fabric.device)
            loss_overlap = torch.tensor(0.0, device=fabric.device)
            if boundary_gt is not None:
                loss_boundary = F.binary_cross_entropy_with_logits(
                    aux_dict["boundary_logits"].squeeze(1), boundary_gt.float(),
                )
            if overlap_gt is not None:
                loss_overlap = F.binary_cross_entropy_with_logits(
                    aux_dict["overlap_logits"].squeeze(1), overlap_gt.float(),
                )

            loss_cooccur = torch.tensor(0.0, device=fabric.device)
            loss_cc_overlap = torch.tensor(0.0, device=fabric.device)
            if cooccur_enable and cooccur_gt is not None:
                loss_cooccur = bce_dice_loss(
                    aux_dict["cooccur_logits"], cooccur_gt,
                    pos_weight=pw_cooccur,
                )
            if cc_overlap_enable and cc_overlap_gt is not None:
                loss_cc_overlap = bce_dice_loss(
                    aux_dict["cc_overlap_logits"], cc_overlap_gt,
                    pos_weight=pw_cc_overlap,
                )

            loss_total = (
                loss_focal + loss_dice_ins + loss_iou
                + w_sem_dice_2ch * loss_sem
                + lambda_boundary * loss_boundary
                + lambda_overlap * loss_overlap
                + lambda_cooccur * loss_cooccur
                + lambda_cc_overlap * loss_cc_overlap
            )

            if not torch.isfinite(loss_total):
                diag_lines = [
                    f"[NonFiniteLoss] epoch={epoch} iter={it + 1}/{len(train_dataloader)}",
                    f"  loss_focal={float(loss_focal.detach().item()) if torch.isfinite(loss_focal.detach()) else 'nan'}",
                    f"  loss_dice_ins={float(loss_dice_ins.detach().item()) if torch.isfinite(loss_dice_ins.detach()) else 'nan'}",
                    f"  loss_sem={float(loss_sem.detach().item()) if torch.isfinite(loss_sem.detach()) else 'nan'}",
                    f"  loss_boundary={float(loss_boundary.detach().item()) if torch.isfinite(loss_boundary.detach()) else 'nan'}",
                    f"  loss_overlap={float(loss_overlap.detach().item()) if torch.isfinite(loss_overlap.detach()) else 'nan'}",
                    f"  loss_iou={float(loss_iou.detach().item()) if torch.isfinite(loss_iou.detach()) else 'nan'}",
                    "  " + _tensor_diag("sem_logits_refined", sem_logits_refined),
                    "  " + _tensor_diag("sem_logits_raw", aux_dict.get("sem_logits_raw", sem_logits_refined)),
                    "  " + _tensor_diag("sem_logits_gated", aux_dict.get("sem_logits_gated", sem_logits_refined)),
                    "  " + _tensor_diag("boundary_logits", aux_dict.get("boundary_logits", sem_logits_refined)),
                    "  " + _tensor_diag("overlap_logits", aux_dict.get("overlap_logits", sem_logits_refined)),
                ]
                try:
                    p_param = model.cell_to_fragment_gate.penalty
                    diag_lines.append(
                        f"  gate_penalty_raw={float(p_param.detach().item()):+.6f} "
                        f"softplus={float(F.softplus(p_param.detach()).item()):+.6f}"
                    )
                except Exception as exc:
                    diag_lines.append(f"  gate_penalty_diag_failed={exc}")
                fabric.print("\n".join(diag_lines))
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch}, iter={it + 1}; aborting before optimizer.step"
                )

            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice_ins.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            sem_losses.update(loss_sem.item(), batch_size)
            boundary_losses.update(loss_boundary.item(), batch_size)
            overlap_losses.update(loss_overlap.item(), batch_size)
            cooccur_losses_m.update(loss_cooccur.item(), batch_size)
            cc_overlap_losses_m.update(loss_cc_overlap.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            if tb_log_interval > 0 and ((it + 1) % tb_log_interval == 0 or (it + 1) == len(train_dataloader)):
                opt_groups_step = getattr(optimizer, "param_groups", None)
                if opt_groups_step is None and hasattr(optimizer, "optimizer"):
                    opt_groups_step = getattr(optimizer.optimizer, "param_groups", [])
                if opt_groups_step is None:
                    opt_groups_step = []
                lr_by_name_step = {
                    str(group.get("name", f"group{i}")): float(group["lr"])
                    for i, group in enumerate(opt_groups_step)
                }
                global_step = (epoch - 1) * len(train_dataloader) + (it + 1)
                train_step_tb = {
                    "train_step/Total": _safe_float(loss_total.item()),
                    "train_step/Sem": _safe_float(loss_sem.item()),
                    "train_step/Bdy": _safe_float(loss_boundary.item()),
                    "train_step/Ovl": _safe_float(loss_overlap.item()),
                    "train_step/Focal": _safe_float(loss_focal.item()),
                    "train_step/DiceIns": _safe_float(loss_dice_ins.item()),
                    "train_step/IoU": _safe_float(loss_iou.item()),
                    "train_step/CoOccur": _safe_float(loss_cooccur.item()),
                    "train_step/CCOverlap": _safe_float(loss_cc_overlap.item()),
                    "lr/sam_step": _safe_float(lr_by_name_step.get("sam")),
                    "lr/sem_step": _safe_float(lr_by_name_step.get("sem")),
                    "lr/aux_heads_step": _safe_float(lr_by_name_step.get("aux_heads")),
                    "lr/apam_fusion_step": _safe_float(lr_by_name_step.get("apam_fusion")),
                    "lr/gate_step": _safe_float(lr_by_name_step.get("gate")),
                }
                fabric.log_dict({k: v for k, v in train_step_tb.items() if v != ""}, step=global_step)

            fabric.print(
                f"Epoch: [{epoch}][{it + 1}/{len(train_dataloader)}]"
                f" | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]"
                f" | Focal [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]"
                f" | Dice(Ins) [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]"
                f" | Sem(2ch) [{sem_losses.val:.4f} ({sem_losses.avg:.4f})]"
                f" | Bdy [{boundary_losses.val:.4f} ({boundary_losses.avg:.4f})]"
                f" | Ovl [{overlap_losses.val:.4f} ({overlap_losses.avg:.4f})]"
                f" | CoOccur [{cooccur_losses_m.val:.4f} ({cooccur_losses_m.avg:.4f})]"
                f" | CCOver [{cc_overlap_losses_m.val:.4f} ({cc_overlap_losses_m.avg:.4f})]"
                f" | IoU [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]"
                f" | Total [{total_losses.val:.4f} ({total_losses.avg:.4f})]"
            )

        opt_groups = getattr(optimizer, "param_groups", None)
        if opt_groups is None and hasattr(optimizer, "optimizer"):
            opt_groups = getattr(optimizer.optimizer, "param_groups", [])
        if opt_groups is None:
            opt_groups = []
        lr_by_name = {str(group.get("name", f"group{i}")): float(group["lr"]) for i, group in enumerate(opt_groups)}
        train_tb = {
            "focal_loss": _safe_float(focal_losses.avg),
            "dice_loss_ins": _safe_float(dice_losses.avg),
            "sem_loss_2ch_dice": _safe_float(sem_losses.avg),
            "boundary_loss": _safe_float(boundary_losses.avg),
            "overlap_loss": _safe_float(overlap_losses.avg),
            "cooccur_loss": _safe_float(cooccur_losses_m.avg),
            "cc_overlap_loss": _safe_float(cc_overlap_losses_m.avg),
            "iou_loss": _safe_float(iou_losses.avg),
            "total_loss": _safe_float(total_losses.avg),
            "train/focal_loss": _safe_float(focal_losses.avg),
            "train/dice_loss_ins": _safe_float(dice_losses.avg),
            "train/sem_loss_2ch_dice": _safe_float(sem_losses.avg),
            "train/boundary_loss": _safe_float(boundary_losses.avg),
            "train/overlap_loss": _safe_float(overlap_losses.avg),
            "train/cooccur_loss": _safe_float(cooccur_losses_m.avg),
            "train/cc_overlap_loss": _safe_float(cc_overlap_losses_m.avg),
            "train/iou_loss": _safe_float(iou_losses.avg),
            "train/total_loss": _safe_float(total_losses.avg),
            "lr/sam": _safe_float(lr_by_name.get("sam")),
            "lr/sem": _safe_float(lr_by_name.get("sem")),
            "lr/aux_heads": _safe_float(lr_by_name.get("aux_heads")),
            "lr/apam_fusion": _safe_float(lr_by_name.get("apam_fusion")),
            "lr/gate": _safe_float(lr_by_name.get("gate")),
        }
        fabric.log_dict({k: v for k, v in train_tb.items() if v != ""}, step=epoch)

        if getattr(fabric, "global_rank", 0) == 0:
            _append_metrics_csv(metrics_csv, {
                "phase": "train",
                "epoch": epoch,
                "step": epoch,
                "train_focal_loss": _safe_float(focal_losses.avg),
                "train_dice_loss_ins": _safe_float(dice_losses.avg),
                "train_sem_loss_2ch_dice": _safe_float(sem_losses.avg),
                "train_boundary_loss": _safe_float(boundary_losses.avg),
                "train_overlap_loss": _safe_float(overlap_losses.avg),
                "train_cooccur_loss": _safe_float(cooccur_losses_m.avg),
                "train_cc_overlap_loss": _safe_float(cc_overlap_losses_m.avg),
                "train_iou_loss": _safe_float(iou_losses.avg),
                "train_total_loss": _safe_float(total_losses.avg),
                "lr_sam": _safe_float(lr_by_name.get("sam")),
                "lr_sem": _safe_float(lr_by_name.get("sem")),
                "lr_aux_heads": _safe_float(lr_by_name.get("aux_heads")),
                "lr_apam_fusion": _safe_float(lr_by_name.get("apam_fusion")),
                "lr_gate": _safe_float(lr_by_name.get("gate")),
            })

    if not early_stop.early_stop:
        validate(
            fabric, model, val_dataloader,
            early_stop, cfg.num_epochs,
            pos_weight_cooccur=pw_cooccur,
            pos_weight_cc_overlap=pw_cc_overlap,
            metrics_csv=metrics_csv,
        )


def configure_opt(cfg: Box, model: Model):
    """
     parameter groups:
        - sam
        - sem
        - aux_heads
        - apam_fusion
        - gate
    """
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

    sam_params = []
    sem_params = []
    aux_head_params = []
    fusion_params = []
    gate_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("apam_fusion."):
            fusion_params.append(p)
        elif name.startswith(("aux_heads.", "cooccur_head.", "cc_overlap_head.")):
            aux_head_params.append(p)
        elif name.startswith("cell_to_fragment_gate."):
            gate_params.append(p)
        elif name.startswith("semantic_decoder."):
            sem_params.append(p)
        elif name.startswith("sam."):
            sam_params.append(p)
        else:
            sem_params.append(p)

    parameters = []
    if sam_params:
        parameters.append({
            "params": sam_params,
            "name": "sam",
            "lr": cfg.opt.learning_rate_sam,
            "weight_decay": cfg.opt.weight_decay,
        })
    if sem_params:
        parameters.append({
            "params": sem_params,
            "name": "sem",
            "lr": cfg.opt.learning_rate_sem,
            "weight_decay": cfg.opt.weight_decay,
        })
    if aux_head_params:
        parameters.append({
            "params": aux_head_params,
            "name": "aux_heads",
            "lr": float(_cfg_get(cfg, ["opt", "learning_rate_aux"], cfg.opt.learning_rate_sem)),
            "weight_decay": cfg.opt.weight_decay,
        })
    if fusion_params:
        parameters.append({
            "params": fusion_params,
            "name": "apam_fusion",
            "lr": float(_cfg_get(cfg, ["opt", "learning_rate_fusion"], 1e-4)),
            "weight_decay": cfg.opt.weight_decay,
        })
    if gate_params:
        parameters.append({
            "params": gate_params,
            "name": "gate",
            "lr": float(_cfg_get(cfg, ["opt", "learning_rate_gate"], 5e-5)),
            "weight_decay": 0.0,
        })

    print("[OptGroups]")
    for group in parameters:
        n_params = sum(p.numel() for p in group["params"])
        print(f"  {group['name']:12s}: {n_params:>10,d}  lr={group['lr']}")

    optimizer = torch.optim.Adam(parameters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def main(cfg: Box) -> None:
    precision = str(_cfg_get(cfg, ["precision"], "32-true"))
    tb_save_dir = str(_cfg_get(cfg, ["tb_save_dir"], cfg.out_dir) or cfg.out_dir)
    tb_name = str(_cfg_get(cfg, ["tb_name"], "lightning-sam") or "lightning-sam")
    fabric = L.Fabric(
        accelerator="auto",
        devices=cfg.num_devices,
        strategy="auto",
        precision=precision,
        loggers=[TensorBoardLogger(tb_save_dir, name=tb_name)],
    )
    fabric.launch()
    seed = int(_cfg_get(cfg, ["seed"], 1344))
    fabric.seed_everything(seed + fabric.global_rank)
    fabric.print(f"[Fabric] precision={precision} seed={seed} tb_save_dir={tb_save_dir} tb_name={tb_name}")

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    init_ckpt = str(_cfg_get(cfg, ["model", "init_from_ckpt"], "") or "")
    if init_ckpt:
        _load_init_checkpoint(model, init_ckpt, fabric)

    import shutil
    shutil.copy("config.py", cfg.out_dir)
    shutil.copy("semantic_decoder.py", cfg.out_dir)
    shutil.copy("cbim_aux_modules.py", cfg.out_dir)
    shutil.copy("model.py", cfg.out_dir)

    train_data, val_data = load_datasets(cfg, model.sam.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op1_alpha", type=float, default=None)
    ap.add_argument("--op3_enable", type=int, choices=[0, 1], default=None)
    ap.add_argument("--cooccur_enable", type=int, choices=[0, 1], default=None)
    ap.add_argument("--cc_overlap_enable", type=int, choices=[0, 1], default=None)
    ap.add_argument("--fusion_enable", type=int, choices=[0, 1], default=None)
    ap.add_argument("--learning_rate_fusion", type=float, default=None)
    ap.add_argument("--fusion_scale_init", type=float, default=None,
                    help="Target initial value for softplus(raw_scale), default 1.0")
    ap.add_argument("--seed", type=int, default=None,
                    help="base random seed; actual per-rank seed is seed + global_rank")
    ap.add_argument("--metrics_csv", type=str, default=None,
                    help="optional epoch-level metrics CSV path for paper figures")
    ap.add_argument("--tb_save_dir", type=str, default=None,
                    help="optional TensorBoard save_dir")
    ap.add_argument("--tb_name", type=str, default=None,
                    help="optional TensorBoard run name")
    ap.add_argument("--tb_log_interval", type=int, default=None,
                    help="batch interval for train_step TensorBoard scalars; <=0 disables batch-level logging")
    return ap.parse_args()


def apply_runtime_overrides(cfg: Box, args) -> None:
    overrides = []
    if args.op1_alpha is not None:
        cfg.model.cbim.op1_alpha = float(args.op1_alpha)
        overrides.append(f"op1_alpha={cfg.model.cbim.op1_alpha}")
    if args.op3_enable is not None:
        cfg.model.cbim.op3_enable = bool(int(args.op3_enable))
        overrides.append(f"op3_enable={cfg.model.cbim.op3_enable}")
    if args.cooccur_enable is not None:
        cfg.model.apam.cooccur_enable = bool(int(args.cooccur_enable))
        overrides.append(f"cooccur_enable={cfg.model.apam.cooccur_enable}")
    if args.cc_overlap_enable is not None:
        cfg.model.apam.cc_overlap_enable = bool(int(args.cc_overlap_enable))
        overrides.append(f"cc_overlap_enable={cfg.model.apam.cc_overlap_enable}")
    if args.fusion_enable is not None:
        cfg.model.apam.fusion_enable = bool(int(args.fusion_enable))
        overrides.append(f"fusion_enable={cfg.model.apam.fusion_enable}")
    if args.learning_rate_fusion is not None:
        cfg.opt.learning_rate_fusion = float(args.learning_rate_fusion)
        overrides.append(f"learning_rate_fusion={cfg.opt.learning_rate_fusion}")
    if args.fusion_scale_init is not None:
        cfg.model.apam.fusion_scale_init = float(args.fusion_scale_init)
        overrides.append(f"fusion_scale_init={cfg.model.apam.fusion_scale_init}")
    if args.seed is not None:
        cfg.seed = int(args.seed)
        overrides.append(f"seed={cfg.seed}")
    if args.metrics_csv is not None:
        cfg.metrics_csv = str(args.metrics_csv)
        overrides.append(f"metrics_csv={cfg.metrics_csv}")
    if args.tb_save_dir is not None:
        cfg.tb_save_dir = str(args.tb_save_dir)
        overrides.append(f"tb_save_dir={cfg.tb_save_dir}")
    if args.tb_name is not None:
        cfg.tb_name = str(args.tb_name)
        overrides.append(f"tb_name={cfg.tb_name}")
    if args.tb_log_interval is not None:
        cfg.tb_log_interval = int(args.tb_log_interval)
        overrides.append(f"tb_log_interval={cfg.tb_log_interval}")
    if overrides:
        print("[RuntimeOverride] " + ", ".join(overrides))


if __name__ == "__main__":
    args = parse_args()
    apply_runtime_overrides(cfg, args)
    main(cfg)
