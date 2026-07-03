"""
Dual-branch model: SAM-based instance branch and semantic fragment branch,
with auxiliary C1 heads, a cell-to-fragment gate, and C2 evidence fusion.
"""

import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from segment_anything import sam_model_registry

from semantic_decoder import SemanticDecoder
from cbim_aux_modules import (
    AuxiliaryHeads,
    CoOccurrenceHead,
    CCOverlapHead,
    APAMEvidenceFusion,
    CellToFragmentGate,
    build_cell_union_logit_from_masks,
)


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(self._cfg_get(cfg, ["num_classes"], 2))

        sam_type = str(self._cfg_get(cfg, ["model", "type"], "vit_b"))
        sam_ckpt = str(self._cfg_get(cfg, ["model", "checkpoint"], ""))
        sam_ckpt = sam_ckpt if sam_ckpt and os.path.exists(sam_ckpt) else None

        self.sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)

        self.semantic_decoder = SemanticDecoder(num_classes=self.num_classes)

        aux_in = int(self._cfg_get(cfg, ["model", "aux_heads", "in_ch"], 256))
        aux_mid = int(self._cfg_get(cfg, ["model", "aux_heads", "mid_ch"], 64))

        self.aux_heads = AuxiliaryHeads(in_ch=aux_in, mid_ch=aux_mid)

        self.cooccur_head = CoOccurrenceHead(in_ch=aux_in, mid_ch=aux_mid)
        self.cc_overlap_head = CCOverlapHead(in_ch=aux_in, mid_ch=aux_mid)

        self.cooccur_enable = bool(self._cfg_get(cfg, ["model", "apam", "cooccur_enable"], True))
        self.cc_overlap_enable = bool(self._cfg_get(cfg, ["model", "apam", "cc_overlap_enable"], True))
        fusion_enable = bool(self._cfg_get(cfg, ["model", "apam", "fusion_enable"], True))
        fusion_scale_init = float(self._cfg_get(cfg, ["model", "apam", "fusion_scale_init"], 1.0))
        self.apam_fusion = APAMEvidenceFusion(
            in_channels=6, mid_channels=16, enable=fusion_enable, scale_init=fusion_scale_init,
        )

        # CBIM switch.
        self.op1_alpha = float(self._cfg_get(cfg, ["model", "cbim", "op1_alpha"], 0.0))
        self.op3_enable = bool(self._cfg_get(cfg, ["model", "cbim", "op3_enable"], True))

        self.cell_to_fragment_gate = CellToFragmentGate(
            penalty_init=float(self._cfg_get(cfg, ["model", "cbim", "penalty_init"], 10.0)),
            detach_gate=bool(self._cfg_get(cfg, ["model", "cbim", "detach_gate"], True)),
            enable=self.op3_enable,
        )

        self._apply_freeze()

    @staticmethod
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

    def _apply_freeze(self):
        freeze_cfg = self._cfg_get(self.cfg, ["model", "freeze"], {}) or {}
        mapping = {
            "image_encoder": getattr(self.sam, "image_encoder", None),
            "prompt_encoder": getattr(self.sam, "prompt_encoder", None),
            "mask_decoder": getattr(self.sam, "mask_decoder", None),
            "sem_decoder": getattr(self, "semantic_decoder", None),
            "semantic_decoder": getattr(self, "semantic_decoder", None),
            "aux_heads": getattr(self, "aux_heads", None),
            "cell_to_fragment_gate": getattr(self, "cell_to_fragment_gate", None),
            "apam_heads": None,       # Handled separately; contains two heads.
            "apam_fusion": getattr(self, "apam_fusion", None),
        }
        # General parameters.
        for name, module in mapping.items():
            if module is None:
                continue
            sf = False
            if isinstance(freeze_cfg, dict):
                sf = bool(freeze_cfg.get(name, False))
            else:
                sf = bool(getattr(freeze_cfg, name, False))
            if sf:
                for p in module.parameters():
                    p.requires_grad = False
        # apam_heads contains two heads.
        apam_heads_freeze = False
        if isinstance(freeze_cfg, dict):
            apam_heads_freeze = bool(freeze_cfg.get("apam_heads", False))
        else:
            apam_heads_freeze = bool(getattr(freeze_cfg, "apam_heads", False))
        if apam_heads_freeze:
            for m in [self.cooccur_head, self.cc_overlap_head]:
                for p in m.parameters():
                    p.requires_grad = False

    def setup(self):
        self.train()

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B,C,H,W], got {tuple(images.shape)}")
        processed = [self.sam.preprocess(img) for img in images]
        return torch.stack(processed, dim=0)

    def _normalize_boxes(self, boxes, batch_size: int, device: torch.device):
        if boxes is None:
            return [torch.zeros((0, 4), dtype=torch.float32, device=device) for _ in range(batch_size)]
        if isinstance(boxes, torch.Tensor):
            if boxes.ndim == 2:
                if batch_size != 1:
                    raise ValueError(f"boxes.shape={tuple(boxes.shape)} does not match bs={batch_size}")
                return [boxes.to(device=device, dtype=torch.float32)]
            if boxes.ndim == 3:
                return [boxes[i].to(device=device, dtype=torch.float32) for i in range(batch_size)]
            raise ValueError(f"Unsupported boxes tensor dimensions: {tuple(boxes.shape)}")
        if isinstance(boxes, (list, tuple)):
            out = []
            for b in boxes:
                if b is None:
                    out.append(torch.zeros((0, 4), dtype=torch.float32, device=device)); continue
                if not isinstance(b, torch.Tensor):
                    b = torch.as_tensor(b, dtype=torch.float32, device=device)
                else:
                    b = b.to(device=device, dtype=torch.float32)
                if b.numel() == 0:
                    b = torch.zeros((0, 4), dtype=torch.float32, device=device)
                elif b.ndim == 1:
                    b = b.unsqueeze(0)
                out.append(b)
            if len(out) != batch_size:
                raise ValueError(f"boxes list length {len(out)} does not match bs={batch_size}")
            return out
        raise TypeError(f"Unsupported boxes type: {type(boxes)}")

    def forward(self, images: torch.Tensor, boxes=None, return_aux: bool = False):
        device = images.device
        bsz, _, h, w = images.shape

        input_images = self._prepare_images(images)
        image_embeddings = self.sam.image_encoder(input_images)  # [B,256,64,64]

        # Semantic branch
        sem_logits_raw = self.semantic_decoder(image_embeddings)
        sem_logits_raw = self.sam.postprocess_masks(
            sem_logits_raw,
            input_size=(input_images.shape[-2], input_images.shape[-1]),
            original_size=(h, w),
        )  # [B, 2, H, W]

        # Aux heads: boundary + overlap
        boundary_logits, overlap_logits = self.aux_heads(image_embeddings, out_size=(h, w))

        if self.cooccur_enable:
            cooccur_logits = self.cooccur_head(image_embeddings, out_size=(h, w))
        else:
            cooccur_logits = torch.zeros_like(boundary_logits)
        if self.cc_overlap_enable:
            cc_overlap_logits = self.cc_overlap_head(image_embeddings, out_size=(h, w))
        else:
            cc_overlap_logits = torch.zeros_like(boundary_logits)

        # Instance branch using the SAM mask decoder loop.
        boxes_list = self._normalize_boxes(boxes, bsz, device)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()

        pred_masks = []
        iou_predictions_out = []
        cell_union_logits_list = []

        for i in range(bsz):
            box_i = boxes_list[i]
            ref_t = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)

            if box_i.numel() == 0:
                pred_masks.append(torch.zeros((0, h, w), dtype=torch.float32, device=device))
                iou_predictions_out.append(torch.zeros((0, 1), dtype=torch.float32, device=device))
                cell_union_logits_list.append(
                    build_cell_union_logit_from_masks(
                        torch.zeros((0, h, w), device=device), ref_tensor=ref_t,
                    )
                )
                continue

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=box_i, masks=None,
            )
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=(input_images.shape[-2], input_images.shape[-1]),
                original_size=(h, w),
            )
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if iou_predictions.ndim == 2 and iou_predictions.shape[1] != 1:
                iou_predictions = iou_predictions[:, :1]

            pred_masks.append(masks)
            iou_predictions_out.append(iou_predictions)

            union_logit_i = build_cell_union_logit_from_masks(masks, ref_tensor=ref_t)
            cell_union_logits_list.append(union_logit_i)

        cell_union_logits = torch.cat(cell_union_logits_list, dim=0)  # [B, 1, H, W]

        # ========================================================
        # op1: boundary bias, sigmoid-normalized; disabled by default when alpha=0.
        # ========================================================
        bg_logits = sem_logits_raw[:, 0:1]
        frag_logits_raw = sem_logits_raw[:, 1:2]

        if self.op1_alpha > 0:
            bias = self.op1_alpha * torch.sigmoid(boundary_logits)
            frag_logits_raw = frag_logits_raw + bias
            cell_union_logits = cell_union_logits + bias

        # ========================================================
        # ========================================================
        frag_logits_gated = self.cell_to_fragment_gate(cell_union_logits, frag_logits_raw)

        # ========================================================
        # Residual injection allows cooccur/cc_overlap supervision to backpropagate to semantic_decoder.
        # ========================================================
        cell_union_logits_for_fusion = cell_union_logits
        # When detach_gate=True is used in the gate path, the fusion path keeps the same protection,
        # preventing refined semantic loss from flowing back into the SAM mask_decoder.
        if getattr(self.cell_to_fragment_gate, "detach_gate", False):
            cell_union_logits_for_fusion = cell_union_logits_for_fusion.detach()
        frag_logits_refined = self.apam_fusion(
            frag_logit=frag_logits_gated,
            boundary_logit=boundary_logits,
            overlap_logit=overlap_logits,
            cooccur_logit=cooccur_logits,
            cc_overlap_logit=cc_overlap_logits,
            cell_union_logit=cell_union_logits_for_fusion,
        )
        # If fusion_enable=False, refined == gated.

        # Build the two-channel output from the refined path used by the main loss.
        sem_logits_refined = torch.cat([bg_logits, frag_logits_refined], dim=1)

        sem_logits_gated = torch.cat([bg_logits, frag_logits_gated], dim=1)

        if not return_aux:
            # Return the refined output from the main path.
            return sem_logits_refined, pred_masks, iou_predictions_out

        aux_dict: Dict[str, torch.Tensor] = {
            "boundary_logits": boundary_logits,
            "overlap_logits": overlap_logits,
            "cooccur_logits": cooccur_logits,            # new
            "cc_overlap_logits": cc_overlap_logits,      # new
            "cell_union_logits": cell_union_logits,
            "frag_logits_raw": frag_logits_raw,
            "frag_logits_gated": frag_logits_gated,
            "frag_logits_refined": frag_logits_refined,
            "sem_logits_raw": sem_logits_raw,
            "sem_logits_gated": sem_logits_gated,
        }
        return sem_logits_refined, pred_masks, iou_predictions_out, aux_dict
