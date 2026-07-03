from box import Box
import json
import os


DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
WEIGHT_ROOT = os.environ.get("WEIGHT_ROOT", "./weights")
CODE_ROOT = os.environ.get("CODE_ROOT", ".")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images", "train")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "images", "val")
TRAIN_JSON = os.path.join(DATA_ROOT, "annotations", "instances_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "annotations", "instances_val.json")
AUX_GT_ROOT = os.path.join(DATA_ROOT, "aux_gt")
TRAIN_AUX_GT_DIR = os.path.join(AUX_GT_ROOT, "train")
VAL_AUX_GT_DIR = os.path.join(AUX_GT_ROOT, "val")

SAM_CKPT = os.environ.get("SAM_CKPT", os.path.join(WEIGHT_ROOT, "sam_vit_b_01ec64.pth"))
YOLO_BEST = os.environ.get("YOLO_BEST", os.path.join(WEIGHT_ROOT, "yolo11_best.pt"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(CODE_ROOT, "out", "training"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(CODE_ROOT, "checkpoints"))
DUAL_CKPT_NAME = os.environ.get("DUAL_CKPT_NAME", "sam_dual.pth")
DUAL_CKPT = os.environ.get("DUAL_CKPT", os.path.join(CKPT_DIR, DUAL_CKPT_NAME))
INIT_CKPT_NAME = os.environ.get("INIT_CKPT_NAME", "sam_dual_init.pth")
INIT_CKPT = os.environ.get("INIT_CKPT", os.path.join(CKPT_DIR, INIT_CKPT_NAME))


def _check_aux_gt_complete(aux_gt_dir: str, ann_json: str) -> dict:
    try:
        with open(ann_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {k: "empty" for k in ["boundary", "overlap", "cooccurrence", "cc_overlap"]}
    expected = {os.path.splitext(im["file_name"])[0] for im in data.get("images", [])}
    result = {}
    for sub in ["boundary", "overlap", "cooccurrence", "cc_overlap"]:
        sub_dir = os.path.join(aux_gt_dir, sub)
        if not os.path.isdir(sub_dir):
            result[sub] = "empty"
            continue
        have = {os.path.splitext(f)[0] for f in os.listdir(sub_dir) if f.endswith(".png")}
        result[sub] = "empty" if not have else ("complete" if have >= expected else "missing")
    return result


def validate_paths(require_init: bool = False, require_aux: bool = False) -> None:
    dirs = [DATA_ROOT, WEIGHT_ROOT, CODE_ROOT, TRAIN_IMG_DIR, VAL_IMG_DIR]
    files_to_check = [TRAIN_JSON, VAL_JSON, SAM_CKPT]
    if require_init:
        files_to_check.append(INIT_CKPT)
    missing = [p for p in dirs if not os.path.isdir(p)] + [p for p in files_to_check if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing configured paths: " + ", ".join(missing))
    if require_aux:
        for aux_dir, ann_json, split in [(TRAIN_AUX_GT_DIR, TRAIN_JSON, "train"), (VAL_AUX_GT_DIR, VAL_JSON, "val")]:
            status = _check_aux_gt_complete(aux_dir, ann_json)
            incomplete = [name for name, state in status.items() if state == "missing"]
            if incomplete:
                raise RuntimeError(f"{split} aux_gt incomplete: {', '.join(incomplete)}")


config = {
    "num_devices": 1,
    "precision": "16-mixed",
    "batch_size": 2,
    "num_workers": 2,
    "num_epochs": 120,
    "num_classes": 2,
    "patience": 5,
    "eval_interval": 1,
    "tb_log_interval": 200,
    "pretrained": False,
    "out_dir": OUT_DIR,
    "visualize_dir": "sam_dual",
    "opt": {
        "learning_rate": 3e-4, "learning_rate_sam": 1e-4, "learning_rate_sem": 1.2e-3,
        "learning_rate_aux": 1e-4, "learning_rate_fusion": 1e-4, "learning_rate_gate": 5e-5,
        "weight_decay": 4e-5, "warmup_steps": 150, "steps": [4000, 9500],
        "decay_factor": 10, "early_stop_min_delta": 0.004,
    },
    "model": {
        "type": "vit_b", "checkpoint": SAM_CKPT, "dual_checkpoint": DUAL_CKPT, "init_from_ckpt": INIT_CKPT,
        "freeze": {"image_encoder": False, "prompt_encoder": True, "mask_decoder": False, "sem_decoder": False, "aux_heads": False, "cell_to_fragment_gate": False, "apam_heads": False, "apam_fusion": False},
        "aux_heads": {"in_ch": 256, "mid_ch": 64},
        "cbim": {"penalty_init": 10.0, "op1_alpha": 0.0, "op2_enable": False, "op3_enable": True, "detach_gate": True},
        "apam": {"cooccur_enable": True, "cc_overlap_enable": True, "fusion_enable": True, "fusion_scale_init": 1.0, "pos_weight_cooccur_fallback": 20.0, "pos_weight_cc_overlap_fallback": 5.0},
        "loss_weights": {"boundary": 0.3, "overlap": 0.3, "cooccur": 0.3, "cc_overlap": 0.3, "sem_dice_2ch": 1.0},
    },
    "dataset": {
        "train": {"root_dir": TRAIN_IMG_DIR, "annotation_file": TRAIN_JSON, "aux_gt_dir": TRAIN_AUX_GT_DIR},
        "val": {"root_dir": VAL_IMG_DIR, "annotation_file": VAL_JSON, "aux_gt_dir": VAL_AUX_GT_DIR},
    },
    "yolo": YOLO_BEST,
}
cfg = Box(config)
