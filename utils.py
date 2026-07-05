import numpy as np
import torch
from pathlib import Path
from pycocotools.coco import COCO
from medpy import metric
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval as _COCOeval

# ===== NumPy 2.x compatibility patch for np.float / np.float_ / np.int =====
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int"):
    np.int = np.int64
# ======================================================================


# ===================== Training utilities =====================

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        # Supports torch.Tensor or float values.
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute IoU for one image or instance, used by the IoU regression loss in train_es.py.

    Supported input shapes:
      - pred_mask: (H, W) or (1, H, W)
      - gt_mask  : (H, W) or (1, H, W)

    Returns:
      - batch_iou: shape = (N, 1); N=1 when called per instance in train_es.py
    """
    # Ensure at least (N, H, W).
    if pred_mask.ndim == 2:
        pred = pred_mask.unsqueeze(0)
        gt = gt_mask.unsqueeze(0)
    else:
        pred = pred_mask
        gt = gt_mask

    pred = (pred >= 0.5).float()
    gt = gt.float()

    # (N, H, W) → (N,)
    intersection = torch.sum(pred * gt, dim=(1, 2))
    union = torch.sum(pred, dim=(1, 2)) + torch.sum(gt, dim=(1, 2)) - intersection

    iou = intersection / (union + eps)
    return iou.unsqueeze(1)  # (N,1)


def calc_dice(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """
    Compute batch-level Dice from (tp, fp, fn, tn) returned by segmentation_models_pytorch.metrics.get_stats.
    The true-negative count is not used by Dice.

    Used by validate() in train_es.py as follows:
        batch_stats = smp.metrics.get_stats(...)
        batch_dice = calc_dice(*batch_stats)
    """
    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    # tn is not used in Dice calculation.

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return dice


def calc_metric(pred: np.ndarray, gt: np.ndarray):
    """
    Compute per-image Dice / HD for fragment semantic segmentation.
    pred, gt: numpy binary arrays
    """
    pred = pred.copy()
    gt = gt.copy()

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd = metric.binary.hd(pred, gt)
        return dice, hd
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    else:
        return 0.0, 500.0


# ===================== Fragment semantic GT construction =====================

def build_fragment_gt_from_instances(instances_json: str, out_json: str, frag_cat_id: int = 2):
    """
    Aggregate fragment instances from a COCO annotation file to build GT for semantic evaluation.
    """
    instances_json = str(instances_json)
    out_json = str(out_json)
    coco = COCO(instances_json)

    dataset = coco.dataset
    frag_cat = None
    for cat in dataset.get("categories", []):
        if int(cat.get("id", -1)) == int(frag_cat_id):
            frag_cat = dict(cat)
            break
    if frag_cat is None:
        frag_cat = {"id": int(frag_cat_id), "name": "fragment"}

    new_dataset = {
        "images": dataset.get("images", []),
        "categories": [frag_cat],
        "annotations": [],
    }
    if "info" in dataset:
        new_dataset["info"] = dataset["info"]
    if "licenses" in dataset:
        new_dataset["licenses"] = dataset["licenses"]

    ann_id = 1
    for img in dataset.get("images", []):
        img_id = int(img["id"])
        h = int(img.get("height", 0))
        w = int(img.get("width", 0))
        union = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[frag_cat_id])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            union = np.logical_or(union, coco.annToMask(ann)).astype(np.uint8)

        if union.any():
            rle = maskUtils.encode(np.asfortranarray(union))
            area = float(maskUtils.area(rle))
            bbox = list(maskUtils.toBbox(rle).astype(float))
            rle["counts"] = rle["counts"].decode("utf-8")

            new_dataset["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(frag_cat_id),
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False)

    print(f"[OK] fragment GT saved: {out_json}")


# ===================== Fragment semantic evaluation: Dice / HD =====================

def evalSemantic(coco, res_json: str, return_hd: bool = False):
    """
    Fragment semantic evaluation:
    Implementation notes:
    1. Read image sizes dynamically;
    2. Iterate over all GT images so images without predictions are still evaluated.
    """
    dice = 0.0
    hd = 0.0
    num = 0

    cocoDT = coco.loadRes(res_json)
    wrongIds = []
    imgIds = [id_ for id_ in coco.getImgIds() if id_ not in wrongIds]

    for imgId in imgIds:
        img_info = coco.loadImgs(imgId)[0]
        h, w = img_info.get('height', 800), img_info.get('width', 800)

        annId = coco.getAnnIds(imgId, 2)
        ann = coco.loadAnns(annId)
        if len(ann) == 0:
            label = np.zeros([h, w], dtype=np.uint8)
        else:
            rles = [a["segmentation"] for a in ann]
            label = maskUtils.decode(rles)
            if label.ndim == 3:
                label = np.logical_or.reduce(label, axis=2).astype(np.uint8)

        annId = cocoDT.getAnnIds(imgId, 2)
        ann = cocoDT.loadAnns(annId)
        if len(ann) == 0:
            pred = np.zeros([h, w], dtype=np.uint8)
        else:
            rles = [a["segmentation"] for a in ann]
            pred = maskUtils.decode(rles)
            if pred.ndim == 3:
                pred = np.logical_or.reduce(pred, axis=2).astype(np.uint8)

        dice_i, hd_i = calc_metric(pred, label)
        dice += dice_i
        hd += hd_i
        num += 1

    mean_dice = dice / max(num, 1)
    mean_hd = hd / max(num, 1)
    print("dice:", mean_dice, "\n", "hd:", mean_hd)
    if return_hd:
        return mean_dice, mean_hd
    return mean_dice, mean_hd


# ===================== Extended COCOeval with F1 / Precision / Recall =====================

class COCOeval(_COCOeval):
    """
    Based on pycocotools.cocoeval.COCOeval:
      - preserves the original AP / AR computation and summarize() output
      - additionally fills self.stats[12], [13], and [14]:
          stats[12] = F1
          stats[13] = Precision
          stats[14] = Recall
    """

    def summarize(self):
        super().summarize()

        if not hasattr(self, "eval") or not self.eval:
            return

        precision = self.eval.get("precision", None)
        recall = self.eval.get("recall", None)

        p_mean = -1.0
        r_mean = -1.0

        if isinstance(precision, np.ndarray):
            valid_p = precision[precision > -1]
            if valid_p.size > 0:
                p_mean = float(valid_p.mean())

        if isinstance(recall, np.ndarray):
            valid_r = recall[recall > -1]
            if valid_r.size > 0:
                r_mean = float(valid_r.mean())

        if p_mean <= 0 or r_mean <= 0:
            f1 = 0.0
        else:
            f1 = 2.0 * p_mean * r_mean / (p_mean + r_mean)

        base = list(self.stats) if getattr(self, "stats", None) is not None else []
        while len(base) < 12:
            base.append(-1.0)

        extra = [f1, p_mean, r_mean]
        self.stats = np.array(base + extra, dtype=np.float32)

        print(f"Extra metrics: F1={f1:.3f}, Precision={p_mean:.3f}, Recall={r_mean:.3f}")
