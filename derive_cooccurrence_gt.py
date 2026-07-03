"""
Derive two auxiliary C1 target masks from COCO-format annotations.

1. cooccurrence: cell_union AND fragment_union.
2. cc_overlap: pixels covered by two or more cell masks.

Outputs:
  {aux_gt_root}/{split}/cooccurrence/{image_stem}.png
  {aux_gt_root}/{split}/cc_overlap/{image_stem}.png
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


# ===== Category IDs, consistent with instances_*.json =====
CAT_CELL = 1
CAT_FRAG = 2


def rasterize_ann(ann, h, w):
    """Rasterize one COCO annotation into a binary mask of shape (H, W)."""
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((h, w), dtype=bool)

    # Polygon format.
    if isinstance(seg, list) and len(seg) > 0:
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
        m = mask_utils.decode(rle).astype(bool)
        return m
    # RLE format (dict).
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects([seg], h, w)[0]
        else:
            rle = seg
        return mask_utils.decode(rle).astype(bool)
    return np.zeros((h, w), dtype=bool)


def derive_one_image(img_info, anns):
    """
    Derive cooccurrence and cc_overlap masks for one image.

    Args:
        img_info: COCO image dict with id, file_name, width, and height.
        anns: all COCO annotations for the image.

    Returns:
        cooccurrence_mask: (H, W) uint8 {0, 255}
        cc_overlap_mask:   (H, W) uint8 {0, 255}
    """
    h, w = img_info["height"], img_info["width"]

    cell_union = np.zeros((h, w), dtype=bool)
    cell_sum = np.zeros((h, w), dtype=np.uint8)  # Accumulate masks to compute sum >= 2.
    frag_union = np.zeros((h, w), dtype=bool)

    for a in anns:
        m = rasterize_ann(a, h, w)
        if m.sum() == 0:
            continue
        if a["category_id"] == CAT_CELL:
            cell_union |= m
            cell_sum = cell_sum + m.astype(np.uint8)
        elif a["category_id"] == CAT_FRAG:
            frag_union |= m

    # cooccurrence: intersection of cell_union and frag_union.
    cooccurrence = (cell_union & frag_union).astype(np.uint8) * 255

    # cc_overlap: pixels covered by at least two cells.
    cc_overlap = (cell_sum >= 2).astype(np.uint8) * 255

    return cooccurrence, cc_overlap


def derive_split(
    coco_json: str,
    img_dir: str,
    out_root: str,
    split_name: str,
    sanity_sample: int = 5,
):
    """
    Iterate through one split, derive cooccurrence and cc_overlap, and save PNG masks.

    Args:
        coco_json: path to instances_{split}.json.
        img_dir: path to images/{split}/, used to verify file_name values.
        out_root: path to aux_gt/{split}/.
        split_name: 'train' or 'val', used only for logging.
        sanity_sample: number of sampled images for summary checks.
    """
    coco = COCO(coco_json)
    cooccur_dir = Path(out_root) / "cooccurrence"
    cc_dir = Path(out_root) / "cc_overlap"
    cooccur_dir.mkdir(parents=True, exist_ok=True)
    cc_dir.mkdir(parents=True, exist_ok=True)

    img_ids = list(coco.imgs.keys())
    print(f"\n[{split_name}] Processing {len(img_ids)} images")
    print(f"  Output directory: {out_root}/cooccurrence/")
    print(f"  Output directory: {out_root}/cc_overlap/")

    # Global statistics.
    total_pos_co = 0
    total_pos_cc = 0
    total_pix = 0
    imgs_with_co = 0
    imgs_with_cc = 0
    n_skipped = 0

    for idx, img_id in enumerate(tqdm(img_ids, desc=split_name)):
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            n_skipped += 1
            continue

        # Optional image file existence check.
        img_path = Path(img_dir) / img_info["file_name"]
        if not img_path.is_file() and idx < 3:
            print(f"[WARN] Image file not found: {img_path} (GT will still be derived)")

        cooccur_mask, cc_mask = derive_one_image(img_info, anns)

        stem = Path(img_info["file_name"]).stem
        cv2.imwrite(str(cooccur_dir / f"{stem}.png"), cooccur_mask)
        cv2.imwrite(str(cc_dir / f"{stem}.png"), cc_mask)

        # Statistics.
        h, w = cooccur_mask.shape
        total_pix += h * w
        pos_co = int((cooccur_mask > 0).sum())
        pos_cc = int((cc_mask > 0).sum())
        total_pos_co += pos_co
        total_pos_cc += pos_cc
        if pos_co > 0:
            imgs_with_co += 1
        if pos_cc > 0:
            imgs_with_cc += 1

    # Summary statistics.
    print(f"\n[{split_name}] Derivation completed:")
    print(f"  Images skipped because no annotations: {n_skipped}")
    print(f"  Images with cooccurrence: {imgs_with_co}/{len(img_ids)} "
          f"= {100 * imgs_with_co / max(len(img_ids), 1):.1f}%")
    print(f"  Images with cc_overlap:   {imgs_with_cc}/{len(img_ids)} "
          f"= {100 * imgs_with_cc / max(len(img_ids), 1):.1f}%")
    print(f"  cooccurrence positive-pixel ratio: "
          f"{100 * total_pos_co / max(total_pix, 1):.3f}% (sparse)")
    print(f"  cc_overlap positive-pixel ratio:   "
          f"{100 * total_pos_cc / max(total_pix, 1):.3f}%")

    # Reference values for pos_weight calculation in train_es.py.
    pos_weight_co = max(total_pix - total_pos_co, 1) / max(total_pos_co, 1)
    pos_weight_cc = max(total_pix - total_pos_cc, 1) / max(total_pos_cc, 1)
    pos_weight_co_clamped = min(pos_weight_co, 20.0)
    pos_weight_cc_clamped = min(pos_weight_cc, 20.0)
    print(f"\n[{split_name}] Suggested BCE pos_weight values, clamped to 20:")
    print(f"  cooccurrence: raw={pos_weight_co:.1f}, "
          f"clamped={pos_weight_co_clamped:.2f}")
    print(f"  cc_overlap:   raw={pos_weight_cc:.1f}, "
          f"clamped={pos_weight_cc_clamped:.2f}")

    # Save statistics to JSON for train_es.py.
    stats_path = Path(out_root) / "apam_gt_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_images": len(img_ids),
            "imgs_with_cooccurrence": imgs_with_co,
            "imgs_with_cc_overlap": imgs_with_cc,
            "total_pos_cooccurrence": total_pos_co,
            "total_pos_cc_overlap": total_pos_cc,
            "total_pixels": total_pix,
            "pos_weight_cooccurrence": float(pos_weight_co_clamped),
            "pos_weight_cc_overlap": float(pos_weight_cc_clamped),
        }, f, indent=2, ensure_ascii=False)
    print(f"[{split_name}] Statistics saved: {stats_path}")


def main():
    from config import (
        TRAIN_JSON, VAL_JSON, TRAIN_IMG_DIR, VAL_IMG_DIR,
        TRAIN_AUX_GT_DIR, VAL_AUX_GT_DIR,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    args = ap.parse_args()

    targets = []
    if args.split in ("train", "both"):
        targets.append(("train", TRAIN_JSON, TRAIN_IMG_DIR, TRAIN_AUX_GT_DIR))
    if args.split in ("val", "both"):
        targets.append(("val", VAL_JSON, VAL_IMG_DIR, VAL_AUX_GT_DIR))

    for name, coco_json, img_dir, out_root in targets:
        if not os.path.isfile(coco_json):
            print(f"[ERR] {name} annotation file not found: {coco_json}")
            continue
        derive_split(
            coco_json=coco_json,
            img_dir=img_dir,
            out_root=out_root,
            split_name=name,
        )

    print("\n[DONE] All splits completed")


if __name__ == "__main__":
    main()
