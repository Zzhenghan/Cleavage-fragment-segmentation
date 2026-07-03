"""
Derive auxiliary boundary and adjacency masks from COCO-format annotations.

Usage:
  python derive_aux_gt.py --ann ${DATA_ROOT}/annotations/instances_val.json \
      --img_dir ${DATA_ROOT}/images/val --out_dir ./aux_gt_val --vis_n 30

Outputs:
  out_dir/boundary/<filename>.png
  out_dir/overlap/<filename>.png
  out_dir/visualize/<filename>.png
"""

import os
import json
import argparse
import random
from collections import defaultdict

import numpy as np
import cv2
try:
    from tqdm import tqdm
except ImportError:
    # Fall back gracefully when tqdm is unavailable.
    def tqdm(x, **kwargs):
        return x
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Save figures in headless environments.
import matplotlib.pyplot as plt


# ----------------------------- Utility functions ----------------------------- #

def poly_to_mask(segmentation, h, w):
    """Convert the polygon segmentation of one annotation to a binary mask.
    segmentation is a list such as [[x1,y1,x2,y2,...], [x1,y1,...], ...].
    Each sublist is a polygon; one instance may contain multiple polygons.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not isinstance(segmentation, list):
        # This script expects polygon format and does not handle RLE.
        return mask
    for poly in segmentation:
        if len(poly) < 6:  # At least 3 points (6 numbers) are required for a polygon.
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def derive_boundary_gt(cell_masks, kernel_size=5, iterations=1):
    """Derive boundary GT.
    For each cell instance mask, compute dilation minus erosion to obtain a ring-like boundary band,
    then merge all instance boundary bands with a pixelwise OR.
    
    kernel_size and iterations control the boundary-band thickness.
    For 800x800 images with median cell area around 23000 pixels,
    kernel=5 and iter=1 produce a boundary band of roughly 2-4 pixels.
    Overly thick bands dilute supervision, while overly thin bands can make BCE highly imbalanced.
    """
    if len(cell_masks) == 0:
        return np.zeros_like(cell_masks[0] if cell_masks else np.zeros((1, 1), dtype=np.uint8))
    
    h, w = cell_masks[0].shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    for m in cell_masks:
        dilated = cv2.dilate(m, kernel, iterations=iterations)
        eroded = cv2.erode(m, kernel, iterations=iterations)
        # dilated minus eroded gives a boundary ring; bitwise operations avoid dtype issues.
        ring = cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))
        boundary = cv2.bitwise_or(boundary, ring)
    return boundary


def derive_overlap_gt(cell_masks, kernel_size=5, iterations=2):
    """Derive overlap GT.
    Slightly dilate each cell mask and compute pairwise intersections between instances.
    The union of all pairwise intersections defines the inter-instance competition region.
    
    iterations controls dilation strength. With kernel=5, iter=2 can cover nearby instances
    that are close but not directly touching. Set iter=0 for strict overlap only.
    The default iter=2 models local uncertainty near adjacent blastomeres.
    """
    h, w = cell_masks[0].shape if cell_masks else (1, 1)
    overlap = np.zeros((h, w), dtype=np.uint8)
    if len(cell_masks) < 2:
        return overlap
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_list = [cv2.dilate(m, kernel, iterations=iterations) for m in cell_masks]
    
    # Compute pairwise intersections and accumulate them into overlap.
    n = len(dilated_list)
    for i in range(n):
        for j in range(i + 1, n):
            inter = cv2.bitwise_and(dilated_list[i], dilated_list[j])
            overlap = cv2.bitwise_or(overlap, inter)
    return overlap


# ----------------------------- Main workflow ----------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, required=True,
                        help="Path to a COCO annotation JSON file, e.g. instances_val.json")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Image directory; required only for visualization")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--vis_n", type=int, default=30,
                        help="Number of randomly sampled images for visual inspection, default 30")
    parser.add_argument("--boundary_kernel", type=int, default=5)
    parser.add_argument("--boundary_iter", type=int, default=1)
    parser.add_argument("--overlap_kernel", type=int, default=5)
    parser.add_argument("--overlap_iter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Output directories.
    os.makedirs(os.path.join(args.out_dir, "boundary"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "overlap"), exist_ok=True)
    if args.img_dir is not None:
        os.makedirs(os.path.join(args.out_dir, "visualize"), exist_ok=True)

    # Read JSON.
    print(f"[1/4] Loading annotation file: {args.ann}")
    with open(args.ann, "r") as f:
        coco = json.load(f)
    
    images = coco["images"]
    # Group annotations by image_id.
    img_id_to_anns = defaultdict(list)
    for a in coco["annotations"]:
        img_id_to_anns[a["image_id"]].append(a)
    
    # Select image IDs for visualization.
    if args.img_dir is not None and args.vis_n > 0:
        vis_ids = set(random.sample([im["id"] for im in images],
                                     min(args.vis_n, len(images))))
    else:
        vis_ids = set()

    # Derive GT.
    print(f"[2/4] Deriving boundary / overlap GT for {len(images)} images")
    skipped = 0
    stats = {"boundary_pixel_ratio": [], "overlap_pixel_ratio": []}
    
    for img in tqdm(images):
        file_name = img["file_name"]
        h, w = img["height"], img["width"]
        anns = img_id_to_anns.get(img["id"], [])
        
        # Use only cells (category_id == 1) for auxiliary GT; fragments are excluded.
        cell_masks = []
        for a in anns:
            if a["category_id"] != 1:
                continue
            m = poly_to_mask(a["segmentation"], h, w)
            if m.sum() > 0:
                cell_masks.append(m)
        
        # Derive masks.
        boundary = derive_boundary_gt(
            cell_masks,
            kernel_size=args.boundary_kernel,
            iterations=args.boundary_iter
        ) if cell_masks else np.zeros((h, w), dtype=np.uint8)
        
        overlap = derive_overlap_gt(
            cell_masks,
            kernel_size=args.overlap_kernel,
            iterations=args.overlap_iter
        ) if cell_masks else np.zeros((h, w), dtype=np.uint8)
        
        # Track positive-pixel ratios to monitor class imbalance.
        stats["boundary_pixel_ratio"].append(boundary.sum() / (h * w))
        stats["overlap_pixel_ratio"].append(overlap.sum() / (h * w))
        
        # Save as 0/255 PNG files for inspection and training.
        stem = os.path.splitext(file_name)[0]
        cv2.imwrite(os.path.join(args.out_dir, "boundary", stem + ".png"),
                    boundary * 255)
        cv2.imwrite(os.path.join(args.out_dir, "overlap", stem + ".png"),
                    overlap * 255)
        
        # Visualization.
        if img["id"] in vis_ids and args.img_dir is not None:
            img_path = os.path.join(args.img_dir, file_name)
            if not os.path.isfile(img_path):
                skipped += 1
                continue
            raw = np.array(Image.open(img_path).convert("RGB"))
            
            # Visualize the cell union.
            cell_union = np.zeros((h, w), dtype=np.uint8)
            for m in cell_masks:
                cell_union = cv2.bitwise_or(cell_union, m)
            
            # Four-panel visualization.
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes[0, 0].imshow(raw)
            axes[0, 0].set_title("Original")
            axes[0, 1].imshow(raw)
            axes[0, 1].imshow(cell_union, alpha=0.35, cmap="Reds")
            axes[0, 1].set_title(f"Cell union (N={len(cell_masks)})")
            axes[1, 0].imshow(raw)
            axes[1, 0].imshow(boundary, alpha=0.6, cmap="cool")
            axes[1, 0].set_title(f"Boundary GT "
                                  f"(ratio={boundary.sum()/(h*w)*100:.2f}%)")
            axes[1, 1].imshow(raw)
            axes[1, 1].imshow(overlap, alpha=0.6, cmap="hot")
            axes[1, 1].set_title(f"Overlap GT "
                                  f"(ratio={overlap.sum()/(h*w)*100:.2f}%)")
            for ax in axes.flat:
                ax.axis("off")
            plt.suptitle(file_name, fontsize=14)
            plt.tight_layout()
            vis_path = os.path.join(args.out_dir, "visualize", stem + ".png")
            plt.savefig(vis_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
    
    # Summary report.
    print(f"\n[3/4] Derivation completed; skipped {skipped} images because source images were missing for visualization")
    b_ratio = np.array(stats["boundary_pixel_ratio"])
    o_ratio = np.array(stats["overlap_pixel_ratio"])
    print(f"\n[Positive-pixel ratio statistics for derived GT]")
    print(f"  Boundary: mean={b_ratio.mean()*100:.2f}%  "
          f"median={np.median(b_ratio)*100:.2f}%  "
          f"max={b_ratio.max()*100:.2f}%")
    print(f"  Overlap : mean={o_ratio.mean()*100:.2f}%  "
          f"median={np.median(o_ratio)*100:.2f}%  "
          f"max={o_ratio.max()*100:.2f}%")
    # Reference ranges: boundary 1-5%; overlap 0.1-2%.
    # boundary > 10% may indicate an overly large kernel; overlap = 0 may indicate insufficient dilation or isolated instances.
    
    print(f"\n[4/4] Output directory: {args.out_dir}")
    print(f"  - boundary/  {len(images)} PNG files")
    print(f"  - overlap/   {len(images)} PNG files")
    if args.img_dir is not None:
        print(f"  - visualize/ {len(vis_ids) - skipped} four-panel visualizations for manual inspection")
    print("\n[Suggestion] During visual inspection, check three items:")
    print("  1. whether boundary covers all cell edges without obvious omissions")
    print("  2. whether overlap appears only near adjacent cell interfaces rather than over large regions")
    print("  3. overlap should be empty for single-cell images")


if __name__ == "__main__":
    main()
