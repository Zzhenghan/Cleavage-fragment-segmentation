import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
from pycocotools.coco import COCO
from ultralytics import YOLO
from tqdm import tqdm


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def clip_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Clip xyxy coordinates to image boundaries.
    """
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = np.clip(x1, 0, max(w - 1, 0))
    y1 = np.clip(y1, 0, max(h - 1, 0))
    x2 = np.clip(x2, 0, max(w - 1, 0))
    y2 = np.clip(y2, 0, max(h - 1, 0))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyxy_to_xywh(box: np.ndarray) -> List[float]:
    """
    xyxy -> coco bbox(x, y, w, h)
    """
    x1, y1, x2, y2 = box.astype(np.float32)
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))
    return [float(x1), float(y1), bw, bh]


def load_gt_meta(gt_json: Path) -> Dict[str, Any]:
    """
    Read GT, build a file_name -> image_id mapping, and keep original images/categories.
    """
    data = json.loads(gt_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    categories = data.get("categories", [])

    file_to_img = {}
    for img in images:
        file_to_img[img["file_name"]] = {
            "image_id": int(img["id"]),
            "width": int(img.get("width", 0)),
            "height": int(img.get("height", 0)),
        }

    return {
        "raw": data,
        "images": images,
        "categories": categories,
        "file_to_img": file_to_img,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True, help="Validation image directory")
    ap.add_argument("--gt", type=str, required=True, help="COCO GT JSON used for file_name -> image_id mapping")
    ap.add_argument("--yolo_w", type=str, required=True, help="YOLO weights path")
    ap.add_argument("--out", type=str, required=True, help="Output COCO bbox detection JSON")

    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--echo", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    gt_json = Path(args.gt)
    yolo_w = Path(args.yolo_w)
    out_json = Path(args.out)

    assert img_dir.exists(), f"[ERR] img_dir does not exist: {img_dir}"
    assert gt_json.exists(), f"[ERR] gt does not exist: {gt_json}"
    assert yolo_w.exists(), f"[ERR] yolo_w does not exist: {yolo_w}"

    ensure_dir(out_json)

    gt_meta = load_gt_meta(gt_json)
    file_to_img = gt_meta["file_to_img"]

    # Class mapping convention:
    # YOLO class 0 -> blastomere (category_id=1)
    # YOLO class 1 -> fragment (category_id=2)
    yolo2coco = {
        0: 1,
        1: 2,
    }

    model = YOLO(str(yolo_w))

    results = []
    missing_imgs = []
    no_pred_count = 0

    # Iterate in GT image order to avoid extra images from the directory.
    for img_info in tqdm(gt_meta["images"], desc="export_bbox"):
        file_name = img_info["file_name"]
        image_id = int(img_info["id"])
        gt_w = int(img_info.get("width", 0))
        gt_h = int(img_info.get("height", 0))

        img_path = img_dir / file_name
        if not img_path.exists():
            missing_imgs.append(file_name)
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            missing_imgs.append(file_name)
            continue

        h, w = img_bgr.shape[:2]
        # If width/height are missing in GT, fall back to actual image dimensions.
        if gt_w <= 0:
            gt_w = w
        if gt_h <= 0:
            gt_h = h

        pred = model.predict(
            source=img_bgr,
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            device=args.device,
            verbose=False,
        )[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            no_pred_count += 1
            continue

        boxes_xyxy = pred.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = pred.boxes.conf.detach().cpu().numpy().astype(np.float32)
        clss = pred.boxes.cls.detach().cpu().numpy().astype(np.int64)

        for box, score, cls_id in zip(boxes_xyxy, scores, clss):
            cls_id = int(cls_id)
            if cls_id not in yolo2coco:
                continue

            box = clip_xyxy(box, gt_w, gt_h)
            bbox_xywh = xyxy_to_xywh(box)

            # Filter degenerate boxes.
            if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
                continue

            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(yolo2coco[cls_id]),
                    "bbox": bbox_xywh,
                    "score": float(score),
                }
            )

    out_json.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    if int(args.echo) == 1:
        print("\n========== [YOLO -> COCO BBox] ==========")
        print(f"GT    : {gt_json}")
        print(f"IMG   : {img_dir}")
        print(f"YOLO  : {yolo_w}")
        print(f"OUT   : {out_json}")
        print(f"Images: {len(gt_meta['images'])}")
        print(f"Dets  : {len(results)}")
        print(f"NoPredImages: {no_pred_count}")
        if len(missing_imgs) > 0:
            print(f"[WARN] Missing/unreadable image count: {len(missing_imgs)}")
            print(f"[WARN] First 20 examples: {missing_imgs[:20]}")
        print("[OK] Export completed")


if __name__ == "__main__":
    main()
