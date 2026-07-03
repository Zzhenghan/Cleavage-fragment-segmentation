import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _to_jsonable(obj):
    """Convert NumPy scalars and arrays to JSON-serializable Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def ensure_info_field(gt_path: Path) -> Path:
    """
    Handle non-strict COCO annotations:
    pycocotools.COCO.loadRes may fail when top-level "info" or "licenses" fields are missing.
    If fields are missing, write a temporary *_fixed.json file and return that path; otherwise return the original path.
    """
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    changed = False

    if "info" not in data:
        data["info"] = {}
        changed = True
    if "licenses" not in data:
        data["licenses"] = []
        changed = True

    for img in data.get("images", []):
        if isinstance(img, dict) and "info" not in img:
            img["info"] = {}
            changed = True

    if not changed:
        return gt_path

    fixed = gt_path.with_name(gt_path.stem + "_fixed.json")
    fixed.write_text(json.dumps(_to_jsonable(data), ensure_ascii=False), encoding="utf-8")
    return fixed


def load_dt_as_res(coco_gt: COCO, dt_path: Path):
    obj = json.loads(dt_path.read_text(encoding="utf-8"))
    return coco_gt.loadRes(obj)


def parse_cat_ids(s: str):
    s = str(s).strip()
    if s == "-1" or s.lower() == "all":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(x) for x in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--cat", default="-1", help="-1=all or '1,2'")
    ap.add_argument("--out", default="")
    ap.add_argument("--echo", type=int, default=1)
    args = ap.parse_args()

    gt_path = Path(args.gt)
    dt_path = Path(args.dt)

    gt_path_fixed = ensure_info_field(gt_path)

    coco_gt = COCO(str(gt_path_fixed))
    coco_dt = load_dt_as_res(coco_gt, dt_path)

    cat_ids = parse_cat_ids(args.cat)
    print("\n========== [COCO BBox] Evaluate ==========")
    print(f"GT: {gt_path}")
    print(f"DT: {dt_path}")
    print("Cat: all" if cat_ids is None else f"Cat: {cat_ids}")

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if cat_ids is not None:
        ev.params.catIds = cat_ids

    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    names = [
        "AP@[0.50:0.95]",
        "AP@0.50",
        "AP@0.75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR@1",
        "AR@10",
        "AR@100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    metrics = {k: float(v) for k, v in zip(names, ev.stats.tolist())}

    payload = {
        "gt": str(gt_path),
        "dt": str(dt_path),
        "cat": "all" if cat_ids is None else cat_ids,
        "metrics": metrics,
        "cocoeval_params": {
            "iouType": ev.params.iouType,
            "iouThrs": ev.params.iouThrs,
            "recThrs": ev.params.recThrs,
            "maxDets": ev.params.maxDets,
            "areaRng": ev.params.areaRng,
            "areaRngLbl": ev.params.areaRngLbl,
            "useCats": int(ev.params.useCats),
            "img_ids": ev.params.imgIds,
            "cat_ids": ev.params.catIds,
        },
    }

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload_json = _to_jsonable(payload)
        out_p.write_text(json.dumps(payload_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] Metrics JSON written: {out_p}")


if __name__ == "__main__":
    main()
