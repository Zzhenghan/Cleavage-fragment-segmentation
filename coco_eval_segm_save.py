import argparse
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any, List

from pycocotools.coco import COCO
from utils import COCOeval as _COCOeval


def _ensure_info_field(src: Path) -> Path:
    """
    Ensure coco_gt.dataset contains 'info' because pycocotools.loadRes accesses it.
    """
    data = json.loads(src.read_text(encoding="utf-8"))
    if "info" not in data:
        data["info"] = {}
    fixed = src.with_name(src.stem + "_fixed" + src.suffix)
    fixed.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return fixed


def _load_dt_as_res(coco_gt: COCO, dt_json: Path) -> COCO:
    obj = json.loads(dt_json.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "annotations" in obj:
        obj = obj["annotations"]
    return coco_gt.loadRes(obj)


def _json_safe(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def eval_blastomere_instance_final(gt_path: Path, dt_path: Path, echo: bool = True, cell_cat_id: int = 1) -> Dict[str, Any]:
    gt_fixed = _ensure_info_field(gt_path)
    coco_gt = COCO(str(gt_fixed))
    coco_dt = _load_dt_as_res(coco_gt, dt_path)

    ev = _COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.params.imgIds = sorted(list(coco_gt.getImgIds()))
    ev.params.catIds = [int(cell_cat_id)]

    if echo:
        print("\n========== [FINAL] blastomere(instance segm) ==========")
        print(f"GT : {gt_path}")
        print(f"DT : {dt_path}")
        print(f"Cat: {cell_cat_id} (blastomere/cell)")

    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    return {
        "mAP_AP_50_95": float(ev.stats[0]),
        "AP_50": float(ev.stats[1]),
        "AP_75": float(ev.stats[2]),
        "AP_s": float(ev.stats[3]),
        "AP_m": float(ev.stats[4]),
        "AP_l": float(ev.stats[5]),
        "F1": float(ev.stats[12]),
        "Precision": float(ev.stats[13]),
        "Recall": float(ev.stats[14]),
    }


def eval_fragment_dice_hd_hd95(sem_gt: Path, dt_path: Path, echo: bool = True) -> Dict[str, Any]:
    sem_fixed = _ensure_info_field(sem_gt)
    coco_sem = COCO(str(sem_fixed))

    from utils import evalSemantic

    buf = io.StringIO()
    with redirect_stdout(buf):
        dice, hd, hd95 = evalSemantic(coco_sem, str(dt_path), return_hd=True)

    out = buf.getvalue()

    empty_ids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Empty GT"):
            parts = line.split()
            if len(parts) >= 3 and parts[-1].isdigit():
                empty_ids.append(int(parts[-1]))

    if echo:
        print("\n========== [FINAL] fragment(semantic segm) ==========")
        print(f"SEM_GT: {sem_gt}")
        print(f"DT    : {dt_path}")
        print(f"Dice  : {float(dice)}")
        print(f"HD    : {float(hd)}")
        print(f"HD95  : {float(hd95)}")
        if len(empty_ids) > 0:
            print(f"[INFO] Empty GT images: {len(empty_ids)}/{len(coco_sem.getImgIds())} (first 20 examples): {empty_ids[:20]}")

    return {
        "Dice": float(dice),
        "HD": float(hd),
        "HD95": float(hd95),
        "empty_gt_count": int(len(empty_ids)),
        "empty_gt_examples": empty_ids[:50],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--cat", default="-1")  # kept for CLI compatibility
    ap.add_argument("--out", required=True)
    ap.add_argument("--echo", type=int, default=1)
    ap.add_argument("--official_only", type=int, default=0)  # kept for CLI compatibility
    ap.add_argument("--sem_gt", type=str, default="")
    ap.add_argument("--paper_eval", type=str, default="")  # not used in the main flow
    ap.add_argument("--cell_cat", type=int, default=1)

    args = ap.parse_args()
    echo = bool(args.echo)

    gt_path = Path(args.gt)
    dt_path = Path(args.dt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "meta": {
            "gt": str(gt_path),
            "dt": str(dt_path),
            "sem_gt": str(args.sem_gt),
            "cell_cat": int(args.cell_cat),
            "note": "Retained paper metrics: blastomere instance AP/F1/Precision/Recall plus fragment Dice/HD95, with fragment full HD reported as an additional diagnostic.",
        }
    }

    payload["blastomere_instance_final"] = eval_blastomere_instance_final(
        gt_path, dt_path, echo=echo, cell_cat_id=int(args.cell_cat)
    )

    if args.sem_gt:
        payload["fragment_semantic_final"] = eval_fragment_dice_hd_hd95(
            Path(args.sem_gt), dt_path, echo=echo
        )
    else:
        payload["fragment_semantic_final"] = {
            "Dice": None,
            "HD": None,
            "HD95": None,
            "warning": "sem_gt not provided"
        }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe), encoding="utf-8")
    if echo:
        print(f"\n[OK] Metrics JSON written: {out_path}")


if __name__ == "__main__":
    main()
