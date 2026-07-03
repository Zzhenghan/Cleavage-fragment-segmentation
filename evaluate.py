from pathlib import Path
import argparse

from pycocotools.coco import COCO
from utils import COCOeval, evalSemantic, build_fragment_gt_from_instances


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins_gt", type=str, default="./data/annotations/instances_val.json")
    ap.add_argument("--segm_dt", type=str, default="./outputs/sam_dual_res.json")
    ap.add_argument("--sem_gt", type=str, default="./data/annotations/fragment_res.json")
    ap.add_argument("--frag_cat_id", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    ins_gt = Path(args.ins_gt)
    segm_dt = Path(args.segm_dt)
    sem_gt = Path(args.sem_gt)

    assert ins_gt.exists(), f"[ERR] GT not found: {ins_gt}"
    assert segm_dt.exists(), f"[ERR] DT not found: {segm_dt}"

    if not sem_gt.exists():
        print(f"[WARN] semantic GT not found: {sem_gt}; building it from instance GT.")
        build_fragment_gt_from_instances(str(ins_gt), str(sem_gt), frag_cat_id=int(args.frag_cat_id))

    print("========== [Evaluator: blastomere segmentation] ==========")
    print(f"GT: {ins_gt}")
    print(f"DT: {segm_dt}")

    coco_gt = COCO(str(ins_gt))
    coco_dt = coco_gt.loadRes(str(segm_dt))
    img_ids = sorted(coco_gt.getImgIds())
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.imgIds = img_ids
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if getattr(coco_eval, "stats", None) is not None and len(coco_eval.stats) >= 15:
        print(f"F1: {float(coco_eval.stats[12]):.6f}")
        print(f"Precision: {float(coco_eval.stats[13]):.6f}")
        print(f"Recall: {float(coco_eval.stats[14]):.6f}")

    print("========== [Evaluator: fragment semantic mask] ==========")
    coco_sem_gt = COCO(str(sem_gt))
    sem_dice, sem_hd, sem_hd95 = evalSemantic(coco_sem_gt, str(segm_dt), return_hd=True)
    print(f"Fragment Dice: {sem_dice:.6f}")
    print(f"Fragment HD: {sem_hd:.6f}")
    print(f"Fragment HD95: {sem_hd95:.6f}")


if __name__ == "__main__":
    main()
