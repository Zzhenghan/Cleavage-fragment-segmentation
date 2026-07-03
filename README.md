# Cleavage Embryo Research Code Release

This folder contains a reduced research-code release for method inspection and partial reproduction of the manuscript methods. It includes model components, training structure, auxiliary-target derivation utilities, and evaluation utilities.

Not included: full training data, trained checkpoints, production inference pipelines, internal experiment automation, figure-generation assets, and private result archives.

## Included Components

- C1 auxiliary heads: co-occurrence and multi-cell projection supervision.
- Cell-to-fragment gate.
- C2 evidence-fusion module.
- Dual-branch training structure.
- Auxiliary target derivation scripts.
- COCO-style segmentation and bbox evaluation utilities.

## Expected Layout

Set paths through environment variables or edit `config.py` locally:

```text
DATA_ROOT=./data
WEIGHT_ROOT=./weights
CODE_ROOT=.
SAM_CKPT=./weights/sam_vit_b_01ec64.pth
YOLO_BEST=./weights/yolo11_best.pt
```

Expected data layout:

```text
data/
  images/train/
  images/val/
  annotations/instances_train.json
  annotations/instances_val.json
  annotations/fragment_res.json
  aux_gt/train/{boundary,overlap,cooccurrence,cc_overlap}/
  aux_gt/val/{boundary,overlap,cooccurrence,cc_overlap}/
```

## Basic Commands

```bash
python derive_aux_gt.py --ann ./data/annotations/instances_val.json --img_dir ./data/images/val --out_dir ./data/aux_gt/val
python derive_cooccurrence_gt.py --split val
python train_es.py --help
python evaluate.py --help
```

## Terminology Mapping

| Code identifier | Manuscript term |
|---|---|
| `CoOccurrenceHead`, `cooccur` | cell-fragment co-occurrence head |
| `CCOverlapHead`, `cc_overlap` | strict multi-cell projection head |
| `overlap` | dilated inter-cell adjacency cue |
| `CellToFragmentGate` | cell-fragment gate |
| `APAMEvidenceFusion` | evidence fusion module |
| `cell_union_logit` | cell-occupancy prior |

The code retains development-time tensor names where changing them would affect checkpoint compatibility.
