"""
Dataset utilities for the dual-branch model.
Loads images, COCO instance annotations, semantic fragment masks, and auxiliary
boundary, adjacency, co-occurrence, and multi-cell projection masks.
"""

import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset

cls_to_id = {"blastomere": 1, "fragment": 2}


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


def _resize_mask_nearest(mask: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    out = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


def _resize_weight_nearest(weight_map: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    out = cv2.resize(weight_map.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.float32)


def _build_embryo_roi_and_weight_map(blast_union: np.ndarray, sem_mask: np.ndarray) -> np.ndarray:
    embryo_roi = np.logical_or(blast_union > 0, sem_mask > 0).astype(np.uint8)
    if embryo_roi.any():
        kernel = np.ones((25, 25), np.uint8)
        embryo_roi = cv2.dilate(embryo_roi, kernel, iterations=2)
        embryo_roi = cv2.morphologyEx(embryo_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(embryo_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull_mask = np.zeros_like(embryo_roi, dtype=np.uint8)
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
            embryo_roi = hull_mask
    outside_roi = 1 - embryo_roi
    return (1.0 + 2.0 * outside_roi.astype(np.float32)).astype(np.float32)


def _build_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = mask.astype(np.uint8)
    dilated = cv2.dilate(m, kernel, iterations=1)
    eroded = cv2.erode(m, kernel, iterations=1)
    return cv2.bitwise_and(dilated, cv2.bitwise_not(eroded)).astype(np.uint8)


def _build_overlap(cell_masks: List[np.ndarray], kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
    if len(cell_masks) == 0:
        return None
    h, w = cell_masks[0].shape
    overlap = np.zeros((h, w), dtype=np.uint8)
    if len(cell_masks) < 2:
        return overlap
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = [cv2.dilate(m.astype(np.uint8), kernel, iterations=iterations) for m in cell_masks]
    n = len(dilated)
    for i in range(n):
        for j in range(i + 1, n):
            inter = cv2.bitwise_and(dilated[i], dilated[j])
            overlap = cv2.bitwise_or(overlap, inter)
    return overlap


def _derive_boundary_overlap(cell_masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(cell_masks) == 0:
        return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
    h, w = cell_masks[0].shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    for m in cell_masks:
        boundary = cv2.bitwise_or(boundary, _build_boundary(m))
    overlap = _build_overlap(cell_masks, kernel_size=5, iterations=2)
    if overlap is None:
        overlap = np.zeros((h, w), dtype=np.uint8)
    return boundary, overlap


def _derive_cooccurrence(cell_masks: List[np.ndarray], sem_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Online fallback used when on-disk GT is unavailable.
    cooccurrence = cell_union AND sem_mask, where sem_mask is the fragment union.
    cc_overlap   = sum(cell_masks) >= 2
    """
    if len(cell_masks) == 0:
        h, w = sem_mask.shape
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    h, w = cell_masks[0].shape
    cell_union = np.zeros((h, w), dtype=bool)
    cell_sum = np.zeros((h, w), dtype=np.uint8)
    for m in cell_masks:
        cell_union |= (m > 0)
        cell_sum = cell_sum + (m > 0).astype(np.uint8)

    frag_bool = sem_mask.astype(bool)
    cooccurrence = (cell_union & frag_bool).astype(np.uint8)
    cc_overlap = (cell_sum >= 2).astype(np.uint8)
    return cooccurrence, cc_overlap


def _safe_read_binary_mask(mask_path: str, h: int, w: int) -> Optional[np.ndarray]:
    if not mask_path or not os.path.isfile(mask_path):
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


class COCODataset(Dataset):
    def __init__(
        self,
        root_dir,
        annotation_file,
        transform=None,
        train=True,
        aux_gt_dir: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.aux_gt_dir = aux_gt_dir
        self.coco = COCO(annotation_file)
        self.image_ids = [i for i in self.coco.imgs.keys()
                          if len(self.coco.getAnnIds(imgIds=i)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def _load_or_derive_aux(
        self, file_name: str, h: int, w: int,
        cell_masks: List[np.ndarray], sem_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (boundary_gt, overlap_gt, cooccurrence_gt, cc_overlap_gt).
        """
        stem = os.path.splitext(file_name)[0]
        boundary_gt = None
        overlap_gt = None
        cooccur_gt = None
        cc_overlap_gt = None

        if self.aux_gt_dir:
            boundary_gt = _safe_read_binary_mask(
                os.path.join(self.aux_gt_dir, "boundary", stem + ".png"), h, w,
            )
            overlap_gt = _safe_read_binary_mask(
                os.path.join(self.aux_gt_dir, "overlap", stem + ".png"), h, w,
            )
            cooccur_gt = _safe_read_binary_mask(
                os.path.join(self.aux_gt_dir, "cooccurrence", stem + ".png"), h, w,
            )
            cc_overlap_gt = _safe_read_binary_mask(
                os.path.join(self.aux_gt_dir, "cc_overlap", stem + ".png"), h, w,
            )

        # Online fallback for boundary / overlap.
        if boundary_gt is None or overlap_gt is None:
            derived_b, derived_o = _derive_boundary_overlap(cell_masks)
            if boundary_gt is None:
                boundary_gt = derived_b
            if overlap_gt is None:
                overlap_gt = derived_o

        # Online fallback for cooccurrence / cc_overlap.
        if cooccur_gt is None or cc_overlap_gt is None:
            derived_co, derived_cc = _derive_cooccurrence(cell_masks, sem_mask)
            if cooccur_gt is None:
                cooccur_gt = derived_co
            if cc_overlap_gt is None:
                cc_overlap_gt = derived_cc

        return (boundary_gt.astype(np.uint8),
                overlap_gt.astype(np.uint8),
                cooccur_gt.astype(np.uint8),
                cc_overlap_gt.astype(np.uint8))

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        file_name = image_info["file_name"]
        image_path = os.path.join(self.root_dir, file_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        bboxes = []
        ins_masks = []
        sem_masks = []
        blast_union = np.zeros((h, w), dtype=np.uint8)

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            mask = self.coco.annToMask(ann).astype(np.uint8)
            if ann["category_id"] == cls_to_id["fragment"]:
                sem_masks.append(mask)
            else:
                bboxes.append([x, y, x + bw, y + bh])
                ins_masks.append(mask)
                blast_union = np.logical_or(blast_union > 0, mask > 0).astype(np.uint8)

        if len(sem_masks) >= 1:
            sem_mask = np.bitwise_or.reduce(sem_masks, axis=0).astype(np.uint8)
        else:
            sem_mask = np.zeros((h, w), dtype=np.uint8)

        loss_weight_map = _build_embryo_roi_and_weight_map(blast_union, sem_mask)

        boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt = self._load_or_derive_aux(
            file_name=file_name, h=h, w=w,
            cell_masks=ins_masks, sem_mask=sem_mask,
        )

        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            ins_masks = []
        else:
            bboxes = np.asarray(bboxes, dtype=np.float32)

        if self.transform:
            (image, ins_masks, bboxes, sem_mask, loss_weight_map,
             boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt) = self.transform(
                image, ins_masks, bboxes, sem_mask, loss_weight_map,
                boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt,
            )

        if len(ins_masks) > 0:
            ins_masks = np.stack(ins_masks, axis=0).astype(np.float32)
        else:
            ins_masks = np.zeros((0, sem_mask.shape[0], sem_mask.shape[1]), dtype=np.float32)

        return (
            image,
            torch.as_tensor(bboxes, dtype=torch.float32),
            torch.as_tensor(ins_masks, dtype=torch.float32),
            torch.as_tensor(sem_mask, dtype=torch.long),
            torch.as_tensor(loss_weight_map, dtype=torch.float32),
            torch.as_tensor(boundary_gt, dtype=torch.float32),
            torch.as_tensor(overlap_gt, dtype=torch.float32),
            torch.as_tensor(cooccur_gt, dtype=torch.float32),
            torch.as_tensor(cc_overlap_gt, dtype=torch.float32),
        )


def collate_fn(batch):
    (images, bboxes, ins_masks, sem_mask, loss_weight_map,
     boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt) = zip(*batch)
    images = torch.stack(images, dim=0)
    sem_mask = torch.stack(sem_mask, dim=0)
    loss_weight_map = torch.stack(loss_weight_map, dim=0)
    boundary_gt = torch.stack(boundary_gt, dim=0)
    overlap_gt = torch.stack(overlap_gt, dim=0)
    cooccur_gt = torch.stack(cooccur_gt, dim=0)
    cc_overlap_gt = torch.stack(cc_overlap_gt, dim=0)
    return (images, bboxes, ins_masks, sem_mask, loss_weight_map,
            boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt)


class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)

    def __call__(self, image, ins_masks, bboxes, sem_mask, loss_weight_map,
                 boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt):
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        new_h, new_w = image.shape[:2]
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()

        ins_masks = [_resize_mask_nearest(m, new_h, new_w) for m in ins_masks]
        sem_mask = _resize_mask_nearest(np.asarray(sem_mask), new_h, new_w)
        boundary_gt = _resize_mask_nearest(np.asarray(boundary_gt), new_h, new_w)
        overlap_gt = _resize_mask_nearest(np.asarray(overlap_gt), new_h, new_w)
        cooccur_gt = _resize_mask_nearest(np.asarray(cooccur_gt), new_h, new_w)
        cc_overlap_gt = _resize_mask_nearest(np.asarray(cc_overlap_gt), new_h, new_w)
        loss_weight_map = _resize_weight_nearest(
            np.asarray(loss_weight_map, dtype=np.float32), new_h, new_w,
        )

        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        image = F.pad(image, (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h), value=0)

        def _pad_mask(m, v=0):
            t = torch.as_tensor(m, dtype=torch.float32)
            return F.pad(t, (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h), value=v).numpy().astype(np.uint8)

        padded_ins = [_pad_mask(m) for m in ins_masks]
        ins_masks = padded_ins

        sem_mask = _pad_mask(sem_mask)
        boundary_gt = _pad_mask(boundary_gt)
        overlap_gt = _pad_mask(overlap_gt)
        cooccur_gt = _pad_mask(cooccur_gt)
        cc_overlap_gt = _pad_mask(cc_overlap_gt)

        loss_weight_map = F.pad(
            torch.as_tensor(loss_weight_map, dtype=torch.float32),
            (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h), value=3.0,
        ).numpy().astype(np.float32)

        if bboxes is not None and len(bboxes) > 0:
            bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
            bboxes = np.asarray(
                [[b[0] + pad_w, b[1] + pad_h, b[2] + pad_w, b[3] + pad_h] for b in bboxes],
                dtype=np.float32,
            )
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        return (image, ins_masks, bboxes, sem_mask, loss_weight_map,
                boundary_gt, overlap_gt, cooccur_gt, cc_overlap_gt)


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(
        root_dir=cfg.dataset.train.root_dir,
        annotation_file=cfg.dataset.train.annotation_file,
        transform=transform, train=True,
        aux_gt_dir=_cfg_get(cfg, ["dataset", "train", "aux_gt_dir"], None),
    )
    val = COCODataset(
        root_dir=cfg.dataset.val.root_dir,
        annotation_file=cfg.dataset.val.annotation_file,
        transform=transform, train=False,
        aux_gt_dir=_cfg_get(cfg, ["dataset", "val", "aux_gt_dir"], None),
    )
    train_dataloader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    from config import cfg
    train, val = load_datasets(cfg, 1024)
    print(f"[dataset test] train={len(train.dataset)}, val={len(val.dataset)}")
