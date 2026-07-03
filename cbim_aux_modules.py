"""
Auxiliary modules for fragment-side supervision and fusion.
Contains the boundary/overlap auxiliary heads, co-occurrence and multi-cell
projection heads (C1), the cell-to-fragment gate, the evidence-fusion module
(C2), and the associated losses.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_EMPTY_LOGIT_FILL = -100.0  # sigmoid(-100) is approximately 0, so the gate is inactive.


# =========================
# Basic modules.
# =========================
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AuxiliaryHeads(nn.Module):
    """Boundary and overlap auxiliary heads."""
    def __init__(self, in_ch: int = 256, mid_ch: int = 64):
        super().__init__()
        self.boundary_head = nn.Sequential(
            ConvGNAct(in_ch, mid_ch, 3, 1),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )
        self.overlap_head = nn.Sequential(
            ConvGNAct(in_ch, mid_ch, 3, 1),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, image_embedding: torch.Tensor, out_size):
        b_logits = self.boundary_head(image_embedding)
        o_logits = self.overlap_head(image_embedding)
        b_out = F.interpolate(b_logits, size=out_size, mode="bilinear", align_corners=False)
        o_out = F.interpolate(o_logits, size=out_size, mode="bilinear", align_corners=False)
        return b_out, o_out


# =========================
# =========================
class CoOccurrenceHead(nn.Module):
    """Predicts the cell-fragment co-occurrence region. Trained with BCE + Dice."""
    def __init__(self, in_ch: int = 256, mid_ch: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvGNAct(in_ch, mid_ch, 3, 1),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, image_embedding: torch.Tensor, out_size):
        logits = self.head(image_embedding)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class CCOverlapHead(nn.Module):
    """Predicts strict multi-cell projection regions. Trained with BCE + Dice."""
    def __init__(self, in_ch: int = 256, mid_ch: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvGNAct(in_ch, mid_ch, 3, 1),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, image_embedding: torch.Tensor, out_size):
        logits = self.head(image_embedding)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


# =========================
# =========================
class APAMEvidenceFusion(nn.Module):
    """
    Evidence-fusion module (C2). Concatenates six logit-space evidence
    channels and predicts an additive residual that refines the fragment logit:
        refined = frag_gated + scale * delta,  scale = softplus(raw_scale)

    The final convolution is zero-initialized so the initial refined output
    matches the gated baseline. The residual scale remains positive through
    softplus and is initialized near 1.0.
    """
    def __init__(
        self,
        in_channels: int = 6,
        mid_channels: int = 16,
        enable: bool = True,
        scale_init: float = 1.0,
    ):
        super().__init__()
        self.enable = enable
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )
        # Zero initialization makes the initial delta equal to 0.
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

        # Invert softplus so softplus(raw_scale) equals scale_init.
        if scale_init <= 0:
            raise ValueError(f"scale_init must be > 0, got: {scale_init}")
        import math
        raw_init = math.log(math.exp(scale_init) - 1.0)
        self.raw_scale = nn.Parameter(torch.tensor(float(raw_init), dtype=torch.float32))

    @property
    def gamma(self) -> torch.Tensor:
        """
        Compatibility field returning the current scale value, softplus(raw_scale).
        This keeps compatibility with code that reads model.apam_fusion.gamma.
        The value represents the current fusion scale, not an additive gamma term.
        """
        return F.softplus(self.raw_scale).detach()

    def forward(
        self,
        frag_logit: torch.Tensor,
        boundary_logit: torch.Tensor,
        overlap_logit: torch.Tensor,
        cooccur_logit: torch.Tensor,
        cc_overlap_logit: torch.Tensor,
        cell_union_logit: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable:
            return frag_logit
        x = torch.cat([
            frag_logit, boundary_logit, overlap_logit,
            cooccur_logit, cc_overlap_logit, cell_union_logit,
        ], dim=1)
        delta = self.fuse(x)
        scale = F.softplus(self.raw_scale)
        return frag_logit + scale * delta


# =========================
# Utility functions.
# =========================
def build_cell_union_logit_from_masks(cell_masks: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    if cell_masks is None or cell_masks.numel() == 0:
        return torch.full_like(ref_tensor, _EMPTY_LOGIT_FILL)

    if cell_masks.dim() == 3:
        cell_masks = cell_masks.unsqueeze(1)
    elif cell_masks.dim() == 4:
        if cell_masks.shape[1] > 1:
            cell_masks = torch.max(cell_masks, dim=1, keepdim=True)[0]
    else:
        raise ValueError(f"Unsupported cell_masks dimensions: {tuple(cell_masks.shape)}")

    if cell_masks.shape[-2:] != ref_tensor.shape[-2:]:
        cell_masks = F.interpolate(
            cell_masks, size=ref_tensor.shape[-2:],
            mode="bilinear", align_corners=False,
        )
    union_logit = torch.max(cell_masks, dim=0, keepdim=True)[0]
    return union_logit


class CellToFragmentGate(nn.Module):
    """Additive penalty gate (op3)."""
    def __init__(self, penalty_init: float = 10.0, detach_gate: bool = True, enable: bool = True):
        super().__init__()
        self.penalty = nn.Parameter(torch.tensor(float(penalty_init)))
        self.detach_gate = detach_gate
        self.enable = enable

    def forward(self, cell_union_logit: torch.Tensor, frag_logit: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return frag_logit
        cell_union_prob = torch.sigmoid(cell_union_logit)
        if self.detach_gate:
            cell_union_prob = cell_union_prob.detach()
        penalty = F.softplus(self.penalty)
        return frag_logit - penalty * cell_union_prob


# =========================
# Loss functions.
# =========================
def dice_loss_2channel(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Two-channel Dice loss in the softmax domain."""
    if target.dim() == 4:
        target = target.squeeze(1)
    assert target.dim() == 3, f"target dim expected 3, got {target.dim()}"

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice_per_class.mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, smooth=1e-6):
    prob = torch.sigmoid(logits).reshape(logits.size(0), -1)
    target = target.reshape(target.size(0), -1).float()
    tp = (prob * target).sum(dim=1)
    fp = (prob * (1.0 - target)).sum(dim=1)
    fn = ((1.0 - prob) * target).sum(dim=1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def sigmoid_dice_loss(logits, target, smooth=1e-6):
    prob = torch.sigmoid(logits).reshape(logits.size(0), -1)
    target = target.reshape(target.size(0), -1).float()
    intersection = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


# =========================
# =========================
def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[float] = None,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Designed for sparse positive classes:
      BCE with pos_weight discourages the all-zero local optimum.
      Dice loss encourages positive-class overlap.
    """
    if logits.dim() == 4 and logits.shape[1] == 1:
        logits_s = logits.squeeze(1)
    elif logits.dim() == 3:
        logits_s = logits
    else:
        raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

    if target.dim() == 4 and target.shape[1] == 1:
        target_s = target.squeeze(1)
    elif target.dim() == 3:
        target_s = target
    else:
        raise ValueError(f"Unsupported target shape: {tuple(target.shape)}")

    target_float = target_s.float()

    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=logits_s.device)
        bce = F.binary_cross_entropy_with_logits(logits_s, target_float, pos_weight=pw)
    else:
        bce = F.binary_cross_entropy_with_logits(logits_s, target_float)

    prob = torch.sigmoid(logits_s)
    B = prob.shape[0]
    pf = prob.reshape(B, -1)
    tf = target_float.reshape(B, -1)
    inter = (pf * tf).sum(dim=1)
    denom = pf.sum(dim=1) + tf.sum(dim=1)
    dice_per_img = (2.0 * inter + smooth) / (denom + smooth)
    dice_loss_val = 1.0 - dice_per_img.mean()

    return bce_weight * bce + dice_weight * dice_loss_val


# =========================
# =========================
def compute_region_dice(
    frag_pred_mask: torch.Tensor,
    frag_gt: torch.Tensor,
    cooccur_gt: torch.Tensor,
    cc_overlap_gt: torch.Tensor,
) -> dict:
    """
    Region-wise fragment Dice for analyzing layered semantic separation.

    Region definitions:
      bg        = ~cooccur & ~cc_overlap
      cf_only   = cooccur & ~cc_overlap     (cell-fragment co-presence only)
      cc_only   = cc_overlap & ~cooccur     (cell-cell projection overlap only)
      cf_and_cc = cooccur & cc_overlap      (intersection region)

    Args:
        frag_pred_mask: [B, H, W] bool  predicted fragment mask
        frag_gt:        [B, H, W] bool  GT fragment mask (sem_gt == 1)
        cooccur_gt:     [B, H, W] bool
        cc_overlap_gt:  [B, H, W] bool

    Returns:
        {'cf_only': f, 'cc_only': f, 'cf_and_cc': f, 'bg': f}
        Returns -1.0 when a region is empty.
    """
    regions = {
        "cf_only":   cooccur_gt & (~cc_overlap_gt),
        "cc_only":   cc_overlap_gt & (~cooccur_gt),
        "cf_and_cc": cooccur_gt & cc_overlap_gt,
        "bg":        (~cooccur_gt) & (~cc_overlap_gt),
    }
    result = {}
    for name, region_mask in regions.items():
        pred_in = (frag_pred_mask & region_mask).float()
        gt_in = (frag_gt & region_mask).float()
        inter = (pred_in * gt_in).sum()
        denom = pred_in.sum() + gt_in.sum()
        if denom.item() < 1e-6:
            result[name] = -1.0
        else:
            result[name] = ((2.0 * inter + 1e-6) / (denom + 1e-6)).item()
    return result


# =========================
# =========================
