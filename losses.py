import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class SemDiceLoss(nn.Module):
    """Semantic Dice loss for 2-class fragment segmentation."""
    def __init__(self, n_classes):
        super(SemDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-4
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1.0, 2.225] if self.n_classes == 2 else [1.0] * self.n_classes
        assert inputs.size() == target.size(), (
            "predict {} & target {} shape do not match".format(inputs.size(), target.size())
        )
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * float(weight[i])
        return loss / self.n_classes


# =========================
# Root-Fix: Fragment Loss
# =========================

class FragmentDiceLoss(nn.Module):
    """
    Dice loss for the fragment channel (class=1).
    logits: [B, 2, H, W]
    target: [B, H, W], value range {0,1}
    """
    def __init__(self, smooth: float = 1.0, fragment_channel: int = 1):
        super().__init__()
        self.smooth = smooth
        self.fragment_channel = fragment_channel

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        prob_fg = probs[:, self.fragment_channel]  # [B,H,W]
        target_fg = (target == self.fragment_channel).float()

        dims = (1, 2)
        intersection = (prob_fg * target_fg).sum(dim=dims)
        denom = prob_fg.sum(dim=dims) + target_fg.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class WeightedBCELoss(nn.Module):
    """
    Pixelwise BCE for the fragment channel, supporting:
    - pos_weight: increases positive-sample weight for sparse positives
    - weight_map: weights hard-negative regions outside the embryo
    """
    def __init__(self, pos_weight: float = 5.0, fragment_channel: int = 1):
        super().__init__()
        self.pos_weight = pos_weight
        self.fragment_channel = fragment_channel

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: torch.Tensor = None
    ) -> torch.Tensor:
        frag_logit = logits[:, self.fragment_channel]          # [B,H,W]
        target_fg = (target == self.fragment_channel).float()  # [B,H,W]

        pos_weight = torch.tensor(self.pos_weight, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            frag_logit,
            target_fg,
            pos_weight=pos_weight,
            reduction="none",
        )  # [B,H,W]

        if weight_map is not None:
            weight_map = weight_map.to(device=logits.device, dtype=loss.dtype)
            loss = loss * weight_map

        return loss.mean()


class FragmentDiceBCELoss(nn.Module):
    """
    Numerical stability detail:
    L_sem = dice_weight * Dice + bce_weight * WeightedBCE
    """
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        pos_weight: float = 5.0,
        fragment_channel: int = 1,
    ):
        super().__init__()
        self.dice = FragmentDiceLoss(fragment_channel=fragment_channel)
        self.bce = WeightedBCELoss(
            pos_weight=pos_weight,
            fragment_channel=fragment_channel
        )
        self.dw = float(dice_weight)
        self.bw = float(bce_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: torch.Tensor = None
    ) -> torch.Tensor:
        loss_dice = self.dice(logits, target)
        loss_bce = self.bce(logits, target, weight_map=weight_map)
        return self.dw * loss_dice + self.bw * loss_bce
