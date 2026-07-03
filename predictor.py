"""
Inference helper for the dual-branch model.
Runs the semantic fragment branch through the same C2 evidence fusion used by
training and returns refined fragment masks or probabilities.
"""

import numpy as np
import torch
from typing import Optional, Tuple

from model import Model
from segment_anything.utils.transforms import ResizeLongestSide
from cbim_aux_modules import build_cell_union_logit_from_masks


class ModelPredictor:
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self.transform = ResizeLongestSide(model.sam.image_encoder.img_size)
        self.reset_image()

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in ["RGB", "BGR"]
        if image_format != self.model.sam.image_format:
            image = image[..., ::-1]
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(self, transformed_image: torch.Tensor, original_image_size: Tuple[int, ...]) -> None:
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.sam.image_encoder.img_size
        )
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.sam.preprocess(transformed_image)
        self.features = self.model.sam.image_encoder(input_image)
        self.is_image_set = True

    def _encode_prompts(self, point_coords, point_labels, boxes, mask_input):
        points = None
        if point_coords is not None:
            if point_coords.dim() == 2:
                point_coords = point_coords.unsqueeze(0)
            if point_labels is None:
                raise ValueError("point_coords must be used together with point_labels")
            if point_labels.dim() == 1:
                point_labels = point_labels.unsqueeze(0)
            points = (point_coords, point_labels)
        if boxes is not None:
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)
            boxes = boxes.to(device=self.device, dtype=torch.float)
        if mask_input is not None:
            if mask_input.dim() == 2:
                mask_input = mask_input.unsqueeze(0).unsqueeze(0)
            elif mask_input.dim() == 3:
                mask_input = mask_input.unsqueeze(1)
            mask_input = mask_input.to(device=self.device, dtype=torch.float)
        return points, boxes, mask_input

    def _empty_instance_outputs(self, multimask_output: bool):
        c = 3 if multimask_output else 1
        empty_masks = torch.zeros(
            (0, c, self.original_size[0], self.original_size[1]),
            dtype=torch.float32, device=self.device,
        )
        empty_iou = torch.zeros((0, c), dtype=torch.float32, device=self.device)
        empty_low = torch.zeros((0, c, 256, 256), dtype=torch.float32, device=self.device)
        return empty_masks, empty_iou, empty_low

    def _predict_instance_torch(
        self, point_coords, point_labels, boxes, mask_input,
        multimask_output: bool, return_logits: bool,
    ):
        has_prompt = (
            (point_coords is not None)
            or (boxes is not None and boxes.numel() > 0)
            or (mask_input is not None)
        )
        if not has_prompt:
            return (*self._empty_instance_outputs(multimask_output), None)

        points, boxes, mask_input = self._encode_prompts(
            point_coords, point_labels, boxes, mask_input,
        )
        sparse_embeddings, dense_embeddings = self.model.sam.prompt_encoder(
            points=points, boxes=boxes, masks=mask_input,
        )
        low_res_masks, iou_predictions = self.model.sam.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks_logits = self.model.sam.postprocess_masks(
            low_res_masks, self.input_size, self.original_size,
        )
        masks_out = masks_logits if return_logits else (
            masks_logits > self.model.sam.mask_threshold
        )
        return masks_out, iou_predictions, low_res_masks, masks_logits

    def _predict_aux_logits_torch(self):
        b_logits, o_logits = self.model.aux_heads(self.features, out_size=self.original_size)
        return b_logits, o_logits

    def _predict_apam_logits_torch(self):
        """APAM logits from the two auxiliary heads."""
        H, W = self.original_size
        zero = torch.zeros((1, 1, H, W), dtype=self.features.dtype, device=self.device)
        if getattr(self.model, "cooccur_enable", True):
            co = self.model.cooccur_head(self.features, out_size=self.original_size)
        else:
            co = zero
        if getattr(self.model, "cc_overlap_enable", True):
            cc = self.model.cc_overlap_head(self.features, out_size=self.original_size)
        else:
            cc = zero
        return co, cc

    def _predict_semantic_logits_torch(
        self,
        instance_logits_for_gate: Optional[torch.Tensor] = None,
        boundary_logits: Optional[torch.Tensor] = None,
        overlap_logits: Optional[torch.Tensor] = None,
        cooccur_logits: Optional[torch.Tensor] = None,
        cc_overlap_logits: Optional[torch.Tensor] = None,
    ):
        """Semantic path matching model.forward."""
        sem_logits_raw = self.model.semantic_decoder(self.features)
        sem_logits_raw = self.model.sam.postprocess_masks(
            sem_logits_raw, self.input_size, self.original_size,
        )

        ref_tensor = torch.zeros(
            (1, 1, sem_logits_raw.shape[-2], sem_logits_raw.shape[-1]),
            dtype=sem_logits_raw.dtype, device=sem_logits_raw.device,
        )

        if instance_logits_for_gate is None or instance_logits_for_gate.numel() == 0:
            cell_union_logit = build_cell_union_logit_from_masks(
                torch.zeros((0,) + ref_tensor.shape[-2:], device=self.device),
                ref_tensor=ref_tensor,
            )
        else:
            cell_union_logit = build_cell_union_logit_from_masks(
                instance_logits_for_gate, ref_tensor=ref_tensor,
            )

        bg_logits = sem_logits_raw[:, 0:1]
        frag_logits_raw = sem_logits_raw[:, 1:2]

        # op1
        alpha = float(getattr(self.model, "op1_alpha", 0.0))
        if alpha > 0 and boundary_logits is not None:
            bias = alpha * torch.sigmoid(boundary_logits)
            frag_logits_raw = frag_logits_raw + bias
            cell_union_logit = cell_union_logit + bias

        # op3
        frag_logits_gated = self.model.cell_to_fragment_gate(cell_union_logit, frag_logits_raw)

        # APAM fusion matching the training path.
        cell_union_logit_for_fusion = cell_union_logit
        if getattr(self.model.cell_to_fragment_gate, "detach_gate", False):
            cell_union_logit_for_fusion = cell_union_logit_for_fusion.detach()
        if all(x is not None for x in [boundary_logits, overlap_logits,
                                        cooccur_logits, cc_overlap_logits]):
            frag_logits_refined = self.model.apam_fusion(
                frag_logit=frag_logits_gated,
                boundary_logit=boundary_logits,
                overlap_logit=overlap_logits,
                cooccur_logit=cooccur_logits,
                cc_overlap_logit=cc_overlap_logits,
                cell_union_logit=cell_union_logit_for_fusion,
            )
        else:
            frag_logits_refined = frag_logits_gated

        sem_logits_refined = torch.cat([bg_logits, frag_logits_refined], dim=1)
        return sem_logits_refined, cell_union_logit

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        return_sem_prob: bool = False,
        return_overlap_prob: bool = False,
    ):
        """
        Return order:
            masks_np, iou_predictions_np, low_res_masks_np, sem_mask_np
            [if return_sem_prob]    sem_prob_np         (H, W) frag refined prob
            [if return_overlap_prob] overlap_prob_np    (H, W) aux overlap prob
        """
        if not self.is_image_set:
            raise RuntimeError("Call .set_image(...) before predict().")

        if point_coords is not None:
            point_coords = np.asarray(point_coords)
            if point_coords.ndim == 1:
                point_coords = point_coords[None, :]
            point_coords = self.transform.apply_coords(point_coords, self.original_size)

        if point_labels is not None:
            point_labels = np.asarray(point_labels)
            if point_labels.ndim == 0:
                point_labels = point_labels[None]

        box_is_single = False
        if box is not None:
            box = np.asarray(box)
            if box.ndim == 1:
                box_is_single = True
                box = box[None, :]
            elif box.ndim == 2 and box.shape[0] == 1:
                box_is_single = True
            box = self.transform.apply_boxes(box, self.original_size)

        if mask_input is not None:
            mask_input = np.asarray(mask_input)
            if mask_input.ndim == 2:
                mask_input = mask_input[None, :, :]
            if mask_input.ndim == 3:
                mask_input = mask_input[:, None, :, :]

        ret = self.predict_torch(
            point_coords=torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            if point_coords is not None else None,
            point_labels=torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if point_labels is not None else None,
            boxes=torch.as_tensor(box, dtype=torch.float, device=self.device)
            if box is not None else None,
            mask_input=torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            if mask_input is not None else None,
            multimask_output=multimask_output,
            return_logits=return_logits,
            return_sem_prob=return_sem_prob,
            return_overlap_prob=return_overlap_prob,
        )
        (masks, iou_predictions, low_res_masks,
         sem_mask, sem_prob, overlap_prob) = ret

        masks_np = masks.detach().cpu().numpy()
        iou_predictions_np = iou_predictions.detach().cpu().numpy()
        low_res_masks_np = low_res_masks.detach().cpu().numpy()
        sem_mask_np = sem_mask[0].astype(np.uint8)

        if box is not None and box_is_single:
            masks_np = masks_np[0]
            iou_predictions_np = iou_predictions_np[0]
            low_res_masks_np = low_res_masks_np[0]
            if not multimask_output:
                masks_np = masks_np[0]
                low_res_masks_np = low_res_masks_np[0]
        else:
            if not multimask_output and masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0]
            if not multimask_output and low_res_masks_np.ndim == 4 and low_res_masks_np.shape[1] == 1:
                low_res_masks_np = low_res_masks_np[:, 0]

        out = [masks_np, iou_predictions_np, low_res_masks_np, sem_mask_np]
        if return_sem_prob:
            out.append(sem_prob[0].detach().cpu().numpy().astype(np.float32))
        if return_overlap_prob:
            out.append(overlap_prob[0].detach().cpu().numpy().astype(np.float32))
        return tuple(out)

    @torch.no_grad()
    def predict_torch(
        self, point_coords, point_labels,
        boxes=None, mask_input=None,
        multimask_output: bool = True,
        return_logits: bool = False,
        return_sem_prob: bool = False,
        return_overlap_prob: bool = False,
    ):
        if not self.is_image_set:
            raise RuntimeError("Call .set_image(...) before predict_torch().")

        masks, iou_predictions, low_res_masks, masks_logits = self._predict_instance_torch(
            point_coords=point_coords, point_labels=point_labels,
            boxes=boxes, mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

        boundary_logits, overlap_logits = self._predict_aux_logits_torch()
        cooccur_logits, cc_overlap_logits = self._predict_apam_logits_torch()

        sem_logits_refined, _ = self._predict_semantic_logits_torch(
            instance_logits_for_gate=masks_logits,
            boundary_logits=boundary_logits,
            overlap_logits=overlap_logits,
            cooccur_logits=cooccur_logits,
            cc_overlap_logits=cc_overlap_logits,
        )

        sem_prob = torch.softmax(sem_logits_refined, dim=1)
        sem_mask = torch.argmax(sem_prob, dim=1).cpu().numpy().astype(np.uint8)

        sem_prob_fg = sem_prob[:, 1] if return_sem_prob else None
        overlap_prob = torch.sigmoid(overlap_logits)[:, 0] if return_overlap_prob else None

        return (masks, iou_predictions, low_res_masks, sem_mask,
                sem_prob_fg, overlap_prob)

    @torch.no_grad()
    def predict_semantic(self, box: Optional[np.ndarray] = None, return_prob: bool = False):
        if not self.is_image_set:
            raise RuntimeError("Call .set_image(...) before predict_semantic().")

        boxes_torch = None
        if box is not None:
            box = np.asarray(box)
            if box.ndim == 1:
                box = box[None, :]
            box = self.transform.apply_boxes(box, self.original_size)
            boxes_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)

        instance_logits = None
        if boxes_torch is not None and boxes_torch.numel() > 0:
            _, _, _, instance_logits = self._predict_instance_torch(
                point_coords=None, point_labels=None,
                boxes=boxes_torch, mask_input=None,
                multimask_output=False, return_logits=True,
            )

        boundary_logits, overlap_logits = self._predict_aux_logits_torch()
        cooccur_logits, cc_overlap_logits = self._predict_apam_logits_torch()

        sem_logits_refined, _ = self._predict_semantic_logits_torch(
            instance_logits_for_gate=instance_logits,
            boundary_logits=boundary_logits,
            overlap_logits=overlap_logits,
            cooccur_logits=cooccur_logits,
            cc_overlap_logits=cc_overlap_logits,
        )
        sem_prob = torch.softmax(sem_logits_refined, dim=1)
        sem_mask = torch.argmax(sem_prob, dim=1).cpu().numpy().astype(np.uint8)[0]

        if return_prob:
            prob = sem_prob[0, 1].detach().cpu().numpy().astype(np.float32)
            return sem_mask, prob
        return sem_mask

    def get_image_embedding(self) -> torch.Tensor:
        if not self.is_image_set:
            raise RuntimeError("Call .set_image(...) before get_image_embedding().")
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.sam.device

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None
