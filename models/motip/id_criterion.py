# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import is_distributed, distributed_world_size, labels_to_one_hot
from utils.box_ops import box_cxcywh_to_xyxy, pairwise_giou


class IDCriterion(nn.Module):
    def __init__(
            self,
            weight: float,
            use_focal_loss: bool,
            spatial_loss: bool = False,
            l1_weight: float = 5.0,
            giou_weight: float = 2.0,
    ):
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        # MOTIP2 Spatial ID Loss (off by default; see ``spatial_loss``):
        self.use_spatial_loss = spatial_loss
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        assert not (self.use_spatial_loss and self.use_focal_loss), \
            "ID_LOSS_SPATIAL needs softmax ID probabilities, it cannot be combined with USE_FOCAL_LOSS."

        if not self.use_focal_loss:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        return

    def forward(self, id_logits, id_labels, id_masks):
        # _B, _G, _T, _N = id_logits.shape
        # Remove the first T for supervision:
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        pass

        # Flatten:
        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        # Filter out the invalid id labels:
        id_logits_flatten = id_logits_flatten[~id_masks_flatten]
        id_labels_flatten = id_labels_flatten[~id_masks_flatten]
        # Calculate the loss:
        if self.use_focal_loss:
            id_labels_flatten_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            id_labels_flatten_one_hot = torch.tensor(id_labels_flatten_one_hot, device=id_logits.device)
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_flatten_one_hot).sum()
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        return loss / num_ids

    def spatial_loss(self, id_logits, id_masks, slot_labels, pred_boxes, gt_boxes):
        """MOTIP2 Spatial ID Loss: the expected box distance of the chosen identity.

        For each detection, take its (detached) predicted box and the GT box of every identity it could be
        given: each identity present in the current frame, plus "newborn" (whose box is the detection's own GT
        box). The cost of an identity is ``L1 + (1 - GIoU)`` against that box, and the loss is
        ``sum_c p_c * cost_c`` with the ID probabilities ``p`` renormalised over that reduced set.
        Confusing a detection with a neighbour is therefore cheap, with a far-away identity expensive.

        Args:
            id_logits: (B*, G, T, N, K+1) as given to ``forward`` (decoder layers concatenated on dim 0).
            id_masks: (B*, G, T, N) True = no object in that slot.
            slot_labels: (B, G, T, N) the ID word of each slot (``trajectory_id_labels``; -1 = padding).
            pred_boxes: (B, G, T, N, 4) cxcywh, the predicted box of the detection matched to each slot.
            gt_boxes: (B, G, T, N, 4) cxcywh, the GT box of each slot's object in this frame.
        Returns:
            ``(loss_l1, loss_giou)``, each normalised by the number of supervised tokens like the CE and
            before the ``l1_weight`` / ``giou_weight`` factors.
        """
        reps = id_logits.shape[0] // slot_labels.shape[0]

        def tile(x):    # layers are concatenated on dim 0, and the first frame is not supervised
            return x.repeat(reps, *([1] * (x.dim() - 1)))[:, :, 1:]

        slot_labels, pred_boxes, gt_boxes = tile(slot_labels), tile(pred_boxes.detach()), tile(gt_boxes)
        id_logits = id_logits[:, :, 1:].float()
        valid = ~id_masks[:, :, 1:]
        # The count is all-reduced on every rank, before anything can return early:
        num_ids = torch.as_tensor([valid.sum()], dtype=torch.float, device=id_logits.device)
        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        n = slot_labels.shape[-1]
        newborn = id_logits.shape[-1] - 1
        # Logit of identity j's word, as seen from detection i (B, G, T, N_i, N_j); absent identities are masked:
        idx = slot_labels.clamp(min=0)[..., None, :].expand(*slot_labels.shape, n)
        candidates = id_logits.gather(-1, idx).masked_fill(~valid[..., None, :], float("-inf"))
        probs = torch.cat([candidates, id_logits[..., newborn:]], dim=-1).softmax(dim=-1)      # (..., N_i, N_j + 1)

        l1 = (pred_boxes[..., :, None, :] - gt_boxes[..., None, :, :]).abs().sum(-1)            # (..., N_i, N_j)
        l1_newborn = (pred_boxes - gt_boxes).abs().sum(-1, keepdim=True)
        giou = pairwise_giou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))
        giou_newborn = torch.diagonal(giou, dim1=-2, dim2=-1)[..., None]
        cost_l1 = torch.cat([l1, l1_newborn], dim=-1)
        cost_giou = 1.0 - torch.cat([giou, giou_newborn], dim=-1)

        loss_l1 = (probs * cost_l1).sum(-1)[valid].sum() / num_ids
        loss_giou = (probs * cost_giou).sum(-1)[valid].sum() / num_ids
        return loss_l1, loss_giou


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config["USE_FOCAL_LOSS"],
        spatial_loss=config.get("ID_LOSS_SPATIAL", False),
        l1_weight=config.get("ID_LOSS_L1_WEIGHT", 5.0),
        giou_weight=config.get("ID_LOSS_GIOU_WEIGHT", 2.0),
    )