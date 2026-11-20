# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from torchvision.ops.boxes import box_area


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    boxes = [
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        (x2 - x1),
        (y2 - y1)
    ]
    return torch.stack(boxes, dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    boxes = [
        (cx - 0.5 * w),
        (cy - 0.5 * h),
        (cx + 0.5 * w),
        (cy + 0.5 * h)
    ]
    return torch.stack(boxes, dim=-1)


def box_cxcywh_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    boxes = [
        (cx - 0.5 * w),
        (cy - 0.5 * h),
        w,
        h
    ]
    return torch.stack(boxes, dim=-1)


def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    boxes = [
        x,
        y,
        x + w,
        y + h
    ]
    return torch.stack(boxes, dim=-1)


def box_iou_union(boxes1, boxes2):
    """
    Args:
        boxes1:
        boxes2:

    Returns: iou, union
    """
    area1 = box_area(boxes1)    # [N, ]
    area2 = box_area(boxes2)    # [M, ]
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter  # [N, M]
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), \
        f"illegal boxes1: {boxes1[~ ((boxes1[:, 2:] >= boxes1[:, :2])[:, 0] & (boxes1[:, 2:] >= boxes1[:, :2])[:, 1])]}"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou_union(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """GIoU between every box of ``boxes1`` (..., N, 4) and ``boxes2`` (..., M, 4) -> (..., N, M).

    Boxes are xyxy. Unlike ``generalized_box_iou`` this accepts leading batch dims and never asserts or
    divides by zero (``eps`` keeps zero-area padding boxes finite), so it is safe on padded boxes.
    """
    a, b = boxes1[..., :, None, :], boxes2[..., None, :, :]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=eps)
    enc = (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    return iou - (enc - union) / enc.clamp(min=eps)
