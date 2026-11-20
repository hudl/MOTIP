"""Statistics of the trajectory ID-switch augmentation (the paper's Table 8: swap %, long-range %, mean GIoU).

Mode-agnostic: it compares the (switched) ``trajectory_ann_idxs`` with the untouched ``unknown_ann_idxs`` of a
transformed clip, so it measures the uniform and the spatial switch the same way.

Definitions, over one clip (T frames, G groups, N slots):
  visible  : slot holds an object in the unknown set (``unknown_id_masks`` False and an annotation index)
  swapped  : visible slot whose trajectory token comes from a different, also visible, object
  far      : swapped slot whose two boxes do not overlap (GIoU <= 0)
  giou     : GIoU between the slot's own box and the box it was swapped to
"""
import torch

from utils.box_ops import box_cxcywh_to_xyxy, pairwise_giou

STAT_KEYS = ("visible", "swapped", "far", "giou_sum")


def empty_switch_stats() -> dict:
    return {k: 0.0 for k in STAT_KEYS}


def clip_switch_stats(annotations: list) -> dict:
    """Stats for one transformed clip (the list of per-frame annotation dicts)."""
    out = empty_switch_stats()
    for ann in annotations:
        boxes = box_cxcywh_to_xyxy(ann["bbox"].float())
        unk = ann["unknown_ann_idxs"][:, 0]            # (G, N)
        traj = ann["trajectory_ann_idxs"][:, 0]
        unk_ok = (~ann["unknown_id_masks"][:, 0]) & (unk >= 0)
        traj_ok = (~ann["trajectory_id_masks"][:, 0]) & (traj >= 0)
        swapped = unk_ok & traj_ok & (traj != unk)
        out["visible"] += float(unk_ok.sum())
        out["swapped"] += float(swapped.sum())
        if swapped.any():
            a = boxes[unk[swapped]]
            b = boxes[traj[swapped]]
            giou = pairwise_giou(a[:, None], b[:, None])[:, 0, 0]
            out["far"] += float((giou <= 0).sum())
            out["giou_sum"] += float(giou.sum())
    return out


def add_switch_stats(total: dict, part: dict) -> dict:
    for k in STAT_KEYS:
        total[k] += part[k]
    return total


def summarize_switch_stats(s: dict) -> dict:
    sw = max(s["swapped"], 1.0)
    return {
        "swap_rate": s["swapped"] / max(s["visible"], 1.0),
        "far_frac": s["far"] / sw,
        "mean_giou": s["giou_sum"] / sw,
        "visible": s["visible"],
        "swapped": s["swapped"],
    }
