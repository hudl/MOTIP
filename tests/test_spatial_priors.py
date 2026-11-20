"""Tests for the MOTIP2 spatial priors: spatial ID switches, spatial ID loss, spatial anchor.

All three are off by default; the first tests in each section pin that the defaults reproduce the old behaviour.
Boxes are normalised cxcywh unless noted.
"""
import copy

import pytest
import torch
import torch.nn as nn

from data.transforms import TurnIntoTrajectoryAndUnknown, build_transforms
from models.motip.id_criterion import IDCriterion
from models.motip.motip import MOTIP
from utils.box_ops import box_cxcywh_to_xyxy, pairwise_giou
from utils.switch_stats import add_switch_stats, clip_switch_stats, empty_switch_stats, summarize_switch_stats

K = 8       # ID vocabulary; column K of the logits is the newborn token


# --------------------------------------------------------------------------------------------------
# pairwise_giou
# --------------------------------------------------------------------------------------------------
def test_pairwise_giou_values():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0], [4.0, 0.0, 6.0, 2.0]])
    g = pairwise_giou(a, b)[0]
    assert g[0] == pytest.approx(1.0)
    # inter 2, union 6 -> IoU 1/3; enclosing 3 x 2 = 6 = union -> GIoU = IoU
    assert g[1] == pytest.approx(1 / 3)
    # disjoint: IoU 0, union 8, enclosing 6 x 2 = 12 -> -(12 - 8) / 12
    assert g[2] == pytest.approx(-1 / 3)


def test_pairwise_giou_batched_and_degenerate():
    boxes = torch.rand(2, 3, 5, 4)
    boxes[..., 2:] += boxes[..., :2]
    assert pairwise_giou(boxes, boxes).shape == (2, 3, 5, 5)
    assert torch.allclose(torch.diagonal(pairwise_giou(boxes, boxes), dim1=-2, dim2=-1), torch.ones(2, 3, 5), atol=1e-5)
    zeros = torch.zeros(1, 4, 4)          # padded slots are all-zero boxes
    assert torch.isfinite(pairwise_giou(zeros, zeros)).all()


# --------------------------------------------------------------------------------------------------
# Spatial ID switches
# --------------------------------------------------------------------------------------------------
def _clip(centres, T=4, G=1, size=0.1):
    """A clip of T frames of fixed boxes at ``centres`` (list of (cx, cy)); one slot per object, all visible.

    Returns annotations in the form ``TurnIntoTrajectoryAndUnknown`` receives them.
    """
    N = len(centres)
    bbox = torch.tensor([[cx, cy, size, size] for cx, cy in centres])
    annotations = []
    for t in range(T):
        annotations.append({
            "bbox": bbox.clone(),
            "id_labels": torch.arange(N)[None, None, :].repeat(G, 1, 1),
            "id_masks": torch.zeros((G, 1, N), dtype=torch.bool),
            "ann_idxs": torch.arange(N)[None, None, :].repeat(G, 1, 1),
            "times": torch.full((G, 1, N), t, dtype=torch.int64),
        })
    return annotations


def _apply(transform, annotations):
    _, out, _ = transform([None] * len(annotations), copy.deepcopy(annotations), None)
    return out


def _stack(out, key):
    return torch.cat([a[key] for a in out], dim=1)


def test_switch_defaults_unchanged():
    clip = _clip([(0.2, 0.5), (0.25, 0.5), (0.7, 0.5), (0.75, 0.5)])
    for kw in (dict(aug_trajectory_switch_prob=0.0), dict(aug_trajectory_switch_prob=0.5)):
        torch.manual_seed(3)
        old = _apply(TurnIntoTrajectoryAndUnknown(K, 0.0, **kw), clip)
        torch.manual_seed(3)
        new = _apply(TurnIntoTrajectoryAndUnknown(K, 0.0, **kw, aug_trajectory_switch_mode="uniform"), clip)
        for key in old[0]:
            assert all(torch.equal(o[key], n[key]) for o, n in zip(old, new)), key


def test_switch_mode_validated_and_spatial_prob_zero_is_noop():
    with pytest.raises(ValueError):
        TurnIntoTrajectoryAndUnknown(K, 0.0, 0.5, aug_trajectory_switch_mode="nearest")
    clip = _clip([(0.2, 0.5), (0.25, 0.5)])
    out = _apply(TurnIntoTrajectoryAndUnknown(K, 0.0, 0.0, aug_trajectory_switch_mode="spatial"), clip)
    assert torch.equal(_stack(out, "trajectory_ann_idxs"), _stack(out, "unknown_ann_idxs"))


def test_build_transforms_reads_new_keys():
    cfg = dict(
        AUG_MAX_SHIFT_RATIO=0.0, AUG_OVERFLOW_BBOX=False, AUG_RESIZE_SCALES=[64], AUG_MAX_SIZE=128,
        AUG_RANDOM_RESIZE=[64], AUG_RANDOM_CROP_MIN=32, AUG_RANDOM_CROP_MAX=64,
        AUG_BRIGHTNESS=0.0, AUG_CONTRAST=0.0, AUG_SATURATION=0.0, AUG_HUE=0.0, AUG_COLOR_JITTER_V2=False,
        NUM_ID_VOCABULARY=K, AUG_NUM_GROUPS=1, AUG_TRAJECTORY_OCCLUSION_PROB=0.0, AUG_TRAJECTORY_SWITCH_PROB=0.3,
    )
    last = build_transforms(cfg).transforms[-1]
    assert (last.aug_trajectory_switch_mode, last.aug_trajectory_switch_sigma) == ("uniform", 0.5)
    last = build_transforms({**cfg, "AUG_TRAJECTORY_SWITCH_MODE": "spatial", "AUG_TRAJECTORY_SWITCH_SIGMA": 0.2}).transforms[-1]
    assert (last.aug_trajectory_switch_mode, last.aug_trajectory_switch_sigma) == ("spatial", 0.2)


def test_spatial_switch_only_swaps_overlapping_boxes():
    # Two tight pairs far apart: (0, 1) on the left, (2, 3) on the right.
    clip = _clip([(0.20, 0.5), (0.23, 0.5), (0.80, 0.5), (0.83, 0.5)], T=40, G=2)
    torch.manual_seed(0)
    out = _apply(
        TurnIntoTrajectoryAndUnknown(K, 0.0, 1.0, aug_trajectory_switch_mode="spatial", aug_trajectory_switch_sigma=0.3),
        clip,
    )
    traj, unk = _stack(out, "trajectory_ann_idxs"), _stack(out, "unknown_ann_idxs")
    assert torch.equal(unk, torch.arange(4)[None, None].expand_as(unk))            # the unknown side is never switched
    swapped = traj != unk
    assert swapped.any(), "with sigma 0.3 some of the overlapping pairs should swap"
    # The permutation stays within a frame (a permutation of the same objects) ...
    assert torch.equal(traj.sort(-1).values, unk.sort(-1).values)
    # ... and an object only ever trades with its overlapping neighbour, never with the far pair:
    left, right = traj[..., :2], traj[..., 2:]
    assert (left < 2).all() and (right >= 2).all()
    # Stats agree: every swap has positive GIoU.
    stats = clip_switch_stats(out)
    assert stats["swapped"] > 0 and stats["far"] == 0 and stats["giou_sum"] > 0


def test_uniform_switch_does_swap_far_boxes():
    clip = _clip([(0.20, 0.5), (0.23, 0.5), (0.80, 0.5), (0.83, 0.5)], T=40, G=2)
    torch.manual_seed(0)
    out = _apply(TurnIntoTrajectoryAndUnknown(K, 0.0, 1.0), clip)
    assert clip_switch_stats(out)["far"] > 0


def test_spatial_switch_respects_occlusion_and_selection():
    # Slot 1 is invisible in every frame: it must neither give nor receive a detection.
    clip = _clip([(0.20, 0.5), (0.22, 0.5), (0.24, 0.5)], T=30)
    for a in clip:
        a["id_masks"][..., 1] = True
        a["ann_idxs"][..., 1] = -1
    torch.manual_seed(1)
    out = _apply(
        TurnIntoTrajectoryAndUnknown(K, 0.0, 1.0, aug_trajectory_switch_mode="spatial", aug_trajectory_switch_sigma=1.0),
        clip,
    )
    traj = _stack(out, "trajectory_ann_idxs")
    assert (traj[..., 1] == -1).all()
    assert set(traj[..., [0, 2]].unique().tolist()) <= {0, 2}
    # prob 0 < p < 1: a lone selected object has nobody to swap with, never crashes.
    torch.manual_seed(2)
    _apply(TurnIntoTrajectoryAndUnknown(K, 0.0, 0.2, aug_trajectory_switch_mode="spatial"), _clip([(0.2, 0.5), (0.4, 0.5)]))


def test_switch_stats_summary():
    s = empty_switch_stats()
    add_switch_stats(s, {"visible": 10.0, "swapped": 2.0, "far": 1.0, "giou_sum": 0.5})
    r = summarize_switch_stats(s)
    assert (r["swap_rate"], r["far_frac"], r["mean_giou"]) == (0.2, 0.5, 0.25)


# --------------------------------------------------------------------------------------------------
# Spatial ID loss
# --------------------------------------------------------------------------------------------------
def _logits_from_probs(probs, fill=5.0):
    """(K+1,) logits whose softmax over the columns in ``probs`` ({col: p}, summing to 1) is exactly ``probs``;
    all other columns are filled with an arbitrary large value the loss must ignore."""
    out = torch.full((K + 1,), fill)
    for col, p in probs.items():
        out[col] = torch.tensor(p).log()
    return out


def _tiny_batch(extra_masked_slot=False):
    """B=1, G=1, T=2, N=2 (N=3 with an invisible slot). Frame 0 is never supervised. Slot 0 carries ID word 3
    and slot 1 ID word 5 (newborn in this frame, which does not matter to the loss: it is a candidate anyway).
    """
    N = 3 if extra_masked_slot else 2
    logits = torch.zeros(1, 1, 2, N, K + 1)
    logits[0, 0, 1, 0] = _logits_from_probs({3: 0.5, 5: 0.3, K: 0.2})
    logits[0, 0, 1, 1] = _logits_from_probs({3: 0.1, 5: 0.6, K: 0.3})
    masks = torch.zeros(1, 1, 2, N, dtype=torch.bool)
    labels = torch.tensor([3, 5, 7][:N]).expand(1, 1, 2, N).clone()
    pred = torch.zeros(1, 1, 2, N, 4)
    gt = torch.zeros(1, 1, 2, N, 4)
    pred[0, 0, 1, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    pred[0, 0, 1, 1] = torch.tensor([0.6, 0.5, 0.2, 0.2])
    gt[0, 0, 1, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    gt[0, 0, 1, 1] = torch.tensor([0.6, 0.5, 0.2, 0.2])
    if extra_masked_slot:
        masks[0, 0, :, 2] = True
        logits[0, 0, 1, :, 7] = 50.0        # the absent identity would dominate if it were not masked out
        pred[0, 0, 1, 2] = gt[0, 0, 1, 2] = torch.tensor([0.9, 0.9, 0.1, 0.1])
    return logits, masks, labels, pred, gt


# Hand computation (frame 1, boxes 0.2 x 0.2; box A at cx 0.5, box B at cx 0.6):
#   cost(A vs B): L1 = 0.1, IoU = 0.02 / 0.06 = 1/3, enclosing area = union -> GIoU = 1/3, so 1 - GIoU = 2/3
#   cost(same box) = 0 and the newborn cost uses the detection's own GT box = 0.
#   det 0 (pred = A, probs id3 .5, id5 .3, newborn .2): L1 = .3 * .1 = .03; GIoU part = .3 * 2/3 = .2
#   det 1 (pred = B, probs id3 .1, id5 .6, newborn .3): L1 = .1 * .1 = .01; GIoU part = .1 * 2/3 = 1/15
#   both are normalised by the 2 supervised tokens.
EXPECT_L1 = (0.03 + 0.01) / 2
EXPECT_GIOU = (0.2 + 1 / 15) / 2


def test_spatial_loss_hand_computed():
    logits, masks, labels, pred, gt = _tiny_batch()
    crit = IDCriterion(weight=1.0, use_focal_loss=False, spatial_loss=True)
    l1, giou = crit.spatial_loss(logits, masks, labels, pred, gt)
    assert l1.item() == pytest.approx(EXPECT_L1, abs=1e-6)
    assert giou.item() == pytest.approx(EXPECT_GIOU, abs=1e-6)


def test_spatial_loss_newborn_uses_own_box():
    # A detection whose predicted box is far from its GT but whose probability mass is all on newborn pays the
    # distance to its own GT box (not to anyone else's).
    logits = torch.zeros(1, 1, 2, 1, K + 1)
    logits[0, 0, 1, 0] = _logits_from_probs({3: 1e-6, K: 1 - 1e-6})
    masks = torch.zeros(1, 1, 2, 1, dtype=torch.bool)
    labels = torch.tensor([3]).expand(1, 1, 2, 1).clone()
    pred = torch.zeros(1, 1, 2, 1, 4)
    gt = torch.zeros(1, 1, 2, 1, 4)
    pred[0, 0, 1, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    gt[0, 0, 1, 0] = torch.tensor([0.6, 0.5, 0.2, 0.2])
    crit = IDCriterion(weight=1.0, use_focal_loss=False, spatial_loss=True)
    l1, giou = crit.spatial_loss(logits, masks, labels, pred, gt)
    # the id-3 candidate is the detection's own slot, so every candidate has the same box: L1 .1, 1 - GIoU 2/3
    assert l1.item() == pytest.approx(0.1, abs=1e-4)
    assert giou.item() == pytest.approx(2 / 3, abs=1e-4)


def test_spatial_loss_masks_absent_identities_and_ignores_first_frame():
    logits, masks, labels, pred, gt = _tiny_batch(extra_masked_slot=True)
    crit = IDCriterion(weight=1.0, use_focal_loss=False, spatial_loss=True)
    l1, giou = crit.spatial_loss(logits, masks, labels, pred, gt)
    assert l1.item() == pytest.approx(EXPECT_L1, abs=1e-6)
    assert giou.item() == pytest.approx(EXPECT_GIOU, abs=1e-6)
    # the first frame is not supervised: garbage there changes nothing
    pred2, logits2 = pred.clone(), logits.clone()
    pred2[:, :, 0] = 0.77
    logits2[:, :, 0] = 3.0
    l1b, giou_b = crit.spatial_loss(logits2, masks, labels, pred2, gt)
    assert (l1b.item(), giou_b.item()) == pytest.approx((l1.item(), giou.item()))


def test_spatial_loss_aux_layers_and_gradients():
    logits, masks, labels, pred, gt = _tiny_batch()
    crit = IDCriterion(weight=1.0, use_focal_loss=False, spatial_loss=True)
    ref = crit.spatial_loss(logits, masks, labels, pred, gt)
    # decoder layers are concatenated on dim 0 (USE_AUX_LOSS); identical layers give the same normalised loss
    l1, giou = crit.spatial_loss(logits.repeat(3, 1, 1, 1, 1), masks.repeat(3, 1, 1, 1), labels, pred, gt)
    assert (l1.item(), giou.item()) == pytest.approx((ref[0].item(), ref[1].item()))
    # gradients reach the ID logits, never the (detached) predicted boxes
    logits = logits.clone().requires_grad_(True)
    pred = pred.clone().requires_grad_(True)
    l1, giou = crit.spatial_loss(logits, masks, labels, pred, gt)
    (l1 + giou).backward()
    assert pred.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_spatial_loss_defaults_and_ce_unchanged():
    crit = IDCriterion(weight=1.0, use_focal_loss=False)
    assert crit.use_spatial_loss is False and (crit.l1_weight, crit.giou_weight) == (5.0, 2.0)
    logits, masks, _, _, _ = _tiny_batch()
    labels = torch.tensor([3, 5]).expand(1, 1, 2, 2).clone()
    on = IDCriterion(weight=1.0, use_focal_loss=False, spatial_loss=True)
    assert crit(logits, labels, masks).item() == on(logits, labels, masks).item()
    with pytest.raises(AssertionError):
        IDCriterion(weight=1.0, use_focal_loss=True, spatial_loss=True)


def test_prepare_for_motip_gt_boxes():
    from train import prepare_for_motip
    T, N, Q = 2, 2, 5
    pred_boxes = torch.rand(T, Q, 4)
    outputs = {"pred_logits": torch.zeros(T, Q, 2), "pred_boxes": pred_boxes, "outputs": torch.rand(T, Q, 6)}
    gt = [torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]), torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.2, 0.2]])]
    indices = [(torch.tensor([4, 1]), torch.tensor([0, 1])), (torch.tensor([2, 3]), torch.tensor([1, 0]))]
    annotations = [[]]
    for t in range(T):
        annotations[0].append({
            "bbox": gt[t],
            "trajectory_id_labels": torch.tensor([[[3, 5]]]), "unknown_id_labels": torch.tensor([[[3, 5]]]),
            "trajectory_times": torch.full((1, 1, N), t), "unknown_times": torch.full((1, 1, N), t),
            "trajectory_id_masks": torch.zeros((1, 1, N), dtype=torch.bool),
            "unknown_id_masks": torch.zeros((1, 1, N), dtype=torch.bool),
            # frame 1 of the trajectory is switched: slot 0 takes detection 1's features, slot 1 detection 0's
            "trajectory_ann_idxs": torch.tensor([[[0, 1]]]) if t == 0 else torch.tensor([[[1, 0]]]),
            "unknown_ann_idxs": torch.tensor([[[0, 1]]]),
        })
    off = prepare_for_motip(outputs, annotations, indices)
    assert "unknown_gt_boxes" not in off
    on = prepare_for_motip(outputs, annotations, indices, with_gt_boxes=True)
    for k in off:
        assert torch.equal(off[k], on[k])
    # GT boxes follow the *unswitched* annotation of each slot ...
    assert torch.equal(on["unknown_gt_boxes"][0, 0], torch.stack(gt))
    # ... and the predicted boxes are those of the matched detection: GT j <-> query indices[t][0][argsort(indices[t][1])][j]
    assert torch.equal(on["unknown_boxes"][0, 0, 1, 0], pred_boxes[1, 3])      # GT 0 of frame 1 matched query 3
    assert torch.equal(on["unknown_boxes"][0, 0, 1, 1], pred_boxes[1, 2])
    # the switched trajectory tokens carry the other detection's box
    assert torch.equal(on["trajectory_boxes"][0, 0, 1, 0], pred_boxes[1, 2])


# --------------------------------------------------------------------------------------------------
# Spatial anchor
# --------------------------------------------------------------------------------------------------
D = 16          # tiny hidden dim for the stubs


class _Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ref_point_head = nn.Linear(2 * d_model, d_model)


class _StubDetr(nn.Module):
    def __init__(self, d_model=D):
        super().__init__()
        self.base = nn.Module()
        self.base.transformer = nn.Module()
        self.base.transformer.decoder = _Decoder(d_model)


class _StubIDDecoder(nn.Module):
    num_id_vocabulary = K
    feature_dim = D

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, seq_info, use_decoder_checkpoint=False):
        self.seen = seq_info
        return "logits"


def _motip(anchor):
    return MOTIP(
        detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
        id_decoder=_StubIDDecoder(), spatial_anchor=anchor, detr_hidden_dim=D,
    )


def _seq_info():
    g = torch.Generator().manual_seed(0)
    return {
        "trajectory_features": torch.randn(1, 1, 3, 4, D, generator=g),
        "unknown_features": torch.randn(1, 1, 3, 2, D, generator=g),
        "trajectory_boxes": torch.rand(1, 1, 3, 4, 4, generator=g),
        "unknown_boxes": torch.rand(1, 1, 3, 2, 4, generator=g),
        "trajectory_masks": torch.zeros(1, 1, 3, 4, dtype=torch.bool),
    }


def test_anchor_off_passes_seq_info_through():
    m = _motip(False)
    seq_info = _seq_info()
    assert m(part="id_decoder", seq_info=seq_info) == "logits"
    assert m.id_decoder.seen is seq_info


def test_anchor_on_adds_box_encoding_to_features_only():
    from rfdetr.models.transformer import gen_sineembed_for_position
    m = _motip(True)
    seq_info = _seq_info()
    before = copy.deepcopy(seq_info)
    m(part="id_decoder", seq_info=seq_info)
    seen = m.id_decoder.seen
    # the caller's dict is untouched
    assert all(torch.equal(seq_info[k], before[k]) for k in before)
    head = m.detr.base.transformer.decoder.ref_point_head
    for k in ("trajectory", "unknown"):
        boxes = seq_info[f"{k}_boxes"]
        feats = seq_info[f"{k}_features"]
        assert seen[f"{k}_features"].shape == feats.shape
        expect = feats + head(gen_sineembed_for_position(boxes.reshape(1, -1, 4), D // 2)).reshape(feats.shape)
        assert torch.allclose(seen[f"{k}_features"], expect, atol=1e-6)
        assert not torch.equal(seen[f"{k}_features"], feats)
        assert torch.equal(seen[f"{k}_boxes"], boxes)
    assert torch.equal(seen["trajectory_masks"], seq_info["trajectory_masks"])


def test_anchor_depends_on_box_not_batch_layout():
    m = _motip(True)
    boxes = torch.rand(2, 3, 4)
    full = m.box_position_encoding(boxes)
    assert full.shape == (2, 3, D)
    assert torch.allclose(full[1], m.box_position_encoding(boxes[1]), atol=1e-6)
    boxes2 = boxes.clone()
    boxes2[0, 0] += 0.1
    assert not torch.allclose(m.box_position_encoding(boxes2)[0, 0], full[0, 0])


def test_anchor_frozen_head_gets_no_gradient():
    m = _motip(True)
    for p in m.detr.parameters():
        p.requires_grad = False
    seq_info = _seq_info()
    for k in ("trajectory_features", "unknown_features"):
        seq_info[k].requires_grad_(True)
    m(part="id_decoder", seq_info=seq_info)
    m.id_decoder.seen["unknown_features"].sum().backward()
    assert seq_info["unknown_features"].grad is not None
    assert all(p.grad is None for p in m.detr.parameters())
    # boxes are detached: no gradient flows to them even when the head is trainable
    m2 = _motip(True)
    seq_info = _seq_info()
    seq_info["unknown_boxes"].requires_grad_(True)
    m2(part="id_decoder", seq_info=seq_info)
    m2.id_decoder.seen["unknown_features"].sum().backward()
    assert seq_info["unknown_boxes"].grad is None
    assert m2.detr.base.transformer.decoder.ref_point_head.weight.grad is not None


def test_anchor_requires_rfdetr_and_matching_dims():
    with pytest.raises(AssertionError):
        MOTIP(detr=_StubDetr(), detr_framework="deformable_detr", only_detr=False, trajectory_modeling=None,
              id_decoder=_StubIDDecoder(), spatial_anchor=True, detr_hidden_dim=D)
    with pytest.raises(AssertionError):
        MOTIP(detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
              id_decoder=_StubIDDecoder(), spatial_anchor=True, detr_hidden_dim=D * 2)


def test_runtime_tracker_feeds_the_boxes_the_anchor_uses():
    """Inference hands the ID decoder the predicted normalised cxcywh boxes of the history and of the current
    detections under the same keys the training seq_info uses, so the anchor sees the same boxes."""
    from models.runtime_tracker import RuntimeTracker
    from utils.misc import distributed_device

    class Rec:
        num_id_vocabulary = K
        num_classes = 1

        def eval(self):
            return self

        def __call__(self, seq_info=None, part=None, **_):
            if part == "id_decoder":
                self.seen = seq_info
                return torch.zeros(1, 1, 1, seq_info["unknown_boxes"].shape[-2], K + 1), None, None
            return seq_info

    dev = distributed_device()
    model = Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(100, 200))
    hist = torch.rand(2, 3, 4, device=dev)
    tr.trajectory_boxes = hist
    tr.trajectory_masks = torch.zeros(2, 3, dtype=torch.bool, device=dev)
    tr.trajectory_id_labels = torch.tensor([[0, 1, 2]] * 2, device=dev)
    tr.trajectory_features = torch.zeros(2, 3, 256, device=dev)
    tr.trajectory_times = torch.arange(2, device=dev)[:, None].expand(2, 3).contiguous()
    det = torch.rand(2, 4, device=dev)
    tr._get_id_pred_labels(boxes=det, output_embeds=torch.zeros(2, 256, device=dev))
    assert torch.equal(model.seen["trajectory_boxes"][0, 0], hist)
    assert torch.equal(model.seen["unknown_boxes"][0, 0, 0], det)


# --------------------------------------------------------------------------------------------------
# Spatial anchor scale (SPATIAL_ANCHOR_SCALE_INIT)
# --------------------------------------------------------------------------------------------------
def _motip_scaled(init):
    return MOTIP(
        detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
        id_decoder=_StubIDDecoder(), spatial_anchor=True, detr_hidden_dim=D,
        spatial_anchor_scale_init=init,
    )


def test_anchor_scale_absent_by_default():
    m = _motip(True)
    assert m.anchor_scale is None
    assert not any(n.endswith("anchor_scale") for n, _ in m.named_parameters())


def test_anchor_scale_zero_init_is_identity():
    m = _motip_scaled(0.0)
    seq_info = _seq_info()
    out = m.add_box_anchor(seq_info)
    for k in ("trajectory", "unknown"):
        assert torch.equal(out[f"{k}_features"], seq_info[f"{k}_features"])


def test_anchor_scale_one_matches_unscaled():
    scaled = _motip_scaled(1.0)
    plain = _motip(True)
    plain.detr = scaled.detr  # same head weights
    seq_info = _seq_info()
    got, ref = scaled.add_box_anchor(seq_info), plain.add_box_anchor(seq_info)
    for k in ("trajectory", "unknown"):
        assert torch.allclose(got[f"{k}_features"], ref[f"{k}_features"], atol=1e-6)


def test_anchor_scale_gets_gradient_at_zero_init():
    m = _motip_scaled(0.0)
    out = m.add_box_anchor(_seq_info())
    (out["unknown_features"].sum() + out["trajectory_features"].sum()).backward()
    assert m.anchor_scale.grad is not None and m.anchor_scale.grad.abs().sum() > 0


def test_load_checkpoint_fills_missing_anchor_scale(tmp_path, monkeypatch):
    import models.misc as misc
    monkeypatch.setattr(misc, "resize_rel_pos_embeds", lambda model, state: None)
    src = _motip(True)  # a checkpoint trained without the scale
    path = tmp_path / "ckpt.pth"
    torch.save({"model": src.state_dict()}, path)
    dst = _motip_scaled(0.0)
    misc.load_checkpoint(dst, str(path))
    assert torch.equal(dst.anchor_scale.detach(), torch.zeros(D))
    for n, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[n], v)
