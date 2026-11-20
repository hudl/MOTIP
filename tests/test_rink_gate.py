"""Tests for the RuntimeTracker rink-space gate: an ID cannot jump further (in metres) than a skater can.

Same stub setup as test_spatial_gate.py. Boxes are cxcywh normalised on a 1000 x 1000 canvas. The test
"homography" maps image px to rink metres as x_m = 0.02 * px_x - 10, y_m = 0.02 * px_y - 10, so a box
at normalised x has feet at x_m = 20 * x - 10 (0.05 of x = 1 m) and y_m = 1 m for the default box.
"""
import numpy as np
import pytest
import torch

from models.runtime_tracker import RuntimeTracker
from utils.misc import distributed_device

K = 10
W = H = 1000
HM = np.array([[0.02, 0.0, -10.0], [0.0, 0.02, -10.0], [0.0, 0.0, 1.0]])
PAN = np.array([[0.02, 0.0, -16.0], [0.0, 0.02, -10.0], [0.0, 0.0, 1.0]])   # camera moved 300 px: x_m = 0.02 px - 16


class _StubModel:
    num_id_vocabulary = K
    num_classes = 1

    def __init__(self):
        self.logits = None

    def eval(self):
        return self

    def __call__(self, seq_info=None, part=None, **_):
        if part == "trajectory_modeling":
            return seq_info
        return self.logits[None, None, None], None, None


def _box(x, h=0.1):
    return [x, 0.5, h / 2, h]


def _m(x):
    """Rink x in metres of a default box at normalised x (feet y is 1 m)."""
    return 20.0 * x - 10.0


def _tracker(history, labels, track_pos=None, **kw):
    """``history[t][n]`` is track n's box (None = unseen); ``track_pos[label]`` its last (x_m, y_m, valid)."""
    dev = distributed_device()
    kw.setdefault("rink_gate", True)
    tr = RuntimeTracker(model=_StubModel(), sequence_hw=(H, W), id_thresh=0.1, newborn_thresh=0.5, **kw)
    t, n = len(history), len(labels)
    boxes = torch.zeros(t, n, 4)
    masks = torch.ones(t, n, dtype=torch.bool)
    for i, row in enumerate(history):
        for j, b in enumerate(row):
            if b is not None:
                boxes[i, j] = torch.tensor(b)
                masks[i, j] = False
    tr.trajectory_boxes = boxes.to(dev)
    tr.trajectory_masks = masks.to(dev)
    tr.trajectory_id_labels = torch.tensor([labels] * t, device=dev)
    tr.trajectory_features = torch.zeros(t, n, 256, device=dev)
    tr.trajectory_times = torch.arange(t, device=dev)[:, None].expand(t, n).contiguous()
    tr.id_label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    tr.next_id = len(labels)
    # By default every track was last seen where its last history box projects under HM.
    for j, lbl in enumerate(labels):
        last = [row[j] for row in history if row[j] is not None]
        if last:
            tr._track_rink[lbl] = (_m(last[-1][0]), 1.0, True)
    for lbl, pos in (track_pos or {}).items():
        tr._track_rink[lbl] = pos
    return tr


def _logits(rows):
    s = torch.full((len(rows), K + 1), 1e-4)
    for i, row in enumerate(rows):
        for col, p in row.items():
            s[i, col] = p
    return s.log().to(distributed_device())


def _labels(tr, rows, dets, rink_H=HM):
    tr.model.logits = _logits(rows)
    boxes = torch.tensor(dets, device=distributed_device())
    if tr.rink_gate:
        tr._set_frame_rink(boxes, rink_H)
    return tr._get_id_pred_labels(boxes=boxes, output_embeds=torch.zeros(len(dets), 256, device=boxes.device)).tolist()


ROW = [{7: 0.9, K: 0.05}]


def test_off_by_default_far_box_keeps_its_id():
    tr = _tracker([[_box(0.1)]], [7], rink_gate=False)
    assert tr.rink_gate is False and tr.spatial_gate is False
    assert _labels(tr, ROW, [_box(0.9)]) == [7]


def test_blocks_beyond_base_plus_speed_and_keeps_within():
    # fps 30, seen last frame: radius 2 + 10/30 = 2.33 m. Track at 0 m (x = 0.5).
    mk = lambda: _tracker([[_box(0.5)]], [7], rink_gate_base_m=2.0, rink_gate_speed_mps=10.0, rink_fps=30.0)
    assert _labels(mk(), ROW, [_box(0.6)]) == [7]        # 2 m
    assert _labels(mk(), ROW, [_box(0.7)]) == [K]        # 4 m


def test_radius_grows_with_seconds_missed():
    # Track last seen 29 frames ago: dt = 30 / 30 s -> 2 + 10 = 12 m. A detection 8 m away (x = 0.9).
    recent = _tracker([[_box(0.5)]], [7])
    old = _tracker([[_box(0.5)]] + [[None]] * 29, [7])
    assert _labels(recent, ROW, [_box(0.9)]) == [K]
    assert _labels(old, ROW, [_box(0.9)]) == [7]


def test_fps_sets_the_speed_term():
    # base 0, 10 m/s, seen last frame: 1/30 s -> 0.333 m but 1/60 s -> 0.167 m. A 0.25 m offset.
    at30 = _tracker([[_box(0.5)]], [7], rink_gate_base_m=0.0, rink_fps=30.0)
    at60 = _tracker([[_box(0.5)]], [7], rink_gate_base_m=0.0, rink_fps=60.0)
    assert _labels(at30, ROW, [_box(0.5125)]) == [7]
    assert _labels(at60, ROW, [_box(0.5125)]) == [K]


def test_camera_pan_does_not_break_a_match_in_rink_space():
    # Same skater, the camera panned 300 px between frames: the box jumps 0.3 in image x but is at
    # the same place on the ice. The image gate (base 0.5 heights) refuses it, the rink gate keeps it.
    track_box, det_box = _box(0.3), _box(0.6)      # x_m = -4 under HM; 0.6 under PAN is x_m = -4
    img = _tracker([[track_box]], [7], rink_gate=False, spatial_gate=True)
    assert _labels(img, ROW, [det_box]) == [K]
    rink = _tracker([[track_box]], [7], track_pos={7: (-4.0, 1.0, True)})
    assert _labels(rink, ROW, [det_box], rink_H=PAN) == [7]


def test_frame_without_a_transform_leaves_pairs_ungated():
    tr = _tracker([[_box(0.1)]], [7])
    assert _labels(tr, ROW, [_box(0.9)], rink_H=None) == [7]
    assert _labels(_tracker([[_box(0.1)]], [7]), ROW, [_box(0.9)], rink_H=np.full((3, 3), np.nan)) == [7]


def test_track_last_seen_without_a_position_is_ungated():
    tr = _tracker([[_box(0.1)]], [7], track_pos={7: (np.nan, np.nan, False)})
    assert _labels(tr, ROW, [_box(0.9)]) == [7]


def test_detection_off_the_rink_is_invalid_and_ungated():
    # Feet at x_m = 20 * 1.0 - 10 = 10 m is on a 31 m rink; shrink the bounds to put it off.
    tr = _tracker([[_box(0.1)]], [7], rink_bounds_m=(5.0, 16.0))
    assert _labels(tr, ROW, [_box(0.9)]) == [7]                    # +8 m: off the 5 m half length
    tr = _tracker([[_box(0.1)]], [7], rink_bounds_m=(31.0, 16.0))
    assert _labels(tr, ROW, [_box(0.9)]) == [K]


def test_invalid_pairs_fall_back_to_the_image_gate_valid_pairs_use_the_rink():
    # Image gate on too (base 0.5 heights = 0.05 in x). Valid rink pairs use the rink decision:
    # det 0.4 is 6 m from track 0.1 -> blocked; det 0.12 is 0.4 m -> allowed.
    mk = lambda: _tracker([[_box(0.1)]], [7], spatial_gate=True, gate_base=0.5, gate_per_step=0.0)
    assert _labels(mk(), ROW, [_box(0.12)]) == [7]
    assert _labels(mk(), ROW, [_box(0.4)]) == [K]
    # No transform this frame -> image gate only: 0.4 is blocked, 0.12 is not.
    assert _labels(mk(), ROW, [_box(0.12)], rink_H=None) == [7]
    assert _labels(mk(), ROW, [_box(0.4)], rink_H=None) == [K]
    # Rink says near but the image gate would refuse (a pan-sized offset): the rink decision wins.
    track_pos = {7: (-4.0, 1.0, True)}
    both = _tracker([[_box(0.3)]], [7], track_pos=track_pos, spatial_gate=True, gate_base=0.5, gate_per_step=0.0)
    assert _labels(both, ROW, [_box(0.6)], rink_H=PAN) == [7]


def test_zeroes_scores_and_leaves_the_rest_alone():
    tr = _tracker([[_box(0.1), _box(0.5)]], [7, 3])
    scores = torch.rand(2, K + 1, device=distributed_device()).softmax(-1)
    boxes = torch.tensor([_box(0.12), _box(0.9)], device=distributed_device())
    tr._set_frame_rink(boxes, HM)
    out = tr._apply_spatial_gate(scores, boxes)
    expected = scores.clone()
    expected[0, 3] = 0        # det 0 (-7.6 m) is 7.6 m from track 3 (0 m)
    expected[1, 7] = 0
    expected[1, 3] = 0
    assert torch.equal(out, expected)
    assert not torch.equal(out, scores)


def test_negative_or_bad_parameters_are_rejected():
    with pytest.raises(ValueError, match="rink_gate_base_m"):
        RuntimeTracker(model=_StubModel(), sequence_hw=(H, W), rink_gate=True, rink_gate_base_m=-1.0)
    with pytest.raises(ValueError, match="rink_fps"):
        RuntimeTracker(model=_StubModel(), sequence_hw=(H, W), rink_gate=True, rink_fps=0.0)


def _process(tr, rows, dets, score, rink_H):
    tr.model.logits = _logits(rows)
    dev = distributed_device()
    n = len(dets)
    tr._process_detections(
        scores=torch.full((n,), score, device=dev), categories=torch.zeros(n, dtype=torch.int64, device=dev),
        boxes=torch.tensor(dets, device=dev), output_embeds=torch.zeros(n, 256, device=dev), rink_H=rink_H,
    )
    return tr.get_track_results()


def test_cut_every_pair_gated_keeps_all_detections_as_newborns():
    tr = _tracker([[_box(0.1), _box(0.15)]], [7, 3])
    res = _process(tr, [{7: 0.9, K: 0.05}, {3: 0.9, K: 0.05}], [_box(0.8), _box(0.9)], score=0.4, rink_H=HM)
    assert tr.gate_stats["cut_frames"] == 1
    assert len(res["id"]) == 2 and not set(res["id"].tolist()) & {0, 1}


def test_partial_gating_is_not_a_cut():
    tr = _tracker([[_box(0.1), _box(0.15)]], [7, 3])
    res = _process(tr, [{7: 0.9, K: 0.05}, {3: 0.9, K: 0.05}], [_box(0.12), _box(0.9)], score=0.4, rink_H=HM)
    assert tr.gate_stats["cut_frames"] == 0
    assert res["id"].tolist() == [0]


def test_positions_are_remembered_per_track_across_frames_end_to_end():
    # Frame 1: one newborn at x_m = -4 (HM). Frame 2 after a pan: the same player (x = 0.6 under PAN)
    # keeps its ID. Frame 3: a detection 10 m away from where it now is -> a newborn, not the old ID.
    tr = RuntimeTracker(model=_StubModel(), sequence_hw=(H, W), rink_gate=True, id_thresh=0.1, newborn_thresh=0.5)
    r1 = _process(tr, [{K: 0.95}], [_box(0.3)], score=0.9, rink_H=HM)
    first = r1["id"].tolist()
    assert len(first) == 1
    lab = int(tr.trajectory_id_labels[0, 0])
    assert tr._track_rink[lab][:2] == pytest.approx((-4.0, 1.0)) and tr._track_rink[lab][2]
    r2 = _process(tr, [{lab: 0.9, K: 0.05}], [_box(0.6)], score=0.9, rink_H=PAN)
    assert r2["id"].tolist() == first
    r3 = _process(tr, [{lab: 0.9, K: 0.05}], [_box(0.6 + 0.5)], score=0.9, rink_H=PAN)    # +10 m
    assert r3["id"].tolist() != first
    assert tr.rink_stats["frames_with_h"] == 3 and tr.rink_stats["gated_rink"] >= 1


def test_goalkeeper_suppression_keeps_rows_aligned():
    # Suppress the first of three detections: positions must follow the surviving boxes.
    tr = RuntimeTracker(model=_StubModel(), sequence_hw=(H, W), rink_gate=True, id_thresh=0.1, newborn_thresh=0.5)
    dev = distributed_device()
    dets = [_box(0.1), _box(0.5), _box(0.8)]
    tr.model.logits = _logits([{K: 0.95}, {K: 0.95}])
    sup = [[70.0, 440.0, 130.0, 560.0]]    # xyxy px: IoU 0.69 with the first box (cx 100)
    tr._process_detections(
        scores=torch.full((3,), 0.9, device=dev), categories=torch.zeros(3, dtype=torch.int64, device=dev),
        boxes=torch.tensor(dets, device=dev), output_embeds=torch.zeros(3, 256, device=dev),
        suppress_boxes=sup, rink_H=HM,
    )
    assert len(tr.get_track_results()["id"]) == 2
    got = sorted(round(v[0], 3) for v in tr._track_rink.values())
    assert got == [0.0, 6.0]
