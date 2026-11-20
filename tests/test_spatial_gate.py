"""Tests for the RuntimeTracker spatial gate: an ID cannot jump further than a player can move.

The tracker is built with its real constructor around a stub model whose ID decoder returns fixed
logits, and its trajectory history is set by hand. Boxes are cxcywh normalised on a 1000 x 1000
canvas, so a box 0.1 tall is 100 px and one box height is 0.1 in x.
"""
import pytest
import torch

from models.runtime_tracker import RuntimeTracker
from utils.misc import distributed_device

K = 10          # ID vocabulary; index K is the newborn token
W = H = 1000


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
        return self.logits[None, None, None], None, None    # (1, 1, 1, D, K + 1)


def _box(x, h=0.1):
    return [x, 0.5, h / 2, h]


def _tracker(history, labels, *, id_thresh=0.1, newborn_thresh=0.5, **kw):
    """``history[t][n]`` is track n's box in history row t, or None when unseen that frame."""
    dev = distributed_device()
    tr = RuntimeTracker(
        model=_StubModel(), sequence_hw=(H, W), id_thresh=id_thresh, newborn_thresh=newborn_thresh, **kw,
    )
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
    return tr


def _logits(rows):
    """Logits whose softmax is ~ the given {id: prob} rows (unlisted ids get a tiny share)."""
    s = torch.full((len(rows), K + 1), 1e-4)
    for i, row in enumerate(rows):
        for col, p in row.items():
            s[i, col] = p
    return s.log().to(distributed_device())


def _labels(tr, rows, dets):
    tr.model.logits = _logits(rows)
    boxes = torch.tensor(dets, device=distributed_device())
    return tr._get_id_pred_labels(boxes=boxes, output_embeds=torch.zeros(len(dets), 256, device=boxes.device)).tolist()


def test_disabled_by_default_far_box_keeps_its_id():
    tr = _tracker([[_box(0.1)]], [7])
    assert tr.spatial_gate is False
    assert _labels(tr, [{7: 0.9, K: 0.05}], [_box(0.9)]) == [7]


def test_gate_blocks_far_box_and_keeps_near_box():
    tr = _tracker([[_box(0.1), _box(0.5)]], [7, 3], spatial_gate=True, gate_base=1.0, gate_per_step=0.0)
    # det 0 is 0.5 heights from track 7; det 1 is 3 heights from track 3 (> 1.0 allowed).
    out = _labels(tr, [{7: 0.9, K: 0.05}, {3: 0.9, K: 0.05}], [_box(0.15), _box(0.8)])
    assert out == [7, K]


def test_radius_grows_with_frames_missed():
    # Track seen once at x=0.1; a detection 2 heights away at x=0.3.
    # base 0.5 + 0.2 * missed: blocked with 0 frames missed (0.5), allowed after 9 (2.3).
    rows, det = [{7: 0.9, K: 0.05}], [_box(0.3)]
    recent = _tracker([[_box(0.1)]], [7], spatial_gate=True)
    old = _tracker([[_box(0.1)]] + [[None]] * 9, [7], spatial_gate=True)
    assert _labels(recent, rows, det) == [K]
    assert _labels(old, rows, det) == [7]


def test_radius_boundary_uses_frames_missed_not_rows():
    # Seen last frame: radius is exactly gate_base (0.5 heights = 0.05 in x).
    ok = _tracker([[_box(0.1)]], [7], spatial_gate=True, gate_base=0.5, gate_per_step=0.2)
    far = _tracker([[_box(0.1)]], [7], spatial_gate=True, gate_base=0.5, gate_per_step=0.2)
    assert _labels(ok, [{7: 0.9, K: 0.05}], [_box(0.14)]) == [7]
    assert _labels(far, [{7: 0.9, K: 0.05}], [_box(0.16)]) == [K]


def test_uses_last_seen_box_not_first():
    # Track moved 0.1 -> 0.8; it was unseen in the final row, so its last box is at 0.8.
    mk = lambda: _tracker([[_box(0.1)], [_box(0.8)], [None]], [7], spatial_gate=True, gate_base=1.0)
    assert _labels(mk(), [{7: 0.9, K: 0.05}], [_box(0.85)]) == [7]
    assert _labels(mk(), [{7: 0.9, K: 0.05}], [_box(0.1)]) == [K]


def test_distance_is_in_units_of_the_tracks_box_height():
    # The same 0.2 (200 px) offset: 2 heights for a short far-side box, 0.5 for a tall near-side one.
    short = _tracker([[_box(0.1, h=0.1)]], [7], spatial_gate=True, gate_base=1.0, gate_per_step=0.0)
    tall = _tracker([[_box(0.1, h=0.4)]], [7], spatial_gate=True, gate_base=1.0, gate_per_step=0.0)
    rows = [{7: 0.9, K: 0.05}]
    assert _labels(short, rows, [_box(0.3)]) == [K]
    assert _labels(tall, rows, [_box(0.3)]) == [7]


def test_gate_zeroes_scores_and_leaves_the_rest_alone():
    tr = _tracker([[_box(0.1), _box(0.5)]], [7, 3], spatial_gate=True, gate_base=1.0, gate_per_step=0.0)
    scores = torch.rand(2, K + 1, device=distributed_device()).softmax(-1)
    boxes = torch.tensor([_box(0.12), _box(0.9)], device=distributed_device())
    out = tr._apply_spatial_gate(scores, boxes)
    expected = scores.clone()
    expected[0, 3] = 0        # det 0 is far from track 3
    expected[1, 7] = 0        # det 1 is far from track 7
    expected[1, 3] = 0        # and from track 3
    assert torch.equal(out, expected)
    assert not torch.equal(out, scores)   # the input was not modified in place


def test_disabled_output_is_identical_whatever_the_gate_parameters():
    torch.manual_seed(0)
    dets = [_box(0.1), _box(0.5), _box(0.9), _box(0.3)]
    rows = [dict(enumerate(torch.rand(K + 1).softmax(0).tolist())) for _ in dets]
    hist, labels = [[_box(0.1), _box(0.5), _box(0.9)]], [1, 4, 7]
    base = _labels(_tracker(hist, labels), rows, dets)
    off = _labels(_tracker(hist, labels, spatial_gate=False, gate_base=0.0, gate_per_step=0.0), rows, dets)
    # A gate wide enough to never fire gives the same assignment as no gate.
    wide = _labels(_tracker(hist, labels, spatial_gate=True, gate_base=1e6), rows, dets)
    assert base == off == wide


def test_negative_parameters_are_rejected():
    with pytest.raises(ValueError, match="gate_base"):
        RuntimeTracker(model=_StubModel(), sequence_hw=(H, W), spatial_gate=True, gate_base=-0.1)


def test_gate_off_never_sets_cut_or_stats():
    tr = _tracker([[_box(0.1), _box(0.2)]], [7, 3])
    _labels(tr, [{7: 0.9, K: 0.05}], [_box(0.9)])
    assert tr._gate_cut is False and tr.gate_stats["frames"] == 0


def _process(tr, rows, dets, score):
    tr.model.logits = _logits(rows)
    dev = distributed_device()
    boxes = torch.tensor(dets, device=dev)
    n = len(dets)
    tr._process_detections(
        scores=torch.full((n,), score, device=dev), categories=torch.zeros(n, dtype=torch.int64, device=dev),
        boxes=boxes, output_embeds=torch.zeros(n, 256, device=dev),
    )
    return tr.get_track_results()


def test_cut_every_pair_gated_keeps_all_detections_as_newborns():
    # Two tracks at the left, every detection far away on the right, scores below newborn_thresh.
    tr = _tracker([[_box(0.1), _box(0.2)]], [7, 3], spatial_gate=True, gate_base=1.0, gate_per_step=0.0)
    res = _process(tr, [{7: 0.9, K: 0.05}, {3: 0.9, K: 0.05}], [_box(0.8), _box(0.9)], score=0.4)
    assert tr.gate_stats["cut_frames"] == 1
    assert len(res["id"]) == 2
    assert len(set(res["id"].tolist())) == 2                 # fresh track ids, not the old ones (0, 1)
    assert not set(res["id"].tolist()) & {0, 1}


def test_partial_gating_is_not_a_cut_and_a_weak_newborn_is_dropped_as_usual():
    tr = _tracker([[_box(0.1), _box(0.2)]], [7, 3], spatial_gate=True, gate_base=1.0, gate_per_step=0.0)
    res = _process(tr, [{7: 0.9, K: 0.05}, {3: 0.9, K: 0.05}], [_box(0.12), _box(0.9)], score=0.4)
    assert tr.gate_stats["cut_frames"] == 0
    assert res["id"].tolist() == [0]                          # det 0 kept track 7 (= output id 0); det 1 dropped


def test_without_the_gate_the_same_far_detections_keep_their_old_ids():
    tr = _tracker([[_box(0.1), _box(0.2)]], [7, 3])
    res = _process(tr, [{7: 0.9, K: 0.05}, {3: 0.9, K: 0.05}], [_box(0.8), _box(0.9)], score=0.4)
    assert sorted(res["id"].tolist()) == [0, 1]
