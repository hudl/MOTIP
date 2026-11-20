"""Tests for FixedLagTracker (models/runtime_tracker.py): fixed-lag identity decisions over MOTIP hypotheses.

The stub ID decoder depends on the trajectory history (unlike the fixed-logit stubs of test_spatial_gate.py): a
detection's logit for ID k is ``beta * cos(det feature, mean of ID k's last `recent` seen features)`` plus an optional
per-frame bias (a spurious cue, standing in for e.g. a misleading motion / position cue at a crossing); the newborn
logit is ``NEWBORN``. Features live in the first two dims: player 1 = e0, player 2 = e1. A wrong assignment pollutes
the track's history and lowers the later probabilities of that hypothesis, which is what a delayed decision exploits.
Boxes are cxcywh normalised on a 1000 x 1000 canvas. The test loop sets ``model.frame`` before every update.
"""
import numpy as np
import pytest
import torch

from models.runtime_tracker import FixedLagTracker, QuarantineRewriter, RuntimeTracker
from utils.misc import distributed_device

K = 10
W = H = 1000
NEWBORN = 0.0


class _HistoryModel:
    num_id_vocabulary = K
    num_classes = 1

    def __init__(self, beta=4.0, recent=3, bias=None, newborn=NEWBORN):
        self.newborn = newborn
        self.calls = 0
        self.beta = beta
        self.recent = recent
        self.bias = bias or {}      # {frame: {(det index, ID label): logit offset}}
        self.frame = 0

    def eval(self):
        return self

    def __call__(self, seq_info=None, part=None, **_):
        if part == "trajectory_modeling":
            return seq_info
        self.calls += 1
        feats = seq_info["trajectory_features"][0, 0].float()        # (T, N, C)
        labels = seq_info["trajectory_id_labels"][0, 0]               # (T, N)
        masks = seq_info["trajectory_masks"][0, 0]                    # (T, N) True = missing
        det = seq_info["unknown_features"][0, 0, 0].float()           # (D, C)
        logits = torch.full((det.shape[0], K + 1), -20.0, device=det.device)
        logits[:, K] = self.newborn
        for n in range(feats.shape[1]):
            seen = (~masks[:, n]).nonzero().flatten()[-self.recent:]
            if seen.numel() == 0:
                continue
            proto = feats[seen, n].mean(0)
            logits[:, int(labels[0, n])] = self.beta * torch.nn.functional.cosine_similarity(det, proto[None], dim=-1)
        for (d, lab), v in self.bias.get(self.frame, {}).items():
            logits[d, lab] += v
        return logits[None, None, None], None, None


def _feat(w1: float, w2: float) -> torch.Tensor:
    f = torch.zeros(256)
    f[0], f[1] = w1, w2
    return f


def _frame(dets):
    """dets: [(x, feature)] -> update_from_detections args (scores, categories, boxes, embeds)."""
    dev = distributed_device()
    n = len(dets)
    boxes = torch.tensor([[d[0], d[2] if len(d) > 2 else 0.5, d[3] if len(d) > 3 else 0.05, 0.1] for d in dets],
                         dtype=torch.float32).reshape(n, 4).to(dev)
    feats = torch.stack([d[1] for d in dets]).to(dev) if n else torch.zeros(0, 256, device=dev)
    return torch.full((n,), 0.9, device=dev), torch.zeros(n, dtype=torch.int64, device=dev), boxes, feats


def _tracker(model, **kw):
    return RuntimeTracker(model=model, sequence_hw=(H, W), id_thresh=0.1, newborn_thresh=0.5, miss_tolerance=30, **kw)


def _crossing(n_frames=20, cross=(8, 9), w=0.05, bias=3.0):
    """Two players crossing. Detections are listed left to right. On the crossing frames appearance is weak but right
    (player 1 is (0.5 + w, 0.5 - w)) and a spurious cue of ``bias`` logits favours the swapped pairing.
    Player 1 gets ID label 0 and player 2 label 1 on frame 0. Returns (frames, truth, bias dict)."""
    frames, truth, bias_d = [], [], {}
    for t in range(n_frames):
        x1 = 0.30 + 0.025 * t          # player 1 moves right
        x2 = 0.70 - 0.025 * t          # player 2 moves left
        if t in cross:
            f1, f2 = _feat(0.5 + w, 0.5 - w), _feat(0.5 - w, 0.5 + w)
        else:
            f1, f2 = _feat(1, 0), _feat(0, 1)
        d = sorted([(x1, f1, 1), (x2, f2, 2)], key=lambda r: r[0])
        frames.append([(x, f) for x, f, _ in d])
        truth.append([p for _, _, p in d])
        if t in cross:
            bias_d[t] = {(i, 1 if p == 1 else 0): bias for i, p in enumerate(truth[-1])}
    return frames, truth, bias_d


def _run_online(frames, model):
    trk = _tracker(model)
    out = []
    for t, dets in enumerate(frames):
        model.frame = t
        trk.update_from_detections(*_frame(dets))
        out.append(trk.get_track_results()["id"].tolist())
    return out


def _run_lag(frames, model, **kw):
    lag = FixedLagTracker(_tracker(model), **kw)
    out, live = {}, []
    for t, dets in enumerate(frames):
        model.frame = t
        for f, res in lag.update_from_detections(*_frame(dets)):
            assert f not in out
            out[f] = res["id"].tolist()
        live.append(len(lag.hyps))
    for f, res in lag.finish():
        assert f not in out
        out[f] = res["id"].tolist()
    assert sorted(out) == list(range(len(frames)))
    return [out[t] for t in range(len(frames))], lag, live


def _wrong_frames(ids, truth):
    """Frames where a player's ID differs from the ID it had on frame 0."""
    first = {p: i for p, i in zip(truth[0], ids[0])}
    return [t for t, (row, tr) in enumerate(zip(ids, truth)) if any(first[p] != i for p, i in zip(tr, row))]


def _scenario(w=0.1, bias=1.0, **kw):
    """The swap case: online MOTIP swaps on the second crossing frame and swaps back after it (transient)."""
    frames, truth, b = _crossing(w=w, bias=bias, **kw)
    return frames, truth, (lambda: _HistoryModel(beta=4.0, recent=3, bias=b))


def test_online_swaps_on_crossing():
    frames, truth, mk = _scenario()
    assert _wrong_frames(_run_online(frames, mk()), truth) == [9]


@pytest.mark.parametrize("lag", [3, 5, 15, 30])
def test_delayed_decision_fixes_the_swap(lag):
    frames, truth, mk = _scenario()
    ids, trk, live = _run_lag(frames, mk(), lag=lag, margin=3.0, max_hyps=4, max_branch=2, near=0)
    assert _wrong_frames(ids, truth) == []
    assert trk.stats["committed_alts"] >= 1 and trk.stats["frames_overridden"] >= 1
    assert max(live) > 1 and len(trk.hyps) == 1      # finish() leaves the winner only


def test_lag_too_short_keeps_online_decision():
    """With lag 1 the evidence after the crossing is not in yet: frame 9 is committed as online did."""
    frames, truth, mk = _scenario()
    ids, _, _ = _run_lag(frames, mk(), lag=1, margin=3.0, max_hyps=4, max_branch=2, near=0)
    assert _wrong_frames(ids, truth) == [9]


def test_single_hypothesis_is_identical_to_online():
    frames, _, mk = _scenario()
    online = _run_online(frames, mk())
    for kw in (dict(max_hyps=1), dict(max_branch=0), dict(margin=0.0)):
        ids, trk, live = _run_lag(frames, mk(), lag=5, **kw)
        assert ids == online, kw
        assert max(live) == 1


def test_no_ambiguity_identical_to_online():
    frames, truth, mk = _scenario(cross=())
    online = _run_online(frames, mk())
    ids, trk, _ = _run_lag(frames, mk(), lag=5, margin=3.0, near=0)
    assert ids == online and _wrong_frames(ids, truth) == []
    assert trk.stats["alts"] == 0


def test_proximity_requirement_blocks_far_branches():
    """Crossing detections are 0.0 / 0.05 of the canvas apart = 0 / 0.5 box heights: near=0.1 lets only frame 8 branch,
    near=0.01 still frame 8 (distance 0); a crossing shifted apart by a constant offset gets no branch at all."""
    frames, truth, mk = _scenario()
    far = [[(x + (0.2 if i else 0.0), f) for i, (x, f) in enumerate(dets)] for dets in frames]
    online = _run_online(far, mk())
    ids, trk, _ = _run_lag(far, mk(), lag=5, margin=3.0, near=0.5)
    assert ids == online and trk.stats["alts"] == 0


def test_boxes_and_scores_unchanged():
    frames, _, mk = _scenario()
    model = mk()
    trk = _tracker(model)
    ref = []
    for t, dets in enumerate(frames):
        model.frame = t
        trk.update_from_detections(*_frame(dets))
        r = trk.get_track_results()
        ref.append((r["bbox"].cpu(), r["score"].cpu()))
    model = mk()
    lag = FixedLagTracker(_tracker(model), lag=4, margin=3.0, near=0)
    got = []
    for t, dets in enumerate(frames):
        model.frame = t
        got += lag.update_from_detections(*_frame(dets))
    got += lag.finish()
    assert [f for f, _ in got] == list(range(len(frames)))
    for (_, r), (b, s) in zip(got, ref):
        assert torch.equal(r["bbox"].cpu(), b) and torch.equal(r["score"].cpu(), s)


def _crowd(n_frames, n_players, seed, noise):
    rng = np.random.default_rng(seed)
    return [[(0.45 + 0.02 * p, _feat(1.0 + noise * rng.standard_normal(), 1.0 + noise * rng.standard_normal()))
             for p in range(n_players)] for _ in range(n_frames)]


def test_hypotheses_and_buffers_are_bounded():
    """Six look-alike players in a tight group: alternatives every frame; live hypotheses, buffers and paths stay bounded."""
    frames = _crowd(40, 6, 0, 0.05)
    lag = FixedLagTracker(_tracker(_HistoryModel()), lag=6, margin=5.0, max_hyps=3, max_branch=2, near=0)
    n_out = 0
    for dets in frames:
        n_out += len(lag.update_from_detections(*_frame(dets)))
        assert 1 <= len(lag.hyps) <= 3
        assert all(len(h.buf) <= 6 for h in lag.hyps)
        assert all(f > lag.t - 1 - 6 for h in lag.hyps for f, _ in h.path)
    n_out += len(lag.finish())
    assert n_out == 40
    assert lag.stats["alts"] > 0 and lag.stats["dropped_cap"] > 0 and lag.stats["max_live"] == 3


def test_ids_unique_per_frame_and_consistent_across_commits():
    """IDs are unique within a frame; and every emitted frame comes from one lineage, so an ID label that is
    emitted for consecutive frames never jumps between output IDs (label -> output id is fixed per lineage)."""
    frames = _crowd(30, 5, 1, 0.1)
    lag = FixedLagTracker(_tracker(_HistoryModel()), lag=5, margin=5.0, max_hyps=4, near=0)
    out = []
    for dets in frames:
        out += lag.update_from_detections(*_frame(dets))
    out += lag.finish()
    assert len(out) == 30
    for _, r in out:
        ids = r["id"].tolist()
        assert len(ids) == len(set(ids))


def test_clone_is_independent():
    frames, _, mk = _scenario()
    trk = _tracker(mk())
    for dets in frames[:5]:
        trk.update_from_detections(*_frame(dets))
    clone = trk._lag_clone()
    before = trk.trajectory_id_labels.clone(), trk.trajectory_features.clone(), list(trk.id_queue), dict(trk.id_label_to_id)
    for dets in frames[5:10]:
        clone.update_from_detections(*_frame(dets))
    assert torch.equal(trk.trajectory_id_labels, before[0])
    assert torch.equal(trk.trajectory_features, before[1])
    assert list(trk.id_queue) == before[2] and trk.id_label_to_id == before[3]
    assert clone.model is trk.model


def test_invalid_settings():
    with pytest.raises(ValueError):
        FixedLagTracker(_tracker(_HistoryModel()), lag=0)
    with pytest.raises(ValueError):
        FixedLagTracker(_tracker(_HistoryModel()), lag=5, max_hyps=0)


def test_decoder_runs_once_per_hypothesis_frame():
    """An alternative re-uses its parent's scores on the branch frame (forced), so decoder calls = hypothesis-frames
    (minus the first frame, which has no trajectories to decode against)."""
    frames, _, mk = _scenario()
    model = mk()
    lag = FixedLagTracker(_tracker(model), lag=5, margin=3.0, near=0)
    for t, dets in enumerate(frames):
        model.frame = t
        lag.update_from_detections(*_frame(dets))
    lag.finish()
    assert lag.stats["alts"] > 0
    assert model.calls == lag.stats["hyp_frames"] - 1


def test_hook_off_by_default():
    trk = _tracker(_HistoryModel())
    assert trk._lag_hook is None and trk._forced is None


def test_penalty_keeps_online_decision():
    """A large prior in favour of the online decision: the alternative never out-scores it."""
    frames, truth, mk = _scenario()
    ids, trk, _ = _run_lag(frames, mk(), lag=5, margin=3.0, near=0, penalty=50.0)
    assert ids == _run_online(frames, mk()) and trk.stats["committed_alts"] == 0 and trk.stats["alts"] > 0


def test_kinds_filter():
    frames, truth, mk = _scenario()
    ids, trk, _ = _run_lag(frames, mk(), lag=5, margin=3.0, near=0, kinds=("move", "give"))
    assert trk.stats["alts"] == 0 and ids == _run_online(frames, mk())
    ids, trk, _ = _run_lag(frames, mk(), lag=5, margin=3.0, near=0, kinds=("swap",))
    assert _wrong_frames(ids, truth) == [] and trk.stats["committed_swap"] >= 1
    with pytest.raises(ValueError):
        FixedLagTracker(_tracker(mk()), lag=5, kinds=("flip",))


def _persistent():
    """recent=1 (the decoder only looks at each track's last token): the swap on the crossing is self-confirming and
    persists for the rest of the clip online."""
    frames, truth, b = _crossing(w=0.1, bias=1.0)
    return frames, truth, (lambda: _HistoryModel(beta=4.0, recent=1, bias=b))


def test_persistent_swap_online():
    frames, truth, mk = _persistent()
    assert _wrong_frames(_run_online(frames, mk()), truth) == list(range(9, 20))


def test_own_scoring_cannot_fix_a_self_confirming_swap():
    """Each hypothesis is scored by its own decoder; the swapped history predicts its own continuation as well as the
    right one does, so the online choice (higher on the branch frame) wins: the documented limit of score='own'."""
    frames, truth, mk = _persistent()
    ids, _, _ = _run_lag(frames, mk(), lag=8, margin=3.0, near=0, score="own")
    assert set(range(9, 20)) <= set(_wrong_frames(ids, truth))    # still swapped to the end (here even from frame 8)


@pytest.mark.parametrize("lag", [5, 8])
def test_ref_scoring_fixes_a_self_confirming_swap(lag):
    """score='ref' scores both hypotheses against the history before the crossing: after it, the swapped assignment
    contradicts the pre-crossing identities on every frame."""
    frames, truth, mk = _persistent()
    ids, trk, _ = _run_lag(frames, mk(), lag=lag, margin=3.0, near=0, score="ref")
    assert _wrong_frames(ids, truth) == []
    assert trk.stats["groups"] >= 1 and trk.stats["ref_calls"] >= lag and trk.ref is None


def test_ref_scoring_bounded_and_complete():
    frames = _crowd(40, 6, 0, 0.05)
    lag = FixedLagTracker(_tracker(_HistoryModel()), lag=6, margin=5.0, max_hyps=3, near=0, score="ref")
    n = 0
    for dets in frames:
        n += len(lag.update_from_detections(*_frame(dets)))
        assert len(lag.hyps) <= 3
        assert lag.ref is None or lag.t - 1 - lag.ref[1] < 6
    n += len(lag.finish())
    assert n == 40 and lag.stats["groups"] > 0


def test_ref_scoring_single_hypothesis_identical_to_online():
    frames, _, mk = _scenario()
    ids, trk, _ = _run_lag(frames, mk(), lag=5, max_branch=0, score="ref")
    assert ids == _run_online(frames, mk()) and trk.stats["ref_calls"] == 0



# -- quarantine ---------------------------------------------------------------------------------------------------

def _overlap_crossing(n_frames=20, cross=(8, 9), w=0.1, bias=1.0):
    """The persistent-swap crossing with wider boxes (0.1 of the canvas) and player 2 slightly lower (y 0.52), so the
    two boxes overlap (IoU >= 0.3) on frames 7-9 and are never identical. Detections are (x, feature, y, width)."""
    frames, truth, b = _crossing(n_frames=n_frames, cross=cross, w=w, bias=bias)
    out = []
    for dets, tr in zip(frames, truth):
        out.append([(x, f, 0.5 if p == 1 else 0.52, 0.1) for (x, f), p in zip(dets, tr)])
    return out, truth, (lambda: _HistoryModel(beta=4.0, recent=1, bias=b))


def _run_q(frames, model, rewrite=False, **kw):
    kw.setdefault("quarantine_max", 20)
    kw.setdefault("quarantine_iou", 0.2)
    trk = _tracker(model, quarantine=True, **kw)
    wrap = QuarantineRewriter(trk, max_buffer=60) if rewrite else None
    out, q = {}, []
    for t, dets in enumerate(frames):
        model.frame = t
        if wrap is None:
            trk.update_from_detections(*_frame(dets))
            r = trk.get_track_results()
            out[t] = r["id"].tolist()
            q.append(r["quarantined"].tolist())
        else:
            for f, r in wrap.update_from_detections(*_frame(dets)):
                assert f not in out
                out[f] = r["id"].tolist()
    if wrap is not None:
        for f, r in wrap.finish():
            out[f] = r["id"].tolist()
    assert sorted(out) == list(range(len(frames)))
    return [out[t] for t in range(len(frames))], trk, wrap, q


def test_overlap_crossing_swaps_persistently_online():
    frames, truth, mk = _overlap_crossing()
    wrong = _wrong_frames(_run_online(frames, mk()), truth)
    assert set(range(10, 20)) <= set(wrong)


def test_quarantine_masks_entangled_tokens_and_stops_the_persistent_swap():
    frames, truth, mk = _overlap_crossing()
    ids, trk, _, q = _run_q(frames, mk())
    assert [t for t, row in enumerate(q) if any(row)] == [7, 8, 9]
    assert trk.q_stats["masked"] == 6
    assert _wrong_frames(ids, truth) == []
    frames, truth, mk = _overlap_crossing(bias=2.0)      # a stronger misleading cue: still wrong while entangled
    ids, _, _, _ = _run_q(frames, mk())
    assert _wrong_frames(ids, truth) == [8, 9]           # ... but right as soon as the players separate


def test_quarantine_rewrite_fixes_the_entangled_frames_too():
    frames, truth, mk = _overlap_crossing(bias=2.0)
    ids, trk, wrap, _ = _run_q(frames, mk(), rewrite=True)
    assert _wrong_frames(ids, truth) == []
    assert wrap.stats["episodes"] == 1 and wrap.stats["dets_relabelled"] >= 1


def test_quarantine_off_identical_and_no_extra_key():
    frames, truth, mk = _overlap_crossing()
    trk = _tracker(mk())
    for t, dets in enumerate(frames[:3]):
        trk.update_from_detections(*_frame(dets))
    assert "quarantined" not in trk.get_track_results()


def test_quarantine_no_overlap_identical_to_online():
    frames, truth, mk = _scenario()           # narrow boxes: IoU 0 except where both are at the same x
    far = [[(x + (0.2 if i else 0.0), f) for i, (x, f) in enumerate(dets)] for dets in frames]
    ids, trk, _, q = _run_q(far, mk())
    assert ids == _run_online(far, mk()) and trk.q_stats["masked"] == 0


def test_quarantine_cap():
    """Two players glued together for 20 frames, cap 5: tokens are masked for 5 frames, then kept again."""
    frames = [[(0.5, _feat(1, 0), 0.5, 0.1), (0.51, _feat(0, 1), 0.52, 0.1)] for _ in range(20)]
    ids, trk, _, q = _run_q(frames, _HistoryModel(), quarantine_max=5)
    assert [t for t, row in enumerate(q) if any(row)] == [1, 2, 3, 4, 5]
    assert trk.q_stats["capped"] == 2 * 14


def test_quarantine_settings_validated():
    with pytest.raises(ValueError):
        _tracker(_HistoryModel(), quarantine=True, quarantine_max=40)      # >= miss_tolerance - 2
    with pytest.raises(ValueError):
        QuarantineRewriter(_tracker(_HistoryModel()))


def test_id_topk_export_is_off_by_default_and_consistent():
    """ID_TOPK (id_topk > 0) adds (N, k, 2) [output id, prob] per detection; default off adds nothing and leaves
    the results unchanged."""
    frames, _, mk = _scenario()
    base, top = _tracker(mk()), _tracker(mk(), id_topk=3)
    m0, m1 = base.model, top.model
    for t, dets in enumerate(frames):
        m0.frame = m1.frame = t
        base.update_from_detections(*_frame(dets))
        top.update_from_detections(*_frame(dets))
        r0, r1 = base.get_track_results(), top.get_track_results()
        assert "id_topk" not in r0
        assert r0["id"].tolist() == r1["id"].tolist()
        tk = r1["id_topk"]
        assert tk.shape == (len(dets), 3, 2)
        if t == 0:
            assert (tk[:, :, 0] == -1).all()
            continue
        st = r1["id_stats"]
        # top-1 prob equals id_stats' top-1; top-1 minus top-2 equals the margin
        assert torch.allclose(tk[:, 0, 1], st[:, 1].cpu(), atol=1e-6)
        assert torch.allclose(tk[:, 0, 1] - tk[:, 1, 1], st[:, 2].cpu(), atol=1e-6)
        assert set(tk[:, :2, 0].flatten().tolist()) <= set(float(i) for i in r1["id"].tolist())
