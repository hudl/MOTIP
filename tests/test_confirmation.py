"""Tests for tracklet confirmation with backfill (RuntimeTracker n_init + ConfirmFilter) and the new-track overlap rule
(RuntimeTracker newborn_max_iou). Same history-dependent stub decoder as test_fixed_lag.py, newborn logit 2: a player
matches its own track at logit 4, a detection that looks like nobody (feature e2) is a newborn."""
import torch

from models.runtime_tracker import ConfirmFilter, FixedLagTracker
from test_fixed_lag import _HistoryModel, _feat, _frame, _tracker


def _ghost():
    f = torch.zeros(256)
    f[2] = 1.0
    return f


def _players(t):
    return [(0.3, _feat(1, 0), 0.5, 0.05), (0.7, _feat(0, 1), 0.5, 0.05)]


def _run(frames, lag=None, **kw):
    model = _HistoryModel(newborn=2.0)
    trk = _tracker(model, **kw)
    wrap = FixedLagTracker(trk, lag=lag, margin=3.0, near=0) if lag else None
    conf = ConfirmFilter(trk.n_init) if trk.n_init > 1 else None
    out = {}

    def emit(ready, final=False):
        if conf is not None:
            ready = conf.push(ready) + (conf.finish() if final else [])
        for f, r in ready:
            assert f not in out
            out[f] = sorted(zip(r["id"].tolist(), [round(float(x), 3) for x in r["bbox"][:, 0].tolist()]))

    for t, dets in enumerate(frames):
        model.frame = t
        if wrap is None:
            trk.update_from_detections(*_frame(dets))
            emit([(t, trk.get_track_results())])
        else:
            emit(wrap.update_from_detections(*_frame(dets)))
    emit(wrap.finish() if wrap else [], final=True)
    assert sorted(out) == list(range(len(frames)))
    return [out[t] for t in range(len(frames))], (wrap.tracker if wrap else trk)


def _with(frames_extra, n=12):
    """Two steady players plus extra detections {frame: [det, ...]}."""
    return [_players(t) + frames_extra.get(t, []) for t in range(n)]


def _ids(rows):
    return {i for row in rows for i, _ in row}


def test_n_init_1_is_unchanged():
    frames = _with({5: [(0.5, _ghost(), 0.5, 0.05)]})
    ref, _ = _run(frames)
    got, trk = _run(frames, n_init=1)
    assert got == ref and "dropped_ids" not in trk.get_track_results()
    assert len(_ids(ref)) == 3                     # the one-frame ghost is a track online


def test_one_frame_ghost_is_dropped_with_its_box():
    frames = _with({5: [(0.5, _ghost(), 0.5, 0.05)]})
    ref, _ = _run(frames)
    got, trk = _run(frames, n_init=2)
    assert len(_ids(got)) == 2 and len(got[5]) == 2
    assert [r for t, r in enumerate(got) if t != 5] == [r for t, r in enumerate(ref) if t != 5]
    assert trk.n_stats["dropped"] == 1
    assert trk.trajectory_id_labels.shape[1] == 2  # its trajectory is gone, it cannot take later detections


def test_new_player_is_backfilled_once_confirmed():
    new = {t: [(0.5, _ghost(), 0.5, 0.05)] for t in range(5, 12)}
    got, trk = _run(_with(new), n_init=3)
    assert len(got[5]) == 3 and len(got[6]) == 3   # tentative frames 5, 6 kept (backfill)
    nid = {i for i, x in got[5]} - {i for i, x in got[4]}
    assert all(nid <= {i for i, _ in got[t]} for t in range(5, 12))
    assert trk.n_stats["confirmed"] >= 1 and trk.n_stats["dropped"] == 0


def test_two_frame_ghost_dropped_with_n_init_3():
    frames = _with({5: [(0.5, _ghost(), 0.5, 0.05)], 6: [(0.5, _ghost(), 0.5, 0.05)]})
    got, trk = _run(frames, n_init=3)
    assert len(_ids(got)) == 2 and len(got[5]) == len(got[6]) == 2


def test_first_frame_tracks_kept_like_a_segment_restart():
    """Every track is newborn on the first frame of a segment; matched again on frame 1 -> confirmed, frame 0 kept."""
    got, trk = _run(_with({}, n=4), n_init=2)
    assert all(len(r) == 2 for r in got) and trk.n_stats["confirmed"] == 2


def test_confirmation_through_fixed_lag():
    frames = _with({5: [(0.5, _ghost(), 0.5, 0.05)]})
    got, trk = _run(frames, lag=4, n_init=2)
    ref, _ = _run(frames, n_init=2)
    assert got == ref


def test_newborn_overlap_rule_blocks_duplicate_only():
    dup = (0.302, _ghost(), 0.5, 0.05)           # IoU ~0.92 with player 1's box
    side = (0.33, _ghost(), 0.5, 0.05)           # IoU ~0.25: a different player
    frames = _with({5: [dup], 7: [side]})
    off, _ = _run(frames)
    on, trk = _run(frames, newborn_max_iou=0.8)
    assert len(off[5]) == 3 and len(on[5]) == 2  # duplicate not given a new ID
    assert len(on[7]) == 3                       # a non-duplicate newborn still starts
    assert [r for t, r in enumerate(on) if t not in (5, 7)] == [r for t, r in enumerate(off) if t not in (5, 7)]
    assert trk.n_stats["blocked_newborns"] == 1
