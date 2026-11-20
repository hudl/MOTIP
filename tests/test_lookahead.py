"""Tests for the look-ahead ID decoder (models/motip/lookahead.py), its inference wrapper (models/lookahead_tracker.py)
and the crossing-contamination augmentation (data/transforms.py TurnIntoTrajectoryAndUnknown._contaminate)."""
import random

import pytest
import torch

from data.transforms import TurnIntoTrajectoryAndUnknown
from models.lookahead_tracker import LookaheadTracker
from models.motip.id_decoder import IDDecoder
from models.motip.lookahead import LookaheadBlock, future_from_unknown, pair_features
from models.runtime_tracker import RuntimeTracker
from utils.misc import distributed_device

VOCAB = 10


def _decoder(lookahead=None, feature_dim=32, id_dim=32, layers=3, rel_pe=20, seed=0):
    torch.manual_seed(seed)
    return IDDecoder(feature_dim=feature_dim, id_dim=id_dim, ffn_dim_ratio=2, num_layers=layers, head_dim=8,
                     num_id_vocabulary=VOCAB, rel_pe_length=rel_pe, use_aux_loss=False, use_shared_aux_head=True,
                     lookahead=lookahead)


def _seq_info(B=1, G=2, T=6, N=4, C=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    times = torch.arange(T)[None, None, :, None].expand(B, G, T, N).clone()
    labels = torch.stack([torch.randperm(VOCAB, generator=g)[:N] for _ in range(B * G)]).reshape(B, G, 1, N).expand(B, G, T, N).clone()
    masks = torch.rand((B, G, T, N), generator=g) < 0.15
    boxes = torch.cat([torch.rand((B, G, T, N, 2), generator=g) * 0.8 + 0.1,
                       torch.rand((B, G, T, N, 2), generator=g) * 0.1 + 0.05], dim=-1)
    return {
        "trajectory_features": torch.randn((B, G, T, N, C), generator=g),
        "trajectory_boxes": boxes.clone(),
        "trajectory_id_labels": labels,
        "trajectory_times": times.clone(),
        "trajectory_masks": masks.clone(),
        "unknown_features": torch.randn((B, G, T, N, C), generator=g),
        "unknown_boxes": boxes.clone(),
        "unknown_id_labels": labels.clone(),
        "unknown_times": times.clone(),
        "unknown_masks": masks.clone(),
    }


def _randomise_lookahead(dec, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in dec.lookahead.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def _la_from(plain, **kw):
    la = _decoder(lookahead=dict(max_steps=kw.pop("max_steps", 3), layers=kw.pop("layers", None)), **kw)
    missing, unexpected = la.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(".lookahead." in "." + k or k.startswith("lookahead.") for k in missing)
    return la


def test_default_off_has_no_lookahead():
    dec = _decoder()
    assert dec.lookahead is None
    assert not any("lookahead" in k for k in dec.state_dict())


def test_building_lookahead_does_not_move_global_rng():
    torch.manual_seed(5)
    a =IDDecoder(32, 32, 2, 3, 8, VOCAB, 20, False, True)
    r1 = torch.rand(3)
    torch.manual_seed(5)
    b = IDDecoder(32, 32, 2, 3, 8, VOCAB, 20, False, True, lookahead=dict(max_steps=3))
    r2 = torch.rand(3)
    assert torch.equal(r1, r2)
    assert torch.equal(a.word_to_embed.weight, b.word_to_embed.weight)


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_fresh_lookahead_is_identity(steps):
    plain = _decoder().eval()
    la = _la_from(plain).eval()
    si = _seq_info()
    with torch.no_grad():
        ref, _, _ = plain(dict(si), use_decoder_checkpoint=False)
        out, _, _ = la({**si, "lookahead_steps": torch.tensor([steps])}, use_decoder_checkpoint=False)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()


def test_training_lag_zero_matches_plain_and_aux_shapes():
    plain = _decoder().train()
    la = _la_from(plain).train()
    si = _seq_info()
    ref, ref_lab, ref_m = plain(dict(si), use_decoder_checkpoint=False)
    out, lab, m = la({**si, "lookahead_steps": torch.tensor([0])}, use_decoder_checkpoint=True)
    assert torch.allclose(ref, out, atol=1e-5)
    assert torch.equal(ref_lab, lab) and torch.equal(ref_m, m)


def test_training_samples_lag_in_range():
    la = _decoder(lookahead=dict(max_steps=3)).train()
    _randomise_lookahead(la)
    si = _seq_info()
    torch.manual_seed(0)
    outs = {tuple(la(dict(si), use_decoder_checkpoint=False)[0].flatten()[:5].tolist()) for _ in range(12)}
    assert len(outs) > 1   # different sampled lags give different outputs


def test_no_id_or_future_leak():
    """Query frame t sees past IDs only (< t) and future detections only within its lag."""
    la = _decoder(lookahead=dict(max_steps=3)).eval()
    _randomise_lookahead(la)
    si = _seq_info(T=8)
    t, lag = 3, 2
    run = lambda s: la({**s, "lookahead_steps": torch.tensor([lag])}, use_decoder_checkpoint=False)[0][:, :, t]
    with torch.no_grad():
        ref = run(si)
        s = {k: v.clone() for k, v in si.items()}
        s["trajectory_id_labels"][:, :, t:] = (s["trajectory_id_labels"][:, :, t:] + 1) % VOCAB   # IDs at >= t
        s["trajectory_features"][:, :, t:] += 5.0                                                     # history at >= t
        assert torch.allclose(run(s), ref, atol=1e-5)
        s = {k: v.clone() for k, v in si.items()}
        s["unknown_features"][:, :, t + lag + 1:] += 5.0                                              # beyond the lag
        s["unknown_id_labels"][:] = 0                                                                 # labels unused
        assert torch.allclose(run(s), ref, atol=1e-5)
        s = {k: v.clone() for k, v in si.items()}
        s["unknown_features"][:, :, t + 1] += 5.0                                                     # within the lag
        assert not torch.allclose(run(s), ref, atol=1e-3)
        s = {k: v.clone() for k, v in si.items()}
        s["trajectory_id_labels"][:, :, :t] = (s["trajectory_id_labels"][:, :, :t] + 1) % VOCAB      # past IDs matter
        assert not torch.allclose(run(s), ref, atol=1e-3)


def test_explicit_future_tokens_and_padding():
    """Inference form: one query frame with given future tokens; all-padding future == no future."""
    la = _decoder(lookahead=dict(max_steps=3)).eval()
    _randomise_lookahead(la)
    si = _seq_info(T=6)
    T = 5
    q = {k: (v[:, :, :T] if k.startswith("trajectory") else v[:, :, T:T + 1]) for k, v in si.items()}
    fut = {"future_features": torch.randn((1, 2, 1, 3, 4, 32)), "future_boxes": si["unknown_boxes"][:, :, :3][:, :, None],
           "future_masks": torch.zeros((1, 2, 1, 3, 4), dtype=torch.bool),
           "future_times": torch.tensor([T + 1, T + 2, T + 3])[None, None, None].expand(1, 2, 1, 3),
           "future_steps": torch.tensor([1, 2, 3])[None, None, None].expand(1, 2, 1, 3)}
    with torch.no_grad():
        base, _, _ = la({**q, "lookahead_steps": torch.tensor([0])}, use_decoder_checkpoint=False)
        withf, _, _ = la({**q, **fut}, use_decoder_checkpoint=False)
        padded, _, _ = la({**q, **fut, "future_masks": torch.ones((1, 2, 1, 3, 4), dtype=torch.bool)},
                          use_decoder_checkpoint=False)
    assert base.shape == withf.shape == (1, 2, 1, 4, VOCAB + 1)
    assert not torch.allclose(base, withf, atol=1e-3)
    assert torch.allclose(base, padded, atol=1e-5)


def test_future_from_unknown_bounds():
    si = _seq_info(B=2, G=1, T=5, N=3)
    fut = future_from_unknown(si, torch.tensor([2, 4]))
    assert fut["future_features"].shape == (2, 1, 5, 4, 3, 32)
    m = fut["future_masks"]
    assert m[0, 0, :, 2:].all()                                    # sample 0: lag 2 -> steps 3, 4 are padding
    assert m[1, 0, 4].all() and m[1, 0, 3, 1:].all()               # past the window end
    assert torch.equal(fut["future_features"][1, 0, 1, 2], si["unknown_features"][1, 0, 4])
    assert torch.equal(m[1, 0, 1, 0], si["unknown_masks"][1, 0, 2])
    assert torch.equal(fut["future_times"][1, 0, 1], torch.tensor([2, 3, 4, 4]))
    assert future_from_unknown(si, torch.tensor([0, 0]))["future_features"].shape[3] == 0


def test_pair_features_and_block_identity():
    boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.2], [0.5, 0.5, 0.1, 0.2], [0.9, 0.5, 0.1, 0.4]]])
    steps = torch.tensor([[0, 1, 2]])
    pf = pair_features(boxes, steps, 2)
    assert pf.shape == (1, 3, 3, 5)
    assert torch.isclose(pf[0, 0, 1, 0], torch.tensor(1.0)) and pf[0, 0, 2, 0] == 0
    assert torch.isclose(pf[0, 0, 2, 3], torch.log(torch.tensor(2.0)))
    assert torch.isclose(pf[0, 0, 2, 4], torch.tensor(1.0))
    blk = LookaheadBlock(16, 2, 2)
    x = torch.randn(1, 3, 16)
    assert torch.equal(blk(x, steps, pf, torch.zeros(1, 3, dtype=torch.bool)), x)


def test_contamination_augmentation():
    random.seed(0)
    torch.manual_seed(0)
    T, N = 8, 3
    tt = TurnIntoTrajectoryAndUnknown(num_id_vocabulary=VOCAB, aug_trajectory_occlusion_prob=0.0,
                                      aug_trajectory_switch_prob=0.0, aug_trajectory_contaminate_prob=1.0,
                                      aug_trajectory_contaminate_max_len=4, aug_trajectory_contaminate_iou=0.1)
    # players 0 and 1 overlap in every frame, player 2 is far away
    bbox = torch.tensor([[0.5, 0.5, 0.1, 0.2], [0.52, 0.5, 0.1, 0.2], [0.1, 0.1, 0.05, 0.1]])
    hits = 0
    for trial in range(30):
        anns = [{"bbox": bbox.clone(), "id_labels": torch.tensor([[[3, 5, 7]]]).expand(2, 1, N).clone(),
                 "id_masks": torch.zeros((2, 1, N), dtype=torch.bool), "ann_idxs": torch.tensor([[[0, 1, 2]]]).expand(2, 1, N).clone(),
                 "times": torch.full((2, 1, N), t)} for t in range(T)]
        _, out, _ = tt(None, anns, None)
        ann = torch.cat([a["trajectory_ann_idxs"] for a in out], dim=1)       # (G, T, N)
        msk = torch.cat([a["trajectory_id_masks"] for a in out], dim=1)
        unk = torch.cat([a["unknown_ann_idxs"] for a in out], dim=1)
        assert torch.equal(unk, torch.tensor([0, 1, 2]).expand(2, T, N))     # queries untouched
        assert torch.equal(ann[:, 0], torch.tensor([0, 1, 2]).expand(2, N))   # never at a track's first frame
        assert (ann[:, :, 2] == 2).all() and not msk[:, :, 2].any()          # the far player never involved
        for g in range(2):
            changed = (ann[g] != torch.tensor([0, 1, 2])) | msk[g]
            rows = changed.any(dim=1).nonzero()[:, 0].tolist()
            if rows:
                hits += 1
                assert rows == list(range(rows[0], rows[0] + len(rows))) and len(rows) <= 4   # one contiguous run
                victim_rows = ann[g, rows]
                assert ((victim_rows[:, 0] == 1) | (victim_rows[:, 1] == 0)).all()
    assert hits == 60   # p = 1: every group of every trial contaminated


# ---------------------------------------------------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------------------------------------------------
class _Model(torch.nn.Module):
    """RuntimeTracker-facing stub around a real (256-d feature) IDDecoder; no detector."""
    num_classes = 1

    def __init__(self, dec):
        super().__init__()
        self.id_decoder = dec
        self.num_id_vocabulary = dec.num_id_vocabulary
        self.seen = []

    def forward(self, seq_info=None, part=None, **_):
        if part == "trajectory_modeling":
            return seq_info
        if "future_steps" in seq_info:
            self.seen.append((~seq_info["future_masks"]).any(dim=-1)[0, 0, 0].tolist())
        else:
            self.seen.append(None)
        return self.id_decoder(seq_info, use_decoder_checkpoint=False)


def _frames(n_frames=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn((3, 256), generator=g)
    out = []
    for t in range(n_frames):
        n = 3 if t % 5 else 2
        boxes = torch.tensor([[0.2 + 0.01 * t, 0.5, 0.05, 0.1], [0.5, 0.5, 0.05, 0.1], [0.8 - 0.01 * t, 0.5, 0.05, 0.1]])[:n]
        emb = protos[:n] + 0.1 * torch.randn((n, 256), generator=g)
        out.append((torch.full((n,), 0.9), torch.zeros(n, dtype=torch.long), boxes, emb))
    return out


def _tracker(model):
    return RuntimeTracker(model=model, sequence_hw=(1000, 1000), miss_tolerance=30, det_thresh=0.3, newborn_thresh=0.5,
                          id_thresh=0.1)


def _run(wrapper_or_tracker, frames, dev):
    res = {}
    for f in frames:
        args = [x.to(dev) for x in f]
        if isinstance(wrapper_or_tracker, LookaheadTracker):
            for t, tr in wrapper_or_tracker.update_from_detections(*args):
                res[t] = tr["id"].tolist()
        else:
            wrapper_or_tracker.update_from_detections(*args)
            res[len(res)] = wrapper_or_tracker.get_track_results()["id"].tolist()
    if isinstance(wrapper_or_tracker, LookaheadTracker):
        for t, tr in wrapper_or_tracker.finish():
            res[t] = tr["id"].tolist()
    return res


def _la_model(randomise=False, max_steps=4):
    dec = _decoder(lookahead=dict(max_steps=max_steps), feature_dim=256, id_dim=32, layers=2, rel_pe=40)
    if randomise:
        _randomise_lookahead(dec)
    return _Model(dec).to(distributed_device()).eval()


def test_lookahead_tracker_identity_model_matches_online_and_delays():
    dev = distributed_device()
    frames = _frames()
    online = _run(_tracker(_la_model()), frames, dev)
    lt = LookaheadTracker(_tracker(_la_model()), lag=8, frame_stride=2)
    emitted = []
    for f in frames:
        emitted.append([t for t, _ in lt.update_from_detections(*[x.to(dev) for x in f])])
    tail = [t for t, _ in lt.finish()]
    assert emitted[:8] == [[]] * 8 and emitted[8] == [0] and emitted[-1] == [len(frames) - 9]
    assert tail == list(range(len(frames) - 8, len(frames)))
    lt2 = LookaheadTracker(_tracker(_la_model()), lag=8, frame_stride=2)
    assert _run(lt2, frames, dev) == online     # fresh (identity) look-ahead: same decisions, only delayed
    assert lt2.stats["frames"] == len(frames) and lt2.stats["frames_with_future"] > 0


def test_lookahead_tracker_future_frames_and_steps():
    dev = distributed_device()
    m = _la_model(randomise=True)
    lt = LookaheadTracker(_tracker(m), lag=8, frame_stride=2)
    _run(lt, _frames(n_frames=12), dev)
    seen = [s for s in m.seen]
    # frame 0 has no history (no decoder call); frames 1..3 see all 4 future steps (t+2..t+8), the tail fewer
    assert seen[0] == [True, True, True, True]
    assert seen[-1] is None or not any(seen[-1])          # the last frame has no future
    n_true = [sum(s) for s in seen if s is not None]
    assert n_true[:3] == [4, 4, 4] and n_true[-1] <= 1
    assert lt.stats["steps"] == 4


def test_lookahead_tracker_trigger_mode():
    dev = distributed_device()
    frames = _frames()
    never = LookaheadTracker(_tracker(_la_model(randomise=True)), lag=8, frame_stride=2, trigger="conf",
                             trigger_p=0.0, trigger_iou=2.0, trigger_react=10_000)
    _run(never, frames, dev)
    assert never.stats["triggered"] == 0 and never.stats["frames_with_future"] == 0
    always = LookaheadTracker(_tracker(_la_model(randomise=True)), lag=8, frame_stride=2, trigger="conf", trigger_p=1.01)
    _run(always, frames, dev)
    assert always.stats["triggered"] == always.stats["frames_with_future"] > 0


def test_lookahead_tracker_needs_lookahead_model():
    plain = _Model(_decoder(feature_dim=256, id_dim=32, layers=2)).to(distributed_device()).eval()
    with pytest.raises(ValueError):
        LookaheadTracker(_tracker(plain), lag=4)


def test_motip_server_selects_lookahead():
    from motip_server import _fixed_lag
    trk = _tracker(_la_model())
    assert _fixed_lag({}, trk) is None
    w = _fixed_lag({"LOOKAHEAD_LAG": 12, "LOOKAHEAD_FRAME_STRIDE": 4}, trk)
    assert isinstance(w, LookaheadTracker) and w.steps == 3
    with pytest.raises(ValueError):
        _fixed_lag({"LOOKAHEAD_LAG": 12, "FIXED_LAG": 5}, trk)


def test_load_checkpoint_fills_lookahead(tmp_path):
    from models.misc import load_checkpoint

    class _W(torch.nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.detr = torch.nn.Linear(2, 2)
            self.id_decoder = dec

    plain = _W(_decoder())
    torch.save({"model": plain.state_dict()}, tmp_path / "c.pth")
    la = _W(_decoder(lookahead=dict(max_steps=3), seed=9))
    own = {k: v.clone() for k, v in la.state_dict().items() if ".lookahead." in k}
    load_checkpoint(la, str(tmp_path / "c.pth"))
    sd = la.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in own.items())
    assert torch.equal(sd["id_decoder.word_to_embed.weight"], plain.state_dict()["id_decoder.word_to_embed.weight"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 inference path needs CUDA")
def test_fp16_inference_matches_fp32():
    """Eval configs run the model in FP16 (INFERENCE_DTYPE); the look-ahead path must stay finite and close."""
    la = _decoder(lookahead=dict(max_steps=3), feature_dim=256, id_dim=32, layers=3, rel_pe=40).eval()
    _randomise_lookahead(la)
    _la_model_and_lt = _Model(la).cuda().eval()
    dev = torch.device("cuda")
    ref = _run(LookaheadTracker(_tracker(_la_model_and_lt), lag=6, frame_stride=2), _frames(), dev)
    half = _Model(_decoder(lookahead=dict(max_steps=3), feature_dim=256, id_dim=32, layers=3, rel_pe=40)).cuda().eval()
    half.id_decoder.load_state_dict(la.state_dict())
    trk = RuntimeTracker(model=half, sequence_hw=(1000, 1000), miss_tolerance=30, det_thresh=0.3, newborn_thresh=0.5,
                         id_thresh=0.1, dtype=torch.float16)
    frames = [(s, c, b.half(), e.half()) for s, c, b, e in _frames()]
    out = _run(LookaheadTracker(trk, lag=6, frame_stride=2), frames, dev)
    assert out == ref
