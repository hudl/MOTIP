"""Tests for the team cue (TEAM_CUE): each ID-decoder token gets a vector computed from its kit-appearance descriptor.

Off by default. The descriptor (utils/kit_feature.py) is NaN where there is none; those tokens get a learned "missing" vector.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.motip.motip import MOTIP  # noqa: E402
from models.motip.team_cue import TeamCue  # noqa: E402
from models.runtime_tracker import RuntimeTracker  # noqa: E402
from train import get_param_groups, prepare_for_motip  # noqa: E402
from utils.detr_cache import DetrCache, SeqCacheWriter, seq_cache_path  # noqa: E402
from utils.kit_feature import KIT_DIM, kit_descriptor, kit_for_norm_boxes  # noqa: E402
from utils.misc import distributed_device  # noqa: E402

D = 16
K = 8


class _StubIDDecoder(nn.Module):
    num_id_vocabulary = K
    feature_dim = D

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, seq_info, use_decoder_checkpoint=False):
        self.seen = seq_info
        return "logits"


class _StubDetr(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(1, 1)


def _motip(team=True, **kw):
    kw = {"in_dim": KIT_DIM, "hidden": 8, **kw}
    return MOTIP(detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
                 id_decoder=_StubIDDecoder(), detr_hidden_dim=D, team_cue=team, team_cue_kwargs=kw)


def _seq_info(with_kit=True, seed=0, masks=False):
    g = torch.Generator().manual_seed(seed)
    out = {
        "trajectory_features": torch.randn(1, 1, 3, 4, D, generator=g),
        "unknown_features": torch.randn(1, 1, 3, 2, D, generator=g),
        "trajectory_boxes": torch.rand(1, 1, 3, 4, 4, generator=g),
        "unknown_boxes": torch.rand(1, 1, 3, 2, 4, generator=g),
        "trajectory_masks": torch.zeros(1, 1, 3, 4, dtype=torch.bool),
        "unknown_masks": torch.zeros(1, 1, 3, 2, dtype=torch.bool),
    }
    if with_kit:
        out["trajectory_kit"] = torch.rand(1, 1, 3, 4, KIT_DIM, generator=g)
        out["unknown_kit"] = torch.rand(1, 1, 3, 2, KIT_DIM, generator=g)
    return out


# --------------------------------------------------------------------------------------------------
# defaults are unchanged
# --------------------------------------------------------------------------------------------------
def test_defaults_off():
    m = _motip(team=False)
    assert m.team_cue is None
    assert not any(n.startswith("team_cue") for n, _ in m.named_parameters())
    assert not any(n.startswith("team_cue") for n in m.state_dict())
    seq_info = _seq_info()
    m(seq_info=seq_info, part="id_decoder")
    assert m.id_decoder.seen is seq_info                # untouched: same object, no copy, no extra term
    assert m.record_cue_stats is False and m._cue_stats == {}


def test_building_the_cue_does_not_move_the_global_rng():
    torch.manual_seed(123)
    _motip(team=False)
    a = torch.rand(3)
    torch.manual_seed(123)
    _motip(team=True)
    assert torch.equal(a, torch.rand(3))


def test_prepare_for_motip_without_kit_has_no_kit_keys():
    seq_info = _prepare(kit=False)
    assert "trajectory_kit" not in seq_info and "unknown_kit" not in seq_info


def test_runtime_tracker_default_has_no_kit_inputs():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(100, 200))
    assert tr.team_cue is False
    _prime(tr)
    tr._get_id_pred_labels(boxes=torch.rand(2, 4, device=distributed_device()),
                           output_embeds=torch.zeros(2, 256, device=distributed_device()))
    assert "trajectory_kit" not in model.seen and "unknown_kit" not in model.seen


def test_cache_default_returns_no_kit(tmp_path):
    _write_cache(tmp_path, kit=True)
    out, _ = DetrCache(tmp_path).detr_outputs(*_batch(), device="cpu")
    assert "kit" not in out
    out, _ = DetrCache(tmp_path, with_kit=True).detr_outputs(*_batch(), device="cpu")
    assert "kit" in out


class _GroupModel(nn.Module):
    def __init__(self, cue):
        super().__init__()
        self.detr = nn.Linear(2, 2)
        self.trajectory_modeling = nn.Linear(2, 2)
        self.id_decoder = nn.Linear(2, 2)
        if cue:
            self.team_cue = TeamCue(in_dim=KIT_DIM, dim=8, hidden=8)


def _group_cfg(**kw):
    return dict(DETR_FRAMEWORK="rf_detr", LR=2e-5, WEIGHT_DECAY=1e-4, LR_LINEAR_PROJ_NAMES=["reference_points"],
                LR_LINEAR_PROJ_SCALE=0.1, LR_DICTIONARY_NAMES=["word_to_embed"], LR_DICTIONARY_SCALE=1.0, **kw)


def test_param_groups_default_unchanged_and_cue_gets_its_own_lr():
    model = _GroupModel(cue=False)
    names = {id(p): n for n, p in model.named_parameters()}
    assert not any(names[id(g["params"])].startswith("team_cue") for g in get_param_groups(model, _group_cfg()))

    model = _GroupModel(cue=True)
    for p in model.detr.parameters():
        p.requires_grad = False
    groups = get_param_groups(model, _group_cfg(LR_TEAM_CUE_SCALE=25.0))
    names = {id(p): n for n, p in model.named_parameters()}
    cue = {names[id(g["params"])]: g for g in groups if names[id(g["params"])].startswith("team_cue.")}
    assert set(cue) == {"team_cue.missing", "team_cue.gate", "team_cue.mlp.0.weight", "team_cue.mlp.0.bias",
                        "team_cue.mlp.2.weight", "team_cue.mlp.2.bias"}
    for n, g in cue.items():
        assert g["lr"] == pytest.approx(2e-5 * 25.0) and g["lr_scale"] == 25.0
        if n.endswith(("bias", "missing", "gate")):
            assert g["weight_decay"] == 0.0
        else:
            assert "weight_decay" not in g
    all_ids = [id(g["params"]) for g in groups]
    assert len(all_ids) == len(set(all_ids))
    assert names[all_ids[-1]].startswith(("trajectory_modeling", "id_decoder"))     # the LR-logging group stays the remainder
    assert {names[i] for i in all_ids} == {n for n, p in model.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------------------------------
# the module
# --------------------------------------------------------------------------------------------------
def test_shapes_and_added_term():
    m = _motip()
    m.eval()                                        # no dropout: the two forwards below must agree
    si = _seq_info()
    out = m.add_team_cue(si)
    assert out is not si and "trajectory_features" in si
    for k in ("trajectory", "unknown"):
        assert out[f"{k}_features"].shape == si[f"{k}_features"].shape
        assert not torch.equal(out[f"{k}_features"], si[f"{k}_features"])
    assert not torch.equal(si["unknown_features"], out["unknown_features"])          # the input dict is not modified in place
    # id_decoder part applies it
    m(seq_info=si, part="id_decoder")
    assert torch.equal(m.id_decoder.seen["unknown_features"], out["unknown_features"])


def test_starts_at_a_meaningful_scale_not_zero():
    """The old anchors started at 0 and never came in: the added term has RMS ~ gate_init per element from step 0."""
    m = _motip(gate_init=0.25, dropout=0.0)
    m.eval()
    cue, used = m.team_cue(torch.rand(2000, KIT_DIM))
    assert used.all()
    rms = cue.pow(2).mean().sqrt().item()
    assert 0.2 < rms < 0.3, rms
    assert float(m.team_cue.gate.detach()) == pytest.approx(0.25)


def test_missing_descriptor_uses_the_missing_vector():
    m = _motip(dropout=0.0)
    with torch.no_grad():
        m.team_cue.missing.copy_(torch.arange(D, dtype=torch.float32))
    kit = torch.rand(5, KIT_DIM)
    kit[1] = float("nan")
    kit[3, 4] = float("nan")                       # one NaN dimension is enough to make the whole descriptor unusable
    m.eval()
    cue, used = m.team_cue(kit)
    assert used.tolist() == [True, False, True, False, True]
    for i in (1, 3):
        assert torch.allclose(cue[i], float(m.team_cue.gate) * torch.arange(D, dtype=torch.float32))
    assert not torch.isnan(cue).any()
    # the zero-initialised missing vector adds nothing: tokens without information are the old behaviour
    assert torch.equal(_motip().team_cue.missing.detach(), torch.zeros(D))
    # a missing key means every token is missing (a caller without kit data still runs)
    out = m.add_team_cue(_seq_info(with_kit=False))
    assert not torch.isnan(out["unknown_features"]).any()


def test_dropout_only_in_train_mode():
    torch.manual_seed(0)
    m = _motip(dropout=0.5)
    kit = torch.rand(4000, KIT_DIM)
    m.train()
    _, used = m.team_cue(kit)
    assert 0.4 < used.float().mean().item() < 0.6
    m.eval()
    _, used = m.team_cue(kit)
    assert used.all()
    a, _ = m.team_cue(kit)
    b, _ = m.team_cue(kit)
    assert torch.equal(a, b)                       # deterministic in eval
    m0 = _motip(dropout=0.0)
    m0.train()
    assert m0.team_cue(kit)[1].all()


def test_noise_only_in_train_mode():
    m = _motip(dropout=0.0, noise_std=0.5)
    kit = torch.rand(64, KIT_DIM)
    m.train()
    assert not torch.equal(m.team_cue(kit)[0], m.team_cue(kit)[0])
    m.eval()
    assert torch.equal(m.team_cue(kit)[0], m.team_cue(kit)[0])


def test_same_descriptor_same_cue_regardless_of_batch_layout():
    m = _motip(dropout=0.0)
    m.eval()
    kit = torch.rand(6, KIT_DIM)
    a, _ = m.team_cue(kit)
    b, _ = m.team_cue(kit.flip(0))
    assert torch.allclose(a, b.flip(0), atol=1e-6)
    assert not torch.allclose(a[0], a[1])           # different kits, different cues


def test_set_stats_standardises():
    m = _motip(dropout=0.0)
    m.eval()
    kit = torch.rand(10, KIT_DIM)
    before, _ = m.team_cue(kit)
    m.team_cue.set_stats(np.full(KIT_DIM, 0.5), np.full(KIT_DIM, 0.1))
    after, _ = m.team_cue(kit)
    assert not torch.allclose(before, after)
    assert torch.allclose(m.team_cue.kit_mean, torch.full((KIT_DIM,), 0.5))
    assert "team_cue.kit_mean" in m.state_dict() and "team_cue.kit_std" in m.state_dict()      # they travel with the checkpoint
    with pytest.raises(AssertionError):
        m.team_cue.set_stats(np.zeros(3), np.ones(3))


def test_gradients_reach_every_parameter():
    m = _motip(dropout=0.3)
    m.train()
    kit = torch.rand(256, KIT_DIM)
    kit[:20] = float("nan")                         # so `missing` is used
    cue, _ = m.team_cue(kit)
    (cue * torch.randn_like(cue)).sum().backward()           # a linear read-out (pow(2) has zero gradient at the zero-init missing vector)
    for n, p in m.team_cue.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_requires_matching_dims():
    class _D(_StubIDDecoder):
        feature_dim = D + 1

    with pytest.raises(AssertionError, match="FEATURE_DIM"):
        MOTIP(detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
              id_decoder=_D(), detr_hidden_dim=D, team_cue=True, team_cue_kwargs={"in_dim": KIT_DIM})


# --------------------------------------------------------------------------------------------------
# contribution logging
# --------------------------------------------------------------------------------------------------
def test_cue_stats_are_recorded_only_when_asked_and_ignore_masked_tokens():
    m = _motip(gate_init=0.5, dropout=0.0)
    m.eval()
    m(seq_info=_seq_info(), part="id_decoder")
    assert m._cue_stats == {}                                       # off by default: no cost at inference
    m.record_cue_stats = True
    si = _seq_info()
    si["trajectory_masks"][0, 0, 0, :] = True                       # masked tokens are zero-feature padding in training
    si["trajectory_features"][0, 0, 0, :] = 0.0
    m(seq_info=si, part="id_decoder")
    st = m.take_cue_stats()
    assert st["kit_rms_ratio"] > 0 and st["kit_used_frac"] == 1.0
    assert st["kit_gate"] == pytest.approx(0.5)
    assert 0.3 < st["kit_added_rms"] < 0.7
    assert m.take_cue_stats().keys() == {"kit_gate", "kit_missing_norm"}      # cleared
    # NaN descriptors lower the used fraction
    si = _seq_info()
    si["unknown_kit"][...] = float("nan")
    m(seq_info=si, part="id_decoder")
    assert 0.0 < m.take_cue_stats()["kit_used_frac"] < 1.0


def test_box_anchor_scale_is_logged_with_the_cue_terms():
    m = _motip(team=False)
    m.anchor_scale = nn.Parameter(torch.full((D,), 0.2))
    m.record_cue_stats = True
    m.spatial_anchor = False
    m._record("box", torch.randn(1, 1, 3, 4, D) * 0.2, torch.randn(1, 1, 3, 4, D), torch.zeros(1, 1, 3, 4, dtype=torch.bool))
    st = m.take_cue_stats()
    assert 0.1 < st["box_rms_ratio"] < 0.4 and st["box_scale_mean"] == pytest.approx(0.2)


# --------------------------------------------------------------------------------------------------
# cache: kit descriptors stored per matched row, looked up by GT object id
# --------------------------------------------------------------------------------------------------
def _write_cache(tmp_path, seq="s1", n=3, kit=True):
    w = SeqCacheWriter(n)
    rng = np.random.default_rng(0)
    for t in range(n):
        ids = np.array([5, 2, 9][: 2 + (t % 2)])
        k = rng.uniform(size=(len(ids), KIT_DIM)) if kit else None
        if kit and t == 1:
            k[0] = np.nan
        w.add_frame(ids, rng.normal(size=(len(ids), 8)), rng.uniform(size=(len(ids), 4)), rng.normal(size=(len(ids), 3)), kit=k)
    w.save(seq_cache_path(tmp_path, "Hockey", "train", seq), {"view": "test"})


def _batch(seq="s1", ids=((5, 2), (2, 5, 9), (2, 5))):
    metas = [[{"dataset": "Hockey", "split": "train", "sequence": seq, "frame_idx": t} for t in range(3)]]
    anns = [[{"id": torch.tensor(i)} for i in ids]]
    return metas, anns


def test_cache_kit_rows_follow_the_annotation_id_order(tmp_path):
    _write_cache(tmp_path)
    c = DetrCache(tmp_path, with_kit=True)
    out, _ = c.detr_outputs(*_batch(), device="cpu")
    z = c.seq("Hockey", "train", "s1")
    assert out["kit"].shape == (3, 3, KIT_DIM)
    # frame 1 asks for ids (2, 5, 9); cached order is (5, 2, 9); row 0 of frame 1 (id 5) was written NaN
    ids1, _, _, _ = z.frame(1)
    assert ids1.tolist() == [5, 2, 9]
    assert torch.equal(out["kit"][1, 0], torch.from_numpy(z.frame_kit(1)[1]))
    assert torch.isnan(out["kit"][1, 1]).all()
    assert torch.isnan(out["kit"][0, 2]).all()                        # padding rows are NaN
    assert out["pred_boxes"].shape[:2] == out["kit"].shape[:2]


def test_cache_without_kit_raises_clearly_when_asked_for_it(tmp_path):
    _write_cache(tmp_path, kit=False)
    with pytest.raises(ValueError, match="no kit"):
        DetrCache(tmp_path, with_kit=True).detr_outputs(*_batch(), device="cpu")
    out, _ = DetrCache(tmp_path).detr_outputs(*_batch(), device="cpu")     # and the old cache still reads
    assert "kit" not in out


# --------------------------------------------------------------------------------------------------
# prepare_for_motip: the kit is gathered exactly like the embeddings
# --------------------------------------------------------------------------------------------------
def _prepare(kit=True):
    B, T, G, N, Q = 1, 2, 1, 2, 5
    pred_boxes = torch.zeros(B * T, Q, 4)
    pred_boxes[:, :, 2:] = 0.1
    outputs = {"pred_logits": torch.zeros(B * T, Q, 2), "outputs": torch.randn(B * T, Q, 4), "pred_boxes": pred_boxes}
    if kit:
        k = torch.full((B * T, Q, KIT_DIM), float("nan"))
        k[:, 3] = torch.arange(KIT_DIM, dtype=torch.float32)            # query 3 matches GT 0
        k[:, 1] = -torch.arange(KIT_DIM, dtype=torch.float32)           # query 1 matches GT 1
        outputs["kit"] = k
    indices = [(torch.tensor([3, 1]), torch.tensor([0, 1])) for _ in range(B * T)]
    anns = [[{
        "bbox": torch.zeros(N, 4),
        "trajectory_id_labels": torch.arange(N)[None, None].repeat(G, 1, 1), "trajectory_id_masks": torch.zeros(G, 1, N, dtype=torch.bool),
        "trajectory_ann_idxs": torch.arange(N)[None, None].repeat(G, 1, 1), "trajectory_times": torch.full((G, 1, N), t),
        "unknown_id_labels": torch.arange(N)[None, None].repeat(G, 1, 1), "unknown_id_masks": torch.zeros(G, 1, N, dtype=torch.bool),
        "unknown_ann_idxs": torch.arange(N).flip(0)[None, None].repeat(G, 1, 1), "unknown_times": torch.full((G, 1, N), t),
    } for t in range(T)]]
    return prepare_for_motip(outputs, anns, indices)


def test_prepare_for_motip_gathers_kit_like_the_features():
    si = _prepare()
    assert si["trajectory_kit"].shape == si["trajectory_boxes"].shape[:-1] + (KIT_DIM,)
    base = torch.arange(KIT_DIM, dtype=torch.float32)
    assert torch.equal(si["trajectory_kit"][0, 0, 0, 0], base) and torch.equal(si["trajectory_kit"][0, 0, 0, 1], -base)
    # the unknown tokens use the reversed annotation order here: slot 0 is GT 1
    assert torch.equal(si["unknown_kit"][0, 0, 1, 0], -base) and torch.equal(si["unknown_kit"][0, 0, 1, 1], base)
    # and the same rows the embeddings come from
    assert torch.equal(si["unknown_features"][0, 0, 0, 0], si["trajectory_features"][0, 0, 0, 1])


# --------------------------------------------------------------------------------------------------
# RuntimeTracker (inference): the history carries the kit descriptors
# --------------------------------------------------------------------------------------------------
class _Rec:
    num_id_vocabulary = K
    num_classes = 1
    seen = None
    team_cue = TeamCue(in_dim=KIT_DIM, dim=D, hidden=4)

    def eval(self):
        return self

    def __call__(self, seq_info=None, part=None, **_):
        if part == "id_decoder":
            self.seen = seq_info
            return torch.zeros(1, 1, 1, seq_info["unknown_boxes"].shape[-2], K + 1, device=distributed_device()), None, None
        return seq_info


def _prime(tr):
    dev = distributed_device()
    tr.trajectory_boxes = torch.rand(2, 3, 4, device=dev)
    tr.trajectory_masks = torch.zeros(2, 3, dtype=torch.bool, device=dev)
    tr.trajectory_id_labels = torch.tensor([[0, 1, 2]] * 2, device=dev)
    tr.trajectory_features = torch.zeros(2, 3, 256, device=dev)
    tr.trajectory_times = torch.arange(2, device=dev)[:, None].expand(2, 3).contiguous()
    tr.trajectory_kit = torch.full((2, 3, KIT_DIM), 0.5, device=dev)


def _det(xs):
    dev = distributed_device()
    boxes = torch.tensor([[x, 0.5, 0.05, 0.1] for x in xs], device=dev)
    n = len(xs)
    return (torch.full((n,), 0.9, device=dev), torch.zeros(n, dtype=torch.int64, device=dev), boxes, torch.zeros(n, 256, device=dev))


def test_runtime_tracker_keeps_the_kit_with_the_history():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(100, 200), team_cue=True)
    k0 = torch.stack([torch.full((KIT_DIM,), 0.1), torch.full((KIT_DIM,), 0.9)])
    tr._process_detections(*_det([0.2, 0.6]), kit=k0)                # frame 0: no history, nothing decoded
    assert tr.trajectory_kit.shape == (1, 2, KIT_DIM)
    got = {int(l): tr.trajectory_kit[0, i, 0].item() for i, l in enumerate(tr.trajectory_id_labels[0])}
    assert sorted(got.values()) == pytest.approx([0.1, 0.9])
    k1 = torch.stack([torch.full((KIT_DIM,), 0.2), torch.full((KIT_DIM,), 0.8)])
    tr._process_detections(*_det([0.2, 0.62]), kit=k1)
    seen = model.seen
    assert seen["trajectory_kit"].shape == (1, 1, 1, 2, KIT_DIM) and seen["unknown_kit"].shape == (1, 1, 1, 2, KIT_DIM)
    assert sorted(seen["trajectory_kit"][0, 0, 0, :, 0].cpu().tolist()) == pytest.approx([0.1, 0.9])     # history = frame 0's
    assert sorted(seen["unknown_kit"][0, 0, 0, :, 0].cpu().tolist()) == pytest.approx([0.2, 0.8])        # current = frame 1's


def test_runtime_tracker_without_kit_gives_nan_tokens_not_a_crash():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(100, 200), team_cue=True)
    tr._process_detections(*_det([0.2, 0.6]))
    tr._process_detections(*_det([0.2, 0.62]))
    assert torch.isnan(model.seen["unknown_kit"]).all() and torch.isnan(model.seen["trajectory_kit"]).all()


def test_runtime_tracker_kit_stays_aligned_through_suppression_and_newborns():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(100, 200), team_cue=True)
    k = torch.tensor([[0.1] * KIT_DIM, [0.5] * KIT_DIM, [0.9] * KIT_DIM])
    tr._process_detections(*_det([0.2, 0.5, 0.8]), kit=k)
    assert sorted(round(v, 3) for v in tr.trajectory_kit[0, :, 0].cpu().tolist()) == [0.1, 0.5, 0.9]
    # detection 1 (0.5) is a goalkeeper to suppress: the others keep their own rows
    gk = [[0.5 * 200 - 5, 0.5 * 100 - 5, 0.5 * 200 + 5, 0.5 * 100 + 5]]
    tr._process_detections(*_det([0.2, 0.5, 0.8]), suppress_boxes=gk, kit=k)
    seen = model.seen["unknown_kit"][0, 0, 0, :, 0].cpu().tolist()
    assert sorted(round(v, 3) for v in seen) == [0.1, 0.9]


def test_runtime_tracker_default_tokens_have_the_declared_kit_dim():
    tr = RuntimeTracker(model=_Rec(), sequence_hw=(100, 200), team_cue=True)
    assert tr._kit_dim == KIT_DIM and tr.trajectory_kit.shape[-1] == KIT_DIM


# --------------------------------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------------------------------
def test_load_checkpoint_fills_missing_team_cue_params(tmp_path, monkeypatch):
    import models.misc as misc
    monkeypatch.setattr(misc, "resize_rel_pos_embeds", lambda model, state: None)
    src = _motip(team=False)                           # a checkpoint trained without the cue (the production ckpt_12)
    path = tmp_path / "ckpt.pth"
    torch.save({"model": src.state_dict()}, path)
    dst = _motip(team=True)
    with torch.no_grad():
        dst.team_cue.gate.fill_(0.7)
        dst.team_cue.kit_mean.fill_(0.3)
    before = {k: v.clone() for k, v in dst.state_dict().items() if k.startswith("team_cue")}
    misc.load_checkpoint(dst, str(path))
    for k, v in before.items():
        assert torch.equal(dst.state_dict()[k], v)      # the load must not touch the init / the set stats
    full = _motip(team=True)                           # and a checkpoint that has them overrides
    with torch.no_grad():
        full.team_cue.gate.fill_(0.05)
    path2 = tmp_path / "ckpt2.pth"
    torch.save({"model": full.state_dict()}, path2)
    misc.load_checkpoint(dst, str(path2))
    assert float(dst.team_cue.gate) == pytest.approx(0.05)


# --------------------------------------------------------------------------------------------------
# the descriptor itself
# --------------------------------------------------------------------------------------------------
def _frame(h=200, w=100, torso=(0, 0, 255), legs=(255, 0, 0)):
    """BGR frame whose box (0,0,w,h) has a red torso (BGR (0,0,255)) and blue legs (BGR (255,0,0))."""
    f = np.full((h, w, 3), 128, np.uint8)
    f[int(0.18 * h):int(0.5 * h)] = torso
    f[int(0.55 * h):int(0.9 * h)] = legs
    return f


def test_kit_descriptor_reads_torso_and_legs_in_rgb_order():
    d = kit_descriptor(_frame(), (0, 0, 100, 200))
    assert d.shape == (KIT_DIM,) and np.isfinite(d).all()
    assert d[:3] == pytest.approx([1.0, 0.0, 0.0], abs=0.02)             # torso: R G B of a red jersey
    assert d[7:10] == pytest.approx([0.0, 0.0, 1.0], abs=0.02)           # legs: blue
    assert d[3] > 0.9 and d[10] > 0.9                                     # saturated
    assert d[5] == pytest.approx(np.cos(0.0) * d[3], abs=0.05)            # hue ~ 0 (red): cos ~ S, sin ~ 0
    assert abs(d[6]) < 0.05


def test_kit_descriptor_invalid_boxes_are_nan():
    f = _frame()
    for b in ((10, 10, 14, 60), (10, 10, 60, 20), (50, 50, 50, 50), (np.nan, 0, 10, 50), (-50, -50, -10, -10), (500, 500, 600, 600)):
        assert np.isnan(kit_descriptor(f, b)).all(), b


def test_kit_is_resolution_independent_and_box_clipping_is_safe():
    a = kit_descriptor(_frame(200, 100), (0, 0, 100, 200))
    import cv2
    big = cv2.resize(_frame(200, 100), (400, 800), interpolation=cv2.INTER_NEAREST)
    assert kit_descriptor(big, (0, 0, 400, 800)) == pytest.approx(a, abs=0.03)
    assert np.isfinite(kit_descriptor(_frame(), (-20, -10, 120, 230))).all()      # a box hanging over the frame edge is clipped


def test_kit_for_norm_boxes_matches_pixel_boxes():
    f = _frame(200, 100)
    got = kit_for_norm_boxes(f, np.array([[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 0.001, 0.001]]))
    assert np.allclose(got[0], kit_descriptor(f, (0, 0, 100, 200)), equal_nan=True)
    assert np.isnan(got[1]).all()


def test_cache_writer_accepts_empty_frames_with_kit(tmp_path):
    w = SeqCacheWriter(3)
    rng = np.random.default_rng(0)
    for n in (2, 0, 1):             # a frame with no GT objects: kit has shape (0, K)
        w.add_frame(np.arange(n), rng.normal(size=(n, 8)), rng.uniform(size=(n, 4)), rng.normal(size=(n, 3)), kit=rng.uniform(size=(n, KIT_DIM)))
    w.save(seq_cache_path(tmp_path, "Hockey", "train", "e"), {})
    c = DetrCache(tmp_path, with_kit=True)
    assert c.frame_kit("Hockey", "train", "e", 1, np.array([], dtype=np.int64)).shape == (0, KIT_DIM)
    assert c.frame_kit("Hockey", "train", "e", 0, np.array([1, 0])).shape == (2, KIT_DIM)
