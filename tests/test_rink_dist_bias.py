"""Tests for the rink-distance attention bias of the ID decoder (cfg RINK_DIST_BIAS) and the rink anchor's
image-anchor-style output scale (cfg RINK_ANCHOR_OUT_SCALE_INIT).

Off by default. Zero-init: a decoder with the bias is exactly the decoder without it at step 0. A pair where either token
has no position gets exactly zero. Training (prepare_for_motip) and inference (RuntimeTracker) feed the decoder the same
feet positions, so the same bias and the same logits.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from models.motip.id_decoder import IDDecoder
from models.motip.motip import MOTIP
from models.motip.rink_anchor import RinkAnchor
from models.motip.rink_dist_bias import DEFAULT_KNOTS_M, RinkDistBias, bias_from_table, pair_interp
from models.motip.trajectory_modeling import TrajectoryModeling
from models.runtime_tracker import RuntimeTracker
from train import get_param_groups, prepare_for_motip
from utils.misc import distributed_device

K = 8                       # ID vocabulary
W0, H0_PX = 1280, 720
H0 = np.array([[0.05, 0.0, -32.0], [0.0, 0.05, -18.0], [0.0, 0.0, 1.0]])   # frame px -> metres
BOUNDS = (31.0, 16.0)


def _decoder(bias=True, dim=16, layers=3, seed=0, **bias_kw):
    torch.manual_seed(seed)
    return IDDecoder(feature_dim=dim, id_dim=dim, ffn_dim_ratio=2, num_layers=layers, head_dim=8, num_id_vocabulary=K,
                     rel_pe_length=10, use_aux_loss=False, use_shared_aux_head=True,
                     rink_dist_bias=(bias_kw if bias else None))


def _seq(dim=16, B=1, G=2, T=3, N=4, seed=0, xy="random"):
    g = torch.Generator().manual_seed(seed)
    s = {
        "trajectory_features": torch.randn(B, G, T, N, dim, generator=g),
        "unknown_features": torch.randn(B, G, T, N, dim, generator=g),
        "trajectory_id_labels": torch.randint(0, K, (B, G, T, N), generator=g),
        "unknown_id_labels": torch.randint(0, K + 1, (B, G, T, N), generator=g),
        "trajectory_times": torch.arange(T)[None, None, :, None].expand(B, G, T, N).contiguous(),
        "unknown_times": torch.arange(T)[None, None, :, None].expand(B, G, T, N).contiguous(),
        "trajectory_masks": torch.rand(B, G, T, N, generator=g) < 0.2,
        "unknown_masks": torch.rand(B, G, T, N, generator=g) < 0.2,
    }
    if xy == "random":
        s["trajectory_rink_xy"] = torch.rand(B, G, T, N, 2, generator=g) * 8 - 4
        s["unknown_rink_xy"] = torch.rand(B, G, T, N, 2, generator=g) * 8 - 4
        s["unknown_rink_xy"][0, 0, 1, 2] = float("nan")          # one token without a position
    elif xy == "nan":
        s["trajectory_rink_xy"] = torch.full((B, G, T, N, 2), float("nan"))
        s["unknown_rink_xy"] = torch.full((B, G, T, N, 2), float("nan"))
    return s


def _poke(dec, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        dec.rink_dist_bias.cross.copy_(torch.randn(dec.rink_dist_bias.cross.shape, generator=g) * scale)
        if dec.rink_dist_bias.self_ is not None:
            dec.rink_dist_bias.self_.copy_(torch.randn(dec.rink_dist_bias.self_.shape, generator=g) * scale)


# ------------------------------------------------------------------------------------------------ defaults / identity
def test_default_off_no_params():
    dec = _decoder(bias=False)
    assert dec.rink_dist_bias is None
    assert not any("rink_dist_bias" in n for n, _ in dec.named_parameters())


def test_building_the_bias_does_not_move_the_rng_nor_the_other_params():
    a, b = _decoder(bias=False), _decoder(bias=True)
    sa, sb = a.state_dict(), b.state_dict()
    for k, v in sa.items():
        assert torch.equal(v, sb[k]), k
    assert set(sb) - set(sa) == {"rink_dist_bias.cross", "rink_dist_bias.self_"}


@pytest.mark.parametrize("xy", ["random", "nan", "missing"])
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("ckpt", [False, True])
def test_zero_init_is_identity(xy, train, ckpt):
    a, b = _decoder(bias=False), _decoder(bias=True)
    a.train(train), b.train(train)
    s = _seq(xy=xy)
    la = a(dict(s), use_decoder_checkpoint=ckpt)[0]
    lb = b(dict(s), use_decoder_checkpoint=ckpt)[0]
    assert torch.allclose(la, lb, atol=1e-6, rtol=0), (la - lb).abs().max()


def test_nonzero_tables_change_logits_and_invalid_xy_makes_them_irrelevant():
    a, b = _decoder(bias=False), _decoder(bias=True)
    _poke(b)
    s = _seq()
    assert not torch.allclose(a(dict(s), False)[0], b(dict(s), False)[0], atol=1e-4)
    # all positions missing -> every pair has zero bias -> the plain decoder again, whatever the tables hold
    s = _seq(xy="nan")
    assert torch.allclose(a(dict(s), False)[0], b(dict(s), False)[0], atol=1e-6)


def test_gradients_reach_both_tables_from_zero_init():
    b = _decoder(bias=True)
    b.train()
    logits = b(_seq(), use_decoder_checkpoint=True)[0]
    logits.float().pow(2).sum().backward()
    assert b.rink_dist_bias.cross.grad is not None and b.rink_dist_bias.cross.grad.abs().sum() > 0
    assert b.rink_dist_bias.self_.grad is not None and b.rink_dist_bias.self_.grad.abs().sum() > 0


def test_self_attn_bias_can_be_switched_off():
    b = _decoder(bias=True, self_attn=False)
    assert b.rink_dist_bias.self_ is None
    _poke(b)
    b(_seq(), False)


# ------------------------------------------------------------------------------------------------ interpolation / layout
def test_interpolation_hand_values_and_clamping():
    knots = torch.tensor(DEFAULT_KNOTS_M)
    q = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [float("nan"), 0.0]]])
    k = torch.tensor([[[1.25, 0.0], [0.0, 2.5], [100.0, 0.0]]])
    lo, frac, valid, dist = pair_interp(q, k, knots)
    table = torch.arange(len(DEFAULT_KNOTS_M), dtype=torch.float32)[:, None] * torch.tensor([1.0, 10.0])  # (K, 2 heads)
    bias = bias_from_table(table, lo, frac, valid).reshape(1, 2, 4, 3)           # (b h) l1 l2 -> b h l1 l2
    i125 = 2 + 0.5            # 1.25 m is half way between knots 1.0 (index 2) and 1.5 (index 3)
    i25 = 4 + 0.5             # 2.5 m between 2.0 (4) and 3.0 (5)
    last = len(DEFAULT_KNOTS_M) - 1
    for h, m in enumerate((1.0, 10.0)):
        assert bias[0, h, 0, 0].item() == pytest.approx(i125 * m)
        assert bias[0, h, 0, 1].item() == pytest.approx(i25 * m)
        assert bias[0, h, 0, 2].item() == pytest.approx(last * m)                 # beyond 30 m: last value
        assert torch.all(bias[0, h, 3] == 0)                                       # invalid query: exactly 0
    assert dist[0, 0, 0].item() == pytest.approx(1.25) and not valid[0, 3].any()


def test_bias_layout_is_batch_major_head_minor():
    rdb = RinkDistBias(num_layers=2, n_heads=3)
    _g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        rdb.cross.copy_(torch.randn(rdb.cross.shape, generator=_g))
    q, k = torch.rand(2, 5, 2) * 10, torch.rand(2, 4, 2) * 10
    interp = rdb.interp(q, k)
    out = rdb.cross_bias(1, interp)
    assert out.shape == (2 * 3, 5, 4)
    d = (q[:, :, None] - k[:, None]).norm(dim=-1)
    kn = torch.tensor(DEFAULT_KNOTS_M)
    for b in range(2):
        for h in range(3):
            want = torch.from_numpy(np.interp(d[b].numpy(), kn.numpy(), rdb.cross[1, :, h].detach().numpy())).float()
            assert torch.allclose(out[b * 3 + h], want, atol=1e-5)


def test_bias_depends_on_distance_only_not_on_flip():
    rdb = RinkDistBias(num_layers=1, n_heads=2)
    _poke_t = torch.randn(rdb.cross.shape)
    with torch.no_grad():
        rdb.cross.copy_(_poke_t)
    q, k = torch.rand(1, 6, 2) * 20 - 10, torch.rand(1, 7, 2) * 20 - 10
    flip = torch.tensor([-1.0, 1.0])
    assert torch.allclose(rdb.cross_bias(0, rdb.interp(q, k)), rdb.cross_bias(0, rdb.interp(q * flip, k * flip)), atol=1e-6)


# ------------------------------------------------------------------------------------------------ rink anchor out_scale
def test_rink_anchor_out_scale_zero_is_identity_and_gets_a_gradient():
    torch.manual_seed(0)
    ra = RinkAnchor(dim=16, out_scale_init=0.0)
    assert ra.mlp[-1].weight.abs().sum() > 0                 # non-zero MLP (the image-anchor recipe)
    xy = torch.rand(5, 2) * 20 - 10
    xy[0] = float("nan")
    out = ra(xy)
    assert torch.equal(out, torch.zeros_like(out))
    out.mul(torch.randn_like(out)).sum().backward()
    assert ra.out_scale.grad is not None and ra.out_scale.grad.abs().sum() > 0
    # default unchanged: zero last layer, no out_scale
    rd = RinkAnchor(dim=16)
    assert rd.out_scale is None and rd.mlp[-1].weight.abs().sum() == 0


def test_param_groups_for_new_params():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.id_decoder = _decoder(bias=True)
            self.rink_anchor = RinkAnchor(dim=16, out_scale_init=0.0)
            self.other = nn.Linear(2, 2)
    cfg = dict(DETR_FRAMEWORK="rf_detr", LR=1e-4, WEIGHT_DECAY=5e-4, LR_LINEAR_PROJ_NAMES=[], LR_LINEAR_PROJ_SCALE=1.0,
               LR_DICTIONARY_NAMES=[], LR_DICTIONARY_SCALE=1.0, LR_RINK_ANCHOR_SCALE=20.0, LR_RINK_ANCHOR_OUT_SCALE=100.0,
               LR_RINK_DIST_BIAS_SCALE=50.0)
    m = _M()
    groups = get_param_groups(m, cfg)
    by_id = {id(g["params"]): g for g in groups}
    named = dict(m.named_parameters())
    g = by_id[id(named["rink_anchor.out_scale"])]
    assert g["lr"] == pytest.approx(1e-2) and g["weight_decay"] == 0.0
    g = by_id[id(named["rink_anchor.mlp.0.weight"])]
    assert g["lr"] == pytest.approx(2e-3) and "weight_decay" not in g
    for n in ("id_decoder.rink_dist_bias.cross", "id_decoder.rink_dist_bias.self_"):
        g = by_id[id(named[n])]
        assert g["lr"] == pytest.approx(5e-3) and g["weight_decay"] == 0.0
    # every parameter is in exactly one group
    assert sorted(id(g["params"]) for g in groups) == sorted(id(p) for p in m.parameters())


# ------------------------------------------------------------------------------------------------ train / inference parity
class _StubDetr(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(1, 1)


def _motip(dim=256, anchor=False, bias=True):
    torch.manual_seed(0)
    dec = IDDecoder(feature_dim=dim, id_dim=dim, ffn_dim_ratio=1, num_layers=2, head_dim=64, num_id_vocabulary=K,
                    rel_pe_length=10, use_aux_loss=False, use_shared_aux_head=True, rink_dist_bias={} if bias else None)
    m = MOTIP(detr=_StubDetr(), detr_framework="rf_detr", only_detr=False,
              trajectory_modeling=TrajectoryModeling(detr_dim=dim, ffn_dim_ratio=1, feature_dim=dim),
              id_decoder=dec, detr_hidden_dim=dim, rink_anchor=anchor,
              rink_anchor_kwargs={"out_scale_init": 0.0} if anchor else None)
    if bias:
        _poke(dec, scale=2.0)
    if anchor:
        with torch.no_grad():
            m.rink_anchor.out_scale.fill_(0.3)
    m.num_classes = 1
    return m.to(distributed_device()).eval()


def _hn():
    return torch.as_tensor(H0 @ np.diag([W0, H0_PX, 1.0]), dtype=torch.float64)


def test_runtime_tracker_keeps_xy_by_itself_for_a_bias_model():
    """The bias works without the anchor: a tracker built without rink_anchor=True still keeps the XY history when the
    model has the bias, and a plain model does not."""
    assert RuntimeTracker(model=_motip(anchor=False, bias=True), sequence_hw=(H0_PX, W0)).rink_anchor
    assert not RuntimeTracker(model=_motip(anchor=False, bias=False), sequence_hw=(H0_PX, W0)).rink_anchor


@pytest.mark.parametrize("anchor", [False, True])
def test_train_and_inference_give_the_same_xy_bias_and_logits(anchor):
    """Frame 0 -> trajectory, frame 1 (camera panned) -> unknown. The RuntimeTracker's decoder call and the training
    path (prepare_for_motip with the per-frame matrices) see the same feet XY, hence the same bias and logits; with the
    bias alone (v1x_rinkdist, the tracker not told about the rink at all) and with the XY anchor as well."""
    dev = distributed_device()
    model = _motip(anchor=anchor)
    D = 256
    g = torch.Generator().manual_seed(3)
    boxes0 = torch.tensor([[0.20, 0.50, 0.05, 0.10], [0.22, 0.52, 0.05, 0.10], [0.60, 0.40, 0.05, 0.12]])
    boxes1 = torch.tensor([[0.23, 0.51, 0.05, 0.10], [0.21, 0.50, 0.05, 0.10], [0.61, 0.41, 0.05, 0.12]])
    emb0, emb1 = torch.randn(3, D, generator=g), torch.randn(3, D, generator=g)
    pan = H0.copy()
    pan[0, 2] -= 6.0

    # --- inference
    seen = {}
    orig = model.id_decoder.forward

    def rec(seq_info, use_decoder_checkpoint=False):
        out = orig(seq_info, use_decoder_checkpoint)
        seen["seq_info"], seen["logits"] = seq_info, out[0]
        return out
    model.id_decoder.forward = rec
    tr = RuntimeTracker(model=model, sequence_hw=(H0_PX, W0), rink_anchor=anchor, rink_bounds_m=BOUNDS,
                        det_thresh=0.1, newborn_thresh=0.1, id_thresh=0.0)
    n = 3
    sc, cat = torch.full((n,), 0.9, device=dev), torch.zeros(n, dtype=torch.int64, device=dev)
    with torch.no_grad():                      # as in inference
        tr._process_detections(sc, cat, boxes0.to(dev), emb0.to(dev), rink_H=H0)
    labels0 = tr.trajectory_id_labels[0].cpu()
    order = [int(torch.nonzero((tr.trajectory_boxes[0].cpu() - b).abs().sum(-1) < 1e-6)[0]) for b in boxes0]
    with torch.no_grad():
        tr._process_detections(sc, cat, boxes1.to(dev), emb1.to(dev), rink_H=pan)
    inf_logits = seen["logits"][0, 0, 0].cpu()                       # (3 unknown, K + 1)
    inf_seq = seen["seq_info"]

    # --- training path: same tokens, labels as the tracker gave them
    B, T, G, N, Q = 1, 2, 1, 3, 5
    pred_boxes = torch.zeros(B * T, Q, 4) + 0.1
    outputs = torch.zeros(B * T, Q, D)
    q_of = [4, 0, 2]
    for j in range(N):
        pred_boxes[0, q_of[j]], outputs[0, q_of[j]] = boxes0[j], emb0[j]
        pred_boxes[1, q_of[j]], outputs[1, q_of[j]] = boxes1[j], emb1[j]
    indices = [(torch.tensor(q_of), torch.arange(N)) for _ in range(B * T)]
    traj_labels = torch.stack([labels0[order[j]] for j in range(N)])
    anns = [[{
        "bbox": torch.zeros(N, 4),
        "trajectory_id_labels": traj_labels[None, None].clone(), "trajectory_id_masks": torch.zeros(G, 1, N, dtype=torch.bool),
        "trajectory_ann_idxs": torch.arange(N)[None, None], "trajectory_times": torch.full((G, 1, N), t),
        "unknown_id_labels": traj_labels[None, None].clone(), "unknown_id_masks": torch.zeros(G, 1, N, dtype=torch.bool),
        "unknown_ann_idxs": torch.arange(N)[None, None], "unknown_times": torch.full((G, 1, N), t),
    } for t in range(T)]]
    anns = [[{k: v.to(dev) for k, v in a.items()} for a in clip] for clip in anns]
    indices = [(a.to(dev), b.to(dev)) for a, b in indices]
    _hpan = torch.as_tensor(pan @ np.diag([W0, H0_PX, 1.0]), dtype=torch.float64)
    rink = {"H": torch.stack([_hn(), _hpan])[None], "flip": torch.zeros(1, T, dtype=torch.bool),
            "bounds": torch.tensor(BOUNDS, dtype=torch.float64).expand(1, T, 2).clone()}
    s = prepare_for_motip({"pred_logits": torch.zeros(B * T, Q, 2, device=dev), "outputs": outputs.to(dev), "pred_boxes": pred_boxes.to(dev)},
                          anns, indices, rink=rink)

    # XY parity, token by token (tracker's trajectory order -> detection order)
    t_inf = inf_seq["trajectory_rink_xy"][0, 0, 0].cpu()[order]
    u_inf = inf_seq["unknown_rink_xy"][0, 0, 0].cpu()
    assert torch.isfinite(t_inf).all() and torch.isfinite(u_inf).all()
    assert torch.allclose(t_inf, s["trajectory_rink_xy"][0, 0, 0].cpu(), atol=1e-3)
    assert torch.allclose(u_inf, s["unknown_rink_xy"][0, 0, 1].cpu(), atol=1e-3)
    # bias parity
    rdb = model.id_decoder.rink_dist_bias
    b_inf = rdb.cross_bias(1, rdb.interp(u_inf[None], t_inf[None]))
    b_tr = rdb.cross_bias(1, rdb.interp(s["unknown_rink_xy"][0, 0, 1][None].cpu(), s["trajectory_rink_xy"][0, 0, 0][None].cpu()))
    assert b_inf.abs().max() > 0.1 and torch.allclose(b_inf, b_tr, atol=1e-3)
    # logits parity: the unknowns of frame 1 attend the trajectory of frame 0 only
    model.id_decoder.forward = orig
    with torch.no_grad():
        s2 = model(seq_info=s, part="trajectory_modeling")
        tr_logits = model(seq_info=s2, part="id_decoder")[0][0, 0, 1].cpu()
    assert torch.allclose(inf_logits, tr_logits, atol=1e-4), (inf_logits - tr_logits).abs().max()
    # and the bias matters for these logits (so the parity is not vacuous)
    with torch.no_grad():
        rdb.cross.zero_()
        rdb.self_.zero_()
        tr0 = model(seq_info=model(seq_info=s, part="trajectory_modeling"), part="id_decoder")[0][0, 0, 1].cpu()
    assert (tr0 - tr_logits).abs().max() > 1e-3
