"""Tests for the hard-negative identity loss on the detector's output embeddings (cfg ID_EMBED_LOSS, default off) and
its optional GT-team term (cfg ID_EMBED_TEAM, default off)."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motip.id_embed_loss import (
    IDEmbedCriterion, IDEmbedHead, gather_matched, hard_negative_mask, supcon_weighted, window_id_embed_loss,
)
from models.motip.motip import MOTIP


class _StubIDDecoder(nn.Module):
    num_id_vocabulary = 8
    feature_dim = 16


class _StubDetr(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(1, 1)


def _motip(head=True):
    return MOTIP(detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
                 id_decoder=_StubIDDecoder(), detr_hidden_dim=16,
                 id_embed_head=dict(in_dim=16, hidden=32, out_dim=8) if head else None)


# ------------------------------------------------------------------------------------------------ defaults
def test_default_off():
    m = _motip(head=False)
    assert m.id_embed_head is None and not any(n.startswith("id_embed_head") for n, _ in m.named_parameters())
    c = IDEmbedCriterion({})
    assert c.weight == 0.1 and c.temp == 0.1 and c.hard_weight == 4.0 and c.hard_iou == 0.1 and c.hard_rink_m == 1.5
    assert c.team is False


def test_building_the_head_does_not_move_the_global_rng():
    torch.manual_seed(5)
    _motip(head=False)
    a = torch.rand(4)
    torch.manual_seed(5)
    _motip(head=True)
    assert torch.equal(a, torch.rand(4))


def test_head_outputs_unit_vectors_and_routes_through_forward():
    m = _motip()
    z = m(part="id_embed", embeds=torch.randn(7, 16))
    assert z.shape == (7, 8) and torch.allclose(z.norm(dim=-1), torch.ones(7), atol=1e-5)


def test_load_checkpoint_fills_missing_head(tmp_path, monkeypatch):
    import models.misc as misc
    monkeypatch.setattr(misc, "resize_rel_pos_embeds", lambda model, state: None)
    path = tmp_path / "ckpt.pth"
    torch.save({"model": _motip(head=False).state_dict()}, path)
    dst = _motip(head=True)
    before = {k: v.clone() for k, v in dst.state_dict().items() if k.startswith("id_embed_head")}
    misc.load_checkpoint(dst, str(path))
    for k, v in before.items():
        assert torch.equal(dst.state_dict()[k], v)


# ------------------------------------------------------------------------------------------------ hard negatives
def test_hard_negative_mask_iou_rink_and_time():
    # tokens: (id, step, box cxcywh)
    ids = torch.tensor([1, 2, 3, 2, 4])
    frames = torch.tensor([0, 0, 0, 3, 0])
    boxes = torch.tensor([
        [0.50, 0.50, 0.10, 0.20],    # id 1
        [0.53, 0.50, 0.10, 0.20],    # id 2, overlaps id 1 (IoU ~0.54) in the same step
        [0.90, 0.10, 0.05, 0.10],    # id 3, far in the image ...
        [0.52, 0.50, 0.10, 0.20],    # id 2 three steps later: overlaps id 1's box but too far in time
        [0.10, 0.90, 0.05, 0.10],    # id 4, far
    ])
    xy = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.0], [float("nan"), 0.0]])   # id 3 is 1 m from id 1 on the rink
    m = hard_negative_mask(ids, frames, boxes, None, hard_iou=0.1, hard_rink_m=1.5, hard_frames=1)
    assert m[0, 1] and m[1, 0] and not m[0, 2] and not m[0, 3] and not m[0, 4]
    assert not m.diagonal().any()
    m = hard_negative_mask(ids, frames, boxes, xy, hard_iou=0.1, hard_rink_m=1.5, hard_frames=1)
    assert m[0, 2] and m[2, 0]                       # rink-near counts
    assert not m[0, 4]                               # NaN feet never count
    m = hard_negative_mask(ids, frames, boxes, None, hard_iou=0.1, hard_rink_m=1.5, hard_frames=3)
    assert m[0, 3]                                   # within 3 steps now
    # same identity is never a negative
    assert not hard_negative_mask(torch.tensor([7, 7]), torch.tensor([0, 1]), boxes[:2], None).any()


# ------------------------------------------------------------------------------------------------ the contrastive term
def _supcon_reference(z, ids, temp):
    """Textbook SupCon (Khosla et al., L_out) without weights."""
    s = z @ z.t() / temp
    M = len(ids)
    total, n = 0.0, 0
    for i in range(M):
        others = [j for j in range(M) if j != i]
        den = torch.logsumexp(torch.stack([s[i, j] for j in others]), 0)
        P = [j for j in others if ids[j] == ids[i]]
        if not P:
            continue
        total = total - sum(s[i, p] - den for p in P) / len(P)
        n += 1
    return total / n


def test_supcon_unit_weights_equals_reference():
    g = torch.Generator().manual_seed(0)
    z = F.normalize(torch.randn(10, 6, generator=g), dim=-1)
    ids = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 4, 5])
    pos = ids[:, None] == ids[None]
    loss, n = supcon_weighted(z, pos, torch.ones(10, 10), 0.1)
    assert n == 7
    assert loss.item() == pytest.approx(_supcon_reference(z, ids, 0.1).item(), rel=1e-5)


def test_no_positive_gives_connected_zero():
    z = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    loss, n = supcon_weighted(z, torch.zeros(3, 3, dtype=torch.bool), torch.ones(3, 3), 0.1)
    assert n == 0 and loss.item() == 0.0
    loss.backward()
    assert z.grad is not None


def test_hard_weight_raises_the_loss_when_hard_negatives_look_alike():
    # two identities, two frames; id 0 and id 1 have near-identical embeddings (a confusable crossing pair)
    base = F.normalize(torch.tensor([[1.0, 0.0], [0.95, 0.31], [1.0, 0.05], [0.94, 0.33]]), dim=-1)
    ids, frames = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])
    boxes = torch.tensor([[0.5, 0.5, 0.1, 0.2], [0.52, 0.5, 0.1, 0.2], [0.5, 0.5, 0.1, 0.2], [0.52, 0.5, 0.1, 0.2]])
    l1 = window_id_embed_loss(base, ids, frames, boxes, hard_weight=1.0)[0]["id_embed"]
    l4 = window_id_embed_loss(base, ids, frames, boxes, hard_weight=4.0)[0]["id_embed"]
    assert l4 > l1
    # far apart boxes: no hard pair, the weight is irrelevant
    far = boxes.clone()
    far[1::2, 0] = 0.9
    a = window_id_embed_loss(base, ids, frames, far, hard_weight=1.0)[0]["id_embed"]
    b = window_id_embed_loss(base, ids, frames, far, hard_weight=4.0)[0]["id_embed"]
    assert a.item() == pytest.approx(b.item())


def test_toy_batch_loss_decreases_and_separates_crossing_partners():
    """A toy window: 4 players over 6 steps, players 0 and 1 cross (overlapping boxes) and start with near-identical
    raw embeddings. Training the head (and the raw embeddings, standing in for the detector) on the loss lowers it and
    pulls same-player pairs above crossing-partner pairs."""
    torch.manual_seed(0)
    T, P, C = 6, 4, 16
    proto = torch.randn(P, C)
    proto[1] = proto[0] + 0.05 * torch.randn(C)
    raw = (proto[None].repeat(T, 1, 1) + 0.3 * torch.randn(T, P, C)).reshape(-1, C).requires_grad_()
    ids = torch.arange(P).repeat(T)
    frames = torch.arange(T).repeat_interleave(P)
    boxes = torch.tensor([[0.3, 0.5, 0.1, 0.2], [0.32, 0.5, 0.1, 0.2], [0.7, 0.3, 0.1, 0.2], [0.1, 0.8, 0.1, 0.2]]).repeat(T, 1)
    head = IDEmbedHead(C, 32, 8)
    opt = torch.optim.Adam(list(head.parameters()) + [raw], lr=1e-2)

    def run():
        out, st = window_id_embed_loss(head(raw), ids, frames, boxes, temp=0.1, hard_weight=4.0)
        return out["id_embed"], st
    first, st0 = run()
    assert st0["n_hard"] == T + 2 * (T - 1) and math.isfinite(first.item())   # (0, 1) pairs: same step + adjacent steps
    for _ in range(150):
        opt.zero_grad()
        loss, _ = run()
        loss.backward()
        opt.step()
    last, st = run()
    # SupCon floor with T - 1 positives per anchor is log(T - 1) (all positives equal, negatives at 0 weight)
    assert last < first - 0.5 and last.item() < math.log(T - 1) + 0.05
    assert st["pos_sim"] > st["hard_sim"] + 0.3


def test_gather_matched_and_gradient_reaches_only_matched_queries():
    B, T, Q, C = 2, 3, 6, 16
    emb = torch.randn(B * T, Q, C, requires_grad=True)
    out = {"outputs": emb}
    anns, idx = [], []
    for b in range(B):
        clip = []
        for t in range(T):
            n = 2 if b == 0 else 3
            clip.append({"id": torch.tensor([10, 11, 12][:n]), "bbox": torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.22, 0.5, 0.1, 0.2], [0.8, 0.5, 0.1, 0.2]][:n])})
            idx.append((torch.tensor([5, 1, 3][:n]), torch.arange(n)))
        anns.append(clip)
    recs = gather_matched(out, idx, anns)
    assert len(recs) == 2 and recs[0]["embeds"].shape == (6, C) and recs[1]["embeds"].shape == (9, C)
    assert recs[0]["ids"].tolist() == [10, 11] * 3 and recs[0]["frames"].tolist() == [0, 0, 1, 1, 2, 2]
    assert torch.equal(recs[1]["embeds"][2], emb[3][3])
    crit = IDEmbedCriterion({"ID_EMBED_LOSS_WEIGHT": 0.5})
    head = IDEmbedHead(C, 32, 8)
    terms, stats = crit(head, out, idx, anns)
    assert terms["total"].item() == pytest.approx(0.5 * terms["id_embed"].item())
    assert stats["id_embed_n_hard"] > 0
    terms["total"].backward()
    g = emb.grad.abs().sum(-1)                                  # (B*T, Q)
    matched = torch.zeros_like(g, dtype=torch.bool)
    for f, (q, _) in enumerate(idx):
        matched[f, q] = True
    assert (g[matched] > 0).all() and (g[~matched] == 0).all()


def test_gather_matched_projects_gt_feet_with_the_rink_batch():
    T, Q, C = 2, 3, 4
    out = {"outputs": torch.randn(T, Q, C)}
    anns = [[{"id": torch.tensor([1]), "bbox": torch.tensor([[0.5, 0.5, 0.1, 0.2]])} for _ in range(T)]]
    idx = [(torch.tensor([2]), torch.tensor([0])) for _ in range(T)]
    H = torch.as_tensor(np.array([[0.05, 0.0, -32.0], [0.0, 0.05, -18.0], [0.0, 0.0, 1.0]]) @ np.diag([1280.0, 720.0, 1.0]))
    rink = {"H": torch.stack([H, torch.full((3, 3), float("nan"), dtype=torch.float64)])[None],
            "flip": torch.tensor([[False, False]]), "bounds": torch.tensor([[[31.0, 16.0], [31.0, 16.0]]], dtype=torch.float64)}
    rec = gather_matched(out, idx, anns, rink=rink)[0]
    assert rec["rink_xy"][0].tolist() == pytest.approx([0.05 * 640 - 32, 0.05 * 432 - 18])
    assert torch.isnan(rec["rink_xy"][1]).all()                 # frame without a homography


# ------------------------------------------------------------------------------------------------ optional team term
def test_team_term_off_by_default_and_needs_labels_when_on():
    out = {"outputs": torch.randn(2, 3, 4)}
    anns = [[{"id": torch.tensor([1, 2]), "bbox": torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.7, 0.5, 0.1, 0.2]])} for _ in range(2)]]
    idx = [(torch.tensor([0, 1]), torch.arange(2)) for _ in range(2)]
    head = IDEmbedHead(4, 8, 4)
    terms, _ = IDEmbedCriterion({})(head, out, idx, anns)
    assert "id_embed_team" not in terms
    crit = IDEmbedCriterion({"ID_EMBED_TEAM": True, "ID_EMBED_TEAM_WEIGHT": 0.2})
    with pytest.raises(RuntimeError, match="gt_team"):
        crit(head, out, idx, anns, metas=[[{}, {}]])
    metas = [[{"gt_team": {1: 0, 2: 1}}, {"gt_team": {1: 0, 2: 1}}]]
    terms, _ = crit(head, out, idx, anns, metas=metas)
    assert terms["id_embed_team"].item() > 0
    assert terms["total"].item() == pytest.approx(0.1 * terms["id_embed"].item() + 0.2 * terms["id_embed_team"].item(), rel=1e-5)


def test_team_term_only_counts_windows_with_both_teams():
    z = F.normalize(torch.randn(4, 4), dim=-1)
    ids, frames = torch.tensor([1, 2, 1, 2]), torch.tensor([0, 0, 1, 1])
    boxes = torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.7, 0.5, 0.1, 0.2]] * 2)
    one_team = window_id_embed_loss(z, ids, frames, boxes, teams=torch.tensor([0, 0, 0, 0]))
    assert one_team[0]["id_embed_team"].item() == 0.0 and one_team[1]["n_team_anchor"] == 0
    both = window_id_embed_loss(z, ids, frames, boxes, teams=torch.tensor([0, 1, 0, 1]))
    assert both[1]["n_team_anchor"] == 4


def test_gt_team_loader(tmp_path):
    from data.rink import load_gt_team
    (tmp_path / "gt").mkdir()
    assert load_gt_team(str(tmp_path)) is None
    (tmp_path / "gt" / "gt_team.txt").write_text("1,0\n2,1\n5,-1\n")
    assert load_gt_team(str(tmp_path)) == {1: 0, 2: 1, 5: -1}


def test_rink_data_gate():
    from data.rink import needs_rink_data
    assert not needs_rink_data({})
    assert needs_rink_data({"RINK_ANCHOR": True}) and needs_rink_data({"RINK_DIST_BIAS": True})
    assert needs_rink_data({"ID_EMBED_LOSS": True})
    assert not needs_rink_data({"ID_EMBED_LOSS": True, "ID_EMBED_HARD_RINK_M": 0.0})


def test_head_runs_inside_the_id_decoder_forward():
    """Training calls the head in the same (last) DDP forward as the ID decoder, so DDP's unused-parameter search from
    that forward's outputs reaches the head."""
    class _Dec(nn.Module):
        num_id_vocabulary = 8
        feature_dim = 16

        def forward(self, seq_info, use_decoder_checkpoint=False):
            return "logits", "labels", "masks"
    m = MOTIP(detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None, id_decoder=_Dec(),
              detr_hidden_dim=16, id_embed_head=dict(in_dim=16, hidden=32, out_dim=8))
    x = torch.randn(5, 16)
    out = m(part="id_decoder", seq_info={}, id_embed_inputs=x)
    assert out[:3] == ("logits", "labels", "masks") and torch.allclose(out[3], m.id_embed_head(x))
    assert len(m(part="id_decoder", seq_info={})) == 3            # without inputs: unchanged
    # gather + loss on the flat head output == the one-call criterion
    crit = IDEmbedCriterion({})
    outs = {"outputs": torch.randn(4, 5, 16)}
    anns = [[{"id": torch.tensor([1, 2]), "bbox": torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.23, 0.5, 0.1, 0.2]])} for _ in range(2)]
            for _ in range(2)]
    idx = [(torch.tensor([0, 3]), torch.arange(2)) for _ in range(4)]
    recs, flat = crit.gather(outs, idx, anns)
    a = crit.loss(recs, m.id_embed_head(flat), "cpu")[0]["total"]
    b = crit(m.id_embed_head, outs, idx, anns)[0]["total"]
    assert torch.allclose(a, b)


def test_stats_have_the_same_keys_whatever_the_batch():
    """Rank invariance: the criterion returns the same loss / stat names for a normal window, a window with no hard
    negative (hard_sim NaN), a single-token window and no matched query at all (the training loop registers NaN stats
    empty, log.Metrics.touch, so every rank keeps the same metric names)."""
    crit = IDEmbedCriterion({"ID_EMBED_HARD_RINK_M": 0.0})
    head = IDEmbedHead(16, 16, 8)
    def rec(ids, frames, boxes):
        n = len(ids)
        return {"embeds": torch.randn(n, 16), "ids": torch.tensor(ids), "frames": torch.tensor(frames),
                "boxes": torch.tensor(boxes, dtype=torch.float32), "rink_xy": None, "teams": None}
    near = rec([1, 2, 1, 2], [0, 0, 1, 1], [[.5, .5, .1, .1], [.52, .5, .1, .1], [.5, .5, .1, .1], [.52, .5, .1, .1]])
    far = rec([1, 2, 1, 2], [0, 0, 1, 1], [[.1, .1, .05, .05], [.9, .9, .05, .05], [.1, .1, .05, .05], [.9, .9, .05, .05]])
    single = rec([1], [0], [[.5, .5, .1, .1]])
    keys = None
    for recs in ([near], [far], [single], [None], [far, None]):
        embeds = [r["embeds"] for r in recs if r is not None]
        flat = torch.cat(embeds) if embeds else torch.zeros(0, 16)
        z = head(flat)
        out, stats = crit.loss(recs, z, "cpu")
        assert set(out) == {"id_embed", "total"}
        if keys is None:
            keys = set(stats)
        assert set(stats) == keys
        out["total"].backward()       # connected graph: every head parameter gets a (possibly zero) gradient
        assert all(p.grad is not None for p in head.parameters())
        head.zero_grad()
    _, st_far = crit.loss([far], head(far["embeds"]), "cpu")
    assert st_far["id_embed_n_hard"] == 0 and math.isnan(st_far["id_embed_hard_sim"])
