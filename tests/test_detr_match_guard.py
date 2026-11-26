"""NaN guard on the matcher inside RF-DETR's own SetCriterion (models/motip GuardedSetMatcher) and the rank-invariant
non-finite batch skip in train.py. The v1x_idloss resume and the s1_long arm C jobs died in that matcher
(rfdetr/models/matcher.py: 'matrix contains invalid numeric entries', called from SetCriterion.forward)."""
import datetime
import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from models.motip import (DETR_MATCH_GUARD_STATS, GuardedSetMatcher, install_criterion_match_guard,
                          take_match_guard_events)

matcher_mod = pytest.importorskip("rfdetr.models.matcher")
lwdetr = pytest.importorskip("rfdetr.models.lwdetr")


def _matcher():
    return matcher_mod.HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)


def _criterion(group_detr=1):
    wd = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    return lwdetr.SetCriterion(2, matcher=_matcher(), weight_dict=wd, focal_alpha=0.25,
                               losses=["labels", "boxes", "cardinality"], group_detr=group_detr)


def _outputs(n_q=8, bs=2, degenerate=True, all_nan=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {"pred_logits": torch.randn(bs, n_q, 2, generator=g),
           "pred_boxes": torch.rand(bs, n_q, 4, generator=g) * 0.3 + 0.2}
    boxes = torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.6, 0.5, 0.08, 0.15], [0.45, 0.45, 0.1, 0.2]])
    tgts = [{"labels": torch.tensor([1, 1, 1]), "boxes": boxes.clone()} for _ in range(bs)]
    if degenerate:
        tgts[0]["boxes"][2] = torch.tensor([0.45, 0.45, 0.0, 0.0])          # zero-area GT
        out["pred_boxes"][0, 3] = torch.tensor([0.45, 0.45, 0.0, 0.0])      # zero-area prediction on top of it
        out["pred_logits"][1, 2, 0] = float("nan")                         # one NaN logit
    if all_nan:
        out["pred_boxes"][:] = float("nan")
        out["pred_logits"][:] = float("nan")
    return out, tgts


def test_unguarded_criterion_crashes_like_the_jobs():
    out, tgts = _outputs()
    with pytest.raises(ValueError, match="invalid numeric entries"):
        _criterion()(out, tgts)


@pytest.mark.parametrize("group_detr", [1, 2])
def test_guarded_criterion_survives_and_counts(group_detr):
    crit = _criterion(group_detr)
    raw = install_criterion_match_guard(crit)
    assert isinstance(crit.matcher, GuardedSetMatcher) and crit.matcher.matcher is raw
    assert install_criterion_match_guard(crit) is raw         # idempotent
    take_match_guard_events()
    out, tgts = _outputs(n_q=8)
    crit.train()
    losses = crit(out, tgts)                    # no ValueError (the NaN logit / 0-area GIoU losses are left to the skip)
    assert torch.isfinite(losses["loss_bbox"])
    ev = take_match_guard_events()
    assert ev["detr_match_sanitized"] >= 1
    assert take_match_guard_events()["detr_match_sanitized"] == 0     # per-step counter resets
    idx = crit.matcher(out, tgts, group_detr=group_detr)
    for (pi, ti) in idx:
        assert len(pi) == 3 * group_detr and len(set(pi.tolist())) == len(pi)


def test_guard_is_identity_on_normal_inputs():
    out, tgts = _outputs(degenerate=False)
    ref = _matcher()(out, tgts, group_detr=2)
    got = GuardedSetMatcher(_matcher())(out, tgts, group_detr=2)
    for (a, b), (c, d) in zip(ref, got):
        assert torch.equal(a, c) and torch.equal(b, d)


def test_grouped_fallback_when_matcher_still_raises():
    def bad(o, t, group_detr=1):
        raise ValueError("matrix contains invalid numeric entries")
    take_match_guard_events()
    out, tgts = _outputs(n_q=8)
    res = GuardedSetMatcher(bad)(out, tgts, group_detr=2)
    assert DETR_MATCH_GUARD_STATS["fallback"] == 1
    for pi, ti in res:
        assert sorted(ti.tolist()) == [0, 0, 1, 1, 2, 2]
        assert (pi[:3] < 4).all() and (pi[3:] >= 4).all()       # one match per query group


def test_all_nan_batch_does_not_raise_and_is_flagged():
    from train import nonfinite_loss_on_any_rank

    class _Acc:
        num_processes = 1
    crit = _criterion()
    install_criterion_match_guard(crit)
    out, tgts = _outputs(all_nan=True)
    losses = crit(out, tgts)                    # no ValueError
    loss = sum(losses[k] * crit.weight_dict[k] for k in losses if k in crit.weight_dict)
    assert nonfinite_loss_on_any_rank(loss, _Acc(), torch.device("cpu"))
    assert not nonfinite_loss_on_any_rank(torch.tensor(1.0), _Acc(), torch.device("cpu"))


# --- DDP: a NaN batch on one rank only. Every rank must do the same collectives and backward and skip the update. -----

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeAccelerator:
    """accelerator.reduce(sum) as an all_reduce over the gloo group."""
    def __init__(self, world):
        self.num_processes = world

    def reduce(self, t, reduction="sum"):
        t = t.clone()
        dist.all_reduce(t)
        return t


def _ddp_worker(rank, world, port, ret):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=30))
    from train import nonfinite_loss_on_any_rank
    torch.manual_seed(0)
    model = nn.parallel.DistributedDataParallel(nn.Linear(4, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    acc = _FakeAccelerator(world)
    before = [p.detach().clone() for p in model.parameters()]
    skips = []
    for step in range(3):
        x = torch.ones(2, 4)
        if step == 1 and rank == 1:
            x[0, 0] = float("nan")              # this rank only, this step only
        loss = model(x).sum()
        skip = nonfinite_loss_on_any_rank(loss, acc, torch.device("cpu"))
        skips.append(skip)
        if skip:
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        loss.backward()                            # same DDP all-reduce on every rank
        if skip:
            opt.zero_grad()
        opt.step()
        opt.zero_grad()
        if step == 1:
            after_skip = [p.detach().clone() for p in model.parameters()]
    dist.barrier()
    ret[rank] = {
        "skips": skips,
        "finite": all(torch.isfinite(p).all().item() for p in model.parameters()),
        "params": [p.detach().clone() for p in model.parameters()],
        "after_skip": after_skip,
    }
    dist.destroy_process_group()


def test_ddp_nan_on_one_rank_is_skipped_on_every_rank_without_desync():
    world = 2
    ret = mp.Manager().dict()
    ctx = mp.spawn(_ddp_worker, args=(world, _free_port(), ret), nprocs=world, join=False)
    deadline = time.time() + 120
    ok = False
    while not ok and time.time() < deadline:
        ok = ctx.join(timeout=5)
    if not ok:
        for p in ctx.processes:
            p.kill()
        pytest.fail("rank desync / hang on a one-rank NaN batch")
    r0, r1 = ret[0], ret[1]
    assert r0["skips"] == r1["skips"] == [False, True, False]
    assert r0["finite"] and r1["finite"]
    for a, b in zip(r0["params"], r1["params"]):
        assert torch.equal(a, b)                   # ranks still identical
    assert all(torch.isfinite(p).all() for p in r0["after_skip"])
