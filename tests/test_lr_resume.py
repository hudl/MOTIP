"""The per-epoch LR schedule on start-up (train.py position_scheduler_after_resume).

A fresh fine-tune from a checkpoint (RESUME_STATES False) must start the schedule at epoch 0. Every v1x run so far stepped
the cosine start_epoch times with the *pretrain's* start_epoch (6, stage-2 checkpoint_5) before the reset, so epoch e used
the LR of e + 6 (2026-11-22 note in sessions/2026-11-21-joint-ablation-rink-idloss.md). A real resume (RESUME_STATES True,
--resume-scheduler true) continues the saved schedule exactly as an uninterrupted run would.
"""
import math

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from models.misc import load_checkpoint
from train import cosine_lr_lambda, position_scheduler_after_resume

LR, EPOCHS, MIN_RATIO = 2.0e-5, 13, 0.01     # the v1x recipe (finetune_crossing_rfdetr_v1.yaml + smoketest cosine)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.detr = nn.Linear(2, 2)     # a 'detr.' key: load_checkpoint takes the full-MOTIP-checkpoint path
        self.head = nn.Linear(2, 2)


def _build():
    model = _Tiny()
    opt = AdamW(model.parameters(), lr=LR)
    sched = LambdaLR(opt, lr_lambda=cosine_lr_lambda(warmup_epochs=0, total_epochs=EPOCHS, min_lr_ratio=MIN_RATIO))
    return model, opt, sched


def _intended(e):
    return LR * (MIN_RATIO + (1 - MIN_RATIO) * 0.5 * (1 + math.cos(math.pi * (e + 1) / EPOCHS)))


def _run_epochs(opt, sched, start, stop, save_at=None, save=None):
    """train.py's epoch loop: the LR of epoch e is the group LR during it; save happens before the end-of-epoch step."""
    lrs = {}
    for e in range(start, stop):
        lrs[e] = opt.param_groups[0]["lr"]
        if save_at is not None and e == save_at:
            save(e)
        sched.step()
    return lrs


def _save(path, model, opt, sched, states):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "states": dict(states)}, path)


def test_fresh_finetune_from_start_epoch_6_checkpoint_starts_the_schedule_at_0(tmp_path):
    # The pretrain: a stage-2 checkpoint at the end of its epoch 5 (start_epoch 6), scheduler 5 steps in.
    m0, o0, s0 = _build()
    for _ in range(5):
        s0.step()
    ck = tmp_path / "checkpoint_5.pth"
    _save(ck, m0, o0, s0, {"start_epoch": 6, "global_step": 6000})

    model, opt, sched = _build()
    states = {"start_epoch": 0, "global_step": 0}
    load_checkpoint(model=model, path=str(ck), optimizer=None, scheduler=None, states=states)   # --resume-* false
    assert states["start_epoch"] == 6
    position_scheduler_after_resume(sched, states, resume_scheduler=False, resume_states=False)
    assert states == {"start_epoch": 0, "global_step": 0}
    lrs = _run_epochs(opt, sched, states["start_epoch"], EPOCHS)
    for e in (0, 6, 12):
        assert lrs[e] == pytest.approx(_intended(e), rel=1e-9), e
    assert lrs[0] == pytest.approx(1.9713e-5, rel=1e-3)     # near the peak, not the old lambda(6) = 0.445 * LR
    assert lrs[6] == pytest.approx(8.909e-6, rel=1e-3)
    assert lrs[12] == pytest.approx(LR * MIN_RATIO, rel=1e-9)
    assert lrs[5] > LR * MIN_RATIO * 10   # the old bug sat at the floor from epoch 6 (lambda(12)) onward; now mid-cosine


def test_old_order_was_six_epochs_ahead(tmp_path):
    """Documents the bug the fix removes: stepping by the checkpoint's start_epoch before the reset."""
    model, opt, sched = _build()
    for _ in range(6):
        sched.step()
    lrs = _run_epochs(opt, sched, 0, EPOCHS)
    assert lrs[0] == pytest.approx(_intended(6))
    assert lrs[6] == pytest.approx(_intended(12))


def test_real_resume_continues_the_saved_schedule(tmp_path):
    ck = tmp_path / "checkpoint_2.pth"
    model, opt, sched = _build()
    states = {"start_epoch": 0, "global_step": 0}
    position_scheduler_after_resume(sched, states, resume_scheduler=False, resume_states=False)   # fresh start

    def save(e):
        _save(ck, model, opt, sched, {"start_epoch": e + 1, "global_step": (e + 1) * 100})
    full = _run_epochs(opt, sched, 0, EPOCHS, save_at=2, save=save)

    m2, o2, s2 = _build()
    st2 = {"start_epoch": 0, "global_step": 0}
    load_checkpoint(model=m2, path=str(ck), optimizer=o2, scheduler=s2, states=st2)   # --resume-optimizer/scheduler true
    position_scheduler_after_resume(s2, st2, resume_scheduler=True, resume_states=True)
    assert st2["start_epoch"] == 3
    resumed = _run_epochs(o2, s2, st2["start_epoch"], EPOCHS)
    for e in range(3, EPOCHS):
        assert resumed[e] == pytest.approx(full[e], rel=1e-12), e
        assert resumed[e] == pytest.approx(_intended(e), rel=1e-9), e


def test_resume_states_without_scheduler_state_steps_by_start_epoch():
    model, opt, sched = _build()
    states = {"start_epoch": 4, "global_step": 400}
    position_scheduler_after_resume(sched, states, resume_scheduler=False, resume_states=True)
    assert states["start_epoch"] == 4
    assert opt.param_groups[0]["lr"] == pytest.approx(_intended(4))
