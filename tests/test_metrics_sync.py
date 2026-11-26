"""log.Metrics.sync must issue identical collectives on every rank even when the ranks hold different metric names
(a statistic skipped as NaN on one rank, or first recorded at different steps). The 2026-11-21 v1x_idloss SageMaker job
hung in NCCL at the first logging step of epoch 3 because one rank had one metric fewer (ALLREDUCE vs ALLGATHER at the
same sequence number). Two gloo CPU processes."""
import datetime
import math
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, ret):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=30))
    import log.log as L
    L.is_distributed = lambda: True
    L.distributed_world_size = lambda: world
    m = L.Metrics()
    if rank == 0:
        m.update("loss", 1.0)
        m.update("id_embed_hard_sim", 0.5)       # defined on rank 0 only
        m.update("id_embed_n_hard", 3.0)
        m.update("only_rank0_late", 7.0)
    else:
        m.update("id_embed_n_hard", 0.0)          # other order on rank 1
        m.update("loss", 3.0)
        m.touch("id_embed_hard_sim")              # NaN on this rank: registered empty
        m.update("only_rank1", 2.0)
    m.sync()
    m.sync()    # twice: still in step
    dist.barrier()
    ret[rank] = {k: (v.global_average, v.average) for k, v in m.metrics.items()}
    dist.destroy_process_group()


def test_sync_with_rank_dependent_metric_names_does_not_hang_and_matches_by_name():
    world = 2
    ret = mp.Manager().dict()
    ctx = mp.spawn(_worker, args=(world, _free_port(), ret), nprocs=world, join=False)
    import time
    deadline = time.time() + 120
    ok = False
    while not ok and time.time() < deadline:   # join returns False while a process still runs
        ok = ctx.join(timeout=5)
    if not ok:      # still running at the deadline: a hang
        for p in ctx.processes:
            p.kill()
        pytest.fail("Metrics.sync hung with rank-dependent metric names")
    for r in range(world):
        got = ret[r]
        assert set(got) == {"loss", "id_embed_hard_sim", "id_embed_n_hard", "only_rank0_late", "only_rank1"}
        assert got["loss"][0] == pytest.approx(2.0)               # matched by name, not by position
        assert got["id_embed_n_hard"][0] == pytest.approx(1.5)
        assert got["id_embed_hard_sim"] == (pytest.approx(0.5), pytest.approx(0.5))   # rank 1's empty value ignored
        assert got["only_rank0_late"][0] == pytest.approx(7.0)
        assert got["only_rank1"][0] == pytest.approx(2.0)


def test_empty_value_statistics_are_nan_not_errors():
    import log.log as L
    m = L.Metrics()
    m.touch("x")
    m.sync()
    assert math.isnan(m["x"].average) and math.isnan(m["x"].global_average) and math.isnan(m["x"].median)
