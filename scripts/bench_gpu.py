"""Speed test bed for the MOTIP tracking loop.

Measures every component of the per-frame budget so the same numbers can be
compared across GPUs (devbox T4 vs a k8s GPU node pool) and across batch sizes.

Arms (select with --arms, default all that can run):

  env      GPU / driver / torch / clock and contention snapshot -- always run
  data     dataloader throughput vs num_workers (JPEG decode + resize + normalize)
  ddetr    MOTIP's deformable DETR: batch size x resolution x dtype
  idhead   trajectory modeling + ID decoder: track count x dtype
  rfdetr   RF-DETR variants x resolution x batch, if rfdetr is importable

Why batch size matters here: the tracking loop is inherently sequential (frame t's
ID assignment depends on t-1) so the ID head cannot batch, but the DETR pass can --
it is a pure per-frame function. `ddetr` reports ms *per frame* (batch latency /
batch size) so the batching win is read off directly.

The CUDA MultiScaleDeformableAttention op is optional. If it is not importable
(any image without the compiled extension), --allow-pytorch-msda swaps in the
reference PyTorch implementation. That is slower, but slower by the same factor on
every GPU, so cross-GPU ratios stay valid. The choice is recorded in the output.

Usage
-----
    python scripts/bench_gpu.py --out /tmp/bench_t4.json
    python scripts/bench_gpu.py --arms ddetr --batches 1,2,4,8
    python scripts/bench_gpu.py --compare /tmp/bench_t4.json /tmp/bench_l4.json

Run it from a consumer repo that vendors MOTIP by pointing MOTIP_ROOT at the
vendored copy; `--config` is relative to that root.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# This script lives in MOTIP's own scripts/, so the repo root is one level up.
# MOTIP_ROOT still overrides, for callers running a vendored copy from elsewhere.
MOTIP_ROOT = Path(
    os.environ.get("MOTIP_ROOT", Path(__file__).resolve().parent.parent)
)
DEFAULT_CONFIG = "configs/eval_stage2_hockey.yaml"

# 16:9 source resized by shorter side, capped on longer -- what SeqDataset produces
# for 720p input at the pipeline's max_shorter=800 / max_longer=1536.
DEFAULT_SIZES = [(800, 1440), (608, 1088), (448, 800)]
DEFAULT_BATCHES = [1, 2, 4, 8]
DEFAULT_TRACK_COUNTS = [8, 12, 24]
DEFAULT_WORKERS = [0, 2, 4, 8]


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def _nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return [line.strip() for line in out.stdout.strip().splitlines()]
    except Exception:
        return []


def _gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception:
        return []


def collect_env(msda: str) -> dict:
    import torch

    name = _nvidia_smi("name")
    clocks = _nvidia_smi("clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu")
    env = {
        "gpu": name[0] if name else "unknown",
        "gpu_count": torch.cuda.device_count(),
        "driver": (_nvidia_smi("driver_version") or ["unknown"])[0],
        "clock_snapshot": clocks[0] if clocks else "unknown",
        "other_gpu_processes": _gpu_processes(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability()),
        "msda": msda,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "k8s_node_pool": os.environ.get("BENCH_NODE_POOL", ""),
    }
    return env


def _parse_clocks(snapshot: str) -> tuple[float, float, float] | None:
    """(sm_mhz, sm_max_mhz, util_pct) from a 'clocks.sm,clocks.max.sm,util,temp' row."""
    try:
        parts = [p.strip() for p in snapshot.split(",")]
        return (
            float(parts[0].split()[0]),
            float(parts[1].split()[0]),
            float(parts[2].split()[0]),
        )
    except Exception:
        return None


def contention_warning(env: dict) -> str | None:
    """A busy GPU inflates every absolute number. Say so loudly rather than silently.

    Two independent signals, both needed: another process holding the GPU, and a
    depressed SM clock *while the GPU is actually working*. An idle GPU sits at its
    minimum clock by design, so the clock alone means nothing -- which is why this
    reads `clock_under_load`, sampled after the first arm has warmed the card, and
    ignores the pre-run snapshot for throttle purposes.
    """
    problems = []
    others = [p for p in env["other_gpu_processes"] if p]
    if others:
        problems.append(f"{len(others)} other process(es) on the GPU: {others}")
    loaded = _parse_clocks(env.get("clock_under_load", ""))
    if loaded:
        sm, sm_max, util = loaded
        if util >= 20.0 and sm < 0.75 * sm_max:
            problems.append(
                f"SM clock {sm:.0f} MHz against a {sm_max:.0f} MHz max at {util:.0f}% "
                "util (thermal/power throttling or a co-tenant)"
            )
    return " | ".join(problems) if problems else None


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def torch_dtype(name: str):
    """Map a dtype name to a torch dtype, rejecting typos loudly."""
    import torch

    try:
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[name]
    except KeyError:
        raise SystemExit(
            f"unknown dtype {name!r}; choose from fp16, bf16, fp32"
        ) from None


def dtype_supported(name: str) -> str | None:
    """Return a warning if the device lacks real support for this dtype, else None.

    bf16 has no tensor-core path before sm_80, so a T4 will run it via emulation
    and post a misleadingly bad number. Worth saying out loud rather than letting
    the table imply bf16 is simply slow.
    """
    import torch

    if name != "bf16":
        return None
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        return f"bf16 has no tensor-core support on sm_{major}{minor} (needs sm_80+)"
    return None


def timed(fn, iters: int, warmup: int) -> dict:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p90_ms": sorted(samples)[int(0.9 * (len(samples) - 1))],
        "iters": iters,
    }


# ---------------------------------------------------------------------------
# MOTIP setup
# ---------------------------------------------------------------------------


def install_motip_path() -> None:
    ops = MOTIP_ROOT / "models" / "ops"
    for p in (str(ops), str(MOTIP_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Optional: a prebuilt MultiScaleDeformableAttention*.so dropped here is
    # picked up automatically. This is how a pod gets the *compiled* kernel
    # without a CUDA toolchain in the image -- the wheel built on the devbox
    # carries cubins for sm_75/86/89/90, so one binary covers T4, A10G and L4 as
    # long as the pod's torch matches the one it was built against. Absent, the
    # run falls back to the reference implementation (~2.7x slower) and says so.
    prebuilt = MOTIP_ROOT / "prebuilt_ops"
    if prebuilt.is_dir() and str(prebuilt) not in sys.path:
        sys.path.insert(0, str(prebuilt))


def resolve_msda(allow_pytorch: bool, force_pytorch: bool = False) -> str:
    """Return 'cuda' if the compiled op imports, else install the PyTorch fallback.

    MOTIP calls MSDeformAttnFunction.apply, which needs the compiled extension.
    The reference PyTorch core ships alongside it, so on an image without the
    extension we can patch the module to use it and still measure the model.

    `force_pytorch` uses the fallback even where the compiled op is available. That
    is how you calibrate: run it on a GPU you also have 'cuda' numbers for, and the
    resulting ratio lets a fallback-only run elsewhere be translated back onto the
    compiled-op scale.
    """
    import torch  # noqa: F401  (must precede the extension import)

    if not force_pytorch:
        try:
            import MultiScaleDeformableAttention  # noqa: F401
            return "cuda"
        except ImportError:
            pass
    if not (allow_pytorch or force_pytorch):
        raise SystemExit(
            "MultiScaleDeformableAttention is not importable. Build it "
            "(third_party/MOTIP/models/ops/make.sh) or pass --allow-pytorch-msda "
            "to fall back to the reference implementation."
        )

    import types

    # models.ops.functions imports the extension at module scope, so on an image
    # without it the import fails before we can patch anything. A placeholder lets
    # the module load; nothing ever calls into it once MSDeformAttnFunction is
    # replaced below.
    sys.modules.setdefault(
        "MultiScaleDeformableAttention",
        types.ModuleType("MultiScaleDeformableAttention"),
    )

    from models.ops.functions import ms_deform_attn_func as msda_func
    from models.ops.modules import ms_deform_attn as msda_mod

    class _PyMSDA:
        """Same .apply signature as the compiled autograd Function."""

        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations,
                  attention_weights, im2col_step):
            # The compiled op takes spatial_shapes as a tensor; the reference
            # implementation splits `value` by H*W and needs plain ints.
            shapes = [(int(h), int(w)) for h, w in spatial_shapes]
            return msda_func.ms_deform_attn_core_pytorch(
                value, shapes, sampling_locations, attention_weights
            )

    msda_mod.MSDeformAttnFunction = _PyMSDA
    return "pytorch-fallback"


def load_config(config: str) -> dict:
    from configs.util import load_super_config
    from utils.misc import yaml_to_dict

    cfg = yaml_to_dict(config)
    return load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))


def build(cfg: dict):
    from models.motip import build as build_motip

    model, _ = build_motip(config=cfg)
    return model.cuda().eval()


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def arm_ddetr(cfg, model, sizes, batches, dtypes, iters, warmup, on_load=None) -> list[dict]:
    """Deformable DETR forward: ms per frame across batch size and resolution.

    Batch is the interesting axis -- the DETR pass is per-frame and stateless, so
    unlike the ID head it can be batched, either over frames of one clip or across
    clips. Anything below 1.0x means the GPU was already saturated at batch 1.
    """
    import torch
    from utils.nested_tensor import nested_tensor_from_tensor_list

    rows = []
    for dtype_name in dtypes:
        dtype = torch_dtype(dtype_name)
        warn = dtype_supported(dtype_name)
        if warn:
            print(f"  note: {warn}")
        detr = model.detr.to(dtype)
        for H, W in sizes:
            for bs in batches:
                images = [
                    torch.randn(3, H, W, device="cuda", dtype=dtype) for _ in range(bs)
                ]
                try:
                    nt = nested_tensor_from_tensor_list(images, 0)
                    nt.tensors = nt.tensors.cuda()
                    nt.mask = nt.mask.cuda()
                    with torch.no_grad():
                        for _ in range(warmup):
                            detr(samples=nt)
                        if on_load is not None:
                            on_load()  # clocks, now that the card is actually loaded
                        t = timed(lambda: detr(samples=nt), iters, warmup=0)
                except torch.cuda.OutOfMemoryError:
                    rows.append({
                        "dtype": dtype_name, "h": H, "w": W, "batch": bs, "oom": True,
                    })
                    torch.cuda.empty_cache()
                    continue
                rows.append({
                    "dtype": dtype_name, "h": H, "w": W, "batch": bs,
                    "dtype_warning": warn,
                    "mpx": round(H * W / 1e6, 3),
                    "batch_ms": round(t["median_ms"], 2),
                    "ms_per_frame": round(t["median_ms"] / bs, 2),
                    "fps": round(1000.0 * bs / t["median_ms"], 2),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                })
                del images, nt
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
    return rows


def arm_idhead(cfg, model, track_counts, dtypes, iters, warmup) -> list[dict]:
    """Trajectory modeling + ID decoder. Sequential by construction: batch is always 1.

    Included because once the detector gets faster this becomes the next wall, and
    because it is launch-bound rather than FLOP-bound -- which is why fp16 buys
    nothing here and a faster GPU will not help it much either.
    """
    import torch

    T = cfg["MISS_TOLERANCE"] - 2
    D = cfg["DETR_HIDDEN_DIM"]
    rows = []
    for dtype_name in dtypes:
        dtype = torch_dtype(dtype_name)
        m = model.to(dtype)
        for N in track_counts:
            tf = torch.randn(1, 1, T, N, D, device="cuda", dtype=dtype)
            tb = torch.rand(1, 1, T, N, 4, device="cuda", dtype=dtype)
            tl = torch.randint(0, cfg["NUM_ID_VOCABULARY"], (1, 1, T, N), device="cuda")
            tt = (
                torch.arange(T, device="cuda", dtype=torch.int64)[None, None, :, None]
                .expand(1, 1, T, N).contiguous()
            )
            tm = torch.zeros(1, 1, T, N, device="cuda", dtype=torch.bool)
            uf = torch.randn(1, 1, 1, N, D, device="cuda", dtype=dtype)
            ub = torch.rand(1, 1, 1, N, 4, device="cuda", dtype=dtype)
            um = torch.zeros(1, 1, 1, N, device="cuda", dtype=torch.bool)
            ut = T * torch.ones(1, 1, 1, N, device="cuda", dtype=torch.int64)

            def run():
                seq = {
                    "trajectory_features": tf, "trajectory_boxes": tb,
                    "trajectory_id_labels": tl, "trajectory_times": tt,
                    "trajectory_masks": tm, "unknown_features": uf,
                    "unknown_boxes": ub, "unknown_masks": um, "unknown_times": ut,
                }
                seq = m(seq_info=seq, part="trajectory_modeling")
                return m(seq_info=seq, part="id_decoder")

            with torch.no_grad():
                t = timed(run, iters, warmup)
            rows.append({
                "dtype": dtype_name, "traj_len": T, "tracks": N,
                "ms": round(t["median_ms"], 2),
            })
            torch.cuda.empty_cache()
    return rows


def arm_data(cfg, workers, frames, gpu_step_ms) -> list[dict]:
    """Dataloader throughput: does the CPU path hide behind the GPU step?

    Two numbers per worker count: the serial cost of decode+resize+normalize, and
    the blocking wait actually observed by a loop whose GPU step takes
    `gpu_step_ms`. The second is what matters -- CPU work only shows up in wall
    clock when it exceeds the GPU step divided by worker count.
    """
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    from data.seq_dataset import SeqDataset

    tmp = Path(os.environ.get("BENCH_TMP", "/tmp")) / "bench_gpu_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = [tmp / f"{i:08d}.jpg" for i in range(1, frames + 1)]
    if not all(p.exists() for p in paths):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
        for p in paths:
            img = np.array(Image.fromarray(base).resize((1280, 720), Image.BILINEAR))
            img = np.clip(
                img.astype(np.int16) + rng.integers(-12, 12, img.shape), 0, 255
            ).astype(np.uint8)
            Image.fromarray(img).save(p, quality=90)
    mean_kib = statistics.mean(p.stat().st_size for p in paths) / 1024

    rows = []
    for nw in workers:
        ds = SeqDataset(
            seq_info={"height": 720, "width": 1280},
            image_paths=[str(p) for p in paths],
            max_shorter=800,
            max_longer=cfg.get("INFERENCE_MAX_LONGER", 1536),
            size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
            dtype=torch.float16,
        )
        dl = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=nw,
            pin_memory=True, collate_fn=lambda x: x[0],
        )
        it = iter(dl)
        for _ in range(min(20, frames // 4)):  # worker spin-up
            next(it)
        waits = []
        while True:
            t0 = time.perf_counter()
            try:
                next(it)
            except StopIteration:
                break
            waits.append((time.perf_counter() - t0) * 1000.0)
            if gpu_step_ms:  # stand in for the GPU step so workers can run ahead
                time.sleep(gpu_step_ms / 1000.0)
        rows.append({
            "num_workers": nw,
            "jpeg_kib": round(mean_kib),
            "simulated_gpu_step_ms": gpu_step_ms,
            "blocking_wait_median_ms": round(statistics.median(waits), 2),
            "blocking_wait_p90_ms": round(sorted(waits)[int(0.9 * (len(waits) - 1))], 2),
        })
        del dl, it
    return rows


def arm_breakdown(cfg, model, sizes, dtypes, iters, warmup) -> list[dict]:
    """Split the detector's time across backbone / encoder / decoder.

    Answers "surely the model is quicker than that" with a decomposition rather
    than a total. Also reports the encoder's token count, because deformable
    attention cost scales with it and it is the number that explains everything
    else: why dropping resolution helps sublinearly (the decoder's 300 queries
    are fixed), and why architectures with one feature level are so much cheaper.

    Timed with CUDA events in forward hooks, so the figures are device time for
    each submodule rather than a wall-clock guess.
    """
    import torch
    from utils.nested_tensor import nested_tensor_from_tensor_list

    rows = []
    for dtype_name in dtypes:
        dtype = torch_dtype(dtype_name)
        detr = model.detr.to(dtype)
        for H, W in sizes:
            images = [torch.randn(3, H, W, device="cuda", dtype=dtype)]
            nt = nested_tensor_from_tensor_list(images, 0)
            nt.tensors = nt.tensors.cuda()
            nt.mask = nt.mask.cuda()

            targets = {
                "backbone": detr.backbone,
                "encoder": detr.transformer.encoder,
                "decoder": detr.transformer.decoder,
            }
            events: dict[str, list] = {k: [] for k in targets}
            handles = []

            def make_pre(name):
                def pre(_m, _inp):
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    events[name].append([start, None])
                return pre

            def make_post(name):
                def post(_m, _inp, _out):
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    events[name][-1][1] = end
                return post

            for name, mod in targets.items():
                handles.append(mod.register_forward_pre_hook(make_pre(name)))
                handles.append(mod.register_forward_hook(make_post(name)))

            # token count per feature level, for the encoder cost story
            tokens = None
            with torch.no_grad():
                for _ in range(max(warmup, 3)):
                    detr(samples=nt)
                for k in events:
                    events[k].clear()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    detr(samples=nt)
                torch.cuda.synchronize()
                total_ms = (time.perf_counter() - t0) / iters * 1000.0

            for h in handles:
                h.remove()

            # Encoder token count: derived from the feature-map strides Deformable
            # DETR builds (8, 16, 32, 64 for 4 levels).
            levels = cfg["DETR_NUM_FEATURE_LEVELS"]
            tokens = sum(
                (H // s) * (W // s) for s in [8 * 2 ** i for i in range(levels)]
            )

            row = {"dtype": dtype_name, "h": H, "w": W, "total_ms": round(total_ms, 2),
                   "encoder_tokens": tokens, "levels": levels,
                   "enc_layers": cfg["DETR_ENC_LAYERS"],
                   "dec_layers": cfg["DETR_DEC_LAYERS"],
                   "queries": cfg["DETR_NUM_QUERIES"]}
            for name, pairs in events.items():
                # decoder fires once per forward; backbone/encoder likewise. Sum
                # each forward then average over iters.
                ms = sum(a.elapsed_time(b) for a, b in pairs if b is not None)
                row[f"{name}_ms"] = round(ms / iters, 2)
                row[f"{name}_pct"] = round(100.0 * (ms / iters) / total_ms, 1)
            rows.append(row)
            del images, nt
            torch.cuda.empty_cache()
    return rows


def arm_compile(cfg, model, sizes, dtypes, iters, warmup) -> list[dict]:
    """torch.compile on the detector and the ID head -- the no-retrain lever.

    Two different bets, so two different modes:

    - detector: default mode. Mostly fusion and kernel-selection wins. The
      compiled deformable-attention op is an opaque autograd Function, so expect
      graph breaks around it and a modest gain at best.
    - ID head: ``reduce-overhead``, which uses CUDA graphs. This is the one worth
      caring about -- the ID head is launch-bound (flat across track count,
      trajectory length and dtype), which is exactly what CUDA graphs fix.

    Compile time is excluded: the timed region runs after warmup, and warmup is
    forced to at least 3 so the first (compiling) call is never measured.

    Eager and compiled are measured **alternately** over several rounds and each
    reduced by median. A single eager-then-compiled pass is not trustworthy on a
    shared GPU: a co-tenant appearing between the two halves shows up as a
    speedup. Measured drift on this box was eager 152 -> 231 ms across two runs
    while compiled held at 140, which would have read as a 1.6x win.
    """
    import torch
    from utils.nested_tensor import nested_tensor_from_tensor_list

    warmup = max(warmup, 3)
    rounds = 3
    rows = []
    H, W = sizes[0]

    def alternate(run_eager, run_compiled):
        """Interleave the two measurements so drift hits both equally."""
        eager_ms, compiled_ms = [], []
        for _ in range(rounds):
            eager_ms.append(timed(run_eager, iters, warmup)["median_ms"])
            compiled_ms.append(timed(run_compiled, iters, warmup)["median_ms"])
        return statistics.median(eager_ms), statistics.median(compiled_ms)

    for dtype_name in dtypes:
        dtype = torch_dtype(dtype_name)

        # --- detector ---
        images = [torch.randn(3, H, W, device="cuda", dtype=dtype)]
        nt = nested_tensor_from_tensor_list(images, 0)
        nt.tensors = nt.tensors.cuda()
        nt.mask = nt.mask.cuda()
        detr = model.detr.to(dtype)
        row = {"component": "detector", "dtype": dtype_name, "h": H, "w": W}
        try:
            compiled = torch.compile(detr)
            with torch.no_grad():
                e_ms, c_ms = alternate(
                    lambda: detr(samples=nt), lambda: compiled(samples=nt)
                )
            row["eager_ms"] = round(e_ms, 2)
            row["compiled_ms"] = round(c_ms, 2)
            row["speedup"] = round(e_ms / c_ms, 3)
        except Exception as e:  # noqa: BLE001 - compile failures are the finding
            with torch.no_grad():
                row["eager_ms"] = round(
                    timed(lambda: detr(samples=nt), iters, warmup)["median_ms"], 2
                )
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
        torch._dynamo.reset()
        torch.cuda.empty_cache()

        # --- ID head, the launch-bound one ---
        T = cfg["MISS_TOLERANCE"] - 2
        D = cfg["DETR_HIDDEN_DIM"]
        N = 12
        m = model.to(dtype)
        tf = torch.randn(1, 1, T, N, D, device="cuda", dtype=dtype)
        tb = torch.rand(1, 1, T, N, 4, device="cuda", dtype=dtype)
        tl = torch.randint(0, cfg["NUM_ID_VOCABULARY"], (1, 1, T, N), device="cuda")
        tt = (torch.arange(T, device="cuda", dtype=torch.int64)[None, None, :, None]
              .expand(1, 1, T, N).contiguous())
        tm = torch.zeros(1, 1, T, N, device="cuda", dtype=torch.bool)
        uf = torch.randn(1, 1, 1, N, D, device="cuda", dtype=dtype)
        ub = torch.rand(1, 1, 1, N, 4, device="cuda", dtype=dtype)
        um = torch.zeros(1, 1, 1, N, device="cuda", dtype=torch.bool)
        ut = T * torch.ones(1, 1, 1, N, device="cuda", dtype=torch.int64)

        def seq_info():
            return {
                "trajectory_features": tf, "trajectory_boxes": tb,
                "trajectory_id_labels": tl, "trajectory_times": tt,
                "trajectory_masks": tm, "unknown_features": uf,
                "unknown_boxes": ub, "unknown_masks": um, "unknown_times": ut,
            }

        def run_eager():
            s = m(seq_info=seq_info(), part="trajectory_modeling")
            return m(seq_info=s, part="id_decoder")

        row = {"component": "id_head", "dtype": dtype_name, "tracks": N,
               "traj_len": T}
        try:
            # Compile the decoder module itself; compiling the MOTIP wrapper
            # would just trace the `match part` dispatch and break immediately.
            comp_traj = torch.compile(m.trajectory_modeling, mode="reduce-overhead")
            comp_id = torch.compile(m.id_decoder, mode="reduce-overhead")

            def run_compiled():
                # reduce-overhead runs under CUDA graphs, which reuse one output
                # buffer. Without a step boundary the next call overwrites
                # tensors the previous one handed back, and torch raises
                # "accessing tensor output of CUDAGraphs that has been
                # overwritten by a subsequent run".
                torch.compiler.cudagraph_mark_step_begin()
                s = comp_traj(seq_info())
                # IDDecoder.forward has no default for use_decoder_checkpoint;
                # the MOTIP wrapper normally supplies it.
                return comp_id(s, use_decoder_checkpoint=False)

            with torch.no_grad():
                e_ms, c_ms = alternate(run_eager, run_compiled)
            row["eager_ms"] = round(e_ms, 2)
            row["compiled_ms"] = round(c_ms, 2)
            row["speedup"] = round(e_ms / c_ms, 3)
        except Exception as e:  # noqa: BLE001
            with torch.no_grad():
                row["eager_ms"] = round(
                    timed(run_eager, iters, warmup)["median_ms"], 2
                )
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    return rows


def arm_rfdetr(variants, resolutions, batches, iters, warmup,
               do_compile: bool = False) -> list[dict]:
    """RF-DETR forward, for the detector-swap comparison. Skipped if not importable.

    Note RF-DETR ships a pure-PyTorch deformable attention, so this arm never needs
    a compiled extension -- but it means an RF-DETR row is not strictly
    apples-to-apples against a 'cuda' MSDA D-DETR row. Check env.msda before
    reading the ratio.
    """
    import types

    import torch

    for name, attrs in {
        "rfdetr.datasets": {"build_dataset": None, "get_coco_api_from_dataset": None,
                            "__path__": []},
        "rfdetr.datasets.coco": {"compute_multi_scale_scales": lambda *a, **k: None},
        "rfdetr.datasets.coco_eval": {"CocoEvaluator": None},
    }.items():
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod

    from rfdetr import config as rfcfg
    from rfdetr.main import populate_args
    from rfdetr.models import build_model

    table = {
        "nano": rfcfg.RFDETRNanoConfig, "small": rfcfg.RFDETRSmallConfig,
        "medium": rfcfg.RFDETRMediumConfig, "base": rfcfg.RFDETRBaseConfig,
    }
    rows = []
    for variant in variants:
        c = table[variant](pretrain_weights=None, num_classes=1)
        for res in resolutions:
            div = c.patch_size * c.num_windows
            if res % div != 0:
                rows.append({"variant": variant, "res": res,
                             "skipped": f"not divisible by {div}"})
                continue
            args = populate_args(
                encoder=c.encoder, hidden_dim=c.hidden_dim, patch_size=c.patch_size,
                num_windows=c.num_windows, dec_layers=c.dec_layers,
                sa_nheads=c.sa_nheads, ca_nheads=c.ca_nheads,
                dec_n_points=c.dec_n_points, num_queries=c.num_queries,
                num_select=c.num_select, projector_scale=c.projector_scale,
                out_feature_indexes=c.out_feature_indexes, two_stage=c.two_stage,
                bbox_reparam=c.bbox_reparam,
                lite_refpoint_refine=c.lite_refpoint_refine, layer_norm=c.layer_norm,
                group_detr=c.group_detr, resolution=res,
                positional_encoding_size=res // c.patch_size,
                ia_bce_loss=c.ia_bce_loss, num_classes=1, pretrain_weights=None,
                force_no_pretrain=True, device="cuda", amp=False, fp16_eval=True,
            )
            for k, v in {"segmentation_head": False, "mask_downsample_ratio": 4,
                         "mask_point_sample_ratio": 16, "mask_ce_loss_coef": 5.0,
                         "mask_dice_loss_coef": 5.0}.items():
                if not hasattr(args, k):
                    setattr(args, k, v)
            built = build_model(args)
            m = (built[0] if isinstance(built, (tuple, list)) else built)
            m = m.cuda().eval().half()
            for bs in batches:
                x = torch.randn(bs, 3, res, res, device="cuda", dtype=torch.half)
                try:
                    with torch.no_grad():
                        t = timed(lambda: m(x), iters, warmup)
                except torch.cuda.OutOfMemoryError:
                    rows.append({"variant": variant, "res": res, "batch": bs,
                                 "oom": True})
                    torch.cuda.empty_cache()
                    continue
                row = {
                    "variant": variant, "res": res, "batch": bs,
                    "mpx": round(res * res / 1e6, 3),
                    "dec_layers": c.dec_layers,
                    "batch_ms": round(t["median_ms"], 2),
                    "ms_per_frame": round(t["median_ms"] / bs, 2),
                    "fps": round(1000.0 * bs / t["median_ms"], 2),
                }
                if do_compile:
                    # Interleaved with eager so GPU drift hits both equally; see
                    # arm_compile for why a sequential eager-then-compiled pass
                    # is not trustworthy on a shared card.
                    try:
                        cm = torch.compile(m)
                        e_s, c_s = [], []
                        with torch.no_grad():
                            for _ in range(3):
                                e_s.append(timed(lambda: m(x), iters, max(warmup, 3))
                                           ["median_ms"])
                                c_s.append(timed(lambda: cm(x), iters, max(warmup, 3))
                                           ["median_ms"])
                        e_ms = statistics.median(e_s)
                        c_ms = statistics.median(c_s)
                        row["eager_ms_per_frame"] = round(e_ms / bs, 2)
                        row["compiled_ms_per_frame"] = round(c_ms / bs, 2)
                        row["compile_speedup"] = round(e_ms / c_ms, 3)
                        row["compiled_fps"] = round(1000.0 * bs / c_ms, 2)
                    except Exception as exc:  # noqa: BLE001
                        row["compile_error"] = f"{type(exc).__name__}: {exc}"[:160]
                    torch._dynamo.reset()
                rows.append(row)
                del x
                torch.cuda.empty_cache()
            del m
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def render(results: dict) -> str:
    env = results["env"]
    out = [
        "",
        f"GPU        {env['gpu']}  (cc {env['capability']}, driver {env['driver']})",
        f"torch      {env['torch']}  cuda {env['cuda']}  msda={env['msda']}",
        f"host       {env['host']}  {env['cpu_count']} cpu"
        + (f"  pool={env['k8s_node_pool']}" if env["k8s_node_pool"] else ""),
        f"clocks     idle {env['clock_snapshot']}",
    ]
    if env.get("clock_under_load"):
        out.append(f"           load {env['clock_under_load']}")
    if results.get("contention"):
        out.append(f"WARNING    contended: {results['contention']}")
        out.append("           absolute ms are inflated; ratios within this run remain valid")

    if results.get("ddetr"):
        out += ["", "deformable DETR (MOTIP detector) -- ms per frame", ""]
        for warn in sorted({r.get("dtype_warning") for r in results["ddetr"]} - {None}):
            out.append(f"  note: {warn}")
        out.append(f"  {'dtype':6s} {'size':>11s} {'Mpx':>5s} {'batch':>6s} "
                   f"{'batch ms':>9s} {'ms/frame':>9s} {'fps':>7s} {'vs bs=1':>8s}")
        base = {}
        for r in results["ddetr"]:
            key = (r["dtype"], r["h"], r["w"])
            if r.get("oom"):
                out.append(f"  {r['dtype']:6s} {r['h']}x{r['w']:<6d} "
                           f"{'':>5s} {r['batch']:>6d} {'OOM':>9s}")
                continue
            base.setdefault(key, r["ms_per_frame"])
            speedup = base[key] / r["ms_per_frame"]
            out.append(
                f"  {r['dtype']:6s} {r['h']}x{r['w']:<6d} {r['mpx']:5.2f} "
                f"{r['batch']:>6d} {r['batch_ms']:9.1f} {r['ms_per_frame']:9.2f} "
                f"{r['fps']:7.2f} {speedup:7.2f}x"
            )

    if results.get("idhead"):
        out += ["", "ID head (trajectory modeling + ID decoder) -- sequential, batch 1", ""]
        out.append(f"  {'dtype':6s} {'traj':>5s} {'tracks':>7s} {'ms':>7s}")
        for r in results["idhead"]:
            out.append(f"  {r['dtype']:6s} {r['traj_len']:>5d} {r['tracks']:>7d} "
                       f"{r['ms']:7.2f}")

    if results.get("data"):
        out += ["", "data path -- blocking wait seen by the loop", ""]
        out.append(f"  {'workers':>7s} {'jpeg KiB':>9s} {'gpu step ms':>12s} "
                   f"{'wait med ms':>12s} {'wait p90 ms':>12s}")
        for r in results["data"]:
            out.append(
                f"  {r['num_workers']:>7d} {r['jpeg_kib']:>9d} "
                f"{r['simulated_gpu_step_ms']:>12.0f} "
                f"{r['blocking_wait_median_ms']:>12.2f} "
                f"{r['blocking_wait_p90_ms']:>12.2f}"
            )

    if results.get("breakdown"):
        out += ["", "detector breakdown -- device time per submodule", ""]
        r0 = results["breakdown"][0]
        out.append(f"  config: {r0['levels']} feature levels, "
                   f"{r0['enc_layers']} enc / {r0['dec_layers']} dec layers, "
                   f"{r0['queries']} queries")
        out.append("")
        out.append(f"  {'size':>11s} {'enc tokens':>11s} {'total':>8s} "
                   f"{'backbone':>17s} {'encoder':>17s} {'decoder':>17s}")
        for r in results["breakdown"]:
            def cell(name):
                return f"{r.get(f'{name}_ms', 0):7.1f} ({r.get(f'{name}_pct', 0):4.1f}%)"
            out.append(
                f"  {r['h']}x{r['w']:<6d} {r['encoder_tokens']:>11,d} "
                f"{r['total_ms']:8.1f} {cell('backbone'):>17s} "
                f"{cell('encoder'):>17s} {cell('decoder'):>17s}"
            )

    if results.get("compile"):
        out += ["", "torch.compile -- eager vs compiled", ""]
        out.append(f"  {'component':10s} {'dtype':6s} {'eager ms':>9s} "
                   f"{'compiled ms':>12s} {'speedup':>8s}")
        for r in results["compile"]:
            if r.get("error"):
                out.append(f"  {r['component']:10s} {r['dtype']:6s} "
                           f"{r['eager_ms']:9.2f} {'FAILED':>12s}")
                out.append(f"      {r['error']}")
                continue
            out.append(f"  {r['component']:10s} {r['dtype']:6s} "
                       f"{r['eager_ms']:9.2f} {r['compiled_ms']:12.2f} "
                       f"{r['speedup']:7.2f}x")

    if results.get("rfdetr"):
        out += ["", "RF-DETR -- ms per frame", ""]
        out.append(f"  {'variant':8s} {'res':>6s} {'Mpx':>5s} {'dec':>4s} {'batch':>6s} "
                   f"{'batch ms':>9s} {'ms/frame':>9s} {'fps':>7s}")
        for r in results["rfdetr"]:
            if r.get("skipped"):
                out.append(f"  {r['variant']:8s} {r['res']:>6d} -- {r['skipped']}")
                continue
            if r.get("oom"):
                out.append(f"  {r['variant']:8s} {r['res']:>6d} {'':>5s} {'':>4s} "
                           f"{r['batch']:>6d} {'OOM':>9s}")
                continue
            line = (
                f"  {r['variant']:8s} {r['res']:>6d} {r['mpx']:5.2f} "
                f"{r['dec_layers']:>4d} {r['batch']:>6d} {r['batch_ms']:9.1f} "
                f"{r['ms_per_frame']:9.2f} {r['fps']:7.2f}"
            )
            if r.get("compiled_ms_per_frame") is not None:
                line += (f"   compiled {r['compiled_ms_per_frame']:7.2f} ms "
                         f"({r['compiled_fps']:6.2f} fps, "
                         f"{r['compile_speedup']:.2f}x)")
            elif r.get("compile_error"):
                line += f"   compile FAILED: {r['compile_error']}"
            out.append(line)

    out.append("")
    return "\n".join(out)


def render_compare(paths: list[str]) -> str:
    """Side-by-side ms/frame for two or more result files (e.g. T4 vs L4)."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    out = ["", "GPU comparison -- deformable DETR ms/frame", ""]
    header = f"  {'dtype':6s} {'size':>11s} {'batch':>6s}"
    for r in runs:
        header += f" {r['env']['gpu'][:16]:>17s}"
    header += f" {'ratio':>8s}"
    out.append(header)
    keyed = []
    for r in runs:
        keyed.append({(x["dtype"], x["h"], x["w"], x["batch"]): x
                      for x in r.get("ddetr", []) if not x.get("oom")})
    for key in sorted(keyed[0]):
        if not all(key in k for k in keyed):
            continue
        line = f"  {key[0]:6s} {key[1]}x{key[2]:<6d} {key[3]:>6d}"
        vals = [k[key]["ms_per_frame"] for k in keyed]
        for v in vals:
            line += f" {v:17.2f}"
        line += f" {vals[0] / vals[-1]:7.2f}x"
        out.append(line)
    msdas = {r["env"]["msda"] for r in runs}
    if len(msdas) > 1:
        out += ["", f"  WARNING: runs used different MSDA backends {msdas} -- "
                    "ratios are not comparable"]
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="ddetr,idhead,data,rfdetr",
                    help="comma list of ddetr,idhead,breakdown,compile,data,rfdetr")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="MOTIP config, relative to third_party/MOTIP")
    ap.add_argument("--batches", type=_int_list, default=DEFAULT_BATCHES)
    ap.add_argument("--sizes", default=",".join(f"{h}x{w}" for h, w in DEFAULT_SIZES),
                    help="comma list of HxW")
    ap.add_argument("--dtypes", default="fp16",
                    help="comma list of fp16,bf16,fp32. bf16 needs sm_80+ for "
                         "tensor cores, so it is emulated (and slow) on a T4")
    ap.add_argument("--tracks", type=_int_list, default=DEFAULT_TRACK_COUNTS)
    ap.add_argument("--workers", type=_int_list, default=DEFAULT_WORKERS)
    ap.add_argument("--data-frames", type=int, default=120)
    ap.add_argument("--data-gpu-step-ms", type=float, default=98.0,
                    help="stand-in GPU step for the data arm; set to the ddetr "
                         "ms/frame you actually measured")
    ap.add_argument("--rfdetr-variants", default="nano,small,medium")
    ap.add_argument("--rfdetr-res", type=_int_list, default=[576, 768, 1088])
    ap.add_argument("--rfdetr-compile", action="store_true",
                    help="also measure each RF-DETR cell under torch.compile, "
                         "interleaved with eager")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--allow-pytorch-msda", action="store_true",
                    help="fall back to reference deformable attention if the "
                         "compiled op is missing (portable, slower, still "
                         "comparable across GPUs)")
    ap.add_argument("--force-pytorch-msda", action="store_true",
                    help="use the reference implementation even where the compiled "
                         "op exists, to calibrate fallback-only runs against it")
    ap.add_argument("--cfg-override", action="append", default=[], metavar="KEY=VAL",
                    help="override a MOTIP config key, e.g. "
                         "--cfg-override DETR_NUM_FEATURE_LEVELS=3. Ints and bools "
                         "are coerced. Timing-only: the weights will not match, so "
                         "use this to price an architecture change, not to score it")
    ap.add_argument("--out", default="", help="write JSON results here")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="render a side-by-side of existing result JSONs and exit")
    ap.add_argument("--quick", action="store_true",
                    help="fewer iterations and a single size, for a smoke test")
    args = ap.parse_args()

    if args.compare:
        print(render_compare(args.compare))
        return 0

    if args.quick:
        args.iters, args.warmup = 6, 3
        args.batches = [1, 4]
        args.sizes = "800x1440"
        args.workers = [2]
        args.tracks = [12]
        args.rfdetr_variants = "medium"
        args.rfdetr_res = [1088]

    sizes = []
    for tok in args.sizes.split(","):
        h, w = tok.strip().lower().split("x")
        sizes.append((int(h), int(w)))
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    arms = {a.strip() for a in args.arms.split(",") if a.strip()}

    if not MOTIP_ROOT.exists():
        raise SystemExit(f"MOTIP_ROOT does not exist: {MOTIP_ROOT}")
    install_motip_path()
    cwd = os.getcwd()
    os.chdir(MOTIP_ROOT)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")

    msda = resolve_msda(args.allow_pytorch_msda, args.force_pytorch_msda)
    results = {"env": collect_env(msda), "args": vars(args) | {"sizes": sizes}}

    needs_model = bool(arms & {"ddetr", "idhead", "compile", "breakdown"})
    cfg = load_config(args.config)
    for item in args.cfg_override:
        key, _, raw = item.partition("=")
        if raw.lower() in ("true", "false"):
            val = raw.lower() == "true"
        else:
            try:
                val = int(raw)
            except ValueError:
                val = raw
        print(f"  cfg override: {key} {cfg.get(key)!r} -> {val!r}")
        cfg[key] = val
    results["cfg_overrides"] = args.cfg_override
    model = build(cfg) if needs_model else None

    CLOCK_QUERY = "clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu"
    loaded_clocks: list[str] = []

    def probe_clocks() -> None:
        """Called from inside the first arm, while the GPU is genuinely busy."""
        if not loaded_clocks:
            loaded_clocks.extend(_nvidia_smi(CLOCK_QUERY))

    if "ddetr" in arms:
        results["ddetr"] = arm_ddetr(cfg, model, sizes, args.batches, dtypes,
                                     args.iters, args.warmup, on_load=probe_clocks)
    if "idhead" in arms:
        id_dtypes = dtypes if len(dtypes) > 1 else ["fp16", "fp32"]
        results["idhead"] = arm_idhead(cfg, model, args.tracks, id_dtypes,
                                       args.iters, args.warmup)
    if "breakdown" in arms:
        results["breakdown"] = arm_breakdown(cfg, model, sizes, dtypes,
                                             args.iters, args.warmup)
    if "compile" in arms:
        results["compile"] = arm_compile(cfg, model, sizes, dtypes,
                                         args.iters, args.warmup)
    if "data" in arms:
        results["data"] = arm_data(cfg, args.workers, args.data_frames,
                                   args.data_gpu_step_ms)
    if "rfdetr" in arms:
        try:
            results["rfdetr"] = arm_rfdetr(
                [v.strip() for v in args.rfdetr_variants.split(",") if v.strip()],
                args.rfdetr_res, args.batches, args.iters, args.warmup,
                do_compile=args.rfdetr_compile,
            )
        except ImportError as e:
            results["rfdetr_skipped"] = f"{type(e).__name__}: {e}"

    results["env"]["clock_under_load"] = loaded_clocks[0] if loaded_clocks else ""
    results["contention"] = contention_warning(results["env"])

    text = render(results)
    print(text)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = Path(cwd) / out
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
