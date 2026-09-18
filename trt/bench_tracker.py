"""MOTIP ID head throughput: eager vs CUDA graphs.

Measures the tracking side of MOTIP (trajectory_modeling + id_decoder) in
isolation.  This is the complement to bench_detector.py -- together they show
the realistic speed ceiling for each half of the pipeline.

Three arms (all run by default):
  shape         Characterise (T, N) pairs across a real clip: how many distinct
                shapes does the ID head see?  This decides whether CUDA graph
                capture will thrash (many N values) or capture once (stable N).
                Run this on at least one real clip before trusting cuda_graphs numbers.
  eager         Baseline: trajectory_modeling + id_decoder, synchronous.
  cuda_graphs   Same but compiled with reduce-overhead (CUDA graphs).  The ID
                head is launch-bound (cost is flat across N, T, dtype), so this
                is where the 7x gain from the compile arm in bench_gpu.py comes
                from.  The padded variant pads trajectories to N=50 so the shape
                is constant and the graph is captured once rather than per-N.

The shape arm needs a real MOTIP inference run and is optional.  The timing arms
use synthetic tensors and need only a MOTIP config.

Usage
-----
    python trt/bench_tracker.py
    python trt/bench_tracker.py --tracks 8,12,24,50 --out tracker_t4.json
    python trt/bench_tracker.py --arms shape --clip-dir /data/hockey/clip_xyz
    python trt/bench_tracker.py --arms eager,cuda_graphs
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

MOTIP_ROOT = Path(os.environ.get("MOTIP_ROOT", Path(__file__).resolve().parent.parent))
DEFAULT_CONFIG = "configs/eval_stage2_hockey.yaml"


def install_motip_path() -> None:
    ops = MOTIP_ROOT / "models" / "ops"
    for p in (str(ops), str(MOTIP_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    prebuilt = MOTIP_ROOT / "prebuilt_ops"
    if prebuilt.is_dir() and str(prebuilt) not in sys.path:
        sys.path.insert(0, str(prebuilt))


def load_config(config: str) -> dict:
    from configs.util import load_super_config
    from utils.misc import yaml_to_dict
    cfg = yaml_to_dict(config)
    return load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))


def build_model(cfg: dict):
    from models.motip import build as build_motip
    model, _ = build_motip(config=cfg)
    return model.cuda().eval()


# ---------------------------------------------------------------------------
# arm: shape characterisation
# ---------------------------------------------------------------------------

def arm_shape(cfg: dict, clip_dir: str) -> dict:
    """Run the tracker on a clip and log (T, N) per frame.

    Needs a real clip -- this arm cannot be run with synthetic data.  Point
    --clip-dir at a directory of JPEG frames that SeqDataset understands.
    """
    from torch.utils.data import DataLoader
    from data.seq_dataset import SeqDataset

    if not clip_dir or not Path(clip_dir).is_dir():
        return {"skipped": "no --clip-dir provided or path does not exist"}

    image_paths = sorted(str(p) for p in Path(clip_dir).glob("*.jpg"))
    if not image_paths:
        return {"skipped": f"no .jpg frames found in {clip_dir}"}

    import models.runtime_tracker as rt_mod
    shape_log: list[tuple[int, int]] = []
    _orig_get_id = rt_mod.RuntimeTracker._get_id_pred_labels

    def _logging_get_id(self, seq_info):
        T = seq_info["trajectory_features"].shape[2]
        N = seq_info["trajectory_features"].shape[3]
        shape_log.append((T, N))
        return _orig_get_id(self, seq_info)

    rt_mod.RuntimeTracker._get_id_pred_labels = _logging_get_id

    ds = SeqDataset(
        seq_info={"height": 720, "width": 1280},
        image_paths=image_paths,
        max_shorter=800,
        max_longer=cfg.get("INFERENCE_MAX_LONGER", 1536),
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        dtype=torch.float16,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    pin_memory=True, collate_fn=lambda x: x[0])

    model = build_model(cfg)
    tracker = __import__("models.runtime_tracker", fromlist=["RuntimeTracker"]).RuntimeTracker(cfg)

    with torch.no_grad():
        for batch in dl:
            frame = batch[0].unsqueeze(0).cuda().half()
            out = model.detr(samples=frame)
            tracker.update(out, frame)

    rt_mod.RuntimeTracker._get_id_pred_labels = _orig_get_id

    distinct = sorted(set(shape_log))
    n_vals = sorted(set(n for _, n in shape_log))
    t_vals = sorted(set(t for t, _ in shape_log))
    return {
        "frames": len(shape_log),
        "distinct_TN_pairs": len(distinct),
        "N_range": [min(n_vals), max(n_vals)] if n_vals else [],
        "T_range": [min(t_vals), max(t_vals)] if t_vals else [],
        "top_N_values": sorted(n_vals, key=lambda n: sum(1 for _, nn in shape_log if nn == n), reverse=True)[:10],
        "verdict": (
            "CUDA graphs will capture once (stable N)" if len(n_vals) <= 3
            else f"CUDA graphs will capture {len(n_vals)} times -- padding to N=50 recommended"
        ),
    }


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------

def _timed(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _make_seq_info(cfg: dict, N: int, dtype, pad_to: int | None = None):
    T = cfg["MISS_TOLERANCE"] - 2
    D = cfg["DETR_HIDDEN_DIM"]
    Np = pad_to if pad_to is not None else N
    tf = torch.randn(1, 1, T, Np, D, device="cuda", dtype=dtype)
    tb = torch.rand(1, 1, T, Np, 4, device="cuda", dtype=dtype)
    tl = torch.randint(0, cfg["NUM_ID_VOCABULARY"], (1, 1, T, Np), device="cuda")
    tt = (torch.arange(T, device="cuda", dtype=torch.int64)[None, None, :, None]
          .expand(1, 1, T, Np).contiguous())
    tm = torch.zeros(1, 1, T, Np, device="cuda", dtype=torch.bool)
    if pad_to is not None and N < pad_to:
        tm[:, :, :, N:] = True  # mask padded slots
    uf = torch.randn(1, 1, 1, Np, D, device="cuda", dtype=dtype)
    ub = torch.rand(1, 1, 1, Np, 4, device="cuda", dtype=dtype)
    um = torch.zeros(1, 1, 1, Np, device="cuda", dtype=torch.bool)
    if pad_to is not None and N < pad_to:
        um[:, :, :, N:] = True
    ut = T * torch.ones(1, 1, 1, Np, device="cuda", dtype=torch.int64)
    return {
        "trajectory_features": tf, "trajectory_boxes": tb,
        "trajectory_id_labels": tl, "trajectory_times": tt,
        "trajectory_masks": tm, "unknown_features": uf,
        "unknown_boxes": ub, "unknown_masks": um, "unknown_times": ut,
    }


# ---------------------------------------------------------------------------
# arms: eager and cuda_graphs
# ---------------------------------------------------------------------------

def arm_timing(cfg: dict, model, track_counts: list[int],
               iters: int, warmup: int, use_cuda_graphs: bool,
               pad_N: int | None) -> list[dict]:
    dtype = torch.float16
    m = model.to(dtype)

    if use_cuda_graphs:
        comp_traj = torch.compile(m.trajectory_modeling, mode="reduce-overhead")
        comp_id = torch.compile(m.id_decoder, mode="reduce-overhead")

    rows = []
    for N in track_counts:
        si = _make_seq_info(cfg, N, dtype, pad_to=pad_N)

        if use_cuda_graphs:
            def run():
                torch.compiler.cudagraph_mark_step_begin()
                s = comp_traj(si)
                return comp_id(s, use_decoder_checkpoint=False)
        else:
            def run():
                s = m(seq_info=si, part="trajectory_modeling")
                return m(seq_info=s, part="id_decoder")

        try:
            warmup_count = max(warmup, 3) if use_cuda_graphs else warmup
            with torch.no_grad():
                samples = _timed(run, iters, warmup_count)
        except Exception as e:
            rows.append({"tracks": N, "padded_to": pad_N, "error": str(e)[:200]})
            torch.cuda.empty_cache()
            if use_cuda_graphs:
                torch._dynamo.reset()
            continue

        med = statistics.median(samples)
        rows.append({
            "tracks": N,
            "padded_to": pad_N,
            "traj_len": cfg["MISS_TOLERANCE"] - 2,
            "median_ms": round(med, 2),
            "min_ms": round(min(samples), 2),
        })
        torch.cuda.empty_cache()

    if use_cuda_graphs:
        torch._dynamo.reset()

    return rows


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(results: dict) -> str:
    out = [
        "",
        f"GPU    {results['gpu']}",
        f"torch  {results['torch']}",
        "",
    ]

    if results.get("shape"):
        s = results["shape"]
        if s.get("skipped"):
            out.append(f"shape arm: skipped ({s['skipped']})")
        else:
            out += [
                f"shape characterisation  ({s['frames']} frames)",
                f"  distinct (T,N) pairs: {s['distinct_TN_pairs']}",
                f"  N range: {s['N_range']}  T range: {s['T_range']}",
                f"  top N values: {s['top_N_values']}",
                f"  {s['verdict']}",
                "",
            ]

    for label, key in [("eager", "eager"), ("cuda_graphs", "cuda_graphs"),
                       ("cuda_graphs_padded", "cuda_graphs_padded")]:
        if not results.get(key):
            continue
        out.append(f"{label}  (traj_len={results[key][0].get('traj_len', '?')})")
        out.append(f"  {'tracks':>7s}  {'median ms':>10s}  {'min ms':>8s}")
        for r in results[key]:
            if r.get("error"):
                out.append(f"  {r['tracks']:>7d}  ERROR: {r['error']}")
                continue
            pad = f" (pad→{r['padded_to']})" if r.get("padded_to") else ""
            out.append(f"  {r['tracks']:>7d}  {r['median_ms']:>10.2f}  {r['min_ms']:>8.2f}{pad}")
        out.append("")

    if results.get("eager") and results.get("cuda_graphs"):
        out.append("speedup summary (cuda_graphs / eager, median)")
        eager_map = {r["tracks"]: r["median_ms"] for r in results["eager"]
                     if "median_ms" in r}
        for r in results.get("cuda_graphs_padded") or results.get("cuda_graphs") or []:
            if "median_ms" not in r or r["tracks"] not in eager_map:
                continue
            sp = eager_map[r["tracks"]] / r["median_ms"]
            out.append(f"  N={r['tracks']:>3d}  {sp:.1f}x")
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="eager,cuda_graphs",
                    help="comma list of shape,eager,cuda_graphs")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tracks", default="8,12,24,50",
                    help="comma-separated track counts to measure")
    ap.add_argument("--pad-n", type=int, default=50,
                    help="pad trajectories to this N for the padded cuda_graphs variant "
                         "(0 = disable padded variant)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--clip-dir", default="",
                    help="path to JPEG frames for the shape arm")
    ap.add_argument("--out", default="", help="write JSON result here")
    ap.add_argument("--results-json", default="",
                    help="append result JSON line here (for SageMaker collection)")
    ap.add_argument("--allow-pytorch-msda", action="store_true")
    args = ap.parse_args()

    if not MOTIP_ROOT.exists():
        raise SystemExit(f"MOTIP_ROOT does not exist: {MOTIP_ROOT}")
    install_motip_path()
    os.chdir(MOTIP_ROOT)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")

    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    track_counts = [int(x) for x in args.tracks.split(",") if x.strip()]

    cfg = load_config(args.config)
    model = None
    if arms & {"eager", "cuda_graphs"}:
        model = build_model(cfg)

    results: dict = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "config": args.config,
    }

    if "shape" in arms:
        results["shape"] = arm_shape(cfg, args.clip_dir)

    if "eager" in arms and model is not None:
        results["eager"] = arm_timing(cfg, model, track_counts,
                                      args.iters, args.warmup,
                                      use_cuda_graphs=False, pad_N=None)

    if "cuda_graphs" in arms and model is not None:
        results["cuda_graphs"] = arm_timing(cfg, model, track_counts,
                                            args.iters, args.warmup,
                                            use_cuda_graphs=True, pad_N=None)
        if args.pad_n > 0:
            padded_counts = [n for n in track_counts if n <= args.pad_n]
            results["cuda_graphs_padded"] = arm_timing(cfg, model, padded_counts,
                                                       args.iters, args.warmup,
                                                       use_cuda_graphs=True,
                                                       pad_N=args.pad_n)

    print(render(results))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    if args.results_json:
        summary = {k: v for k, v in results.items() if k != "shape"}  # shape has no timing
        with open(args.results_json, "a") as f:
            f.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
