"""TRT engine throughput: ms/frame across batch sizes.

Loads a pre-built TRT engine and drives it at configurable batch sizes.
Uses synthetic frames (random fp16 tensors) -- timing is identical to real
frames for a GPU-bound operator, and this measures the speed ceiling.

Two things this tells you:
  - At batch=1: the latency budget per frame available to the rest of the loop
  - At batch>1: whether the GPU is still underutilised (if fps keeps climbing,
    it is; if it plateaus, you've hit memory bandwidth or compute saturation)

The detector pass is stateless (each frame is independent), so batching is
always safe.  The ID head is NOT batchable across frames -- use bench_tracker.py
for that.

Usage
-----
    python trt/bench_detector.py trt/engines/rfdetr_large_1088_det_sm75_fp16.engine
    python trt/bench_detector.py <engine> --batches 1,2,4,8 --out t4_detector.json
    python trt/bench_detector.py --compare t4_detector.json a10g_detector.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


def _nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return [l.strip() for l in out.stdout.strip().splitlines()]
    except Exception:
        return []


def _gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return []


def collect_env() -> dict:
    name = _nvidia_smi("name")
    clocks = _nvidia_smi("clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu")
    return {
        "gpu": name[0] if name else torch.cuda.get_device_name(0),
        "capability": "sm" + "".join(str(x) for x in torch.cuda.get_device_capability()),
        "driver": (_nvidia_smi("driver_version") or ["unknown"])[0],
        "clock_snapshot": clocks[0] if clocks else "unknown",
        "other_gpu_processes": _gpu_processes(),
        "torch": torch.__version__,
        "k8s_node_pool": os.environ.get("BENCH_NODE_POOL", ""),
    }


def load_engine(path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    blob = path.read_bytes()
    engine = runtime.deserialize_cuda_engine(blob)
    trt_ver = getattr(trt, "__version__", "unknown")
    return engine, trt_ver


def run_batch(engine, ctx, batch: int, iters: int, warmup: int) -> list[float]:
    """Allocate IO buffers for this batch size, warm up, then time."""
    import tensorrt as trt

    dtype_map = {}
    for name, dt in (("float32", torch.float32), ("float16", torch.float16),
                     ("int32", torch.int32), ("int64", torch.int64),
                     ("int8", torch.int8), ("bool", torch.bool),
                     ("bfloat16", torch.bfloat16)):
        enum = getattr(trt, name, None)
        if enum is not None:
            dtype_map[enum] = dt

    # Infer per-tensor shapes: replace dim-0 (batch) with our batch size
    tensors = {}
    for i in range(engine.num_io_tensors):
        tname = engine.get_tensor_name(i)
        base_shape = list(ctx.get_tensor_shape(tname))
        base_shape[0] = batch
        dtype = dtype_map[engine.get_tensor_dtype(tname)]
        t = torch.zeros(base_shape, dtype=dtype, device="cuda")
        if engine.get_tensor_mode(tname).name == "INPUT":
            t = torch.randn(base_shape, dtype=torch.float16, device="cuda")
        tensors[tname] = t
        ctx.set_tensor_address(tname, t.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def bench_engine(engine_path: Path, batches: list[int],
                 iters: int, warmup: int) -> list[dict]:
    engine, trt_ver = load_engine(engine_path)
    ctx = engine.create_execution_context()

    # Detect the input tensor and its base shape
    import tensorrt as trt
    input_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n).name == "INPUT":
            input_name = n
            break
    base_shape = list(ctx.get_tensor_shape(input_name))
    # base_shape[0] might be 1 or a fixed batch from the build
    built_batch = base_shape[0]

    rows = []
    for bs in batches:
        if bs != built_batch and built_batch > 0:
            # Dynamic axes let us vary batch; if the engine was built with a fixed
            # batch != bs, skip with explanation.
            rows.append({
                "batch": bs,
                "skipped": f"engine built with fixed batch={built_batch}; rebuild with --batch {bs}",
            })
            continue
        try:
            samples = run_batch(engine, ctx, bs, iters, warmup)
        except Exception as e:
            rows.append({"batch": bs, "error": str(e)[:200]})
            continue

        med = statistics.median(samples)
        lo = min(samples)
        rows.append({
            "batch": bs,
            "batch_ms_median": round(med, 2),
            "batch_ms_min": round(lo, 2),
            "ms_per_frame_median": round(med / bs, 2),
            "ms_per_frame_min": round(lo / bs, 2),
            "fps_median": round(1000.0 * bs / med, 1),
            "fps_min": round(1000.0 * bs / lo, 1),
            "contention_spread": round(med / lo, 2),
        })
    return rows, trt_ver


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(results: dict) -> str:
    env = results["env"]
    engine = results["engine"]
    rows = results["rows"]

    out = [
        "",
        f"GPU        {env['gpu']}  ({env['capability']}, driver {env['driver']})",
        f"torch      {env['torch']}",
        f"TRT        {results['trt']}",
        f"engine     {engine}",
        f"clocks     idle {env['clock_snapshot']}",
    ]
    if env.get("other_gpu_processes"):
        out.append(f"WARNING    {len(env['other_gpu_processes'])} other process(es) on GPU "
                   "-- absolute ms inflated; ratios still valid")

    out += ["", f"  {'batch':>6s}  {'batch ms':>10s}  {'ms/frame':>10s}  {'fps':>8s}  {'spread':>7s}", ""]
    base_fps = None
    for r in rows:
        if r.get("skipped"):
            out.append(f"  batch {r['batch']:>2d}  skipped: {r['skipped']}")
            continue
        if r.get("error"):
            out.append(f"  batch {r['batch']:>2d}  ERROR: {r['error']}")
            continue
        if base_fps is None:
            base_fps = r["fps_min"]
        scaling = r["fps_min"] / base_fps
        out.append(
            f"  {r['batch']:>6d}  {r['batch_ms_min']:>10.2f}  {r['ms_per_frame_min']:>10.2f}"
            f"  {r['fps_min']:>8.1f}  {r['contention_spread']:>6.2f}x"
            f"  ({scaling:.2f}x vs batch=1)"
        )
    out.append("")
    return "\n".join(out)


def render_compare(paths: list[str]) -> str:
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    out = ["", "GPU comparison -- ms/frame at min (uncontended)", ""]
    header = f"  {'batch':>6s}"
    for r in runs:
        header += f"  {r['env']['gpu'][:18]:>20s}"
    header += f"  {'ratio':>7s}"
    out.append(header)

    keyed = [{r["batch"]: r for r in run["rows"] if "ms_per_frame_min" in r}
             for run in runs]
    all_batches = sorted(keyed[0].keys())
    for bs in all_batches:
        if not all(bs in k for k in keyed):
            continue
        line = f"  {bs:>6d}"
        vals = [k[bs]["ms_per_frame_min"] for k in keyed]
        for v in vals:
            line += f"  {v:>20.2f}"
        ratio = vals[0] / vals[-1] if vals[-1] > 0 else float("nan")
        line += f"  {ratio:>7.2f}x"
        out.append(line)
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine", nargs="?", help="path to .engine file")
    ap.add_argument("--batches", default="1,2,4,8",
                    help="comma-separated batch sizes to test")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="", help="write JSON result here")
    ap.add_argument("--results-json", default="",
                    help="append result JSON line here (for SageMaker collection)")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="render a side-by-side of existing result JSONs and exit")
    args = ap.parse_args()

    if args.compare:
        print(render_compare(args.compare))
        return 0

    if not args.engine:
        ap.error("engine path required (or --compare)")

    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise SystemExit(f"engine not found: {engine_path}")

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    env = collect_env()
    rows, trt_ver = bench_engine(engine_path, batches, args.iters, args.warmup)

    results = {
        "engine": str(engine_path),
        "trt": trt_ver,
        "env": env,
        "rows": rows,
    }
    text = render(results)
    print(text)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    if args.results_json:
        summary = {
            "engine": str(engine_path),
            "gpu": env["gpu"],
            "capability": env["capability"],
            "rows": rows,
        }
        with open(args.results_json, "a") as f:
            f.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
