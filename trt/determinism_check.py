"""Check TRT engine determinism: run same input N times, compare outputs."""
import sys, json, argparse
from pathlib import Path
import torch

def _load_engine(path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(Path(path).read_bytes())

def _run_trt(engine, x):
    import tensorrt as trt
    ctx = engine.create_execution_context()
    dtype_map = {}
    for name, dt in (("float32", torch.float32), ("float16", torch.float16),
                     ("int32", torch.int32), ("int64", torch.int64)):
        enum = getattr(trt, name, None)
        if enum is not None:
            dtype_map[enum] = dt
    tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(ctx.get_tensor_shape(name))
        dtype = dtype_map[engine.get_tensor_dtype(name)]
        tensors[name] = torch.zeros(shape, dtype=dtype, device="cuda")
        ctx.set_tensor_address(name, tensors[name].data_ptr())
    inp = tensors["input"]
    inp.copy_(x.half() if inp.dtype == torch.float16 else x)
    ctx.set_input_shape("input", list(x.shape))
    stream = torch.cuda.Stream()
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {k: v.float() for k, v in tensors.items() if k != "input"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--res", type=int, default=1088)
    args = ap.parse_args()

    engine = _load_engine(args.engine)
    print(f"Engine loaded: {args.engine}")

    # Test with 3 different fixed inputs
    torch.manual_seed(42)
    inputs = []
    for _ in range(3):
        raw = torch.rand(1, 3, args.res, args.res, device="cuda")
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        inputs.append((raw - mean) / std)

    all_pass = True
    for inp_idx, x in enumerate(inputs):
        first = None
        for run in range(args.runs):
            out = _run_trt(engine, x)
            if first is None:
                first = {k: v.clone() for k, v in out.items()}
                continue
            for k, v in out.items():
                diff = (first[k] - v).abs().max().item()
                if diff > 0:
                    print(f"  input {inp_idx} run {run}: {k} max_diff={diff:.2e}  NON-DETERMINISTIC")
                    all_pass = False
                else:
                    print(f"  input {inp_idx} run {run}: {k} EXACT match")

    if all_pass:
        print("\nPASS: engine is fully deterministic across all runs")
    else:
        print("\nFAIL: non-determinism detected")

if __name__ == "__main__":
    main()
