"""Check if a single warmup run stabilises the fp16 engine's bifurcation."""
import sys, torch, torch.nn.functional as F
from pathlib import Path

def _load_engine(path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    return trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())

def _run(engine, x):
    import tensorrt as trt
    ctx = engine.create_execution_context()
    dtype_map = {}
    for name, dt in (("float32",torch.float32),("float16",torch.float16),
                     ("int32",torch.int32),("int64",torch.int64)):
        e = getattr(trt, name, None)
        if e: dtype_map[e] = dt
    tensors = {}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        s = tuple(ctx.get_tensor_shape(n))
        d = dtype_map[engine.get_tensor_dtype(n)]
        tensors[n] = torch.zeros(s, dtype=d, device="cuda")
        ctx.set_tensor_address(n, tensors[n].data_ptr())
    tensors["input"].copy_(x.half() if tensors["input"].dtype==torch.float16 else x)
    ctx.set_input_shape("input", list(x.shape))
    stream = torch.cuda.Stream()
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {k: v.float() for k,v in tensors.items() if k!="input"}

path = sys.argv[1]
engine = _load_engine(path)
print(f"Loaded: {path}")

torch.manual_seed(42)
def make_input():
    raw = torch.rand(1,3,1088,1088,device="cuda")
    mean = torch.tensor([0.485,0.456,0.406],device="cuda").view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225],device="cuda").view(1,3,1,1)
    return (raw-mean)/std

# dummy warmup with zeros
dummy = torch.zeros(1,3,1088,1088,device="cuda")
_run(engine, dummy)
print("Warmup done with zero input.")

inputs = [make_input() for _ in range(3)]
for inp_i, x in enumerate(inputs):
    first = None
    for run in range(5):
        out = _run(engine, x)
        if first is None:
            first = {k: v.clone() for k,v in out.items()}
            continue
        for k,v in out.items():
            diff = (first[k]-v).abs().max().item()
            status = "EXACT" if diff==0 else f"diff={diff:.2e}"
            print(f"  input {inp_i} run {run}: {k} {status}")

