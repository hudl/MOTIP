"""TensorRT detector wrapper for MOTIP inference.

Provides TRTDetectorWrapper — a drop-in replacement for model.detr that runs the
RF-DETR detector via a TRT engine and returns the same dict MOTIP expects:
  {"pred_logits": ..., "pred_boxes": ..., "outputs": query_embeds}

Usage
-----
    from trt.trt_wrapper import load_engine, TRTDetectorWrapper

    engine = load_engine("trt/engines/rfdetr_small_576_motip_ckpt_sm75_fp16.engine")
    object.__setattr__(model, "detr", TRTDetectorWrapper(engine, res=576))

The engine must have been built with --motip (three outputs: boxes, logits, query_embeds).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

# TRT 8.6 bindings location — checked in priority order.
# On the host: rfdetr-bench isolated venv (has both libs and bindings).
# In the devcontainer: packages are copied into the system dist-packages, and
# cuDNN 8 is at the system path; only libnvinfer needs explicit preloading.
_TRT_SEARCH_PATHS = [
    # host venv path
    "/home/ubuntu/experiments/rfdetr-bench/.venv/lib/python3.10/site-packages",
    # devcontainer system dist-packages
    "/usr/local/lib/python3.10/dist-packages",
]
_CUDNN8_CANDIDATES = [
    "/home/ubuntu/experiments/cudnn8/nvidia/cudnn/lib/libcudnn.so.8",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.8",
]


def _ensure_trt_importable() -> None:
    import sys
    import ctypes
    from pathlib import Path as _Path

    # Already set up
    if any("tensorrt_bindings" in p for p in sys.path):
        return

    # Find which search path has the packages
    sp = None
    for candidate in _TRT_SEARCH_PATHS:
        if _Path(f"{candidate}/tensorrt_bindings").exists():
            sp = candidate
            break
    if sp is None:
        raise RuntimeError("TRT packages not found — run: copy tensorrt_bindings/libs into the container")

    # Preload cuDNN 8 first, then libnvinfer
    for cudnn_path in _CUDNN8_CANDIDATES:
        if _Path(cudnn_path).exists():
            ctypes.CDLL(cudnn_path, mode=ctypes.RTLD_GLOBAL)
            break

    for lib in ["libnvinfer.so.8", "libnvinfer_plugin.so.8"]:
        lib_path = f"{sp}/tensorrt_libs/{lib}"
        if _Path(lib_path).exists():
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

    sys.path.insert(0, f"{sp}/tensorrt_bindings")


def load_engine(path: str | Path):
    """Deserialise a TRT engine file and return a CUDA engine object."""
    _ensure_trt_importable()
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(Path(path).read_bytes())


class TRTDetectorWrapper:
    """Cached-context TRT wrapper for MOTIP.

    Creates execution context and output buffers once at init, then reuses them
    on every forward pass to avoid per-frame allocation overhead.

    Parameters
    ----------
    engine
        Deserialised TRT CUDA engine (from load_engine()).
    res
        Square input resolution the engine was built for (e.g. 576).
    """

    def __init__(self, engine, res: int = 576) -> None:
        _ensure_trt_importable()
        import tensorrt as trt

        self.res = res
        self._ctx = engine.create_execution_context()

        dtype_map = {
            getattr(trt, n): t
            for n, t in (
                ("float32", torch.float32),
                ("float16", torch.float16),
                ("int32", torch.int32),
                ("int64", torch.int64),
            )
            if hasattr(trt, n)
        }

        self._tensors: dict[str, torch.Tensor] = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(self._ctx.get_tensor_shape(name))
            dtype = dtype_map[engine.get_tensor_dtype(name)]
            self._tensors[name] = torch.zeros(shape, dtype=dtype, device="cuda")
            self._ctx.set_tensor_address(name, self._tensors[name].data_ptr())

        self._stream = torch.cuda.Stream()

    def __call__(self, samples) -> dict[str, torch.Tensor]:
        img = samples.tensors.to("cuda", dtype=torch.float32)
        if img.shape[2] != self.res or img.shape[3] != self.res:
            img = F.interpolate(img, size=(self.res, self.res),
                                mode="bilinear", align_corners=False)

        inp = self._tensors["input"]
        inp.copy_(img.half() if inp.dtype == torch.float16 else img)

        self._ctx.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        # logits/boxes → float32 for post-processing; query_embeds keeps the
        # engine's native dtype (float16 for fp16 engines) so it matches the
        # float16 trajectory feature buffer in RuntimeTracker.
        return {
            "pred_logits": self._tensors["logits"].float(),
            "pred_boxes":  self._tensors["boxes"].float(),
            "outputs":     self._tensors["query_embeds"].float(),
        }

    def eval(self) -> "TRTDetectorWrapper":
        return self

    def train(self, mode: bool = True) -> "TRTDetectorWrapper":
        return self

    def parameters(self):
        return iter([])
