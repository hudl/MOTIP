"""RF-DETR -> ONNX -> TensorRT fp16.

Two output modes:
  detector  (default)  boxes + logits -- measures detector ceiling
  --motip              boxes + logits + query_embeds (hs[-1]) -- required for
                       MOTIP inference; without this the engine can detect but
                       the ID head has nothing to do ReID on

Checkpoint formats supported (auto-detected):
  MOTIP stage-2  ck["model"] with detr.base.* keys (e.g. rfdetr_stage2_ckpt4.pth)
  ihc-od / rfdetr Lightning  ck["state_dict"] with model.* keys (e.g. rfdetr_best_ema.pth)
  Raw state dict  bare {param: tensor} mapping

Usage
-----
    python trt/build_engine.py large 1088 --motip --checkpoint /path/to/ckpt.pth
    python trt/build_engine.py large 1088 --motip  # random weights (speed only)
    python trt/build_engine.py large 1088 --batch 4 --out trt/engines/

Run via trt/run.sh on the devbox (sets LD_LIBRARY_PATH for the TRT 8.6 / cuDNN 8
co-existence).  On SageMaker the entrypoint calls this directly.

Engines are GPU-architecture-specific.  The filename embeds the SM capability so
you cannot accidentally load a T4 (sm_75) engine on an A10G (sm_86).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# RF-DETR model builder
# ---------------------------------------------------------------------------

def parse_size(spec: str) -> tuple[int, int]:
    if "x" in spec.lower():
        h, w = spec.lower().split("x")
        return int(h), int(w)
    s = int(spec)
    return s, s


def peek_checkpoint_hparams(path: str) -> dict:
    """Quickly read class count and group_detr from checkpoint before building model."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    raw = ck.get("model", ck.get("state_dict", ck))
    prefix = "detr.base." if any(k.startswith("detr.base.") for k in raw) else (
             "model." if any(k.startswith("model.") for k in raw) else "")
    sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)} if prefix else raw
    out = {}
    if "class_embed.weight" in sd:
        # RF-DETR adds one background class internally, so the linear has num_classes+1 rows
        out["num_classes"] = sd["class_embed.weight"].shape[0] - 1
    if "refpoint_embed.weight" in sd:
        nq = sd["refpoint_embed.weight"].shape[0]
        out["group_detr"] = max(1, nq // 300)
    # Count decoder layers from actual checkpoint weights
    import re as _re
    dec_idxs = {int(_re.search(r'transformer\.decoder\.layers\.(\d+)', k).group(1))
                for k in sd if _re.search(r'transformer\.decoder\.layers\.(\d+)', k)}
    if dec_idxs:
        out["dec_layers"] = max(dec_idxs) + 1
    return out


def build_rfdetr(variant: str, res: int, num_classes: int = 1, group_detr_override: int = 0, dec_layers_override: int = 0):
    from rfdetr import config as rfcfg
    from rfdetr.main import populate_args
    from rfdetr.models import build_model as _build

    table = {
        "nano": rfcfg.RFDETRNanoConfig, "small": rfcfg.RFDETRSmallConfig,
        "medium": rfcfg.RFDETRMediumConfig, "base": rfcfg.RFDETRBaseConfig,
        "large": rfcfg.RFDETRLargeConfig,
    }
    cfg = table[variant](pretrain_weights=None, num_classes=num_classes)
    div = cfg.patch_size * cfg.num_windows
    if res % div != 0:
        raise SystemExit(f"{variant}@{res}: resolution must be divisible by {div}")
    group_detr = group_detr_override if group_detr_override > 0 else cfg.group_detr
    args = populate_args(
        encoder=cfg.encoder, hidden_dim=cfg.hidden_dim, patch_size=cfg.patch_size,
        num_windows=cfg.num_windows, dec_layers=dec_layers_override if dec_layers_override > 0 else cfg.dec_layers,
        sa_nheads=cfg.sa_nheads, ca_nheads=cfg.ca_nheads,
        dec_n_points=cfg.dec_n_points,
        num_queries=getattr(cfg, "num_queries", None) or 300,
        num_select=getattr(cfg, "num_select", None) or 300,
        projector_scale=cfg.projector_scale,
        out_feature_indexes=cfg.out_feature_indexes, two_stage=cfg.two_stage,
        bbox_reparam=cfg.bbox_reparam,
        lite_refpoint_refine=cfg.lite_refpoint_refine, layer_norm=cfg.layer_norm,
        group_detr=group_detr, resolution=res,
        positional_encoding_size=res // cfg.patch_size, ia_bce_loss=cfg.ia_bce_loss,
        num_classes=num_classes, pretrain_weights=None, force_no_pretrain=True,
        device="cuda", amp=False, fp16_eval=True,
    )
    for k, v in {"segmentation_head": False, "mask_downsample_ratio": 4,
                 "mask_point_sample_ratio": 16, "mask_ce_loss_coef": 5.0,
                 "mask_dice_loss_coef": 5.0}.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    built = _build(args)
    model = built[0] if isinstance(built, (tuple, list)) else built
    return model.cuda().eval(), cfg


# ---------------------------------------------------------------------------
# Checkpoint loading (auto-detects format)
# ---------------------------------------------------------------------------

def load_checkpoint(model, path: str) -> str:
    """Load trained weights into the RF-DETR model.  Returns a description of the
    format detected so the caller can print it.

    Handled formats:
      MOTIP stage-2    ck["model"] keys start with "detr.base." -- strip that prefix
      ihc-od Lightning ck["state_dict"] keys start with "model." -- strip that prefix
      raw state dict   bare {param: tensor} -- load directly
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ck, dict):
        raise SystemExit(f"unexpected checkpoint type: {type(ck)}")

    # MOTIP stage-2: ck["model"] has detr.base.* keys
    if "model" in ck and isinstance(ck["model"], dict):
        raw = ck["model"]
        prefix = "detr.base."
        if any(k.startswith(prefix) for k in raw):
            sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
            source = f"MOTIP stage-2 (stripped '{prefix}', {len(sd)} params)"
        else:
            sd = raw
            source = f"model dict ({len(sd)} params)"

    # Lightning checkpoint (ihc-od or rfdetr native): ck["state_dict"] with model.* keys
    elif "state_dict" in ck:
        raw = ck["state_dict"]
        prefix = "model."
        if any(k.startswith(prefix) for k in raw):
            sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        else:
            sd = raw
        source = f"Lightning checkpoint (epoch {ck.get('epoch', '?')}, {len(sd)} params)"

    else:
        # Assume it is already a plain state dict
        sd = ck
        source = f"raw state dict ({len(sd)} params)"

    # Drop keys where tensor shape doesn't match (e.g. position embeddings at a
    # different resolution -- DINOv2 interpolates those itself at runtime).
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k not in model_sd or model_sd[k].shape == v.shape}
    skipped = [k for k in sd if k in model_sd and model_sd[k].shape != sd[k].shape]
    if skipped:
        print(f"  skipping {len(skipped)} shape-mismatched keys: {skipped[:4]}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # _kp_active_mask is an RF-DETR buffer not present in some builds -- silence it
    unexpected = [k for k in unexpected if "_kp_active_mask" not in k]
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    return source


# ---------------------------------------------------------------------------
# MOTIP patch: add hs[-1] (query embeddings) as a third output
# ---------------------------------------------------------------------------

def patch_motip_forward_export(model) -> None:
    """Replace model.forward_export with a version that also returns hs[-1].

    The stock forward_export returns (boxes, logits).  MOTIP's ID head needs
    the last decoder layer's query embeddings -- that is what it does ReID on.
    We replicate the forward_export body exactly, adding one extra return value,
    and bind it to the instance so the class is unchanged.

    Shape: hs[-1] is (B, num_queries, hidden_dim) = (B, 300, 256) for all
    current RF-DETR variants, matching MOTIP's DETR_HIDDEN_DIM=256.
    """

    def motip_forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight
        )
        if hs is not None:
            if self.bbox_reparam:
                delta = self.bbox_embed(hs)
                cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                wh = delta[..., 2:].clamp(-10, 10).exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            query_embeds = hs
        else:
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            query_embeds = hs_enc

        return outputs_coord, outputs_class, query_embeds

    model.forward_export = types.MethodType(motip_forward_export, model)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class _ExportWrapper(torch.nn.Module):
    def __init__(self, model, motip: bool):
        super().__init__()
        self.model = model
        self.motip = motip

    def forward(self, x):
        result = self.model.forward_export(x)
        if self.motip:
            return result[0], result[1], result[2]
        return result[0], result[1]


def _patch_projector_layernorm(model) -> None:
    """projector.LayerNorm.forward uses (x.size(3),) for normalized_shape, which is a
    dynamic tensor value that the ONNX exporter cannot constant-fold.  self.normalized_shape
    is identical and is a plain Python tuple set in __init__ -- use that instead."""
    import torch.nn.functional as _F
    try:
        from rfdetr.models.backbone.projector import LayerNorm as _ProjLN
    except ImportError:
        return

    def _static_forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = _F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

    patched = 0
    for m in model.modules():
        if isinstance(m, _ProjLN):
            m.forward = types.MethodType(_static_forward, m)
            patched += 1
    if patched:
        print(f"  patched {patched} projector LayerNorm(s) for static ONNX export")


def export_onnx(model, hw: tuple[int, int], path: Path, motip: bool,
                batch: int) -> None:
    if hasattr(model, "export"):
        model.export()
    elif hasattr(model.backbone, "export"):
        model.backbone.export()
    else:
        raise SystemExit("no export() hook found on model or backbone")

    _patch_projector_layernorm(model)

    if motip:
        patch_motip_forward_export(model)

    wrapper = _ExportWrapper(model, motip).cuda().eval()
    h, w = hw
    x = torch.randn(batch, 3, h, w, device="cuda")

    real_interpolate = torch.nn.functional.interpolate

    def _no_aa(*args, **kwargs):
        kwargs.pop("antialias", None)
        return real_interpolate(*args, **kwargs)

    output_names = ["boxes", "logits", "query_embeds"] if motip else ["boxes", "logits"]
    dynamic = {"input": {0: "batch"}, "boxes": {0: "batch"}, "logits": {0: "batch"}}
    if motip:
        dynamic["query_embeds"] = {0: "batch"}

    torch.nn.functional.interpolate = _no_aa
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, x, str(path),
                opset_version=int(os.environ.get("BENCH_OPSET", "17")),
                input_names=["input"],
                output_names=output_names,
                dynamic_axes=dynamic,
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        torch.nn.functional.interpolate = real_interpolate


# ---------------------------------------------------------------------------
# TRT engine build + benchmark
# ---------------------------------------------------------------------------

def _trt_major() -> int:
    import tensorrt as trt
    vi = getattr(trt, "version_info", None)
    if vi is not None:
        return int(vi[0])
    v = getattr(trt, "__version__", "8.0.0")
    return int(str(v).split(".")[0])


def build_engine(onnx_path: Path, engine_path: Path, workspace_mb: int = 4096,
                  batch: int = 1, fp32_decoder: bool = False, no_fp16: bool = False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    major = _trt_major()
    if major >= 10:
        network = builder.create_network()
    else:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  onnx parse error:", parser.get_error(i))
            raise SystemExit("ONNX -> TRT parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if not no_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Mixed-precision: backbone (DINOv2 ViT) stays fully in fp16 for speed;
    # post-backbone layers (projector, encoder, decoder) have their LayerNorm
    # and Softmax ops pinned to fp32 to avoid the precision collapse that flat
    # fp16 causes in deformable-attention transformer blocks.
    _fp32_types = {trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX}
    _last_backbone = max(
        (_i for _i in range(network.num_layers)
         if "/backbone/" in network.get_layer(_i).name),
        default=-1,
    )
    _n_fp32 = 0
    for _i in range(network.num_layers):
        _layer = network.get_layer(_i)
        is_post = _i > _last_backbone
        _skip_types = {trt.LayerType.SHAPE, trt.LayerType.CONSTANT}
        if is_post and (fp32_decoder or _layer.type in _fp32_types) and _layer.type not in _skip_types:
            try:
                _layer.precision = trt.DataType.FLOAT
                for _j in range(_layer.num_outputs):
                    _layer.set_output_type(_j, trt.DataType.FLOAT)
                _n_fp32 += 1
            except Exception:
                pass
    if _n_fp32:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        mode = "ALL post-backbone" if fp32_decoder else "NORM/SOFTMAX post-backbone"
        print(f"  mixed-precision: backbone fp16 (last at layer {_last_backbone}), "
              f"forced {_n_fp32} {mode} layers to fp32")

    # Optimization profile required for any dynamic-shape network
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = [batch if d == -1 else d for d in t.shape]
        profile.set_shape(t.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise SystemExit("engine build returned None")
    blob = bytes(serialized)
    engine_path.write_bytes(blob)
    return blob, build_s


def benchmark_engine(blob: bytes, iters: int = 50, warmup: int = 20) -> tuple[list[float], list]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    ctx = engine.create_execution_context()

    dtype_map = {}
    for name, dt in (("float32", torch.float32), ("float16", torch.float16),
                     ("int32", torch.int32), ("int64", torch.int64),
                     ("int8", torch.int8), ("bool", torch.bool),
                     ("bfloat16", torch.bfloat16)):
        enum = getattr(trt, name, None)
        if enum is not None:
            dtype_map[enum] = dt

    tensors, io_meta = {}, []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(ctx.get_tensor_shape(name))
        dtype = dtype_map[engine.get_tensor_dtype(name)]
        t = torch.zeros(shape, dtype=dtype, device="cuda")
        tensors[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
        io_meta.append((name, list(shape), str(dtype)))

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
    return samples, io_meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", help="nano|small|medium|base|large")
    ap.add_argument("res", help="square resolution (e.g. 1088) or HxW")
    ap.add_argument("--motip", action="store_true",
                    help="add hs[-1] (query_embeds) as third output")
    ap.add_argument("--checkpoint", default="",
                    help="path to .pth/.ckpt file; auto-detects MOTIP stage-2 and "                         "Lightning formats.  Omit for random weights (speed only).")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="trt/engines")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--no-fp16", action="store_true",
                    help="build pure fp32 engine (baseline for ID head quality check)")
    ap.add_argument("--fp32-decoder", action="store_true",
                    help="pin ALL post-backbone layers to fp32 (better query_embeds for MOTIP)")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--results-json", default="")
    args = ap.parse_args()

    h, w = parse_size(args.res)
    res = max(h, w)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tag = f"{h}x{w}" if h != w else str(res)
    mode_tag = "motip" if args.motip else "det"
    batch_tag = f"_b{args.batch}" if args.batch > 1 else ""
    sm = "sm" + "".join(str(x) for x in torch.cuda.get_device_capability())
    ckpt_tag = "_ckpt" if args.checkpoint else "_rand"
    onnx_path = out / f"rfdetr_{args.variant}_{tag}_{mode_tag}{batch_tag}{ckpt_tag}.onnx"
    dec_tag = "_fp32dec" if getattr(args, 'fp32_decoder', False) else ""
    prec_tag = "_fp32" if getattr(args, 'no_fp16', False) else "_fp16"
    engine_path = out / f"rfdetr_{args.variant}_{tag}_{mode_tag}{batch_tag}{ckpt_tag}{dec_tag}_{sm}{prec_tag}.engine"

    import tensorrt as trt
    trt_ver = getattr(trt, "__version__", "unknown")
    print(f"TensorRT {trt_ver}  GPU: {torch.cuda.get_device_name(0)}  sm_{sm[2:]}")

    # Peek checkpoint first so we can build the model with matching hparams
    ck_path = args.checkpoint
    if ck_path and ck_path.startswith("s3://"):
        import subprocess, hashlib
        cache_name = hashlib.md5(ck_path.encode()).hexdigest()[:12] + ".pth"
        local = f"/tmp/trt_ckpt_{cache_name}"
        if not __import__("os").path.exists(local):
            print(f"downloading {ck_path} ...")
            subprocess.run(["aws", "s3", "cp", ck_path, local], check=True)
        else:
            print(f"using cached checkpoint: {local}")
        ck_path = local

    ck_hparams = {}
    if ck_path:
        ck_hparams = peek_checkpoint_hparams(ck_path)
        print(f"checkpoint hparams: {ck_hparams}")

    num_classes = ck_hparams.get("num_classes", 1)
    group_detr_override = ck_hparams.get("group_detr", 0)
    dec_layers_override = ck_hparams.get("dec_layers", 0)

    model, cfg = build_rfdetr(args.variant, res, num_classes=num_classes,
                               group_detr_override=group_detr_override,
                               dec_layers_override=dec_layers_override)
    print(f"built {args.variant}@{res}  dec={cfg.dec_layers} hidden={cfg.hidden_dim}  "          f"num_classes={num_classes}  group_detr={group_detr_override or cfg.group_detr}")

    if ck_path:
        fmt = load_checkpoint(model, ck_path)
        print(f"checkpoint loaded: {fmt}")
    else:
        print("no checkpoint -- using random weights (speed ceiling only)")

    if not onnx_path.exists():
        print(f"exporting onnx -> {onnx_path} ...")
        export_onnx(model, (h, w), onnx_path, args.motip, args.batch)
        print(f"  {onnx_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"onnx exists: {onnx_path}")

    del model
    torch.cuda.empty_cache()

    print(f"building fp16 engine -> {engine_path} ...")
    blob, build_s = build_engine(onnx_path, engine_path, args.workspace_mb, args.batch,
                              fp32_decoder=getattr(args, 'fp32_decoder', False),
                              no_fp16=getattr(args, 'no_fp16', False))
    print(f"  {engine_path.stat().st_size / 1e6:.1f} MB  built in {build_s:.0f}s")

    result = {
        "variant": args.variant, "res": tag, "mpx": round(h * w / 1e6, 3),
        "motip": args.motip, "batch": args.batch, "checkpoint": bool(args.checkpoint),
        "sm": sm, "gpu": torch.cuda.get_device_name(0),
        "trt": trt_ver, "build_seconds": round(build_s, 1),
        "engine_mb": round(len(blob) / 1e6, 1),
        "engine_path": str(engine_path),
    }

    if not args.skip_bench:
        print("benchmarking ...")
        samples, io_meta = benchmark_engine(blob, args.iters, args.warmup)
        med, lo = statistics.median(samples), min(samples)
        result.update({
            "median_ms": round(med, 2), "min_ms": round(lo, 2),
            "median_fps": round(1000.0 / med, 2), "min_fps": round(1000.0 / lo, 2),
            "io": io_meta,
        })
        print(f"  median {med:.2f} ms  min {lo:.2f} ms  ({result['min_fps']:.1f} fps at min)")

    print("RESULT " + json.dumps(result))
    if args.results_json:
        with open(args.results_json, "a") as f:
            f.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
