"""Compare TRT fp16 engine outputs against PyTorch fp32 reference.

Usage
-----
    python trt/validate_engine.py \\
        trt/engines/rfdetr_large_1088_motip_ckpt_sm75_fp16.engine \\
        --checkpoint /path/or/s3/ckpt.pth \\
        [--n 8] [--res 1088] [--variant large] [--out results.json]

Run via trt/run.sh on the devbox.
"""

from __future__ import annotations
import argparse, json, sys, statistics
from pathlib import Path
import torch
import torch.nn.functional as F


def _load_engine(path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(Path(path).read_bytes())


def _run_trt(engine, x: torch.Tensor):
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


def _build_model(variant, res, num_classes, group_detr=1):
    from rfdetr import config as rfcfg
    from rfdetr.main import populate_args
    from rfdetr.models import build_model as _build

    table = {
        "nano": rfcfg.RFDETRNanoConfig, "small": rfcfg.RFDETRSmallConfig,
        "medium": rfcfg.RFDETRMediumConfig, "base": rfcfg.RFDETRBaseConfig,
        "large": rfcfg.RFDETRLargeConfig,
    }
    cfg = table[variant](pretrain_weights=None, num_classes=num_classes)
    args = populate_args(
        encoder=cfg.encoder, hidden_dim=cfg.hidden_dim, patch_size=cfg.patch_size,
        num_windows=cfg.num_windows, dec_layers=cfg.dec_layers,
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
    return model.cuda().eval()


def _patch_motip_forward_export(model):
    """Apply MOTIP patch AFTER model.export() so it isn't overwritten."""
    import types

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
                wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            query_embeds = hs  # [B, 300, 256] in forward_export mode
        else:
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            query_embeds = hs_enc
        return outputs_coord, outputs_class, query_embeds

    model.forward_export = types.MethodType(motip_forward_export, model)


def _load_checkpoint(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    raw = ck.get("model", ck.get("state_dict", ck))
    prefix = "detr.base." if any(k.startswith("detr.base.") for k in raw) else (
             "model." if any(k.startswith("model.") for k in raw) else "")
    sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)} if prefix else raw
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k not in model_sd or model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)


def _make_input(res: int) -> torch.Tensor:
    """Uniform random pixels in [0,1] normalised with ImageNet stats.

    Values end up roughly in [-2.1, 2.6] — safe for fp16 (max ~65504).
    """
    raw = torch.rand(1, 3, res, res, device="cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    return (raw - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", help="path to .engine file")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--variant", default="large")
    ap.add_argument("--res", type=int, default=1088)
    ap.add_argument("--n", type=int, default=8, help="number of frames to test")
    ap.add_argument("--out", default="", help="save JSON results to this path")
    args = ap.parse_args()

    engine = _load_engine(args.engine)

    import tensorrt as trt
    ctx0 = engine.create_execution_context()
    num_classes_plus1 = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name == "logits":
            num_classes_plus1 = tuple(ctx0.get_tensor_shape(name))[-1]
            break
    num_classes = (num_classes_plus1 or 5) - 1
    print(f"engine num_classes={num_classes}")

    ck_path = args.checkpoint
    if ck_path.startswith("s3://"):
        import subprocess, hashlib, os
        cache = "/tmp/trt_ckpt_" + hashlib.md5(ck_path.encode()).hexdigest()[:12] + ".pth"
        if not os.path.exists(cache):
            subprocess.run(["aws", "s3", "cp", ck_path, cache], check=True)
        ck_path = cache

    model = _build_model(args.variant, args.res, num_classes)
    if ck_path:
        _load_checkpoint(model, ck_path)
        print("checkpoint loaded")

    # export() first — it may reset forward_export; patch afterwards
    if hasattr(model, "export"):
        model.export()
    elif hasattr(model.backbone, "export"):
        model.backbone.export()

    _patch_motip_forward_export(model)

    box_diffs, logit_diffs, cosims = [], [], []
    frames_data = []
    torch.manual_seed(42)

    # Strip antialias from F.interpolate — ONNX was exported with this patch applied;
    # without it here, DINOv2's interpolate_pos_encoding differs between PT and TRT.
    real_interpolate = torch.nn.functional.interpolate
    def _no_aa(*a, **kw):
        kw.pop("antialias", None)
        return real_interpolate(*a, **kw)
    torch.nn.functional.interpolate = _no_aa

    for i in range(args.n):
        x = _make_input(args.res)

        with torch.no_grad():
            boxes_pt, logits_pt, qe_pt = model.forward_export(x)

        trt_out = _run_trt(engine, x)
        boxes_trt = trt_out["boxes"]
        logits_trt = trt_out["logits"]
        qe_trt = trt_out["query_embeds"]

        # check for NaN/inf
        for label, t in [("pt_boxes", boxes_pt), ("pt_logits", logits_pt),
                         ("pt_qe", qe_pt), ("trt_boxes", boxes_trt),
                         ("trt_logits", logits_trt), ("trt_qe", qe_trt)]:
            if not torch.isfinite(t).all():
                print(f"  WARNING frame {i}: {label} has NaN/inf "
                      f"(nan={t.isnan().sum()}, inf={t.isinf().sum()})")

        box_diff = (boxes_pt - boxes_trt).abs().max().item()
        logit_diff = (logits_pt - logits_trt).abs().max().item()
        box_diffs.append(box_diff)
        logit_diffs.append(logit_diff)

        pt_norm  = F.normalize(qe_pt,  dim=-1)   # [1, 300, 256]
        trt_norm = F.normalize(qe_trt, dim=-1)
        per_query_cosim = (pt_norm * trt_norm).sum(dim=-1).squeeze(0)  # [300]
        frame_cosim = per_query_cosim.mean().item()
        cosims.append(frame_cosim)

        # top-5 boxes by max logit (before softmax) for a quick sanity check
        top_scores_pt,  top_idx_pt  = logits_pt[0,  :, :-1].max(-1)
        top_scores_trt, top_idx_trt = logits_trt[0, :, :-1].max(-1)
        top5_pt  = top_scores_pt.topk(5)
        top5_trt = top_scores_trt.topk(5)

        frame_rec = {
            "frame": i,
            "box_max_diff": round(box_diff, 6),
            "logit_max_diff": round(logit_diff, 6),
            "query_cosim_mean": round(frame_cosim, 6),
            "query_cosim_min": round(per_query_cosim.min().item(), 6),
            "query_cosim_hist": [round(v, 4) for v in per_query_cosim.tolist()],
            "top5_boxes_pt":  boxes_pt[0, top5_pt.indices].tolist(),
            "top5_boxes_trt": boxes_trt[0, top5_trt.indices].tolist(),
            "top5_scores_pt":  [round(v, 4) for v in top5_pt.values.tolist()],
            "top5_scores_trt": [round(v, 4) for v in top5_trt.values.tolist()],
            "logit_pt_flat100":  [round(v, 4) for v in logits_pt[0,  :100, :].flatten().tolist()],
            "logit_trt_flat100": [round(v, 4) for v in logits_trt[0, :100, :].flatten().tolist()],
            "box_pt_flat100":    [round(v, 4) for v in boxes_pt[0,  :100, :].flatten().tolist()],
            "box_trt_flat100":   [round(v, 4) for v in boxes_trt[0, :100, :].flatten().tolist()],
        }
        frames_data.append(frame_rec)

        print(f"  frame {i}: box_diff={box_diff:.4f}  logit_diff={logit_diff:.4f}  "
              f"qe_cosim={frame_cosim:.4f}  "
              f"qe_min={per_query_cosim.min():.4f}  qe_max={per_query_cosim.max():.4f}")

    print(f"\nResults over {args.n} frames:")
    print(f"  boxes        max_abs_diff  mean={statistics.mean(box_diffs):.5f}  max={max(box_diffs):.5f}")
    print(f"  logits       max_abs_diff  mean={statistics.mean(logit_diffs):.5f}  max={max(logit_diffs):.5f}")
    print(f"  query_embeds cosine_sim    mean={statistics.mean(cosims):.5f}  min={min(cosims):.5f}")

    summary = {
        "engine": str(args.engine),
        "n_frames": args.n,
        "res": args.res,
        "variant": args.variant,
        "num_classes": num_classes,
        "summary": {
            "box_diff_mean": round(statistics.mean(box_diffs), 6),
            "box_diff_max": round(max(box_diffs), 6),
            "logit_diff_mean": round(statistics.mean(logit_diffs), 6),
            "logit_diff_max": round(max(logit_diffs), 6),
            "cosim_mean": round(statistics.mean(cosims), 6),
            "cosim_min": round(min(cosims), 6),
        },
        "frames": frames_data,
    }

    out_path = args.out or "trt_validate_results.json"
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to: {out_path}")

    if min(cosims) > 0.99:
        print("\nPASS  query_embeds cosine_sim > 0.99 -- fp16 drift negligible for ReID")
    elif min(cosims) > 0.95:
        print("\nWARN  query_embeds cosine_sim 0.95-0.99 -- minor fp16 drift, likely fine")
    else:
        print("\nFAIL  query_embeds cosine_sim < 0.95 -- fp16 precision loss too large")


if __name__ == "__main__":
    sys.exit(main())
