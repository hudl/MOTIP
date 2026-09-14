"""Validate TRT engine on real hockey frames.

Loads actual JPEG frames from the devbox, runs both PyTorch fp32 and TRT engine,
reports per-query cosine similarity and draws detected boxes for visual comparison.

Usage (via run.sh from MOTIP repo root):
    python trt/validate_real.py \\
        trt/engines/rfdetr_large_1088_motip_ckpt_sm75_fp32.engine \\
        /home/ubuntu/sip-tracking-experiments/_render_92fef/snap_F1*.jpg \\
        --checkpoint s3://... \\
        --topk 20 --score_thresh -3.0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F


# ─── image loading ────────────────────────────────────────────────────────────

def load_frame(path: str, res: int) -> torch.Tensor:
    """Load JPEG/PNG, resize to (res, res), ImageNet-normalize → [1,3,H,W] cuda float32."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((res, res), Image.BILINEAR)
    t = TF.to_tensor(img).unsqueeze(0).cuda()   # [1,3,H,W] in [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    return (t - mean) / std, (orig_w, orig_h)


# ─── model helpers (same as validate_engine.py) ───────────────────────────────

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
    return (built[0] if isinstance(built, (tuple, list)) else built).cuda().eval()


def _patch_motip(model):
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
            query_embeds = hs
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
        e = getattr(trt, name, None)
        if e is not None:
            dtype_map[e] = dt
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


# ─── visualization helper ─────────────────────────────────────────────────────

def draw_boxes(image_path: str, boxes_pt, logits_pt, boxes_trt, logits_trt,
               res: int, topk: int, score_thresh: float, out_path: str):
    """Draw top-K boxes for PT (green) and TRT (red) on the original image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  PIL not available, skipping visualization")
        return

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    draw = ImageDraw.Draw(img)

    def get_top_boxes(boxes, logits, topk, thresh):
        # boxes: [1, 300, 4] cx/cy/w/h normalized, logits: [1, 300, num_classes+1]
        # score = max logit over foreground classes (all but last background class)
        scores = logits[0, :, :-1].max(-1).values  # [300]
        mask = scores > thresh
        if mask.sum() == 0:
            topk_scores, topk_idx = scores.topk(min(topk, 300))
        else:
            sel_scores = scores[mask]
            sel_boxes = boxes[0][mask]
            topk_n = min(topk, sel_scores.shape[0])
            topk_scores, top_sub = sel_scores.topk(topk_n)
            topk_idx = mask.nonzero().squeeze(-1)[top_sub]
        return boxes[0][topk_idx].cpu().numpy(), scores[topk_idx].cpu().numpy()

    def cx_cy_wh_to_xyxy(box, W, H):
        cx, cy, w, h = box
        x1 = (cx - w/2) * W
        y1 = (cy - h/2) * H
        x2 = (cx + w/2) * W
        y2 = (cy + h/2) * H
        return x1, y1, x2, y2

    pt_boxes, pt_scores = get_top_boxes(boxes_pt, logits_pt, topk, score_thresh)
    trt_boxes, trt_scores = get_top_boxes(boxes_trt, logits_trt, topk, score_thresh)

    for i, (box, score) in enumerate(zip(trt_boxes, trt_scores)):
        x1, y1, x2, y2 = cx_cy_wh_to_xyxy(box, orig_w, orig_h)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 12), f"TRT {score:.2f}", fill="red")

    for i, (box, score) in enumerate(zip(pt_boxes, pt_scores)):
        x1, y1, x2, y2 = cx_cy_wh_to_xyxy(box, orig_w, orig_h)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1, y2 + 2), f"PT {score:.2f}", fill="lime")

    img.save(out_path)
    print(f"  visualization saved to {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--variant", default="large")
    ap.add_argument("--res", type=int, default=1088)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--score_thresh", type=float, default=-3.5)
    ap.add_argument("--out", default="trt/trt_real_results.json")
    ap.add_argument("--viz_dir", default="trt/viz")
    args = ap.parse_args()

    import os
    os.makedirs(args.viz_dir, exist_ok=True)

    engine = _load_engine(args.engine)
    import tensorrt as trt
    ctx0 = engine.create_execution_context()
    num_classes_plus1 = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if n == "logits":
            num_classes_plus1 = tuple(ctx0.get_tensor_shape(n))[-1]
    num_classes = (num_classes_plus1 or 5) - 1
    print(f"engine num_classes={num_classes}")

    ck_path = args.checkpoint
    if ck_path.startswith("s3://"):
        import subprocess, hashlib
        cache = "/tmp/trt_ckpt_" + hashlib.md5(ck_path.encode()).hexdigest()[:12] + ".pth"
        if not os.path.exists(cache):
            subprocess.run(["aws", "s3", "cp", ck_path, cache], check=True)
        ck_path = cache

    model = _build_model(args.variant, args.res, num_classes)
    if ck_path:
        _load_checkpoint(model, ck_path)
        print("checkpoint loaded")

    if hasattr(model, "export"):
        model.export()
    elif hasattr(model.backbone, "export"):
        model.backbone.export()
    _patch_motip(model)

    frames_data = []
    all_cosims_topk = []

    # Match the ONNX export's _no_aa patch so DINOv2 pos-encoding interpolation agrees.
    real_interpolate = torch.nn.functional.interpolate
    def _no_aa(*a, **kw):
        kw.pop("antialias", None)
        return real_interpolate(*a, **kw)
    torch.nn.functional.interpolate = _no_aa

    for img_path in args.images:
        x, (orig_w, orig_h) = load_frame(img_path, args.res)
        img_name = Path(img_path).stem

        with torch.no_grad():
            boxes_pt, logits_pt, qe_pt = model.forward_export(x)

        trt_out = _run_trt(engine, x)
        boxes_trt = trt_out["boxes"]
        logits_trt = trt_out["logits"]
        qe_trt = trt_out["query_embeds"]

        # per-query cosim
        pt_norm  = F.normalize(qe_pt,  dim=-1)
        trt_norm = F.normalize(qe_trt, dim=-1)
        per_q_cosim = (pt_norm * trt_norm).sum(dim=-1).squeeze(0)  # [300]

        # top-K detections by PT score
        pt_scores = logits_pt[0, :, :-1].max(-1).values
        topk_idx = pt_scores.topk(args.topk).indices
        topk_cosims = per_q_cosim[topk_idx].tolist()
        topk_scores_pt  = pt_scores[topk_idx].tolist()
        topk_scores_trt = logits_trt[0, :, :-1].max(-1).values[topk_idx].tolist()
        topk_boxes_pt  = boxes_pt[0, topk_idx].tolist()
        topk_boxes_trt = boxes_trt[0, topk_idx].tolist()

        all_cosims_topk.extend(topk_cosims)

        n_detected_pt  = (pt_scores > args.score_thresh).sum().item()
        n_detected_trt = (logits_trt[0, :, :-1].max(-1).values > args.score_thresh).sum().item()

        print(f"  {img_name}: n_detected PT={n_detected_pt} TRT={n_detected_trt} | "
              f"top-{args.topk} cosim mean={sum(topk_cosims)/len(topk_cosims):.4f} "
              f"min={min(topk_cosims):.4f}")

        # visualize
        viz_path = f"{args.viz_dir}/{img_name}_compare.jpg"
        draw_boxes(img_path, boxes_pt, logits_pt, boxes_trt, logits_trt,
                   args.res, args.topk, args.score_thresh, viz_path)

        frames_data.append({
            "image": img_name,
            "n_detected_pt": n_detected_pt,
            "n_detected_trt": n_detected_trt,
            "topk_cosim_mean": round(sum(topk_cosims)/len(topk_cosims), 5),
            "topk_cosim_min": round(min(topk_cosims), 5),
            "topk_scores_pt": [round(s, 4) for s in topk_scores_pt],
            "topk_scores_trt": [round(s, 4) for s in topk_scores_trt],
            "topk_boxes_pt": topk_boxes_pt,
            "topk_boxes_trt": topk_boxes_trt,
            "topk_cosims": [round(c, 5) for c in topk_cosims],
        })

    overall_mean = sum(all_cosims_topk) / len(all_cosims_topk) if all_cosims_topk else 0
    overall_min = min(all_cosims_topk) if all_cosims_topk else 0
    print(f"\nOverall top-{args.topk} query cosim: mean={overall_mean:.5f}  min={overall_min:.5f}")

    import statistics
    if len(all_cosims_topk) > 1:
        print(f"  stdev={statistics.stdev(all_cosims_topk):.5f}")

    if overall_min > 0.99:
        print("\nPASS  top-K cosim > 0.99 -- fp precision drift negligible")
    elif overall_min > 0.95:
        print("\nWARN  top-K cosim 0.95-0.99 -- minor drift, likely fine for ReID")
    else:
        print(f"\nFAIL  top-K cosim {overall_min:.3f} < 0.95")

    Path(args.out).write_text(json.dumps({
        "engine": args.engine, "frames": frames_data,
        "overall_topk_cosim_mean": round(overall_mean, 5),
        "overall_topk_cosim_min": round(overall_min, 5),
    }, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
