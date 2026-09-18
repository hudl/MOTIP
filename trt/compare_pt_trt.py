"""
Compare PyTorch vs TRT MOTIP tracking: speed, detection overlap, side-by-side video.

Runs both engines on the same frames, measures per-frame latency, renders a
split-screen mp4, and prints a timing / fidelity summary table.

Usage (from MOTIP repo root via run.sh):
    ./trt/run.sh compare_pt_trt \\
        trt/engines/rfdetr_large_1088_motip_ckpt_sm75_fp16.engine \\
        /home/ubuntu/trt_test_frames \\
        --checkpoint /tmp/trt_ckpt_cab8f975e423.pth \\
        [--n_frames 300] [--warmup 5] [--out /tmp/pt_vs_trt.mp4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

MOTIP_ROOT = Path(os.environ.get("MOTIP_ROOT", Path(__file__).resolve().parent.parent))
for _p in (str(MOTIP_ROOT / "models" / "ops"), str(MOTIP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_prebuilt = MOTIP_ROOT / "prebuilt_ops"
if _prebuilt.is_dir() and str(_prebuilt) not in sys.path:
    sys.path.insert(0, str(_prebuilt))


# ── antialias shim (matches ONNX export) ─────────────────────────────────────
_real_interp = torch.nn.functional.interpolate
def _no_aa(*a, **kw):
    kw.pop("antialias", None)
    return _real_interp(*a, **kw)
torch.nn.functional.interpolate = _no_aa


# ── TRT helpers ───────────────────────────────────────────────────────────────
from trt.trt_wrapper import load_engine as _load_engine, TRTDetectorWrapper


# ── NestedTensor (real MOTIP class, has .decompose()) ────────────────────────
def _NestedTensor(tensors):
    from utils.nested_tensor import NestedTensor
    mask = torch.zeros(tensors.shape[0], tensors.shape[2], tensors.shape[3],
                       dtype=torch.bool, device=tensors.device)
    return NestedTensor(tensors=tensors, mask=mask)


# ── model building ────────────────────────────────────────────────────────────

def _motip_cfg():
    from configs.util import load_super_config
    from utils.misc import yaml_to_dict
    cfg = yaml_to_dict(str(MOTIP_ROOT / "configs" / "rfdetr_motip_hockey_smoketest.yaml"))
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    cfg.update({
        "DEVICE": "cuda",
        "INFERENCE_MODE": "evaluate",
        "ONLY_DETR": False,
        "RFDETR_GROUP_DETR": 1,  # checkpoint has [300,4] refpoints; group_detr=13 from yaml skips this key
    })
    return cfg


def _build_motip(checkpoint_path: str, num_classes: int = 3):
    """Build MOTIP for TRT side: loads only trajectory/id_decoder weights (DETR replaced by engine)."""
    from models.motip import build as build_motip
    cfg = _motip_cfg()
    cfg["NUM_CLASSES"] = num_classes
    model, _ = build_motip(config=cfg)
    model = model.cuda().eval()
    if checkpoint_path:
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw = ck.get("model", ck.get("state_dict", ck))
        model_sd = model.state_dict()
        filtered = {k: v for k, v in raw.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"  loaded {len(filtered)}/{len(model_sd)} tensors (TRT path)")
    return model, cfg


def _build_motip_pt(checkpoint_path: str, num_classes: int = 3):
    """Build MOTIP for PT side: loads backbone weights properly via build_engine's loader."""
    # build_engine.load_checkpoint strips the 'detr.base.' prefix so it matches
    # the raw RF-DETR state dict (model.detr.base).
    sys.path.insert(0, str(MOTIP_ROOT / "trt"))
    from build_engine import load_checkpoint as trt_load_checkpoint
    from models.motip import build as build_motip

    cfg = _motip_cfg()
    cfg["NUM_CLASSES"] = num_classes
    model, _ = build_motip(config=cfg)
    model = model.cuda().eval()

    if checkpoint_path:
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw = ck.get("model", ck.get("state_dict", ck))

        # 1. Load backbone + transformer into model.detr.base (rfdetr model)
        src = trt_load_checkpoint(model.detr.base, checkpoint_path)
        print(f"  backbone/transformer loaded ({src})")

        # 2. Load trajectory_modeling + id_decoder from the same checkpoint
        model_sd = model.state_dict()
        id_keys = {k: v for k, v in raw.items()
                   if not k.startswith("detr.base.") and
                   k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(id_keys, strict=False)
        print(f"  ID head loaded ({len(id_keys)} tensors)")

    return model, cfg


# ── frame loading ─────────────────────────────────────────────────────────────
def _load_frame(path: Path, res: int) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(path).convert("RGB")
    img = img.resize((res, res), Image.BILINEAR)
    t = TF.to_tensor(img).unsqueeze(0).cuda()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    return (t - mean) / std


# ── cv2 renderer ──────────────────────────────────────────────────────────────
_PALETTE_BGR = [
    (233, 180,  86), (0,  159, 230), (115, 158,   0), (167, 121, 204),
    (178, 114,   0), (0,   94, 213), ( 66, 228, 240), (255, 128,   0),
    (0,  200, 100), (200,   0, 100), (100,   0, 200), (0,  100, 200),
    (200, 100,   0), (50,  200,  50), (200,  50,  50), (50,  50, 200),
    (180, 180,   0), (0,  180, 180), (180,   0, 180), (128, 128, 128),
]

def _tid_color(tid: int):
    return _PALETTE_BGR[int(tid) % len(_PALETTE_BGR)]


def _render_frame(frame_bgr: np.ndarray, result: dict, label: str,
                  elapsed_ms: float) -> np.ndarray:
    """Draw bboxes+IDs on frame, add HUD. Returns a copy."""
    import cv2
    img = frame_bgr.copy()
    if result:
        ids    = result["id"].tolist()
        bboxes = result["bbox"].tolist()   # [x, y, w, h] pixels
        scores = result["score"].tolist()
        for tid, (x, y, w, h), s in zip(ids, bboxes, scores):
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            color = _tid_color(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            lbl = f"{tid}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(y1-th-4, 0)), (x1+tw+4, y1), color, -1)
            cv2.putText(img, lbl, (x1+2, max(y1-3, th)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD bar
    n = len(result["id"]) if result else 0
    hud = f"{label}  {elapsed_ms:.1f}ms  {n} tracks"
    (hw, hh), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (0, 0), (hw + 12, hh + 10), (20, 20, 20), -1)
    cv2.putText(img, hud, (6, hh + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def _make_side_by_side(left: np.ndarray, right: np.ndarray,
                       frame_idx: int, total: int) -> np.ndarray:
    import cv2
    h, w = left.shape[:2]
    div = np.full((h, 4, 3), 40, dtype=np.uint8)
    out = np.concatenate([left, div, right], axis=1)
    # Frame counter
    lbl = f"frame {frame_idx+1}/{total}"
    cv2.putText(out, lbl, (w + 6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine",       help="TRT .engine file")
    ap.add_argument("frame_dir",    help="directory of JPEG/PNG frames")
    ap.add_argument("--checkpoint", default="/tmp/trt_ckpt_cab8f975e423.pth")
    ap.add_argument("--num_classes",  type=int,   default=3)
    ap.add_argument("--det_thresh",   type=float, default=0.5)
    ap.add_argument("--newborn_thresh", type=float, default=0.5)
    ap.add_argument("--res",          type=int,   default=576)
    ap.add_argument("--n_frames",     type=int,   default=300,
                    help="frames to process (loops dir if needed)")
    ap.add_argument("--warmup",       type=int,   default=5,
                    help="warmup frames excluded from timing stats")
    ap.add_argument("--out",          default="/tmp/pt_vs_trt.mp4",
                    help="output mp4 path")
    ap.add_argument("--no_video",     action="store_true",
                    help="skip video render, print stats only")
    args = ap.parse_args()

    from models.runtime_tracker import RuntimeTracker
    import cv2

    # ── load frames (loop if needed) ─────────────────────────────────────────
    raw_paths = sorted(Path(args.frame_dir).glob("*.jpg")) + \
                sorted(Path(args.frame_dir).glob("*.png"))
    if not raw_paths:
        print(f"ERROR: no jpg/png frames in {args.frame_dir}"); return 1
    frame_paths = []
    while len(frame_paths) < args.n_frames:
        frame_paths.extend(raw_paths)
    frame_paths = frame_paths[:args.n_frames]
    print(f"Using {len(frame_paths)} frames "
          f"({'looped ' if len(frame_paths) > len(raw_paths) else ''}"
          f"from {len(raw_paths)} unique)")

    # frames are loaded on-demand per-frame to avoid OOM on long sequences

    def _make_tracker(model, cfg):
        return RuntimeTracker(
            model=model, sequence_hw=(args.res, args.res),
            det_thresh=args.det_thresh, newborn_thresh=args.newborn_thresh,
            id_thresh=cfg.get("ID_THRESH", 0.1),
            miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
            max_tracks=cfg.get("MAX_TRACKS", 0),
            area_thresh=cfg.get("AREA_THRESH", 100),
        )

    # ── build PT model (proper backbone loading) ──────────────────────────────
    print("\n[1/2] Building PyTorch model (full backbone + ID head)...")
    model_pt, cfg = _build_motip_pt(args.checkpoint, args.num_classes)

    # ── build TRT model ───────────────────────────────────────────────────────
    print("[2/2] Loading TRT engine...")
    engine = _load_engine(args.engine)
    model_trt, _ = _build_motip(args.checkpoint, args.num_classes)
    object.__setattr__(model_trt, "detr", TRTDetectorWrapper(engine, res=args.res))

    # ── warmup both (builds CUDA graphs / JIT) ────────────────────────────────
    print(f"\nWarmup ({args.warmup} frames)...")
    _wpt  = _make_tracker(model_pt,  cfg)
    _wtrt = _make_tracker(model_trt, cfg)
    with torch.no_grad():
        for i in range(min(args.warmup, len(frame_paths))):
            _nt = _NestedTensor(_load_frame(frame_paths[i], args.res))
            _wpt.update(_nt)
            _wtrt.update(_nt)
    del _wpt, _wtrt

    # Fresh trackers after warmup (clean state)
    tracker_pt  = _make_tracker(model_pt,  cfg)
    tracker_trt = _make_tracker(model_trt, cfg)

    # ── benchmark loop ────────────────────────────────────────────────────────
    print(f"Running {len(frame_paths)} frames...\n")
    bgr_frames = []
    pt_times, trt_times = [], []
    pt_results, trt_results = [], []

    header = f"{'Frame':>6}  {'PT(ms)':>8}  {'TRT(ms)':>8}  {'Speedup':>8}  {'PT ids':>8}  {'TRT ids':>8}"
    print(header)
    print("-" * len(header))

    with torch.no_grad():
        for fi, fp in enumerate(frame_paths):
            nt = _NestedTensor(_load_frame(fp, args.res))
            if not args.no_video:
                _img = cv2.imread(str(fp))
                if _img is None:
                    from PIL import Image as _PIL
                    import numpy as _np2
                    _img = _np2.array(_PIL.open(fp).convert("RGB"))[:, :, ::-1].copy()
                if _img.shape[0] != args.res or _img.shape[1] != args.res:
                    _img = cv2.resize(_img, (args.res, args.res))
                bgr_frames.append(_img)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tracker_pt.update(nt)
            torch.cuda.synchronize()
            pt_ms = (time.perf_counter() - t0) * 1000
            pt_r = tracker_pt.get_track_results()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tracker_trt.update(nt)
            torch.cuda.synchronize()
            trt_ms = (time.perf_counter() - t0) * 1000
            trt_r = tracker_trt.get_track_results()

            pt_times.append(pt_ms)
            trt_times.append(trt_ms)
            pt_results.append(pt_r)
            trt_results.append(trt_r)

            if fi % 20 == 0 or fi < 5:
                pt_n  = len(pt_r["id"])  if pt_r  else 0
                trt_n = len(trt_r["id"]) if trt_r else 0
                spd = pt_ms / trt_ms if trt_ms > 0 else 0
                print(f"{fi:>6}  {pt_ms:>8.1f}  {trt_ms:>8.1f}  {spd:>8.2f}x  {pt_n:>8}  {trt_n:>8}")

    # ── stats summary ─────────────────────────────────────────────────────────
    import numpy as _np
    pt_arr  = _np.array(pt_times)
    trt_arr = _np.array(trt_times)

    def _pct(arr, p): return float(_np.percentile(arr, p))

    print(f"\n{'='*60}")
    print(f"{'Metric':<24}  {'PT fp32':>12}  {'TRT fp16':>12}")
    print(f"{'-'*60}")
    print(f"{'Mean (ms)':<24}  {pt_arr.mean():>12.1f}  {trt_arr.mean():>12.1f}")
    print(f"{'Median (ms)':<24}  {_pct(pt_arr,50):>12.1f}  {_pct(trt_arr,50):>12.1f}")
    print(f"{'p95 (ms)':<24}  {_pct(pt_arr,95):>12.1f}  {_pct(trt_arr,95):>12.1f}")
    print(f"{'Min (ms)':<24}  {pt_arr.min():>12.1f}  {trt_arr.min():>12.1f}")
    print(f"{'Max (ms)':<24}  {pt_arr.max():>12.1f}  {trt_arr.max():>12.1f}")
    mean_spd = pt_arr.mean() / trt_arr.mean()
    print(f"{'Speedup (mean)':<24}  {'':>12}  {mean_spd:>11.2f}x")
    pt_fps  = 1000 / pt_arr.mean()
    trt_fps = 1000 / trt_arr.mean()
    print(f"{'FPS (mean)':<24}  {pt_fps:>12.1f}  {trt_fps:>12.1f}")

    # ID stats
    pt_unique  = len({int(i) for r in pt_results  if r for i in r["id"].tolist()})
    trt_unique = len({int(i) for r in trt_results if r for i in r["id"].tolist()})
    print(f"{'Unique track IDs':<24}  {pt_unique:>12}  {trt_unique:>12}")
    pt_avg_det  = _np.mean([len(r["id"]) if r else 0 for r in pt_results])
    trt_avg_det = _np.mean([len(r["id"]) if r else 0 for r in trt_results])
    print(f"{'Avg detections/frame':<24}  {pt_avg_det:>12.1f}  {trt_avg_det:>12.1f}")
    print(f"{'='*60}")

    if args.no_video:
        return 0

    # ── render side-by-side video ─────────────────────────────────────────────
    print(f"\nRendering {args.out} ...")
    H, W = args.res, args.res
    total_w = 2 * W + 4
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(args.out, fourcc, 10.0, (total_w, H))

    for fi, (bgr, pt_r, trt_r, pt_ms, trt_ms) in enumerate(
            zip(bgr_frames, pt_results, trt_results, pt_times, trt_times)):
        left  = _render_frame(bgr, pt_r,  f"PyTorch fp32", pt_ms)
        right = _render_frame(bgr, trt_r, f"TRT fp16",     trt_ms)
        out   = _make_side_by_side(left, right, fi, len(frame_paths))
        writer.write(out)
        if fi % 30 == 0:
            print(f"  rendered {fi}/{len(frame_paths)}")

    writer.release()
    print(f"Saved → {args.out}  ({len(frame_paths)} frames @ 10fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
