"""
render_3way.py -- render a 3-panel comparison video from saved eval_trt results JSON.

Usage (from MOTIP root):
  python trt/render_3way.py \
      /path/to/img1  /tmp/eval_trt_results.json \
      --out /tmp/3way.mp4 --n_frames 300 --fps 15
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

ID_PALETTE = [
    (255,  60,  60), (60, 180, 255), ( 60, 220,  90), (255, 180,  30),
    (200,  80, 255), (255, 120,  30), ( 30, 220, 200), (200, 200,  30),
    (255, 130, 160), (100, 160, 255), (160, 255, 100), (255, 200, 100),
    (220, 120, 255), (100, 255, 200), (255,  90, 120), (140, 200, 255),
    ( 90, 255, 150), (255, 160,  60), (160,  90, 255), ( 80, 220, 255),
    (255,  80, 200), (180, 255,  80), (255, 130, 100), (100, 180, 220),
    (220, 160,  80), (100, 255, 130), (180,  80, 255), (255,  80, 140),
    ( 80, 200, 180), (200, 255,  60), (255, 100, 180), (120, 140, 255),
    (255, 220,  80), ( 80, 255, 180), (200,  80, 200), (255, 160, 120),
    (140, 255, 160), ( 80, 180, 255), (255, 200, 160), (180, 255, 140),
]

LABELS = {
    "PT_fp32":  "PyTorch fp32",
    "TRT_fp32": "TRT fp32",
    "TRT_fp16": "TRT fp16",
}
ORDER = ["PT_fp32", "TRT_fp32", "TRT_fp16"]

def render_panel(bgr, frame_dets, label, res):
    img = bgr.copy()
    for (tid, x, y, w, h, sc) in frame_dets:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(res-1, x2), min(res-1, y2)
        color = ID_PALETTE[tid % len(ID_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lbl = str(tid)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(y1-th-4, 0)), (x1+tw+6, y1), color, -1)
        cv2.putText(img, lbl, (x1+2, max(y1-3, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    # label bar at top
    (hw, hh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (0, 0), (hw+14, hh+10), (20, 20, 20), -1)
    cv2.putText(img, label, (7, hh+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("results_json")
    ap.add_argument("--out", default="/tmp/3way_compare.mp4")
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--res", type=int, default=576)
    args = ap.parse_args()

    data = json.load(open(args.results_json))
    # Build per-frame lookup: tracker -> frame -> [(id, x, y, w, h, sc), ...]
    frame_dets = {k: {} for k in ORDER}
    for k in ORDER:
        for row in data.get(k, []):
            fr = int(row[0])
            frame_dets[k].setdefault(fr, []).append(
                (int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]))
            )

    frame_paths = sorted(Path(args.frame_dir).glob("*.jpg"))[:args.n_frames]
    n = len(frame_paths)
    print(f"Rendering {n} frames -> {args.out}")

    res = args.res
    gap = 6
    total_w = 3 * res + 2 * gap
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (total_w, res))

    for fi, fp in enumerate(frame_paths):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            from PIL import Image
            bgr = np.array(Image.open(fp).convert("RGB"))[:, :, ::-1].copy()
        if bgr.shape[0] != res or bgr.shape[1] != res:
            bgr = cv2.resize(bgr, (res, res))

        frame_num = fi + 1
        panels = []
        for k in ORDER:
            dets = frame_dets[k].get(frame_num, [])
            panels.append(render_panel(bgr, dets, LABELS[k], res))

        div = np.full((res, gap, 3), 30, dtype=np.uint8)
        composite = np.concatenate([panels[0], div, panels[1], div, panels[2]], axis=1)

        # frame counter bottom-center
        fc = f"{frame_num}/{n}"
        (fw, fh), _ = cv2.getTextSize(fc, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(composite, fc, (total_w//2 - fw//2, res - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        writer.write(composite)
        if fi % 30 == 0:
            print(f"  {fi}/{n}")

    writer.release()
    print(f"Done -> {args.out}")

if __name__ == "__main__":
    main()
