# IHC: per-detection "kit appearance" descriptor for the MOTIP ID decoder (TEAM_CUE).
#
# Definition (version 1, 14 numbers, NaN row = no descriptor):
#   Two body regions of the detection box (fractions of the box; the box is clipped to the frame first):
#     torso = x in [0.15, 0.85] * w, y in [0.18, 0.50] * h     (jersey, shoulders)
#     legs  = x in [0.20, 0.80] * w, y in [0.55, 0.90] * h     (pants shell, socks)
#   each cropped from the BGR frame and area-resized to 6 x 8 px (w x h), then 7 numbers per region:
#     mean R, G, B in [0, 1]; mean saturation and value (OpenCV HSV) in [0, 1];
#     saturation-weighted hue vector (mean of S * cos(h), mean of S * sin(h)), in [-1, 1].
#   Order: torso (R, G, B, S, V, hue_cos, hue_sin), legs (same). A box smaller than 8 x 24 px after clipping gives NaN.
# It uses nothing but the frame and the box, so it is identical for training frames (dataset JPEGs, cached GT-matched
# detector boxes) and for inference (video frames, detections). No ReID model, no clustering, no per-clip state.
import numpy as np

KIT_VERSION = 1
KIT_DIM = 14
REGIONS = (((0.15, 0.85), (0.18, 0.50)), ((0.20, 0.80), (0.55, 0.90)))   # ((x0, x1), (y0, y1)) of the box
PATCH_WH = (6, 8)
MIN_W, MIN_H = 8.0, 24.0


def _region_stats(bgr_patch: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(bgr_patch, PATCH_WH, interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    theta = hsv[..., 0] * (2.0 * np.pi / 180.0)
    return np.array([*rgb.reshape(-1, 3).mean(0), sat.mean(), (hsv[..., 2] / 255.0).mean(),
                     (sat * np.cos(theta)).mean(), (sat * np.sin(theta)).mean()], dtype=np.float32)


def kit_descriptor(frame_bgr: np.ndarray, box_xyxy) -> np.ndarray:
    """(KIT_DIM,) float32 descriptor of one box in pixel xyxy of ``frame_bgr`` (H, W, 3 uint8); NaN when unusable."""
    out = np.full(KIT_DIM, np.nan, dtype=np.float32)
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    if not np.isfinite([x1, y1, x2, y2]).all():
        return out
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w_img), x2), min(float(h_img), y2)
    w, h = x2 - x1, y2 - y1
    if w < MIN_W or h < MIN_H:
        return out
    parts = []
    for (rx0, rx1), (ry0, ry1) in REGIONS:
        xa, xb = int(round(x1 + rx0 * w)), int(round(x1 + rx1 * w))
        ya, yb = int(round(y1 + ry0 * h)), int(round(y1 + ry1 * h))
        xb, yb = max(xb, xa + 1), max(yb, ya + 1)
        parts.append(_region_stats(np.ascontiguousarray(frame_bgr[ya:yb, xa:xb])))
    return np.concatenate(parts)


def kit_for_boxes(frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """(n, KIT_DIM) for (n, 4) pixel-xyxy boxes."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    out = np.full((len(boxes_xyxy), KIT_DIM), np.nan, dtype=np.float32)
    for i, b in enumerate(boxes_xyxy):
        out[i] = kit_descriptor(frame_bgr, b)
    return out


def cxcywh_norm_to_xyxy(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    """Normalised cxcywh (the detector's / the cache's boxes) -> pixel xyxy of a ``width`` x ``height`` frame."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * width, b[:, 1] * height, b[:, 2] * width, b[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def kit_for_norm_boxes(frame_bgr: np.ndarray, boxes_cxcywh: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    return kit_for_boxes(frame_bgr, cxcywh_norm_to_xyxy(boxes_cxcywh, w, h))
