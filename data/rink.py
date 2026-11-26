# IHC: rink-coordinate data for the MOTIP rink anchor (numpy / torch only; no sip_tracking import, because the
# SageMaker job only ships this repo).
#
# Per sequence, ``rink/homography.jsonl`` (written by ``scripts/data_prep/ice_hockey/add_rink_sidecar.py`` in
# sip-tracking) holds ihc-pipeline's predicted homography for every sequence frame, and ``rink/rink_meta.json`` the
# metres -> template matrix and the rink half extent. ``H`` maps rink-template px -> frame px, so
#
#     frame px -> rink metres = inv(H @ metres_to_template_px)
#
# Rink metres: origin centre ice, +x along the long axis, +y up in the template. A player's position is their feet:
# the bbox bottom-centre.
#
# Training augmentation (resize, crop, horizontal flip) moves boxes in the image. ``RinkAug`` tracks that as a
# per-axis affine map so the boxes the ID decoder sees (normalised cxcywh in the augmented image) can be taken back
# to original-frame pixels and projected with the frame's own homography.

import json
import os
from dataclasses import dataclass

import numpy as np

HOMOGRAPHY_FILE = "homography.jsonl"
META_FILE = "rink_meta.json"
GT_RINK_FILE = "gt_rink.txt"


def img_to_metres(h, metres_to_template_px) -> np.ndarray | None:
    """3x3 frame px -> rink metres from ihc-pipeline's H, or None when it is degenerate (same rule as
    ``sip_tracking.rink.load_rink_transforms``)."""
    h = np.asarray(h, dtype=np.float64)
    m = np.asarray(metres_to_template_px, dtype=np.float64)
    try:
        k = np.linalg.inv(h @ m)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(k)) or abs(np.linalg.det(h)) <= 1e-12:
        return None
    return k / k[2, 2] if abs(k[2, 2]) > 1e-12 else k


@dataclass
class SequenceRink:
    """Image -> rink transforms of one dataset sequence, indexed by 0-based sequence frame
    (frame ``i`` is ``img1/{i + 1:08d}.jpg``, ``frame = i + 1`` in gt.txt)."""

    img_to_rink: np.ndarray        # (F, 3, 3) float64, NaN where the frame has no homography
    half_extent_m: tuple           # (half length, half width) of the playing surface
    margin_m: float = 1.0

    @property
    def n_frames(self) -> int:
        return int(self.img_to_rink.shape[0])

    @property
    def bounds_m(self) -> tuple:
        """(x, y) bounds inside which a projected feet point counts as on the rink."""
        return (self.half_extent_m[0] + self.margin_m, self.half_extent_m[1] + self.margin_m)

    def frame_H(self, frame_idx: int) -> np.ndarray:
        """3x3 for the 0-based frame, NaN where there is none / out of range."""
        if 0 <= frame_idx < self.n_frames:
            return self.img_to_rink[frame_idx]
        return np.full((3, 3), np.nan)


def load_sequence_rink(seq_dir: str) -> SequenceRink | None:
    """The sequence's rink transforms, or None when it has no sidecar (older dataset, or no homography file)."""
    meta_path = os.path.join(seq_dir, "rink", META_FILE)
    h_path = os.path.join(seq_dir, "rink", HOMOGRAPHY_FILE)
    if not (os.path.exists(meta_path) and os.path.exists(h_path)):
        return None
    with open(meta_path) as fh:
        meta = json.load(fh)
    m = meta.get("metres_to_template_px")
    half = meta.get("half_extent_m")
    if m is None or half is None:        # clip without a homography stem: every frame invalid
        return None
    rows = []
    with open(h_path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    n = max((int(r["frame"]) for r in rows), default=0)
    out = np.full((n, 3, 3), np.nan)
    for r in rows:
        if r.get("H") is None:
            continue
        k = img_to_metres(r["H"], m)
        if k is not None:
            out[int(r["frame"]) - 1] = k
    return SequenceRink(img_to_rink=out, half_extent_m=(float(half[0]), float(half[1])),
                        margin_m=float(meta.get("margin_m", 1.0)))


def load_gt_rink(seq_dir: str) -> np.ndarray | None:
    """(N, 6) float array ``frame, track_id, x_m, y_m, valid, h_ok``, row-aligned with ``gt/gt.txt``."""
    path = os.path.join(seq_dir, "rink", GT_RINK_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    return np.array([[float(v) for v in r] for r in rows], dtype=np.float64).reshape(-1, 6)


# ----------------------------------------------------------------------------------------------------------
# Augmentation tracking
# ----------------------------------------------------------------------------------------------------------

def aug_init(width: float, height: float) -> dict:
    """Identity: original px -> current image px is ``x' = ax * x + bx``, ``y' = ay * y + by``."""
    return {"ax": 1.0, "bx": 0.0, "ay": 1.0, "by": 0.0, "w": float(width), "h": float(height), "flip": False}


def aug_flip(a: dict) -> None:
    """Horizontal flip of the current image (``x' -> w - x'``)."""
    a["ax"], a["bx"] = -a["ax"], a["w"] - a["bx"]
    a["flip"] = not a["flip"]


def aug_resize(a: dict, new_w: float, new_h: float) -> None:
    rx, ry = new_w / a["w"], new_h / a["h"]
    a["ax"] *= rx
    a["bx"] *= rx
    a["ay"] *= ry
    a["by"] *= ry
    a["w"], a["h"] = float(new_w), float(new_h)


def aug_crop(a: dict, top: float, left: float, new_h: float, new_w: float) -> None:
    a["bx"] -= left
    a["by"] -= top
    a["w"], a["h"] = float(new_w), float(new_h)


def aug_norm_to_orig(a: dict) -> np.ndarray:
    """3x3 taking a point normalised to the current image ([0, 1] in x and y) to original-frame pixels."""
    return np.array([
        [a["w"] / a["ax"], 0.0, -a["bx"] / a["ax"]],
        [0.0, a["h"] / a["ay"], -a["by"] / a["ay"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def apply_aug(metas: list, fn, *args) -> None:
    """Apply an ``aug_*`` update to every frame's ``rink_aug`` (no-op when the rink anchor is off)."""
    for m in metas:
        if "rink_aug" in m:
            fn(m["rink_aug"], *args)


def finalize_meta(meta: dict) -> None:
    """Replace ``meta['rink_aug']`` by what the training loop needs: ``rink_Hn`` (3x3, normalised augmented-image
    coords -> rink metres, NaN where the frame has no homography), ``rink_flip`` and ``rink_bounds`` (2,)."""
    a = meta.pop("rink_aug")
    h = np.asarray(meta.pop("rink_H"), dtype=np.float64)
    meta["rink_Hn"] = h @ aug_norm_to_orig(a)
    meta["rink_flip"] = bool(a["flip"])


def needs_rink_data(config: dict) -> bool:
    """Whether training needs the per-frame rink transforms (the sidecar loader, MultiRinkFinalize and the per-token feet
    XY in prepare_for_motip): the rink anchor, the rink-distance attention bias, or the identity loss's rink-distance
    hard negatives (ID_EMBED_LOSS with ID_EMBED_HARD_RINK_M > 0)."""
    return bool(
        config.get("RINK_ANCHOR", False)
        or config.get("RINK_DIST_BIAS", False)
        or (config.get("ID_EMBED_LOSS", False) and float(config.get("ID_EMBED_HARD_RINK_M", 1.5)) > 0)
    )


def load_gt_team(sequence_dir: str) -> dict | None:
    """``{gt track id: team}`` from ``<seq>/gt/gt_team.txt`` (lines ``track_id,team``; team an int, -1 = unknown), or None
    without the file. Training-only labels for the optional ID_EMBED_TEAM term."""
    path = os.path.join(sequence_dir, "gt", "gt_team.txt")
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                a, b = line.split(",")[:2]
                out[int(float(a))] = int(float(b))
    return out
