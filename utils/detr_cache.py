"""Cache of a FROZEN detector's outputs for ID-decoder training (``DETR_CACHE_DIR``).

With ``DETR_FREEZE`` the detector forward is 77-92% of a training step, yet the ID head only ever reads three things of it:
for every GT object of a frame, the last-layer query that the Hungarian matcher assigns to it -- its embedding
(``outputs``, 256-d), its predicted box and its class logits. So the cache stores exactly those, per sequence and frame,
keyed by the GT object id:

    <cache_dir>/<dataset>/<split>/<sequence>.npz
        ptr     (T + 1,)   int64   rows of frame t are ptr[t]:ptr[t + 1]   (t = 0-based frame index in the sequence)
        ids     (M,)       int64   GT object id of each row (the ``id`` column of gt.txt)
        emb     (M, D)     float32 query embedding            (``detr_outputs["outputs"]`` row)
        boxes   (M, 4)     float32 predicted box, normalised cxcywh of the canonical (unflipped) view
        logits  (M, C)     float32 class logits
        info    ()         str     json: version, view, resolution, dtype, detector checkpoint

The match was computed once on the un-augmented frame against that frame's GT. ``match_last_layer`` costs (class, L1, GIoU)
are invariant to a horizontal flip applied to predictions AND GT, so the match stays valid under the one geometric
augmentation that survives caching (see ``flip_boxes``); crops and rescales do not, see the session note.

Training looks rows up by (sequence, frame_idx, object id) and rebuilds ``detr_outputs`` / ``detr_indices`` in the exact
layout ``prepare_for_motip`` expects, with identity indices: row i of the rebuilt outputs IS the query matched to GT object i.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

CACHE_VERSION = 1


def seq_cache_path(cache_dir: str | os.PathLike, dataset: str, split: str, sequence: str) -> Path:
    return Path(cache_dir) / dataset / split / f"{sequence}.npz"


class SeqCacheWriter:
    """Accumulates one sequence frame by frame (in frame order), then writes the ``.npz``."""

    def __init__(self, n_frames: int):
        self.n_frames = n_frames
        self._ids: list[np.ndarray] = []
        self._emb: list[np.ndarray] = []
        self._boxes: list[np.ndarray] = []
        self._logits: list[np.ndarray] = []
        self._kit: list[np.ndarray] = []
        self._ptr = [0]

    def add_frame(self, ids: np.ndarray, emb: np.ndarray, boxes: np.ndarray, logits: np.ndarray,
                  kit: np.ndarray | None = None) -> None:
        assert len(self._ptr) <= self.n_frames, "more frames than the sequence has"
        n = len(ids)
        assert emb.shape[0] == boxes.shape[0] == logits.shape[0] == n
        self._ids.append(np.asarray(ids, dtype=np.int64))
        self._emb.append(np.asarray(emb, dtype=np.float32).reshape(n, emb.shape[-1]))
        self._boxes.append(np.asarray(boxes, dtype=np.float32).reshape(n, 4))
        self._logits.append(np.asarray(logits, dtype=np.float32).reshape(n, logits.shape[-1]))
        if kit is not None:   # optional (TEAM_CUE): the kit descriptor of each row, utils/kit_feature.py, NaN = none
            assert len(kit) == n
            self._kit.append(np.asarray(kit, dtype=np.float32).reshape(n, np.shape(kit)[-1]))
        self._ptr.append(self._ptr[-1] + n)

    def save(self, path: str | os.PathLike, info: dict) -> None:
        assert len(self._ptr) == self.n_frames + 1, f"wrote {len(self._ptr) - 1} of {self.n_frames} frames"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cat = lambda xs, shape: np.concatenate(xs) if xs and sum(len(x) for x in xs) else np.zeros(shape, np.float32)  # noqa: E731
        d = self._emb[0].shape[1] if self._emb else 0
        c = self._logits[0].shape[1] if self._logits else 0
        tmp = path.with_name(path.name + ".tmp.npz")
        extra = {}
        if self._kit:
            assert len(self._kit) == len(self._ids), "kit must be given for every frame or for none"
            extra["kit"] = np.concatenate(self._kit) if sum(len(x) for x in self._kit) else np.zeros((0, self._kit[0].shape[1]), np.float32)
        np.savez(
            tmp,
            **extra,
            ptr=np.asarray(self._ptr, dtype=np.int64),
            ids=np.concatenate(self._ids) if self._ids else np.zeros(0, np.int64),
            emb=cat(self._emb, (0, d)), boxes=cat(self._boxes, (0, 4)), logits=cat(self._logits, (0, c)),
            info=np.asarray(json.dumps({"version": CACHE_VERSION, **info})),
        )
        os.replace(tmp, path)  # a reader never sees a half-written sequence


class _Seq:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.ptr, self.ids, self.emb, self.boxes, self.logits = (z[k] for k in ("ptr", "ids", "emb", "boxes", "logits"))
            self.info = json.loads(str(z["info"]))
            self.kit = z["kit"] if "kit" in z.files else None
        if self.info.get("version") != CACHE_VERSION:
            raise ValueError(f"{path}: cache version {self.info.get('version')} != {CACHE_VERSION}")
        self.n_frames = len(self.ptr) - 1

    def frame(self, t: int):
        a, b = int(self.ptr[t]), int(self.ptr[t + 1])
        return self.ids[a:b], self.emb[a:b], self.boxes[a:b], self.logits[a:b]

    def frame_kit(self, t: int):
        a, b = int(self.ptr[t]), int(self.ptr[t + 1])
        return self.kit[a:b]


def flip_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Horizontal flip of normalised cxcywh boxes: cx -> 1 - cx."""
    out = boxes.clone()
    out[..., 0] = 1.0 - out[..., 0]
    return out


class DetrCache:
    """Reader. One instance per process / dataloader consumer; sequences are loaded lazily and kept in a small LRU."""

    def __init__(self, cache_dir: str | os.PathLike, max_open: int = 32, with_kit: bool = False):
        self.cache_dir = Path(cache_dir)
        self.max_open = max_open
        self.with_kit = with_kit   # TEAM_CUE: also return the cached kit descriptors as ``detr_outputs["kit"]``
        self._open: OrderedDict[tuple, _Seq] = OrderedDict()

    def seq(self, dataset: str, split: str, sequence: str) -> _Seq:
        key = (dataset, split, sequence)
        if key in self._open:
            self._open.move_to_end(key)
            return self._open[key]
        path = seq_cache_path(self.cache_dir, dataset, split, sequence)
        if not path.exists():
            raise FileNotFoundError(f"DETR cache has no sequence {dataset}/{split}/{sequence}: {path}")
        s = _Seq(path)
        self._open[key] = s
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return s

    def frame_rows(self, dataset: str, split: str, sequence: str, frame_idx: int, obj_ids: np.ndarray):
        """(emb, boxes, logits) of the rows for ``obj_ids`` (in that order) of one frame."""
        s = self.seq(dataset, split, sequence)
        if not 0 <= frame_idx < s.n_frames:
            raise IndexError(f"{sequence}: frame {frame_idx} outside the cached 0..{s.n_frames - 1}")
        ids, emb, boxes, logits = s.frame(frame_idx)
        if len(obj_ids) == 0:
            return emb[:0], boxes[:0], logits[:0]
        order = {int(i): r for r, i in enumerate(ids)}
        try:
            rows = np.fromiter((order[int(i)] for i in obj_ids), dtype=np.int64, count=len(obj_ids))
        except KeyError as e:
            raise KeyError(f"{sequence} frame {frame_idx}: GT object {e.args[0]} is not in the cache "
                           f"(cached ids {ids.tolist()}); the cache was built for different labels") from None
        return emb[rows], boxes[rows], logits[rows]

    def frame_kit(self, dataset: str, split: str, sequence: str, frame_idx: int, obj_ids: np.ndarray) -> np.ndarray:
        """(len(obj_ids), K) kit descriptors of the rows for ``obj_ids`` (NaN rows = no descriptor)."""
        s = self.seq(dataset, split, sequence)
        if s.kit is None:
            raise ValueError(f"{sequence}: the cache has no kit descriptors (build it with scripts/build_detr_cache.py --kit)")
        ids = s.ids[int(s.ptr[frame_idx]): int(s.ptr[frame_idx + 1])]
        kit = s.frame_kit(frame_idx)
        if len(obj_ids) == 0:
            return kit[:0]
        order = {int(i): r for r, i in enumerate(ids)}
        return kit[np.fromiter((order[int(i)] for i in obj_ids), dtype=np.int64, count=len(obj_ids))]

    def detr_outputs(self, metas, annotations, device) -> tuple[dict, list]:
        """The frozen detector's ``detr_outputs`` and matcher ``detr_indices`` for a batch, from the cache.

        Args:
            metas: (B, T) frame metas of the batch (``dataset``, ``split``, ``sequence``, ``frame_idx``, optional ``detr_flip``).
            annotations: (B, T) annotations (their ``id`` order is the row order of the rebuilt outputs).
        Returns ``({"pred_logits", "pred_boxes", "outputs"}`` of shape (B*T, Nmax, .), ``[(arange(N), arange(N))] * (B*T))``.
        """
        embs, boxes_l, logits_l, n_l, kit_l = [], [], [], [], []
        for b_metas, b_anns in zip(metas, annotations):
            for meta, ann in zip(b_metas, b_anns):
                emb, boxes, logits = self.frame_rows(
                    meta["dataset"], meta["split"], meta["sequence"], int(meta["frame_idx"]), ann["id"].cpu().numpy())
                boxes = torch.from_numpy(boxes)
                if meta.get("detr_flip", False):
                    boxes = flip_boxes(boxes)
                embs.append(torch.from_numpy(emb))
                boxes_l.append(boxes)
                logits_l.append(torch.from_numpy(logits))
                n_l.append(len(emb))
                if self.with_kit:
                    kit_l.append(torch.from_numpy(self.frame_kit(
                        meta["dataset"], meta["split"], meta["sequence"], int(meta["frame_idx"]), ann["id"].cpu().numpy())))
        n_max = max(max(n_l), 1)
        d, c = embs[0].shape[-1], logits_l[0].shape[-1]
        flat = len(embs)
        out_emb = torch.zeros(flat, n_max, d)
        out_box = torch.zeros(flat, n_max, 4)
        out_logit = torch.full((flat, n_max, c), -20.0)  # padding rows are never indexed; a low logit if one ever is
        for i, (e, bx, lg) in enumerate(zip(embs, boxes_l, logits_l)):
            out_emb[i, : len(e)], out_box[i, : len(e)], out_logit[i, : len(e)] = e, bx, lg
        outputs = {
            "pred_logits": out_logit.to(device, non_blocking=True),
            "pred_boxes": out_box.to(device, non_blocking=True),
            "outputs": out_emb.to(device, non_blocking=True),
        }
        if self.with_kit:
            out_kit = torch.full((flat, n_max, kit_l[0].shape[-1]), float("nan"))
            for i, k in enumerate(kit_l):
                out_kit[i, : len(k)] = k
            outputs["kit"] = out_kit.to(device, non_blocking=True)
        indices = [(torch.arange(n, dtype=torch.int64), torch.arange(n, dtype=torch.int64)) for n in n_l]
        return outputs, indices
