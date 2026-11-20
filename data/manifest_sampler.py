# IHC: training windows from an explicit manifest (``SAMPLE_MANIFEST``) instead of NaiveSampler's every-start sweep.
"""ManifestSampler: draw each epoch's training windows from a JSON list of candidate windows.

The manifest (written by ``scripts/data_prep/ice_hockey/make_window_manifest.py``) names every candidate window over the
dataset's sequences, so one dataset + one DETR cache serves any window recipe (event-centred, uniform, gap windows, mixes,
temporal steps):

    {"version": 1, "dataset": "Hockey", "split": "train", "samples_per_epoch": 1907,
     "windows": [{"seq": "013d2b36_c00_f00000", "start": 12, "length": 60, "step": 1, "weight": 1.0, "tag": "event"}, ...],
     "recipe": {...free-form provenance...}}

``start`` is the 0-based index of the first frame in the sequence, frames are ``start + step * i`` for i < length.
Windows that do not fit the sequence, or that touch a frame with an illegal annotation (no GT box, duplicate ids: the same
``is_legal`` rule NaiveSampler applies), are dropped once at construction and counted per tag. Each epoch draws
``samples_per_epoch`` windows with probability proportional to ``weight`` (without replacement while there are enough
windows, else with replacement), deterministically from ``seed + epoch`` so every process of a multi-GPU run draws the
same list. Windows of different lengths in one manifest need ``BATCH_SIZE`` 1.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import torch
from torch.utils.data.sampler import Sampler


def resolve_manifest_path(path: str, data_root: str | None) -> str:
    """As given if it exists, else relative to DATA_ROOT (manifests ship inside the dataset for SageMaker)."""
    if os.path.exists(path) or os.path.isabs(path) or not data_root:
        return path
    return os.path.join(data_root, path)


class ManifestSampler(Sampler):
    def __init__(self, data_source, manifest: str | dict, seed: int = 1025, samples_per_epoch: int | None = None):
        super().__init__()
        m = manifest if isinstance(manifest, dict) else json.load(open(manifest))
        assert m.get("version") == 1, f"unknown manifest version {m.get('version')}"
        self.dataset, self.split = m.get("dataset", "Hockey"), m.get("split", "train")
        self.seed = seed
        infos = data_source.sequence_infos[self.dataset][self.split]
        legal = data_source.ann_is_legals[self.dataset][self.split]
        kept, dropped = [], Counter()
        for w in m["windows"]:
            seq, start, length, step = w["seq"], int(w["start"]), int(w["length"]), int(w.get("step", 1))
            tag = w.get("tag", "")
            if seq not in infos:
                dropped[f"{tag}:missing_seq"] += 1
                continue
            idxs = [start + step * i for i in range(length)]
            if start < 0 or idxs[-1] >= infos[seq]["length"]:
                dropped[f"{tag}:outside"] += 1
                continue
            if not bool(legal[seq][torch.tensor(idxs)].all()):
                dropped[f"{tag}:illegal"] += 1
                continue
            weight = float(w.get("weight", 1.0))
            if weight <= 0:
                dropped[f"{tag}:zero_weight"] += 1
                continue
            kept.append({"seq": seq, "frame_idxs": idxs, "weight": weight, "tag": tag})
        assert kept, f"no legal window in the manifest (dropped: {dict(dropped)})"
        self.windows = kept
        self.dropped = dict(dropped)
        self.weights = torch.tensor([w["weight"] for w in kept], dtype=torch.float64)
        self.samples_per_epoch = int(samples_per_epoch or m.get("samples_per_epoch") or len(kept))
        self.lengths = sorted({len(w["frame_idxs"]) for w in kept})
        self.sample_infos = None
        self.last_epoch_stats = None

    @property
    def variable_length(self) -> bool:
        return len(self.lengths) > 1

    def prepare_for_epoch(self, epoch: int):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        n = self.samples_per_epoch
        replacement = n > len(self.windows)
        picks = torch.multinomial(self.weights, n, replacement=replacement, generator=g).tolist()
        self.sample_infos = [{"dataset": self.dataset, "split": self.split, "sequence": self.windows[i]["seq"],
                              "frame_idxs": self.windows[i]["frame_idxs"]} for i in picks]
        tags = Counter(self.windows[i]["tag"] for i in picks)
        frames = Counter()
        for i in picks:
            frames[self.windows[i]["tag"]] += len(self.windows[i]["frame_idxs"])
        # keys NaiveSampler reports (train.py logs them), plus the manifest-specific ones
        self.last_epoch_stats = {
            "samples": n, "sample_length": self.lengths[0] if len(self.lengths) == 1 else f"{self.lengths[0]}-{self.lengths[-1]}",
            "sequences": len({w["seq"] for w in self.windows}), "short": 0, "short_kept": 0, "no_window": 0,
            "candidates": len(self.windows), "dropped": self.dropped, "replacement": replacement,
            "tags": dict(tags), "window_frames": dict(frames),
        }

    def __iter__(self):
        return iter(self.sample_infos)

    def __len__(self):
        return len(self.sample_infos) if self.sample_infos is not None else self.samples_per_epoch
