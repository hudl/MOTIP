"""Precompute a FROZEN detector's matched outputs for ID-decoder training (``DETR_CACHE_DIR``, see utils/detr_cache.py).

    python scripts/build_detr_cache.py --config configs/<train config>.yaml --data-root <root of Hockey/> \
        --out <cache dir> [--detr-pretrain <stage-1 .pth> | --resume-model <checkpoint .pth>] \
        [--split train] [--shard 0 --nshards 4] [--device cuda:0]

Every frame of every sequence goes through the detector once, un-augmented, in the canonical view (shorter side
``--shorter`` (default 576 = the inference resolution), longer side <= ``--max-longer``, aspect preserved, ImageNet
normalisation; exactly the transforms the training pipeline applies when it draws a scale), the last layer is matched to
that frame's GT with the training matcher, and the matched queries' embedding / box / logits are stored by GT object id.
Shards are disjoint sequence subsets (``sequences[shard::nshards]``): run one process per GPU. Finished sequences are
skipped, so a crashed build resumes.

``--video``: frames come from the source clip instead of ``img1/`` JPEGs (datasets without frames, e.g. the full-clip pool
of ``scripts/data_prep/ice_hockey/build_pool_dataset.py``). Each sequence's ``meta.json`` names ``clip_s3`` and the raw
frame range ``window_start .. window_end`` (stride 1); the clip is downloaded once and decoded in one sequential OpenCV
pass (BGR -> RGB, the decode inference uses), feeding every chunk that contains the frame. Shards are then disjoint CLIP
subsets.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.detr_cache import SeqCacheWriter, seq_cache_path  # noqa: E402
from utils.kit_feature import KIT_DIM, KIT_VERSION, kit_for_norm_boxes  # noqa: E402
from utils.nested_tensor import nested_tensor_from_tensor_list  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def canonical_hw(h: int, w: int, shorter: int, max_longer: int) -> tuple[int, int]:
    """New (h, w): ``MultiRandomResize.get_new_hw`` with a single size."""
    mn, mx = float(min(h, w)), float(max(h, w))
    if mx / mn * shorter > max_longer:
        shorter = int(math.floor(max_longer * mn / mx))
    if w < h:
        return int(round(shorter * h / w)), shorter
    return shorter, int(round(shorter * w / h))


def canonical_batch(images: list[Image.Image], shorter: int, max_longer: int, device):
    """PIL frames of one sequence -> normalised NestedTensor, as the training pipeline + loop produce it."""
    w, h = images[0].size
    nh, nw = canonical_hw(h, w, shorter, max_longer)
    tensors = []
    for im in images:
        im = v2.functional.resize(im, [nh, nw])
        tensors.append(v2.functional.to_image(im))
    nested = nested_tensor_from_tensor_list(tensors, size_divisibility=0)
    nested.tensors = v2.functional.to_dtype(nested.tensors, dtype=torch.float32, scale=True)
    nested.tensors = v2.functional.normalize(nested.tensors, mean=MEAN, std=STD)
    nested.tensors = nested.tensors * (~nested.mask[:, None, ...]).to(torch.float32)
    nested.tensors, nested.mask = nested.tensors.contiguous().to(device), nested.mask.to(device)
    return nested


def gt_targets(annotations: list[dict], width: int, height: int, device) -> list[dict]:
    """Detector-criterion targets of un-augmented frames: normalised cxcywh boxes (the pipeline's, flip / crop off)."""
    from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh

    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    out = []
    for a in annotations:
        cxcywh = box_xyxy_to_cxcywh(box_xywh_to_xyxy(a["bbox"])) / scale
        out.append({"boxes": cxcywh.to(device), "labels": a["category"].to(device)})
    return out


@torch.no_grad()
def matched_rows(model, criterion, nested, targets, annotations):
    """Detector forward + last-layer match -> per frame (ids, emb, boxes, logits) of the GT objects, in GT order."""
    out = model(frames=nested, part="detr")
    indices = criterion.match_last_layer(outputs=out, targets=targets, batch_len=10)
    rows = []
    for i, (src, tgt) in enumerate(indices):
        q = src[torch.argsort(tgt)]
        rows.append((
            annotations[i]["id"].numpy(),
            out["outputs"][i][q].float().cpu().numpy(),
            out["pred_boxes"][i][q].float().cpu().numpy(),
            out["pred_logits"][i][q].float().cpu().numpy(),
        ))
    return rows


def load_model(config: dict, detr_pretrain: str | None, resume_model: str | None, device):
    from models.misc import load_checkpoint, load_detr_pretrain
    from models.motip import build as build_motip

    model, criterion = build_motip(config=config)
    if detr_pretrain:
        load_detr_pretrain(model=model, pretrain_path=detr_pretrain, num_classes=config["NUM_CLASSES"],
                           default_class_idx=config.get("DETR_DEFAULT_CLASS_IDX"))
    if resume_model:
        load_checkpoint(model=model, path=resume_model)
    return model.to(device).eval(), criterion


def _download(uri: str, dst: str) -> None:
    import boto3
    from urllib.parse import urlparse

    u = urlparse(uri)
    boto3.client("s3").download_file(u.netloc, u.path.lstrip("/"), dst)


def build_from_video(a, ds, infos, anns, model, criterion, device, info) -> None:
    """--video: one sequential decode per clip; each frame goes to every (unfinished) sequence whose range holds it."""
    import json
    import tempfile
    from collections import defaultdict

    import cv2

    split_dir = Path(a.data_root) / a.dataset / a.split
    by_clip = defaultdict(list)
    for seq in sorted(infos):
        m = json.loads((split_dir / seq / "meta.json").read_text())
        assert int(m.get("stride", 1)) == 1, f"{seq}: --video needs stride-1 sequences"
        by_clip[m["clip_s3"]].append((int(m["window_start"]), int(m["window_end"]), seq))
    clips = sorted(by_clip)[a.shard :: a.nshards][: a.max_seqs]
    t0, n_frames = time.time(), 0
    for ci, clip in enumerate(clips):
        todo = [(s, e, q) for s, e, q in sorted(by_clip[clip]) if not seq_cache_path(a.out, a.dataset, a.split, q).exists()]
        if not todo:
            continue
        for s, e, q in todo:
            assert infos[q]["length"] == e - s, f"{q}: seqLength {infos[q]['length']} != window {e - s}"
        writers = {q: SeqCacheWriter(e - s) for s, e, q in todo}
        pending = {q: [] for _, _, q in todo}   # (local frame idx, PIL frame) waiting for a batch
        last = max(e for _, e, _ in todo)

        def flush(q, start):
            if not pending[q]:
                return
            idx = [i for i, _ in pending[q]]
            frames = [fr for _, fr in pending[q]]
            W, H = infos[q]["width"], infos[q]["height"]
            nested = canonical_batch(frames, a.shorter, a.max_longer, device)
            targets = gt_targets([anns[q][i] for i in idx], W, H, device)
            rows = matched_rows(model, criterion, nested, targets, [anns[q][i] for i in idx])
            for fr, (ids, emb, boxes, logits) in zip(frames, rows):
                kit = kit_for_norm_boxes(np.ascontiguousarray(np.asarray(fr)[..., ::-1]), boxes) if a.kit else None
                writers[q].add_frame(ids, emb, boxes, logits, kit=kit)
            pending[q] = []

        with tempfile.TemporaryDirectory() as td:
            vp = os.path.join(td, "clip.mp4")
            _download(clip, vp)
            cap = cv2.VideoCapture(vp)
            f = 0
            while f < last:
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"{clip}: video ended at frame {f}, sequences need {last}")
                frame = None
                for s, e, q in todo:
                    if s <= f < e:
                        if frame is None:
                            frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        pending[q].append((f - s, frame))
                        if len(pending[q]) == a.batch or f == e - 1:
                            flush(q, s)
                f += 1
            cap.release()
        for s, e, q in todo:
            writers[q].save(seq_cache_path(a.out, a.dataset, a.split, q), {**info, "source": "video", "clip_s3": clip,
                                                                          "window_start": s, "window_end": e})
            n_frames += e - s
        print(f"[shard {a.shard}] clip {ci + 1}/{len(clips)} {clip.rsplit('/', 2)[-2][:8]}: {len(todo)} seqs, "
              f"{n_frames / (time.time() - t0):.1f} frames/s", flush=True)
    print(f"done: {n_frames} frames in {time.time() - t0:.0f}s -> {a.out}")


def build_config(path: str) -> dict:
    from configs.util import load_super_config
    from utils.misc import yaml_to_dict

    cfg = yaml_to_dict(path)
    return load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="Hockey")
    ap.add_argument("--split", default="train")
    ap.add_argument("--detr-pretrain", default=None, help="detector-only checkpoint (DETR_PRETRAIN)")
    ap.add_argument("--resume-model", default=None, help="full MOTIP checkpoint whose detector to cache (RESUME_MODEL)")
    ap.add_argument("--shorter", type=int, default=576)
    ap.add_argument("--max-longer", type=int, default=1440)
    ap.add_argument("--batch", type=int, default=8, help="frames per detector forward")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--video", action="store_true",
                    help="decode frames from meta.json clip_s3 (no img1/ needed); shards by clip")
    ap.add_argument("--kit", action="store_true",
                    help="also store the kit descriptor (utils/kit_feature.py) of every matched row, computed from the "
                         "un-augmented frame and the matched PREDICTED box (what inference sees), for TEAM_CUE")
    a = ap.parse_args()
    assert a.detr_pretrain or a.resume_model, "give the detector weights: --detr-pretrain or --resume-model"

    cfg = build_config(a.config)
    cfg["DETR_FREEZE"] = True
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from data.joint_dataset import dataset_classes

    ds = dataset_classes[a.dataset](data_root=a.data_root, split=a.split, load_annotation=True)
    infos, paths, anns = ds.get_sequence_infos(), ds.get_image_paths(), ds.get_annotations()
    seqs = sorted(infos)[a.shard :: a.nshards][: a.max_seqs]
    model, criterion = load_model(cfg, a.detr_pretrain, a.resume_model, device)
    for p in model.parameters():
        p.requires_grad = False
    info = {"view": f"shorter{a.shorter}_max{a.max_longer}_unflipped", "dtype": "fp32",
            "detr_pretrain": a.detr_pretrain, "resume_model": a.resume_model, "num_classes": cfg["NUM_CLASSES"],
            **({"kit_version": KIT_VERSION, "kit_dim": KIT_DIM} if a.kit else {})}
    t0, n_frames = time.time(), 0
    if a.video:
        build_from_video(a, ds, infos, anns, model, criterion, device, info)
        return
    for si, seq in enumerate(seqs):
        out_path = seq_cache_path(a.out, a.dataset, a.split, seq)
        if out_path.exists():
            continue
        W, H, T = infos[seq]["width"], infos[seq]["height"], infos[seq]["length"]
        writer = SeqCacheWriter(T)
        for s in range(0, T, a.batch):
            idx = list(range(s, min(T, s + a.batch)))
            frames = [Image.open(paths[seq][i]).convert("RGB") for i in idx]
            nested = canonical_batch(frames, a.shorter, a.max_longer, device)
            targets = gt_targets([anns[seq][i] for i in idx], W, H, device)
            rows = matched_rows(model, criterion, nested, targets, [anns[seq][i] for i in idx])
            for fr, (ids, emb, boxes, logits) in zip(frames, rows):
                kit = kit_for_norm_boxes(np.ascontiguousarray(np.asarray(fr)[..., ::-1]), boxes) if a.kit else None
                writer.add_frame(ids, emb, boxes, logits, kit=kit)
        writer.save(out_path, info)
        n_frames += T
        print(f"[shard {a.shard}] {si + 1}/{len(seqs)} {seq}: {T} frames, {n_frames / (time.time() - t0):.1f} frames/s", flush=True)
    print(f"done: {n_frames} frames in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
