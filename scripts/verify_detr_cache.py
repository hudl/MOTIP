"""Check that training from the DETR cache equals training with the live frozen detector, on the same batches.

    python scripts/verify_detr_cache.py --config <train config> --data-root <root> --cache <dir built by build_detr_cache.py> \
        --detr-pretrain <ckpt> [--windows 6 --length 20]

For sampled windows (the real dataset pipeline in cache mode: placeholder images, trajectory augmentation incl. the configured
ID-switch mode, flip off) it compares, per window:
  * ``prepare_for_motip`` of the cached rows against the same call on a live detector forward + match of the same frames
    (every tensor; max abs difference),
  * the ID loss (CE + spatial terms when configured) of one forward / backward through the ID head from each, same seed,
  * and, as information, how far the cached-and-box-flipped rows are from a live forward on the mirrored frames (the
    approximation of the flip augmentation: the embeddings are NOT recomputed).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_detr_cache import build_config, canonical_batch, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--detr-pretrain", default=None)
    ap.add_argument("--resume-model", default=None)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--shorter", type=int, default=576)
    ap.add_argument("--max-longer", type=int, default=1440)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from data import build_dataset
    from data.util import collate_fn
    from models.motip.id_criterion import build as build_id_criterion
    from train import annotations_to_flatten_detr_targets, prepare_for_motip
    from utils.detr_cache import DetrCache, flip_boxes

    cfg = build_config(a.config)
    cfg.update(DATA_ROOT=a.data_root, DETR_FREEZE=True, DETR_CACHE_DIR=a.cache, DETR_CACHE_FLIP=False)
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, criterion = load_model(cfg, a.detr_pretrain, a.resume_model, device)
    model.train()
    model.detr.eval()  # as in training: ID head in train mode (dropout, labels), frozen detector in eval
    id_criterion = build_id_criterion(config=cfg)
    cache = DetrCache(a.cache)
    ds = build_dataset(config=cfg)
    seqs = [(d, s, q) for d in ds.sequence_infos for s in ds.sequence_infos[d] for q in ds.sequence_infos[d][s]]
    rng = random.Random(0)
    worst = {"tensors": 0.0, "loss": 0.0}
    flip_cos, flip_box = [], []
    for w in range(a.windows):
        d, s, q = rng.choice(seqs)
        info = ds.sequence_infos[d][s][q]
        start = rng.randrange(0, max(1, info["length"] - a.length))
        idxs = list(range(start, min(info["length"], start + a.length)))
        torch.manual_seed(w), random.seed(w), np.random.seed(w)
        images, annotations, metas = ds[{"dataset": d, "split": s, "sequence": q, "frame_idxs": idxs}]
        batch = collate_fn([(images, annotations, metas)])
        annotations, metas = batch["annotations"], batch["metas"]

        # cached
        c_out, c_idx = cache.detr_outputs(metas, annotations, device)
        # live: same frames through the detector, matched against the (normalised, cxcywh) GT of the same annotations
        frames = [Image.open(ds.image_paths[d][s][q][i]).convert("RGB") for i in idxs]
        with torch.no_grad():
            l_out = model(frames=canonical_batch(frames, a.shorter, a.max_longer, device), part="detr")
            targets = annotations_to_flatten_detr_targets(annotations, device)
            l_idx = criterion.match_last_layer(outputs=l_out, targets=targets, batch_len=10)

        res = {}
        for name, (o, ix) in {"cached": (c_out, c_idx), "live": (l_out, l_idx)}.items():
            torch.manual_seed(1234 + w)
            seq_info = prepare_for_motip(detr_outputs=o, annotations=annotations, detr_indices=ix,
                                         with_gt_boxes=id_criterion.use_spatial_loss)
            si = model(seq_info=dict(seq_info), part="trajectory_modeling")
            logits, gts, masks = model(seq_info=si, part="id_decoder", use_decoder_checkpoint=False)
            loss = id_criterion(id_logits=logits, id_labels=gts, id_masks=masks)
            if id_criterion.use_spatial_loss:
                l1, gi = id_criterion.spatial_loss(id_logits=logits, id_masks=masks, slot_labels=si["trajectory_id_labels"],
                                                   pred_boxes=si["unknown_boxes"], gt_boxes=si["unknown_gt_boxes"])
                loss = loss + id_criterion.l1_weight * l1 + id_criterion.giou_weight * gi
            res[name] = (seq_info, loss.item())
        diffs = {k: (res["cached"][0][k].float() - res["live"][0][k].float()).abs().max().item() for k in res["live"][0]
                 if res["live"][0][k].dtype.is_floating_point and res["live"][0][k].numel()}
        exact = all(torch.equal(res["cached"][0][k], res["live"][0][k]) for k in res["live"][0]
                    if not res["live"][0][k].dtype.is_floating_point)
        worst["tensors"] = max(worst["tensors"], max(diffs.values()))
        worst["loss"] = max(worst["loss"], abs(res["cached"][1] - res["live"][1]))
        print(f"window {w} {q} frames {idxs[0]}..{idxs[-1]}: loss cached {res['cached'][1]:.6f} live {res['live'][1]:.6f}; "
              f"max |d| over seq_info tensors {max(diffs.values()):.2e}; integer tensors equal: {exact}", flush=True)

        # information: cached-and-box-flipped vs a live forward on the mirrored frames
        with torch.no_grad():
            mirrored = [ImageOps.mirror(f) for f in frames[:8]]
            f_out = model(frames=canonical_batch(mirrored, a.shorter, a.max_longer, device), part="detr")
            f_targets = [{"boxes": flip_boxes(t["boxes"]), "labels": t["labels"]} for t in targets[:8]]
            f_idx = criterion.match_last_layer(outputs=f_out, targets=f_targets, batch_len=10)
        for i, (src, tgt) in enumerate(f_idx):
            order = torch.argsort(tgt)
            live_e, live_b = f_out["outputs"][i][src[order]], f_out["pred_boxes"][i][src[order]]
            n = len(order)
            if n:
                flip_cos.append(torch.nn.functional.cosine_similarity(live_e, c_out["outputs"][i, :n], dim=-1).mean().item())
                flip_box.append((live_b - flip_boxes(c_out["pred_boxes"][i, :n])).abs().max().item())
    print(f"\nworst over {a.windows} windows: seq_info tensors {worst['tensors']:.2e}, loss {worst['loss']:.2e}")
    if flip_cos:
        print(f"flip approximation (information): mean cos(live mirrored embedding, cached embedding) {np.mean(flip_cos):.4f} "
              f"(min over frames {np.min(flip_cos):.4f}); max |box diff| mean {np.mean(flip_box):.4f}")


if __name__ == "__main__":
    main()
