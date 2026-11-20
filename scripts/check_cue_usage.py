"""Is the new input used? Post-training check of a TEAM_CUE / SPATIAL_ANCHOR checkpoint.

    python scripts/check_cue_usage.py --config configs/finetune_teamcue_t2.yaml --checkpoint <checkpoint_N.pth> \
        --data-root <lean dataset root (Hockey/train/<seq>/gt, seqinfo.ini)> --cache-dir <DETR cache built with --kit> \
        [--init-checkpoint <the checkpoint training started from>] [--batches 16]

Prints, for the checkpoint (cue dropout / noise off):
  1. the learned scalars: team-cue gate, norm of the "missing" vector, box-anchor scale (mean / max), and how far the cue MLP moved from
     its (deterministic) initialisation, and the kit statistics buffers;
  2. the size of each added term relative to the detection features, over ``--batches`` sample windows of the training data;
  3. the ID cross-entropy on the same windows with the cue as trained, with the kit descriptors blanked (all tokens "missing"), with the
     box anchor scaled to 0, and with both off: if the decoder uses an input, removing it must raise the loss.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from configs.util import load_super_config
    from data import build_dataset
    from data.naive_sampler import NaiveSampler
    from data.util import collate_fn
    from models.misc import load_checkpoint
    from models.motip import build as build_motip
    from models.motip.id_criterion import build as build_id_criterion
    from torch.utils.data import DataLoader
    from train import prepare_for_motip
    from utils.detr_cache import DetrCache
    from utils.misc import yaml_to_dict

    cfg = yaml_to_dict(a.config)
    cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    cfg.update(DATA_ROOT=a.data_root, DETR_CACHE_DIR=a.cache_dir, NUM_WORKERS=0, BATCH_SIZE=1, DETR_FREEZE=True)
    dev = torch.device(a.device)

    torch.manual_seed(0)
    model, _ = build_motip(config=cfg)
    init_cue = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("team_cue.")}   # deterministic init (seeded in build)
    load_checkpoint(model, path=a.checkpoint)
    # train mode: the ID decoder only returns its labels / masks (for the loss) in train mode. Nothing stochastic is left on in this model
    # (the decoder has no dropout); the cue's own dropout / noise are switched off here.
    model = model.to(dev).train()
    if model.team_cue is not None:
        model.team_cue.dropout = 0.0
        model.team_cue.noise_std = 0.0
    model.record_cue_stats = True
    cue, box = model.team_cue, model.anchor_scale
    print(f"checkpoint {a.checkpoint}")
    print(f"TEAM_CUE: {cue is not None}   SPATIAL_ANCHOR: {bool(cfg.get('SPATIAL_ANCHOR', False))} (scale param: {box is not None})")
    if cue is not None:
        sd = {k: v for k, v in model.state_dict().items() if k.startswith("team_cue.")}
        print(f"  gate = {float(cue.gate):.4f} (init {float(init_cue['team_cue.gate']):.4f})   |missing| = {float(cue.missing.norm()):.4f}")
        for k in ("team_cue.mlp.0.weight", "team_cue.mlp.2.weight", "team_cue.mlp.2.bias"):
            d = (sd[k].cpu() - init_cue[k]).norm() / init_cue[k].norm().clamp(min=1e-8)
            print(f"  {k}: |w| {float(sd[k].norm()):.3f}, moved {100 * float(d):.1f}% of its init norm")
        print(f"  kit_mean[:4] {cue.kit_mean[:4].tolist()}   kit_std[:4] {cue.kit_std[:4].tolist()}")
    if box is not None:
        print(f"  box anchor scale: mean {float(box.abs().mean()):.4f}  max {float(box.abs().max()):.4f}  min {float(box.abs().min()):.4f}")
    if a.init_checkpoint:
        init_sd = torch.load(a.init_checkpoint, map_location="cpu", weights_only=False)["model"]
        sd = model.state_dict()
        moved = {k: ((sd[k].cpu() - init_sd[k]).norm() / init_sd[k].norm().clamp(min=1e-8)).item()
                 for k in sd if k in init_sd and (k.startswith("id_decoder") or k.startswith("trajectory_modeling")) and init_sd[k].dtype.is_floating_point}
        print(f"  ID decoder / trajectory weights moved from the starting checkpoint: mean {100 * sum(moved.values()) / len(moved):.2f}% "
              f"max {100 * max(moved.values()):.2f}% of their norm ({len(moved)} tensors)")

    dataset = build_dataset(config=cfg)
    sampler = NaiveSampler(data_source=dataset, sample_steps=cfg["SAMPLE_STEPS"], sample_lengths=cfg["SAMPLE_LENGTHS"],
                           sample_intervals=cfg["SAMPLE_INTERVALS"], length_per_iteration=cfg["LENGTH_PER_ITERATION"],
                           sample_stride=cfg.get("SAMPLE_STRIDE", 1), short_sequence_mode=cfg.get("SAMPLE_SHORT_SEQUENCES", "drop"))
    sampler.prepare_for_epoch(epoch=0)
    loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=1, num_workers=0, collate_fn=collate_fn)
    detr_cache = DetrCache(a.cache_dir, with_kit=bool(cfg.get("TEAM_CUE", False)))
    id_criterion = build_id_criterion(config=cfg)

    variants = {"as trained": (False, False), "kit blanked (all tokens missing)": (True, False),
                "box anchor scale -> 0": (False, True), "both removed": (True, True)}
    ce = {k: [] for k in variants}
    saved_scale = box.detach().clone() if box is not None else None
    stride = max(1, len(loader) // a.batches)
    n = 0
    with torch.no_grad():
        for i, samples in enumerate(loader):
            if i % stride:
                continue
            annotations, metas = samples["annotations"], samples["metas"]
            detr_outputs, detr_indices = detr_cache.detr_outputs(metas, annotations, dev)
            seq_info = prepare_for_motip(detr_outputs=detr_outputs, annotations=annotations, detr_indices=detr_indices)
            seq_info = model(seq_info=seq_info, part="trajectory_modeling")
            for name, (blank_kit, zero_box) in variants.items():
                si = dict(seq_info)
                if blank_kit:
                    si.pop("trajectory_kit", None)
                    si.pop("unknown_kit", None)
                if zero_box and box is not None:
                    box.copy_(torch.zeros_like(box))
                logits, labels, masks = model(seq_info=si, part="id_decoder")
                ce[name].append(float(id_criterion(id_logits=logits, id_labels=labels, id_masks=masks)))
                if zero_box and box is not None:
                    box.copy_(saved_scale)
                if name == "as trained":
                    st = model.take_cue_stats()
                    if n == 0:
                        stats = {k: [] for k in st}
                    for k, v in st.items():
                        stats[k].append(v)
                else:
                    model.take_cue_stats()
            n += 1
            if n >= a.batches:
                break
    print(f"\nadded term relative to the detection features (mean over {n} windows, eval mode):")
    for k in sorted(stats):
        if k.endswith(("rms_ratio", "added_rms", "used_frac")):
            print(f"  {k:24s} {sum(stats[k]) / len(stats[k]):.4f}")
    print(f"\nID cross-entropy on the same {n} training windows (lower is better; the cue is used if removing it raises the loss):")
    base = sum(ce["as trained"]) / n
    for name in variants:
        m = sum(ce[name]) / n
        print(f"  {name:36s} {m:.4f}   ({m - base:+.4f})")


if __name__ == "__main__":
    main()
