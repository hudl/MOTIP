"""Table-8-style statistics of the trajectory ID-switch augmentation on real training clips.

Samples clips with the real sampler + transform pipeline (blank images of the right size: only the boxes and
IDs matter here, and it keeps the run light), then re-applies the final ``TurnIntoTrajectoryAndUnknown``
step to the *same* clips for every (mode, selection prob) and reports, per setting:

  swap_rate  swapped objects / visible objects
  far_frac   fraction of swapped objects whose two boxes do not overlap (GIoU <= 0)
  mean_giou  mean GIoU of the swapped pairs

Usage (from third_party/MOTIP):
  python scripts/measure_switch_stats.py --config configs/finetune_crossing_rfdetr_s1_long.yaml \
      --data-root /mnt/s3files/.../motip_crossing_dataset_s1_long --n-clips 60 --probs 0.5 1.0
"""
import argparse
import copy
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.util import load_super_config
from data import build_dataset
from data.joint_dataset import JointDataset
from data.naive_sampler import NaiveSampler
from data.transforms import TurnIntoTrajectoryAndUnknown
from utils.misc import yaml_to_dict
from utils.switch_stats import add_switch_stats, clip_switch_stats, empty_switch_stats, summarize_switch_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n-clips", type=int, default=60)
    ap.add_argument("--probs", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.5])
    ap.add_argument("--modes", nargs="+", default=["uniform", "spatial"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml_to_dict(a.config)
    cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    cfg["DATA_ROOT"] = a.data_root
    cfg["DEVICE"] = "cpu"
    torch.manual_seed(a.seed)

    # Blank images of the sequence's size: boxes and IDs do not depend on the pixels.
    sizes = {}

    def fake_open(path):
        seq_dir = os.path.dirname(os.path.dirname(path))
        if seq_dir not in sizes:
            ini = dict(l.strip().replace(" ", "").split("=", 1) for l in open(os.path.join(seq_dir, "seqinfo.ini")) if "=" in l)
            sizes[seq_dir] = (int(ini["imwidth"]), int(ini["imheight"]))
        return Image.new("RGB", sizes[seq_dir])

    import data.joint_dataset as jd
    jd.Image.open = fake_open

    dataset = build_dataset(config=cfg)
    sampler = NaiveSampler(
        data_source=dataset, sample_steps=cfg["SAMPLE_STEPS"], sample_lengths=cfg["SAMPLE_LENGTHS"],
        sample_intervals=cfg["SAMPLE_INTERVALS"], length_per_iteration=cfg["LENGTH_PER_ITERATION"],
        sample_stride=cfg.get("SAMPLE_STRIDE", 1), short_sequence_mode=cfg.get("SAMPLE_SHORT_SEQUENCES", "drop"),
    )
    sampler.prepare_for_epoch(epoch=0)
    infos = list(iter(sampler))
    step = max(len(infos) // a.n_clips, 1)
    infos = infos[::step][:a.n_clips]
    pre_transforms = dataset.transforms.transforms[:-1]     # everything before TurnIntoTrajectoryAndUnknown
    print(f"{len(infos)} clips of {len(infos[0]['frame_idxs'])} frames, AUG_NUM_GROUPS={cfg['AUG_NUM_GROUPS']}, "
          f"occlusion prob={cfg['AUG_TRAJECTORY_OCCLUSION_PROB']}", flush=True)

    settings = [(m, p, s if m == "spatial" else None) for m in a.modes for p in a.probs
                for s in (a.sigmas if m == "spatial" else [None])]
    totals = {k: empty_switch_stats() for k in settings}
    for i, info in enumerate(infos):
        # JointDataset.__getitem__ minus the final transform:
        saved = dataset.transforms.transforms
        dataset.transforms.transforms = pre_transforms
        images, annotations, metas = dataset[info]
        dataset.transforms.transforms = saved
        for (mode, prob, sigma) in settings:
            t = TurnIntoTrajectoryAndUnknown(
                num_id_vocabulary=cfg["NUM_ID_VOCABULARY"],
                aug_trajectory_occlusion_prob=cfg["AUG_TRAJECTORY_OCCLUSION_PROB"],
                aug_trajectory_switch_prob=prob, aug_trajectory_switch_mode=mode,
                aug_trajectory_switch_sigma=sigma if sigma is not None else 0.5,
            )
            _, out, _ = t(images, copy.deepcopy(annotations), metas)
            add_switch_stats(totals[(mode, prob, sigma)], clip_switch_stats(out))
        if i % 10 == 0:
            print(f"  clip {i + 1}/{len(infos)}", flush=True)

    print(f"\n{'mode':8s} {'prob':>5s} {'sigma':>5s} {'swap_rate':>10s} {'far_frac(GIoU<=0)':>18s} {'mean_giou':>10s} {'visible':>9s} {'swapped':>9s}")
    for k, tot in totals.items():
        r = summarize_switch_stats(tot)
        print(f"{k[0]:8s} {k[1]:5.2f} {'' if k[2] is None else k[2]:>5} {r['swap_rate']:10.4f} {r['far_frac']:18.4f} "
              f"{r['mean_giou']:10.4f} {int(r['visible']):9d} {int(r['swapped']):9d}")


if __name__ == "__main__":
    main()
