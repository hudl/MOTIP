"""Persistent MOTIP inference server.

Loads the model once on startup, then processes sequences on demand.
Protocol: reads JSON-line requests from stdin, writes JSON-line responses to stdout.

Request format:
  {"seq_dir": "/path/to/seq_dir", "seq_name": "t3_123"}
  seq_dir must contain img1/ with numbered .jpg files and seqinfo.ini.

Response format:
  {"status": "ok", "tracks": {<frame_1indexed>: [{"track_id": int, "bbox": [x1,y1,x2,y2]}]}}
  or {"status": "error", "message": "..."}

Startup signal: prints "MOTIP_READY" to stdout when model is loaded.
"""
import json
import os
import sys
import time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import torch
from torch.utils.data import DataLoader

# Add MOTIP to path
MOTIP_ROOT = os.environ.get("MOTIP_ROOT", "/workspaces/sip-tracking-experiments/third_party/MOTIP")
sys.path.insert(0, os.path.join(MOTIP_ROOT, "models", "ops"))
sys.path.insert(0, MOTIP_ROOT)

from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from data.seq_dataset import SeqDataset
from models.runtime_tracker import RuntimeTracker
from models.motip import build as build_motip
from models.misc import load_checkpoint
from accelerate import Accelerator


def load_model(config_path: str, checkpoint_path: str):
    """Load MOTIP model and return (model, config, accelerator)."""
    cfg = yaml_to_dict(config_path)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))

    accelerator = Accelerator()
    model, _ = build_motip(config=cfg)
    load_checkpoint(model, path=checkpoint_path)
    model = accelerator.prepare(model)
    return model, cfg, accelerator


def run_sequence(model, cfg, seq_dir: str, seq_name: str) -> dict:
    """Run inference on one sequence. Returns {frame_1indexed: [detections]}."""
    from configparser import ConfigParser

    ini = ConfigParser()
    ini.read(os.path.join(seq_dir, "seqinfo.ini"))
    seq_info = {
        "name": seq_name,
        "img_dir": os.path.join(seq_dir, ini["Sequence"]["imdir"]),
        "seq_length": int(ini["Sequence"]["seqlength"]),
        "width": int(ini["Sequence"]["imwidth"]),
        "height": int(ini["Sequence"]["imheight"]),
        "ext": ini["Sequence"].get("imext", ".jpg"),
    }

    image_paths = []
    for i in range(1, seq_info["seq_length"] + 1):
        image_paths.append(os.path.join(seq_info["img_dir"], f"{i:08d}{seq_info['ext']}"))

    dtype = torch.float32

    sequence_dataset = SeqDataset(
        seq_info=seq_info,
        image_paths=image_paths,
        max_shorter=800,
        max_longer=cfg.get("INFERENCE_MAX_LONGER", 1536),
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        dtype=dtype,
    )
    sequence_loader = DataLoader(
        dataset=sequence_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    sequence_hw = sequence_dataset.seq_hw()
    runtime_tracker = RuntimeTracker(
        model=model,
        sequence_hw=sequence_hw,
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=cfg["MISS_TOLERANCE"],
        det_thresh=cfg["DET_THRESH"],
        newborn_thresh=cfg["NEWBORN_THRESH"],
        id_thresh=cfg["ID_THRESH"],
        area_thresh=cfg.get("AREA_THRESH", 0),
        only_detr=cfg.get("INFERENCE_ONLY_DETR", False) if cfg.get("INFERENCE_ONLY_DETR") is not None else cfg.get("ONLY_DETR", False),
        dtype=dtype,
        max_tracks=cfg.get("MAX_TRACKS", 0),
    )

    results = {}
    for t, (image, image_path) in enumerate(sequence_loader):
        image.tensors = image.tensors.cuda()
        image.mask = image.mask.cuda()
        runtime_tracker.update(image=image)
        track_results = runtime_tracker.get_track_results()

        frame_dets = []
        for obj_id, score, category, bbox in zip(
            track_results["id"],
            track_results["score"],
            track_results["category"],
            track_results["bbox"],
        ):
            x, y, w, h = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
            frame_dets.append({
                "track_id": obj_id.item(),
                "bbox": [x, y, x + w, y + h],
            })
        results[t + 1] = frame_dets

    return results


def main():
    config_path = os.environ.get("MOTIP_CONFIG",
        "/workspaces/sip-tracking-experiments/third_party/MOTIP/configs/eval_stage2_hockey.yaml")
    checkpoint_path = os.environ.get("MOTIP_CHECKPOINT",
        "/workspaces/sip-tracking-experiments/third_party/MOTIP/outputs/motip_crossing_finetune_v1/checkpoint_20.pth")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.stderr.write(f"MOTIP server: loading model from {checkpoint_path}...\n")
    model, cfg, accelerator = load_model(config_path, checkpoint_path)
    sys.stderr.write("MOTIP server: model loaded.\n")

    # Signal ready
    sys.stdout.write("MOTIP_READY\n")
    sys.stdout.flush()

    # Main loop: read requests from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            seq_dir = req["seq_dir"]
            seq_name = req.get("seq_name", "t3_seq")

            t0 = time.time()
            tracks = run_sequence(model, cfg, seq_dir, seq_name)
            elapsed = time.time() - t0

            resp = {"status": "ok", "tracks": tracks, "elapsed": elapsed}
        except Exception as e:
            resp = {"status": "error", "message": str(e)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
