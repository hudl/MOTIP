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
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

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
from utils.nested_tensor import nested_tensor_from_tensor_list


def load_model(config_path: str, checkpoint_path: str):
    """Load MOTIP model and return (model, config, accelerator).

    If MOTIP_TRT_ENGINE is set, the RF-DETR detector is replaced with a
    TensorRT engine; only the trajectory/ID decoder weights are loaded from
    the checkpoint.  The returned accelerator is None in that case (the TRT
    wrapper has no PyTorch parameters to prepare).
    """
    cfg = yaml_to_dict(config_path)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))

    trt_engine_path = os.environ.get("MOTIP_TRT_ENGINE", "")
    if trt_engine_path:
        sys.path.insert(0, os.path.join(MOTIP_ROOT, "trt"))
        from trt_wrapper import load_engine, TRTDetectorWrapper

        model, _ = build_motip(config=cfg)
        # Load only non-detector weights (trajectory_modeling + id_decoder).
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw = ck.get("model", ck.get("state_dict", ck))
        model_sd = model.state_dict()
        filtered = {
            k: v for k, v in raw.items()
            if not k.startswith("detr.base.")
            and k in model_sd
            and model_sd[k].shape == v.shape
        }
        model.load_state_dict(filtered, strict=False)
        model.cuda().eval()

        engine = load_engine(trt_engine_path)
        trt_res = int(os.environ.get("MOTIP_TRT_RES", "576"))
        object.__setattr__(model, "detr", TRTDetectorWrapper(engine, res=trt_res))
        sys.stderr.write(
            f"MOTIP server: TRT engine loaded from {trt_engine_path} (res={trt_res})\n"
        )
        return model, cfg, None

    accelerator = Accelerator()
    model, _ = build_motip(config=cfg)
    load_checkpoint(model, path=checkpoint_path)
    model = accelerator.prepare(model)
    model.eval()
    return model, cfg, accelerator


class NumpySeqDataset(Dataset):
    """SeqDataset variant that serves in-memory BGR numpy frames instead of JPEG files.

    Skips the encode→write→read→decode cycle. The transform pipeline is
    identical to SeqDataset; only _load differs.
    """

    def __init__(self, frames, vid_w, vid_h, max_shorter=800, max_longer=1536,
                 size_divisibility=0, dtype=torch.float32):
        self.frames = frames
        self.vid_w = vid_w
        self.vid_h = vid_h
        self.size_divisibility = size_divisibility
        self.dtype = dtype
        self.transform = v2.Compose([
            v2.Resize(size=max_shorter, max_size=max_longer),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        image = self._load(self.frames[item])
        transformed_image = self.transform(image)
        if self.dtype != torch.float32:
            transformed_image = transformed_image.to(self.dtype)
        return nested_tensor_from_tensor_list([transformed_image], self.size_divisibility), item

    def seq_hw(self):
        return self.vid_h, self.vid_w

    @staticmethod
    def _load(frame):
        # OpenCV gives BGR; PIL expects RGB.
        return Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))


def run_detr_pass(model, cfg, frames: list, vid_w: int, vid_h: int) -> list:
    """DETR-only forward over all frames. Returns a list of per-frame detection tuples.

    Each element is ``(scores, categories, boxes, output_embeds)`` — CPU tensors,
    already threshold-filtered by ``RuntimeTracker._get_activate_detections``.
    Pass this as ``det_cache`` to ``run_sequence_from_frames`` to skip re-running
    the backbone in the ID-assignment pass.
    """
    dtype_str = cfg.get("INFERENCE_DTYPE", "FP32")
    dtype = torch.float16 if dtype_str == "FP16" else torch.float32

    dataset = NumpySeqDataset(
        frames=frames,
        vid_w=vid_w,
        vid_h=vid_h,
        max_shorter=cfg.get("INFERENCE_MAX_SHORTER", 800),
        max_longer=cfg.get("INFERENCE_MAX_LONGER", 1536),
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        dtype=dtype,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )
    # Temporary tracker just to call _get_activate_detections with the right thresholds.
    _tracker = RuntimeTracker(
        model=model,
        sequence_hw=dataset.seq_hw(),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=cfg["MISS_TOLERANCE"],
        det_thresh=cfg["DET_THRESH"],
        newborn_thresh=cfg["NEWBORN_THRESH"],
        id_thresh=cfg["ID_THRESH"],
        area_thresh=cfg.get("AREA_THRESH", 0),
        only_detr=True,
        dtype=dtype,
        max_tracks=cfg.get("MAX_TRACKS", 0),
    )
    det_cache = []
    with torch.no_grad():
        for image, _ in loader:
            image.tensors = image.tensors.cuda()
            image.mask = image.mask.cuda()
            detr_out = model(frames=image, part="detr")
            scores, categories, boxes, output_embeds = _tracker._get_activate_detections(detr_out)
            det_cache.append((
                scores.cpu(), categories.cpu(), boxes.cpu(), output_embeds.cpu(),
            ))
    return det_cache


def run_sequence_from_frames(
    model, cfg, frames: list, vid_w: int, vid_h: int,
    det_cache: list | None = None,
) -> dict:
    """Run inference on pre-decoded BGR numpy frames. Returns {frame_1indexed: [detections]}.

    Drop-in for run_sequence when frames are already in memory — no JPEG I/O.

    Pass ``det_cache`` (a slice of the list returned by ``run_detr_pass``) to skip
    re-running the DETR backbone entirely; only the ID decoder runs per frame.
    """
    dtype_str = cfg.get("INFERENCE_DTYPE", "FP32")
    dtype = torch.float16 if dtype_str == "FP16" else torch.float32

    sequence_hw = (vid_h, vid_w)
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
        only_detr=(cfg.get("INFERENCE_ONLY_DETR", False)
                   if cfg.get("INFERENCE_ONLY_DETR") is not None
                   else cfg.get("ONLY_DETR", False)),
        dtype=dtype,
        max_tracks=cfg.get("MAX_TRACKS", 0),
    )

    results = {}

    if det_cache is not None:
        # Fast path: DETR already ran, only run ID decoder.
        for t, (scores, categories, boxes, output_embeds) in enumerate(det_cache):
            runtime_tracker.update_from_detections(
                scores.to("cuda"), categories.to("cuda"),
                boxes.to("cuda"), output_embeds.to("cuda", dtype=dtype),
            )
            track_results = runtime_tracker.get_track_results()
            bboxes_cpu = track_results["bbox"].cpu()
            ids_list = track_results["id"].tolist()
            results[t + 1] = [
                {"track_id": oid, "bbox": [x, y, x + w, y + h]}
                for oid, (x, y, w, h) in zip(ids_list, bboxes_cpu.tolist())
            ]
        return results

    # Normal path: run full model (DETR + ID decoder) per frame.
    dataset = NumpySeqDataset(
        frames=frames,
        vid_w=vid_w,
        vid_h=vid_h,
        max_shorter=cfg.get("INFERENCE_MAX_SHORTER", 800),
        max_longer=cfg.get("INFERENCE_MAX_LONGER", 1536),
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        dtype=dtype,
    )
    sequence_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    for t, (image, _) in enumerate(sequence_loader):
        image.tensors = image.tensors.cuda()
        image.mask = image.mask.cuda()
        runtime_tracker.update(image=image)
        track_results = runtime_tracker.get_track_results()

        bboxes_cpu = track_results["bbox"].cpu()
        ids_list = track_results["id"].tolist()
        results[t + 1] = [
            {"track_id": oid, "bbox": [x, y, x + w, y + h]}
            for oid, (x, y, w, h) in zip(ids_list, bboxes_cpu.tolist())
        ]

    return results


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

    dtype_str = cfg.get("INFERENCE_DTYPE", "FP32")
    dtype = torch.float16 if dtype_str == "FP16" else torch.float32

    sequence_dataset = SeqDataset(
        seq_info=seq_info,
        image_paths=image_paths,
        max_shorter=cfg.get("INFERENCE_MAX_SHORTER", 800),
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

        bboxes_cpu = track_results["bbox"].cpu()
        ids_list = track_results["id"].tolist()
        frame_dets = [
            {"track_id": oid, "bbox": [x, y, x + w, y + h]}
            for oid, (x, y, w, h) in zip(ids_list, bboxes_cpu.tolist())
        ]
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
