"""Run a trained DETR stage-1 checkpoint on an image or video frame.

Usage (from third_party/MOTIP/):
  python scripts/eval_detr_checkpoint.py \
      --checkpoint /path/to/checkpoint_4.pth \
      --image /path/to/frame.jpg \
      [--threshold 0.3] [--output out.jpg] [--max-size 1440]

Or point at a video and pick a frame:
  python scripts/eval_detr_checkpoint.py \
      --checkpoint /path/to/checkpoint_4.pth \
      --video /path/to/clip.mp4 \
      [--frame 0] [--threshold 0.3] [--output out.jpg]
"""
import argparse
import sys
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.util import load_super_config, update_config
from utils.misc import yaml_to_dict
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from structures.args import Args
from utils.nested_tensor import nested_tensor_from_tensor_list


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

AMF_CONFIG_PATH = "./configs/pretrain_r50_deformable_detr_amf.yaml"


def build_config(config_path: str) -> dict:
    config = yaml_to_dict(config_path)
    config = load_super_config(config, config_path)
    config = update_config(config)
    config.setdefault("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    return config


def build_detr(config: dict):
    args = Args()
    args.backbone          = config["BACKBONE"]
    args.lr_backbone       = config["LR"] * config["LR_BACKBONE_SCALE"]
    args.dilation          = config["DILATION"]
    args.num_classes       = config["NUM_CLASSES"]
    args.device            = config["DEVICE"]
    args.num_queries       = config["DETR_NUM_QUERIES"]
    args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    args.aux_loss          = config["DETR_AUX_LOSS"]
    args.with_box_refine   = config["DETR_WITH_BOX_REFINE"]
    args.two_stage         = config["DETR_TWO_STAGE"]
    args.hidden_dim        = config["DETR_HIDDEN_DIM"]
    args.masks             = config["DETR_MASKS"]
    args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    args.nheads            = config["DETR_NUM_HEADS"]
    args.enc_layers        = config["DETR_ENC_LAYERS"]
    args.dec_layers        = config["DETR_DEC_LAYERS"]
    args.dim_feedforward   = config["DETR_DIM_FEEDFORWARD"]
    args.dropout           = config["DETR_DROPOUT"]
    args.dec_n_points      = config["DETR_DEC_N_POINTS"]
    args.enc_n_points      = config["DETR_ENC_N_POINTS"]
    args.cls_loss_coef     = config["DETR_CLS_LOSS_COEF"]
    args.bbox_loss_coef    = config["DETR_BBOX_LOSS_COEF"]
    args.giou_loss_coef    = config["DETR_GIOU_LOSS_COEF"]
    args.focal_alpha       = config["DETR_FOCAL_ALPHA"]
    args.set_cost_class    = config["DETR_SET_COST_CLASS"]
    args.set_cost_bbox     = config["DETR_SET_COST_BBOX"]
    args.set_cost_giou     = config["DETR_SET_COST_GIOU"]
    detr, _, _ = build_deformable_detr(args=args)
    return detr


def load_image(path: str, max_size: int) -> tuple:
    """Returns (tensor CHW float32, orig_hw)."""
    img_bgr = cv2.imread(path)
    assert img_bgr is not None, f"Could not read {path}"
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize so max side <= max_size
    scale = min(max_size / max(orig_h, orig_w), 1.0)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img_rgb = cv2.resize(img_rgb, (new_w, new_h))
    tensor = TF.to_tensor(img_rgb)                              # [0,1] float CHW
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor, (orig_h, orig_w), (new_h, new_w)


def frame_from_video(video_path: str, frame_idx: int, tmp_path: str = "/tmp/_detr_frame.jpg") -> str:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"Could not read frame {frame_idx} from {video_path}"
    cv2.imwrite(tmp_path, frame)
    return tmp_path


def run(checkpoint: str, image_path: str, threshold: float, max_size: int, output: str):
    config = build_config(AMF_CONFIG_PATH)
    device = torch.device(config["DEVICE"])

    print(f"Building DETR ({config['BACKBONE']}, {config['DETR_NUM_QUERIES']} queries)...")
    detr = build_detr(config).to(device).eval()

    print(f"Loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    detr.load_state_dict(state["model"])

    tensor, orig_hw, resized_hw = load_image(image_path, max_size)
    nested = nested_tensor_from_tensor_list([tensor]).to(device)

    print("Running inference...")
    with torch.no_grad():
        out = detr(nested)

    # out["pred_logits"]: [1, Q, num_classes]  (logits, use sigmoid for focal)
    # out["pred_boxes"]:  [1, Q, 4]            (cx cy w h, normalised)
    scores = out["pred_logits"][0].sigmoid().max(-1).values.cpu()
    boxes  = out["pred_boxes"][0].cpu()

    keep = scores > threshold
    scores = scores[keep]
    boxes  = boxes[keep]
    print(f"  {keep.sum().item()} detections above threshold {threshold}")

    # Draw on the resized image
    img_bgr = cv2.imread(image_path)
    rh, rw = resized_hw
    oh, ow = orig_hw
    img_draw = cv2.resize(img_bgr, (rw, rh))

    for score, box in sorted(zip(scores.tolist(), boxes.tolist()), reverse=True):
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * rw)
        y1 = int((cy - h / 2) * rh)
        x2 = int((cx + w / 2) * rw)
        y2 = int((cy + h / 2) * rh)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, f"{score:.2f}", (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imwrite(output, img_draw)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image")
    group.add_argument("--video")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-size", type=int, default=1440)
    parser.add_argument("--output", default="detr_eval_out.jpg")
    args = parser.parse_args()

    image_path = args.image
    if args.video:
        image_path = frame_from_video(args.video, args.frame)

    run(args.checkpoint, image_path, args.threshold, args.max_size, args.output)


if __name__ == "__main__":
    main()
