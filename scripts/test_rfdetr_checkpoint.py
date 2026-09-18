"""
Throwaway script: load a stage-1 RF-DETR checkpoint and run detection on a
single image to check the model has actually learned something.
Usage (run from MOTIP root):
  python scripts/test_rfdetr_checkpoint.py \
      --checkpoint /path/to/checkpoint_0.pth \
      --image     /path/to/frame.jpg \
      [--thresh 0.3] [--out detections.jpg]
"""
import argparse, os, sys
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image, ImageDraw

_MOTIP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MOTIP not in sys.path:
    sys.path.insert(0, _MOTIP)

from configs.util import load_super_config, update_config, yaml_to_dict
from models.motip import build as build_motip_model

def load_model(checkpoint_path, config_path, device):
    cfg = yaml_to_dict(config_path)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    import argparse
    update_config(cfg, argparse.Namespace())
    model, _ = build_motip_model(cfg)
    model = model.detr   # stage-1 checkpoint only saves the DETR sub-model
    model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

def preprocess(image_path, device):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    scale = min(1440 / max(H, W), 1.0)
    nH, nW = int(H * scale), int(W * scale)
    img = img.resize((nW, nH), Image.BILINEAR)
    tensor = F.to_image(img)
    tensor = F.to_dtype(tensor, torch.float32, scale=True)
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(device)
    mask = torch.zeros(1, nH, nW, dtype=torch.bool, device=device)
    from utils.nested_tensor import NestedTensor
    return NestedTensor(tensor, mask), (nH, nW), (H, W)

def draw_boxes(image_path, boxes_xyxy, scores, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for box, score in zip(boxes_xyxy, scores):
        x0, y0, x1, y1 = [int(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, max(0, y0 - 12)), f"{score:.2f}", fill="red")
    img.save(out_path)

@torch.no_grad()
def run(checkpoint, image, config, thresh, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint, config, device)
    nested, (nH, nW), (origH, origW) = preprocess(image, device)
    outputs = model(nested)
    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]
    num_classes = cfg.get("NUM_CLASSES", 1)
    scores = logits[..., :num_classes].sigmoid().max(-1).values
    keep = scores > thresh
    scores_k = scores[keep].cpu()
    boxes_k = boxes[keep].cpu()
    print(f"Detections above thresh={thresh}: {keep.sum().item()}")
    for i, (s, b) in enumerate(zip(scores_k, boxes_k)):
        cx, cy, w, h = b.tolist()
        print(f"  [{i:3d}] score={s:.3f}  cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")
    sx, sy = origW / nW, origH / nH
    xyxy = torch.stack([
        (boxes_k[:, 0] - boxes_k[:, 2] / 2) * nW * sx,
        (boxes_k[:, 1] - boxes_k[:, 3] / 2) * nH * sy,
        (boxes_k[:, 0] + boxes_k[:, 2] / 2) * nW * sx,
        (boxes_k[:, 1] + boxes_k[:, 3] / 2) * nH * sy,
    ], dim=1)
    draw_boxes(image, xyxy, scores_k, out)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--config", default="./configs/pretrain_rfdetr_hockey_real.yaml")
    p.add_argument("--thresh", type=float, default=0.3)
    p.add_argument("--out", default="detections.jpg")
    args = p.parse_args()
    run(args.checkpoint, args.image, args.config, args.thresh, args.out)
