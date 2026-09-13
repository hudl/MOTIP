#!/bin/bash
set -euo pipefail
echo "=== RF-DETR native stage-1 detection training (IA-BCE loss, no MOTIP harness) ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')":${LD_LIBRARY_PATH:-}
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

echo "=== Installing rfdetr 1.10.0 (transformers-v5 compatible) ==="
pip install --no-cache-dir "rfdetr[train,loggers]==1.10.0"

echo "=== Converting MOT hockey data -> RF-DETR COCO format ==="
python - <<'PYEOF'
import os, json, configparser
from pathlib import Path

TRAIN_ROOT = Path(os.environ["SM_CHANNEL_TRAIN"]) / "Hockey"
OUT_DIR = Path("/tmp/rfdetr_data")
OUT_DIR.mkdir(exist_ok=True)

CATEGORIES = [
    {"id": 1, "name": "player",     "supercategory": "person"},
    {"id": 2, "name": "goalkeeper", "supercategory": "person"},
    {"id": 3, "name": "referee",    "supercategory": "person"},
]

def convert_split(mot_split_dir, rfdetr_name):
    out_split_dir = OUT_DIR / rfdetr_name
    out_split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations, ann_id, img_id = [], [], 0, 0

    for seq_dir in sorted(mot_split_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        cfg = configparser.ConfigParser()
        cfg.read(seq_dir / "seqinfo.ini")
        img_width  = int(cfg["Sequence"]["imWidth"])
        img_height = int(cfg["Sequence"]["imHeight"])
        seq_name = seq_dir.name

        link_path = out_split_dir / seq_name
        if not link_path.exists():
            os.symlink(seq_dir, link_path)

        img1_dir = seq_dir / "img1"
        frame_to_imgid = {}
        for img_path in sorted(img1_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img_id += 1
            frame_num = int(img_path.stem)
            images.append({
                "id": img_id,
                "file_name": f"{seq_name}/img1/{img_path.name}",
                "width": img_width,
                "height": img_height,
            })
            frame_to_imgid[frame_num] = img_id

        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"  WARNING: no gt.txt in {seq_dir}")
            continue
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                frame_id = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                class_id  = int(float(parts[7]))
                if frame_id not in frame_to_imgid or class_id not in (1, 2, 3):
                    continue
                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": frame_to_imgid[frame_id],
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })

    ann_file = out_split_dir / "_annotations.coco.json"
    with open(ann_file, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES}, f)
    print(f"  {mot_split_dir.name} -> {rfdetr_name}: {len(images)} images, {len(annotations)} annotations")

for mot_name, rfdetr_name in [("train", "train"), ("val", "valid")]:
    mot_split = TRAIN_ROOT / mot_name
    if mot_split.exists():
        convert_split(mot_split, rfdetr_name)
    else:
        print(f"WARNING: {mot_split} not found")

# RF-DETR loads a test split; reuse valid to avoid FileNotFoundError
test_dir = OUT_DIR / "test"
if not test_dir.exists():
    os.symlink(OUT_DIR / "valid", test_dir)
print("Conversion complete:", [d.name for d in OUT_DIR.iterdir()])
PYEOF

echo "=== Writing rfdetr_train.py ==="
cat > /tmp/rfdetr_train.py <<'TRAINEOF'
"""RF-DETR native torchrun entry point (SageMaker)."""
import argparse
from rfdetr import RFDETRMedium

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", default="/opt/ml/checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_encoder", type=float, default=1.5e-4)
    p.add_argument("--lr_vit_layer_decay", type=float, default=0.8)
    p.add_argument("--warmup_epochs", type=float, default=2.0)
    p.add_argument("--checkpoint_interval", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--run", default=None)
    args = p.parse_args()
    RFDETRMedium().train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_vit_layer_decay=args.lr_vit_layer_decay,
        warmup_epochs=args.warmup_epochs,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
        run_test=False,
        mlflow=True,
        run=args.run,
    )

if __name__ == "__main__":
    main()
TRAINEOF

echo "=== Launching RF-DETR training (4x GPU via torchrun) ==="
torchrun --nproc_per_node=4 /tmp/rfdetr_train.py \
  --dataset_dir /tmp/rfdetr_data \
  --output_dir /opt/ml/checkpoints \
  --epochs 20 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 1e-4 \
  --lr_encoder 1.5e-4 \
  --lr_vit_layer_decay 0.8 \
  --warmup_epochs 2 \
  --checkpoint_interval 1 \
  --num_workers 4 \
  --run rfdetr_native_stage1_hockey

echo "=== Training complete ==="
