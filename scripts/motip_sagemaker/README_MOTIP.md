# MOTIP Training & Inference

End-to-end guide for training MOTIP on American Football (AMF/STAD) and Ice Hockey data,
and running inference with the resulting checkpoints.

## Architecture overview

MOTIP trains in two stages:

1. **Stage 1 — DETR pretrain** (detection only, no ID head)
   - Learns to detect players from static crops
   - Input: detection-format data (images + COCO-style annotations)
   - Output: DETR checkpoint (weights for backbone + encoder + decoder)

2. **Stage 2 — Full MOTIP** (detector + ID association head)
   - Learns to associate detections across frames (track)
   - Input: MOT-format sequences (frames + gt.txt with track IDs)
   - Initialised from stage-1 DETR checkpoint via `--detr-pretrain`
   - Output: full MOTIP checkpoint (can run end-to-end tracking)

## Data locations (S3)

### American football

| Asset | S3 path |
|-------|---------|
| Stage-1 training data (detection) | `s3://hudl-experiments-v1/finlay/amfb_detection/` |
| Stage-2 training data (AMF tracking) | `s3://hudl-experiments-v1/finlay/amfb_motip_stage2/motip_hockey_data/` |
| Stage-2 training data (STAD v2 tracking) | `s3://hudl-experiments/touchdown/datasets/tracking_stad_v2/` |
| Stage-1 checkpoints | `s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1/checkpoints/` |
| Stage-2 checkpoints (AMF) | `s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1/checkpoints/` |
| Stage-2 checkpoints (STAD) | `s3://hudl-experiments-v1/finlay/motip_amf_stad_stage2_v1/checkpoints/` |

### Ice hockey

| Asset | S3 path |
|-------|---------|
| Stage-1/2 training data | `s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/` |
| Crossing fine-tune data | `s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset/` |
| Crossing fine-tune data (stride-1) | `s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset_s1/` |
| Stage-1 checkpoints | `s3://hudl-experiments-v1/finlay/motip_hockey_stage1_pretrain_v1/checkpoints/` |
| Stage-2 checkpoints | `s3://hudl-experiments-v1/finlay/motip_hockey_stage2_real_v1/checkpoints/` |
| Crossing fine-tune checkpoints | `s3://hudl-experiments-v1/finlay/motip_crossing_finetune_v2/checkpoints/` |
| Crossing fine-tune checkpoints (stride-1) | `s3://hudl-experiments-v1/finlay/motip_crossing_finetune_s1/checkpoints/` |

### Shared

| Asset | S3 path |
|-------|---------|
| COCO/SportsMOT pretrained DETR (stage-1 init) | `s3://hudl-experiments-v1/finlay/motip_hockey_smoketest/pretrain/r50_deformable_detr_coco_sportsmot.pth` |

### Key checkpoints

- **Stage-1 final**: `s3://...motip_amf_stage1_pretrain_v1/checkpoints/motip-hockey-stage1-amf-2026-07-23-10-44-08/checkpoint_14.pth`
  - 20 epochs on AMF detection data, saved at epoch 14 (last milestone before LR decay)
- **Stage-2 (current best)**: `s3://...motip_amf_stage2_v1/checkpoints/motip-hockey-stage2-amf-2026-07-24-15-18-34/checkpoint_1.pth`
  - 3 epochs (0,1,2) but only checkpoint_1 was saved (save_per_epoch bug, now fixed)
  - Local copy: `third_party/MOTIP/outputs/motip_amf_stage2_v1/checkpoint_1.pth`

## Training (SageMaker)

### Prerequisites

- Run all submission commands from **inside the devcontainer** (`root@<container_id>:/workspaces/sip-tracking-clean`).
  The devcontainer's venv already has `sagemaker` and `boto3` installed — no separate venv needed.
- AWS credentials must be active (run `aws sts get-caller-identity` to verify).

### Step 1: Prepare staging directory

```bash
bash scripts/motip_sagemaker/prepare_motip_staging.sh
```

This copies `third_party/MOTIP/` to `/tmp/motip_sm_staging/` with entrypoints,
pruning weights/videos/build artifacts.

### Available stages

| Stage key | Sport | Description |
|-----------|-------|-------------|
| `stage1-real` | Ice hockey | Stage 1 — DETR detector pretrain on real hockey tracking data |
| `stage2-real` | Ice hockey | Stage 2 — full MOTIP on real hockey tracking data |
| `crossing-finetune` | Ice hockey | Fine-tune stage-2 checkpoint with crossing-specific config |
| `crossing-finetune-s1` | Ice hockey | Crossing fine-tune from a specific stage-2 checkpoint (stride-1) |
| `stage1-amf` | American football | Stage 1 — DETR detector pretrain on AMF detection dataset |
| `stage2-amf` | American football | Stage 2 — full MOTIP on AMF tracking data |
| `stage2-amf-stad` | American football | Stage 2 — full MOTIP on STAD v2 tracking data (reuses AMF stage-1 checkpoint) |

### Step 2: Submit job

```bash
# American football — stage 1 (detection pretrain):
python scripts/motip_sagemaker/submit_motip_sagemaker.py stage1-amf

# American football — stage 2 (full MOTIP tracking):
python scripts/motip_sagemaker/submit_motip_sagemaker.py stage2-amf

# Ice hockey — stage 1:
python scripts/motip_sagemaker/submit_motip_sagemaker.py stage1-real

# Ice hockey — stage 2:
python scripts/motip_sagemaker/submit_motip_sagemaker.py stage2-real
```

### Step 3: Monitor

```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix <job-name>
```

Or check the SageMaker console: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs

### Training configs

| Stage | Config | Epochs | LR schedule | Instance |
|-------|--------|--------|-------------|----------|
| 1 | `configs/pretrain_r50_deformable_detr_amf.yaml` | 20 | 1e-4, decay at ep15 | ml.g5.12xlarge (4xA10G) |
| 2 | `configs/r50_deformable_detr_motip_amf.yaml` | 3 | 1e-4, decay at ep2 | ml.g5.12xlarge (4xA10G) |

Stage-2 uses `SAMPLE_STRIDE: 10` to keep steps/epoch manageable (~5.8k steps).

### Checkpoint save cadence

Controlled by `--save-checkpoint-per-epoch N` in the entrypoint. The file is
saved when `(epoch + 1) % N == 0`, so:
- N=1: saves every epoch (checkpoint_0, checkpoint_1, checkpoint_2)
- N=2: saves epoch 1 only (for 3-epoch training — misses the final epoch!)

**Current entrypoint uses N=1** (fixed 2026-07-25).

## Converting AMF data to MOT format

Raw AMF tracking plays (JSON + MP4 pairs on S3) must be converted to
DanceTrack/MOT layout before MOTIP can train on them.

```bash
# Convert all plays for a competition and upload:
python scripts/amfb_to_mot.py \
  --s3_root s3://hudl-experiments-v1/finlay/amfb_tracking/dataset_v1 \
  --out_root /tmp/amfb_mot \
  --upload_to s3://hudl-experiments-v1/finlay/amfb_motip_stage2/motip_hockey_data \
  --competition_id 1409

# Convert a single match locally:
python scripts/amfb_to_mot.py \
  --s3_root s3://hudl-experiments-v1/finlay/amfb_tracking/dataset_v1 \
  --out_root /tmp/amfb_mot \
  --match_id 1502813
```

Output layout:
```
<out_root>/AMFTracking/train/<match>_p<part>_<uuid>/
    img1/00000001.jpg ...
    gt/gt.txt
    seqinfo.ini
```

## Running inference (devbox)

All inference runs inside the devcontainer using the project `.venv`.

### Quick eval on local sequences

```bash
# 1. Convert a play to MOT format (if not already done):
python scripts/amfb_to_mot.py --s3_root s3://... --out_root /tmp/amfb_eval --match_id <id>

# 2. Create a seqmap listing which sequences to evaluate:
ls /tmp/amfb_eval/AMFTracking/train/ > /tmp/amfb_eval/AMFTracking/train_seqmap.txt

# 3. Run inference + TrackEval:
cd /workspaces/sip-tracking-experiments/third_party/MOTIP
/workspaces/sip-tracking-experiments/.venv/bin/python submit_and_evaluate.py \
  --config-path /tmp/amfb_eval_config.yaml \
  --inference-mode evaluate
```

### Eval config template

```yaml
SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_amf.yaml

INFERENCE_DATASET: AMFTracking
INFERENCE_SPLIT: train
INFERENCE_MODEL: /workspaces/sip-tracking-experiments/third_party/MOTIP/outputs/motip_amf_stage2_v1/checkpoint_1.pth
INFERENCE_MODE: evaluate
INFERENCE_MAX_LONGER: 1536

DATA_ROOT: /tmp/amfb_eval
OUTPUTS_DIR: /tmp/amfb_eval_out

DET_THRESH: 0.1
NEWBORN_THRESH: 0.1
ID_THRESH: 0.1
MISS_TOLERANCE: 30
```

### Key inference parameters

| Param | Default | Notes |
|-------|---------|-------|
| `DET_THRESH` | 0.5 | Detection confidence threshold. **Use 0.1 for AMFB** — the model outputs very low confidences; 0.3 loses ~9% recall |
| `NEWBORN_THRESH` | 0.5 | Threshold for spawning new tracks. Match to DET_THRESH (0.1 for AMFB) |
| `ID_THRESH` | 0.1 | ID association threshold |
| `MISS_TOLERANCE` | 30 | Frames before a lost track is killed |
| `INFERENCE_MAX_LONGER` | 1536 | Max image dimension (longer side) |

### Important: use the project .venv

```bash
/workspaces/sip-tracking-experiments/.venv/bin/python
```

NOT system python. The `.venv` has torch 2.8+cu128 and a pre-built
MultiScaleDeformableAttention extension. System python (torch 2.3+cu121)
has driver/CUDA mismatches.

## Baseline results (checkpoint_1, 2 sequences from match 1502813)

| det_thresh | HOTA | MOTA | IDF1 | DetRe | DetPr | IDSW |
|------------|------|------|------|-------|-------|------|
| 0.5 | 62.5 | 71.8 | 80.5 | 64.3% | 83.6% | 15 |
| **0.3** | **69.8** | **88.1** | **88.2** | **82.3%** | 78.2% | 29 |

dt=0.3 is better than 0.5, but **0.1 is required for AMFB** — see point_mot results below.

## Checkpoint 5 results (5 epochs, play 38695f12, point_mot eval)

Point-based MOT metrics (dist_threshold=80px) against full AMFB GT (5236 points
across 238 frames). Point association uses bottom-centre of predicted bbox vs GT
ground_position — avoids the partial-bbox problem that inflates IoU FN.

| det_thresh | MOTA | IDF1 | Precision | Recall | IDSW | Mean dist | FP | FN |
|------------|------|------|-----------|--------|------|-----------|----|----| 
| 0.3 | 0.903 | 0.954 | 99.9% | 91.3% | 50 | 3.9 px | 3 | 454 |
| **0.1** | **0.893** | **0.978** | 97.2% | **98.3%** | 323 | 4.7 px | 148 | 87 |

dt=0.1 recovers nearly all GT points (87 FN vs 454) at the cost of more ID
switches (323 vs 50) — the low-confidence detections flicker. MISS_TOLERANCE
or post-processing may help.

## File reference

```
scripts/motip_sagemaker/
  prepare_motip_staging.sh          # Copies MOTIP source to /tmp/motip_sm_staging
  submit_motip_sagemaker.py         # SageMaker job submission (all stages)
  motip_sm_entrypoint_stage1_amf.sh # Stage-1 AMF entrypoint
  motip_sm_entrypoint_stage2_amf.sh # Stage-2 AMF entrypoint
  README_MOTIP.md                   # This file

third_party/MOTIP/
  train.py                          # Training loop
  submit_and_evaluate.py            # Inference + TrackEval
  configs/
    pretrain_r50_deformable_detr_amf.yaml  # Stage-1 config
    r50_deformable_detr_motip_amf.yaml     # Stage-2 config
  data/
    amf_detection.py              # AMFDetection dataset (stage 1)
    amf_tracking.py               # AMFTracking dataset (stage 2)
  outputs/motip_amf_stage2_v1/
    checkpoint_1.pth              # Local copy of best checkpoint

scripts/amfb_to_mot.py                # Raw AMF play -> MOT format converter
```


## Crossing fine-tune (hockey identity correction)

Fine-tunes the stage-2 hockey checkpoint on crossing-focused sequences to
improve re-ID through player crossings (high-IoU overlap events).

### Dataset

Built from golden_50 clips by extracting 80-frame stride-4 windows around each
crossing event (IoU >= 0.45 between two tracks for >= 3 frames).

| Split | Sequences | Source |
|-------|-----------|--------|
| train | 1551 | 41 clips from golden_50 |
| val | 205 | 9 held-out clips (same val split as original hockey training) |

Val clips (by prefix): 005254c3, 08d9c172, 0baebce3, 35a6df02, 39f231f4,
4500df42, a673db5b, a9b71947, b4b96185.

Dataset location: `s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset/`
(also on EFS at `/mnt/s3files/faceoff/tracking_workgroup/tracking_experiments/motip_crossing_dataset/`)

Built by: `/tmp/_build_crossing_dataset.py` (on devbox)

### Training

```bash
# 1. Prepare staging
bash scripts/motip_sagemaker/prepare_motip_staging.sh

# 2. Submit (4x A10G, ~2-3h expected)
python scripts/motip_sagemaker/submit_motip_sagemaker.py crossing-finetune
```

| Config | Base checkpoint | Epochs | LR | Instance |
|--------|----------------|--------|-----|----------|
| `configs/finetune_crossing_v1.yaml` | `stage2_g7e_checkpoint_12.pth` (epoch 12) | 13-21 (8 new) | 2e-5 | ml.g5.12xlarge (4xA10G) |

Key settings:
- `RESUME_OPTIMIZER: False` / `RESUME_SCHEDULER: False` -- fresh optimizer, avoids inheriting stale momentum
- `AUG_*: 0.0` -- colour jitter disabled (torchvision API incompatibility in SageMaker image)
- `SAMPLE_INTERVALS: [4]` / `SAMPLE_LENGTHS: [20]` -- matches training domain (stride-4 sequences)
- `SAVE_CHECKPOINT_PER_EPOCH: 1` -- saves every epoch

### Checkpoints

Output: `s3://hudl-experiments-v1/finlay/motip_crossing_finetune_v1/checkpoints/<job-name>/`

Base checkpoint stored at: `s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1/checkpoints/crossing_finetune_base/stage2_g7e_checkpoint_12.pth`

## Crossing fine-tune (hockey identity correction)

Fine-tunes the stage-2 hockey checkpoint on crossing-focused sequences to
improve re-ID through player crossings (high-IoU overlap events).

### Dataset

Built from golden_50 clips by extracting 80-frame stride-4 windows around each
crossing event (IoU >= 0.45 between two tracks for >= 3 frames).

| Split | Sequences | Source |
|-------|-----------|--------|
| train | 1551 | 41 clips from golden_50 |
| val | 205 | 9 held-out clips (same val split as original hockey training) |

Val clips (by prefix): 005254c3, 08d9c172, 0baebce3, 35a6df02, 39f231f4,
4500df42, a673db5b, a9b71947, b4b96185.

Dataset location: 
(also on EFS at )

Built by:  (copied to devbox )

### Training



| Config | Base checkpoint | Epochs | LR | Instance |
|--------|----------------|--------|-----|----------|
|  |  (epoch 12) | 13→21 (8 new) | 2e-5 | ml.g5.12xlarge (4xA10G) |

Key settings:
-  /  — fresh optimizer, avoids inheriting stale momentum
-  — colour jitter disabled (torchvision API incompatibility in SageMaker image)
-  /  — matches training domain (stride-4 sequences)
-  — saves every epoch

### Checkpoints

Output: 

Base checkpoint stored at: 

### Per-crossing evaluation

Evaluates MOTIP's ability to correct OFSort identity swaps at crossing events.
For each crossing in golden_50, extracts an 80-frame stride-4 window, runs MOTIP
from scratch, and compares its identity assignment against GT.

**Script**: 

**How to run**:


**Important**: The  export is mandatory. Without it, torch in
 cannot find  and falls back to CPU (10x slower).

**Metrics**:
- TP: MOTIP correctly detects a swap that OFSort made
- FP: MOTIP says swap but OFSort was correct
- TN: Both agree no swap happened
- FN: Swap exists but MOTIP missed it
- Net = TP - FP (must be >0 for MOTIP to add value)
- Precision = TP / (TP + FP)

**OFSort baseline**: Gets 150/237 events right (63%), swaps 87/237 (37%).

**Results (2026-08-01)**:
| Checkpoint | Epochs fine-tuned | Precision | Net |
|---|---|---|---|
| 12 (baseline) | 0 | 36.6% | -22 |
| 13 | +1 | 38.8% | -18 |
| 14 | +2 | 38.3% | -19 |
| 16 | +4 | 36.4% | -21 |

Conclusion: Fine-tuning gives marginal improvement at epoch 1 then regresses.
MOTIP precision stays ~37% — below the 50% threshold needed for net-positive
correction.

### Per-crossing evaluation

Evaluates MOTIP's ability to correct OFSort identity swaps at crossing events.
For each crossing in golden_50, extracts an 80-frame stride-4 window, runs MOTIP
from scratch, and compares its identity assignment against GT.

**Script**: `/workspaces/sip-tracking-experiments/_motip_s4_full_eval.py`

**How to run**:
```bash
cd /workspaces/sip-tracking-experiments

# Set LD_LIBRARY_PATH (required -- .venv torch needs cusparseLt)
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

# Run baseline (checkpoint 12):
.venv/bin/python _motip_s4_full_eval.py > /tmp/eval_baseline.log 2>&1

# Run a different checkpoint (copy + sed):
cp _motip_s4_full_eval.py _motip_s4_eval_ckptXX.py
sed -i 's|stage2_g7e_checkpoint_12.pth|../motip_crossing_finetune_v1/checkpoint_XX.pth|' _motip_s4_eval_ckptXX.py
.venv/bin/python _motip_s4_eval_ckptXX.py > /tmp/eval_ckptXX.log 2>&1
```

**Important**: The LD_LIBRARY_PATH export is mandatory. Without it, torch in
.venv cannot find libcusparseLt.so.0 and falls back to CPU (10x slower).

**Metrics**:
- TP: MOTIP correctly detects a swap that OFSort made
- FP: MOTIP says swap but OFSort was correct
- TN: Both agree no swap happened
- FN: Swap exists but MOTIP missed it
- Net = TP - FP (must be >0 for MOTIP to add value)
- Precision = TP / (TP + FP)

**OFSort baseline**: Gets 150/237 events right (63%), swaps 87/237 (37%).

**Results (2026-08-01)**:
| Checkpoint | Epochs fine-tuned | Precision | Net |
|---|---|---|---|
| 12 (baseline) | 0 | 36.6% | -22 |
| 13 | +1 | 38.8% | -18 |
| 14 | +2 | 38.3% | -19 |
| 16 | +4 | 36.4% | -21 |

Conclusion: Fine-tuning gives marginal improvement at epoch 1 then regresses.
MOTIP precision stays ~37% -- below the 50% threshold needed for net-positive
correction.

---

## MOTIP AMF STAD Stage 2

Training on the new STAD tracking dataset v2 (`s3://hudl-experiments/touchdown/datasets/tracking_stad_v2`).

### Stage 1 → Stage 2 lineage

| Item | Value |
|------|-------|
| Stage-1 run | `motip-hockey-stage1-amf-2026-08-16-10-09-44` (full 20 epochs) |
| Stage-1 checkpoint | `.../checkpoint_19.pth` — **final epoch, use this** |
| Stage-1 checkpoint path | `s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1/checkpoints/motip-hockey-stage1-amf-2026-08-16-10-09-44/checkpoint_19.pth` |
| Stage-1 dataset | `s3://hudl-experiments-v1/finlay/amfb_detection/` |
| Stage-2 dataset | `s3://hudl-experiments/touchdown/datasets/tracking_stad_v2/` (dataset still being populated, 1700+ clips as of 2026-08-19) |
| Stage-2 config | `configs/r50_deformable_detr_motip_amf_stad.yaml` |
| Stage-2 output | `s3://hudl-experiments-v1/finlay/motip_amf_stad_stage2_v1/checkpoints/` |
| MLflow experiment | `motip-amf-stad-stage2` |

### Training configs

| Stage | Config | Epochs | LR schedule | Instance |
|-------|--------|--------|-------------|----------|
| 1 | `configs/pretrain_r50_deformable_detr_amf.yaml` | 20 | 1e-4, decay at ep15 | ml.g5.12xlarge |
| 2 | `configs/r50_deformable_detr_motip_amf_stad.yaml` | 8 | 1e-4, decay at ep6 | ml.g5.12xlarge |

Stage-2 uses `SAMPLE_STRIDE: 10`, matching AMF v1 scale (~1700+ clips, dataset still being populated).

### Smoketest (local, before full run)

Run from inside the devcontainer to verify data access, the dataset class, and one training epoch
before committing to a multi-hour SageMaker job:

```bash
bash scripts/motip_sagemaker/run_motip_stad_local_smoketest.sh
```

This downloads 3 STAD clips + `checkpoint_19.pth`, then runs 1 epoch on 1 GPU with
`SAMPLE_STRIDE: 50` (config: `r50_deformable_detr_motip_amf_stad_smoketest.yaml`).
Completes in ~5 minutes. Output checkpoint lands in `/tmp/motip_stad_smoketest_out/`.

### How to run

```bash
# 1. Prepare staging
bash scripts/motip_sagemaker/prepare_motip_staging.sh

# 2. Submit
python scripts/motip_sagemaker/submit_motip_sagemaker.py stage2-amf-stad
```

### Cross-bucket access note

The STAD dataset is in `s3://hudl-experiments` (not the usual `hudl-experiments-v1`).
If the SageMaker job fails to pull training data, confirm that the execution role
`arn:aws:iam::690616407375:role/p-sagemaker-execution-role` has `s3:GetObject`
permission on `arn:aws:s3:::hudl-experiments/*`.
