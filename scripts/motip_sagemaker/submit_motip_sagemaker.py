"""Standalone MOTIP SageMaker smoke-test submission.

The shared ihc_common.training.train_sagemaker launcher is built for the
Lightning-CLI-shaped pipelines (data.init_args train/val/test paths,
trainer.devices, a Lightning-specific default entrypoint) and requires a
project-local entrypoints/prepare_sagemaker_source.sh that MOTIP doesn't
have. Rather than force-fit MOTIP into that shape, this reuses the same
proven building blocks (IAM role, bucket convention, ModelTrainer,
image_uris.retrieve) directly, with a MOTIP-specific entrypoint.
"""
import os
import subprocess
import sys
from datetime import datetime

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ["SAGEMAKER_SUPPRESS_V2_WARNING"] = "1"

import boto3
import sagemaker
import sagemaker.session
import sagemaker.image_uris
from sagemaker.modules.configs import (
    CheckpointConfig,
    Compute,
    InputData,
    Networking,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.modules.train import ModelTrainer

STAGE = sys.argv[1] if len(sys.argv) > 1 else "stage1"  # "stage1" | "stage2" | "stage1-real" | "stage2-real" | "mgpu-smoke" | "stage1-amf" | "stage2-amf" | "stage2-amf-resume" | "stage3-amf-stad" | "rfdetr-stage1-hockey" | "rfdetr-stage1-real"

ENTRYPOINTS = {
    "stage1": "motip_sm_entrypoint.sh",
    "stage2": "motip_sm_entrypoint_stage2.sh",
    "stage1-real": "motip_sm_entrypoint_stage1_real.sh",
    "stage2-real": "motip_sm_entrypoint_stage2_real.sh",
    "mgpu-smoke": "motip_sm_entrypoint_mgpu_smoke.sh",
    "stage1-amf": "motip_sm_entrypoint_stage1_amf.sh",
    "stage1-amf-resume": "motip_sm_entrypoint_stage1_amf_resume.sh",
    "stage2-amf": "motip_sm_entrypoint_stage2_amf.sh",
    "stage2-amf-resume": "motip_sm_entrypoint_stage2_amf_resume.sh",
    "stage2-amf-resume2": "motip_sm_entrypoint_stage2_amf_resume2.sh",
    "crossing-finetune": "motip_sm_entrypoint_crossing_finetune.sh",
    "crossing-finetune-s1": "motip_sm_entrypoint_crossing_finetune_s1.sh",
    "crossing-finetune-s1-resume": "motip_sm_entrypoint_crossing_finetune_s1_resume.sh",
    "stage2-amf-stad": "motip_sm_entrypoint_stage2_amf_stad.sh",
    "stage3-amf-stad": "motip_sm_entrypoint_stage3_amf_stad.sh",
    "stage3b-amf-stad": "motip_sm_entrypoint_stage3b_amf_stad.sh",
    "stage3b-continue-amf-stad": "motip_sm_entrypoint_stage3b_continue_amf_stad.sh",
    "rfdetr-stage1-hockey": "motip_sm_entrypoint_rfdetr_stage1_hockey.sh",
    "rfdetr-stage1-real": "motip_sm_entrypoint_rfdetr_stage1_real.sh",
}
ENTRYPOINT = ENTRYPOINTS[STAGE]
IS_REAL = STAGE.endswith("-real")
IS_STAGE2_REAL = STAGE == "stage2-real"
IS_MGPU_SMOKE = STAGE == "mgpu-smoke"
IS_AMF_STAGE1 = STAGE == "stage1-amf"
IS_AMF_STAGE1_RESUME = STAGE == "stage1-amf-resume"
IS_AMF_STAGE2 = STAGE == "stage2-amf"
IS_AMF_STAGE2_RESUME = STAGE == "stage2-amf-resume"
IS_AMF_STAGE2_RESUME2 = STAGE == "stage2-amf-resume2"
IS_AMF_STAD_STAGE2 = STAGE == "stage2-amf-stad"
IS_AMF_STAD_STAGE3 = STAGE == "stage3-amf-stad"
IS_AMF_STAD_STAGE3B = STAGE == "stage3b-amf-stad"
IS_AMF_STAD_STAGE3B_CONTINUE = STAGE == "stage3b-continue-amf-stad"
IS_AMF = IS_AMF_STAGE1 or IS_AMF_STAGE1_RESUME or IS_AMF_STAGE2 or IS_AMF_STAGE2_RESUME or IS_AMF_STAGE2_RESUME2 or IS_AMF_STAD_STAGE2 or IS_AMF_STAD_STAGE3 or IS_AMF_STAD_STAGE3B or IS_AMF_STAD_STAGE3B_CONTINUE
IS_RFDETR_STAGE1_HOCKEY = STAGE == "rfdetr-stage1-hockey"
IS_RFDETR_STAGE1_REAL = STAGE == "rfdetr-stage1-real"
IS_CROSSING_FINETUNE = STAGE == "crossing-finetune"
IS_CROSSING_FINETUNE_S1 = STAGE == "crossing-finetune-s1"
IS_CROSSING_FINETUNE_S1_RESUME = STAGE == "crossing-finetune-s1-resume"

ROLE = "arn:aws:iam::690616407375:role/p-sagemaker-execution-role"
SOURCE_DIR = "/tmp/motip_sm_staging"

if IS_CROSSING_FINETUNE_S1_RESUME:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset_s1"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_crossing_finetune_s1"
    INSTANCE_TYPE = "ml.g7e.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 36 * 3600  # 3 epochs at ~10.5h each
elif IS_CROSSING_FINETUNE_S1:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset_s1"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_crossing_finetune_s1"
    INSTANCE_TYPE = "ml.g7e.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 12 * 3600  # stride-1 longer sequences, ~8-10h expected
elif IS_CROSSING_FINETUNE:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_crossing_finetune_v2"
    INSTANCE_TYPE = "ml.g7e.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 6 * 3600  # 8 epochs on small dataset, ~2-3h expected
elif IS_AMF_STAGE1:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/amfb_detection"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 30 * 3600  # 20 epochs on 4x A10G, ~24h with buffer
elif IS_AMF_STAGE1_RESUME:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/amfb_detection"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 14 * 3600  # epochs 10-19 (~11 more at ~4300s/epoch)
elif IS_AMF_STAGE2:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/amfb_motip_stage2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 16 * 3600  # 20 epochs of full MOTIP on AMF tracking data
elif IS_AMF_STAGE2_RESUME:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/amfb_motip_stage2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 24 * 3600  # epochs 2-5 (4 more epochs from checkpoint_1)
elif IS_AMF_STAGE2_RESUME2:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/amfb_motip_stage2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 16 * 3600  # epochs 6-8 (3 more epochs from checkpoint_5)
elif IS_AMF_STAD_STAGE2:
    # Dataset is in a different bucket from the usual hudl-experiments-v1.
    DATA_BUCKET_PREFIX = "s3://hudl-experiments/touchdown/datasets/tracking_stad_v2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stad_stage2_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 48 * 3600  # 8 epochs on 32-clip dataset; steps/epoch unknown, generous budget
elif IS_AMF_STAD_STAGE3:
    # Stage 3: ID consolidation — same STAD data, resume from stage-2 checkpoint.
    DATA_BUCKET_PREFIX = "s3://hudl-experiments/touchdown/datasets/tracking_stad_v2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stad_stage3_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 10 * 3600  # 3 epochs, generous buffer
elif IS_AMF_STAD_STAGE3B:
    # Stage 3b: crossing/fast-mover finetuning — longer sequences, later start, small switch prob.
    DATA_BUCKET_PREFIX = "s3://hudl-experiments/touchdown/datasets/tracking_stad_v2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stad_stage3b_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 48 * 3600  # generous buffer; 3 epochs x ~7h each
elif IS_AMF_STAD_STAGE3B_CONTINUE:
    # Stage 3b continue: epochs 3-5 from checkpoint_2, same data and output bucket.
    DATA_BUCKET_PREFIX = "s3://hudl-experiments/touchdown/datasets/tracking_stad_v2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_amf_stad_stage3b_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 48 * 3600  # 3 more epochs x ~6h each
elif IS_RFDETR_STAGE1_HOCKEY:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_hockey_smoketest/data/motip_hockey_data/Hockey"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_rfdetr_stage1_hockey"
    INSTANCE_TYPE = "ml.g5.2xlarge"   # single A10G — RF-DETR ViT-S fits comfortably
    NUM_INSTANCES = 1
    MAX_RUNTIME = 30 * 3600            # 20 epochs; expect ~20-24h on a single A10G
elif IS_RFDETR_STAGE1_REAL:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_rfdetr_stage1_real_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"  # 4x A10G
    NUM_INSTANCES = 1
    MAX_RUNTIME = 90 * 3600  # 20 epochs x ~3.8h each = ~76h on 4x A10G (DINOv2 is slower than ResNet-50)
elif IS_MGPU_SMOKE:
    # Small 20-sequence dataset already in S3 (no download wait), but the
    # real multi-GPU instance — isolate GPU/EFA/distributed issues fast
    # before re-attempting the full dataset.
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_hockey_smoketest"
    OUTPUT_BUCKET_PREFIX = DATA_BUCKET_PREFIX
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 3600
elif IS_REAL:
    # Real training pool, synced automatically from EFS — no manual upload
    # needed (confirmed: /mnt/s3files/faceoff/tracking_workgroup/... mirrors
    # here within moments of EFS writes).
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments"
    OUTPUT_BUCKET_PREFIX = (
        "s3://hudl-experiments-v1/finlay/motip_hockey_stage2_real_v1" if IS_STAGE2_REAL
        else "s3://hudl-experiments-v1/finlay/motip_hockey_stage1_pretrain_v1"
    )
    INSTANCE_TYPE = "ml.g5.12xlarge"  # 4x A10G — real training scale
    NUM_INSTANCES = 1
    # Stage 2 is the intended overnight run (13 epochs, full ID-association
    # head) — give it more runway than stage 1's 6h cap.
    MAX_RUNTIME = 12 * 3600 if IS_STAGE2_REAL else 6 * 3600
else:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_hockey_smoketest"
    OUTPUT_BUCKET_PREFIX = DATA_BUCKET_PREFIX
    INSTANCE_TYPE = "ml.g4dn.xlarge"  # single T4 — parity with the devbox smoke test
    NUM_INSTANCES = 1
    MAX_RUNTIME = 3600

session = sagemaker.session.Session(boto3.Session(region_name="us-east-1"))

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"motip-hockey-{STAGE}-{timestamp}"

print(f"Retrieving SageMaker PyTorch training image for {INSTANCE_TYPE}...")
pytorch_image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=session.boto_region_name,
    version="2.8.0",
    instance_type=INSTANCE_TYPE,
    image_scope="training",
)
print(f"Using image: {pytorch_image_uri}")

_prepare = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_motip_staging.sh")
print("Preparing staging directory...")
subprocess.run(["bash", _prepare], check=True)
print(f"Staging ready at {SOURCE_DIR}")

source_code = SourceCode(source_dir=SOURCE_DIR, command=f"bash {ENTRYPOINT}")

compute_config = Compute(
    instance_type=INSTANCE_TYPE,
    instance_count=NUM_INSTANCES,
    keep_alive_period_in_seconds=0,
    volume_size_in_gb=100 if (IS_REAL or IS_RFDETR_STAGE1_REAL or IS_AMF_STAGE2 or IS_AMF_STAGE2_RESUME or IS_AMF_STAGE2_RESUME2 or IS_AMF_STAD_STAGE2 or IS_AMF_STAD_STAGE3 or IS_AMF_STAD_STAGE3B or IS_AMF_STAD_STAGE3B_CONTINUE) else 50,
)
output_data_config = OutputDataConfig(s3_output_path=f"{OUTPUT_BUCKET_PREFIX}/output")
checkpoint_config = CheckpointConfig(
    s3_uri=f"{OUTPUT_BUCKET_PREFIX}/checkpoints/{job_name}",
    local_path="/opt/ml/checkpoints",
)
stopping_condition = StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME)

# aml-mlflow.hudltools.com is only reachable from inside this VPC — without
# this, the job times out trying to reach it (confirmed: ConnectTimeoutError,
# not an auth/API error). Reusing the same VPC/subnet/security-group IDs the
# action-recognition pipeline already uses successfully for MLflow access.
networking_config = Networking(
    subnets=["subnet-0866e06a57d4d3de7", "subnet-057459b3b7db638cb"],
    security_group_ids=["sg-0cdb02c95c81f8fb7"],
    enable_network_isolation=False,
)

model_trainer = ModelTrainer(
    training_image=pytorch_image_uri,
    source_code=source_code,
    base_job_name=job_name,
    compute=compute_config,
    stopping_condition=stopping_condition,
    output_data_config=output_data_config,
    checkpoint_config=checkpoint_config,
    role=ROLE,
    networking=networking_config,
    tags=[{"key": "Squad", "value": "Faceoff"}],
    environment={
        "MLFLOW_TRACKING_URI": "https://aml-mlflow.hudltools.com",
        "MLFLOW_EXPERIMENT_NAME": (
            "motip-amf-stage1" if (IS_AMF_STAGE1 or IS_AMF_STAGE1_RESUME)
            else "motip-amf-stage2" if (IS_AMF_STAGE2 or IS_AMF_STAGE2_RESUME or IS_AMF_STAGE2_RESUME2)
            else "motip-amf-stad-stage3b" if (IS_AMF_STAD_STAGE3B or IS_AMF_STAD_STAGE3B_CONTINUE)
            else "motip-amf-stad-stage3" if IS_AMF_STAD_STAGE3
            else "motip-amf-stad-stage2" if IS_AMF_STAD_STAGE2
            else f"motip-hockey-{STAGE}"
        ),
        "MLFLOW_WORKSPACE": "faceoff",
        "MLFLOW_RUN_NAME": job_name,
    },
)

# Pretrain channel:
#   stage2-real:        our own stage-1 hockey checkpoint
#   stage1-amf:         SportsMOT COCO weights (same as hockey stage 1)
#   stage2-amf:         our own AMF stage-1 checkpoint (checkpoint_14 is the last one)
#   stage2-amf-resume:  stage-2 checkpoint_1.pth (epoch 1, resume full training)
PRETRAIN_PREFIX = (
    "s3://hudl-experiments-v1/finlay/motip_crossing_finetune_s1/checkpoints/motip-hockey-crossing-finetune-s1-2026-08-02-08-58-41"
    if IS_CROSSING_FINETUNE_S1_RESUME
    else
    "s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1/checkpoints/crossing_finetune_base"
    if (IS_CROSSING_FINETUNE or IS_CROSSING_FINETUNE_S1)
    else
    "s3://hudl-experiments-v1/finlay/motip_hockey_stage1_pretrain_v1/checkpoints/"
    "motip-hockey-stage1-real-2026-07-03-17-41-35"
    if IS_STAGE2_REAL
    else "s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1/checkpoints/"
    "motip-hockey-stage1-amf-2026-07-23-10-44-08"
    if IS_AMF_STAGE2
    else "s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1/checkpoints/"
    "motip-hockey-stage2-amf-resume-2026-07-25-20-22-35"
    if IS_AMF_STAGE2_RESUME2
    else "s3://hudl-experiments-v1/finlay/motip_amf_stage2_v1/checkpoints/"
    "motip-hockey-stage2-amf-2026-07-24-15-18-34"
    if IS_AMF_STAGE2_RESUME
    else "s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1/checkpoints/"
    "motip-hockey-stage1-amf-2026-08-14-22-21-06"
    if IS_AMF_STAGE1_RESUME
    else "s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1/checkpoints/"
    "motip-hockey-stage1-amf-2026-08-16-10-09-44"
    if IS_AMF_STAD_STAGE2
    else "s3://hudl-experiments-v1/finlay/motip_amf_stad_stage2_v1/checkpoints/"
    "motip-hockey-stage2-amf-stad-2026-08-20-09-49-26"
    if IS_AMF_STAD_STAGE3
    else "s3://hudl-experiments-v1/finlay/motip_amf_stad_stage2_v1/checkpoints/"
    "motip-hockey-stage2-amf-stad-2026-08-20-09-49-26"
    if IS_AMF_STAD_STAGE3B
    else "s3://hudl-experiments-v1/finlay/motip_amf_stad_stage3b_v1/checkpoints/"
    "motip-hockey-stage3b-amf-stad-2026-08-23-07-26-45"
    if IS_AMF_STAD_STAGE3B_CONTINUE
    else "s3://hudl-experiments-v1/finlay/motip_hockey_smoketest/pretrain"
)
TRAIN_DATA_SOURCE = (
    DATA_BUCKET_PREFIX if (IS_CROSSING_FINETUNE or IS_CROSSING_FINETUNE_S1 or IS_CROSSING_FINETUNE_S1_RESUME)
    else
    DATA_BUCKET_PREFIX if (IS_AMF_STAGE1 or IS_AMF_STAGE1_RESUME)  # amfb_detection/train/images/ is at root
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE2  # STADTracking: train/ at bucket root, no sub_dir wrapper
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE3  # same STAD data as stage 2
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE3B  # same STAD data
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE3B_CONTINUE  # same STAD data
    else f"{DATA_BUCKET_PREFIX}/motip_hockey_data" if IS_REAL
    else DATA_BUCKET_PREFIX if IS_RFDETR_STAGE1_HOCKEY  # channel root IS the data dir
    else f"{DATA_BUCKET_PREFIX}/motip_hockey_data" if IS_RFDETR_STAGE1_REAL
    else f"{DATA_BUCKET_PREFIX}/data/motip_hockey_data" if not (IS_AMF_STAGE2 or IS_AMF_STAGE2_RESUME or IS_AMF_STAGE2_RESUME2)
    else f"{DATA_BUCKET_PREFIX}/motip_hockey_data"  # mounts AMFTracking/train/ at channel root
)

input_data_config = [InputData(channel_name="train", data_source=TRAIN_DATA_SOURCE)]
if not (IS_RFDETR_STAGE1_HOCKEY or IS_RFDETR_STAGE1_REAL):
    # RF-DETR loads DINOv2 backbone from HuggingFace; no S3 pretrain channel needed
    input_data_config.append(InputData(channel_name="pretrain", data_source=PRETRAIN_PREFIX))

print("\nSubmitting job...")
model_trainer.train(input_data_config=input_data_config, wait=False)
print(f"\nSubmitted job: {job_name}")
print(f"Monitor: https://console.aws.amazon.com/sagemaker/home?region={session.boto_region_name}#/jobs")
print(f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {job_name}")
