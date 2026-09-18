"""Submit a TRT build+bench job to SageMaker on a specific GPU type.

Each job: installs deps, builds a TRT engine on the target GPU, runs
bench_detector.py, and (optionally) bench_tracker.py.  Results land in S3.

Because TRT engines are GPU-architecture-specific, you need a separate job for
each GPU you want numbers from -- you cannot reuse a T4 engine on an A10G.

Usage
-----
    python trt/sagemaker/submit.py --gpu t4
    python trt/sagemaker/submit.py --gpu a10g --variant large --res 1088
    python trt/sagemaker/submit.py --gpu a10g --variant large --res 704
    python trt/sagemaker/submit.py --gpu t4 --skip-tracker

GPU choices
-----------
  t4    ml.g4dn.xlarge   (1x T4,   sm_75, ~300 GB/s)
  a10g  ml.g5.2xlarge    (1x A10G, sm_86, ~600 GB/s, 1.93x T4 for this workload)
  l4    ml.g6.2xlarge    (1x L4,   sm_89, ~448 GB/s, 1.58x T4 -- slower than A10G)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ["SAGEMAKER_SUPPRESS_V2_WARNING"] = "1"

ROLE = "arn:aws:iam::690616407375:role/p-sagemaker-execution-role"
OUTPUT_BUCKET = "s3://hudl-experiments-v1/finlay/motip_trt_bench"
STAGE_DIR = "/tmp/motip_trt_bench_staging"

GPU_INSTANCES = {
    "t4":   "ml.g4dn.xlarge",
    "a10g": "ml.g5.2xlarge",
    "l4":   "ml.g6.2xlarge",
}


def prepare_staging(motip_root: Path) -> None:
    """Copy MOTIP source + trt/ scripts into a staging dir for SageMaker upload."""
    import shutil
    if Path(STAGE_DIR).exists():
        shutil.rmtree(STAGE_DIR)
    shutil.copytree(motip_root, STAGE_DIR, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pth", "*.mp4", "*.avi",
        "outputs", "datasets", "engines",
    ))
    # Copy entrypoint to staging root (SageMaker runs from there)
    entrypoint = motip_root / "trt" / "sagemaker" / "entrypoint.sh"
    shutil.copy(entrypoint, STAGE_DIR)
    size = sum(f.stat().st_size for f in Path(STAGE_DIR).rglob("*") if f.is_file())
    print(f"  staged {size / 1e6:.1f} MB to {STAGE_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=list(GPU_INSTANCES), required=True,
                    help="target GPU type")
    ap.add_argument("--variant", default="large", help="RF-DETR variant")
    ap.add_argument("--res", default="1088", help="resolution (e.g. 1088 or 800x1440)")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--skip-tracker", action="store_true",
                    help="skip bench_tracker.py (faster job, detector numbers only)")
    ap.add_argument("--motip-config", default="configs/eval_stage2_hockey.yaml")
    ap.add_argument("--max-runtime-hours", type=int, default=2)
    args = ap.parse_args()

    import boto3
    import sagemaker
    import sagemaker.image_uris
    import sagemaker.session
    from sagemaker.modules.configs import (
        CheckpointConfig, Compute, Networking, OutputDataConfig,
        SourceCode, StoppingCondition,
    )
    from sagemaker.modules.train import ModelTrainer

    instance_type = GPU_INSTANCES[args.gpu]
    session = sagemaker.session.Session(boto3.Session(region_name="us-east-1"))
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"motip-trt-bench-{args.gpu}-{args.variant}-{args.res}-{timestamp}"

    motip_root = Path(__file__).resolve().parent.parent.parent
    print(f"Preparing staging from {motip_root} ...")
    prepare_staging(motip_root)

    print(f"Retrieving SageMaker image for {instance_type} ...")
    image_uri = sagemaker.image_uris.retrieve(
        framework="pytorch", region=session.boto_region_name,
        version="2.8.0", instance_type=instance_type, image_scope="training",
    )
    print(f"  image: {image_uri}")

    output_prefix = f"{OUTPUT_BUCKET}/{args.gpu}/{args.variant}_{args.res}"
    source_code = SourceCode(source_dir=STAGE_DIR, command="bash entrypoint.sh")
    compute_config = Compute(
        instance_type=instance_type, instance_count=1,
        keep_alive_period_in_seconds=0, volume_size_in_gb=50,
    )
    output_config = OutputDataConfig(s3_output_path=f"{output_prefix}/output")
    checkpoint_config = CheckpointConfig(
        s3_uri=f"{output_prefix}/checkpoints/{job_name}",
        local_path="/opt/ml/checkpoints",
    )
    stopping = StoppingCondition(max_runtime_in_seconds=args.max_runtime_hours * 3600)
    # Same VPC as other MOTIP jobs (for MLflow access, if needed)
    networking = Networking(
        subnets=["subnet-0866e06a57d4d3de7", "subnet-057459b3b7db638cb"],
        security_group_ids=["sg-0cdb02c95c81f8fb7"],
        enable_network_isolation=False,
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        source_code=source_code,
        base_job_name=job_name,
        compute=compute_config,
        stopping_condition=stopping,
        output_data_config=output_config,
        checkpoint_config=checkpoint_config,
        role=ROLE,
        networking=networking,
        tags=[{"key": "Squad", "value": "Faceoff"}],
        environment={
            "TRT_VARIANT": args.variant,
            "TRT_RES": args.res,
            "TRT_BATCHES": args.batches,
            "TRT_SKIP_TRACKER": "1" if args.skip_tracker else "0",
            "TRT_MOTIP_CONFIG": args.motip_config,
            "BENCH_NODE_POOL": args.gpu,
        },
    )

    print(f"\nSubmitting {job_name} on {instance_type} ({args.gpu}) ...")
    model_trainer.train(wait=False)
    print(f"\nSubmitted: {job_name}")
    print(f"Results -> {output_prefix}/output/")
    print(f"Monitor: aws logs tail /aws/sagemaker/TrainingJobs --follow "
          f"--log-stream-name-prefix {job_name}")
    print(f"Fetch results: aws s3 cp {output_prefix}/output/ . --recursive "
          f"--profile rd-thor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
