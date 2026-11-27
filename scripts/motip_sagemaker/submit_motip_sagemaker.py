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

STAGE = sys.argv[1] if len(sys.argv) > 1 else "stage1"  # "stage1" | "stage2" | "stage1-real" | "stage2-real" | "mgpu-smoke" | "stage1-amf" | "stage2-amf" | "stage2-amf-resume" | "stage3-amf-stad" | "rfdetr-stage1-hockey" | "rfdetr-stage1-real" | "rfdetr-native-stage1-real"

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
    "rfdetr-native-stage1-real": "motip_sm_entrypoint_rfdetr_native_stage1.sh",
    "rfdetr-stage2-hockey": "motip_sm_entrypoint_rfdetr_stage2_hockey.sh",
    "rfdetr-stage2-hockey-resume": "motip_sm_entrypoint_rfdetr_stage2_hockey_resume.sh",
    "rfdetr-crossing-finetune": "motip_sm_entrypoint_rfdetr_crossing_finetune.sh",
    "rfdetr-crossing-finetune-s2": "motip_sm_entrypoint_rfdetr_crossing_finetune_s2.sh",
    "rfdetr-crossing-finetune-s1-long": "motip_sm_entrypoint_rfdetr_crossing_finetune_s1_long.sh",
    # MOTIP2 spatial-priors ablation on s1_long (same recipe, frozen RF-DETR, from v1 checkpoint_12):
    #   a = spatial ID switches, b = a + spatial ID loss, c = b + spatial anchor.
    # (Short names: SageMaker caps job names at 60 chars.)
    "rfdetr-s1-long-priors-a": "motip_sm_entrypoint_rfdetr_s1_long_priors_a.sh",
    "rfdetr-s1-long-priors-b": "motip_sm_entrypoint_rfdetr_s1_long_priors_b.sh",
    "rfdetr-s1-long-priors-c": "motip_sm_entrypoint_rfdetr_s1_long_priors_c.sh",
    # c2 = arm c resumed from its own checkpoint_3 after the epoch-4 matcher NaN crash (pretrain channel below).
    "rfdetr-s1-long-priors-c2": "motip_sm_entrypoint_rfdetr_s1_long_priors_c2.sh",
    # + the rink anchor (needs the dataset's rink/ sidecar, in the train and val sequences): d = b + rink anchor,
    # e = c + rink anchor. The train channel is the whole dataset prefix, so rink/ files are downloaded with it.
    "rfdetr-s1-long-priors-d": "motip_sm_entrypoint_rfdetr_s1_long_priors_d.sh",
    "rfdetr-s1-long-priors-e": "motip_sm_entrypoint_rfdetr_s1_long_priors_e.sh",
    # MOTIP2 priors, fair test along the production lineage (RF-DETR joint training, no frozen detector, original recipes):
    #   v1x  = the v1 crossing fine-tune (motip_crossing_dataset, stride 4, length 20, LR 2e-5, from the stage-2 checkpoint_5)
    #   s2x  = RF-DETR stage 2 on motip_hockey_data from the stage-1 detector (rfdetr_best_ema.pth)
    # ctrl = the recipe unchanged, priors = + spatial switches / spatial ID loss / image anchor. Short names: 60-char job names.
    # To run v1x on top of an s2x arm: PRETRAIN_PREFIX_OVERRIDE=<s2x checkpoints/<job> S3 dir> RESUME_CKPT_NAME=checkpoint_5.pth.
    "rfdetr-v1x-ctrl": "motip_sm_entrypoint_rfdetr_v1x_ctrl.sh",
    "rfdetr-v1x-priors": "motip_sm_entrypoint_rfdetr_v1x_priors.sh",
    # v1x-priors + one change each (2026-11-21-joint-ablation-rink-idloss): rinkxy = rink XY anchor (out-scale recipe);
    # rinkdist = rink-distance attention bias; idloss = hard-negative identity loss on the detector embeddings.
    # Same data / pretrain / instance as v1x.
    "rfdetr-v1x-rinkxy": "motip_sm_entrypoint_rfdetr_v1x_rinkxy.sh",
    "rfdetr-v1x-rinkdist": "motip_sm_entrypoint_rfdetr_v1x_rinkdist.sh",
    "rfdetr-v1x-idloss": "motip_sm_entrypoint_rfdetr_v1x_idloss.sh",
    # idloss resumed from its own checkpoint_2 (epochs 3-12) after the epoch-3 NCCL hang (2026-11-22); same output prefix as idloss.
    "rfdetr-v1x-idloss-resume": "motip_sm_entrypoint_rfdetr_v1x_idloss_resume.sh",
    # 2026-11-24 (sessions/2026-11-24-v1x-dist-fixes.md): rinkdist re-run with the LR-schedule fix (cosine from epoch 0) and the
    # RF-DETR criterion matcher NaN guard. Same data / pretrain / instance / cap as rinkdist.
    "rfdetr-v1x-rinkdist-fix": "motip_sm_entrypoint_rfdetr_v1x_rinkdist_fix.sh",
    # ... resumed from its own checkpoint_9 (epochs 10-12) after the epoch-10 CUDA OOM (2026-11-25); same output prefix.
    "rfdetr-v1x-rinkdist-fix-resume": "motip_sm_entrypoint_rfdetr_v1x_rinkdist_fix_resume.sh",
    # Team-cue ablation (frozen RF-DETR from the production v1 checkpoint_12, detector outputs read from a DETR cache, new inputs
    # trained with the ID decoder). One entrypoint; the arm is the stage suffix (TC_ARM): t0 control, t1 + team cue, t2 + team cue + box anchor.
    # Channels: train = lean copy of motip_crossing_dataset (gt / seqinfo, no frames), cache = DETR cache + kit descriptors + kit_stats.json,
    # pretrain = checkpoint_12.pth only. TC_CACHE_URI overrides the cache location.
    "rfdetr-tc-t0": "motip_sm_entrypoint_rfdetr_teamcue.sh",
    "rfdetr-tc-t1": "motip_sm_entrypoint_rfdetr_teamcue.sh",
    "rfdetr-tc-t2": "motip_sm_entrypoint_rfdetr_teamcue.sh",
    # Look-ahead ID decoder (models/motip/lookahead.py): same channels as the team-cue arms (lean dataset, DETR cache of the
    # frozen checkpoint_12 detector, checkpoint_12); LA_ARM = stage suffix: ctrl (lag 0) | l8 (lag 0..8 steps). Config finetune_lookahead_<arm>.yaml.
    "rfdetr-la-ctrl": "motip_sm_entrypoint_rfdetr_lookahead.sh",
    "rfdetr-la-l8": "motip_sm_entrypoint_rfdetr_lookahead.sh",
    # Data-recipe study (2026-11-20 data-curation note): frozen checkpoint_12 detector via its DETR cache over the clean
    # full-clip pool motip_pool_v1 (41 tracking_train clips = tracking_train minus the tracking_val_v1 games, no frames; the cache is built over motip_pool_v2, a superset with identical chunks); POOL_ARM = stage suffix picks
    # configs/finetune_pool_<arm>.yaml and its SAMPLE_MANIFEST. POOL_CACHE_URI overrides the cache location.
    "rfdetr-pool-c1": "motip_sm_entrypoint_rfdetr_pool.sh",
    "rfdetr-pool-w1": "motip_sm_entrypoint_rfdetr_pool.sh",
    "rfdetr-pool-m50": "motip_sm_entrypoint_rfdetr_pool.sh",
    "rfdetr-pool-a1": "motip_sm_entrypoint_rfdetr_pool.sh",
    "rfdetr-pool-vs": "motip_sm_entrypoint_rfdetr_pool.sh",
    "rfdetr-s2x-ctrl": "motip_sm_entrypoint_rfdetr_s2x_ctrl.sh",
    "rfdetr-s2x-priors": "motip_sm_entrypoint_rfdetr_s2x_priors.sh",
    "stage1-stad-ablation": "motip_sm_entrypoint_stage1_stad_ablation.sh",
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
IS_STAD_ABLATION = STAGE == "stage1-stad-ablation"
IS_AMF = IS_AMF_STAGE1 or IS_AMF_STAGE1_RESUME or IS_AMF_STAGE2 or IS_AMF_STAGE2_RESUME or IS_AMF_STAGE2_RESUME2 or IS_AMF_STAD_STAGE2 or IS_AMF_STAD_STAGE3 or IS_AMF_STAD_STAGE3B or IS_AMF_STAD_STAGE3B_CONTINUE or IS_STAD_ABLATION
IS_RFDETR_STAGE1_HOCKEY = STAGE == "rfdetr-stage1-hockey"
IS_RFDETR_STAGE1_REAL = STAGE == "rfdetr-stage1-real"
IS_RFDETR_NATIVE_STAGE1 = STAGE == "rfdetr-native-stage1-real"
IS_RFDETR_V1X = STAGE.startswith("rfdetr-v1x-")
IS_TEAMCUE = STAGE.startswith("rfdetr-tc-")
TC_ARM = STAGE.rsplit("-", 1)[1] if IS_TEAMCUE else None   # t0 | t1 | t2
IS_LOOKAHEAD = STAGE.startswith("rfdetr-la-")
LA_ARM = STAGE.rsplit("-", 1)[1] if IS_LOOKAHEAD else None  # ctrl | l8
IS_POOL = STAGE.startswith("rfdetr-pool-")
POOL_ARM = STAGE.rsplit("-", 1)[1] if IS_POOL else None  # c1 | w1 | m50 | a1 | vs
IS_RFDETR_S2X = STAGE.startswith("rfdetr-s2x-")
IS_V1X_IDLOSS_RESUME = STAGE == "rfdetr-v1x-idloss-resume"
V1X_FIX_ARMS = {"rfdetr-v1x-rinkdist-fix": "rinkdist_fix", "rfdetr-v1x-rinkdist-fix-resume": "rinkdist_fix"}
IS_V1X_RINKDIST_FIX_RESUME = STAGE == "rfdetr-v1x-rinkdist-fix-resume"
V1X_S2X_ARM = ("idloss" if IS_V1X_IDLOSS_RESUME
               else V1X_FIX_ARMS[STAGE] if STAGE in V1X_FIX_ARMS
               else STAGE.rsplit("-", 1)[1] if (IS_RFDETR_V1X or IS_RFDETR_S2X) else None)  # ctrl | priors | rinkxy | rinkdist | idloss | *_fix
# The idloss resume reads exactly this checkpoint (end of epoch 2 of the hung job, with optimizer + scheduler + states):
# The rinkdist_fix resume reads exactly this checkpoint (end of epoch 9 of the OOM job, with optimizer + scheduler + states):
V1X_RINKDIST_FIX_RESUME_CKPT = ("s3://hudl-experiments-v1/finlay/motip_rfdetr_v1x_rinkdist_fix/checkpoints/"
                                "motip-hockey-rfdetr-v1x-rinkdist-fix-2026-11-25-00-25-57/checkpoint_9.pth")
V1X_IDLOSS_RESUME_CKPT = ("s3://hudl-experiments-v1/finlay/motip_rfdetr_v1x_idloss/checkpoints/"
                          "motip-hockey-rfdetr-v1x-idloss-2026-11-21-18-41-57/checkpoint_2.pth")
IS_RFDETR_STAGE2_HOCKEY = STAGE == "rfdetr-stage2-hockey" or IS_RFDETR_S2X
IS_RFDETR_STAGE2_HOCKEY_RESUME = STAGE == "rfdetr-stage2-hockey-resume"
IS_RFDETR_CROSSING_FINETUNE_S2 = STAGE == "rfdetr-crossing-finetune-s2"
# The priors arms share every s1_long setting (data, base checkpoint, instance, runtime); only the config differs.
PRIORS_ARM = STAGE.rsplit("-", 1)[1] if STAGE.startswith("rfdetr-s1-long-priors-") else None
IS_RFDETR_CROSSING_FINETUNE_S1_LONG = STAGE == "rfdetr-crossing-finetune-s1-long" or PRIORS_ARM is not None
# s2/s1_long share the v1 crossing-finetune plumbing (pretrain channel, data layout); only data/output prefixes differ.
IS_RFDETR_CROSSING_FINETUNE = STAGE == "rfdetr-crossing-finetune" or IS_RFDETR_CROSSING_FINETUNE_S2 or IS_RFDETR_CROSSING_FINETUNE_S1_LONG or IS_RFDETR_V1X
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
elif IS_RFDETR_CROSSING_FINETUNE_S2 or IS_RFDETR_CROSSING_FINETUNE_S1_LONG:
    # Frozen RF-DETR with per-epoch validation. A/B: s2 = per-event stride-2 windows,
    # s1_long = stride-1 merged +-100-frame windows (build_crossing_dataset.py defaults).
    _ds = "s2" if IS_RFDETR_CROSSING_FINETUNE_S2 else "s1_long"
    DATA_BUCKET_PREFIX = f"s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset_{_ds}"
    OUTPUT_BUCKET_PREFIX = f"s3://hudl-experiments-v1/finlay/motip_rfdetr_crossing_finetune_{_ds}" + (
        f"_priors_{PRIORS_ARM}" if PRIORS_ARM else ""
    )
    INSTANCE_TYPE = "ml.g5.12xlarge"  # 4x A10G, same as RF-DETR stage 2 and the v1 crossing fine-tune
    NUM_INSTANCES = 1
    MAX_RUNTIME = 36 * 3600  # 13 epochs + 14 val passes
elif IS_LOOKAHEAD:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/teamcue/dataset_lean"
    OUTPUT_BUCKET_PREFIX = f"s3://hudl-experiments-v1/finlay/motip_rfdetr_lookahead_{LA_ARM}"
    INSTANCE_TYPE = os.environ.get("LA_INSTANCE_TYPE", "ml.g5.12xlarge")
    NUM_INSTANCES = 1
    MAX_RUNTIME = 12 * 3600  # estimate ~2 h (ctrl) / ~4-5 h (l8) for 13 cached epochs on 4x A10G
elif IS_POOL:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_pool_v1"
    OUTPUT_BUCKET_PREFIX = f"s3://hudl-experiments-v1/finlay/motip_rfdetr_pool_{POOL_ARM}"
    INSTANCE_TYPE = os.environ.get("POOL_INSTANCE_TYPE", "ml.g5.12xlarge")
    NUM_INSTANCES = 1
    MAX_RUNTIME = 24 * 3600  # fresh ID decoder, ~6M window-frames: ~6.5-8.5 h expected on 4x A10G
elif IS_TEAMCUE:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/teamcue/dataset_lean"
    OUTPUT_BUCKET_PREFIX = f"s3://hudl-experiments-v1/finlay/motip_rfdetr_teamcue_{TC_ARM}"
    INSTANCE_TYPE = os.environ.get("TC_INSTANCE_TYPE", "ml.g5.12xlarge")
    NUM_INSTANCES = 1
    MAX_RUNTIME = 12 * 3600  # ~1.5-3 h expected for 13 cached epochs on 4x A10G
elif IS_RFDETR_V1X:
    # The v1 crossing fine-tune recipe (joint detector + ID decoder). ~1,550 train sequences, 5.7k windows/epoch of 20 frames.
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset"
    OUTPUT_BUCKET_PREFIX = f"s3://hudl-experiments-v1/finlay/motip_rfdetr_v1x_{V1X_S2X_ARM}"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 30 * 3600  # estimate ~10-15 h for 13 epochs (see the 2026-11-19 session note); 30 h cap
    if IS_V1X_RINKDIST_FIX_RESUME:
        MAX_RUNTIME = 7 * 3600  # 3 epochs x ~1.2 h measured + ~25 min setup ~ 4 h (~$23); 7 h cap = $40 max (approved up to $40)
    if IS_V1X_IDLOSS_RESUME:
        MAX_RUNTIME = 12 * 3600  # 10 epochs x ~1.1 h measured + setup; budget ~$62 approved (12 h cap = $68 max)
elif IS_RFDETR_S2X:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_hockey_data"
    OUTPUT_BUCKET_PREFIX = f"s3://hudl-experiments-v1/finlay/motip_rfdetr_s2x_{V1X_S2X_ARM}"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 200 * 3600  # 13 epochs x ~14 h (stage-2 resume comment; my estimate 13 h) ~ 175-180 h; SageMaker allows up to 28 days
elif IS_RFDETR_CROSSING_FINETUNE:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_crossing_dataset"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_rfdetr_crossing_finetune_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 36 * 3600  # 13 epochs on crossing dataset
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
elif IS_RFDETR_NATIVE_STAGE1:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_rfdetr_native_stage1_v1"
    INSTANCE_TYPE = "ml.g5.12xlarge"  # 4x A10G
    NUM_INSTANCES = 1
    MAX_RUNTIME = 72 * 3600  # 20 epochs; RF-DETR native is faster than MOTIP harness (~3h/epoch on 4x A10G)
elif IS_RFDETR_STAGE2_HOCKEY:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_hockey_data"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_rfdetr_stage2_hockey_v2"
    INSTANCE_TYPE = "ml.g5.12xlarge"   # 4x A10G — same as D-DETR stage-2 hockey
    NUM_INSTANCES = 1
    MAX_RUNTIME = 72 * 3600             # 13 epochs on full dataset; D-DETR took ~16h, allow headroom
elif IS_RFDETR_STAGE2_HOCKEY_RESUME:
    DATA_BUCKET_PREFIX = "s3://hudl-experiments-v1/faceoff/metaflow/data/tracking_workgroup/tracking_experiments/motip_hockey_data"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_rfdetr_stage2_hockey_v2"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 72 * 3600             # epochs 5-8: ~4 epochs x ~14h each
elif IS_STAD_ABLATION:
    # Ablation: DETR detection-only training on STAD v2 data.
    # Same strategy as stage1-amf (no ID decoder), same data as stage2-amf-stad.
    # Isolates data vs training-strategy effect on DetA.
    DATA_BUCKET_PREFIX = "s3://hudl-experiments/touchdown/datasets/tracking_stad_v2"
    OUTPUT_BUCKET_PREFIX = "s3://hudl-experiments-v1/finlay/motip_stad_detr_pretrain_ablation"
    INSTANCE_TYPE = "ml.g5.12xlarge"
    NUM_INSTANCES = 1
    MAX_RUNTIME = 30 * 3600  # 20 epochs; expect ~24h on 4x A10G
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
DRY_RUN = os.environ.get("MOTIP_SM_DRY_RUN", "") not in ("", "0")  # resolve + print the job, submit nothing
job_name = f"motip-hockey-{STAGE}-{timestamp}"
# SageMaker caps job names at 63 chars and ModelTrainer appends its own suffix, truncating the result; a base
# longer than 60 can be cut on a '-' (invalid). Drop the 'hockey-' infix for long stage names.
if len(job_name) > 60:
    job_name = f"motip-{STAGE}-{timestamp}"
assert len(job_name) <= 60, f"job name too long for SageMaker: {job_name}"

print(f"Retrieving SageMaker PyTorch training image for {INSTANCE_TYPE}...")
pytorch_image_uri = "<dry-run: pytorch 2.8.0 training image>" if DRY_RUN else sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=session.boto_region_name,
    version="2.8.0",
    instance_type=INSTANCE_TYPE,
    image_scope="training",
)
print(f"Using image: {pytorch_image_uri}")

_prepare = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_motip_staging.sh")
print("Preparing staging directory...")
# The staging dir /tmp/motip_sm_staging is shared and rebuilt (rm -rf + cp) by every submit, and ModelTrainer uploads it at
# train(): two submits at once race on it (2026-11-21: a v1x submit failed that way). Submits therefore hold an exclusive
# lock from staging to upload; a second concurrent submit waits here. Run submits one at a time anyway.
import fcntl
_staging_lock = open("/tmp/motip_sm_staging.lock", "w")
if not DRY_RUN:
    print("Waiting for the staging lock (/tmp/motip_sm_staging.lock)...")
    fcntl.flock(_staging_lock, fcntl.LOCK_EX)
    subprocess.run(["bash", _prepare], check=True)
print(f"Staging ready at {SOURCE_DIR}")

source_code = SourceCode(source_dir=SOURCE_DIR, command=f"bash {ENTRYPOINT}")

compute_config = Compute(
    instance_type=INSTANCE_TYPE,
    instance_count=NUM_INSTANCES,
    keep_alive_period_in_seconds=0,
    volume_size_in_gb=100 if (IS_REAL or IS_RFDETR_STAGE1_REAL or IS_RFDETR_NATIVE_STAGE1 or IS_AMF_STAGE2 or IS_AMF_STAGE2_RESUME or IS_AMF_STAGE2_RESUME2 or IS_AMF_STAD_STAGE2 or IS_AMF_STAD_STAGE3 or IS_AMF_STAD_STAGE3B or IS_AMF_STAD_STAGE3B_CONTINUE or IS_STAD_ABLATION) else 150 if (IS_RFDETR_CROSSING_FINETUNE_S2 or IS_RFDETR_CROSSING_FINETUNE_S1_LONG) else 50,
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
    tags=[{"key": "Squad", "value": "faceoff"}],
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
        "MOTIP_CHECKPOINT_S3": f"{OUTPUT_BUCKET_PREFIX}/checkpoints/{job_name}",
        **({"RESUME_CKPT_NAME": os.environ["RESUME_CKPT_NAME"]} if os.environ.get("RESUME_CKPT_NAME") else {}),
        **({"TC_ARM": TC_ARM} if IS_TEAMCUE else {}),
        **({"LA_ARM": LA_ARM} if IS_LOOKAHEAD else {}),
        **({"POOL_ARM": POOL_ARM, "POOL_INIT": os.environ.get("POOL_INIT", "fresh"),
            "POOL_DETECTOR": os.environ.get("POOL_DETECTOR", "stage1"),
            "POOL_DETR_FILE": "ckpt12_detector_only.pth" if os.environ.get("POOL_DETECTOR") == "ckpt12" else "rfdetr_best_ema.pth"}
           if IS_POOL else {}),
    },
)

# Pretrain channel:
#   stage2-real:        our own stage-1 hockey checkpoint
#   stage1-amf:         SportsMOT COCO weights (same as hockey stage 1)
#   stage2-amf:         our own AMF stage-1 checkpoint (checkpoint_14 is the last one)
#   stage2-amf-resume:  stage-2 checkpoint_1.pth (epoch 1, resume full training)
PRETRAIN_PREFIX = os.environ.get("PRETRAIN_PREFIX_OVERRIDE") or (
    # team-cue arms: the production v1 checkpoint_12 on its own (a copy of the one in .../motip_rfdetr_crossing_finetune_v1/checkpoints/...-13-03-48)
    # pool arms: POOL_INIT=fresh (default) needs a detector-only checkpoint: POOL_DETECTOR=stage1 (default) the ihc-od RF-DETR,
    # POOL_DETECTOR=ckpt12 the production detector; POOL_INIT=ckpt12 the whole checkpoint_12.
    ("s3://hudl-experiments-v1/finlay/teamcue/base_ckpt12" if os.environ.get("POOL_INIT") == "ckpt12"
     else "s3://hudl-experiments-v1/finlay/motip_pool_v1/pretrain" if os.environ.get("POOL_DETECTOR") == "ckpt12"
     else "s3://hudl-experiments-v1/finlay/motip_rfdetr_pretrain")
    if IS_POOL
    else
    "s3://hudl-experiments-v1/finlay/teamcue/base_ckpt12"
    if (IS_TEAMCUE or IS_LOOKAHEAD)
    else
    # v1x idloss resume: the S3 prefix of the single checkpoint file (the channel then holds checkpoint_2.pth only).
    V1X_IDLOSS_RESUME_CKPT
    if IS_V1X_IDLOSS_RESUME
    else
    V1X_RINKDIST_FIX_RESUME_CKPT
    if IS_V1X_RINKDIST_FIX_RESUME
    else
    # priors-c2 resumes arm c from its own epoch-3 checkpoint.
    "s3://hudl-experiments-v1/finlay/motip_rfdetr_crossing_finetune_s1_long_priors_c/checkpoints/motip-hockey-rfdetr-s1-long-priors-c-2026-11-18-16-50-53"
    if PRIORS_ARM == "c2"
    else
    # s2/s1_long A/B continue from the production model: v1 crossing fine-tune checkpoint_12.
    "s3://hudl-experiments-v1/finlay/motip_rfdetr_crossing_finetune_v1/checkpoints/motip-hockey-rfdetr-crossing-finetune-2026-09-14-13-03-48"
    if (IS_RFDETR_CROSSING_FINETUNE_S2 or IS_RFDETR_CROSSING_FINETUNE_S1_LONG)
    else
    "s3://hudl-experiments-v1/finlay/motip_rfdetr_stage2_hockey_v2/checkpoints/motip-hockey-rfdetr-stage2-hockey-resume-2026-09-13-11-26-49"
    if IS_RFDETR_CROSSING_FINETUNE
    else
    "s3://hudl-experiments-v1/finlay/motip_rfdetr_stage2_hockey_v2/checkpoints/motip-hockey-rfdetr-stage2-hockey-2026-09-07-17-58-12"
    if IS_RFDETR_STAGE2_HOCKEY_RESUME
    else
    "s3://hudl-experiments-v1/finlay/motip_rfdetr_pretrain"
    if IS_RFDETR_STAGE2_HOCKEY
    else
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
    DATA_BUCKET_PREFIX if (IS_TEAMCUE or IS_LOOKAHEAD or IS_POOL)
    else DATA_BUCKET_PREFIX if (IS_CROSSING_FINETUNE or IS_CROSSING_FINETUNE_S1 or IS_CROSSING_FINETUNE_S1_RESUME)
    else
    DATA_BUCKET_PREFIX if (IS_AMF_STAGE1 or IS_AMF_STAGE1_RESUME)  # amfb_detection/train/images/ is at root
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE2  # STADTracking: train/ at bucket root, no sub_dir wrapper
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE3  # same STAD data as stage 2
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE3B  # same STAD data
    else DATA_BUCKET_PREFIX if IS_AMF_STAD_STAGE3B_CONTINUE  # same STAD data
    else DATA_BUCKET_PREFIX if IS_STAD_ABLATION  # STAD v2 at channel root, no sub_dir wrapper
    else f"{DATA_BUCKET_PREFIX}/motip_hockey_data" if IS_REAL
    else DATA_BUCKET_PREFIX if IS_RFDETR_CROSSING_FINETUNE  # crossing data layout
    else DATA_BUCKET_PREFIX if IS_RFDETR_STAGE2_HOCKEY_RESUME  # same data layout as original
    else DATA_BUCKET_PREFIX if IS_RFDETR_STAGE2_HOCKEY  # hockey stage-2 data root has Hockey/ subdir
    else DATA_BUCKET_PREFIX if IS_RFDETR_STAGE1_HOCKEY  # channel root IS the data dir
    else f"{DATA_BUCKET_PREFIX}/motip_hockey_data" if IS_RFDETR_STAGE1_REAL
    else f"{DATA_BUCKET_PREFIX}/motip_hockey_data" if IS_RFDETR_NATIVE_STAGE1
    else f"{DATA_BUCKET_PREFIX}/data/motip_hockey_data" if not (IS_AMF_STAGE2 or IS_AMF_STAGE2_RESUME or IS_AMF_STAGE2_RESUME2)
    else f"{DATA_BUCKET_PREFIX}/motip_hockey_data"  # mounts AMFTracking/train/ at channel root
)

input_data_config = [InputData(channel_name="train", data_source=TRAIN_DATA_SOURCE)]
if not (IS_RFDETR_STAGE1_HOCKEY or IS_RFDETR_STAGE1_REAL or IS_RFDETR_NATIVE_STAGE1) or IS_RFDETR_STAGE2_HOCKEY or IS_RFDETR_STAGE2_HOCKEY_RESUME or IS_RFDETR_CROSSING_FINETUNE:
    # RF-DETR loads DINOv2 backbone from HuggingFace; no S3 pretrain channel needed
    input_data_config.append(InputData(channel_name="pretrain", data_source=PRETRAIN_PREFIX))

if IS_TEAMCUE or IS_LOOKAHEAD:
    TC_CACHE_URI = os.environ.get("TC_CACHE_URI", "s3://hudl-experiments-v1/finlay/teamcue/detr_cache_ckpt12_kit")
    input_data_config.append(InputData(channel_name="cache", data_source=TC_CACHE_URI))
    print(f"arm {TC_ARM or LA_ARM}: cache channel {TC_CACHE_URI}")

if IS_POOL:
    # caches are built over motip_pool_v2 (all 50 clips; v1's chunks are identical): pool-s1 = stage-1 detector, pool-v2 = ckpt_12's
    POOL_CACHE_URI = os.environ.get("POOL_CACHE_URI", "s3://hudl-experiments-v1/finlay/motip_pool_v2/"
                                    + ("pool-v2" if os.environ.get("POOL_DETECTOR") == "ckpt12" else "pool-s1") + "/cache")
    input_data_config.append(InputData(channel_name="cache", data_source=POOL_CACHE_URI))
    print(f"pool arm {POOL_ARM}: cache channel {POOL_CACHE_URI}")

if DRY_RUN:
    print(f"\n[DRY RUN] stage {STAGE}: job {job_name}, entrypoint {ENTRYPOINT}, {NUM_INSTANCES}x {INSTANCE_TYPE}, "
          f"max runtime {MAX_RUNTIME / 3600:.0f} h")
    print(f"[DRY RUN] checkpoints -> {OUTPUT_BUCKET_PREFIX}/checkpoints/{job_name}")
    for _c in input_data_config:
        print(f"[DRY RUN] channel {_c.channel_name}: {_c.data_source}")
    print(f"[DRY RUN] environment: { {k: v for k, v in model_trainer.environment.items()} }")
    print("[DRY RUN] nothing submitted (unset MOTIP_SM_DRY_RUN to submit).")
    sys.exit(0)
print("\nSubmitting job...")
model_trainer.train(input_data_config=input_data_config, wait=False)
fcntl.flock(_staging_lock, fcntl.LOCK_UN)   # source uploaded: the next submit may restage
print(f"\nSubmitted job: {job_name}")
print(f"Monitor: https://console.aws.amazon.com/sagemaker/home?region={session.boto_region_name}#/jobs")
print(f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {job_name}")
