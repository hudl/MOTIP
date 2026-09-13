"""Run scripts/bench_gpu.py on a chosen EKS GPU node pool, to compare GPUs.

Lives at MOTIP's root on purpose: Metaflow packages the directory containing the
flow file, so putting it here ships MOTIP itself (models/, configs/, utils/,
data/, structures/, scripts/) into the pod. Nothing needs to vendor or copy MOTIP
anywhere, and the harness the pod runs is always the one in this commit.

There is no dependency on aml-ice-hockey. The Hudl cluster wiring is five env
vars, applied by ``apply_cluster_env()`` below with the same values
``aml_ice_hockey.ihc_common.utils.flow_runner_settings`` uses; ``@kubernetes``
comes from Metaflow itself.

Node pools:

    GPU_BENCH_TARGET=g4  → GPU-G4-4   T4,   sm_75, fp16 tensor cores, no bf16 (default)
    GPU_BENCH_TARGET=g5  → GPU-G5-4   A10G, sm_86, bf16 tensor cores
    GPU_BENCH_TARGET=g6  → GPU-G6-4   L4,   sm_89, bf16 + fp8

G5 is what ihc-pipeline's detection and reid steps use. Per its
``2026-08-09-reid-detection-gpu-g5-batch-512`` note, the 6.4x they measured came
from CLIP-ReID's *bf16* autocast gate activating on A10 — a path the T4 does not
have at all. MOTIP runs fp16 today, so do not expect 6.4x here: the honest
expectation is closer to the raw fp16 throughput ratio (~1.9x on paper) less
whatever is launch-bound. MOTIP's deformable attention *does* have a bf16 branch
(``models/ops/modules/ms_deform_attn.py``), so ``--dtypes fp16,bf16`` is worth
running on g5/g6 to see whether the same gate helps the detector.

Target is an env var rather than a Parameter because ``node_selector`` is
evaluated at class-definition time, before Parameter values exist.

Examples
--------
    # On the cluster's T4 pool -- the like-for-like baseline against the devbox
    GPU_BENCH_TARGET=g4 python dag_speed_bench.py run

    # A10G, the pool ihc-pipeline uses, with bf16 in the sweep
    GPU_BENCH_TARGET=g5 python dag_speed_bench.py run --dtypes fp16,bf16

    # L4
    GPU_BENCH_TARGET=g6 python dag_speed_bench.py run --dtypes fp16,bf16

    # Locally, no pod (skips the @kubernetes decorator entirely)
    BENCH_LOCAL=1 python dag_speed_bench.py run

Results come back as Metaflow artifacts, so no PVC is required:

    python dag_speed_bench.py dump <run-id>/benchmark
    # or fetch results_json and write it out:
    python -c "from metaflow import Run; \
        open('g6.json','w').write(Run('SpeedBenchFlow/<run-id>').data.results_json)"

Then compare pools:

    python scripts/bench_gpu.py --compare g4.json g6.json

Deformable attention in the pod
-------------------------------
The stable-gpu image does not ship MOTIP's compiled
``MultiScaleDeformableAttention``, so ``allow_pytorch_msda`` (default true) lets
the harness fall back to the reference implementation. Measured on the devbox T4
that costs 2.6-2.75x (87.7 -> 241.2 ms/frame at 800x1440). It is the same fallback
on both pools, so compare fallback-to-fallback and never against a compiled-op
number -- the harness records which backend ran and ``--compare`` warns if they
are mixed.
"""

from __future__ import annotations

import os
from enum import Enum

# Same values as aml-ice-hockey's MetaflowClusterSettings defaults. Duplicated
# rather than imported so MOTIP stays standalone; if the cluster moves, these are
# the five lines to change.
_CLUSTER_ENV = {
    "METAFLOW_DEFAULT_DATASTORE": "s3",
    "METAFLOW_DATASTORE_SYSROOT_S3": "s3://hudl-experiments-v1/faceoff/metaflow-argo/",
    "METAFLOW_KUBERNETES_NAMESPACE": "workflows",
    "METAFLOW_SERVICE_URL": "https://metaflow-metadata.hudltools.com",
    "METAFLOW_DEFAULT_METADATA": "service",
    # Metaflow packages only .py by default, so the configs/ YAMLs would be
    # missing in the pod and the harness dies on yaml_to_dict. Kept narrow on
    # purpose: MOTIP carries checkpoints and datasets that must not be packaged.
    # .so is here so prebuilt_ops/ ships too: that gives the pod MOTIP's
    # *compiled* deformable-attention kernel, which the stable-gpu image lacks
    # and cannot build (no CUDA toolchain). Without it every cluster number
    # carries the ~2.7x reference-implementation penalty. Safe to include: the
    # MOTIP tree has no other .so, so this adds exactly one 11 MB file.
    "METAFLOW_PACKAGE_SUFFIXES": ".py,.yaml,.yml,.so",
}

_LOCAL = os.environ.get("BENCH_LOCAL", "").lower() in ("1", "true", "yes")


def apply_cluster_env() -> None:
    """Point Metaflow at the Hudl cluster, without clobbering an explicit override."""
    for key, value in _CLUSTER_ENV.items():
        os.environ.setdefault(key, value)


# Must run BEFORE `import metaflow`: Metaflow snapshots its configuration at
# import time, so setting these afterwards is silently ignored and @kubernetes
# fails with "requires --datastore=s3".
if not _LOCAL:
    apply_cluster_env()

from metaflow import FlowSpec, Parameter, step  # noqa: E402

_ECR = "690616407375.dkr.ecr.us-east-1.amazonaws.com/icehockey/ihc-pipeline"
_GPU_IMAGE = os.environ.get("METAFLOW_KUBERNETES_GPU_IMAGE", f"{_ECR}:stable-gpu")


class GpuTarget(str, Enum):
    """EKS GPU node-pool targets. Member name = env value, value = ``group`` label.

    Only the 4-vCPU pools are modelled: this benchmark is single-GPU and not
    CPU-bound, so the multi-GPU pools would queue longer for no extra signal.
    """

    g4 = "GPU-G4-4"
    g5 = "GPU-G5-4"
    g6 = "GPU-G6-4"


def resolve_gpu_node_selector() -> dict[str, str]:
    """Node selector from ``GPU_BENCH_TARGET`` (default g4/T4).

    Raises on an unknown target so a typo fails at launch rather than silently
    benchmarking whatever the scheduler picked, which would produce a
    plausible-looking table attributed to the wrong GPU.
    """
    target = os.environ.get("GPU_BENCH_TARGET", "g4")
    try:
        return {"group": GpuTarget[target].value}
    except KeyError:
        raise ValueError(
            f"GPU_BENCH_TARGET={target!r} is not a valid target; "
            f"choose one of {[t.name for t in GpuTarget]}"
        ) from None


_TARGET = os.environ.get("GPU_BENCH_TARGET", "g4")
_NODE_SELECTOR = resolve_gpu_node_selector()

# What nvidia-smi should report for each pool. Checked after the run, because a
# node_selector is not a guarantee: ihc-action-recognition's
# 2026-07-15-DEL-14760 note records that the autoscaler's node-template labels
# land only on nodes it provisions, so a pre-existing node can carry a `group`
# label that does not match its actual hardware. Without this check a mislabelled
# (or unpinned) pod yields a perfectly plausible table attributed to the wrong GPU.
_EXPECTED_GPU = {"g4": "T4", "g5": "A10", "g6": "L4"}


def gpu_step(**kwargs):
    """``@kubernetes`` for cluster runs, a no-op decorator when BENCH_LOCAL is set."""
    if _LOCAL:
        return lambda fn: fn
    from metaflow import kubernetes  # noqa: PLC0415

    return kubernetes(
        annotations={"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"},
        **kwargs,
    )


class SpeedBenchFlow(FlowSpec):
    """Single GPU step: run scripts/bench_gpu.py on a pinned node pool."""

    arms = Parameter(
        "arms",
        default="ddetr,idhead,data",
        help="Harness arms: ddetr,idhead,data,rfdetr. rfdetr needs rfdetr "
        "importable in the image, which stable-gpu does not provide.",
    )
    sizes = Parameter(
        "sizes",
        default="800x1440,608x1088,448x800",
        help="Comma list of HxW input sizes for the ddetr arm.",
    )
    batches = Parameter(
        "batches",
        default="1,2,4,8",
        help="Batch sizes for the ddetr arm. The DETR pass is per-frame and "
        "stateless so it batches; the ID head is sequential and cannot.",
    )
    dtypes = Parameter(
        "dtypes",
        default="fp16",
        help="Comma list of fp16,bf16,fp32. bf16 only has tensor-core support from "
        "sm_80 up, so it is worth adding on g5/g6 and expected to look bad on the "
        "T4 (sm_75), which has no bf16 path.",
    )
    motip_config = Parameter(
        "motip_config",
        default="configs/eval_stage2_hockey.yaml",
        help="MOTIP config, relative to the repo root.",
    )
    rfdetr_variants = Parameter(
        "rfdetr_variants",
        default="nano,medium",
        help="RF-DETR variants for the rfdetr arm.",
    )
    rfdetr_compile = Parameter(
        "rfdetr_compile",
        default=False,
        help="Also measure RF-DETR under torch.compile, interleaved with eager.",
    )
    iters = Parameter("iters", default=20, help="Timed iterations per cell.")
    warmup = Parameter("warmup", default=10, help="Warmup iterations per cell.")
    allow_pytorch_msda = Parameter(
        "allow_pytorch_msda",
        default=True,
        help="Fall back to reference deformable attention when the compiled op is "
        "absent. Required for stable-gpu; see the module docstring.",
    )

    @step
    def start(self):
        """Record which pool this run is pinned to (CPU)."""
        self.target = _TARGET
        self.node_pool = _NODE_SELECTOR["group"]
        print(f"target={self.target} node_pool={self.node_pool} local={_LOCAL}")
        self.next(self.benchmark)

    # No memory request on purpose. These pools are all xlarge (4 vCPU / 16 GiB),
    # and gpu=1 + node_selector already guarantees exclusive placement, so a
    # memory request buys nothing and can make the pod unschedulable: asking for
    # 16384 MB on a 16 GiB node leaves nothing for system overhead, and the
    # autoscaler answers "NotTriggerScaleUp: 1 Insufficient memory" rather than
    # provisioning. Same lesson as ihc-jersey-numbers' 2026-08-06 note, which
    # removed @resources(memory=...) from its GPU-G4-4 step for this reason.
    @gpu_step(
        image=_GPU_IMAGE,
        gpu=1,
        gpu_vendor="nvidia",
        node_selector=_NODE_SELECTOR,
    )
    @step
    def benchmark(self):
        """Run the harness on the pinned GPU; keep the JSON as an artifact."""
        import json  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        # Metaflow unpacks the code package into the working directory, so the
        # repo root is cwd and the harness sits beside the rest of MOTIP.
        root = Path.cwd()
        script = root / "scripts" / "bench_gpu.py"
        if not script.exists():
            raise FileNotFoundError(
                f"{script} missing from the code package. Metaflow packages the "
                "flow file's directory, so this flow must stay at the repo root."
            )
        # Checked separately from the script: a .py always ships, a .yaml only
        # ships if METAFLOW_PACKAGE_SUFFIXES includes it. Failing here names the
        # cause instead of surfacing as a bare FileNotFoundError from yaml_to_dict.
        if not (root / self.motip_config).exists():
            raise FileNotFoundError(
                f"{self.motip_config} is not in the code package. Metaflow "
                "packages only .py by default; METAFLOW_PACKAGE_SUFFIXES must "
                "include .yaml (set in _CLUSTER_ENV)."
            )

        out_json = Path(tempfile.gettempdir()) / f"bench_{self.target}.json"
        cmd = [
            sys.executable, str(script),
            "--arms", self.arms,
            "--sizes", self.sizes,
            "--batches", self.batches,
            "--dtypes", self.dtypes,
            "--rfdetr-variants", self.rfdetr_variants,
            "--config", self.motip_config,
            "--iters", str(self.iters),
            "--warmup", str(self.warmup),
            "--out", str(out_json),
        ]
        if self.allow_pytorch_msda:
            cmd.append("--allow-pytorch-msda")
        if self.rfdetr_compile:
            cmd.append("--rfdetr-compile")

        env = dict(os.environ)
        env["MOTIP_ROOT"] = str(root)
        env["BENCH_NODE_POOL"] = self.node_pool
        env["BENCH_TMP"] = tempfile.gettempdir()

        print("running:", " ".join(cmd))
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            raise RuntimeError(f"bench_gpu.py exited {proc.returncode}")

        self.table = proc.stdout
        self.results_json = out_json.read_text()
        self.results = json.loads(self.results_json)
        self.gpu = self.results["env"]["gpu"]
        self.msda = self.results["env"]["msda"]

        expected = _EXPECTED_GPU.get(self.target)
        if expected and expected.lower() not in self.gpu.lower():
            raise RuntimeError(
                f"pinned to {self.node_pool} for target {self.target!r}, which "
                f"should be a {expected}, but nvidia-smi reports {self.gpu!r}. "
                "Either the node_selector was not applied or the node's `group` "
                "label does not match its hardware. Refusing to report these "
                "numbers under the wrong GPU."
            )
        self.next(self.end)

    @step
    def end(self):
        """Summarise and say how to pull the JSON back out."""
        from metaflow import current  # noqa: PLC0415

        run_id = current.run_id
        exports = " ".join(
            f"{k}={v}" for k, v in _CLUSTER_ENV.items()
            if k != "METAFLOW_PACKAGE_SUFFIXES"
        )
        print(f"benchmarked {self.gpu} on {self.node_pool} (msda={self.msda})")
        print(
            "fetch the JSON with:\n"
            # The exports are not optional: a bare `python -c` has none of the
            # cluster config this module applies at import, and the client then
            # fails with a misleading MetaflowNotFound("Run(...) does not exist").
            f"  export {exports}\n"
            "  python -c \"from metaflow import Run; "
            f"open('{self.target}.json','w')"
            f".write(Run('SpeedBenchFlow/{run_id}').data.results_json)\"\n"
            "then: python scripts/bench_gpu.py --compare g4.json g5.json"
        )


if __name__ == "__main__":
    SpeedBenchFlow()
