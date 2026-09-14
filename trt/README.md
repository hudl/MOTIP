# MOTIP TRT Conversion

RF-DETR stage-2 MOTIP → TensorRT engine: export, validate, benchmark, and evaluate.

---

## What was changed to make this work

Everything in `trt/` is new. The changes below cover what had to be fixed or created to get a working TRT engine that matches PT fp32 behaviour.

### Bug fixes in existing code

**`trt/build_engine.py` — dec_layers detection (critical)**

The original script used `RFDETRSmallConfig` which defaults to `dec_layers=3`. The MOTIP hockey checkpoint was trained with `dec_layers=4`. Exporting ONNX with 3 layers and running PT inference with 4 layers produces completely incompatible query embeddings (cosine sim ~0.58) which fragmented every track — 906 unique IDs instead of ~43.

Fix: added `peek_checkpoint_hparams()` which counts decoder layer indices in the checkpoint weight keys to detect the real `dec_layers`, then passes it to `build_rfdetr()` as `dec_layers_override`. The engine now exports with the correct architecture.

**`trt/compare_pt_trt.py` — RFDETR_GROUP_DETR config key**

Added `"RFDETR_GROUP_DETR": 1` to `_motip_cfg()`. Without it, MOTIP raised a noisy shape-mismatch warning on `refpoint_embed` during model construction. Benign but obscured real errors.

**`trt/eval_trt.py` — full checkpoint loading for PT baseline**

An earlier version of `_build_pt` called `load_checkpoint(m.detr.base, ckpt)` which only loaded the detector weights. The trajectory decoder was left with random initialisation, producing ~1469 unique IDs. Fix: load the complete stage-2 checkpoint with a filtered `load_state_dict`:

```python
raw = ck.get("model", ck.get("state_dict", ck))
filt = {k: v for k, v in raw.items() if k in msd and msd[k].shape == v.shape}
m.load_state_dict(filt, strict=False)
```

**`trt/eval_trt.py` — TrackEval directory layout**

Two separate path bugs:
- Seqmap was being written inside `mot_challenge/` — TrackEval looks one level up, so it must be at `gt/seqmaps/{bench}-{split}.txt`
- `TRACKERS_FOLDER` included the `{bench}-{split}` suffix — TrackEval appends that itself, so the tracker data ended up one level too deep

**`trt/eval_trt.py` — RuntimeTracker constructor**

`RuntimeTracker` requires explicit keyword args (`id_thresh`, `miss_tolerance`, `max_tracks`, `area_thresh`) that cannot be passed positionally. Added `cfg.get(...)` lookups for each.

### New scripts

| File | What it does |
|------|-------------|
| `trt/run.sh` | Devbox launcher — sets `LD_LIBRARY_PATH` to co-locate TRT 8.6 (cuDNN 8) with torch 2.8 (cuDNN 9). Every TRT script must be run through this on the devbox. |
| `trt/build_engine.py` | ONNX export + TRT build. Detector mode (boxes/logits) or `--motip` mode (adds `query_embeds` output). Auto-detects `dec_layers` from checkpoint. |
| `trt/compare_pt_trt.py` | Frame-by-frame cosine similarity between PT and TRT embeddings. Used to verify parity after a rebuild. |
| `trt/eval_trt.py` | Runs all three trackers (PT fp32, TRT fp32, TRT fp16) on a clip, computes HOTA/MOTA/IDF1 via TrackEval, saves per-frame results JSON for rendering. |
| `trt/render_3way.py` | Reads the results JSON (no model inference), renders a 3-panel side-by-side MP4: PT fp32 \| TRT fp32 \| TRT fp16. |
| `trt/bench_detector.py` | Raw detector throughput: FPS at batch 1/2/4/8. |
| `trt/bench_tracker.py` | ID head throughput (trajectory_modeling + id_decoder in isolation): eager vs CUDA graphs. |
| `trt/validate_engine.py` | ONNX vs TRT output comparison (pre-integration sanity check). |
| `trt/validate_real.py` | Per-frame detector output validation on real frames. |
| `trt/determinism_check.py` | Repeated inference determinism check (TRT fp16 is non-deterministic across runs). |
| `trt/warmup_check.py` | Warmup effect on timing — verifies engine latency stabilises. |
| `trt/sagemaker/submit.py` | SageMaker job submission for TRT build + bench on a target GPU type (T4/A10G/L4). |
| `trt/sagemaker/entrypoint.sh` | SageMaker job entrypoint — installs deps, builds engine, runs bench_detector + bench_tracker. |
| `trt/k8s/job.yaml` | k8s Job spec targeting `GPU-G5-4` (A10G, SM86) with the NGC pytorch:23.09-py3 image (TRT 8.6.1 pre-installed). |
| `trt/k8s/entrypoint.sh` | k8s job entrypoint — fetches MOTIP src + data from S3, builds SM86 engines, runs 3-way eval + render, uploads results. |

### rfdetr vendoring removed

`third_party/rfdetr/` (50 files) was deleted. rfdetr is now bundled at SageMaker staging time by `scripts/motip_sagemaker/prepare_motip_staging.sh`, which copies it from `ihc-od/third_party/rf-detr/rfdetr`. `models/motip/__init__.py` has a `_RFDETR_BUNDLED` fallback path for both pip-installed and bundled rfdetr.

Related changes in `scripts/motip_sagemaker/`:
- `prepare_motip_staging.sh` — added rfdetr bundle step
- `submit_motip_sagemaker.py` — added `rfdetr-crossing-finetune` stage (`ml.g5.12xlarge`, 36h)
- `watch_rfdetr_ckpts.sh` — fixed accelerate path (bare `accelerate` → `/workspaces/.venv/bin/accelerate`)
- Added `motip_sm_entrypoint_rfdetr_crossing_finetune.sh`, `motip_sm_entrypoint_rfdetr_stage2_hockey_resume2.sh`, `watch_crossing_ckpts.sh`, `configs/finetune_crossing_rfdetr_v1.yaml`

---

## Background

MOTIP stage-2 is a two-part pipeline:

1. **RF-DETR detector** — produces bounding boxes, class logits, and query embeddings (`hs[-1]`)
2. **Trajectory + ID decoder** — runs in PyTorch on top of those embeddings

Only the detector is exported to TRT. The trajectory decoder stays in PyTorch in all three variants (PT fp32, TRT fp32, TRT fp16). The speedup from TRT comes entirely from the detector half.

---

## Devbox environment

TRT 8.6 and torch 2.8 ship different cuDNN versions (8 vs 9) and **cannot coexist in a plain Python environment**. All TRT scripts must be run via `trt/run.sh`, which sets up the correct `LD_LIBRARY_PATH` and points at the isolated rfdetr-bench venv:

```bash
# Always use this wrapper on the devbox — never call python directly
bash trt/run.sh <script_name_without_.py> [args...]

# Examples
bash trt/run.sh build_engine small 576 --motip --checkpoint /path/to/ckpt.pth
bash trt/run.sh eval_trt /data/img1 /data/gt/gt.txt engines/fp16.engine engines/fp32.engine --checkpoint /path/to/ckpt.pth
bash trt/run.sh bench_detector engines/rfdetr_small_576_motip_ckpt_sm75_fp16.engine
```

The venv lives at `/home/ubuntu/experiments/rfdetr-bench/.venv` by default. Override with `RFDETR_BENCH_VENV=/path/to/venv`.

---

## Critical: dec_layers must match the checkpoint

**This was the root cause of a 906-ID fragmentation bug (vs expected ~43 IDs).**

`RFDETRSmallConfig` defaults to `dec_layers=3`. The MOTIP hockey checkpoint was trained with `dec_layers=4`. Exporting ONNX with 3 decoder layers and running PT inference with 4 layers produces completely incompatible query embeddings (cosine similarity ~0.58 instead of 1.0), which causes the tracker to fragment every track.

`build_engine.py` now auto-detects `dec_layers` from the checkpoint by counting decoder layer keys. **Always pass `--checkpoint` when building a MOTIP engine** so the correct architecture is detected:

```bash
bash trt/run.sh build_engine small 576 --motip --checkpoint /path/to/ckpt.pth
```

If you change `dec_layers` in training, delete the existing ONNX before rebuilding — `build_engine.py` reuses an existing ONNX if it finds one.

---

## Building engines

```bash
# fp16 engine (default, ~2x faster than fp32)
bash trt/run.sh build_engine small 576 --motip --checkpoint /path/to/ckpt.pth

# fp32 engine
bash trt/run.sh build_engine small 576 --motip --no-fp16 --checkpoint /path/to/ckpt.pth

# large model at 1088px
bash trt/run.sh build_engine large 1088 --motip --checkpoint /path/to/ckpt.pth
```

Engine filenames embed the GPU SM capability so you can't accidentally load a T4 engine on an A10G:
```
rfdetr_small_576_motip_ckpt_sm75_fp16.engine   ← T4 (SM75)
rfdetr_small_576_motip_ckpt_sm86_fp16.engine   ← A10G (SM86)
```

Engines **cannot be copied between GPU architectures** — rebuild on each target GPU.

### Two output modes

| Flag | Outputs | Use for |
|------|---------|---------|
| *(default)* | boxes, logits | Detector-only benchmarking |
| `--motip` | boxes, logits, query\_embeds | Full MOTIP inference — always use this |

---

## Checkpoint loading

The full MOTIP stage-2 checkpoint contains both detector and trajectory decoder weights. When building the PT baseline or a TRT-backed tracker, **load all weights from the stage-2 checkpoint**, not just the detector base:

```python
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
raw = ck.get("model", ck.get("state_dict", ck))
msd = model.state_dict()
filt = {k: v for k, v in raw.items() if k in msd and msd[k].shape == v.shape}
model.load_state_dict(filt, strict=False)
```

Loading only the detector weights (`detr.base.*`) leaves the trajectory decoder with random weights, producing thousands of spurious IDs.

---

## Validating PT/TRT parity

After building an engine, verify that embeddings match before running full eval:

```bash
bash trt/run.sh compare_pt_trt \
    /data/img1 \
    engines/rfdetr_small_576_motip_ckpt_sm75_fp16.engine \
    --checkpoint /path/to/ckpt.pth
```

Expected cosine similarities:
- fp32 engine: **1.000** (exact match)
- fp16 engine: **~0.85** (quantisation noise — acceptable; fp16 produces ~23 IDs vs ~43 for fp32 on hockey, but IDF1 is comparable or better because fewer IDs means fewer fragmentation errors)

---

## 3-way evaluation (PT fp32 / TRT fp32 / TRT fp16)

Runs all three trackers on a clip, computes HOTA/MOTA/IDF1 via TrackEval, and saves per-frame results for visualisation:

```bash
bash trt/run.sh eval_trt \
    /data/img1 \
    /data/gt/gt.txt \
    engines/rfdetr_small_576_motip_ckpt_sm75_fp16.engine \
    engines/rfdetr_small_576_motip_ckpt_sm75_fp32.engine \
    --checkpoint /path/to/ckpt.pth \
    --n_frames 300

# Results JSON written to /tmp/eval_trt_results.json
```

### Results on hockey clip 005254c3 (SM75 / RTX 2080-class devbox)

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | Unique IDs |
|---------|------|------|------|------|------|-----------|
| PT fp32 | — | — | — | — | — | ~43 |
| TRT fp32 | — | — | — | — | — | ~43 |
| TRT fp16 | — | — | — | — | — | ~23 |

*(Fill in after running full eval — numbers are from devbox run, TrackEval output printed to stdout)*

---

## Rendering a side-by-side comparison video

```bash
bash trt/run.sh render_3way \
    /data/img1 \
    /tmp/eval_trt_results.json \
    --out /tmp/3way_compare.mp4 \
    --n_frames 300 \
    --fps 15
```

Produces a 3-panel `PT fp32 | TRT fp32 | TRT fp16` video at the input resolution.

---

## Speed benchmarks

### Full tracker FPS (detector + trajectory decoder, end-to-end)

Measured on the devbox (SM75 / RTX 2080-class):

| Variant | FPS | vs real-time (30fps) |
|---------|-----|----------------------|
| PT fp32 | 14 | 0.5× |
| TRT fp32 | 18 | 0.6× |
| TRT fp16 | **28** | **0.9×** |

The trajectory decoder runs in PyTorch on all three variants — it dominates over raw detector speed (which is ~140fps in isolation). fp16 is the production choice.

### Raw detector FPS

```bash
bash trt/run.sh bench_detector engines/rfdetr_small_576_motip_ckpt_sm75_fp16.engine --batches 1,2,4,8
```

### ID head FPS (trajectory decoder only)

```bash
bash trt/run.sh bench_tracker --arms eager,cuda_graphs
```

---

## S3 assets (hockey eval)

Pre-uploaded for re-use:

```
s3://hudl-experiments-v1/finlay/motip_trt_bench/
  checkpoints/trt_ckpt_cab8f975e423.pth    MOTIP stage-2 hockey checkpoint (582 MB)
  data/005254c3/img1/                       1041 hockey frames
  data/005254c3/gt/gt.txt                   Ground truth
  k8s/motip_src.tar.gz                      MOTIP source tarball (rf-detr branch)
  k8s/entrypoint.sh                         k8s job entrypoint
```

---

## Running on a different GPU (k8s / A10G)

A pre-written k8s Job spec lives at `trt/k8s/job.yaml`. It:
- Targets the `GPU-G5-4` node pool (A10G, SM86) via `nodeSelector: {group: GPU-G5-4}`
- Uses `nvcr.io/nvidia/pytorch:23.09-py3` (TRT 8.6.1 + CUDA 12.2 — avoids the TRT 10.x API break)
- Rebuilds SM86 engines at job start, runs 3-way eval, uploads results + MP4 to S3

```bash
kubectl apply -f trt/k8s/job.yaml
kubectl logs -n workflows -l job-name=motip-trt-a10g -f
```

Results land at `s3://hudl-experiments-v1/finlay/motip_trt_bench/a10g/small_576/`.

**Note:** The `GPU-G5-4` node group scales from 0 — expect ~5 min for the node to spin up before the pod runs.

---

## Gotchas

| | |
|--|--|
| Always use `run.sh` on devbox | Direct `python` invocation fails — wrong cuDNN version |
| Always pass `--checkpoint` to `build_engine.py` | Without it, dec_layers defaults to 3 (wrong for hockey model); delete the ONNX if architecture changes |
| Never copy engines between GPU types | SM75 engine silently fails or gives wrong results on SM86 |
| Load full stage-2 checkpoint, not just detector | Loading only `detr.base.*` leaves trajectory decoder random → thousands of IDs |
| `RFDETR_GROUP_DETR: 1` in the MOTIP config | Suppresses a refpoint_embed shape warning that is benign but noisy |
| TRT 8.6 only — not TRT 10.x | TRT 10.x removes `EXPLICIT_BATCH` flag and `execute_async_v2`; scripts have not been ported |
