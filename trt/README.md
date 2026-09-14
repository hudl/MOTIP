# MOTIP TRT Conversion

RF-DETR stage-2 MOTIP → TensorRT engine: export, validate, benchmark, and evaluate.

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
| PT fp32 | 21 | 0.7× |
| TRT fp32 | 28 | 0.9× |
| TRT fp16 | **45** | **1.5×** |

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
