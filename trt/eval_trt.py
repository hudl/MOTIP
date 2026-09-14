"""
eval_trt.py -- evaluate PT fp32, TRT fp16, TRT fp32 against MOT GT.

Usage (from MOTIP root):
  MOTIP_ROOT=$(pwd) <venv>/python trt/eval_trt.py \
      /path/to/seq/img1  /path/to/seq/gt/gt.txt \
      trt/engines/rfdetr_small_576_motip_ckpt_sm75_fp16.engine \
      trt/engines/rfdetr_small_576_motip_ckpt_sm75_fp32.engine \
      --checkpoint /tmp/trt_ckpt_cab8f975e423.pth
"""
from __future__ import annotations
import argparse, os, sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

MOTIP_ROOT = Path(os.environ.get("MOTIP_ROOT", Path(__file__).resolve().parent.parent))
TRACKEVAL_ROOT = MOTIP_ROOT / "TrackEval"
for p in (str(MOTIP_ROOT/"models"/"ops"), str(MOTIP_ROOT), str(MOTIP_ROOT/"trt"), str(TRACKEVAL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_real_interp = torch.nn.functional.interpolate
def _no_aa(*a, **kw):
    kw.pop("antialias", None)
    return _real_interp(*a, **kw)
torch.nn.functional.interpolate = _no_aa


def _load_frame(path: Path, res: int) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
    return TF.to_tensor(img).unsqueeze(0).cuda()

def _NestedTensor(t):
    from utils.nested_tensor import NestedTensor
    return NestedTensor(t, torch.zeros(t.shape[0],t.shape[2],t.shape[3],
                                       dtype=torch.bool, device=t.device))

def _motip_cfg(num_classes=3):
    from configs.util import load_super_config
    from utils.misc import yaml_to_dict
    cfg = yaml_to_dict(str(MOTIP_ROOT/"configs"/"rfdetr_motip_hockey_smoketest.yaml"))
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    cfg.update({"DEVICE":"cuda","INFERENCE_MODE":"evaluate","ONLY_DETR":False,
                "NUM_CLASSES":num_classes,"RFDETR_GROUP_DETR":1})
    return cfg

def _build_pt(ckpt, num_classes=3):
    from models.motip import build as build_motip
    cfg = _motip_cfg(num_classes)
    m, _ = build_motip(cfg)
    m.cuda().eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = ck.get("model", ck.get("state_dict", ck))
    msd = m.state_dict()
    filt = {k: v for k, v in raw.items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(filt, strict=False)
    return m, cfg

def _build_trt(engine_path, ckpt, num_classes=3):
    from models.motip import build as build_motip
    cfg = _motip_cfg(num_classes)
    m, _ = build_motip(cfg)
    m.cuda().eval()
    # load only trajectory/id_decoder weights
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = ck.get("model", ck.get("state_dict", ck))
    msd = m.state_dict()
    filt = {k:v for k,v in raw.items()
            if not k.startswith("detr.base.") and k in msd and msd[k].shape==v.shape}
    m.load_state_dict(filt, strict=False)
    # replace detr with TRT wrapper
    from compare_pt_trt import TRTDetectorWrapper, _load_engine
    engine = _load_engine(engine_path)
    object.__setattr__(m, "detr", TRTDetectorWrapper(engine, res=576))
    return m, cfg

def _run_tracker(model, cfg, frame_paths, res, det_thresh, newborn_thresh):
    from models.runtime_tracker import RuntimeTracker
    tracker = RuntimeTracker(
        model=model, sequence_hw=(res, res),
        det_thresh=det_thresh, newborn_thresh=newborn_thresh,
        id_thresh=cfg.get("ID_THRESH", 0.1),
        miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
        max_tracks=cfg.get("MAX_TRACKS", 0),
        area_thresh=cfg.get("AREA_THRESH", 100),
    )
    results = []
    for fi, fp in enumerate(frame_paths):
        t = _load_frame(fp, res)
        nt = _NestedTensor(t)
        with torch.no_grad():
            tracker.update(nt)
        tr = tracker.get_track_results()
        if tr and len(tr.get("id", [])):
            ids   = tr["id"].cpu().numpy()
            boxes = tr["bbox"].cpu().numpy()  # x1,y1,w,h in 'res' pixels
            scores = tr["score"].cpu().numpy()
            for tid, box, sc in zip(ids, boxes, scores):
                results.append((fi+1, int(tid), float(box[0]), float(box[1]),
                                 float(box[2]), float(box[3]), float(sc)))
    return results

def _save_mot(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.4f},-1,-1,-1\n")

def _run_trackeval(gt_txt, tracker_results, n_frames, res, orig_w=1280, orig_h=720):
    import trackeval
    tmp = Path(tempfile.mkdtemp())
    seq = "005254c3"; bench = "HOCKEY"; split = "val"; bsplit = f"{bench}-{split}"
    gt_base = tmp / "gt" / "mot_challenge" / bsplit
    (gt_base/seq/"gt").mkdir(parents=True)
    # TrackEval looks one level up from GT_FOLDER for seqmaps by default
    seqmap_dir = tmp / "gt" / "seqmaps"
    seqmap_dir.mkdir(parents=True)
    seqmap_file = seqmap_dir / f"{bsplit}.txt"
    with open(seqmap_file, "w") as f:
        f.write("name\n"+seq+"\n")
    with open(gt_base/seq/"seqinfo.ini","w") as f:
        f.write(f"[Sequence]\nname={seq}\nseqLength={n_frames}\nimWidth={res}\nimHeight={res}\nimExt=.jpg\n")
    sx, sy = res/orig_w, res/orig_h
    with open(gt_base/seq/"gt"/"gt.txt","w") as f:
        for line in open(gt_txt):
            p = line.strip().split(",")
            if int(p[0]) > n_frames: continue
            x,y,w,h = float(p[2])*sx, float(p[3])*sy, float(p[4])*sx, float(p[5])*sy
            f.write(f"{p[0]},{p[1]},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
    # TrackEval expects: TRACKERS_FOLDER/{bsplit}/{tracker_name}/data/{seq}.txt
    trackers_root = tmp/"trackers"/"mot_challenge"
    for name, rows in tracker_results.items():
        d = trackers_root/bsplit/name/"data"; d.mkdir(parents=True)
        _save_mot(rows, d/f"{seq}.txt")
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"PRINT_RESULTS":True,"PRINT_ONLY_COMBINED":True,
                        "TIME_PROGRESS":False,"OUTPUT_SUMMARY":False,
                        "OUTPUT_DETAILED":False,"PLOT_CURVES":False,
                        "DISPLAY_LESS_PROGRESS":True})
    ds_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_config.update({"GT_FOLDER":str(tmp/"gt"/"mot_challenge"),
                      "TRACKERS_FOLDER":str(trackers_root),
                      "BENCHMARK":bench,"SPLIT_TO_EVAL":split,
                      "SEQMAP_FILE":str(seqmap_file),
                      "DO_PREPROC":False,"CLASSES_TO_EVAL":["pedestrian"]})
    metrics_config = {"METRICS":["HOTA","CLEAR","Identity"],"THRESHOLD":0.5}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(ds_config)]
    metrics_list = [getattr(trackeval.metrics, m)({"THRESHOLD":0.5})
                    for m in metrics_config["METRICS"]]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("gt_txt")
    ap.add_argument("fp16_engine")
    ap.add_argument("fp32_engine")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--res", type=int, default=576)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--det_thresh", type=float, default=0.5)
    ap.add_argument("--newborn_thresh", type=float, default=0.6)
    args = ap.parse_args()

    frame_paths = sorted(Path(args.frame_dir).glob("*.jpg"))[:args.n_frames]
    n = len(frame_paths)
    print(f"Using {n} frames")

    import json

    all_results = {}

    print("\n--- PT fp32 ---")
    pt, cfg = _build_pt(args.checkpoint, args.num_classes)
    import time; _t0 = time.perf_counter()
    pt_r = _run_tracker(pt, cfg, frame_paths, args.res, args.det_thresh, args.newborn_thresh)
    all_results["PT_fp32"] = pt_r
    print(f"  {len(set(r[1] for r in pt_r))} unique IDs, {len(pt_r)} dets  |  {n/(time.perf_counter()-_t0):.1f} fps")
    del pt; torch.cuda.empty_cache()

    print("\n--- TRT fp16 ---")
    trt16, cfg16 = _build_trt(args.fp16_engine, args.checkpoint, args.num_classes)
    _t0 = time.perf_counter()
    trt16_r = _run_tracker(trt16, cfg16, frame_paths, args.res, args.det_thresh, args.newborn_thresh)
    all_results["TRT_fp16"] = trt16_r
    print(f"  {len(set(r[1] for r in trt16_r))} unique IDs, {len(trt16_r)} dets  |  {n/(time.perf_counter()-_t0):.1f} fps")
    del trt16; torch.cuda.empty_cache()

    print("\n--- TRT fp32 ---")
    trt32, cfg32 = _build_trt(args.fp32_engine, args.checkpoint, args.num_classes)
    _t0 = time.perf_counter()
    trt32_r = _run_tracker(trt32, cfg32, frame_paths, args.res, args.det_thresh, args.newborn_thresh)
    all_results["TRT_fp32"] = trt32_r
    print(f"  {len(set(r[1] for r in trt32_r))} unique IDs, {len(trt32_r)} dets  |  {n/(time.perf_counter()-_t0):.1f} fps")
    del trt32; torch.cuda.empty_cache()

    # Save per-tracker results for visualization
    out_json = "/tmp/eval_trt_results.json"
    with open(out_json, "w") as f:
        json.dump({k: [list(r) for r in v] for k, v in all_results.items()}, f)
    print(f"\nSaved results to {out_json}")

    print("\n--- TrackEval ---")
    res = _run_trackeval(args.gt_txt, all_results, n, args.res)

    print("\n")
    print(f"{'Tracker':<12}  {'HOTA':>7}  {'DetA':>7}  {'AssA':>7}  {'MOTA':>7}  {'IDF1':>7}  {'IDs':>5}")
    print("-" * 65)
    for name in ["PT_fp32", "TRT_fp16", "TRT_fp32"]:
        try:
            r = res["MotChallenge2DBox"][name]["COMBINED_SEQ"]["pedestrian"]
            hota = r["HOTA"]["HOTA"].mean() * 100
            deta = r["HOTA"]["DetA"].mean() * 100
            assa = r["HOTA"]["AssA"].mean() * 100
            mota = r["CLEAR"]["MOTA"] * 100
            idf1 = r["Identity"]["IDF1"] * 100
            uid  = len(set(x[1] for x in all_results[name]))
            print(f"{name:<12}  {hota:>7.1f}  {deta:>7.1f}  {assa:>7.1f}  {mota:>7.1f}  {idf1:>7.1f}  {uid:>5}")
        except Exception as e:
            print(f"{name:<12}  eval error: {e}")

if __name__ == "__main__":
    main()
