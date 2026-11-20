"""DETR_CACHE_DIR: cache format / lookup, and the cache-mode dataset (placeholder images, normalised boxes)."""
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.detr_cache import DetrCache, SeqCacheWriter, flip_boxes, seq_cache_path  # noqa: E402


def _write(tmp_path, seq="s1", n=3):
    w = SeqCacheWriter(n)
    rng = np.random.default_rng(0)
    for t in range(n):
        ids = np.array([5, 2, 9][: 2 + (t % 2)])
        w.add_frame(ids, rng.normal(size=(len(ids), 8)), rng.uniform(size=(len(ids), 4)), rng.normal(size=(len(ids), 3)))
    w.save(seq_cache_path(tmp_path, "Hockey", "train", seq), {"view": "test"})
    return w


def _batch(seq="s1", ids=((5, 2), (2, 5, 9), (2, 5)), flip=False):
    metas = [[{"dataset": "Hockey", "split": "train", "sequence": seq, "frame_idx": t, "detr_flip": flip} for t in range(3)]]
    anns = [[{"id": torch.tensor(i)} for i in ids]]
    return metas, anns


def test_rows_follow_the_annotation_id_order(tmp_path):
    _write(tmp_path)
    c = DetrCache(tmp_path)
    out, idx = c.detr_outputs(*_batch(), device="cpu")
    z = c.seq("Hockey", "train", "s1")
    # frame 1 asks for ids (2, 5, 9); cached order is (5, 2, 9)
    assert torch.allclose(out["outputs"][1, 0], torch.from_numpy(z.frame(1)[1][1]))
    assert torch.allclose(out["outputs"][1, 1], torch.from_numpy(z.frame(1)[1][0]))
    assert out["outputs"].shape == (3, 3, 8) and out["pred_logits"].shape == (3, 3, 3) and out["pred_boxes"].shape == (3, 3, 4)
    for (a, b), n in zip(idx, (2, 3, 2)):
        assert a.tolist() == b.tolist() == list(range(n))


def test_flip_negates_cx_only_and_matches_the_matcher_symmetry(tmp_path):
    _write(tmp_path)
    c = DetrCache(tmp_path)
    plain, _ = c.detr_outputs(*_batch(), device="cpu")
    flipped, _ = c.detr_outputs(*_batch(flip=True), device="cpu")
    for t, n in enumerate((2, 3, 2)):  # padding rows are never indexed
        assert torch.allclose(flipped["pred_boxes"][t, :n, 0], 1 - plain["pred_boxes"][t, :n, 0])
    assert torch.equal(flipped["pred_boxes"][..., 1:], plain["pred_boxes"][..., 1:])
    assert torch.equal(flipped["outputs"], plain["outputs"])
    assert torch.allclose(flip_boxes(flip_boxes(plain["pred_boxes"])), plain["pred_boxes"])


def test_missing_object_and_sequence_raise_clearly(tmp_path):
    _write(tmp_path)
    c = DetrCache(tmp_path)
    with pytest.raises(KeyError, match="GT object 7"):
        c.detr_outputs(*_batch(ids=((5, 7), (2, 5, 9), (2, 5))), device="cpu")
    with pytest.raises(FileNotFoundError):
        c.detr_outputs(*_batch(seq="nope"), device="cpu")


def test_writer_refuses_an_incomplete_sequence(tmp_path):
    w = SeqCacheWriter(3)
    w.add_frame(np.array([1]), np.zeros((1, 8)), np.zeros((1, 4)), np.zeros((1, 3)))
    with pytest.raises(AssertionError):
        w.save(tmp_path / "x.npz", {})


def test_empty_frames_round_trip(tmp_path):
    w = SeqCacheWriter(2)
    w.add_frame(np.zeros(0, np.int64), np.zeros((0, 8)), np.zeros((0, 4)), np.zeros((0, 3)))
    w.add_frame(np.array([4]), np.ones((1, 8)), np.full((1, 4), 0.5), np.zeros((1, 3)))
    w.save(seq_cache_path(tmp_path, "Hockey", "train", "e"), {})
    c = DetrCache(tmp_path)
    metas = [[{"dataset": "Hockey", "split": "train", "sequence": "e", "frame_idx": t} for t in range(2)]]
    anns = [[{"id": torch.zeros(0, dtype=torch.int64)}, {"id": torch.tensor([4])}]]
    out, idx = c.detr_outputs(metas, anns, "cpu")
    assert [len(a) for a, _ in idx] == [0, 1] and torch.equal(out["outputs"][1, 0], torch.ones(8))


def _make_dataset(root, w=640, h=360, n=6):
    seq = root / "Hockey" / "train" / "seqA"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    for i in range(1, n + 1):
        Image.new("RGB", (w, h)).save(seq / "img1" / f"{i:08d}.jpg")
    (seq / "gt" / "gt.txt").write_text("".join(f"{f},{o},{64 * o},{36 * o},{32},{72},1,1,1\n" for f in range(1, n + 1) for o in (1, 2)))
    (seq / "seqinfo.ini").write_text(f"[Sequence]\nname=seqA\nimDir=img1\nframeRate=30\nseqLength={n}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n")


def _config(cache_dir, flip):
    return {"DETR_CACHE_DIR": cache_dir, "DETR_CACHE_FLIP": flip, "NUM_ID_VOCABULARY": 50, "AUG_NUM_GROUPS": 2,
            "AUG_TRAJECTORY_OCCLUSION_PROB": 0.0, "AUG_TRAJECTORY_SWITCH_PROB": 0.0, "DATA_ROOT": None, "DATASETS": ["Hockey"],
            "DATASET_SPLITS": ["train"]}


@pytest.mark.parametrize("flip", [False, True])
def test_cache_mode_dataset_gives_normalised_boxes_without_decoding_images(tmp_path, flip):
    from data import build_dataset

    _make_dataset(tmp_path)
    cfg = _config("/unused", flip)
    cfg["DATA_ROOT"] = str(tmp_path)
    ds = build_dataset(cfg)
    for t in ds.transforms.transforms:
        if hasattr(t, "p"):
            t.p = 1.0 if flip else 0.0
    # delete the images: cache mode must not need them
    for p in (tmp_path / "Hockey/train/seqA/img1").iterdir():
        p.unlink()
    images, anns, metas = ds[{"dataset": "Hockey", "split": "train", "sequence": "seqA", "frame_idxs": [0, 1, 2]}]
    assert images.shape[-2:] == (22, 40)  # 360/16, 640/16 placeholders
    for a, m in zip(anns, metas):
        assert m.get("detr_flip", False) is flip
        ids = a["id"].tolist()
        for obj, box in zip(ids, a["bbox"]):
            cx, cy = (64 * obj + 16) / 640, (36 * obj + 36) / 360
            assert box[0].item() == pytest.approx(1 - cx if flip else cx, abs=1e-5)
            assert box[1].item() == pytest.approx(cy, abs=1e-5)
            assert box[2].item() == pytest.approx(32 / 640, abs=1e-5) and box[3].item() == pytest.approx(72 / 360, abs=1e-5)


def test_default_transforms_do_not_record_a_flip_meta(tmp_path):
    from data import build_dataset

    _make_dataset(tmp_path)
    cfg = _config(None, False)
    cfg.update({"DATA_ROOT": str(tmp_path), "AUG_RESIZE_SCALES": [160], "AUG_MAX_SIZE": 400, "AUG_RANDOM_RESIZE": [100],
                "AUG_RANDOM_CROP_MIN": 50, "AUG_RANDOM_CROP_MAX": 60, "AUG_MAX_SHIFT_RATIO": 0.0, "AUG_OVERFLOW_BBOX": False,
                "AUG_BRIGHTNESS": 0.0, "AUG_CONTRAST": 0.0, "AUG_SATURATION": 0.0, "AUG_HUE": 0.0, "AUG_COLOR_JITTER_V2": False})
    ds = build_dataset(cfg)
    assert not ds.detr_cache
    for _ in range(4):
        _, _, metas = ds[{"dataset": "Hockey", "split": "train", "sequence": "seqA", "frame_idxs": [0, 1]}]
        assert all("detr_flip" not in m for m in metas)
