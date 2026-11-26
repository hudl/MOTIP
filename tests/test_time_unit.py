"""TIME_UNIT: GenerateIDLabels time indexes in steps (MOTIP default) or in real frame offsets."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.transforms import GenerateIDLabels  # noqa: E402


def _call(frame_idxs, unit, ids=((3, 7), (7, 3), (3,), (7, 3))):
    anns = [{"id": torch.tensor(i, dtype=torch.int64)} for i in ids[: len(frame_idxs)]]
    metas = [{"frame_idx": f} for f in frame_idxs]
    t = GenerateIDLabels(num_id_vocabulary=50, aug_num_groups=2, num_training_ids=50, time_unit=unit)
    _, anns, _ = t([None] * len(frame_idxs), anns, metas)
    return torch.cat([a["times"] for a in anns], dim=1)   # (G, T, N)


def test_step_mode_is_the_original_arange():
    times = _call([10, 14, 18, 22], "step")
    assert times.shape == (2, 4, 2)
    assert torch.equal(times[0, :, 0], torch.tensor([0, 1, 2, 3])) and torch.equal(times[1], times[0])


def test_frame_mode_matches_step_mode_at_stride_1():
    assert torch.equal(_call([100, 101, 102, 103], "frame"), _call([100, 101, 102, 103], "step"))


def test_frame_mode_uses_real_offsets_at_stride_4():
    times = _call([12, 16, 20, 24], "frame")
    assert torch.equal(times[0, :, 1], torch.tensor([0, 4, 8, 12]))


def test_frame_mode_irregular_steps():
    assert torch.equal(_call([5, 6, 8, 12], "frame")[1, :, 0], torch.tensor([0, 1, 3, 7]))


def test_frame_mode_rejects_non_increasing_frames():
    with pytest.raises(AssertionError):
        _call([5, 5, 6, 7], "frame")


def test_unknown_unit_rejected():
    with pytest.raises(AssertionError):
        GenerateIDLabels(num_id_vocabulary=50, aug_num_groups=1, num_training_ids=50, time_unit="seconds")


def test_builders_pass_the_config_key():
    src = open(os.path.join(os.path.dirname(__file__), "..", "data", "transforms.py")).read()
    assert src.count('time_unit=config.get("TIME_UNIT", "step")') == 2
