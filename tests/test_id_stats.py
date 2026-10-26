"""Tests for RuntimeTracker._id_stats: the per-detection ID-decoder confidence.

A tracker is built without a model (only the state _id_stats reads), with an ID vocabulary
of 5 (index 5 is the newborn token).
"""
import math

import torch

from models.runtime_tracker import RuntimeTracker

VOCAB = 5


def _tracker(live_labels: list[int], scores: list[list[float]] | None) -> RuntimeTracker:
    t = object.__new__(RuntimeTracker)
    t.num_id_vocabulary = VOCAB
    # (T, N): every frame of a trajectory carries its label; T = 0 means no trajectories yet
    t.trajectory_id_labels = torch.tensor([live_labels, live_labels]) if live_labels else torch.zeros((0, 0), dtype=torch.int64)
    t._frame_id_scores = torch.tensor(scores) if scores is not None else None
    return t


def test_label_score_and_margin_over_live_ids_only() -> None:
    # live IDs 1 and 3; ID 4 has no trajectory but the decoder still gives it 0.30
    scores = [[0.0, 0.50, 0.0, 0.15, 0.30, 0.05]]
    out = _tracker([1, 3], scores)._id_stats(torch.tensor([1]))
    assert math.isclose(out[0, 0].item(), 0.50, abs_tol=1e-6)  # the label it was given
    assert math.isclose(out[0, 1].item(), 0.50, abs_tol=1e-6)  # best live ID
    assert math.isclose(out[0, 2].item(), 0.35, abs_tol=1e-6)  # 0.50 - 0.15, not 0.50 - 0.30


def test_newborn_label_reports_the_newborn_probability() -> None:
    scores = [[0.0, 0.20, 0.0, 0.10, 0.0, 0.70]]
    out = _tracker([1, 3], scores)._id_stats(torch.tensor([VOCAB]))
    assert math.isclose(out[0, 0].item(), 0.70, abs_tol=1e-6)
    assert math.isclose(out[0, 1].item(), 0.20, abs_tol=1e-6)
    assert math.isclose(out[0, 2].item(), 0.10, abs_tol=1e-6)


def test_one_live_trajectory_margin_is_its_probability() -> None:
    out = _tracker([2], [[0.0, 0.0, 0.8, 0.0, 0.1, 0.1]])._id_stats(torch.tensor([2]))
    assert math.isclose(out[0, 2].item(), 0.8, abs_tol=1e-6)


def test_no_trajectories_is_all_nan() -> None:
    out = _tracker([], None)._id_stats(torch.tensor([VOCAB, VOCAB]))
    assert out.shape == (2, 3)
    assert torch.isnan(out).all()
