"""The last-layer match guard: RF-DETR's matcher crashes on one degenerate box; the guard must not."""
import pytest
import torch

from models.motip import MATCH_MIN_WH, guarded_match, l1_fallback_match, sanitize_match_inputs

matcher_mod = pytest.importorskip("rfdetr.models.matcher")


def _matcher():
    return matcher_mod.HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)


def _inputs(n_q=6, degenerate=True):
    g = torch.Generator().manual_seed(0)
    single = {
        "pred_logits": torch.randn(1, n_q, 2, generator=g),
        "pred_boxes": torch.rand(1, n_q, 4, generator=g) * 0.3 + 0.2,
    }
    boxes = torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.6, 0.5, 0.08, 0.15], [0.45, 0.45, 0.1, 0.2]])
    if degenerate:
        boxes[2] = torch.tensor([0.45, 0.45, 0.0, 0.0])   # zero-area GT box
        single["pred_boxes"][0, 3] = torch.tensor([0.45, 0.45, 0.0, 0.0])  # zero-area prediction on top of it
    return single, [{"labels": torch.tensor([1, 1, 1]), "boxes": boxes}]


def test_rfdetr_matcher_crashes_on_degenerate_box():
    single, targets = _inputs()
    with pytest.raises(ValueError):
        _matcher()(single, targets)


def test_guard_matches_degenerate_box():
    single, targets = _inputs()
    out = guarded_match(_matcher(), single, targets)
    assert len(out) == 1
    pi, ti = out[0]
    assert sorted(ti.tolist()) == [0, 1, 2]
    assert len(set(pi.tolist())) == 3


def test_guard_is_identity_on_normal_inputs():
    single, targets = _inputs(degenerate=False)
    ref = _matcher()(single, targets)
    got = guarded_match(_matcher(), single, targets)
    for (a, b), (c, d) in zip(ref, got):
        assert torch.equal(a, c) and torch.equal(b, d)


def test_sanitize_only_touches_bad_values():
    single, targets = _inputs(degenerate=False)
    s2, t2, changed = sanitize_match_inputs(single, targets)
    assert not changed
    assert all(torch.equal(single[k], s2[k]) for k in single)
    assert torch.equal(t2[0]["boxes"], targets[0]["boxes"])
    single["pred_logits"][0, 0, 0] = float("nan")
    s3, _, changed = sanitize_match_inputs(single, targets)
    assert changed and torch.isfinite(s3["pred_logits"]).all()


def test_sanitize_clamps_wh():
    single, targets = _inputs()
    s2, t2, changed = sanitize_match_inputs(single, targets)
    assert changed
    assert (t2[0]["boxes"][:, 2:] >= MATCH_MIN_WH).all()
    assert (s2["pred_boxes"][..., 2:] >= MATCH_MIN_WH).all()
    assert targets[0]["boxes"][2, 2] == 0.0  # caller's targets untouched


def test_fallback_when_matcher_still_raises():
    def bad(single, targets):
        raise ValueError("matrix contains invalid numeric entries")
    single, targets = _inputs()
    out = guarded_match(bad, single, targets)
    assert sorted(out[0][1].tolist()) == [0, 1, 2]


def test_l1_fallback_empty_target():
    single, _ = _inputs()
    out = l1_fallback_match(single, [{"labels": torch.tensor([]), "boxes": torch.zeros(0, 4)}])
    assert out[0][0].numel() == 0 and out[0][1].numel() == 0
