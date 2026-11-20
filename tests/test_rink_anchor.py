"""Tests for the rink anchor: each ID-decoder token gets an encoding of its player's position in rink metres.

Off by default. Zero-init: a model with the anchor on is exactly the model without it at step 0. Boxes are normalised
cxcywh unless noted. The synthetic "homography" maps frame px to metres linearly (see ``H0``).
"""
import copy
import json
import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import data.rink as R
from data.transforms import (
    MultiBoxXYWHtoXYXY, MultiBoxXYXYtoCXCYWH, MultiCompose, MultiNormalizeBoundingBoxes, MultiRandomCrop,
    MultiRandomHorizontalFlip, MultiRandomResize, MultiRinkFinalize, MultiStack, MultiToTensor, build_transforms,
)
from models.motip.motip import MOTIP
from models.motip.rink_anchor import RinkAnchor, project_feet
from models.runtime_tracker import RuntimeTracker
from train import get_param_groups, prepare_for_motip, rink_batch_from_metas
from utils.misc import distributed_device

D = 16
K = 8
W0, H0_PX = 1280, 720
# frame px -> metres: x_m = 0.05 * px_x - 32, y_m = 0.05 * px_y - 18  (so the frame spans [-32, 32] x [-18, 18])
H0 = np.array([[0.05, 0.0, -32.0], [0.0, 0.05, -18.0], [0.0, 0.0, 1.0]])
BOUNDS = (31.0, 16.0)


# --------------------------------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------------------------------
class _StubIDDecoder(nn.Module):
    num_id_vocabulary = K
    feature_dim = D

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, seq_info, use_decoder_checkpoint=False):
        self.seen = seq_info
        return "logits"


class _StubDetr(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(1, 1)


def _motip(rink=True, **kw):
    return MOTIP(
        detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
        id_decoder=_StubIDDecoder(), detr_hidden_dim=D, rink_anchor=rink, rink_anchor_kwargs=kw or None,
    )


def _seq_info(with_xy=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {
        "trajectory_features": torch.randn(1, 1, 3, 4, D, generator=g),
        "unknown_features": torch.randn(1, 1, 3, 2, D, generator=g),
        "trajectory_boxes": torch.rand(1, 1, 3, 4, 4, generator=g),
        "unknown_boxes": torch.rand(1, 1, 3, 2, 4, generator=g),
    }
    if with_xy:
        out["trajectory_rink_xy"] = torch.rand(1, 1, 3, 4, 2, generator=g) * 40 - 20
        out["unknown_rink_xy"] = torch.rand(1, 1, 3, 2, 2, generator=g) * 40 - 20
    return out


def _poke(m):
    """Make the anchor non-trivial (as after training)."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for t in (m.rink_anchor.mlp[-1].weight, m.rink_anchor.mlp[-1].bias, m.rink_anchor.no_position):
            t.copy_(torch.randn(t.shape, generator=g))


# --------------------------------------------------------------------------------------------------
# defaults are unchanged
# --------------------------------------------------------------------------------------------------
def test_defaults_off():
    m = _motip(rink=False)
    assert m.rink_anchor is None
    assert not any(n.startswith("rink_anchor") for n, _ in m.named_parameters())
    # the id_decoder part passes seq_info untouched when the anchor is off
    seq_info = _seq_info()
    m(seq_info=seq_info, part="id_decoder")
    assert m.id_decoder.seen is seq_info


def test_default_transforms_have_no_rink_step_and_datasets_no_rink_metas():
    from data.joint_dataset import JointDataset
    cfg = _aug_cfg()
    assert not any(isinstance(t, MultiRinkFinalize) for t in build_transforms(cfg).transforms)
    assert any(isinstance(t, MultiRinkFinalize) for t in build_transforms({**cfg, "RINK_ANCHOR": True}).transforms)
    assert JointDataset(data_root=".", datasets=[], splits=[]).rink_anchor is False


def test_prepare_for_motip_without_rink_has_no_rink_keys():
    seq_info, _ = _prepare(rink=None)
    assert "trajectory_rink_xy" not in seq_info and "unknown_rink_xy" not in seq_info


def test_runtime_tracker_default_has_no_rink_inputs():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(100, 200))
    assert tr.rink_anchor is False
    _prime(tr)
    tr._get_id_pred_labels(boxes=torch.rand(2, 4, device=distributed_device()),
                           output_embeds=torch.zeros(2, 256, device=distributed_device()))
    assert "trajectory_rink_xy" not in model.seen and "unknown_rink_xy" not in model.seen


def test_param_groups_default_unchanged():
    model = _GroupModel(rink=False)
    groups = get_param_groups(model, _group_cfg())
    names = {id(p): n for n, p in model.named_parameters()}
    assert not any(names[id(g["params"])].startswith("rink_anchor") for g in groups)


# --------------------------------------------------------------------------------------------------
# the module
# --------------------------------------------------------------------------------------------------
def test_zero_init_is_identity_and_invalid_tokens_use_no_position():
    m = _motip()
    seq_info = _seq_info()
    seq_info["unknown_rink_xy"][0, 0, 0, 0] = float("nan")
    out = m.add_rink_anchor(seq_info)
    for k in ("trajectory", "unknown"):
        assert torch.equal(out[f"{k}_features"], seq_info[f"{k}_features"])
    assert torch.equal(m.rink_anchor.no_position.detach(), torch.zeros(D))
    # the rest of seq_info is shared, the features are not modified in place
    assert out["trajectory_boxes"] is seq_info["trajectory_boxes"]


def test_missing_xy_means_all_tokens_invalid_not_a_crash():
    m = _motip()
    _poke(m)
    seq_info = _seq_info(with_xy=False)
    out = m.add_rink_anchor(seq_info)
    for k in ("trajectory", "unknown"):
        assert torch.allclose(out[f"{k}_features"], seq_info[f"{k}_features"] + m.rink_anchor.no_position)


def test_valid_and_invalid_tokens_get_different_embeddings():
    m = _motip()
    _poke(m)
    m.eval()
    seq_info = _seq_info()
    seq_info["trajectory_rink_xy"][0, 0, 1, 2] = float("nan")
    out = m.add_rink_anchor(seq_info)
    delta = out["trajectory_features"] - seq_info["trajectory_features"]
    assert torch.allclose(delta[0, 0, 1, 2], m.rink_anchor.no_position, atol=1e-4)         # invalid -> no-position embedding
    valid = delta[0, 0, 0, 0]
    assert not torch.allclose(valid, m.rink_anchor.no_position)
    assert torch.allclose(valid, m.rink_anchor(seq_info["trajectory_rink_xy"][0, 0, 0, 0]), atol=1e-4)
    # half-NaN (one coordinate) is invalid as a whole
    assert torch.allclose(m.rink_anchor(torch.tensor([1.0, float("nan")])), m.rink_anchor.no_position)


def test_encoding_depends_on_position_not_batch_layout():
    m = _motip()
    _poke(m)
    m.eval()
    xy = torch.tensor([[1.0, 2.0], [-20.0, 7.5], [1.0, 2.0]])
    out = m.rink_anchor(xy)
    assert torch.allclose(out[0], out[2]) and not torch.allclose(out[0], out[1])
    assert torch.allclose(m.rink_anchor(xy[:1])[0], out[0], atol=1e-6)
    assert not torch.allclose(m.rink_anchor(torch.tensor([[1.0, 2.0]])), m.rink_anchor(torch.tensor([[2.0, 2.0]])))


def test_dropout_only_in_train_mode():
    m = _motip(dropout=1.0)
    _poke(m)
    xy = torch.rand(64, 2) * 20 - 10
    m.train()
    assert torch.allclose(m.rink_anchor(xy), m.rink_anchor.no_position.expand(64, D))       # every token dropped
    m.eval()
    out = m.rink_anchor(xy)
    assert not torch.allclose(out, m.rink_anchor.no_position.expand(64, D))               # eval: never dropped
    assert torch.equal(out, m.rink_anchor(xy))                                              # and deterministic
    # p = 0.25: about a quarter of the tokens are replaced in train mode
    m2 = _motip(dropout=0.25)
    _poke(m2)
    m2.train()
    torch.manual_seed(0)
    xy = torch.rand(4000, 2) * 20 - 10
    dropped = torch.isclose(m2.rink_anchor(xy), m2.rink_anchor.no_position.expand(4000, D)).all(dim=-1).float().mean()
    assert dropped.item() == pytest.approx(0.25, abs=0.03)


def test_gradients_reach_every_parameter_after_zero_init():
    m = _motip()
    seq_info = _seq_info()
    seq_info["unknown_rink_xy"][0, 0, 0, 0] = float("nan")          # one invalid token -> no_position gets a gradient
    out = m.add_rink_anchor(seq_info)
    (out["unknown_features"].sum() + out["trajectory_features"].sum()).backward()
    # at zero init the last layer's weight gets gradient (the hidden activations are non-zero)
    assert m.rink_anchor.mlp[-1].weight.grad.abs().sum() > 0
    assert m.rink_anchor.mlp[-1].bias.grad.abs().sum() > 0
    assert m.rink_anchor.no_position.grad.abs().sum() > 0


def test_fp16_model_half_keeps_fp32_frequencies():
    m = _motip()
    _poke(m)
    xy = torch.tensor([[29.5, -14.2]])
    ref = m.rink_anchor.encode(xy)
    m.half()
    assert torch.allclose(m.rink_anchor.encode(xy), ref)          # no fp16 rounding of the frequencies


def test_requires_matching_dims():
    class Dec(_StubIDDecoder):
        feature_dim = D + 1
    with pytest.raises(AssertionError):
        MOTIP(detr=_StubDetr(), detr_framework="rf_detr", only_detr=False, trajectory_modeling=None,
              id_decoder=Dec(), detr_hidden_dim=D, rink_anchor=True)


# --------------------------------------------------------------------------------------------------
# projection (training: normalised boxes + per-frame matrix)
# --------------------------------------------------------------------------------------------------
def _hn(h=H0, w=W0, hh=H0_PX):
    """normalised current-image coords -> metres for an un-augmented frame"""
    return torch.as_tensor(h @ np.diag([w, hh, 1.0]), dtype=torch.float64)


def test_project_feet_matches_hand_value():
    box = torch.tensor([[0.5, 0.5, 0.1, 0.2]])           # feet = (0.5, 0.6) -> px (640, 432)
    xy = project_feet(box, _hn(), torch.tensor(False), torch.tensor(BOUNDS))
    assert xy[0].tolist() == pytest.approx([0.05 * 640 - 32, 0.05 * 432 - 18])


def test_project_feet_invalid_cases_are_nan():
    boxes = torch.tensor([[0.5, 0.5, 0.1, 0.2], [0.98, 0.99, 0.1, 0.2], [0.5, 0.5, 0.1, 0.2]])
    h = torch.stack([_hn(), _hn(), torch.full((3, 3), float("nan"), dtype=torch.float64)])
    xy = project_feet(boxes, h, torch.zeros(3, dtype=torch.bool), torch.tensor(BOUNDS))
    assert torch.isfinite(xy[0]).all()
    assert torch.isnan(xy[1]).all()           # lands off the rink (x ~ 31.5)
    assert torch.isnan(xy[2]).all()           # no homography
    # bounds = NaN (sequence without a sidecar) -> invalid
    assert torch.isnan(project_feet(boxes[:1], _hn(), torch.tensor(False), torch.full((2,), float("nan")))).all()


def test_project_feet_flip_negates_x_only():
    box = torch.tensor([[0.3, 0.5, 0.1, 0.2]])
    a = project_feet(box, _hn(), torch.tensor(False), torch.tensor(BOUNDS))
    b = project_feet(box, _hn(), torch.tensor(True), torch.tensor(BOUNDS))
    assert b[0, 0] == pytest.approx(-a[0, 0].item()) and b[0, 1] == pytest.approx(a[0, 1].item())


def test_training_projection_equals_inference_projection():
    """Training projects normalised boxes with ``H @ S`` (S = norm -> original px); the RuntimeTracker multiplies by the
    frame size first. Same metres."""
    tr = RuntimeTracker(model=_Rec(), sequence_hw=(H0_PX, W0), rink_anchor=True, rink_bounds_m=BOUNDS)
    boxes = torch.rand(30, 4, device=distributed_device()) * torch.tensor([0.8, 0.4, 0.1, 0.2], device=distributed_device()) + 0.1
    tr._set_frame_rink(boxes, H0)
    inf = torch.as_tensor(tr._rink_all[:, :3])
    train = project_feet(boxes.cpu(), _hn(), torch.tensor(False), torch.tensor(BOUNDS))
    ok = inf[:, 2] > 0.5
    assert ok.any() and torch.equal(ok, torch.isfinite(train).all(-1))
    assert torch.allclose(inf[ok, :2].float(), train[ok], atol=1e-3)


# --------------------------------------------------------------------------------------------------
# augmentation tracking: flip / resize / crop are inverted, flip negates x
# --------------------------------------------------------------------------------------------------
def _aug_cfg():
    return dict(
        AUG_MAX_SHIFT_RATIO=0.0, AUG_OVERFLOW_BBOX=False, AUG_RESIZE_SCALES=[480, 512], AUG_MAX_SIZE=1000,
        AUG_RANDOM_RESIZE=[400, 500], AUG_RANDOM_CROP_MIN=300, AUG_RANDOM_CROP_MAX=450,
        AUG_BRIGHTNESS=0.0, AUG_CONTRAST=0.0, AUG_SATURATION=0.0, AUG_HUE=0.0, AUG_COLOR_JITTER_V2=False,
        NUM_ID_VOCABULARY=K, AUG_NUM_GROUPS=1, AUG_TRAJECTORY_OCCLUSION_PROB=0.0, AUG_TRAJECTORY_SWITCH_PROB=0.0,
    )


def _clip(T=3, n=4, seed=0):
    rng = np.random.RandomState(seed)
    images = [Image.new("RGB", (W0, H0_PX), (10, 20, 30)) for _ in range(T)]
    xywh = np.stack([rng.uniform(100, 1000, n), rng.uniform(200, 560, n), rng.uniform(30, 90, n), rng.uniform(60, 140, n)], 1)
    anns = [{
        "bbox": torch.tensor(xywh, dtype=torch.float32), "category": torch.zeros(n, dtype=torch.int64),
        "id": torch.arange(n), "visibility": torch.ones(n), "is_legal": True,
    } for _ in range(T)]
    metas = [{"is_static": False, "rink_H": H0.copy(), "rink_bounds": np.asarray(BOUNDS),
              "rink_aug": R.aug_init(W0, H0_PX)} for _ in range(T)]
    return images, anns, metas, xywh


def _pipeline(flip_p, with_crop):
    steps = [MultiBoxXYWHtoXYXY(), MultiRandomHorizontalFlip(p=flip_p), MultiRandomResize(sizes=[480, 512], max_size=1000)]
    if with_crop:
        steps += [MultiRandomCrop(min_size=300, max_size=450, overflow_bbox=True),
                  MultiRandomResize(sizes=[400, 500], max_size=1000)]
    steps += [MultiBoxXYXYtoCXCYWH(), MultiToTensor(), MultiStack(), MultiNormalizeBoundingBoxes(), MultiRinkFinalize()]
    return MultiCompose(steps)


@pytest.mark.parametrize("flip_p,with_crop", [(0.0, False), (1.0, False), (0.0, True), (1.0, True)])
def test_augmentation_is_inverted_and_flip_negates_x(flip_p, with_crop):
    images, anns, metas, xywh = _clip()
    random.seed(3)
    torch.manual_seed(3)
    imgs, anns2, metas2 = _pipeline(flip_p, with_crop)(images, anns, metas)
    # ground truth: project the ORIGINAL (un-augmented) feet with the original matrix
    feet = np.stack([xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3], np.ones(len(xywh))], 1)
    ref = feet @ H0.T
    ref = ref[:, :2] / ref[:, 2:3]
    batch = rink_batch_from_metas([metas2])
    assert bool(batch["flip"][0, 0]) == (flip_p == 1.0)
    for t in range(len(metas2)):
        boxes, ids = anns2[t]["bbox"], anns2[t]["id"].numpy()
        assert len(ids) == len(xywh) or with_crop, "only the crop may drop boxes"
        # the crop leaves boxes unclamped (overflow_bbox) and drops those fully outside; they may leave the crop, so
        # use a wide bounds box: this checks the inversion, not the rink gate
        xy = project_feet(boxes, batch["H"][0, t], batch["flip"][0, t], torch.tensor([1e9, 1e9], dtype=torch.float64))
        want = ref[ids] * np.array([-1.0 if flip_p == 1.0 else 1.0, 1.0])
        assert np.allclose(xy.numpy(), want, atol=0.02), (xy.numpy(), want)       # metres; boxes are float32 px / size


def test_crop_actually_applied_in_some_cases():
    """Guard for the test above: the crop branch must really run (otherwise the crop inversion is untested)."""
    cropped = 0
    for seed in range(8):
        images, anns, metas, _ = _clip(seed=seed)
        random.seed(seed)
        torch.manual_seed(seed)
        _, _, m2 = _pipeline(0.0, True)(images, anns, metas)
        cropped += int(not np.allclose(m2[0]["rink_Hn"], (H0 @ R.aug_norm_to_orig(R.aug_init(W0, H0_PX)))))
    assert cropped > 0


def test_flip_matches_a_mirrored_homography():
    """The flipped image is the mirror of the frame; the mirrored homography ``diag(-1, 1, 1) @ H @ F`` (F the pixel flip)
    sends its feet to ``-x`` of the original (and is a legitimate rink homography: see the session note)."""
    xy = np.array([[200.0, 400.0], [900.0, 300.0]])
    pts = np.concatenate([xy, np.ones((2, 1))], 1)
    orig = (pts @ H0.T)
    orig = orig[:, :2] / orig[:, 2:3]
    flipped = pts.copy()
    flipped[:, 0] = W0 - flipped[:, 0]                     # where the same feet land in the flipped image
    F = np.array([[-1.0, 0, W0], [0, 1, 0], [0, 0, 1]])
    mirrored = np.diag([-1.0, 1.0, 1.0]) @ H0 @ F          # flipped px -> metres of the mirrored scene
    got = flipped @ mirrored.T
    got = got[:, :2] / got[:, 2:3]
    assert np.allclose(got, orig * np.array([-1.0, 1.0]), atol=1e-9)


# --------------------------------------------------------------------------------------------------
# prepare_for_motip with the rink inputs
# --------------------------------------------------------------------------------------------------
def _prepare(rink, flips=(False, False)):
    """Two frames, one clip, 2 GT objects, one group; query q matches GT j via ``detr_indices``."""
    B, T, G, N, Q = 1, 2, 1, 2, 5
    pred_boxes = torch.zeros(B * T, Q, 4)
    pred_boxes[:, :, 2:] = 0.1
    pred_boxes[0, 3] = torch.tensor([0.25, 0.5, 0.1, 0.2])
    pred_boxes[0, 1] = torch.tensor([0.75, 0.5, 0.1, 0.2])
    pred_boxes[1, 3] = torch.tensor([0.26, 0.5, 0.1, 0.2])
    pred_boxes[1, 1] = torch.tensor([0.74, 0.5, 0.1, 0.2])
    outputs = {"pred_logits": torch.zeros(B * T, Q, 2), "outputs": torch.randn(B * T, Q, 4), "pred_boxes": pred_boxes}
    indices = [(torch.tensor([3, 1]), torch.tensor([0, 1])) for _ in range(B * T)]
    anns = [[{
        "bbox": torch.zeros(N, 4),
        "trajectory_id_labels": torch.arange(N)[None, None].repeat(G, 1, 1), "trajectory_id_masks": torch.zeros(G, 1, N, dtype=torch.bool),
        "trajectory_ann_idxs": torch.arange(N)[None, None].repeat(G, 1, 1), "trajectory_times": torch.full((G, 1, N), t),
        "unknown_id_labels": torch.arange(N)[None, None].repeat(G, 1, 1), "unknown_id_masks": torch.zeros(G, 1, N, dtype=torch.bool),
        "unknown_ann_idxs": torch.arange(N)[None, None].repeat(G, 1, 1), "unknown_times": torch.full((G, 1, N), t),
    } for t in range(T)]]
    if rink is not None:
        rink = {"H": torch.stack([_hn(), _hn()])[None], "flip": torch.tensor([list(flips)]),
                "bounds": torch.tensor(BOUNDS, dtype=torch.float64).expand(1, T, 2).clone()}
    return prepare_for_motip(outputs, anns, indices, rink=rink), pred_boxes


def test_prepare_for_motip_projects_the_predicted_boxes_of_each_token():
    seq_info, pred_boxes = _prepare(rink=True)
    assert seq_info["trajectory_rink_xy"].shape == seq_info["trajectory_boxes"].shape[:-1] + (2,)
    for t in range(2):
        for j, q in enumerate((3, 1)):
            want = project_feet(pred_boxes[t, q][None], _hn(), torch.tensor(False), torch.tensor(BOUNDS))[0]
            assert torch.allclose(seq_info["unknown_rink_xy"][0, 0, t, j], want)
            assert torch.allclose(seq_info["trajectory_rink_xy"][0, 0, t, j], want)


def test_prepare_for_motip_flip_and_missing_frames():
    (a, _), (b, _) = _prepare(rink=True), _prepare(rink=True, flips=(True, False))
    assert torch.allclose(b["unknown_rink_xy"][0, 0, 0, :, 0], -a["unknown_rink_xy"][0, 0, 0, :, 0])
    assert torch.equal(b["unknown_rink_xy"][0, 0, 1], a["unknown_rink_xy"][0, 0, 1])
    # a frame without a homography: all NaN there, the other frame untouched
    seq_info, _ = _prepare(rink=True)
    metas = [[{"rink_Hn": np.full((3, 3), np.nan), "rink_flip": False, "rink_bounds": np.full(2, np.nan)},
              {"rink_Hn": _hn().numpy(), "rink_flip": False, "rink_bounds": np.asarray(BOUNDS)}]]
    batch = rink_batch_from_metas(metas)
    assert torch.isnan(batch["H"][0, 0]).all() and torch.isfinite(batch["H"][0, 1]).all()
    # metas without any rink key (a dataset without a sidecar) become all-NaN too
    nb = rink_batch_from_metas([[{}, {}]])
    assert torch.isnan(nb["H"]).all() and torch.isnan(nb["bounds"]).all() and not nb["flip"].any()


# --------------------------------------------------------------------------------------------------
# RuntimeTracker (inference): the history carries the feet positions
# --------------------------------------------------------------------------------------------------
class _Rec:
    num_id_vocabulary = K
    num_classes = 1
    seen = None

    def eval(self):
        return self

    def __call__(self, seq_info=None, part=None, **_):
        if part == "id_decoder":
            self.seen = seq_info
            return torch.zeros(1, 1, 1, seq_info["unknown_boxes"].shape[-2], K + 1, device=distributed_device()), None, None
        return seq_info


def _prime(tr):
    dev = distributed_device()
    tr.trajectory_boxes = torch.rand(2, 3, 4, device=dev)
    tr.trajectory_masks = torch.zeros(2, 3, dtype=torch.bool, device=dev)
    tr.trajectory_id_labels = torch.tensor([[0, 1, 2]] * 2, device=dev)
    tr.trajectory_features = torch.zeros(2, 3, 256, device=dev)
    tr.trajectory_times = torch.arange(2, device=dev)[:, None].expand(2, 3).contiguous()
    tr.trajectory_rink = torch.zeros(2, 3, 2, device=dev)


def _det(xs):
    dev = distributed_device()
    boxes = torch.tensor([[x, 0.5, 0.05, 0.1] for x in xs], device=dev)
    n = len(xs)
    return (torch.full((n,), 0.9, device=dev), torch.zeros(n, dtype=torch.int64, device=dev), boxes, torch.zeros(n, 256, device=dev))


def _feet_m(x, h=H0):
    px = np.array([x * W0, 0.55 * H0_PX, 1.0])
    p = h @ px
    return p[:2] / p[2]


def test_runtime_tracker_keeps_feet_xy_with_the_history():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(H0_PX, W0), rink_anchor=True, rink_bounds_m=BOUNDS)
    tr._process_detections(*_det([0.2, 0.6]), rink_H=H0)               # frame 0: no history, nothing decoded
    assert tr.trajectory_rink.shape == (1, 2, 2)
    got = {int(l): tr.trajectory_rink[0, i].cpu().numpy() for i, l in enumerate(tr.trajectory_id_labels[0])}
    assert sorted(got) == [0, 1]
    assert np.allclose(sorted(g[0] for g in got.values()), sorted(_feet_m(x)[0] for x in (0.2, 0.6)), atol=1e-4)
    # the camera pans: the same pixel is another place on the rink
    pan = H0.copy()
    pan[0, 2] -= 6.0
    tr._process_detections(*_det([0.2, 0.62]), rink_H=pan)
    seen = model.seen
    assert seen["trajectory_rink_xy"].shape == (1, 1, 1, 2, 2)
    assert seen["unknown_rink_xy"].shape == (1, 1, 1, 2, 2)
    assert np.allclose(sorted(seen["trajectory_rink_xy"][0, 0, 0, :, 0].cpu().tolist()),
                       sorted(_feet_m(x)[0] for x in (0.2, 0.6)), atol=1e-4)           # history keeps its own frame's H
    assert np.allclose(sorted(seen["unknown_rink_xy"][0, 0, 0, :, 0].cpu().tolist()),
                       sorted(_feet_m(x, pan)[0] for x in (0.2, 0.62)), atol=1e-4)
    # history is now two frames long and aligned with boxes
    assert tr.trajectory_rink.shape[:2] == tr.trajectory_boxes.shape[:2]


def test_runtime_tracker_missing_homography_gives_invalid_tokens_not_a_crash():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(H0_PX, W0), rink_anchor=True, rink_bounds_m=BOUNDS)
    tr._process_detections(*_det([0.2, 0.6]), rink_H=None)
    assert torch.isnan(tr.trajectory_rink).all()
    tr._process_detections(*_det([0.2, 0.6]), rink_H=np.full((3, 3), np.nan))
    assert torch.isnan(model.seen["unknown_rink_xy"]).all() and torch.isnan(model.seen["trajectory_rink_xy"]).all()
    tr._process_detections(*_det([0.2, 0.6]), rink_H=H0)                 # a good frame afterwards works
    assert torch.isfinite(model.seen["unknown_rink_xy"]).all()
    # a detection that lands off the rink is invalid on its own
    tr._process_detections(*_det([0.2, 1.0]), rink_H=H0)
    u = model.seen["unknown_rink_xy"][0, 0, 0]
    assert torch.isfinite(u).all(-1).sum() == 1


def test_runtime_tracker_history_stays_aligned_through_newborns_and_dropouts():
    model = _Rec()
    tr = RuntimeTracker(model=model, sequence_hw=(H0_PX, W0), rink_anchor=True, rink_bounds_m=BOUNDS, miss_tolerance=5)
    for xs in ([0.2], [0.2, 0.5], [0.5, 0.8], [0.8], [0.2, 0.8]):
        tr._process_detections(*_det(xs), rink_H=H0)
        assert tr.trajectory_rink.shape[:2] == tr.trajectory_boxes.shape[:2] == tr.trajectory_features.shape[:2]
        visible = ~tr.trajectory_masks
        # wherever a token is visible its stored xy is finite and matches the stored box's feet
        for t, n in visible.nonzero().cpu().tolist():
            b = tr.trajectory_boxes[t, n].cpu().numpy()
            want = (H0 @ np.array([b[0] * W0, (b[1] + b[3] / 2) * H0_PX, 1.0]))[:2]
            assert np.allclose(tr.trajectory_rink[t, n].cpu().numpy(), want, atol=1e-3)
        assert torch.isnan(tr.trajectory_rink[visible]).sum() == 0


# --------------------------------------------------------------------------------------------------
# checkpoint loading and optimiser groups
# --------------------------------------------------------------------------------------------------
def test_load_checkpoint_fills_missing_rink_anchor_params(tmp_path, monkeypatch):
    import models.misc as misc
    monkeypatch.setattr(misc, "resize_rel_pos_embeds", lambda model, state: None)
    src = _motip(rink=False)                          # a checkpoint trained without the rink anchor
    path = tmp_path / "ckpt.pth"
    torch.save({"model": src.state_dict()}, path)
    dst = _motip(rink=True)
    _poke(dst)                                        # whatever the init is, the load must not touch it
    before = {k: v.clone() for k, v in dst.state_dict().items() if k.startswith("rink_anchor")}
    misc.load_checkpoint(dst, str(path))
    for k, v in before.items():
        assert torch.equal(dst.state_dict()[k], v)
    # and a checkpoint that has them overrides the init
    full = _motip(rink=True)
    path2 = tmp_path / "ckpt2.pth"
    torch.save({"model": full.state_dict()}, path2)
    misc.load_checkpoint(dst, str(path2))
    for k, v in full.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


class _GroupModel(nn.Module):
    def __init__(self, rink):
        super().__init__()
        self.detr = nn.Linear(2, 2)
        self.trajectory_modeling = nn.Linear(2, 2)
        self.id_decoder = nn.Linear(2, 2)
        if rink:
            self.rink_anchor = RinkAnchor(dim=8, n_freq=4, hidden=8)


def _group_cfg(**kw):
    return dict(DETR_FRAMEWORK="rf_detr", LR=2e-5, WEIGHT_DECAY=1e-4, LR_LINEAR_PROJ_NAMES=["reference_points"],
                LR_LINEAR_PROJ_SCALE=0.1, LR_DICTIONARY_NAMES=["word_to_embed"], LR_DICTIONARY_SCALE=1.0, **kw)


def test_param_groups_rink_anchor_gets_own_lr_and_no_wd_on_bias_and_embedding():
    model = _GroupModel(rink=True)
    for p in model.detr.parameters():
        p.requires_grad = False
    groups = get_param_groups(model, _group_cfg(LR_RINK_ANCHOR_SCALE=20.0))
    names = {id(p): n for n, p in model.named_parameters()}
    rink = [(names[id(g["params"])], g) for g in groups if names[id(g["params"])].startswith("rink_anchor.")]
    assert len(rink) == 5          # two weights, two biases, no_position
    for n, g in rink:
        assert g["lr"] == pytest.approx(2e-5 * 20.0) and g["lr_scale"] == 20.0
        if n.endswith("bias") or n.endswith("no_position"):
            assert g["weight_decay"] == 0.0
        else:
            assert "weight_decay" not in g          # the optimiser default (WEIGHT_DECAY) applies
    # each parameter is in exactly one group, and the last group (used for LR logging) is the remaining params
    all_ids = [id(g["params"]) for g in groups]
    assert len(all_ids) == len(set(all_ids))
    assert names[all_ids[-1]].startswith(("trajectory_modeling", "id_decoder"))
    assert {names[i] for i in all_ids} == {n for n, p in model.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------------------------------
# sidecar loader: row alignment with gt.txt, frame mapping, projection reproduces gt_rink.txt
# --------------------------------------------------------------------------------------------------
def _write_sequence(root, gt_boxes, h_by_frame, margin=1.0):
    """gt_boxes: list of (frame, tid, x, y, w, h) 1-based frames. h_by_frame: {frame: image->metres 3x3 or None}"""
    seq = os.path.join(root, "seq0")
    os.makedirs(os.path.join(seq, "gt"), exist_ok=True)
    os.makedirs(os.path.join(seq, "rink"), exist_ok=True)
    m = np.diag([2.0, 2.0, 1.0])                                  # metres -> template px (x2)
    inv_m = np.linalg.inv(m)
    with open(os.path.join(seq, "gt", "gt.txt"), "w") as f:
        for fr, tid, x, y, w, h in gt_boxes:
            f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
    with open(os.path.join(seq, "rink", "homography.jsonl"), "w") as f:
        for fr in range(1, max(h_by_frame) + 1):
            k = h_by_frame.get(fr)
            # raw H (template px -> frame px) such that inv(H @ m) = k  ->  H = inv(k) @ inv(m)
            raw = None if k is None else (np.linalg.inv(k) @ inv_m).tolist()
            f.write(json.dumps({"frame": fr, "src_frame": 100 + fr, "H": raw}) + "\n")
    with open(os.path.join(seq, "rink", "rink_meta.json"), "w") as f:
        json.dump({"metres_to_template_px": m.tolist(), "half_extent_m": [30.0, 15.0], "margin_m": margin}, f)
    rows = []
    rink = R.load_sequence_rink(seq)
    for fr, tid, x, y, w, h in gt_boxes:
        k = h_by_frame.get(fr)
        if k is None:
            rows.append(f"{fr},{tid},nan,nan,0,0")
            continue
        p = k @ np.array([x + w / 2, y + h, 1.0])
        xy = p[:2] / p[2]
        ok = abs(xy[0]) <= 31 and abs(xy[1]) <= 16
        rows.append(f"{fr},{tid},{xy[0]:.3f},{xy[1]:.3f},{int(ok)},1" if ok else f"{fr},{tid},nan,nan,0,1")
    with open(os.path.join(seq, "rink", "gt_rink.txt"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return seq, rink


def test_loader_alignment_with_gt_txt_and_projection_reproduces_gt_rink(tmp_path):
    k1 = H0
    k3 = H0.copy()
    k3[0, 2] += 4.0
    boxes = [(1, 5, 100, 200, 40, 90), (1, 7, 600, 300, 50, 100), (2, 5, 110, 205, 40, 90), (2, 7, 610, 300, 50, 100),
             (3, 5, 120, 210, 40, 90), (3, 7, 1270, 700, 50, 100)]          # (3, 7) is off the rink
    seq, rink = _write_sequence(str(tmp_path), boxes, {1: k1, 2: None, 3: k3})
    assert rink.n_frames == 3 and rink.bounds_m == (31.0, 16.0)
    gt = [ln.split(",") for ln in open(os.path.join(seq, "gt", "gt.txt")).read().split()]
    side = R.load_gt_rink(seq)
    assert side.shape == (len(gt), 6)
    assert [int(g[0]) for g in gt] == side[:, 0].astype(int).tolist()           # row i <-> gt.txt row i
    assert [int(g[1]) for g in gt] == side[:, 1].astype(int).tolist()
    for i, g in enumerate(gt):
        x, y, w, h = map(float, g[2:6])
        fr = int(g[0])
        feet = np.array([x + w / 2, y + h, 1.0])
        proj = rink.frame_H(fr - 1) @ feet                                      # 1-based gt frame -> 0-based index
        xy = proj[:2] / proj[2]
        if side[i, 4] == 1:
            assert xy == pytest.approx(side[i, 2:4], abs=2e-3)
        else:
            assert np.isnan(side[i, 2:4]).all()
    assert np.isnan(rink.frame_H(1)).all()                                      # frame 2 had no homography
    assert np.isnan(rink.frame_H(99)).all() and np.isnan(rink.frame_H(-1)).all()
    assert side[5, 4] == 0 and side[5, 5] == 1                                  # off rink: h_ok but not valid
    assert side[2, 4] == 0 and side[2, 5] == 0                                  # no homography: neither


def test_loader_returns_none_without_a_sidecar(tmp_path):
    os.makedirs(tmp_path / "s")
    assert R.load_sequence_rink(str(tmp_path / "s")) is None
    assert R.load_gt_rink(str(tmp_path / "s")) is None
    os.makedirs(tmp_path / "s" / "rink")
    json.dump({"metres_to_template_px": None, "half_extent_m": None}, open(tmp_path / "s" / "rink" / "rink_meta.json", "w"))
    open(tmp_path / "s" / "rink" / "homography.jsonl", "w").write("")
    assert R.load_sequence_rink(str(tmp_path / "s")) is None                   # clip without a homography stem


def test_hockey_dataset_loads_rink_only_when_asked(tmp_path):
    root = tmp_path / "ds" / "Hockey" / "train"
    seq = root / "seq0"
    os.makedirs(seq / "img1")
    os.makedirs(seq / "gt")
    (seq / "seqinfo.ini").write_text("[Sequence]\nname=seq0\nimWidth=1280\nimHeight=720\nseqLength=3\n")
    (seq / "gt" / "gt.txt").write_text("1,1,10,10,20,40,1,1,1\n2,1,12,10,20,40,1,1,1\n3,1,14,10,20,40,1,1,1\n")
    for i in (1, 2, 3):
        Image.new("RGB", (1280, 720)).save(seq / "img1" / f"{i:08d}.jpg")
    _write_sequence(str(root), [(1, 1, 10, 10, 20, 40), (2, 1, 12, 10, 20, 40), (3, 1, 14, 10, 20, 40)], {1: H0, 2: H0, 3: None})
    # _write_sequence made root/seq0 too: same dir, its gt/rink files are fine
    from data.hockey import Hockey
    assert Hockey(data_root=str(tmp_path / "ds"), split="train").get_rink() == {}
    r = Hockey(data_root=str(tmp_path / "ds"), split="train", load_rink=True).get_rink()
    assert set(r) == {"seq0"} and r["seq0"].n_frames == 3
    # JointDataset: metas carry the rink inputs only when the anchor is on, and the real pipeline finishes them
    from data.joint_dataset import JointDataset
    for on in (False, True):
        ds = JointDataset(data_root=str(tmp_path / "ds"), datasets=["Hockey"], splits=["train"],
                          transforms=_pipeline(0.0, False), dataset_kwargs={"Hockey": {"load_rink": on}}, rink_anchor=on)
        _, _, metas = ds[{"dataset": "Hockey", "split": "train", "sequence": "seq0", "frame_idxs": [0, 1, 2]}]
        assert ("rink_Hn" in metas[0]) == on
        if on:
            assert np.isfinite(metas[0]["rink_Hn"]).all() and np.isnan(metas[2]["rink_Hn"]).all()     # frame 3: no H


# --------------------------------------------------------------------------------------------------
# a real checkpoint: the zero-init anchor leaves the ID decoder's inputs (and logits) untouched
# --------------------------------------------------------------------------------------------------
CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rfdetr_stage2_ckpt4.pth")


@pytest.mark.skipif(not os.path.exists(CKPT), reason="rfdetr_stage2_ckpt4.pth not available")
def test_real_checkpoint_zero_init_identity():
    from configs.util import load_super_config
    from models.misc import load_checkpoint
    from models.motip import build as build_motip
    from utils.misc import yaml_to_dict

    def build(extra):
        cfg = yaml_to_dict("configs/finetune_crossing_rfdetr_s1_long.yaml")
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
        cfg.update({"DETR_PRETRAIN": None, "NUM_CLASSES": 3, **extra})
        model, _ = build_motip(config=cfg)
        load_checkpoint(model, path=CKPT)
        return model.eval()

    plain, rink = build({}), build({"RINK_ANCHOR": True, "RINK_ANCHOR_DROPOUT": 0.1})
    assert rink.rink_anchor is not None and plain.rink_anchor is None
    # every shared parameter is identical (the same checkpoint), and the new ones are zero where it matters
    ps = plain.state_dict()
    for k, v in rink.state_dict().items():
        if k in ps:
            assert torch.equal(v, ps[k]), k
    assert torch.equal(rink.rink_anchor.mlp[-1].weight, torch.zeros_like(rink.rink_anchor.mlp[-1].weight))
    assert torch.equal(rink.rink_anchor.no_position, torch.zeros_like(rink.rink_anchor.no_position))
    torch.manual_seed(0)
    d = plain.detr_hidden_dim
    seq_info = {
        "trajectory_features": torch.randn(1, 1, 4, 5, d), "unknown_features": torch.randn(1, 1, 4, 3, d),
        "trajectory_boxes": torch.rand(1, 1, 4, 5, 4) * 0.5 + 0.2, "unknown_boxes": torch.rand(1, 1, 4, 3, 4) * 0.5 + 0.2,
        "trajectory_id_labels": torch.randint(0, 10, (1, 1, 4, 5)), "trajectory_times": torch.arange(4)[None, None, :, None].expand(1, 1, 4, 5).contiguous(),
        "trajectory_masks": torch.zeros(1, 1, 4, 5, dtype=torch.bool),
        "unknown_id_labels": torch.randint(0, 10, (1, 1, 4, 3)), "unknown_times": torch.arange(4)[None, None, :, None].expand(1, 1, 4, 3).contiguous(),
        "unknown_masks": torch.zeros(1, 1, 4, 3, dtype=torch.bool),
    }
    with_xy = dict(seq_info)
    with_xy["trajectory_rink_xy"] = torch.rand(1, 1, 4, 5, 2) * 40 - 20
    with_xy["unknown_rink_xy"] = torch.rand(1, 1, 4, 3, 2) * 40 - 20
    with_xy["unknown_rink_xy"][0, 0, 0, 0] = float("nan")
    with torch.no_grad():
        ref, *_ = plain(seq_info=copy.copy(plain(seq_info=copy.copy(seq_info), part="trajectory_modeling")), part="id_decoder")
        got, *_ = rink(seq_info=copy.copy(rink(seq_info=copy.copy(with_xy), part="trajectory_modeling")), part="id_decoder")
        got_missing, *_ = rink(seq_info=copy.copy(rink(seq_info=copy.copy(seq_info), part="trajectory_modeling")), part="id_decoder")
    assert torch.equal(ref, got) and torch.equal(ref, got_missing)


def test_building_the_anchor_does_not_move_the_global_rng():
    """Arms with and without the anchor must draw the same data order / augmentations for the same seed."""
    torch.manual_seed(123)
    _motip(rink=False)
    a = torch.rand(3)
    torch.manual_seed(123)
    _motip(rink=True)
    b = torch.rand(3)
    assert torch.equal(a, b)
    # and the init itself is deterministic whatever the global seed
    torch.manual_seed(1)
    w1 = _motip(rink=True).rink_anchor.mlp[0].weight.clone()
    torch.manual_seed(2)
    assert torch.equal(w1, _motip(rink=True).rink_anchor.mlp[0].weight)
