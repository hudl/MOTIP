# Copyright (c) Ruopeng Gao. All Rights Reserved.

import sys
import os

from .motip import MOTIP
from structures.args import Args
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder

# Prefer the bundled copy (available in both local dev and SageMaker code channel).
# Fall back to the ihc-od repo path for developers who don't have the copy.
_RFDETR_BUNDLED = os.path.join(os.path.dirname(__file__), "../../third_party")
_RFDETR_REPO = os.path.join(
    os.path.dirname(__file__),
    "../../../../third_party/aml-ice-hockey/ihc-od/third_party/rf-detr",
)



# --------------------------------------------------------------------------------------------------
# Last-layer matching guard (RF-DETR). RF-DETR's HungarianMatcher replaces inf/nan costs with
# `C.max() * 2`, but `C.max()` is NaN as soon as one entry is NaN, so a single degenerate box (GIoU 0/0)
# makes linear_sum_assignment raise and kills the whole DDP job. These keep the match finite.
# --------------------------------------------------------------------------------------------------
MATCH_MIN_WH = 1e-4   # normalised; ~0.1 px at 1920 wide, far below any real box
MATCH_GUARD_STATS = {"calls": 0, "sanitized": 0, "fallback": 0}


def sanitize_match_inputs(single: dict, targets: list) -> tuple[dict, list, bool]:
    """Finite logits/boxes and boxes with a minimum w/h. Returns (single, targets, changed)."""
    import torch
    changed = False
    out = {}
    for k, v in single.items():
        if not torch.isfinite(v).all():
            v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
            changed = True
        out[k] = v
    b = out["pred_boxes"]
    if (b[..., 2:] < MATCH_MIN_WH).any():
        b = b.clone()
        b[..., 2:] = b[..., 2:].clamp(min=MATCH_MIN_WH)
        out["pred_boxes"] = b
        changed = True
    new_targets = []
    for t in targets:
        tb = t["boxes"]
        if tb.numel() and ((tb[..., 2:] < MATCH_MIN_WH).any() or not torch.isfinite(tb).all()):
            tb = torch.nan_to_num(tb, nan=0.5, posinf=1.0, neginf=0.0).clone()
            tb[..., 2:] = tb[..., 2:].clamp(min=MATCH_MIN_WH)
            t = {**t, "boxes": tb}
            changed = True
        new_targets.append(t)
    return out, new_targets, changed


def l1_fallback_match(single: dict, targets: list, group_detr: int = 1) -> list:
    """Hungarian on finite L1 box cost only, the same (pred_idx, tgt_idx) format as the RF-DETR matcher
    (with group_detr > 1: one match per query group, query indices offset by the group, as RF-DETR does)."""
    import torch
    from scipy.optimize import linear_sum_assignment
    res = []
    boxes = single["pred_boxes"]
    g_q = boxes.shape[1] // max(1, group_detr)
    for i, t in enumerate(targets):
        tb = t["boxes"]
        if tb.numel() == 0:
            e = torch.as_tensor([], dtype=torch.int64)
            res.append((e, e.clone()))
            continue
        rs, ks = [], []
        for g in range(max(1, group_detr)):
            c = torch.cdist(boxes[i, g * g_q:(g + 1) * g_q].float(), tb.float(), p=1)
            c = torch.nan_to_num(c, nan=1e6, posinf=1e6, neginf=1e6).cpu().numpy()
            r, k = linear_sum_assignment(c)
            rs.append(torch.as_tensor(r, dtype=torch.int64) + g * g_q)
            ks.append(torch.as_tensor(k, dtype=torch.int64))
        res.append((torch.cat(rs), torch.cat(ks)))
    return res


def guarded_match(matcher, single: dict, targets: list) -> list:
    """matcher(single, targets) on sanitised inputs; L1 fallback if it still fails. Never raises on NaN."""
    import torch
    MATCH_GUARD_STATS["calls"] += 1
    single, targets, changed = sanitize_match_inputs(single, targets)
    if changed:
        MATCH_GUARD_STATS["sanitized"] += 1
    try:
        with torch.no_grad():
            return matcher(single, targets)
    except ValueError:
        MATCH_GUARD_STATS["fallback"] += 1
        with torch.no_grad():
            return l1_fallback_match(single, targets)


# The same protection for the matcher *inside* RF-DETR's own SetCriterion (lwdetr.py SetCriterion.forward: last layer, every
# aux layer, the encoder output), which the job-killing 'matrix contains invalid numeric entries' came from (arm C, idloss resume).
DETR_MATCH_GUARD_STATS = {"calls": 0, "sanitized": 0, "fallback": 0}


class GuardedSetMatcher:
    """Drop-in for SetCriterion.matcher: matcher(outputs, targets, group_detr) on sanitised inputs, the grouped L1 fallback
    if it still raises. Never raises on non-finite costs. Counts events in DETR_MATCH_GUARD_STATS (train.py logs them).
    Matching does not depend on other ranks, so this changes nothing about the collectives; a batch whose loss is still
    non-finite is zero-weighted on every rank in train.py (nonfinite_loss_skip)."""

    def __init__(self, matcher):
        self.matcher = matcher

    def __call__(self, outputs, targets, group_detr=1):
        import torch
        DETR_MATCH_GUARD_STATS["calls"] += 1
        single = {k: outputs[k] for k in ("pred_logits", "pred_boxes")}
        single, targets, changed = sanitize_match_inputs(single, targets)
        if changed:
            DETR_MATCH_GUARD_STATS["sanitized"] += 1
        try:
            with torch.no_grad():
                return self.matcher(single, targets, group_detr=group_detr)
        except ValueError:
            DETR_MATCH_GUARD_STATS["fallback"] += 1
            with torch.no_grad():
                return l1_fallback_match(single, targets, group_detr=group_detr)


def install_criterion_match_guard(criterion):
    """Replace criterion.matcher (RF-DETR SetCriterion) with GuardedSetMatcher; returns the raw matcher."""
    raw = criterion.matcher
    if isinstance(raw, GuardedSetMatcher):
        return raw.matcher
    del criterion.matcher                       # an nn.Module child: remove it before setting a plain attribute
    criterion.matcher = GuardedSetMatcher(raw)
    return raw


def take_match_guard_events() -> dict:
    """Guard events since the last call (both matchers), for per-step logging; resets the event counters."""
    out = {
        "detr_match_sanitized": DETR_MATCH_GUARD_STATS["sanitized"],
        "detr_match_fallback": DETR_MATCH_GUARD_STATS["fallback"],
        "last_match_sanitized": MATCH_GUARD_STATS["sanitized"],
        "last_match_fallback": MATCH_GUARD_STATS["fallback"],
    }
    for d in (DETR_MATCH_GUARD_STATS, MATCH_GUARD_STATS):
        d["sanitized"] = 0
        d["fallback"] = 0
    return out


def _build_rf_detr(config, detr_args):
    """Build an LWDETR model and return (model, criterion).

    Uses MOTIP's own criterion (built from D-DETR args) so the detr_indices
    plumbing for the ID loss is preserved.  RF-DETR's IA-BCE criterion belongs
    in the ihc-od stage-1 pretrain, not here.
    """
    # If rfdetr is already installed (e.g. via pip), use it directly.
    # Otherwise use the bundled copy (MOTIP/third_party/rfdetr) which is
    # included in the SageMaker code channel and the local dev tree.
    # The top-level stub bypasses rfdetr/__init__.py's supervision import.
    if 'rfdetr' not in sys.modules:
        try:
            import rfdetr  # prefer pip-installed (e.g. 1.10.0) over bundled copy
        except ImportError:
            _repo = _RFDETR_BUNDLED if os.path.isdir(os.path.join(_RFDETR_BUNDLED, 'rfdetr')) else _RFDETR_REPO
            if _repo not in sys.path:
                sys.path.insert(0, _repo)
            try:
                import rfdetr  # noqa: F401
            except ImportError:
                import types
                _stub = types.ModuleType('rfdetr')
                _stub.__path__ = [os.path.join(_repo, 'rfdetr')]
                _stub.__package__ = 'rfdetr'
                sys.modules['rfdetr'] = _stub

    from rfdetr.models.lwdetr import build_model as build_lwdetr

    rf_args = Args()
    # --- Architecture (RF-DETR medium by default) ---
    rf_args.encoder = config.get("RFDETR_ENCODER", "dinov2_windowed_small")
    rf_args.vit_encoder_num_layers = config.get("RFDETR_VIT_ENCODER_NUM_LAYERS", 12)
    rf_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    rf_args.out_feature_indexes = config.get("RFDETR_OUT_FEATURE_INDEXES", [3, 6, 9, 12])
    rf_args.projector_scale = config.get("RFDETR_PROJECTOR_SCALE", ["P4"])
    rf_args.dec_layers = config.get("RFDETR_DEC_LAYERS", 4)
    rf_args.patch_size = config.get("RFDETR_PATCH_SIZE", 14)
    rf_args.num_windows = config.get("RFDETR_NUM_WINDOWS", 2)
    rf_args.positional_encoding_size = config.get("RFDETR_POSITIONAL_ENCODING_SIZE", 37)
    rf_args.resolution = config.get("RFDETR_RESOLUTION", 560)
    # --- Shared with MOTIP config ---
    rf_args.num_classes = config["NUM_CLASSES"]
    rf_args.device = config["DEVICE"]
    rf_args.num_queries = config["DETR_NUM_QUERIES"]
    rf_args.aux_loss = config["DETR_AUX_LOSS"]
    # group_detr: native RF-DETR 1.10.0 uses 13 groups (3900 queries); rfdetr criterion handles

    rf_args.group_detr = config.get("RFDETR_GROUP_DETR", 1)
    rf_args.two_stage = config["DETR_TWO_STAGE"]
    rf_args.lite_refpoint_refine = config.get("RFDETR_LITE_REFPOINT_REFINE", True)
    rf_args.bbox_reparam = config.get("RFDETR_BBOX_REPARAM", True)
    # --- Defaults / not used in stage-2 ---
    rf_args.pretrain_weights = None       # weights come from DETR_PRETRAIN
    rf_args.pretrained_encoder = None
    rf_args.window_block_indexes = None
    rf_args.drop_path = 0.0
    rf_args.use_cls_token = False
    rf_args.position_embedding = "sine"
    rf_args.freeze_encoder = False
    rf_args.layer_norm = True
    rf_args.rms_norm = False
    rf_args.backbone_lora = False
    rf_args.force_no_pretrain = False      # allow DINOv2 backbone weights to load
    rf_args.gradient_checkpointing = False
    rf_args.encoder_only = False
    rf_args.backbone_only = False
    rf_args.segmentation_head = False
    rf_args.ia_bce_loss = config.get("IA_BCE_LOSS", False)
    rf_args.dual_projector = False
    rf_args.dual_projector_kp_only = False
    rf_args.use_varifocal_loss = False
    rf_args.use_position_supervised_loss = False
    rf_args.sum_group_losses = False
    # --- Transformer args (defaults from RF-DETR main.py) ---
    rf_args.sa_nheads = 8
    rf_args.ca_nheads = config.get("RFDETR_CA_NHEADS", 8)
    rf_args.dropout = 0.0
    rf_args.dim_feedforward = 2048
    rf_args.dec_n_points = config.get("RFDETR_DEC_N_POINTS", 4)
    rf_args.decoder_norm = "LN"
    rf_args.num_select = config.get("DETR_NUM_QUERIES", 300)
    # Loss coefs forwarded for completeness (MOTIP criterion uses detr_args values)
    rf_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    rf_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    rf_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    rf_args.focal_alpha = config["DETR_FOCAL_ALPHA"]

    _base = build_lwdetr(rf_args)

    # Wrap LWDETR to inject the 'outputs' key MOTIP's ID loss reads.
    # RF-DETR's forward() doesn't expose decoder query embeddings by default;
    # a forward hook on the transformer captures hs without touching ihc-od.
    # The wrapper also pads inputs to a multiple of block_size (patch_size *
    # num_windows = 32) because DINOv2 requires this alignment.
    import torch.nn as _nn
    import torch.nn.functional as _F
    from utils.nested_tensor import NestedTensor as _NestedTensor

    _block_size = rf_args.patch_size * rf_args.num_windows

    class _RFDETRWithOutputs(_nn.Module):
        def __init__(self, base, block_size):
            super().__init__()
            self.base = base
            self._hs_last = None
            self._block_size = block_size
            base.transformer.register_forward_hook(self._capture_hs)

        def _capture_hs(self, module, inp, out):
            hs = out[0]   # (layers, B, Q, D); None in enc-only two_stage path
            if hs is not None:
                self._hs_last = hs

        def _pad_to_block(self, samples):
            tensors, mask = samples.decompose()
            H, W = tensors.shape[-2:]
            bs = self._block_size
            # RF-DETR's windowed self-attention reshapes patches assuming a
            # square spatial layout (num_h_patches == num_w_patches), so pad
            # to a square whose side is max(H,W) rounded up to block_size.
            side = ((max(H, W) + bs - 1) // bs) * bs
            pad_h, pad_w = side - H, side - W
            if pad_h == 0 and pad_w == 0:
                return samples
            tensors = _F.pad(tensors, [0, pad_w, 0, pad_h])
            mask = _F.pad(mask.float(), [0, pad_w, 0, pad_h], value=1.0).bool()
            return _NestedTensor(tensors, mask)

        def forward(self, samples):
            samples = self._pad_to_block(samples)
            samples.no_padding = False  # rfdetr 1.10.0 backbone requires this attr
            result = self.base(samples=samples)
            if self._hs_last is not None:
                result['outputs'] = self._hs_last[-1]
            return result

    detr_model = _RFDETRWithOutputs(_base, block_size=_block_size)

    # --- Criterion: use RF-DETR's own IA-BCE criterion so stage-1 training is
    # identical to native RF-DETR training (same loss, same matcher, same LR schedule).
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors as _build_rfdetr_crit
    import torch as _torch

    # Matcher cost coefficients (same values as the D-DETR args, kept in sync via config).
    rf_args.set_cost_class = detr_args.set_cost_class
    rf_args.set_cost_bbox = detr_args.set_cost_bbox
    rf_args.set_cost_giou = detr_args.set_cost_giou
    rf_args.segmentation_head = False
    rf_args.sum_group_losses = config.get("RFDETR_SUM_GROUP_LOSSES", False)

    _rfdetr_crit, _ = _build_rfdetr_crit(rf_args)
    _rfdetr_crit.to(_torch.device(config["DEVICE"]))
    # NaN guard on the criterion's own matcher (the raw matcher is kept for match_last_layer, which guards it itself).
    _raw_matcher = install_criterion_match_guard(_rfdetr_crit)

    class _MotipCriterionWrapper:
        """Wraps RF-DETR SetCriterion to match MOTIP's (loss_dict, indices) interface.

        Stage 1 (ONLY_DETR=True): detr_indices are not consumed (prepare_for_motip
        is skipped), so returning dummy indices is fine.

        Stage 2 (ONLY_DETR=False): we re-run the matcher on just the first
        num_queries (300) queries so prepare_for_motip gets clean 1-to-1
        per-image assignments, independent of the group_detr=13 replication used
        for the loss.  The extra matcher call is cheap relative to the forward pass.
        """
        def __init__(self, c, num_queries, raw_matcher):
            self._c = c
            self._num_queries = num_queries
            self._raw_matcher = raw_matcher
            self.weight_dict = c.weight_dict

        def __call__(self, outputs, targets, **kwargs):
            losses = self._c(outputs, targets)
            return losses, self.match_last_layer(outputs, targets)

        def match_last_layer(self, outputs, targets, **kwargs):
            # Re-run matcher on first 300 queries only to get 1-to-1 assignments
            # that prepare_for_motip expects (group_detr=13 gives 13*N matches).
            # Also used alone by DETR_FREEZE training, where no DETR losses are computed.
            single = {
                "pred_logits": outputs["pred_logits"][:, :self._num_queries, :],
                "pred_boxes":  outputs["pred_boxes"][:, :self._num_queries, :],
            }
            return guarded_match(self._raw_matcher, single, targets)

        def train(self, mode=True):
            self._c.train(mode)
            return self

        def eval(self):
            self._c.eval()
            return self

        def to(self, *args, **kwargs):
            self._c.to(*args, **kwargs)
            return self

        def parameters(self):
            return self._c.parameters()

    detr_criterion = _MotipCriterionWrapper(_rfdetr_crit, num_queries=rf_args.num_queries, raw_matcher=_raw_matcher)

    return detr_model, detr_criterion


def build(config: dict):
    # Generate DETR args (used by the D-DETR criterion regardless of framework):
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]

    detr_framework = config["DETR_FRAMEWORK"].lower()
    match detr_framework:
        case "deformable_detr":
            from models.deformable_detr.deformable_detr import build as build_deformable_detr
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case "rf_detr":
            detr, detr_criterion = _build_rf_detr(config, detr_args)
        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build each component:
    # 1. trajectory modeling (currently, only FFNs are used):
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
    ) if config["ONLY_DETR"] is False else None
    # 2. ID decoder:
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
        # Look-ahead ID decoder (models/motip/lookahead.py), off by default.
        lookahead=dict(
            max_steps=int(config.get("LOOKAHEAD_MAX_STEPS", 8)),
            layers=config.get("LOOKAHEAD_LAYERS", None),
            geo_hidden=int(config.get("LOOKAHEAD_GEO_HIDDEN", 32)),
        ) if config.get("LOOKAHEAD", False) else None,
        # Rink-distance attention bias (models/motip/rink_dist_bias.py), off by default.
        rink_dist_bias=dict(
            **({"knots_m": tuple(config["RINK_DIST_BIAS_KNOTS_M"])} if config.get("RINK_DIST_BIAS_KNOTS_M") else {}),
            self_attn=bool(config.get("RINK_DIST_BIAS_SELF_ATTN", True)),
        ) if config.get("RINK_DIST_BIAS", False) else None,
    ) if config["ONLY_DETR"] is False else None
    if config.get("RINK_DIST_BIAS", False):
        # Works with or without RINK_ANCHOR: the bias reads only the per-token feet XY (training: prepare_for_motip with the
        # rink sidecar, gated by data.rink.needs_rink_data; inference: RuntimeTracker keeps the XY history whenever the
        # model has the bias, from the per-frame homographies passed to update*(rink_H=)).
        if config.get("LOOKAHEAD", False):
            raise NotImplementedError("LOOKAHEAD does not carry rink XY for future tokens yet (RINK_DIST_BIAS).")
    if config.get("LOOKAHEAD", False) and (config.get("RINK_ANCHOR", False) or config.get("TEAM_CUE", False)):
        raise NotImplementedError("LOOKAHEAD does not carry rink XY / kit descriptors for future tokens yet.")

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
        spatial_anchor=config.get("SPATIAL_ANCHOR", False),
        spatial_anchor_scale_init=config.get("SPATIAL_ANCHOR_SCALE_INIT", None),
        detr_hidden_dim=config["DETR_HIDDEN_DIM"],
        rink_anchor=config.get("RINK_ANCHOR", False),
        rink_anchor_kwargs=dict(
            dropout=config.get("RINK_ANCHOR_DROPOUT", 0.0),
            n_freq=config.get("RINK_ANCHOR_N_FREQ", 24),
            min_period_m=config.get("RINK_ANCHOR_MIN_PERIOD_M", 1.0),
            max_period_m=config.get("RINK_ANCHOR_MAX_PERIOD_M", 64.0),
            out_scale_init=config.get("RINK_ANCHOR_OUT_SCALE_INIT", None),
        ),
        team_cue=config.get("TEAM_CUE", False),
        team_cue_kwargs=dict(
            in_dim=config.get("TEAM_CUE_DIM", 14),
            hidden=config.get("TEAM_CUE_HIDDEN", 128),
            gate_init=config.get("TEAM_CUE_GATE_INIT", 0.25),
            dropout=config.get("TEAM_CUE_DROPOUT", 0.1),
            noise_std=config.get("TEAM_CUE_NOISE", 0.0),
        ),
        id_embed_head=dict(
            in_dim=config["DETR_HIDDEN_DIM"],
            hidden=int(config.get("ID_EMBED_HIDDEN", 256)),
            out_dim=int(config.get("ID_EMBED_DIM", 128)),
        ) if config.get("ID_EMBED_LOSS", False) else None,
    )
    # Store num_classes so RuntimeTracker can slice logits to the real class
    # channels (RF-DETR's build_model adds +1 for the background slot).
    motip_model.num_classes = config["NUM_CLASSES"]

    return motip_model, detr_criterion
