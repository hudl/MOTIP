# Copyright (c) Ruopeng Gao. All Rights Reserved.

import sys
import os

from .motip import MOTIP
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder

# Prefer the bundled copy (available in both local dev and SageMaker code channel).
# Fall back to the ihc-od repo path for developers who don't have the copy.
_RFDETR_BUNDLED = os.path.join(os.path.dirname(__file__), "../../third_party")
_RFDETR_REPO = os.path.join(
    os.path.dirname(__file__),
    "../../../../third_party/aml-ice-hockey/ihc-od/third_party/rf-detr",
)


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
    # group_detr must be 1: train-time pred_logits otherwise has 3900 queries
    # and Hungarian would match across all groups rather than independently.
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
    rf_args.ia_bce_loss = False      # stage-1 loss; not used in MOTIP training
    rf_args.use_varifocal_loss = False
    rf_args.use_position_supervised_loss = False
    rf_args.sum_group_losses = False
    # --- Transformer args (defaults from RF-DETR main.py) ---
    rf_args.sa_nheads = 8
    rf_args.ca_nheads = 8
    rf_args.dropout = 0.0
    rf_args.dim_feedforward = 2048
    rf_args.dec_n_points = 4
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
            result = self.base(samples=samples)
            if self._hs_last is not None:
                result['outputs'] = self._hs_last[-1]
            return result

    detr_model = _RFDETRWithOutputs(_base, block_size=_block_size)

    # Use MOTIP's architecture-agnostic criterion, not RF-DETR's IA-BCE variant.
    # build_deformable_detr returns (model, criterion, postprocessors); we only
    # need the criterion.
    _, detr_criterion, _ = build_deformable_detr(args=detr_args)

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
    ) if config["ONLY_DETR"] is False else None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )
    # Store num_classes so RuntimeTracker can slice logits to the real class
    # channels (RF-DETR's build_model adds +1 for the background slot).
    motip_model.num_classes = config["NUM_CLASSES"]

    return motip_model, detr_criterion
