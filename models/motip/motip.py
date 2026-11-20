# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MOTIP(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_framework: str,
            only_detr: bool,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
            spatial_anchor: bool = False,
            detr_hidden_dim: int = 256,
            spatial_anchor_scale_init: float | None = None,
            rink_anchor: bool = False,
            rink_anchor_kwargs: dict | None = None,
            team_cue: bool = False,
            team_cue_kwargs: dict | None = None,
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder
        # MOTIP2 Spatial Anchor: the detector's box positional encoding is added to every ID-decoder token.
        self.spatial_anchor = spatial_anchor
        self.detr_hidden_dim = detr_hidden_dim
        if self.spatial_anchor:
            assert detr_framework == "rf_detr", "SPATIAL_ANCHOR reuses RF-DETR's ref_point_head, it needs DETR_FRAMEWORK: rf_detr."
            assert id_decoder is not None and id_decoder.feature_dim == detr_hidden_dim, \
                "SPATIAL_ANCHOR adds a detr_hidden_dim-wide encoding to the features, so FEATURE_DIM must equal DETR_HIDDEN_DIM."
        # Optional learnable per-channel scale on the anchor. Initialised at 0 it leaves a checkpoint trained
        # without the anchor exactly unchanged at step 0 and lets training bring the anchor in gradually.
        # None keeps the plain (unscaled) MOTIP2 anchor.
        self.anchor_scale = None
        if self.spatial_anchor and spatial_anchor_scale_init is not None:
            self.anchor_scale = nn.Parameter(torch.full((detr_hidden_dim,), float(spatial_anchor_scale_init)))

        # Rink anchor: each token's player position in rink metres (from the frame's homography), encoded and added
        # to the detection-feature half of every trajectory / unknown token (see models/motip/rink_anchor.py).
        self.rink_anchor = None
        if rink_anchor:
            from .rink_anchor import RinkAnchor
            assert id_decoder is not None and id_decoder.feature_dim == detr_hidden_dim, \
                "RINK_ANCHOR adds a detr_hidden_dim-wide encoding to the features, so FEATURE_DIM must equal DETR_HIDDEN_DIM."
            # Initialised from a private RNG state: building the anchor must not move the global RNG, or a run with the
            # anchor would draw different data order / augmentations than the same run without it (the DataLoader's
            # base seed comes from the global generator), and arms D / E would not see the data arms B / C see.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self.rink_anchor = RinkAnchor(dim=detr_hidden_dim, **(rink_anchor_kwargs or {}))

        # Team cue: each token's kit-appearance descriptor (utils/kit_feature.py) -> MLP -> gate, added to the detection-feature
        # half of every trajectory / unknown token (see models/motip/team_cue.py). Off by default: no module, no parameters.
        self.team_cue = None
        if team_cue:
            from .team_cue import TeamCue
            assert id_decoder is not None and id_decoder.feature_dim == detr_hidden_dim, \
                "TEAM_CUE adds a detr_hidden_dim-wide encoding to the features, so FEATURE_DIM must equal DETR_HIDDEN_DIM."
            with torch.random.fork_rng(devices=[]):   # as for the rink anchor: building it must not move the global RNG
                torch.manual_seed(1)
                self.team_cue = TeamCue(dim=detr_hidden_dim, **(team_cue_kwargs or {}))
        # Size of each added term relative to the detection features, recorded on every id_decoder forward while
        # ``record_cue_stats`` is on (training loop / check script); read and cleared with ``take_cue_stats``.
        self.record_cue_stats = False
        self._cue_stats: dict[str, list] = {}

        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000           # hack implementation

        return

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                if "use_checkpoint" in kwargs:
                    return checkpoint(
                        self.detr, frames,
                        use_reentrant=False,
                    )
                else:
                    return self.detr(samples=frames)
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                return self.trajectory_modeling(seq_info)
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                if self.spatial_anchor:
                    seq_info = self.add_box_anchor(seq_info)
                if self.rink_anchor is not None:
                    seq_info = self.add_rink_anchor(seq_info)
                if self.team_cue is not None:
                    seq_info = self.add_team_cue(seq_info)
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"MOTIP forwarding doesn't support part={kwargs['part']}.")

    def box_position_encoding(self, boxes: torch.Tensor) -> torch.Tensor:
        """RF-DETR's query positional encoding of boxes: ``ref_point_head(gen_sineembed_for_position(box))``.

        Args:
            boxes: (..., 4) normalised cxcywh, the detector's predicted boxes (used detached).
        Returns:
            (..., detr_hidden_dim)
        """
        from rfdetr.models.transformer import gen_sineembed_for_position
        head = self.detr.base.transformer.decoder.ref_point_head
        flat = boxes.detach().reshape(1, -1, 4).to(next(head.parameters()).dtype)
        pos = head(gen_sineembed_for_position(flat, self.detr_hidden_dim // 2))
        return pos.reshape(*boxes.shape[:-1], self.detr_hidden_dim)

    def add_box_anchor(self, seq_info: dict) -> dict:
        """MOTIP2 Spatial Anchor: add each token's box encoding to the detection-feature half of every
        trajectory and unknown token, before the ID fusion (the decoder concatenates feature and ID embedding).
        Returns a shallow copy, the caller's ``seq_info`` is not modified."""
        seq_info = dict(seq_info)
        # "future": LOOKAHEAD inference tokens (in training they are built inside the decoder from the anchored unknowns).
        for k in ("trajectory", "unknown") + (("future",) if "future_features" in seq_info else ()):
            features = seq_info[f"{k}_features"]
            pos = self.box_position_encoding(seq_info[f"{k}_boxes"])
            if self.anchor_scale is not None:
                pos = pos * self.anchor_scale.to(pos.dtype)
            if self.record_cue_stats:
                self._record("box", pos, features, seq_info.get(f"{k}_masks"))
            seq_info[f"{k}_features"] = features + pos.to(features.dtype)
        return seq_info

    def add_rink_anchor(self, seq_info: dict) -> dict:
        """Rink anchor: add ``rink_anchor(xy)`` of each token's feet position in rink metres to the detection-feature
        half of every trajectory and unknown token. ``seq_info['{trajectory,unknown}_rink_xy']`` is (..., 2) metres,
        NaN where the position is invalid; a missing key means every token is invalid (no homography available), so
        a caller without rink data still runs. Returns a shallow copy."""
        seq_info = dict(seq_info)
        for k in ("trajectory", "unknown"):
            features = seq_info[f"{k}_features"]
            xy = seq_info.get(f"{k}_rink_xy")
            if xy is None:
                xy = torch.full((*features.shape[:-1], 2), float("nan"), device=features.device)
            seq_info[f"{k}_features"] = features + self.rink_anchor(xy.to(features.device)).to(features.dtype)
        return seq_info

    def add_team_cue(self, seq_info: dict) -> dict:
        """Add ``team_cue(kit)`` of each token's kit descriptor to the detection-feature half of every trajectory and
        unknown token. ``seq_info['{trajectory,unknown}_kit']`` is (..., K), NaN where there is none; a missing key means
        every token has none (a caller without kit data still runs, tokens get the learned "missing" vector).
        Returns a shallow copy."""
        seq_info = dict(seq_info)
        for k in ("trajectory", "unknown"):
            features = seq_info[f"{k}_features"]
            kit = seq_info.get(f"{k}_kit")
            if kit is None:
                kit = torch.full((*features.shape[:-1], self.team_cue.in_dim), float("nan"), device=features.device)
            cue, used = self.team_cue(kit.to(features.device))
            if self.record_cue_stats:
                self._record("kit", cue, features, seq_info.get(f"{k}_masks"), used=used)
            seq_info[f"{k}_features"] = features + cue.to(features.dtype)
        return seq_info

    @torch.no_grad()
    def _record(self, name: str, added: torch.Tensor, features: torch.Tensor, masks, used: torch.Tensor | None = None) -> None:
        """Log RMS(added) / RMS(features) over the real (unmasked) tokens, and the share of tokens with a descriptor."""
        real = ~masks if masks is not None else torch.ones(features.shape[:-1], dtype=torch.bool, device=features.device)
        if not real.any():
            return
        a, f = added.float()[real], features.float()[real]
        st = self._cue_stats
        st.setdefault(f"{name}_added_rms", []).append(a.pow(2).mean().sqrt().item())
        st.setdefault(f"{name}_feat_rms", []).append(f.pow(2).mean().sqrt().item())
        st.setdefault(f"{name}_rms_ratio", []).append((a.pow(2).mean().sqrt() / f.pow(2).mean().sqrt().clamp(min=1e-8)).item())
        if used is not None:
            st.setdefault(f"{name}_used_frac", []).append(used[real].float().mean().item())

    def take_cue_stats(self) -> dict[str, float]:
        """Mean of what was recorded since the last call (plus the gate / scale values), and clear it."""
        out = {k: sum(v) / len(v) for k, v in self._cue_stats.items() if v}
        self._cue_stats = {}
        if self.team_cue is not None:
            out["kit_gate"] = float(self.team_cue.gate.detach())
            out["kit_missing_norm"] = float(self.team_cue.missing.detach().norm())
        if self.anchor_scale is not None:
            out["box_scale_mean"] = float(self.anchor_scale.detach().abs().mean())
            out["box_scale_max"] = float(self.anchor_scale.detach().abs().max())
        return out
