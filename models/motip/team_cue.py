# IHC: team / kit appearance cue for the MOTIP ID decoder (TEAM_CUE).
#
# Each ID-decoder token (every trajectory token and every current detection) carries the kit descriptor of the detection it was
# built from (utils/kit_feature.py: 14 colour numbers of the torso and leg regions of its box). This module turns it into a
# vector added to the detection-feature half of the token, next to the box anchor (models/motip/motip.py).
#
#   kit (..., K) NaN = none -> (kit - mean) / std -> MLP(K, hidden, dim) -> LayerNorm (no affine) -> * gate
#   tokens without a descriptor (and, in training, a fraction ``dropout`` of the valid ones) get ``missing * gate`` instead.
#
# Why this shape: the previous position anchors were zero-initialised (per-channel scale / zero last layer) and trained at a 2e-5
# LR, and stayed at ~0.3-1% of the feature magnitude after 13 epochs: the decoder never used them. Here the added term starts at a
# fixed, meaningful size (gate = ``gate_init`` per element, the LayerNorm makes the vector unit-RMS per element before the gate;
# the detector features are ~0.9 RMS per element), a plain scalar gate carries the size, and the whole module gets its own LR group.
# ``missing`` starts at zero (no information adds nothing), the gate is a scalar so a "no use" solution is one number away.
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeamCue(nn.Module):
    def __init__(self, in_dim: int = 14, dim: int = 256, hidden: int = 128, gate_init: float = 0.25, dropout: float = 0.1,
                 noise_std: float = 0.0):
        super().__init__()
        assert 0.0 <= dropout <= 1.0, dropout
        self.in_dim, self.dim, self.dropout, self.noise_std = in_dim, dim, float(dropout), float(noise_std)
        # Standardisation of the descriptor (dataset statistics, scripts/compute_kit_stats.py); buffers, so they travel with
        # the checkpoint and inference needs no stats file.
        self.register_buffer("kit_mean", torch.zeros(in_dim))
        self.register_buffer("kit_std", torch.ones(in_dim))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.missing = nn.Parameter(torch.zeros(dim))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def set_stats(self, mean, std) -> None:
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        assert mean.shape == std.shape == (self.in_dim,), (mean.shape, std.shape)
        self.kit_mean.copy_(mean)
        self.kit_std.copy_(std.clamp(min=1e-3))

    def forward(self, kit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """kit (..., in_dim), NaN rows = no descriptor -> (cue (..., dim), used (...) bool: the cue is from a descriptor)."""
        valid = torch.isfinite(kit).all(dim=-1)
        if self.training and self.dropout > 0.0:
            valid = valid & (torch.rand(valid.shape, device=valid.device) >= self.dropout)
        x = torch.where(valid.unsqueeze(-1), kit, self.kit_mean.to(kit.dtype))
        x = (x - self.kit_mean) / self.kit_std
        if self.training and self.noise_std > 0.0:
            x = x + self.noise_std * torch.randn_like(x)
        w = self.mlp[0].weight
        e = self.mlp(x.to(w.dtype))
        e = F.layer_norm(e, e.shape[-1:])
        e = torch.where(valid.unsqueeze(-1), e, self.missing.to(e.dtype).expand_as(e))
        return e * self.gate.to(e.dtype), valid
