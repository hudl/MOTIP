# IHC: rink anchor for the MOTIP ID decoder: each token's player position in rink metres (from the per-frame rink
# homography) is Fourier-encoded, passed through a small MLP and added to the token's detection-feature half, next to
# (and independent of) the image-space box anchor. See obsidian-vault/sessions/2026-11-18-rink-anchor.md.

import math

import torch
import torch.nn as nn


def project_feet(boxes: torch.Tensor, h_norm: torch.Tensor, flip: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Feet (bbox bottom-centre) of normalised cxcywh boxes in rink metres.

    Args:
        boxes: (..., 4) normalised cxcywh in the (augmented) image the boxes live in.
        h_norm: (..., 3, 3) float64, that image's normalised coords -> rink metres, NaN rows where the frame has no
            homography. Broadcastable against ``boxes[..., 0]`` (e.g. ``(B, 1, T, 1, 3, 3)`` for ``(B, G, T, N, 4)``).
        flip: (...) bool, broadcastable. The training pipeline's horizontal flip mirrors the image; the flipped image
            shows the same rink seen by a camera at the mirrored position, so the rink x of every player is negated
            (the rink is symmetric about x = 0). ``h_norm`` already undoes the flip in pixel space.
        bounds: (..., 2) (x, y) half extents + margin in metres inside which a point counts as on the rink.
    Returns:
        (..., 2) float32 metres, NaN where the position is invalid (no homography, point at infinity, off the rink).
    """
    b = boxes.detach().to(torch.float64)
    p = torch.stack([b[..., 0], b[..., 1] + b[..., 3] / 2.0, torch.ones_like(b[..., 0])], dim=-1)   # (..., 3)
    h_norm = h_norm.to(device=p.device, dtype=torch.float64)
    q = (h_norm @ p.unsqueeze(-1)).squeeze(-1)                                                       # (..., 3)
    w = q[..., 2]
    with torch.no_grad():
        xy = q[..., :2] / torch.where(w.abs() > 1e-9, w, torch.full_like(w, float("nan"))).unsqueeze(-1)
    bounds = bounds.to(device=p.device, dtype=torch.float64)
    ok = torch.isfinite(xy).all(dim=-1) & (xy[..., 0].abs() <= bounds[..., 0]) & (xy[..., 1].abs() <= bounds[..., 1])
    flip = flip.to(device=p.device).bool()
    x = torch.where(flip, -xy[..., 0], xy[..., 0])
    out = torch.stack([x, xy[..., 1]], dim=-1)
    out = torch.where(ok.unsqueeze(-1), out, torch.full_like(out, float("nan")))
    return out.to(torch.float32)


class RinkAnchor(nn.Module):
    """``(x, y)`` metres (NaN = no position) -> ``(dim,)`` added to a token's feature.

    Fourier features of the metric coordinates (``n_freq`` log-spaced periods from ``max_period_m`` down to
    ``min_period_m``, sin and cos, for x and y), a 2-layer MLP, and a learned "no position" vector for invalid
    tokens. The last MLP layer and the no-position vector start at zero, so a model with the anchor on is exactly
    the model without it at step 0. ``dropout`` replaces a valid token's position by "no position" in training.
    """

    def __init__(self, dim: int = 256, n_freq: int = 24, hidden: int = 256, min_period_m: float = 1.0,
                 max_period_m: float = 64.0, dropout: float = 0.0):
        super().__init__()
        assert 0.0 <= dropout <= 1.0, dropout
        self.dim, self.n_freq, self.dropout = dim, n_freq, float(dropout)
        self.min_period_m, self.max_period_m = float(min_period_m), float(max_period_m)
        self.mlp = nn.Sequential(nn.Linear(4 * n_freq, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.no_position = nn.Parameter(torch.zeros(dim))

    def encode(self, xy: torch.Tensor) -> torch.Tensor:
        """(..., 2) finite metres -> (..., 4 * n_freq)."""
        # Frequencies are rebuilt in fp32 on every call (a registered buffer would be rounded to fp16 by model.half()).
        periods = torch.logspace(math.log10(self.max_period_m), math.log10(self.min_period_m), self.n_freq,
                                 device=xy.device, dtype=torch.float32)
        ang = xy.float().unsqueeze(-1) * (2.0 * math.pi / periods)   # (..., 2, n_freq)
        return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)   # (..., 4 * n_freq)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(xy).all(dim=-1)
        if self.training and self.dropout > 0.0:
            valid = valid & (torch.rand(valid.shape, device=valid.device) >= self.dropout)
        feat = self.encode(torch.where(valid.unsqueeze(-1), xy, torch.zeros_like(xy)))
        pos = self.mlp(feat.to(self.mlp[0].weight.dtype))
        return torch.where(valid.unsqueeze(-1), pos, self.no_position.to(pos.dtype).expand_as(pos))
