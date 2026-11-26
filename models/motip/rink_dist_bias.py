# IHC: relative rink-distance attention bias for the MOTIP ID decoder. For every (query, key) token pair the decoder's
# attention logits get a learned per-layer, per-head bias that is a piecewise-linear function of the distance in metres
# between the two tokens' feet on the rink (the per-frame-homography XY the rink anchor uses). The tables start at zero
# (step 0 == the decoder without the bias) and a pair where either token has no position (no homography, feet off the
# rink) gets exactly zero. See obsidian-vault/sessions/2026-11-21-joint-ablation-rink-idloss.md.

import einops
import torch
import torch.nn as nn

# Knots (metres) of the piecewise-linear distance function; distances beyond the last knot use the last value. Denser
# where crossings happen (0-3 m), coarser beyond.
DEFAULT_KNOTS_M = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0, 20.0, 30.0)


def pair_interp(xy_q: torch.Tensor, xy_k: torch.Tensor, knots: torch.Tensor):
    """Interpolation weights of the pairwise feet distance on ``knots``.

    Args:
        xy_q: (B, Lq, 2) metres, NaN = no position.  xy_k: (B, Lk, 2).  knots: (K,) increasing, knots[0] = 0.
    Returns:
        ``(lo, frac, valid, dist)``: lo (B, Lq, Lk) int64 lower knot index in [0, K-2], frac (B, Lq, Lk) float32 in [0, 1],
        valid (B, Lq, Lk) bool (both tokens have a position), dist (B, Lq, Lk) float32 metres (0 where invalid).
    """
    q = xy_q.detach().float()
    k = xy_k.detach().float().to(q.device)
    valid = torch.isfinite(q).all(-1)[:, :, None] & torch.isfinite(k).all(-1)[:, None, :]
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    dist = (q[:, :, None, :] - k[:, None, :, :]).norm(dim=-1)
    dist = torch.where(valid, dist, torch.zeros_like(dist))
    kn = knots.to(device=dist.device, dtype=torch.float32)
    d = dist.clamp(max=float(kn[-1]))
    lo = (torch.searchsorted(kn, d.contiguous(), right=True) - 1).clamp(min=0, max=kn.numel() - 2)
    frac = ((d - kn[lo]) / (kn[lo + 1] - kn[lo])).clamp(0.0, 1.0)
    return lo, frac, valid, dist


def bias_from_table(table: torch.Tensor, lo: torch.Tensor, frac: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(K, H) table + the interpolation of ``pair_interp`` -> (B * H, Lq, Lk) float32, in the head-minor layout of
    ``nn.MultiheadAttention``'s 3-D ``attn_mask`` (as the decoder's rel-PE bias)."""
    t = table.float().to(lo.device)
    b = t[lo] * (1.0 - frac)[..., None] + t[lo + 1] * frac[..., None]          # (B, Lq, Lk, H)
    b = b * valid[..., None].to(b.dtype)
    return einops.rearrange(b, "b l1 l2 h -> (b h) l1 l2")


class RinkDistBias(nn.Module):
    """Learned bias tables over pairwise rink distance: ``cross`` (num_layers, K, H) for the unknown -> trajectory
    cross-attention and, with ``self_attn``, ``self_`` (num_layers - 1, K, H) for the unknown <-> unknown (same frame)
    self-attention of layers 1... Zero-initialised."""

    def __init__(self, num_layers: int, n_heads: int, knots_m=DEFAULT_KNOTS_M, self_attn: bool = True):
        super().__init__()
        knots = torch.as_tensor([float(k) for k in knots_m], dtype=torch.float32)
        assert knots.numel() >= 2 and float(knots[0]) == 0.0 and bool((knots[1:] > knots[:-1]).all()), knots_m
        self.register_buffer("knots", knots, persistent=False)
        self.cross = nn.Parameter(torch.zeros(num_layers, knots.numel(), n_heads))
        self.self_ = nn.Parameter(torch.zeros(max(num_layers - 1, 0), knots.numel(), n_heads)) if self_attn else None

    def interp(self, xy_q, xy_k):
        return pair_interp(xy_q, xy_k, self.knots)

    def cross_bias(self, layer: int, interp) -> torch.Tensor:
        lo, frac, valid, _ = interp
        return bias_from_table(self.cross[layer], lo, frac, valid)

    def self_bias(self, layer: int, interp) -> torch.Tensor | None:
        """``layer`` is the decoder layer (>= 1); its self-attention uses table row ``layer - 1``."""
        if self.self_ is None or interp is None:
            return None
        lo, frac, valid, _ = interp
        return bias_from_table(self.self_[layer - 1], lo, frac, valid)
