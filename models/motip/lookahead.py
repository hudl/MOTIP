# Look-ahead ID decoder (cfg LOOKAHEAD, default off). Design: obsidian-vault/sessions/2026-11-20-fixed-lag-decisions.md
# ("Design note: a trained look-ahead ID decoder") and 2026-11-20-lookahead-decoder.md.
#
# The causal MOTIP ID decoder labels the detections of frame t from the labelled history (times < t). Here it can also
# use the detections of the next K frames ("future tokens", no ID labels: they are not decided yet):
#
#   1. Future tokens are decoded exactly like frame-t queries with the same layers and weights (same-frame self-attention,
#      cross-attention over the labelled history, FFN), with the history cut at the query frame t (never any ID at a
#      time >= t) and their own (later) time in the relative positional encoding. So each future detection carries the
#      decoder's view of "which past identity is this", taken at a frame where the players may have separated.
#   2. After the cross-attention of the layers in LOOKAHEAD_LAYERS, a look-ahead block attends jointly over
#      [frame-t queries ; future tokens] of the same query frame, with a per-head bias from the SIGNED step offset between
#      tokens (key step - query step, in [-K, K]; the extension of the relative PE to negative offsets) and from the box
#      geometry of the pair (IoU, centre offset / height, log height ratio, |step offset|). This is where "the detection at
#      t + 5 is where this query goes" is learned. Pre-norm, residual, zero-initialised output projection / biases: a
#      fresh block is the identity, so a checkpoint trained without it decodes identically at step 0 (at any lag).
#   3. The output head is unchanged and reads the frame-t queries only.
#
# Lag units. K is in DECODER STEPS. Training: one step = one sampled frame of the window (motip_crossing_dataset is stride 4,
# so a step is 4 raw frames there), the lag K' is sampled per sample uniformly in 0..LOOKAHEAD_MAX_STEPS. Inference
# (LookaheadTracker): future frames are taken every LOOKAHEAD_FRAME_STRIDE raw frames (t + s, t + 2s, ..., <= t + lag),
# so the future tokens are spaced as in training. The future tokens' time against the history (relative PE of the
# pretrained cross-attention) uses the same units as the history does (training: step index; inference: raw frames),
# which is how MOTIP already treats its history.

import einops
import torch
import torch.nn as nn


def pair_features(boxes: torch.Tensor, steps: torch.Tensor, max_steps: int) -> torch.Tensor:
    """(M, S, 4) cxcywh boxes, (M, S) int steps -> (M, S, S, 5) pair features [query i, key j]:
    IoU, (dx, dy) of the centres / query height (clamped, / 4), log height ratio (clamped), |step offset| / max_steps."""
    b = boxes.float()
    q, k = b[:, :, None, :], b[:, None, :, :]
    h = q[..., 3].clamp(min=1e-4)
    dx = ((k[..., 0] - q[..., 0]) / h).clamp(-8.0, 8.0) / 4.0
    dy = ((k[..., 1] - q[..., 1]) / h).clamp(-8.0, 8.0) / 4.0
    lh = torch.log(k[..., 3].clamp(min=1e-4) / h).clamp(-3.0, 3.0)
    qx1, qy1, qx2, qy2 = q[..., 0] - q[..., 2] / 2, q[..., 1] - q[..., 3] / 2, q[..., 0] + q[..., 2] / 2, q[..., 1] + q[..., 3] / 2
    kx1, ky1, kx2, ky2 = k[..., 0] - k[..., 2] / 2, k[..., 1] - k[..., 3] / 2, k[..., 0] + k[..., 2] / 2, k[..., 1] + k[..., 3] / 2
    iw = (torch.minimum(qx2, kx2) - torch.maximum(qx1, kx1)).clamp(min=0)
    ih = (torch.minimum(qy2, ky2) - torch.maximum(qy1, ky1)).clamp(min=0)
    inter = iw * ih
    union = (q[..., 2] * q[..., 3]).clamp(min=0) + (k[..., 2] * k[..., 3]).clamp(min=0) - inter
    iou = inter / union.clamp(min=1e-8)
    dt = (steps[:, None, :] - steps[:, :, None]).abs().float() / float(max(max_steps, 1))
    return torch.stack([iou, dx, dy, lh, dt], dim=-1)


class LookaheadBlock(nn.Module):
    """Joint attention over one query frame's [frame-t queries ; future tokens]. Identity at initialisation."""

    def __init__(self, dim: int, n_heads: int, max_steps: int, geo_hidden: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.max_steps = int(max_steps)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=0.0, batch_first=True,
                                          add_zero_attn=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        # Signed step offset (key - query) in [-K, K] -> per-head bias; zero-init (no prior).
        self.time_bias = nn.Parameter(torch.zeros(2 * max(self.max_steps, 1) + 1, n_heads))
        self.geo = nn.Sequential(nn.Linear(5, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, n_heads))
        nn.init.zeros_(self.geo[2].weight)
        nn.init.zeros_(self.geo[2].bias)

    def forward(self, x: torch.Tensor, steps: torch.Tensor, pair: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """x (M, S, C); steps (M, S) int; pair (M, S, S, 5) from pair_features; pad (M, S) bool, True = no token."""
        K = max(self.max_steps, 1)
        rel = (steps[:, None, :] - steps[:, :, None]).clamp(-K, K) + K                     # (M, S, S)
        bias = self.time_bias[rel] + self.geo(pair.to(x.dtype))                              # (M, S, S, H)
        bias = einops.rearrange(bias, "m q k h -> (m h) q k").to(x.dtype)
        kpm = torch.zeros(pad.shape, dtype=x.dtype, device=x.device).masked_fill(pad, float("-inf"))
        y = self.norm(x)
        out, _ = self.attn(query=y, key=y, value=y, key_padding_mask=kpm, attn_mask=bias, need_weights=False)
        return x + out


class Lookahead(nn.Module):
    """The look-ahead parameters of an IDDecoder: one LookaheadBlock per layer in ``layers`` and the learned "future /
    unlabelled" offset added to the empty-ID embedding of future tokens (zero-init)."""

    def __init__(self, dim: int, id_dim: int, n_heads: int, num_layers: int, max_steps: int, layers=None,
                 geo_hidden: int = 32):
        super().__init__()
        self.max_steps = int(max_steps)
        if max_steps < 0:
            raise ValueError(f"LOOKAHEAD_MAX_STEPS must be >= 0, not {max_steps}")
        layers = list(range(num_layers)) if layers is None else [int(l) for l in layers]
        if not layers or any(l < 0 or l >= num_layers for l in layers):
            raise ValueError(f"LOOKAHEAD_LAYERS must be a non-empty subset of 0..{num_layers - 1}, not {layers}")
        self.layers = sorted(set(layers))
        self.blocks = nn.ModuleDict({str(l): LookaheadBlock(dim, n_heads, max_steps, geo_hidden) for l in self.layers})
        self.future_id_delta = nn.Parameter(torch.zeros(id_dim))


def future_from_unknown(seq_info: dict, steps: torch.Tensor) -> dict:
    """Training: the future tokens of every query frame t are the unknown tokens (detections) of frames t+1..t+K of the
    same window, without their labels. ``steps`` (B,) = each sample's lag K' (decoder steps); K = max(steps) capped at
    T - 1; tokens beyond a sample's lag or past the window end are padding.

    Returns ``future_{features,boxes}`` (B, G, T, K, N, ·), ``future_masks`` (B, G, T, K, N) (True = padding),
    ``future_times`` (B, G, T, K) (time used against the history: the future frame's own time) and ``future_steps``
    (B, G, T, K) = 1..K."""
    f, bx, m, tm = (seq_info["unknown_features"], seq_info["unknown_boxes"], seq_info["unknown_masks"],
                    seq_info["unknown_times"])
    B, G, T, N, C = f.shape
    K = int(min(int(steps.max().item()) if steps.numel() else 0, T - 1))
    dev = f.device
    t = torch.arange(T, device=dev)[:, None]                     # (T, 1)
    j = torch.arange(1, K + 1, device=dev)[None, :]              # (1, K)
    src = (t + j).clamp(max=T - 1)                               # (T, K)
    beyond = (t + j) > (T - 1)                                   # (T, K)
    feat = f[:, :, src]                                          # (B, G, T, K, N, C)
    boxes = bx[:, :, src]
    masks = m[:, :, src].clone()
    over_lag = j > steps.to(dev)[:, None]                        # (B, K)
    pad = beyond[None, None, :, :, None] | over_lag[:, None, None, :, None]   # (B, 1, T, K, 1)
    masks = masks | pad
    times = tm[:, :, src, 0] if N > 0 else torch.zeros((B, G, T, K), dtype=torch.int64, device=dev)
    fsteps = j.expand(T, K)[None, None].expand(B, G, T, K)
    return {"future_features": feat, "future_boxes": boxes, "future_masks": masks, "future_times": times,
            "future_steps": fsteps}


def lookahead_forward(dec, seq_info: dict, use_decoder_checkpoint: bool):
    """IDDecoder.forward with look-ahead (``dec.lookahead`` set). Same return as IDDecoder.forward.

    Future tokens come from ``seq_info["future_*"]`` (inference, see LookaheadTracker; all five keys of
    ``future_from_unknown``), else, in training, from the window's own later frames with a lag of
    ``seq_info["lookahead_steps"]`` (B,) or a lag drawn uniformly in 0..LOOKAHEAD_MAX_STEPS per sample. With no future
    tokens (lag 0) the blocks still run over the frame-t queries alone, so the model is the same at every lag."""
    from torch.utils.checkpoint import checkpoint
    la = dec.lookahead
    traj_f = seq_info["trajectory_features"]
    unk_f = seq_info["unknown_features"]
    traj_lab = seq_info["trajectory_id_labels"]
    unk_lab = seq_info.get("unknown_id_labels")
    traj_t, unk_t = seq_info["trajectory_times"], seq_info["unknown_times"]
    traj_m, unk_m = seq_info["trajectory_masks"], seq_info["unknown_masks"]
    B, G, T, N, _ = traj_f.shape
    _, _, Tq, Nq, _ = unk_f.shape
    dev = traj_f.device

    if "future_features" in seq_info:
        fut = {k: seq_info[k] for k in ("future_features", "future_boxes", "future_masks", "future_times", "future_steps")}
    else:
        steps = seq_info.get("lookahead_steps")
        if steps is None:
            steps = torch.randint(0, la.max_steps + 1, (B,)) if (dec.training and la.max_steps > 0) \
                else torch.zeros((B,), dtype=torch.int64)
        steps = torch.as_tensor(steps, dtype=torch.int64).reshape(B).clamp(0, la.max_steps)
        fut = future_from_unknown(seq_info, steps)
    if fut["future_features"].shape[3] == 0 or fut["future_features"].shape[4] == 0:
        # No future (lag 0 / end of a sequence): one all-padding step, which is exactly equivalent (padding is never
        # attended to) and keeps a single code path.
        z = torch.zeros((B, G, Tq, 1), dtype=torch.int64, device=dev)
        fut = {"future_features": unk_f.new_zeros((B, G, Tq, 1, 1, unk_f.shape[-1])),
               "future_boxes": unk_f.new_zeros((B, G, Tq, 1, 1, 4)),
               "future_masks": torch.ones((B, G, Tq, 1, 1), dtype=torch.bool, device=dev),
               "future_times": (unk_t[..., :1] if Nq > 0 else z).to(dev), "future_steps": z + 1}
    fut_f = fut["future_features"]                                # (B, G, Tq, K, Nf, C)
    K, Nf = fut_f.shape[3], fut_f.shape[4]
    fut_m = fut["future_masks"].bool()
    fut_steps = fut["future_steps"].to(dev).long()

    traj_e = torch.cat([traj_f, dec.id_label_to_embed(traj_lab)], dim=-1)
    unk_e = torch.cat([unk_f, dec.generate_empty_id_embed(unk_f)], dim=-1)
    fut_e = torch.cat([fut_f, dec.generate_empty_id_embed(fut_f) + la.future_id_delta.to(dec.dtype)], dim=-1)
    C2 = unk_e.shape[-1]
    H = dec.n_heads
    R = dec.rel_pe_length
    BG = B * G

    # --- cross-attention masks over the history, for [unknown rows ; future rows] per (b g) ---
    q_unk = Tq * Nq
    cut_unk = einops.rearrange(unk_t, "b g t n -> (b g) (t n)")
    cut_fut = einops.repeat(unk_t[..., 0] if Nq > 0 else torch.zeros((B, G, Tq), dtype=torch.int64, device=dev),
                            "b g t -> (b g) (t k n)", k=K, n=Nf)
    pe_fut = einops.repeat(fut["future_times"].to(dev), "b g t k -> (b g) (t k n)", n=Nf)
    q_cut = torch.cat([cut_unk, cut_fut], dim=1)                  # (BG, Q)
    q_pe = torch.cat([cut_unk, pe_fut], dim=1)
    k_t = einops.rearrange(traj_t, "b g t n -> (b g) (t n)")      # (BG, Tk)
    cross_mask = k_t[:, None, :] >= q_cut[:, :, None]             # (BG, Q, Tk)
    cross_mask = torch.zeros(cross_mask.shape, dtype=dec.dtype, device=dev).masked_fill(cross_mask, float("-inf"))
    cross_mask = einops.repeat(cross_mask, "bg q k -> (bg h) q k", h=H).contiguous()
    rel_idx = (q_pe[:, :, None] - k_t[:, None, :]).clamp(0, R - 1)  # (BG, Q, Tk)
    cross_kpm = einops.rearrange(traj_m, "b g t n -> (b g) (t n)")
    cross_kpm = torch.zeros(cross_kpm.shape, dtype=dec.dtype, device=dev).masked_fill(cross_kpm, float("-inf"))
    self_kpm_unk = einops.rearrange(unk_m, "b g t n -> (b g t) n").contiguous()
    self_kpm_fut = einops.rearrange(fut_m, "b g t k n -> (b g t k) n").contiguous()

    # --- look-ahead sets: per (b g t), [Nq frame-t queries ; K*Nf future tokens] ---
    la_steps = torch.cat([torch.zeros((B, G, Tq, Nq), dtype=torch.int64, device=dev),
                          einops.repeat(fut_steps, "b g t k -> b g t (k n)", n=Nf)], dim=-1)
    la_steps = einops.rearrange(la_steps, "b g t s -> (b g t) s")
    la_boxes = torch.cat([seq_info["unknown_boxes"].to(dev),
                          einops.rearrange(fut["future_boxes"].to(dev), "b g t k n c -> b g t (k n) c")], dim=-2)
    la_pair = pair_features(einops.rearrange(la_boxes, "b g t s c -> (b g t) s c"), la_steps, la.max_steps)
    la_pad = torch.cat([unk_m, einops.rearrange(fut_m, "b g t k n -> b g t (k n)")], dim=-1)
    la_pad = einops.rearrange(la_pad, "b g t s -> (b g t) s")

    def _layer(layer, unk_e, fut_e):
        if layer > 0:
            sa = dec.self_attn_layers[layer - 1]
            sn = dec.self_attn_norm_layers[layer - 1]
            x = einops.rearrange(unk_e, "b g t n c -> (b g t) n c").contiguous()
            o, _ = sa(query=x, key=x, value=x, key_padding_mask=self_kpm_unk)
            unk_e = einops.rearrange(sn(x + o), "(b g t) n c -> b g t n c", b=B, g=G, t=Tq)
            if K > 0 and Nf > 0:
                x = einops.rearrange(fut_e, "b g t k n c -> (b g t k) n c").contiguous()
                o, _ = sa(query=x, key=x, value=x, key_padding_mask=self_kpm_fut)
                fut_e = einops.rearrange(sn(x + o), "(b g t k) n c -> b g t k n c", b=B, g=G, t=Tq, k=K)
        q = torch.cat([einops.rearrange(unk_e, "b g t n c -> (b g) (t n) c"),
                       einops.rearrange(fut_e, "b g t k n c -> (b g) (t k n) c")], dim=1).contiguous()
        kv = einops.rearrange(traj_e, "b g t n c -> (b g) (t n) c").contiguous()
        pe = einops.rearrange(dec.rel_pos_embeds[layer][rel_idx], "bg q k h -> (bg h) q k")
        o, _ = dec.cross_attn_layers[layer](query=q, key=kv, value=kv, key_padding_mask=cross_kpm,
                                            attn_mask=cross_mask + pe)
        x = dec.cross_attn_norm_layers[layer](q + o)
        x = dec.ffn_norm_layers[layer](x + dec.ffn_layers[layer](x))
        unk_e = einops.rearrange(x[:, :q_unk], "(b g) (t n) c -> b g t n c", b=B, g=G, t=Tq)
        fut_e = einops.rearrange(x[:, q_unk:], "(b g) (t k n) c -> b g t k n c", b=B, g=G, t=Tq, k=K)
        if str(layer) in la.blocks:
            s = torch.cat([unk_e, einops.rearrange(fut_e, "b g t k n c -> b g t (k n) c")], dim=-2)
            s = la.blocks[str(layer)](einops.rearrange(s, "b g t s c -> (b g t) s c"), la_steps, la_pair, la_pad)
            s = einops.rearrange(s, "(b g t) s c -> b g t s c", b=B, g=G, t=Tq)
            unk_e = s[..., :Nq, :]
            fut_e = einops.rearrange(s[..., Nq:, :], "b g t (k n) c -> b g t k n c", k=K, n=Nf)
        return unk_e, fut_e

    all_logits = all_labels = all_masks = None
    lab = None
    for layer in range(dec.num_layers):
        if use_decoder_checkpoint:
            unk_e, fut_e = checkpoint(_layer, layer, unk_e, fut_e, use_reentrant=False)
        else:
            unk_e, fut_e = _layer(layer, unk_e, fut_e)
        logits = dec.embed_to_word_layers[layer](unk_e[..., -dec.id_dim:])
        masks = unk_m.clone()
        lab = None if not dec.training else unk_lab
        if all_logits is None:
            all_logits, all_labels, all_masks = logits, lab, masks
        else:
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_labels = torch.cat([all_labels, lab], dim=0) if lab is not None else None
            all_masks = torch.cat([all_masks, masks], dim=0)
    if dec.training and dec.use_aux_loss:
        return all_logits, all_labels, all_masks
    return logits, lab, masks
