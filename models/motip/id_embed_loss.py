# IHC: hard-negative identity loss on the RF-DETR output embeddings the MOTIP ID decoder consumes (cfg ID_EMBED_LOSS,
# default off: no head, no parameters, no loss). A small projection head maps each GT-matched query's embedding to a
# unit vector; a supervised-contrastive (InfoNCE) term over every GT-matched query of a training window pulls the same
# GT identity in other frames together and pushes the other identities apart, with the negatives that overlap or nearly
# overlap the anchor (box IoU > hard_iou, or feet closer than hard_rink_m metres on the rink) within hard_frames sampled
# steps up-weighted by hard_weight. The loss reaches the detector through the frames that run with gradients
# (DETR_NUM_TRAIN_FRAMES); the head keeps the raw embedding free of a metric-space constraint.
# Optional, off: an auxiliary GT-team term on the same projected embeddings (same team positive, other team negative;
# training-only labels from a per-sequence gt/gt_team.txt sidecar, nothing at inference).
# See obsidian-vault/sessions/2026-11-21-joint-ablation-rink-idloss.md.

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDEmbedHead(nn.Module):
    """Linear -> GELU -> Linear -> L2 normalise."""

    def __init__(self, in_dim: int = 256, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.to(self.net[0].weight.dtype)).float(), dim=-1)


def _box_iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(M, 4) x (M', 4) normalised cxcywh -> (M, M') IoU."""
    def xyxy(x):
        return torch.cat([x[:, :2] - x[:, 2:] / 2, x[:, :2] + x[:, 2:] / 2], dim=-1)
    a, b = xyxy(a.float()), xyxy(b.float())
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-8)


def hard_negative_mask(ids, frames, boxes, rink_xy=None, hard_iou: float = 0.1, hard_rink_m: float = 1.5,
                       hard_frames: int = 1) -> torch.Tensor:
    """(M, M) bool: pairs of different identities whose GT boxes overlap (IoU > hard_iou) or whose feet are closer than
    hard_rink_m metres (when ``rink_xy`` is given and both are finite), at sampled steps at most ``hard_frames`` apart."""
    diff_id = ids[:, None] != ids[None, :]
    near_t = (frames[:, None] - frames[None, :]).abs() <= hard_frames
    near = _box_iou_cxcywh(boxes, boxes) > hard_iou
    if rink_xy is not None and hard_rink_m > 0:
        xy = rink_xy.float()
        ok = torch.isfinite(xy).all(-1)
        d = (torch.nan_to_num(xy)[:, None] - torch.nan_to_num(xy)[None]).norm(dim=-1)
        near = near | (ok[:, None] & ok[None, :] & (d < hard_rink_m))
    return diff_id & near_t & near


def supcon_weighted(z: torch.Tensor, pos: torch.Tensor, neg_weight: torch.Tensor, temp: float):
    """Supervised contrastive loss with per-pair negative weights.

    z (M, D) unit vectors; pos (M, M) bool positives (diagonal ignored); neg_weight (M, M) >= 0 weight of each pair in
    the denominator when it is a negative (positives count 1). Loss of anchor i = -mean_{p in P(i)} log(e^{s_ip} /
    (sum_{p'} e^{s_ip'} + sum_n w_in e^{s_in})); averaged over anchors with at least one positive.
    Returns (loss, n_anchors). A window with no positive pair gives (z.sum() * 0, 0) so the graph stays connected."""
    M = z.shape[0]
    eye = torch.eye(M, dtype=torch.bool, device=z.device)
    pos = pos & ~eye
    has = pos.any(1)
    if M < 2 or not has.any():
        return z.sum() * 0.0, 0
    s = (z @ z.t()) / temp
    w = torch.where(pos, torch.ones_like(neg_weight), neg_weight).masked_fill(eye, 0.0)
    log_w = torch.where(w > 0, w.clamp(min=1e-12).log(), torch.full_like(w, float("-inf")))
    log_den = torch.logsumexp(s + log_w, dim=1)                                # (M,)
    lp = (s - log_den[:, None]).masked_fill(~pos, 0.0)
    per = -(lp.sum(1) / pos.sum(1).clamp(min=1))
    return per[has].mean(), int(has.sum())


def window_id_embed_loss(z, ids, frames, boxes, rink_xy=None, temp: float = 0.1, hard_weight: float = 4.0,
                         hard_iou: float = 0.1, hard_rink_m: float = 1.5, hard_frames: int = 1, teams=None,
                         team_temp: float = 0.2):
    """Loss terms and stats of one training window. ``z`` (M, D) projected unit vectors of the GT-matched queries,
    ``ids`` (M,) GT identity, ``frames`` (M,) sampled-step index, ``boxes`` (M, 4) GT cxcywh, ``rink_xy`` (M, 2) metres
    or None, ``teams`` (M,) int (-1 unknown) or None (no team term)."""
    hard = hard_negative_mask(ids, frames, boxes, rink_xy, hard_iou, hard_rink_m, hard_frames)
    pos = (ids[:, None] == ids[None, :])
    neg_w = torch.ones(hard.shape, dtype=z.dtype, device=z.device) + (float(hard_weight) - 1.0) * hard.to(z.dtype)
    loss, n_anchor = supcon_weighted(z, pos, neg_w, temp)
    with torch.no_grad():
        s = z @ z.t()
        eye = torch.eye(len(ids), dtype=torch.bool, device=z.device)
        pos_ = pos & ~eye
        stats = {
            "n_anchor": n_anchor, "n_tokens": int(len(ids)), "n_hard": int(hard.sum()) // 2,
            "pos_sim": float(s[pos_].mean()) if pos_.any() else float("nan"),
            "neg_sim": float(s[~pos].mean()) if (~pos).any() else float("nan"),
            "hard_sim": float(s[hard].mean()) if hard.any() else float("nan"),
        }
    out = {"id_embed": loss}
    if teams is not None:
        known = teams >= 0
        tpos = known[:, None] & known[None, :] & (teams[:, None] == teams[None, :])
        tneg = known[:, None] & known[None, :] & (teams[:, None] != teams[None, :])
        # an anchor counts only when its window has both teams (otherwise there is no negative to separate from)
        tpos = tpos & tneg.any(1, keepdim=True)
        team_loss, n_t = supcon_weighted(z, tpos, tneg.to(z.dtype), team_temp)
        out["id_embed_team"] = team_loss
        stats["n_team_anchor"] = n_t
    return out, stats


def gather_matched(detr_outputs, detr_indices, annotations, rink=None, metas=None):
    """Per window b: the matched queries' (pre-head) embeddings (M_b, C), GT ids, sampled-step index, GT boxes, feet xy
    (when ``rink`` from train.rink_batch_from_metas is given; projected from the GT box) and GT team (from
    ``metas[b][t]['gt_team']`` {id: team} when present, else None)."""
    _B, _T = len(annotations), len(annotations[0])
    embeds = detr_outputs["outputs"]
    dev = embeds.device
    out = []
    for b in range(_B):
        e, i, f, bx, tm = [], [], [], [], []
        any_team = metas is not None and any("gt_team" in m for m in metas[b])
        for t in range(_T):
            flat = b * _T + t
            q_idx, gt_idx = detr_indices[flat]
            if len(q_idx) == 0:
                continue
            q_idx, gt_idx = q_idx.to(dev), gt_idx.to(dev)
            e.append(embeds[flat][q_idx])
            ids_t = annotations[b][t]["id"].to(dev)[gt_idx]
            i.append(ids_t)
            f.append(torch.full((len(q_idx),), t, dtype=torch.int64, device=dev))
            bx.append(annotations[b][t]["bbox"].to(dev)[gt_idx])
            if any_team:
                tmap = metas[b][t].get("gt_team", {})
                tm.append(torch.tensor([int(tmap.get(int(x), -1)) for x in ids_t.tolist()], dtype=torch.int64, device=dev))
        if not e:
            out.append(None)
            continue
        rec = {"embeds": torch.cat(e), "ids": torch.cat(i), "frames": torch.cat(f), "boxes": torch.cat(bx),
               "rink_xy": None, "teams": torch.cat(tm) if any_team else None}
        if rink is not None:
            from models.motip.rink_anchor import project_feet
            H = rink["H"][b].to(dev)[rec["frames"]]
            rec["rink_xy"] = project_feet(rec["boxes"], H, rink["flip"][b].to(dev)[rec["frames"]],
                                          rink["bounds"][b].to(dev)[rec["frames"]])
        out.append(rec)
    return out


class IDEmbedCriterion:
    """Config-driven wrapper: ``__call__(head, detr_outputs, detr_indices, annotations, rink, metas)`` -> (dict of
    weighted loss terms, dict of stats)."""

    def __init__(self, config: dict):
        self.weight = float(config.get("ID_EMBED_LOSS_WEIGHT", 0.1))
        self.temp = float(config.get("ID_EMBED_TEMP", 0.1))
        self.hard_weight = float(config.get("ID_EMBED_HARD_WEIGHT", 4.0))
        self.hard_iou = float(config.get("ID_EMBED_HARD_IOU", 0.1))
        self.hard_rink_m = float(config.get("ID_EMBED_HARD_RINK_M", 1.5))
        self.hard_frames = int(config.get("ID_EMBED_HARD_FRAMES", 1))
        self.team = bool(config.get("ID_EMBED_TEAM", False))
        self.team_weight = float(config.get("ID_EMBED_TEAM_WEIGHT", 0.05))
        self.team_temp = float(config.get("ID_EMBED_TEAM_TEMP", 0.2))

    def gather(self, detr_outputs, detr_indices, annotations, rink=None, metas=None):
        """-> (records per window, (M_total, C) concatenation of their pre-head embeddings for one head call)."""
        recs = gather_matched(detr_outputs, detr_indices, annotations, rink=rink, metas=metas if self.team else None)
        embeds = [r["embeds"] for r in recs if r is not None]
        flat = torch.cat(embeds) if embeds else detr_outputs["outputs"][:1, :0].reshape(0, detr_outputs["outputs"].shape[-1])
        return recs, flat

    def __call__(self, head, detr_outputs, detr_indices, annotations, rink=None, metas=None):
        recs, flat = self.gather(detr_outputs, detr_indices, annotations, rink=rink, metas=metas)
        return self.loss(recs, head(flat), detr_outputs["outputs"].device)

    def loss(self, recs, z_flat, device):
        """Loss terms from ``gather``'s records and the head's output on its concatenation (same order)."""
        losses, team_losses, stats = [], [], []
        off = 0
        for rec in recs:
            if rec is None:
                continue
            z = z_flat[off: off + len(rec["embeds"])]
            off += len(rec["embeds"])
            if self.team and rec["teams"] is None:
                raise RuntimeError("ID_EMBED_TEAM needs GT team labels: a per-sequence gt/gt_team.txt and Hockey(load_team=True).")
            terms, st = window_id_embed_loss(
                z, rec["ids"], rec["frames"], rec["boxes"], rec["rink_xy"], temp=self.temp, hard_weight=self.hard_weight,
                hard_iou=self.hard_iou, hard_rink_m=self.hard_rink_m, hard_frames=self.hard_frames,
                teams=rec["teams"] if self.team else None, team_temp=self.team_temp)
            losses.append(terms["id_embed"])
            if "id_embed_team" in terms:
                team_losses.append(terms["id_embed_team"])
            stats.append(st)
        dev = device
        # no window with a match: a zero that still touches the head's (empty) output, so the graph stays connected
        raw = torch.stack(losses).mean() if losses else z_flat.sum() * 0.0
        out = {"id_embed": raw}
        total = self.weight * raw
        if self.team:
            raw_t = torch.stack(team_losses).mean() if team_losses else torch.zeros((), device=dev)
            out["id_embed_team"] = raw_t
            total = total + self.team_weight * raw_t
        out["total"] = total
        agg = {}
        for k in ("pos_sim", "neg_sim", "hard_sim"):
            v = [s[k] for s in stats if s[k] == s[k]]
            agg[f"id_embed_{k}"] = sum(v) / len(v) if v else float("nan")
        agg["id_embed_n_hard"] = float(sum(s["n_hard"] for s in stats))
        agg["id_embed_n_anchor"] = float(sum(s["n_anchor"] for s in stats))
        return out, agg
