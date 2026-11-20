# Delayed (look-ahead) ID decisions for a MOTIP model built with LOOKAHEAD (models/motip/lookahead.py). Inference only,
# opt-in: cfg LOOKAHEAD_LAG (raw frames) > 0 in motip_server._fixed_lag(). Same interface as FixedLagTracker.

from collections import deque

import torch

from models.misc import get_model
from models.runtime_tracker import RuntimeTracker
from utils.box_ops import box_cxcywh_to_xyxy


def _iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(n, 4) x (m, 4) xyxy -> (n, m) IoU."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class LookaheadTracker:
    """Frame t is decided when frame t + ``lag`` has arrived: the ID decoder of the wrapped RuntimeTracker sees, next to
    the labelled history, the detections of frames t + s, t + 2s, ... <= t + lag (s = ``frame_stride``) as unlabelled
    future tokens. The decision is then committed exactly like an online one (the history stays causal: it holds only
    committed IDs of frames < t + 1). No hypotheses; one decoder call per frame (two in a triggered frame of
    ``trigger`` mode). ``finish()`` decides the buffered tail with whatever future is left.

    ``trigger`` = "always": every frame uses its future. "conf": first decode without future; use the future only when a
    detection's best probability is < ``trigger_p``, or a live track unseen for 1..``trigger_lost_max`` frames has its
    last box at IoU >= ``trigger_iou`` with a detection, or a detection's best ID is a track unseen >= ``trigger_react``
    frames (a reactivation). The decoder output without future is the L = 0 output of the same checkpoint."""

    def __init__(self, tracker: RuntimeTracker, lag: int, frame_stride: int = 4, trigger: str = "always",
                 trigger_p: float = 0.99, trigger_iou: float = 0.3, trigger_lost_max: int = 30, trigger_react: int = 5):
        if int(lag) < 1 or int(frame_stride) < 1:
            raise ValueError(f"need lag >= 1 and frame_stride >= 1, not {lag}, {frame_stride}")
        if trigger not in ("always", "conf"):
            raise ValueError(f"LOOKAHEAD_TRIGGER must be 'always' or 'conf', not {trigger!r}")
        la = getattr(get_model(tracker.model).id_decoder, "lookahead", None)
        if la is None:
            raise ValueError("LOOKAHEAD_LAG > 0 needs a model built with LOOKAHEAD: True")
        self._trk = tracker
        self.lag = int(lag)
        self.stride = int(frame_stride)
        self.steps = self.lag // self.stride
        self.trigger = trigger
        self.trigger_p = float(trigger_p)
        self.trigger_iou = float(trigger_iou)
        self.trigger_lost_max = int(trigger_lost_max)
        self.trigger_react = int(trigger_react)
        self.buf = deque()
        self.t = 0
        self.stats = {"frames": 0, "frames_with_future": 0, "future_tokens": 0, "triggered": 0,
                      "trig_conf": 0, "trig_lost": 0, "trig_react": 0, "steps": self.steps,
                      "model_max_steps": int(la.max_steps)}

    @property
    def tracker(self) -> RuntimeTracker:
        return self._trk

    @torch.no_grad()
    def update(self, image, suppress_boxes=None, _det_cache=None, rink_H=None, frame_bgr=None) -> list:
        trk = self._trk
        detr_out = trk.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = trk._get_activate_detections(detr_out=detr_out)
        kit = trk._kit_from_frame(frame_bgr, boxes) if (trk.team_cue and frame_bgr is not None) else None
        if _det_cache is not None:
            _det_cache.append((scores, categories, boxes, output_embeds) if kit is None else (scores, categories, boxes, output_embeds, kit))
        return self.update_from_detections(scores, categories, boxes, output_embeds, suppress_boxes, rink_H=rink_H, kit=kit)

    @torch.no_grad()
    def update_from_detections(self, scores, categories, boxes, output_embeds, suppress_boxes=None, rink_H=None,
                               kit=None) -> list:
        """Buffer this frame; return ``[(frame, track_results)]`` for the frames decided now (0-based call index)."""
        self.buf.append((self.t, (scores, categories, boxes, output_embeds, suppress_boxes), dict(rink_H=rink_H, kit=kit)))
        self.t += 1
        out = []
        while self.buf and self.buf[-1][0] - self.buf[0][0] >= self.lag:
            out.append(self._decide())
        return out

    def finish(self) -> list:
        out = []
        while self.buf:
            out.append(self._decide())
        return out

    # -- internals ----------------------------------------------------------------------------------------------------
    def _future_dets(self, t: int):
        """[(step j, boxes (n, 4) cxcywh, embeds (n, C))] of the buffered frames t + j*s, j = 1..steps."""
        out = []
        for tt, args, _ in self.buf:
            d = tt - t
            if d <= 0 or d % self.stride or d // self.stride > self.steps:
                continue
            _, _, boxes, emb, sup = args
            keep = self._suppress_keep(boxes, sup)
            out.append((d // self.stride, boxes[keep], emb[keep]))
        return out

    def _suppress_keep(self, boxes: torch.Tensor, suppress_boxes) -> torch.Tensor:
        """RuntimeTracker's goalkeeper suppression rule (IoU > 0.5 with a suppress box drops the detection)."""
        keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        if suppress_boxes is None or len(suppress_boxes) == 0 or len(boxes) == 0:
            return keep
        xyxy = box_cxcywh_to_xyxy(boxes) * self._trk.bbox_unnorm
        sup = torch.as_tensor(suppress_boxes, dtype=xyxy.dtype, device=xyxy.device).reshape(-1, 4)
        return ~(_iou_matrix(xyxy, sup) > 0.5).any(dim=1)

    def _future_tensors(self, fut, like: torch.Tensor, t_now: int) -> dict:
        K = self.steps
        nf = max([len(b) for _, b, _ in fut] + [1])
        C = like.shape[-1]
        dev, dt = like.device, like.dtype
        feat = torch.zeros((1, 1, 1, K, nf, C), dtype=dt, device=dev)
        boxes = torch.zeros((1, 1, 1, K, nf, 4), dtype=dt, device=dev)
        masks = torch.ones((1, 1, 1, K, nf), dtype=torch.bool, device=dev)
        for j, b, e in fut:
            n = len(b)
            feat[0, 0, 0, j - 1, :n] = e.to(device=dev, dtype=dt)
            boxes[0, 0, 0, j - 1, :n] = b.to(device=dev, dtype=dt)
            masks[0, 0, 0, j - 1, :n] = False
        steps = torch.arange(1, K + 1, device=dev)[None, None, None]
        times = t_now + self.stride * steps     # raw frames, the units of the history at inference
        return {"future_features": feat, "future_boxes": boxes, "future_masks": masks,
                "future_times": times.long(), "future_steps": steps.long()}

    def _triggered(self, trk: RuntimeTracker, seq_info: dict) -> bool:
        st = self.stats
        info = dict(seq_info)
        info = trk.model(seq_info=info, part="trajectory_modeling")
        logits, _, _ = trk.model(seq_info=info, part="id_decoder")
        p = logits[0, 0, 0].float().softmax(-1)               # (D, vocab + 1)
        if p.shape[0] == 0:
            return False
        best_p, best = p.max(dim=-1)
        hit = False
        if bool((best_p < self.trigger_p).any()):
            st["trig_conf"] += 1
            hit = True
        T = trk.trajectory_masks.shape[0]
        if T > 0:
            seen = ~trk.trajectory_masks                       # (T, N)
            any_seen = seen.any(dim=0)
            last = torch.where(seen, torch.arange(T, device=seen.device)[:, None], -1).max(dim=0).values
            unseen = (T - 1 - last)                            # frames since last seen (0 = seen in the previous frame)
            lost = any_seen & (unseen >= 1) & (unseen <= self.trigger_lost_max)
            if bool(lost.any()):
                idx = lost.nonzero()[:, 0]
                lb = trk.trajectory_boxes[last[idx], idx]       # (L, 4) cxcywh
                det = seq_info["unknown_boxes"][0, 0, 0]
                if len(det) and bool((_iou_matrix(box_cxcywh_to_xyxy(lb.float()), box_cxcywh_to_xyxy(det.float()))
                                      >= self.trigger_iou).any()):
                    st["trig_lost"] += 1
                    hit = True
            labels = trk.trajectory_id_labels[0]               # (N,) ID label of each column
            react = (unseen >= self.trigger_react) & any_seen
            if bool(react.any()):
                react_labels = set(labels[react].tolist())
                if any(int(b) in react_labels for b in best.tolist()):
                    st["trig_react"] += 1
                    hit = True
        return hit

    def _decide(self):
        t, args, kw = self.buf.popleft()
        trk = self._trk
        fut = self._future_dets(t)
        st = self.stats
        st["frames"] += 1

        def _hook(trk_, seq_info):
            if self.steps == 0 or not fut:
                return seq_info
            if self.trigger == "conf":
                if not self._triggered(trk_, seq_info):
                    return seq_info
                st["triggered"] += 1
            t_now = int(seq_info["unknown_times"].reshape(-1)[0].item()) if seq_info["unknown_times"].numel() \
                else trk_.trajectory_times.shape[0]
            ft = self._future_tensors(fut, seq_info["unknown_features"], t_now)
            st["frames_with_future"] += 1
            st["future_tokens"] += int((~ft["future_masks"]).sum().item())
            return {**seq_info, **ft}

        trk._future = _hook
        try:
            trk.update_from_detections(*args, **kw)
        finally:
            trk._future = None
        return t, trk.get_track_results()
