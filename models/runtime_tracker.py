# Copyright (c) Ruopeng Gao. All Rights Reserved.

import copy
import math

import numpy as np
import torch
import einops
from scipy.optimize import linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh, box_cxcywh_to_xyxy
from models.misc import get_model


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            max_tracks: int = 0,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            max_displacement_bh: float = 0.0,
            dtype: torch.dtype = torch.float32,
            # Spatial gate (off by default; see _apply_spatial_gate):
            spatial_gate: bool = False,
            gate_base: float = 0.5,
            gate_per_step: float = 0.2,
            # Rink-space gate (off by default; see _apply_spatial_gate). Needs per-frame image->rink
            # transforms from update(rink_H=...); frames / tracks without one are left to the image
            # gate (if on) or ungated.
            rink_gate: bool = False,
            rink_gate_base_m: float = 2.0,
            rink_gate_speed_mps: float = 10.0,
            rink_fps: float = 30.0,
            rink_bounds_m: tuple = (31.0, 16.0),
            # Rink anchor (off by default; needs a model built with RINK_ANCHOR): each detection's feet in rink
            # metres is kept with the trajectory history and fed to the ID decoder. Needs the same per-frame
            # transforms as the rink gate (update(rink_H=)); a frame without one gives invalid ("no position") tokens.
            rink_anchor: bool = False,
            # Team cue (off by default; needs a model built with TEAM_CUE): each detection's kit descriptor
            # (utils/kit_feature.py) is kept with the trajectory history and fed to the ID decoder. The caller passes it
            # per frame (update_from_detections(kit=) from a precomputed sidecar, or update(frame_bgr=) to compute it);
            # a frame without one gives NaN ("missing") tokens.
            team_cue: bool = False,
            # Quarantine (off by default; see _quarantine_labels): while a track's detection overlaps another
            # track's detection (box IoU >= quarantine_iou), or the last box of a live track that has no detection
            # this frame, its token for that frame is stored as "missing", so entangled frames do not enter the
            # identity history and the decoder keeps matching against the pre-crossing tokens (as after an
            # occlusion). At most quarantine_max consecutive frames per track; then tokens are kept again until the
            # track is no longer entangled. IDs are still assigned every frame (causal); QuarantineRewriter can
            # rewrite the entangled frames' IDs once the group separates.
            quarantine: bool = False,
            quarantine_iou: float = 0.3,
            quarantine_max: int = 30,
            # Tracklet confirmation (n_init = 1: off, current behaviour): a newborn track is tentative until it has been
            # matched in n_init consecutive frames (birth frame included). A tentative track that misses a frame is
            # dropped (trajectory removed, so it cannot take later detections) and its output ID is reported in the
            # frame's results as "dropped_ids"; ConfirmFilter deletes its earlier boxes from the output. A confirmed
            # track keeps its tentative frames under the same ID (backfill: nothing is lost for a real new player).
            n_init: int = 1,
            # New-track overlap rule (0 = off): a detection that would start a newborn track while its box has IoU >=
            # newborn_max_iou with a confirmed track's box in this frame (its matched detection, or its last-seen box
            # if it has none) is left unassigned for the frame (no output, no token). Not NMS: matched detections and
            # existing tracks are never touched.
            newborn_max_iou: float = 0.0,
            # ID-decoder top-k export (0 = off, current behaviour): per detection, the k best live IDs (output ids) and
            # their probabilities, in track results "id_topk" (N, k, 2) = (output id, prob); -1 / NaN pads. Analysis only.
            id_topk: int = 0,
    ):
        self.model = model
        self.id_topk = int(id_topk)
        self.model.eval()

        self.dtype = dtype

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.max_tracks = max_tracks
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.max_displacement_bh = max_displacement_bh
        if gate_base < 0 or gate_per_step < 0:
            raise ValueError(f"gate_base and gate_per_step must be >= 0, not {gate_base}, {gate_per_step}")
        self.spatial_gate = bool(spatial_gate)
        self.gate_base = float(gate_base)
        self.gate_per_step = float(gate_per_step)
        # Set by the gate on a frame where it removed every (detection, track) candidate (a likely
        # camera cut); makes that frame's detections newborns that skip the newborn score filter.
        self._gate_cut = False
        self.gate_stats = {"frames": 0, "pairs": 0, "gated": 0, "cut_frames": 0}
        if rink_gate_base_m < 0 or rink_gate_speed_mps < 0:
            raise ValueError(
                f"rink_gate_base_m and rink_gate_speed_mps must be >= 0, not {rink_gate_base_m}, {rink_gate_speed_mps}")
        if rink_gate and rink_fps <= 0:
            raise ValueError(f"rink_fps must be > 0, not {rink_fps}")
        self.rink_gate = bool(rink_gate)
        # ``rink_anchor`` here means "keep each token's feet XY and feed it to the ID decoder": needed by the rink anchor and
        # by the rink-distance attention bias (RINK_DIST_BIAS), so it is switched on by itself for a model with the bias.
        self.rink_anchor = bool(rink_anchor) or (
            getattr(getattr(get_model(model), "id_decoder", None), "rink_dist_bias", None) is not None)
        self.team_cue = bool(team_cue)
        self._kit_dim = int(get_model(model).team_cue.in_dim) if self.team_cue else 0
        self._kit_cur = None             # (D, K) float32 kit descriptors aligned with the boxes the ID decoder sees
        self.rink_gate_base_m = float(rink_gate_base_m)
        self.rink_gate_speed_mps = float(rink_gate_speed_mps)
        self.rink_fps = float(rink_fps)
        # (x, y) half extents in metres inside which a projected feet point counts as on the rink
        # (the playing surface plus the caller's margin).
        self.rink_bounds_m = (float(rink_bounds_m[0]), float(rink_bounds_m[1]))
        # This frame's image px -> rink metres matrix (3x3) or None, set by update*(); per-detection
        # feet positions of the whole input, and of the detections that survived goalkeeper suppression.
        self._rink_H = None
        self._rink_all = None            # (D_in, 3) float64: x, y, valid
        self._rink_cur = None            # (D, 3), aligned with the boxes the ID decoder sees
        # ID label -> (x, y, valid) of the track's feet in the frame it was last seen.
        self._track_rink = {}
        self.rink_stats = {"frames": 0, "frames_with_h": 0, "pairs": 0, "pairs_rink": 0, "gated_rink": 0}
        self.num_id_vocabulary = get_model(model).num_id_vocabulary
        self.num_classes = getattr(get_model(model), "num_classes", None)

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        # Rink anchor: feet (x, y) in rink metres of every trajectory token, NaN = no position. Always fp32.
        self.trajectory_rink = torch.zeros(
            (0, 0, 2), dtype=torch.float32, device=distributed_device(),
        )
        # Team cue: kit descriptor of every trajectory token, NaN = none. Always fp32.
        self.trajectory_kit = torch.zeros(
            (0, 0, max(self._kit_dim, 1)), dtype=torch.float32, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )
        # self.trajectory_features = torch.zeros(())

        self.current_track_results = {}
        self._frame_id_scores = None
        # Fixed-lag decisions (FixedLagTracker, off unless a caller wraps this tracker): a callback that sees every
        # frame's ID scores and assignment (it may snapshot the tracker to branch), and a one-shot forced
        # (id_scores, id_labels, gate_cut) that replaces the ID decoder + assignment of the next frame.
        self._lag_hook = None
        self._forced = None
        # Look-ahead decisions (LookaheadTracker, models/lookahead_tracker.py; off unless a caller wraps this tracker): a
        # one-shot callable (tracker, seq_info) -> seq_info that adds the next frames' detections as future tokens.
        self._future = None
        if quarantine and not (0.0 < quarantine_iou <= 1.0 and 1 <= quarantine_max < max(miss_tolerance - 2, 2)):
            raise ValueError(f"need 0 < quarantine_iou <= 1 and 1 <= quarantine_max < miss_tolerance - 2, "
                             f"not {quarantine_iou}, {quarantine_max} (miss_tolerance {miss_tolerance})")
        self.quarantine = bool(quarantine)
        self.quarantine_iou = float(quarantine_iou)
        self.quarantine_max = int(quarantine_max)
        self._q_count = {}        # ID label -> consecutive entangled frames
        self.q_stats = {"frames": 0, "tokens": 0, "masked": 0, "capped": 0}
        if int(n_init) < 1 or not 0.0 <= newborn_max_iou <= 1.0:
            raise ValueError(f"need n_init >= 1 and 0 <= newborn_max_iou <= 1, not {n_init}, {newborn_max_iou}")
        self.n_init = int(n_init)
        self.newborn_max_iou = float(newborn_max_iou)
        self._tentative = {}      # ID label -> consecutive matched frames since birth (only while < n_init)
        self.n_stats = {"born": 0, "confirmed": 0, "dropped": 0, "blocked_newborns": 0}
        return

    @torch.no_grad()
    def update(self, image, suppress_boxes=None, _det_cache=None, rink_H=None, frame_bgr=None):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = self._get_activate_detections(detr_out=detr_out)
        kit = self._kit_from_frame(frame_bgr, boxes) if (self.team_cue and frame_bgr is not None) else None
        if _det_cache is not None:
            _det_cache.append((scores, categories, boxes, output_embeds) if kit is None else (scores, categories, boxes, output_embeds, kit))
        self._process_detections(scores, categories, boxes, output_embeds, suppress_boxes, rink_H=rink_H, kit=kit)

    def _kit_from_frame(self, frame_bgr, boxes: torch.Tensor) -> torch.Tensor:
        """Kit descriptors (D, K) of this frame's detections (normalised cxcywh ``boxes``) from the original-resolution BGR frame."""
        from utils.kit_feature import kit_for_boxes
        xyxy = (box_cxcywh_to_xyxy(boxes.float()) * self.bbox_unnorm.float()).cpu().numpy()
        return torch.from_numpy(kit_for_boxes(frame_bgr, xyxy))

    @torch.no_grad()
    def update_from_detections(self, scores, categories, boxes, output_embeds, suppress_boxes=None, rink_H=None, kit=None):
        """Like update() but with pre-computed DETR outputs — skips the model forward pass.

        ``scores``, ``categories``, ``boxes``, ``output_embeds`` are the outputs of
        ``_get_activate_detections`` (already threshold-filtered), moved to the correct
        device by the caller.  Using this path avoids re-running the backbone when
        detections were already computed in a prior pass (e.g. a DETR pre-pass that
        also served segment planning).

        ``rink_H`` is this frame's 3x3 image px -> rink metres matrix, or None / NaN when the frame
        has none; only the rink gate (``rink_gate=True``) reads it.

        ``kit`` is this frame's (D, K) kit descriptors, one row per input detection (NaN row = none); only the team cue
        (``team_cue=True``) reads it.
        """
        self._process_detections(scores, categories, boxes, output_embeds, suppress_boxes, rink_H=rink_H, kit=kit)

    def _process_detections(self, scores, categories, boxes, output_embeds, suppress_boxes=None, rink_H=None, kit=None):
        """Apply goalie suppression, ID decoder, trajectory update from pre-extracted detections."""
        if self.rink_gate or self.rink_anchor:
            self._set_frame_rink(boxes, rink_H)
        if self.team_cue:
            kit = torch.full((boxes.shape[0], self._kit_dim), float("nan")) if kit is None else torch.as_tensor(kit).float().reshape(boxes.shape[0], self._kit_dim)
            kit = kit.to(distributed_device())
        # Index of each detection in this frame's input (the det_cache entry), kept through
        # every filter below so a track row can be joined back to its stored embedding.
        det_idx = torch.arange(boxes.shape[0], device=boxes.device)
        # Suppress detections matching goalkeeper boxes
        if suppress_boxes is not None and len(suppress_boxes) > 0 and len(boxes) > 0:
            boxes_xyxy = box_cxcywh_to_xyxy(boxes) * self.bbox_unnorm
            suppress_t = torch.tensor(suppress_boxes, dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
            keep = torch.ones(len(boxes_xyxy), dtype=torch.bool, device=boxes_xyxy.device)
            for i in range(len(boxes_xyxy)):
                for j in range(len(suppress_t)):
                    inter_x1 = torch.max(boxes_xyxy[i, 0], suppress_t[j, 0])
                    inter_y1 = torch.max(boxes_xyxy[i, 1], suppress_t[j, 1])
                    inter_x2 = torch.min(boxes_xyxy[i, 2], suppress_t[j, 2])
                    inter_y2 = torch.min(boxes_xyxy[i, 3], suppress_t[j, 3])
                    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
                    area_a = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
                    area_b = (suppress_t[j, 2] - suppress_t[j, 0]) * (suppress_t[j, 3] - suppress_t[j, 1])
                    iou_val = inter / (area_a + area_b - inter + 1e-6)
                    if iou_val > 0.5:
                        keep[i] = False
                        break
            scores = scores[keep]
            categories = categories[keep]
            boxes = boxes[keep]
            output_embeds = output_embeds[keep]
            det_idx = det_idx[keep]
            if self.team_cue:
                kit = kit[keep]
        if self.rink_gate or self.rink_anchor:
            self._rink_cur = self._rink_all[det_idx.cpu().numpy()]
        self._kit_cur = kit if self.team_cue else None
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)

        # Filter out illegal newborn detections:
        n_active = self.trajectory_id_labels.shape[1] if self.trajectory_id_labels.shape[0] > 0 else 0
        if self.max_tracks > 0 and n_active >= self.max_tracks:
            # Force-assign newborns to best existing track instead of dropping
            newborn_mask = (id_pred_labels == self.num_id_vocabulary)
            if newborn_mask.any() and hasattr(self, '_last_id_scores') and self._last_id_scores is not None:
                trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
                # IDs already claimed by non-newborn detections in this frame
                already_assigned = set()
                for i in range(len(id_pred_labels)):
                    if id_pred_labels[i].item() != self.num_id_vocabulary:
                        already_assigned.add(id_pred_labels[i].item())
                valid_ids = sorted(trajectory_id_labels_set - already_assigned)
                if valid_ids:
                    # Greedy: assign each newborn to best available ID, removing it from pool
                    newborn_idxs = newborn_mask.nonzero(as_tuple=False).squeeze(-1)
                    if newborn_idxs.dim() == 0:
                        newborn_idxs = newborn_idxs.unsqueeze(0)
                    for idx in newborn_idxs:
                        if not valid_ids:
                            break
                        id_scores_row = self._last_id_scores[idx]
                        best_score = -1.0
                        best_id = None
                        for vid in valid_ids:
                            if vid < id_scores_row.shape[0]:
                                s = id_scores_row[vid].item()
                                if s > best_score:
                                    best_score = s
                                    best_id = vid
                        if best_id is not None and best_score > self.id_thresh:
                            id_pred_labels[idx] = best_id
                            valid_ids.remove(best_id)
            # Now apply normal newborn filter (remaining unassigned newborns get dropped)
            keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        else:
            keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        if self.newborn_max_iou > 0 and not self._gate_cut:
            keep_idxs = keep_idxs & ~self._blocked_newborns(id_pred_labels, boxes)
        if self._gate_cut:
            # The gate rejected every candidate for every track: most likely a camera cut. Every
            # detection is a newborn here, so keep them all rather than filtering by newborn_thresh.
            keep_idxs = torch.ones_like(keep_idxs)
        # After the max-tracks block, which can move a newborn onto an existing ID
        id_stats = self._id_stats(id_pred_labels)
        id_topk = self._id_topk_stats(id_pred_labels.shape[0]) if self.id_topk > 0 else None
        if id_topk is not None:
            id_topk = id_topk[keep_idxs.cpu()]
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        det_idx = det_idx[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]
        id_stats = id_stats[keep_idxs]

        # A hack implementation, before assign new id labels, update the id_queue to ensure the uniqueness of id labels:
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep_idxs = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            det_idx = det_idx[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]
            id_stats = id_stats[keep_idxs]
            if id_topk is not None:
                id_topk = id_topk[keep_idxs.cpu()]
        pass

        # Assign new id labels (in place: id_pred_labels becomes id_labels, so remember the newborns first):
        born_mask = (id_pred_labels == self.num_id_vocabulary).clone()
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)

        if len(torch.unique(id_labels)) != len(id_labels):
            print(id_labels, id_labels.shape)
            exit(-1)

        masked = self._quarantine_labels(boxes, id_labels) if self.quarantine else set()
        dropped_ids = self._update_confirmation(id_labels, born_mask) if self.n_init > 1 else []

        # Update the results:
        self.current_track_results = {
            "score": scores,
            "category": categories,
            # "bbox": boxes * self.bbox_unnorm,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
            # Decoder output embeddings, one row per detection and in the same order as
            # "bbox". These are what the id_decoder consumes to judge identity, so they
            # are usable as appearance features by an external tracker.
            "embed": output_embeds,
            # (N, 3) per detection: ID-decoder prob of the label it was given (newborn token
            # included), top-1 prob over existing IDs, top-1 minus top-2 margin. NaN when no
            # trajectory existed to decode against (first frame / only_detr).
            "id_stats": id_stats,
            # (N,) index of each detection in this frame's det_cache entry (= the stored
            # embeds file), in the same order as "bbox".
            "det_idx": det_idx,
        }
        if id_topk is not None:
            self.current_track_results["id_topk"] = id_topk
        if self.n_init > 1:
            # Output IDs of tentative tracks dropped this frame (their earlier boxes are spurious), and per detection
            # whether its track is still tentative.
            self.current_track_results["dropped_ids"] = dropped_ids
            self.current_track_results["tentative"] = torch.tensor(
                [int(l) in self._tentative for l in id_labels.tolist()], dtype=torch.bool)
        if self.quarantine:
            # (N,) bool: this detection's token was quarantined (stored as missing) this frame.
            self.current_track_results["quarantined"] = torch.tensor(
                [int(l) in masked for l in id_labels.tolist()], dtype=torch.bool)

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Remember where each assigned track's feet were (for the rink gate on later frames):
        if self.rink_gate:
            feet = self._rink_all[det_idx.cpu().numpy()]
            for lab, row in zip(id_labels.tolist(), feet):
                self._track_rink[int(lab)] = (float(row[0]), float(row[1]), bool(row[2]))

        # Update trajectory infos:
        self._update_trajectory_infos(
            boxes=boxes, output_embeds=output_embeds, id_labels=id_labels,
            rink_xy=self._rink_xy(self._rink_all[det_idx.cpu().numpy()]) if self.rink_anchor else None,
            kit=self._kit_cur, masked_labels=masked,
        )

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks()
        pass
        return

    def get_track_results(self):
        return self.current_track_results

    def _blocked_newborns(self, id_pred_labels: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """(D,) bool: newborn detections whose box has IoU >= newborn_max_iou with a confirmed track's box this frame."""
        V = self.num_id_vocabulary
        nb = id_pred_labels == V
        out = torch.zeros_like(nb)
        if not bool(nb.any()) or self.trajectory_id_labels.shape[0] == 0:
            return out
        live = self.trajectory_id_labels[0].tolist()
        pred = id_pred_labels.tolist()
        ref = []
        tm = self.trajectory_masks
        rows = torch.arange(tm.shape[0], device=tm.device)[:, None].expand_as(tm)
        last_t = torch.where(~tm, rows, torch.full_like(rows, -1)).max(dim=0).values
        for k, l in enumerate(live):
            if l in self._tentative:
                continue
            if l in pred:
                ref.append(boxes[pred.index(l)].float())
            elif int(last_t[k]) >= 0:
                ref.append(self.trajectory_boxes[int(last_t[k]), k].float())
        if not ref:
            return out
        iou = _box_iou(box_cxcywh_to_xyxy(boxes.float()), box_cxcywh_to_xyxy(torch.stack(ref)))
        out = nb & (iou >= self.newborn_max_iou).any(dim=1)
        self.n_stats["blocked_newborns"] += int(out.sum())
        return out

    def _update_confirmation(self, id_labels: torch.Tensor, born_mask: torch.Tensor) -> list:
        """Advance tentative tracks by this frame's assignment; drop the ones that missed it. Returns their output IDs."""
        labels = [int(l) for l in id_labels.tolist()]
        born = {l for l, b in zip(labels, born_mask.tolist()) if b}
        here = set(labels)
        dropped = []
        for l in list(self._tentative):
            if l in born:
                continue                      # label word reused by a newborn this frame (old trajectory already gone)
            if l in here:
                self._tentative[l] += 1
                if self._tentative[l] >= self.n_init:
                    del self._tentative[l]
                    self.n_stats["confirmed"] += 1
            else:
                del self._tentative[l]
                dropped.append(l)
        for l in born:
            self._tentative[l] = 1
            self.n_stats["born"] += 1
            if self.n_init <= 1:
                del self._tentative[l]
        if dropped:
            self.n_stats["dropped"] += len(dropped)
            if self.trajectory_id_labels.shape[0] > 0:
                keep = ~torch.isin(self.trajectory_id_labels[0],
                                   torch.tensor(dropped, device=self.trajectory_id_labels.device))
                for name in self._LAG_TENSORS:
                    t = getattr(self, name)
                    if t.dim() >= 2 and t.shape[1] == keep.shape[0]:
                        setattr(self, name, t[:, keep])
            out_ids = []
            for l in dropped:
                self._track_rink.pop(l, None)
                if l in self.id_label_to_id:
                    out_ids.append(int(self.id_label_to_id[l]))
            return out_ids
        return []

    def _quarantine_labels(self, boxes: torch.Tensor, id_labels: torch.Tensor) -> set:
        """ID labels (of tracks that existed before this frame) whose token this frame is quarantined.

        Entangled: the label's detection has box IoU >= quarantine_iou with another detection of this frame, or with
        the last-seen box of a live track that has no detection this frame (two players in one box). Counts consecutive
        entangled frames per label; past quarantine_max the token is kept again until the label disentangles.
        """
        st = self.q_stats
        st["frames"] += 1
        labels = [int(l) for l in id_labels.tolist()]
        st["tokens"] += len(labels)
        existing = set(self.trajectory_id_labels[0].tolist()) if self.trajectory_id_labels.shape[0] > 0 else set()
        ent = set()
        if len(labels) > 0 and existing:
            xyxy = box_cxcywh_to_xyxy(boxes.float())
            iou = _box_iou(xyxy, xyxy)
            iou.fill_diagonal_(0)
            close = (iou >= self.quarantine_iou).any(dim=1).tolist()
            ent = {l for l, c in zip(labels, close) if c and l in existing}
            # live tracks without a detection this frame: their last-seen box
            missing = [k for k, l in enumerate(self.trajectory_id_labels[0].tolist()) if l not in set(labels)]
            if missing:
                tm = self.trajectory_masks
                rows = torch.arange(tm.shape[0], device=tm.device)[:, None].expand_as(tm)
                last_t = torch.where(~tm, rows, torch.full_like(rows, -1)).max(dim=0).values
                mk = torch.tensor(missing, device=tm.device)
                ok = last_t[mk] >= 0
                if bool(ok.any()):
                    last = self.trajectory_boxes[last_t[mk][ok], mk[ok]].float()
                    hit = (_box_iou(xyxy, box_cxcywh_to_xyxy(last)) >= self.quarantine_iou).any(dim=1).tolist()
                    ent |= {l for l, c in zip(labels, hit) if c and l in existing}
        masked = set()
        for l in labels:
            if l not in ent:
                self._q_count.pop(l, None)
                continue
            n = self._q_count.get(l, 0)
            if n < self.quarantine_max:
                masked.add(l)
                self._q_count[l] = n + 1
            else:
                st["capped"] += 1
        for l in list(self._q_count):
            if l not in labels:
                self._q_count.pop(l)
        st["masked"] += len(masked)
        return masked

    # Every per-hypothesis tensor of the tracker state. Cloned (not shared) by _lag_clone: _update_trajectory_infos
    # writes into a view of trajectory_id_labels in place.
    _LAG_TENSORS = ("trajectory_features", "trajectory_boxes", "trajectory_rink", "trajectory_kit",
                    "trajectory_id_labels", "trajectory_times", "trajectory_masks")

    @torch.no_grad()
    def _lag_ref_scores(self, boxes: torch.Tensor, output_embeds: torch.Tensor, gap: int, rink_cur=None, kit_cur=None):
        """ID-decoder probabilities (D, V+1) of this frame's detections against THIS tracker's (frozen) history as if
        ``gap`` frames had passed since it: the last ``T - gap`` history frames are kept and the query sits ``gap``
        frames after them, i.e. the tracks look unseen for ``gap`` frames (an occlusion, in distribution for the
        decoder). Ungated. None when there is no history left. Used by FixedLagTracker(score="ref")."""
        T = self.trajectory_features.shape[0]
        keep = T - int(gap)
        if T == 0 or keep < 1 or self.trajectory_id_labels.shape[1] == 0 or boxes.shape[0] == 0:
            return None
        dev = distributed_device()
        n = self.trajectory_id_labels.shape[1]
        sl = slice(T - keep, T)
        seq_info = {
            "trajectory_features": self.trajectory_features[sl][None, None, ...],
            "trajectory_boxes": self.trajectory_boxes[sl][None, None, ...],
            "trajectory_id_labels": self.trajectory_id_labels[sl][None, None, ...],
            "trajectory_times": torch.arange(keep, dtype=torch.int64, device=dev)[:, None].expand(keep, n)[None, None, ...],
            "trajectory_masks": self.trajectory_masks[sl][None, None, ...],
            "unknown_features": output_embeds[None, None, None, ...],
            "unknown_boxes": boxes[None, None, None, ...],
            "unknown_masks": torch.zeros((1, 1, 1, output_embeds.shape[0]), dtype=torch.bool, device=dev),
            "unknown_times": T * torch.ones((1, 1, 1, output_embeds.shape[0]), dtype=torch.int64, device=dev),
        }
        if self.rink_anchor:
            seq_info["trajectory_rink_xy"] = self.trajectory_rink[sl][None, None, ...]
            seq_info["unknown_rink_xy"] = self._rink_xy(rink_cur)[None, None, None, ...]
        if self.team_cue:
            seq_info["trajectory_kit"] = self.trajectory_kit[sl][None, None, ...]
            seq_info["unknown_kit"] = kit_cur[None, None, None, ...]
        seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
        id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
        id_logits = id_logits[0, 0, 0]
        return id_logits.softmax(dim=-1) if not self.use_sigmoid else id_logits.sigmoid()

    def _lag_clone(self) -> "RuntimeTracker":
        """Independent copy of the tracking state that shares the model (for FixedLagTracker hypotheses).

        Taken from inside ``_get_id_pred_labels`` (the lag hook), i.e. before this frame touched any persistent state,
        so the copy can re-run the same frame with a forced assignment.
        """
        new = copy.copy(self)
        for name in self._LAG_TENSORS:
            setattr(new, name, getattr(self, name).clone())
        new.id_queue = copy.deepcopy(self.id_queue)
        new.id_label_to_id = dict(self.id_label_to_id)
        new._track_rink = dict(self._track_rink)
        new._q_count = dict(self._q_count)
        new._tentative = dict(self._tentative)
        new.n_stats = dict(self.n_stats)
        new.q_stats = dict(self.q_stats)
        new.gate_stats = dict(self.gate_stats)
        new.rink_stats = dict(self.rink_stats)
        new.current_track_results = {}
        new._lag_hook = None
        new._forced = None
        return new

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        # Slice to NUM_CLASSES channels before max: RF-DETR adds an extra
        # background logit (num_classes+1 channels at build_model time).
        nc = self.num_classes
        scores = logits[..., :nc].sigmoid() if nc is not None else logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        # Selecting:
        # logits = logits[activate_indices]
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    def _id_topk_stats(self, n: int) -> torch.Tensor:
        """(n, id_topk, 2) per detection: the id_topk most probable live IDs as OUTPUT ids (id_label_to_id, taken
        before this frame's newborn labels are assigned) and their probabilities; -1 / NaN when fewer are live or
        nothing was decoded (same cases as _id_stats)."""
        k = self.id_topk
        out = torch.full((n, k, 2), float("nan"), dtype=torch.float32)
        out[:, :, 0] = -1
        s = self._frame_id_scores
        live_ids = self.trajectory_id_labels[0].unique() if self.trajectory_id_labels.shape[0] > 0 else None
        if s is None or n == 0 or s.shape[0] != n or live_ids is None or live_ids.numel() == 0:
            return out
        live = s.float()[:, live_ids]
        kk = min(k, live.shape[1])
        vals, idx = live.topk(kk, dim=-1)
        labels = live_ids[idx].cpu().tolist()
        out[:, :kk, 1] = vals.cpu()
        out[:, :kk, 0] = torch.tensor([[float(self.id_label_to_id.get(int(l), -1)) for l in row] for row in labels],
                                      dtype=torch.float32).reshape(n, kk)
        return out

    def _id_stats(self, labels: torch.Tensor) -> torch.Tensor:
        """Return (N, 3) per detection: P(label it was given), top-1 P over live IDs, margin.

        The label column includes the newborn token. Top-1 and the top-1 minus top-2 margin
        are taken only over the IDs of live trajectories, since the decoder also gives a little
        probability to unused IDs; with one live trajectory the margin is its probability.
        All NaN when there was nothing live to decode against: the first frame, only_detr, or
        a history whose trajectories have all gone inactive (columns dropped, rows kept).
        """
        n = labels.shape[0]
        out = torch.full((n, 3), float("nan"), dtype=torch.float32, device=labels.device)
        s = self._frame_id_scores
        live_ids = self.trajectory_id_labels[0].unique() if self.trajectory_id_labels.shape[0] > 0 else None
        if s is None or n == 0 or s.shape[0] != n or live_ids is None or live_ids.numel() == 0:
            return out
        s = s.float()
        out[:, 0] = s[torch.arange(n, device=s.device), labels.clamp(max=s.shape[1] - 1)]
        live = s[:, live_ids]
        top2 = live.topk(min(2, live.shape[1]), dim=-1).values
        out[:, 1] = top2[:, 0]
        out[:, 2] = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
        return out

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        self._frame_id_scores = None
        self._gate_cut = False
        if self._forced is not None:
            # FixedLagTracker alternative hypothesis: this frame's (gated) scores were computed on the parent hypothesis,
            # whose history up to the previous frame is identical; only the assignment differs.
            id_scores, id_labels, self._gate_cut = self._forced
            self._forced = None
            self._last_id_scores = id_scores
            self._frame_id_scores = id_scores
            return torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            if self.rink_anchor:
                seq_info["trajectory_rink_xy"] = self.trajectory_rink[None, None, ...]
                seq_info["unknown_rink_xy"] = self._rink_xy(self._rink_cur)[None, None, None, ...]
            if self.team_cue:
                seq_info["trajectory_kit"] = self.trajectory_kit[None, None, ...]
                seq_info["unknown_kit"] = self._kit_cur[None, None, None, ...]
            if self._future is not None:
                seq_info = self._future(self, seq_info)
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 4b. optional spatial gate: a box cannot take an ID whose track was last seen too far away.
            if self.spatial_gate or self.rink_gate:
                id_scores = self._apply_spatial_gate(id_scores=id_scores, boxes=boxes)
            # 5. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores, boxes=boxes)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            if self._lag_hook is not None:
                self._lag_hook(self, id_scores, list(id_labels), boxes, output_embeds)
            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            self._last_id_scores = id_scores
            self._frame_id_scores = id_scores
            return id_pred_labels

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            if self.rink_anchor:
                self.trajectory_rink = self.trajectory_rink[:, ~trajectory_remove_idxs]
            if self.team_cue:
                self.trajectory_kit = self.trajectory_kit[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _rink_xy(self, rows) -> torch.Tensor:
        """(D, 3) rows of (x, y, valid) from ``_set_frame_rink`` -> (D, 2) float32 metres, NaN where invalid."""
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
        xy = np.where(rows[:, 2:3] > 0.5, rows[:, :2], np.nan)
        return torch.as_tensor(xy, dtype=torch.float32, device=distributed_device())

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor,
                                 rink_xy: torch.Tensor | None = None, kit: torch.Tensor | None = None,
                                 masked_labels: set | None = None):
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        if self.rink_anchor:
            self.trajectory_rink = self.trajectory_rink[-self.miss_tolerance + 2:, ...]
        if self.team_cue:
            self.trajectory_kit = self.trajectory_kit[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            newborn_id_labels = torch.tensor(list(newborn_id_labels), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels)
            _id_labels = einops.repeat(newborn_id_labels, 'n -> t n', t=_T)
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _features = torch.zeros(
                (_T, _N, 256), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            if self.rink_anchor:
                self.trajectory_rink = torch.cat([
                    self.trajectory_rink,
                    torch.full((_T, _N, 2), float("nan"), dtype=torch.float32, device=distributed_device()),
                ], dim=1)
            if self.team_cue:
                self.trajectory_kit = torch.cat([
                    self.trajectory_kit,
                    torch.full((_T, _N, self._kit_dim), float("nan"), dtype=torch.float32, device=distributed_device()),
                ], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, 256), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        current_rink = torch.full((_N, 2), float("nan"), dtype=torch.float32, device=distributed_device())
        current_kit = torch.full((_N, max(self._kit_dim, 1)), float("nan"), dtype=torch.float32, device=distributed_device())
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        current_features[current_idxs] = output_embeds[idxs]
        current_boxes[current_idxs] = boxes[idxs]
        if rink_xy is not None:
            current_rink[current_idxs] = rink_xy[idxs]
        if kit is not None:
            current_kit[current_idxs] = kit[idxs].float()
        current_masks[current_idxs] = False
        if masked_labels:
            # Quarantined tokens: stored exactly like a frame in which the track had no detection.
            q = torch.tensor([int(l) in masked_labels for l in current_id_labels.tolist()], dtype=torch.bool,
                             device=current_masks.device)
            current_masks[q] = True
            current_features[q] = 0
            current_boxes[q] = 0
            current_rink[q] = float("nan")
            current_kit[q] = float("nan")
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        if self.rink_anchor:
            self.trajectory_rink = torch.cat([self.trajectory_rink, current_rink[None, ...]], dim=0).contiguous()
        if self.team_cue:
            self.trajectory_kit = torch.cat([self.trajectory_kit, current_kit[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        # 4.4. a hack implementation to fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        if self.rink_anchor:
            self.trajectory_rink = self.trajectory_rink[:, is_active]
        if self.team_cue:
            self.trajectory_kit = self.trajectory_kit[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        return

    def _set_frame_rink(self, boxes: torch.Tensor, rink_H) -> None:
        """Project this frame's input detections' feet (bbox bottom-centre) to rink metres.

        ``self._rink_all`` is (D, 3): x, y, valid. A detection is valid when the frame has a finite
        transform and its feet land inside ``rink_bounds_m``.
        """
        n = int(boxes.shape[0])
        out = np.full((n, 3), np.nan)
        out[:, 2] = 0.0
        H = None
        if rink_H is not None:
            H = np.asarray(rink_H, dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(H)):
                H = None
        self._rink_H = H
        self.rink_stats["frames"] += 1
        if H is not None and n > 0:
            self.rink_stats["frames_with_h"] += 1
            b = boxes.detach().float().cpu().numpy().astype(np.float64)
            scale = self.bbox_unnorm.float().cpu().numpy().astype(np.float64)
            feet = np.stack([b[:, 0] * scale[0], (b[:, 1] + b[:, 3] / 2.0) * scale[1], np.ones(n)], axis=1)
            p = feet @ H.T
            with np.errstate(divide="ignore", invalid="ignore"):
                xy = p[:, :2] / p[:, 2:3]
            ok = (np.abs(p[:, 2]) > 1e-9) & np.isfinite(xy).all(axis=1)
            ok &= (np.abs(xy[:, 0]) <= self.rink_bounds_m[0]) & (np.abs(xy[:, 1]) <= self.rink_bounds_m[1])
            out[ok, :2] = xy[ok]
            out[:, 2] = ok.astype(np.float64)
        self._rink_all = out
        self._rink_cur = out

    def _apply_spatial_gate(self, id_scores: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Zero ``id_scores[det, id]`` where detection ``det`` is too far from track ``id``'s last box.

        Image gate (``spatial_gate``): distances are between box centres in pixels, in units of the
        track's last-seen box height (scale-aware: a far-side player moves fewer pixels). The allowed
        radius is ``gate_base + gate_per_step * frames_missed`` heights, with ``frames_missed`` = 0
        when the track was seen in the previous frame.

        Rink gate (``rink_gate``): distances are between feet positions (bbox bottom-centre) in rink
        metres, which do not move with the camera. The allowed radius is
        ``rink_gate_base_m + rink_gate_speed_mps * dt`` with ``dt = (frames_missed + 1) / rink_fps``
        seconds. It applies to a (detection, track) pair only when both feet positions are valid (the
        frame has a transform and both points are on the rink); other pairs use the image gate if it
        is on and are ungated otherwise.

        A zeroed score falls below ``id_thresh`` in the assignment, so the detection becomes a newborn
        (or takes another track) as usual. If every (detection, track) pair is gated (at least two
        tracks), ``self._gate_cut`` is set: a likely camera cut, where the detections become newborns
        instead of being dropped.
        """
        t, n = self.trajectory_masks.shape
        if t == 0 or n == 0 or boxes.shape[0] == 0:
            return id_scores
        seen = ~self.trajectory_masks                                   # (T, N)
        rows = torch.arange(t, device=seen.device)[:, None].expand(t, n)
        last_t = torch.where(seen, rows, torch.full_like(rows, -1)).max(dim=0).values   # (N,)
        # Inactive tracks are filtered every update, so this should not happen; leave them ungated.
        unseen = last_t < 0
        last_t = last_t.clamp(min=0)
        frames_missed = (t - 1 - last_t).clamp(min=0).float()
        labels = self.trajectory_id_labels[0]

        too_far = torch.zeros((boxes.shape[0], n), dtype=torch.bool, device=id_scores.device)
        if self.spatial_gate:
            last_boxes = self.trajectory_boxes[last_t, torch.arange(n, device=seen.device)]  # (N, 4) cxcywh
            scale = self.bbox_unnorm.float()
            trk_c = last_boxes[:, :2].float() * scale[:2]
            trk_h = last_boxes[:, 3].float() * scale[3]
            det_c = boxes[:, :2].float() * scale[:2]
            dist = torch.cdist(det_c, trk_c)                                 # (D, N) pixels
            limit = (self.gate_base + self.gate_per_step * frames_missed)[None, :] * trk_h[None, :]
            too_far = (dist > limit) & ~unseen[None, :]
        if self.rink_gate:
            det = torch.as_tensor(self._rink_cur, dtype=torch.float64, device=id_scores.device)   # (D, 3)
            trk = np.array([self._track_rink.get(int(l), (np.nan, np.nan, False)) for l in labels.tolist()],
                           dtype=np.float64).reshape(n, 3)
            trk = torch.as_tensor(trk, device=id_scores.device)
            both = (det[:, 2:3] > 0.5) & (trk[None, :, 2] > 0.5) & ~unseen[None, :]          # (D, N)
            dm = torch.cdist(torch.nan_to_num(det[:, :2]), torch.nan_to_num(trk[:, :2]))     # (D, N) metres
            dt = (frames_missed.double() + 1.0) / self.rink_fps
            limit_m = self.rink_gate_base_m + self.rink_gate_speed_mps * dt                   # (N,)
            far_rink = dm > limit_m[None, :]
            too_far = torch.where(both, far_rink, too_far)
            rs = self.rink_stats
            rs["pairs"] += int(both.numel())
            rs["pairs_rink"] += int(both.sum())
            rs["gated_rink"] += int((far_rink & both).sum())
        gated = id_scores.clone()
        gated[:, labels] = torch.where(too_far, torch.zeros_like(gated[:, labels]), gated[:, labels])
        st = self.gate_stats
        st["frames"] += 1
        st["pairs"] += int(too_far.numel())
        st["gated"] += int(too_far.sum())
        if n >= 2 and bool(too_far.all()):
            self._gate_cut = True
            st["cut_frames"] += 1
        return gated

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        if len(id_scores) > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, len(id_scores) - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        for _ in range(len(match_rows)):
            _id = match_cols[_]
            if _id not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            elif _id >= self.num_id_vocabulary:
                id_labels.append(self.num_id_vocabulary)
            elif id_scores[match_rows[_], _id] < self.id_thresh:
                id_labels.append(self.num_id_vocabulary)
            else:
                id_labels.append(_id)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor, boxes: torch.Tensor = None):
        id_labels = list()  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs

        # Spatial tiebreaker: when multiple detections want the same label with
        # near-equal confidence, prefer the one closest to the track's last position.
        if boxes is not None and self.trajectory_boxes.shape[0] > 0:
            last_traj_boxes = self.trajectory_boxes[-1]  # (N_traj, 4) cxcywh normalized
            last_traj_labels = self.trajectory_id_labels[-1]  # (N_traj,)
            last_traj_masks = self.trajectory_masks[-1]  # (N_traj,) True=missing

            # Group detections by their argmax label
            label_to_dets = dict()
            for i in range(len(object_max_id_labels)):
                lbl = object_max_id_labels[i].item()
                if lbl == self.num_id_vocabulary or lbl not in trajectory_id_labels_set:
                    continue
                if lbl not in label_to_dets:
                    label_to_dets[lbl] = []
                label_to_dets[lbl].append((i, object_max_confs[i].item()))

            # For labels claimed by multiple detections with near-equal conf, pick closest
            TIEBREAK_MARGIN = 0.05
            for lbl, dets in label_to_dets.items():
                if len(dets) < 2:
                    continue
                dets.sort(key=lambda x: -x[1])  # sort by conf descending
                top_conf = dets[0][1]
                # Check if runner-up is within margin
                if dets[1][1] >= top_conf - TIEBREAK_MARGIN:
                    # Find track's last known position
                    match_mask = (last_traj_labels == lbl)
                    if not match_mask.any():
                        continue
                    traj_idx = match_mask.nonzero(as_tuple=False)[0].item()
                    if last_traj_masks[traj_idx]:
                        continue  # track was missing last frame, can't use position
                    traj_box = last_traj_boxes[traj_idx]
                    traj_cx = traj_box[0] * self.bbox_unnorm[0]
                    traj_cy = traj_box[1] * self.bbox_unnorm[1]

                    # Find closest detection among tied candidates
                    best_idx = None
                    best_dist = float('inf')
                    for det_i, det_conf in dets:
                        if det_conf < top_conf - TIEBREAK_MARGIN:
                            break  # below margin
                        det_cx = boxes[det_i, 0] * self.bbox_unnorm[0]
                        det_cy = boxes[det_i, 1] * self.bbox_unnorm[1]
                        dist = ((det_cx - traj_cx)**2 + (det_cy - traj_cy)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = det_i

                    # Promote the closest detection to be the "max" by boosting its conf
                    if best_idx is not None and best_idx != dets[0][0]:
                        object_max_confs[best_idx] = top_conf + 1e-4
                        # Demote the original winner
                        object_max_confs[dets[0][0]] = top_conf - 1e-4

        # Get the max confs of each ID label:
        id_max_confs = dict()
        for conf, id_label in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if id_label not in id_max_confs:
                id_max_confs[id_label] = conf
            else:
                id_max_confs[id_label] = max(id_max_confs[id_label], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0  # special token

        # Assign ID labels:
        for _ in range(len(object_max_id_labels)):
            if object_max_id_labels[_].item() not in trajectory_id_labels_set:         # not in tracked IDs -> newborn
                id_labels.append(self.num_id_vocabulary)
            else:
                _id_label = object_max_id_labels[_].item()
                _conf = object_max_confs[_].item()
                if _conf < self.id_thresh or _conf < id_max_confs[_id_label]:  # low conf or not the max conf -> newborn
                    id_labels.append(self.num_id_vocabulary)
                elif _id_label in id_labels:
                    id_labels.append(self.num_id_vocabulary)
                else:                                                          # normal case
                    id_labels.append(_id_label)

        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        # Get the max confs of each object:
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:    # a very rare case
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)

        # Assign ID labels:
        for _ in range(len(id_max_obj_idxs)):
            _obj_idx, _id_label, _conf = id_max_obj_idxs[_].item(), _, id_max_confs[_].item()
            if _conf < self.id_thresh or _conf < object_max_confs[_obj_idx]:
                pass
            elif _id_label not in trajectory_id_labels_set:
                pass
            else:
                id_labels[_obj_idx] = _id_label

        return id_labels


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(A, B) IoU of xyxy boxes."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class ConfirmFilter:
    """Output side of tracklet confirmation (RuntimeTracker n_init > 1): holds emitted frames for n_init - 1 frames
    and deletes the boxes of every track reported in a later frame's ``dropped_ids``. A track that is confirmed keeps
    all its frames (backfill). Takes frames in order from any source (the tracker, FixedLagTracker, QuarantineRewriter).
    """

    def __init__(self, n_init: int):
        self.hold = max(int(n_init) - 1, 0)
        self.buf = []
        self.stats = {"boxes_removed": 0}

    def push(self, ready: list) -> list:
        out = []
        for f, res in ready:
            drop = set(res.get("dropped_ids") or [])
            if drop:
                self.buf = [(g, self._without(r, drop)) for g, r in self.buf]
            self.buf.append((f, res))
            while len(self.buf) > self.hold:
                out.append(self.buf.pop(0))
        return out

    def finish(self) -> list:
        out, self.buf = self.buf, []
        return out

    def _without(self, res: dict, drop: set) -> dict:
        ids = res["id"]
        keep = torch.tensor([int(i) not in drop for i in ids.tolist()], dtype=torch.bool)
        if bool(keep.all()):
            return res
        self.stats["boxes_removed"] += int((~keep).sum())
        out = dict(res)
        for k, v in res.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == keep.shape[0] and k != "dropped_ids":
                out[k] = v[keep.to(v.device)]
        return out


class QuarantineRewriter:
    """Rewrites the IDs of quarantined (entangled) frames once the group separates (opt-in, ``QUARANTINE_REWRITE``).

    Wraps a RuntimeTracker built with ``quarantine=True``. Frames are buffered from the first frame with a quarantined
    detection until a frame without any (the episode closes; adaptive lag) or ``max_buffer`` frames. Then, from the
    last frame backwards, the episode's entangled IDs (every output ID quarantined in the episode) are carried from
    frame f+1 to frame f by box continuity: Hungarian on IoU between the entangled detections of the two frames, pairs
    with IoU >= ``link_iou``; each linked detection of frame f takes its successor's ID, the others keep theirs when
    free or take a freed entangled ID. A frame where that would duplicate an ID keeps its online IDs. Detections not
    involved and every box / score are unchanged. So the entangled frames get the IDs that the decision at separation
    (made against the clean pre-crossing histories) gave the players they continue into.
    """

    def __init__(self, tracker: RuntimeTracker, max_buffer: int = 60, link_iou: float = 0.3):
        if not tracker.quarantine:
            raise ValueError("QuarantineRewriter needs a tracker built with quarantine=True")
        self.tracker = tracker
        self.max_buffer = int(max_buffer)
        self.link_iou = float(link_iou)
        self.t = 0
        self.buf = []            # [(frame, results)]
        self.ent_ids = set()
        self.stats = {"episodes": 0, "frames_buffered": 0, "dets_relabelled": 0, "frames_conflict": 0,
                      "forced_flush": 0}

    @torch.no_grad()
    def update(self, image, **kw) -> list:
        self.tracker.update(image, **kw)
        return self._step()

    @torch.no_grad()
    def update_from_detections(self, *args, **kw) -> list:
        self.tracker.update_from_detections(*args, **kw)
        return self._step()

    def finish(self) -> list:
        return self._flush()

    def _step(self) -> list:
        t = self.t
        self.t += 1
        res = self.tracker.get_track_results()
        q = res.get("quarantined")
        q_ids = set(res["id"][q].tolist()) if q is not None and len(q) else set()
        if not self.buf and not q_ids:
            return [(t, res)]
        self.buf.append((t, res))
        self.ent_ids |= q_ids
        if not q_ids:
            return self._flush()
        if len(self.buf) >= self.max_buffer:
            self.stats["forced_flush"] += 1
            return self._flush()
        return []

    def _flush(self) -> list:
        if not self.buf:
            return []
        self.stats["episodes"] += 1
        self.stats["frames_buffered"] += len(self.buf)
        E = self.ent_ids
        out = [self.buf[-1]]
        nxt = self.buf[-1][1]
        for f, cur in reversed(self.buf[:-1]):
            cur = self._relink(cur, nxt, E)
            out.append((f, cur))
            nxt = cur
        self.buf, self.ent_ids = [], set()
        return out[::-1]

    def _relink(self, cur: dict, nxt: dict, E: set) -> dict:
        ids = cur["id"].tolist()
        ci = [i for i, x in enumerate(ids) if x in E]
        nids = nxt["id"].tolist()
        ni = [j for j, x in enumerate(nids) if x in E]
        if not ci or not ni:
            return cur
        def xyxy(b):
            b = b.float().cpu()
            return torch.cat([b[:, :2], b[:, :2] + b[:, 2:]], dim=1)
        iou = _box_iou(xyxy(cur["bbox"][ci]), xyxy(nxt["bbox"][ni])).numpy()
        rows, cols = linear_sum_assignment(-iou)
        new = {}
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.link_iou:
                new[ci[r]] = nids[ni[c]]
        used = set(new.values())
        free = [ids[i] for i in ci if ids[i] not in used]
        for i in ci:
            if i in new:
                continue
            if ids[i] not in used:
                new[i] = ids[i]
            else:
                cand = [x for x in free if x not in used]
                if not cand:
                    self.stats["frames_conflict"] += 1
                    return cur
                new[i] = cand[0]
            used.add(new[i])
        final = list(ids)
        for i, x in new.items():
            final[i] = x
        if len(set(final)) != len(final):
            self.stats["frames_conflict"] += 1
            return cur
        changed = sum(1 for a, b in zip(ids, final) if a != b)
        if not changed:
            return cur
        self.stats["dets_relabelled"] += changed
        out = dict(cur)
        out["id"] = torch.tensor(final, dtype=torch.int64)
        return out


class _LagHypothesis:
    """One identity hypothesis of a FixedLagTracker: its own tracker state (history), cumulative log-probability,
    the non-default choices it took at branch points (``path``: ((frame, key), ...), only the uncommitted ones) and
    the per-frame results of the frames not emitted yet (``buf``: [(frame, track_results), ...])."""
    __slots__ = ("trk", "logp", "path", "buf")

    def __init__(self, trk, logp: float, path: tuple, buf: list):
        self.trk, self.logp, self.path, self.buf = trk, logp, path, buf


class FixedLagTracker:
    """Fixed-lag identity decisions on top of a causal RuntimeTracker (inference only, opt-in; ``FIXED_LAG`` > 0).

    MOTIP commits every frame's IDs immediately. Here an ambiguous assignment is not committed: when, at frame t, an
    alternative assignment of one hypothesis is within ``margin`` (natural log) of its Hungarian assignment, the
    alternative is kept as a second hypothesis. Each hypothesis carries its own copy of the tracker state, so the ID
    decoder runs forward under each history, and each accumulates the log-probability of its own assignments
    (sum over detections of log p(det -> its label), newborn column included). Frame t is emitted at t + ``lag``
    (N-scan pruning): the best hypothesis by cumulative log-probability is taken, every hypothesis that disagrees with
    it on a decision at or before t is dropped, and frame t's results come from it, i.e. the IDs of the buffered frames
    are rewritten to the winning hypothesis. ``finish()`` emits the rest at the end of the sequence.

    Alternatives (each only when both the default and the alternative probabilities are >= the tracker's id_thresh,
    and the detections / tracks involved are within ``near`` box heights of each other, centre to centre; ``near`` <= 0
    disables the proximity test):
      - "swap": detections i, j hold live IDs a, b -> i takes b, j takes a (a crossing);
      - "move": detection i holds a, live track b is unmatched this frame -> i takes b, a is unmatched (one box for
        two players during an occlusion);
      - "give": detection i holds a, detection j is a newborn -> j takes a, i becomes a newborn (an ID steal).
    At most ``max_branch`` alternatives (smallest log-probability gap first) per hypothesis per frame, and at most
    ``max_hyps`` live hypotheses (lowest cumulative log-probability dropped), so memory is bounded by
    ``max_hyps`` tracker states and ``lag`` + 1 buffered frames each. With a single live hypothesis the decisions are
    exactly the online ones: identical boxes, scores and IDs, only delayed.

    ``score`` = "own" ranks hypotheses by the log-probability of their assignments under their OWN decoder (each
    conditioned on its own history: the plain chain rule, but a wrong history can make its own later predictions
    confident: "self-confirmation"). ``score`` = "ref" instead scores every hypothesis of a branch group against the
    same frozen reference: the tracker state just before the group's first branch (frame t0); at frame t each
    hypothesis' assignment is scored by the decoder run on that reference history with the tracks unseen for
    t - t0 frames (``RuntimeTracker._lag_ref_scores``; one extra decoder call per frame while a group is open; labels the
    reference does not know count as newborns). So the hypotheses are compared on the same evidence: which assignment of
    the window's detections best matches the identities as they were before the ambiguity. A group is closed (best
    hypothesis kept) ``lag`` frames after t0 at the latest, which bounds the reference's age.

    ``kinds`` restricts the alternatives to a subset of ("swap", "move", "give"). ``penalty`` (nats) is subtracted from
    an alternative's log-probability when it is created: a prior in favour of the online decision, so an alternative
    must out-score it by ``penalty`` over the lag window to be committed (0 = plain maximum likelihood).
    """

    def __init__(self, tracker: RuntimeTracker, lag: int, margin: float = 2.0, max_hyps: int = 4,
                 max_branch: int = 2, near: float = 1.0, eps: float = 1e-6, penalty: float = 0.0,
                 kinds: tuple = ("swap", "move", "give"), score: str = "own"):
        if int(lag) < 1:
            raise ValueError(f"lag must be >= 1 frame, not {lag}")
        if max_hyps < 1 or max_branch < 0 or margin < 0:
            raise ValueError(f"need max_hyps >= 1, max_branch >= 0, margin >= 0; got {max_hyps}, {max_branch}, {margin}")
        self.lag = int(lag)
        self.margin = float(margin)
        self.max_hyps = int(max_hyps)
        self.max_branch = int(max_branch)
        self.near = float(near)
        self.eps = float(eps)
        self.penalty = float(penalty)
        self.kinds = frozenset(kinds)
        if score not in ("own", "ref"):
            raise ValueError(f"score must be 'own' or 'ref', not {score!r}")
        self.score = score
        self.ref = None          # score="ref": (reference tracker snapshot, t0) of the open branch group
        self._ref_cache = None   # (frame, P_ref numpy or None, set of labels the reference knows)
        if not self.kinds <= {"swap", "move", "give"}:
            raise ValueError(f"unknown alternative kinds {sorted(self.kinds - {'swap', 'move', 'give'})}")
        self.hyps = [_LagHypothesis(tracker, 0.0, (), [])]
        self.t = 0
        self.stats = {"frames": 0, "hyp_frames": 0, "branch_frames": 0, "alts": 0, "dropped_cap": 0,
                      "dropped_scan": 0, "max_live": 1, "committed_alts": 0, "frames_overridden": 0,
                      "committed_swap": 0, "committed_move": 0, "committed_give": 0, "groups": 0, "ref_calls": 0,
                      "group_closed_early": 0}

    @property
    def tracker(self) -> RuntimeTracker:
        """The current best hypothesis' tracker (for its stats / settings)."""
        return self.hyps[0].trk

    # -- per frame ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def update(self, image, suppress_boxes=None, _det_cache=None, rink_H=None, frame_bgr=None) -> list:
        """RuntimeTracker.update with the DETR forward run once for all hypotheses. Returns the frames ready now."""
        trk = self.tracker
        detr_out = trk.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = trk._get_activate_detections(detr_out=detr_out)
        kit = trk._kit_from_frame(frame_bgr, boxes) if (trk.team_cue and frame_bgr is not None) else None
        if _det_cache is not None:
            _det_cache.append((scores, categories, boxes, output_embeds) if kit is None else (scores, categories, boxes, output_embeds, kit))
        return self.update_from_detections(scores, categories, boxes, output_embeds, suppress_boxes, rink_H=rink_H, kit=kit)

    @torch.no_grad()
    def update_from_detections(self, scores, categories, boxes, output_embeds, suppress_boxes=None, rink_H=None,
                               kit=None) -> list:
        """Advance every hypothesis by one frame. Returns ``[(frame, track_results), ...]`` for the frames that are
        final now (frame = 0-based index of the update call; empty for the first ``lag`` frames)."""
        t = self.t
        self.t += 1
        args = (scores, categories, boxes, output_embeds, suppress_boxes)
        kw = dict(rink_H=rink_H, kit=kit)
        nxt = []
        branched = False
        for h in self.hyps:
            ctx = {"logp": 0.0, "alts": [], "t": t, "reset": False}
            h.trk._lag_hook = lambda trk, s, labels, bx, emb, _ctx=ctx: self._on_frame(trk, s, labels, bx, emb, _ctx)
            try:
                h.trk.update_from_detections(*args, **kw)
            finally:
                h.trk._lag_hook = None
            base, prev_buf = (0.0 if ctx["reset"] else h.logp), h.buf
            h.logp = base + ctx["logp"]
            h.buf = prev_buf + [(t, h.trk.get_track_results())]
            nxt.append(h)
            for key, clone, alt_labels, alt_logp, s, gate_cut in ctx["alts"]:
                clone._forced = (s, alt_labels, gate_cut)
                clone.update_from_detections(*args, **kw)
                nxt.append(_LagHypothesis(clone, base + alt_logp, h.path + ((t, key),),
                                          prev_buf + [(t, clone.get_track_results())]))
                self.stats["alts"] += 1
                branched = True
        st = self.stats
        st["frames"] += 1
        st["hyp_frames"] += len(self.hyps)
        st["branch_frames"] += int(branched)
        nxt.sort(key=lambda h: -h.logp)          # stable: at a tie the parent (default) stays ahead of its alternative
        if len(nxt) > self.max_hyps:
            st["dropped_cap"] += len(nxt) - self.max_hyps
            nxt = nxt[:self.max_hyps]
        st["max_live"] = max(st["max_live"], len(nxt))
        if self.ref is not None and len(nxt) > 1 and t - self.ref[1] >= self.lag:
            st["group_closed_early"] += 1
            st["dropped_scan"] += len(nxt) - 1
            nxt = nxt[:1]
        if len(nxt) == 1:
            self.ref = None
        self.hyps = nxt
        ready = t - self.lag
        return self._commit(ready) if ready >= 0 else []

    def finish(self) -> list:
        """End of the sequence: commit the best hypothesis and return every frame still buffered."""
        if not self.hyps or not self.hyps[0].buf:
            return []
        return self._commit(self.hyps[0].buf[-1][0])

    def _commit(self, ready: int) -> list:
        """Keep only the hypotheses that agree with the best one on every decision up to frame ``ready``; emit the
        best one's frames up to ``ready``."""
        best = self.hyps[0]
        def decided(h):
            return tuple(e for e in h.path if e[0] <= ready)
        pref = decided(best)
        keep = [h for h in self.hyps if decided(h) == pref]
        self.stats["dropped_scan"] += len(self.hyps) - len(keep)
        self.stats["committed_alts"] += len(pref)
        for _, key in pref:
            self.stats["committed_" + key[0]] += 1
        if pref:
            first = min(e[0] for e in pref)
            self.stats["frames_overridden"] += sum(1 for f, _ in best.buf if first <= f <= ready)
        n = sum(1 for f, _ in best.buf if f <= ready)
        out = best.buf[:n]
        for h in keep:
            h.buf = h.buf[n:]
            h.path = tuple(e for e in h.path if e[0] > ready)
        self.hyps = keep
        return out

    # -- branching ------------------------------------------------------------------------------------------------
    def _ref_logp(self, trk: RuntimeTracker, boxes, output_embeds, t: int):
        """(log P_ref (D, V+1) or None, labels known to the reference) for frame t, computed once per frame."""
        if self._ref_cache is None or self._ref_cache[0] != t:
            ref, t0 = self.ref
            P = ref._lag_ref_scores(boxes, output_embeds, gap=t - t0, rink_cur=trk._rink_cur, kit_cur=trk._kit_cur)
            self.stats["ref_calls"] += 1
            known = set(ref.trajectory_id_labels[0].tolist()) if ref.trajectory_id_labels.shape[0] > 0 else set()
            lp = None if P is None else np.log(np.maximum(P.float().cpu().numpy(), self.eps))
            self._ref_cache = (t, lp, known)
        return self._ref_cache[1], self._ref_cache[2]

    def _ref_score(self, lp, known, cols, V) -> float:
        if lp is None:
            return 0.0
        return float(sum(lp[i, c if c in known else V] for i, c in enumerate(cols)))

    def _on_frame(self, trk: RuntimeTracker, id_scores: torch.Tensor, labels: list, boxes: torch.Tensor,
                  output_embeds: torch.Tensor, ctx: dict):
        """Lag hook (called inside trk._get_id_pred_labels): score this frame's assignment; snapshot alternatives."""
        V = trk.num_id_vocabulary
        P = id_scores.float().cpu().numpy()
        logP = np.log(np.maximum(P, self.eps))
        D = len(labels)
        cols = [min(int(l), V) for l in labels]
        ctx["logp"] = float(sum(logP[i, cols[i]] for i in range(D)))
        use_ref = self.score == "ref"
        if use_ref and self.ref is not None:
            lp, known = self._ref_logp(trk, boxes, output_embeds, ctx["t"])
            ctx["logp"] = self._ref_score(lp, known, cols, V)
        if self.max_hyps <= 1 or self.max_branch <= 0 or D == 0:
            return
        thr = trk.id_thresh
        scale = trk.bbox_unnorm.float().cpu().numpy()
        b = boxes.float().cpu().numpy().reshape(-1, 4)
        cx, cy, hh = b[:, 0] * scale[0], b[:, 1] * scale[1], b[:, 3] * scale[3]

        def close(x1, y1, h1, x2, y2, h2):
            return self.near <= 0 or math.hypot(x1 - x2, y1 - y2) <= self.near * max(h1, h2)

        held = [i for i in range(D) if cols[i] != V]
        newborn = [i for i in range(D) if cols[i] == V]
        cands = []
        for ii, i in enumerate(held):
            a = cols[i]
            for j in held[ii + 1:]:
                bb = cols[j]
                if P[i, bb] < thr or P[j, a] < thr or not close(cx[i], cy[i], hh[i], cx[j], cy[j], hh[j]):
                    continue
                delta = logP[i, a] + logP[j, bb] - logP[i, bb] - logP[j, a]
                if delta <= self.margin and "swap" in self.kinds:
                    alt = list(cols); alt[i], alt[j] = bb, a
                    cands.append((delta, ("swap", min(a, bb), max(a, bb)), alt))
            for j in newborn:
                if P[j, a] < thr or not close(cx[i], cy[i], hh[i], cx[j], cy[j], hh[j]):
                    continue
                delta = logP[i, a] + logP[j, V] - logP[i, V] - logP[j, a]
                if delta <= self.margin and "give" in self.kinds:
                    alt = list(cols); alt[i], alt[j] = V, a
                    cands.append((delta, ("give", a, j), alt))
        live = trk.trajectory_id_labels[0].tolist() if trk.trajectory_id_labels.shape[0] > 0 else []
        unmatched = [l for l in live if l not in set(cols) and l < V]
        if unmatched and held and "move" in self.kinds:
            tb, tm = trk.trajectory_boxes, trk.trajectory_masks
            seen = ~tm
            rows = torch.arange(tm.shape[0], device=tm.device)[:, None].expand_as(tm)
            last_t = torch.where(seen, rows, torch.full_like(rows, -1)).max(dim=0).values
            last = tb[last_t.clamp(min=0), torch.arange(tm.shape[1], device=tm.device)].float().cpu().numpy()
            col_of = {l: k for k, l in enumerate(live)}
            for i in held:
                a = cols[i]
                for bb in unmatched:
                    k = col_of[bb]
                    if P[i, bb] < thr or int(last_t[k]) < 0:
                        continue
                    if not close(cx[i], cy[i], hh[i], last[k, 0] * scale[0], last[k, 1] * scale[1], last[k, 3] * scale[3]):
                        continue
                    delta = logP[i, a] - logP[i, bb]
                    if delta <= self.margin:
                        alt = list(cols); alt[i] = bb
                        cands.append((delta, ("move", a, bb), alt))
        if not cands:
            return
        cands.sort(key=lambda c: c[0])
        cands = cands[:self.max_branch]
        if use_ref:
            if self.ref is None:
                # Open a branch group: the reference is this (single) hypothesis' state before this frame.
                self.ref = (trk._lag_clone(), ctx["t"])
                self._ref_cache = None
                self.stats["groups"] += 1
                ctx["reset"] = True
            lp, known = self._ref_logp(trk, boxes, output_embeds, ctx["t"])
            ctx["logp"] = self._ref_score(lp, known, cols, V)
            for delta, key, alt in cands:
                ctx["alts"].append((key, trk._lag_clone(), alt, self._ref_score(lp, known, alt, V) - self.penalty,
                                    id_scores, trk._gate_cut))
            return
        for delta, key, alt in cands:
            ctx["alts"].append((key, trk._lag_clone(), alt, ctx["logp"] - float(delta) - self.penalty, id_scores,
                                trk._gate_cut))
