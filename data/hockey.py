# Copyright (c) Ruopeng Gao. All Rights Reserved.
# IHC: hockey huddle resolver dataset. Subclass of DanceTrack; same on-disk layout.

import os

from .dancetrack import DanceTrack


class Hockey(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "Hockey",
            split: str = "train",
            load_annotation: bool = True,
            load_rink: bool = False,
            load_team: bool = False,
    ):
        super(Hockey, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        # Rink anchor: per-sequence image -> rink transforms from <seq>/rink/ (sequences without it are skipped).
        self.rink = {}
        if load_rink:
            from .rink import load_sequence_rink
            for sequence_name in self.sequence_infos:
                _r = load_sequence_rink(self._get_sequence_dir(self.data_dir, self.split, sequence_name))
                if _r is not None:
                    self.rink[sequence_name] = _r

        # ID_EMBED_TEAM (training only): {gt track id: team} per sequence from <seq>/gt/gt_team.txt (missing -> none).
        self.team = {}
        if load_team:
            from .rink import load_gt_team
            for sequence_name in self.sequence_infos:
                _t = load_gt_team(self._get_sequence_dir(self.data_dir, self.split, sequence_name))
                if _t is not None:
                    self.team[sequence_name] = _t

    def get_rink(self):
        return self.rink

    def get_team(self):
        return self.team

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        # Match SportsMOT (1-indexed 6-digit). extract_hard_windows produces
        # 00000001.jpg (8-digit), but DanceTrack._get_image_path uses 8-digit
        # too — verified by reading dancetrack.py. Keep parent's default.
        return DanceTrack._get_image_path(sequence_dir, frame_idx)
