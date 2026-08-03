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
    ):
        super(Hockey, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        # Match SportsMOT (1-indexed 6-digit). extract_hard_windows produces
        # 00000001.jpg (8-digit), but DanceTrack._get_image_path uses 8-digit
        # too — verified by reading dancetrack.py. Keep parent's default.
        return DanceTrack._get_image_path(sequence_dir, frame_idx)
