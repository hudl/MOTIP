# IHC-PATCH: STADTracking — STAD player tracking dataset for MOTIP stage 2.
# Layout: <data_root>/train/<clip_id>/{img1/, gt/gt.txt, seqinfo.ini}
# No sub_dir wrapper — train/ sits directly under the channel root.

import os
from configparser import ConfigParser

from .dancetrack import DanceTrack


class STADTracking(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(STADTracking, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

    def _get_sequence_names(self):
        # Only return sequences that have a valid seqinfo.ini with a [Sequence]
        # section — some STAD clips were uploaded without one and would crash
        # configparser with KeyError: 'Sequence' during _get_sequence_infos().
        split_dir = os.path.join(self.data_dir, self.split)
        valid = []
        skipped = 0
        for name in os.listdir(split_dir):
            ini_path = os.path.join(split_dir, name, "seqinfo.ini")
            ini = ConfigParser()
            ini.read(ini_path)
            if "Sequence" in ini:
                valid.append(name)
            else:
                skipped += 1
        if skipped:
            print(f"[STADTracking] WARNING: skipped {skipped} sequences with missing/invalid seqinfo.ini")
        return valid

    def _get_annotations(self):
        # Override to skip gt.txt rows where frame_id exceeds seqlength — can
        # happen when seqinfo.ini seqlength and gt.txt are slightly out of sync.
        from .util import is_legal, append_annotation
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)
        clipped_seqs = 0
        for sequence_name in sequence_names:
            seq_len = self.sequence_infos[sequence_name]["length"]
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "gt", "gt.txt")
            with open(gt_file_path, "r") as gt_file:
                had_overrun = False
                for line in gt_file:
                    line = line.strip().split(",")
                    frame_id, obj_id, x, y, w, h, _, _, _ = line
                    frame_id, obj_id = map(int, [frame_id, obj_id])
                    ann_index = frame_id - 1
                    if ann_index >= seq_len:
                        had_overrun = True
                        continue
                    x, y, w, h = map(float, [x, y, w, h])
                    annotations[sequence_name][ann_index] = append_annotation(
                        annotation=annotations[sequence_name][ann_index],
                        obj_id=obj_id,
                        category=0,
                        bbox=[x, y, w, h],
                        visibility=1.0,
                    )
                if had_overrun:
                    clipped_seqs += 1
        if clipped_seqs:
            print(f"[STADTracking] WARNING: {clipped_seqs} sequences had gt.txt frame_ids beyond seqlength (rows skipped)")
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations
