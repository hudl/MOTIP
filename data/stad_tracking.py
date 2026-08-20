# IHC-PATCH: STADTracking — STAD player tracking dataset for MOTIP stage 2.
# Layout: <data_root>/train/<clip_id>/{img1/, gt/gt.txt, seqinfo.ini}
# No sub_dir wrapper — train/ sits directly under the channel root.

import os
from configparser import ConfigParser

from .dancetrack import DanceTrack

# Minimum frames needed to form one valid sample (sample_length=30 at interval=1).
_MIN_SAMPLE_FRAMES = 30


class STADTracking(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "",
            split: str = "train",
            load_annotation: bool = True,
            start_frame_fraction: float = 1.0,
    ):
        # Fraction of each clip's frames that are eligible as start positions.
        # 1.0 = full clip; 0.2 = first 20%; 0.0 = frame 0 only.
        self.start_frame_fraction = start_frame_fraction
        super(STADTracking, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

    def get_sequence_infos(self):
        if self.start_frame_fraction >= 1.0:
            return self.sequence_infos
        # Cap the reported length so the sampler only picks starts from the
        # first (start_frame_fraction * real_length) frames.  Internal uses
        # (annotation loading, image paths) still see the real length via
        # self.sequence_infos, so nothing else breaks.
        limited = {}
        for seq, meta in self.sequence_infos.items():
            real_len = meta["length"]
            capped = min(real_len, max(_MIN_SAMPLE_FRAMES, int(real_len * self.start_frame_fraction)))
            limited[seq] = {**meta, "length": capped}
        return limited

    def _get_sequence_names(self):
        # Filter out sequences with any of:
        #   - missing/invalid seqinfo.ini (no [Sequence] section)
        #   - image count != seqlength (frame gaps cause FileNotFoundError in DataLoader)
        split_dir = os.path.join(self.data_dir, self.split)
        valid = []
        skip_ini = 0
        skip_frames = 0
        for name in os.listdir(split_dir):
            seq_dir = os.path.join(split_dir, name)
            if not os.path.isdir(seq_dir):
                continue
            ini_path = os.path.join(seq_dir, "seqinfo.ini")
            ini = ConfigParser()
            ini.read(ini_path)
            if "Sequence" not in ini:
                skip_ini += 1
                continue
            try:
                seq_len = int(ini["Sequence"]["seqlength"])
            except (KeyError, ValueError):
                skip_ini += 1
                continue
            img_dir = os.path.join(seq_dir, "img1")
            try:
                n_images = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            except FileNotFoundError:
                skip_frames += 1
                continue
            if n_images != seq_len:
                skip_frames += 1
                continue
            valid.append(name)
        if skip_ini:
            print(f"[STADTracking] WARNING: skipped {skip_ini} sequences with missing/invalid seqinfo.ini")
        if skip_frames:
            print(f"[STADTracking] WARNING: skipped {skip_frames} sequences with frame count != seqlength")
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
