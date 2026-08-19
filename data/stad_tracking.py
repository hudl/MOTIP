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
