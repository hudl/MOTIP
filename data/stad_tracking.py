# IHC-PATCH: STADTracking — STAD player tracking dataset for MOTIP stage 2.
# Layout: <data_root>/train/<clip_id>/{img1/, gt/gt.txt, seqinfo.ini}
# No sub_dir wrapper — train/ sits directly under the channel root.

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
