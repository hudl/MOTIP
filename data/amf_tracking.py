# IHC-PATCH: AMFTracking — AMF player tracking dataset for MOTIP stage 2.
# Same on-disk layout as DanceTrack/Hockey (8-digit jpg, gt/gt.txt, seqinfo.ini).

from .dancetrack import DanceTrack


class AMFTracking(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "AMFTracking",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(AMFTracking, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
