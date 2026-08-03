# IHC-PATCH: AMF player detection dataset for DETR pretraining.
# Same on-disk layout as CrowdHuman (images/<name>.jpg, gts/<name>.txt).

import os
from .crowdhuman import CrowdHuman


class AMFDetection(CrowdHuman):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "AMFDetection",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(AMFDetection, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

    def _get_sequence_names(self):
        # Only include frames that have both an image and a gt file — the
        # S3 sync and live extraction can leave orphaned images with no gt.
        image_dir = os.path.join(self.data_dir, self.split, "images")
        gt_dir = os.path.join(self.data_dir, self.split, "gts")
        names = []
        for fname in os.listdir(image_dir):
            if not fname.endswith(".jpg"):
                continue
            name = fname[:-4]
            if os.path.exists(os.path.join(gt_dir, f"{name}.txt")):
                names.append(name)
        return names
