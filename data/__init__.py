# Copyright (c) Ruopeng Gao. All Rights Reserved.

from torch.utils.data import DataLoader

from .joint_dataset import JointDataset
from .transforms import build_transforms
from .util import collate_fn


def build_dataset(config: dict):
    dataset_kwargs = {}
    start_frame_fraction = config.get("START_FRAME_FRACTION", 1.0)
    if start_frame_fraction < 1.0:
        dataset_kwargs["STADTracking"] = {"start_frame_fraction": start_frame_fraction}
    from .rink import needs_rink_data
    _rink = needs_rink_data(config)
    if _rink:
        dataset_kwargs["Hockey"] = {"load_rink": True}
    _team = bool(config.get("ID_EMBED_LOSS", False) and config.get("ID_EMBED_TEAM", False))
    if _team:
        dataset_kwargs.setdefault("Hockey", {})["load_team"] = True
    return JointDataset(
        data_root=config["DATA_ROOT"],
        datasets=config["DATASETS"],
        splits=config["DATASET_SPLITS"],
        transforms=build_transforms(config),
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        dataset_kwargs=dataset_kwargs,
        rink_anchor=_rink,
        load_team=_team,
        detr_cache=bool(config.get("DETR_CACHE_DIR")),
    )


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
