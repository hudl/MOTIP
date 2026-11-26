# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import math
import contextlib
import numpy as np
import torch
import einops
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from collections import defaultdict
from torchvision.transforms import v2
from typing import Any, Generator, List

from models.motip import build as build_motip
from models.motip import take_match_guard_events
from models.motip.id_criterion import build as build_id_criterion
from runtime_option import runtime_option
from utils.misc import yaml_to_dict, set_seed
from configs.util import load_super_config, update_config
from log.logger import Logger
from data import build_dataset
from data.naive_sampler import NaiveSampler
from data.util import collate_fn
from log.log import TPS, Metrics
from models.misc import load_detr_pretrain, save_checkpoint, load_checkpoint
from models.misc import get_model
from utils.nested_tensor import NestedTensor
from utils.switch_stats import empty_switch_stats, add_switch_stats, clip_switch_stats, summarize_switch_stats
from submit_and_evaluate import submit_and_evaluate_one_model


import data.rink as _rink_mod

def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    # RF-DETR has params unused in some forward passes (windowed attn); DDP requires find_unused_parameters=True
    _ddp_kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True)] if config.get("DETR_FRAMEWORK") == "rf_detr" else []
    accelerator = Accelerator(kwargs_handlers=_ddp_kwargs)
    state = PartialState()
    # Also, we set the seed:
    set_seed(config["SEED"])
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.

    # Init Logger:
    logger = Logger(
        logdir=os.path.join(outputs_dir, "train"),
        use_wandb=config["USE_WANDB"],
        config=config,
        exp_owner=config["EXP_OWNER"],
        exp_project=config["EXP_PROJECT"],
        exp_group=config["EXP_GROUP"],
        exp_name=config["EXP_NAME"],
    )
    logger.info(f"We init the logger at {logger.logdir}.")
    if config["USE_WANDB"] is False:
        logger.warning("The wandb is not used in this experiment.")
    logger.info(f"The distributed type is {state.distributed_type}.")
    logger.config(config=config)

    # Build training dataset:
    train_dataset = build_dataset(config=config)
    logger.dataset(train_dataset)
    # Cached frozen-detector outputs (skips the detector forward; see utils/detr_cache.py):
    detr_cache = None
    if config.get("DETR_CACHE_DIR"):
        assert config.get("DETR_FREEZE", False) and not config["ONLY_DETR"], "DETR_CACHE_DIR needs DETR_FREEZE and an ID head to train."
        from utils.detr_cache import DetrCache
        detr_cache = DetrCache(config["DETR_CACHE_DIR"], with_kit=config.get("TEAM_CUE", False),
                              max_open=config.get("DETR_CACHE_MAX_OPEN", 32))  # sequences kept in RAM (LRU)
        logger.info(f"DETR_CACHE_DIR={config['DETR_CACHE_DIR']}: detector forward skipped, flip aug on cached boxes={config.get('DETR_CACHE_FLIP', False)}.")
    # Build training data sampler:
    if "DATASET_WEIGHTS" in config:
        data_weights = defaultdict(lambda: defaultdict())
        for _ in range(len(config["DATASET_WEIGHTS"])):
            data_weights[config["DATASETS"][_]][config["DATASET_SPLITS"][_]] = config["DATASET_WEIGHTS"][_]
        data_weights = dict(data_weights)
    else:
        data_weights = None
    if config.get("SAMPLE_MANIFEST"):
        # Explicit candidate windows (data/manifest_sampler.py); SAMPLE_LENGTHS / INTERVALS / STRIDE are then unused.
        from data.manifest_sampler import ManifestSampler, resolve_manifest_path
        train_sampler = ManifestSampler(
            data_source=train_dataset,
            manifest=resolve_manifest_path(config["SAMPLE_MANIFEST"], config.get("DATA_ROOT")),
            seed=config["SEED"],
            samples_per_epoch=config.get("SAMPLE_MANIFEST_SAMPLES"),
        )
        if train_sampler.variable_length:
            assert config["BATCH_SIZE"] == 1, f"SAMPLE_MANIFEST has windows of lengths {train_sampler.lengths}; needs BATCH_SIZE 1."
        if config.get("TIME_UNIT", "step") == "frame":
            _span = max(w["frame_idxs"][-1] - w["frame_idxs"][0] for w in train_sampler.windows)
            assert _span < config["REL_PE_LENGTH"], f"TIME_UNIT frame: windows span {_span} frames, REL_PE_LENGTH is {config['REL_PE_LENGTH']}"
        logger.info(f"SAMPLE_MANIFEST={config['SAMPLE_MANIFEST']}: {len(train_sampler.windows)} candidate windows, "
                    f"{train_sampler.samples_per_epoch} per epoch, lengths {train_sampler.lengths}, dropped {train_sampler.dropped}.")
    else:
        train_sampler = NaiveSampler(
            data_source=train_dataset,
            sample_steps=config["SAMPLE_STEPS"],
            sample_lengths=config["SAMPLE_LENGTHS"],
            sample_intervals=config["SAMPLE_INTERVALS"],
            length_per_iteration=config["LENGTH_PER_ITERATION"],
            data_weights=data_weights,
            sample_stride=config.get("SAMPLE_STRIDE", 1),
            short_sequence_mode=config.get("SAMPLE_SHORT_SEQUENCES", "drop"),
        )
    if config.get("SAMPLE_SHORT_SEQUENCES", "drop") == "whole":
        assert config["BATCH_SIZE"] == 1, "SAMPLE_SHORT_SEQUENCES: whole gives variable-length samples; needs BATCH_SIZE 1."
    # Build training data loader:
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
        prefetch_factor=config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }

    # Build MOTIP model:
    model, detr_criterion = build_motip(config=config)
    # Load the pre-trained DETR (skip if DETR_PRETRAIN is None, e.g. resume runs):
    if config["DETR_PRETRAIN"] is not None:
        load_detr_pretrain(
            model=model, pretrain_path=config["DETR_PRETRAIN"], num_classes=config["NUM_CLASSES"],
            default_class_idx=config["DETR_DEFAULT_CLASS_IDX"] if "DETR_DEFAULT_CLASS_IDX" in config else None,
        )
        detr_path = config["DETR_PRETRAIN"]
        logger.success(
            log=f"Load the pre-trained DETR from '{detr_path}'. "
        )
    # Build Loss Function:
    id_criterion = build_id_criterion(config=config)
    # ID_EMBED_LOSS: hard-negative identity loss on the detector's output embeddings (models/motip/id_embed_loss.py).
    id_embed_criterion = None
    if config.get("ID_EMBED_LOSS", False):
        from models.motip.id_embed_loss import IDEmbedCriterion
        assert not config["ONLY_DETR"], "ID_EMBED_LOSS needs the MOTIP part (GT-matched queries)."
        id_embed_criterion = IDEmbedCriterion(config)

    # Build Optimizer:
    # DETR_FREEZE: detector fully frozen (eval mode, no_grad forward, no DETR losses — only the
    # last-layer matching the ID loss needs). Implies DETR_NUM_TRAIN_FRAMES = 0.
    detr_freeze = config.get("DETR_FREEZE", False)
    if detr_freeze:
        assert not config["ONLY_DETR"], "DETR_FREEZE with ONLY_DETR leaves nothing to train."
        config["DETR_NUM_TRAIN_FRAMES"] = 0
    if config["DETR_NUM_TRAIN_FRAMES"] == 0:
        for n, p in model.named_parameters():
            if n.startswith("detr."):
                p.requires_grad = False     # only train the MOTIP part.
    _non_detr_named_detr = [n for n, _ in model.named_parameters() if "detr" in n and not n.startswith("detr.")]
    assert len(_non_detr_named_detr) == 0, f"Params outside the detector contain 'detr': {_non_detr_named_detr[:5]}"
    _n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Trainable params: {_n_trainable:,}, frozen params: {_n_frozen:,} (DETR_FREEZE={detr_freeze}).")
    if config.get("SPATIAL_ANCHOR", False):
        # The anchor reuses the detector's ref_point_head; with a frozen detector it must stay frozen.
        _head = [p for n, p in model.named_parameters() if "ref_point_head" in n]
        assert len(_head) > 0, "SPATIAL_ANCHOR: no ref_point_head found in the detector."
        _head_trainable = any(p.requires_grad for p in _head)
        logger.info(f"SPATIAL_ANCHOR on: ref_point_head {sum(p.numel() for p in _head):,} params, trainable={_head_trainable}.")
        if detr_freeze:
            assert not _head_trainable, "DETR_FREEZE must keep the ref_point_head used by SPATIAL_ANCHOR frozen."
    if config.get("RINK_ANCHOR", False):
        _rink = [(n, p) for n, p in model.named_parameters() if n.startswith("rink_anchor.")]
        assert len(_rink) > 0, "RINK_ANCHOR: the model has no rink_anchor parameters."
        assert all(p.requires_grad for _, p in _rink), "RINK_ANCHOR parameters must be trainable."
        logger.info(
            f"RINK_ANCHOR on: {sum(p.numel() for _, p in _rink):,} params, dropout={config.get('RINK_ANCHOR_DROPOUT', 0.0)}, "
            f"lr x{config.get('LR_RINK_ANCHOR_SCALE', 1.0)}."
        )
    if config.get("TEAM_CUE", False):
        _cue = [(n, p) for n, p in model.named_parameters() if n.startswith("team_cue.")]
        assert len(_cue) > 0, "TEAM_CUE: the model has no team_cue parameters."
        assert all(p.requires_grad for _, p in _cue), "TEAM_CUE parameters must be trainable."
        assert detr_cache is not None and config.get("TEAM_CUE", False), "TEAM_CUE training needs DETR_CACHE_DIR built with --kit (the kit descriptors)."
        _stats_path = config.get("KIT_STATS_PATH") or os.path.join(config["DETR_CACHE_DIR"], "kit_stats.json")
        import json
        with open(_stats_path) as _fh:
            _stats = json.load(_fh)
        model.team_cue.set_stats(_stats["mean"], _stats["std"])
        logger.info(
            f"TEAM_CUE on: {sum(p.numel() for _, p in _cue):,} params, dropout={config.get('TEAM_CUE_DROPOUT', 0.1)}, "
            f"gate init={config.get('TEAM_CUE_GATE_INIT', 0.25)}, lr x{config.get('LR_TEAM_CUE_SCALE', 1.0)}, "
            f"kit stats from {_stats_path} ({_stats['n_rows']:,} rows)."
        )
    if config.get("RINK_DIST_BIAS", False):
        _rdb = [(n, p) for n, p in model.named_parameters() if ".rink_dist_bias." in n]
        assert len(_rdb) > 0 and all(p.requires_grad for _, p in _rdb), "RINK_DIST_BIAS: no trainable rink_dist_bias parameters."
        logger.info(f"RINK_DIST_BIAS on: {sum(p.numel() for _, p in _rdb):,} params, lr x{config.get('LR_RINK_DIST_BIAS_SCALE', 1.0)}.")
    if config.get("ID_EMBED_LOSS", False):
        _head = [(n, p) for n, p in model.named_parameters() if n.startswith("id_embed_head.")]
        assert len(_head) > 0, "ID_EMBED_LOSS: the model has no id_embed_head."
        _detr_trainable = any(p.requires_grad for n, p in model.named_parameters() if n.startswith("detr."))
        logger.info(
            f"ID_EMBED_LOSS on: head {sum(p.numel() for _, p in _head):,} params, weight {id_embed_criterion.weight}, "
            f"temp {id_embed_criterion.temp}, hard x{id_embed_criterion.hard_weight} (IoU > {id_embed_criterion.hard_iou} or "
            f"< {id_embed_criterion.hard_rink_m} m within {id_embed_criterion.hard_frames} steps), team={id_embed_criterion.team}, "
            f"detector trainable={_detr_trainable} (DETR_NUM_TRAIN_FRAMES={config['DETR_NUM_TRAIN_FRAMES']}).")
    # Record the size of the added cue terms (kit, box anchor) on every ID-decoder forward and log them (cue_* metrics).
    model.record_cue_stats = bool(config.get("TEAM_CUE", False) or config.get("SPATIAL_ANCHOR", False)
                                  or config.get("RINK_ANCHOR", False))
    param_groups = get_param_groups(model, config)
    optimizer = AdamW(
        params=param_groups,
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    lr_scheduler_type = config.get("LR_SCHEDULER", "multistep")
    if lr_scheduler_type == "cosine":
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=cosine_lr_lambda(
            warmup_epochs=config["LR_WARMUP_EPOCHS"], total_epochs=config["EPOCHS"],
            min_lr_ratio=config.get("LR_MIN_RATIO", 0.01),
        ))
    else:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=config["SCHEDULER_MILESTONES"],
            gamma=config["SCHEDULER_GAMMA"],
        )

    # Other infos:
    only_detr = config["ONLY_DETR"]

    # Resuming:
    if config["RESUME_MODEL"] is not None:
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states,
        )
        # Different processing on scheduler (and the RESUME_STATES reset, which has to come first):
        position_scheduler_after_resume(
            scheduler=scheduler, train_states=train_states,
            resume_scheduler=config["RESUME_SCHEDULER"], resume_states=config.get("RESUME_STATES", True),
        )
        logger.success(
            log=f"Resume the model from '{config['RESUME_MODEL']}', "
                f"optimizer={config['RESUME_OPTIMIZER']}, "
                f"scheduler={config['RESUME_SCHEDULER']}, "
                f"states={train_states}. "
                f"Start from epoch {train_states['start_epoch']}, step {train_states['global_step']}."
        )

    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer,
        # device_placement=[False]        # whether to place the data on the device
    )

    # Validation (decoupled from checkpoint saving):
    eval_per_epoch = config.get("EVAL_PER_EPOCH", config["SAVE_CHECKPOINT_PER_EPOCH"])
    best_val = {"HOTA": -1.0, "AssA": -1.0, "epoch": None}

    def run_eval(eval_tag: str):
        eval_metrics = submit_and_evaluate_one_model(
            is_evaluate=True,
            accelerator=accelerator,
            state=state,
            logger=logger,
            model=model,
            data_root=config["DATA_ROOT"],
            dataset=config["INFERENCE_DATASET"],
            data_split=config["INFERENCE_SPLIT"],
            outputs_dir=os.path.join(outputs_dir, "train", "eval_during_train", eval_tag),
            image_max_longer=config["INFERENCE_MAX_LONGER"],
            size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
            miss_tolerance=config["MISS_TOLERANCE"],
            use_sigmoid=config["USE_FOCAL_LOSS"] if "USE_FOCAL_LOSS" in config else False,
            assignment_protocol=config["ASSIGNMENT_PROTOCOL"] if "ASSIGNMENT_PROTOCOL" in config else "hungarian",
            det_thresh=config["DET_THRESH"],
            newborn_thresh=config["NEWBORN_THRESH"],
            id_thresh=config["ID_THRESH"],
            area_thresh=config["AREA_THRESH"],
            inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
            else config["ONLY_DETR"],
            max_sequences=config.get("INFERENCE_MAX_SEQS"),
            spatial_gate=config.get("SPATIAL_GATE", False),
            gate_base=config.get("GATE_BASE", 0.5),
            gate_per_step=config.get("GATE_PER_STEP", 0.2),
            rink_anchor=config.get("RINK_ANCHOR", False) or config.get("RINK_DIST_BIAS", False),
        )
        eval_metrics.sync()
        return eval_metrics

    do_eval = config["INFERENCE_DATASET"] is not None
    if do_eval:
        assert config["INFERENCE_SPLIT"] is not None, f"Please set the INFERENCE_SPLIT for inference."
    if do_eval and config.get("EVAL_BEFORE_TRAIN", False):
        # Baseline of the starting checkpoint — the number the fine-tune has to beat.
        eval_metrics = run_eval("before_train")
        logger.metrics(
            log=f"[Eval before training] ",
            metrics=eval_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_step"],
            prefix="val",
        )

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        logger.info(log=f"Start training epoch {epoch}. LR (group 0 / last): "
                        f"{optimizer.param_groups[0]['lr']:.6e} / {optimizer.param_groups[-1]['lr']:.6e}.")
        epoch_start_timestamp = TPS.timestamp()
        # Prepare the sampler for the current epoch:
        # Update AUG_TRAJECTORY_SWITCH_PROB if a schedule is configured:
        switch_prob_schedule = config.get('AUG_TRAJECTORY_SWITCH_PROB_SCHEDULE', None)
        if switch_prob_schedule is not None:
            schedule = sorted(switch_prob_schedule, key=lambda x: x[0])
            if epoch <= schedule[0][0]:
                switch_prob = schedule[0][1]
            elif epoch >= schedule[-1][0]:
                switch_prob = schedule[-1][1]
            else:
                for i in range(len(schedule) - 1):
                    e0, p0 = schedule[i]
                    e1, p1 = schedule[i + 1]
                    if e0 <= epoch < e1:
                        switch_prob = p0 + (p1 - p0) * (epoch - e0) / (e1 - e0)
                        break
            train_dataset.set_aug_trajectory_switch_prob(switch_prob)
            logger.info(f"AUG_TRAJECTORY_SWITCH_PROB set to {switch_prob:.4f} for epoch {epoch}.")
        train_sampler.prepare_for_epoch(epoch=epoch)
        _st = train_sampler.last_epoch_stats
        logger.info(
            f"Sampler epoch {epoch}: {_st['samples']} samples of length {_st['sample_length']} from {_st['sequences']} sequences; "
            f"{_st['short']} shorter than the window ({_st['short_kept']} kept whole, {_st['short'] - _st['short_kept']} dropped), "
            f"{_st['no_window']} long enough but with no legal window."
        )
        if "tags" in _st:
            logger.info(f"Manifest epoch {epoch}: windows per tag {_st['tags']}, window-frames per tag {_st['window_frames']}, "
                        f"candidates {_st['candidates']}, with replacement {_st['replacement']}.")
        # Train one epoch:
        train_metrics = train_one_epoch(
            accelerator=accelerator,
            logger=logger,
            states=train_states,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            detr_criterion=detr_criterion,
            id_criterion=id_criterion,
            optimizer=optimizer,
            only_detr=only_detr,
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            lr_warmup_tgt_lr=config["LR"],
            detr_num_train_frames=config["DETR_NUM_TRAIN_FRAMES"],
            detr_num_checkpoint_frames=config["DETR_NUM_CHECKPOINT_FRAMES"],
            detr_criterion_batch_len=config.get("DETR_CRITERION_BATCH_LEN", 10),
            use_decoder_checkpoint=config["USE_DECODER_CHECKPOINT"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            separate_clip_norm=config.get("SEPARATE_CLIP_NORM", True),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            # For multi last checkpoints:
            outputs_dir=outputs_dir,
            is_last_epochs=(epoch == config["EPOCHS"] - 1),
            multi_last_checkpoints=config["MULTI_LAST_CHECKPOINTS"],
            detr_freeze=detr_freeze,
            detr_frozen_autocast=config.get("DETR_FROZEN_AUTOCAST", "none"),
            log_switch_stats=(
                config.get("AUG_TRAJECTORY_SWITCH_PROB", 0.0) > 0.0
                or config.get("AUG_TRAJECTORY_SWITCH_PROB_SCHEDULE") is not None
            ),
            rink_anchor=_rink_mod.needs_rink_data(config),
            detr_cache=detr_cache,
            id_embed_criterion=id_embed_criterion,
            id_embed_log_detr_grad_every=int(config.get("ID_EMBED_LOG_DETR_GRAD_EVERY", 0)),
        )

        _cue_line = {k: round(float(train_metrics.metrics[k].global_average), 4) for k in list(train_metrics.metrics.keys()) if str(k).startswith("cue_")}
        if _cue_line:
            logger.info(f"CUE-CHECK epoch {epoch} (mean over the epoch): {_cue_line}")
        # Get learning rate:
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["lr"].update(lr)
        train_metrics["lr"].sync()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.metrics(
            log=f"[Finish epoch: {epoch}] [Time: {time_per_epoch}] ",
            metrics=train_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_step"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch",
        )

        # Save checkpoint:
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_checkpoint(
                model=model,
                path=os.path.join(outputs_dir, f"checkpoint_{epoch}.pth"),
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler,
                only_detr=only_detr,
            )

        # Validation every EVAL_PER_EPOCH epochs (and always on the last epoch):
        is_last_epoch = epoch == config["EPOCHS"] - 1
        if do_eval and ((epoch + 1) % eval_per_epoch == 0 or is_last_epoch):
            eval_metrics = run_eval(f"epoch_{epoch}")
            logger.metrics(
                log=f"[Eval epoch: {epoch}] ",
                metrics=eval_metrics,
                fmt="{global_average:.4f}",
                statistic="global_average",
                global_step=train_states["global_step"],
                prefix="val",
                x_axis_step=epoch,
                x_axis_name="epoch",
            )
            _hota = eval_metrics["HOTA"].global_average
            _assa = eval_metrics["AssA"].global_average
            if (_hota, _assa) > (best_val["HOTA"], best_val["AssA"]):
                best_val.update({"HOTA": _hota, "AssA": _assa, "epoch": epoch})
                save_checkpoint(
                    model=model,
                    path=os.path.join(outputs_dir, "checkpoint_best.pth"),
                    states=train_states,
                    optimizer=None,
                    scheduler=None,
                    only_detr=only_detr,
                )
                logger.success(log=f"New best val HOTA {_hota:.4f} (AssA {_assa:.4f}) at epoch {epoch}.")
                logger.set_tags({"best_epoch": epoch, "best_val_HOTA": f"{_hota:.4f}"})

        logger.success(log=f"Finish training epoch {epoch}.")
        # Prepare for next step:
        scheduler.step()
    if best_val["epoch"] is not None:
        logger.success(log=f"Best val HOTA {best_val['HOTA']:.4f} (AssA {best_val['AssA']:.4f}) at epoch {best_val['epoch']}.")
    logger.finish(status="FINISHED")
    pass


def train_one_epoch(
        # Infos:
        accelerator: Accelerator,
        logger: Logger,
        states: dict,
        epoch: int,
        dataloader: DataLoader,
        model,
        detr_criterion,
        id_criterion,
        optimizer,
        only_detr,
        lr_warmup_epochs: int,
        lr_warmup_tgt_lr: float,
        detr_num_train_frames: int,
        detr_num_checkpoint_frames: int,
        detr_criterion_batch_len: int,
        use_decoder_checkpoint: bool,
        accumulate_steps: int = 1,
        separate_clip_norm: bool = True,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
        # For multi last checkpoints:
        outputs_dir: str = None,
        is_last_epochs: bool = False,
        multi_last_checkpoints: int = 0,
        # Frozen detector:
        detr_freeze: bool = False,
        detr_frozen_autocast: str = "none",
        # Log the realised ID-switch statistics (swap rate, far-swap fraction, mean GIoU of swaps):
        log_switch_stats: bool = False,
        # Rink anchor: batch the per-frame projection inputs from the metas and pass them to prepare_for_motip:
        rink_anchor: bool = False,
        # DETR_CACHE_DIR: a utils.detr_cache.DetrCache of the frozen detector's outputs (skips the detector forward).
        detr_cache=None,
        # ID_EMBED_LOSS: models.motip.id_embed_loss.IDEmbedCriterion (None = off); every N steps also log the norm of its
        # gradient on the detector parameters (0 = never; an extra autograd.grad, for smoke tests).
        id_embed_criterion=None,
        id_embed_log_detr_grad_every: int = 0,
):
    current_last_checkpoint_idx = 0
    if id_embed_log_detr_grad_every > 0 and accelerator.num_processes > 1:
        # An extra torch.autograd.grad(retain_graph=True) inside a DDP step is unsupported by DDP: smoke tests only.
        logger.warning("ID_EMBED_LOG_DETR_GRAD_EVERY > 0 is single-process only (autograd.grad inside DDP); disabled.")
        id_embed_log_detr_grad_every = 0
    assert detr_cache is None or detr_freeze, "DETR_CACHE_DIR needs DETR_FREEZE: a trained detector's outputs change."

    model.train()
    if detr_freeze:
        get_model(model).detr.eval()
    _frozen_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(detr_frozen_autocast).lower())
    assert _frozen_dtype is not None or str(detr_frozen_autocast).lower() in ("none", "fp32"), \
        f"Unknown DETR_FROZEN_AUTOCAST '{detr_frozen_autocast}', expected bf16/fp16/none."
    tps = TPS()     # time per step
    metrics = Metrics()
    optimizer.zero_grad()
    step_timestamp = tps.timestamp()
    device = accelerator.device
    _B = dataloader.batch_sampler.batch_size
    _num_gts_per_frame = 0

    # Prepare for gradient clip norm:
    model_without_ddp = get_model(model)
    detr_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if "detr" in name:
            detr_params.append(param)
        else:
            other_params.append(param)

    for step, samples in enumerate(dataloader):
        images, annotations, metas = samples["images"], samples["annotations"], samples["metas"]
        if detr_cache is None:
            # Normalize the images:
            # (Normally, it should be done in the dataloader, but here we do it in the training loop (on cuda).)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            images.tensors = v2.functional.to_dtype(images.tensors, dtype=torch.float32, scale=True)
            images.tensors = v2.functional.normalize(images.tensors, mean=mean, std=std)
            # A hack implementation to recover 0.0 in the masked regions:
            images.tensors = images.tensors * (~images.mask[:, :, None, ...]).to(torch.float32)
            images.tensors = images.tensors.contiguous()

        # Learning rate warmup:
        if epoch < lr_warmup_epochs:
            # Do warmup:
            lr_warmup(
                optimizer=optimizer,
                epoch=epoch, curr_iter=step, tgt_lr=lr_warmup_tgt_lr,
                warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=len(dataloader),
            )

        if detr_cache is None:
            _B, _T = len(annotations), len(annotations[0])
            detr_num_train_frames = min(detr_num_train_frames, _T)

            # Prepare the DETR targets from the annotations:
            detr_targets_flatten = annotations_to_flatten_detr_targets(annotations=annotations, device=device)

            # Select the training and no_grad frames:
            random_frame_idxs = torch.randperm(_T, device=device)   # use these random indices to select the frames.
            go_back_frame_idxs = torch.argsort(random_frame_idxs)   # use these indices to go back to the original order.
            go_back_frame_idxs_flatten = torch.cat([
                go_back_frame_idxs + _T * b for b in range(_B)
            ])      # only used for the DETR's criterion.
            # Split random_frame_idxs into training and no_grad frame indices:
            detr_train_frame_idxs = random_frame_idxs[:detr_num_train_frames]
            detr_no_grad_frame_idxs = random_frame_idxs[detr_num_train_frames:]

            detr_outputs_flatten_idxs = torch.arange(_B * _T, device=device)
            detr_outputs_flatten_idxs = einops.rearrange(detr_outputs_flatten_idxs, "(b t) -> b t", b=_B)
            detr_outputs_flatten_idxs = torch.cat([
                einops.rearrange(detr_outputs_flatten_idxs[:, :detr_num_train_frames], "b t -> (b t)"),
                einops.rearrange(detr_outputs_flatten_idxs[:, detr_num_train_frames:], "b t -> (b t)"),
            ], dim=0)
            detr_outputs_flatten_go_back_idxs = torch.argsort(detr_outputs_flatten_idxs)
            pass
            # Select the training and no_grad frames:
            detr_train_frames = nested_tensor_index_select(images, dim=1, index=detr_train_frame_idxs)
            detr_no_grad_frames = nested_tensor_index_select(images, dim=1, index=detr_no_grad_frame_idxs)

            # Prepare for the DETR forward function, turn the (B, T, ...) images to (B*T, ...) (or said flatten):
            detr_train_frames.tensors = einops.rearrange(detr_train_frames.tensors, "b t c h w -> (b t) c h w").contiguous()
            detr_train_frames.mask = einops.rearrange(detr_train_frames.mask, "b t h w -> (b t) h w").contiguous()
            detr_no_grad_frames.tensors = einops.rearrange(detr_no_grad_frames.tensors, "b t c h w -> (b t) c h w").contiguous()
            detr_no_grad_frames.mask = einops.rearrange(detr_no_grad_frames.mask, "b t h w -> (b t) h w").contiguous()

            # TODO: For DeNoise (e.g., in DINO-DETR),
            #       need to split the detr_targets_flatten into training and no_grad parts.

            # DETR forward:
            # 1. no_grad frames:
            if _T > detr_num_train_frames:      # do have no_grad frames (if not, skip this part)
                _frozen_autocast = torch.autocast(device_type="cuda", dtype=_frozen_dtype) \
                    if (detr_freeze and _frozen_dtype is not None) else contextlib.nullcontext()
                with torch.no_grad(), _frozen_autocast:
                    if detr_num_checkpoint_frames == 0 or detr_num_checkpoint_frames * 4 >= len(detr_no_grad_frames):
                        # Directly forward the no_grad frames:
                        detr_no_grad_outputs = model(frames=detr_no_grad_frames, part="detr")
                    else:
                        # Split the no_grad frames into batched iterations (reduce the memory usage):
                        detr_no_grad_outputs = None
                        for batch_samples in batch_iterator(
                            detr_num_checkpoint_frames * 4,
                            detr_no_grad_frames,
                        ):
                            batch_frames = batch_samples[0]
                            _ = model(frames=batch_frames, part="detr")
                            detr_no_grad_outputs = tensor_dict_cat(detr_no_grad_outputs, _, dim=0)
            else:                               # no no_grad frames
                detr_no_grad_outputs = None

            # 2. training frames:
            if detr_num_train_frames > 0:
                if detr_num_checkpoint_frames == 0 or detr_num_checkpoint_frames >= len(detr_train_frames):
                    # Directly forward the training frames:
                    detr_train_outputs = model(frames=detr_train_frames, part="detr")
                else:
                    # Split the training frames into batched iterations (reduce the memory usage):
                    detr_train_outputs = None
                    for batch_samples in batch_iterator(
                        detr_num_checkpoint_frames,
                        detr_train_frames,
                    ):
                        batch_frames = batch_samples[0]
                        _ = model(frames=batch_frames, part="detr", use_checkpoint=True)
                        detr_train_outputs = tensor_dict_cat(detr_train_outputs, _, dim=0)
            else:
                detr_train_outputs = None

            # Combine training and no_grad outputs:
            detr_outputs = tensor_dict_cat(detr_train_outputs, detr_no_grad_outputs, dim=0)
            # Recover the order of the outputs:
            detr_outputs = tensor_dict_index_select(detr_outputs, index=detr_outputs_flatten_go_back_idxs, dim=0)
            detr_outputs = tensor_dict_index_select(detr_outputs, index=go_back_frame_idxs_flatten, dim=0)

            # DETR criterion:
            if detr_freeze:
                # Frozen detector: no DETR losses, only the last-layer matching the ID loss needs.
                if _frozen_dtype is not None:
                    detr_outputs = tensor_dict_to_float32(detr_outputs)
                detr_indices = detr_criterion.match_last_layer(
                    outputs=detr_outputs, targets=detr_targets_flatten, batch_len=detr_criterion_batch_len,
                )
                detr_loss_dict = {}
            else:
                detr_loss_dict, detr_indices = detr_criterion(outputs=detr_outputs, targets=detr_targets_flatten, batch_len=detr_criterion_batch_len)

        else:
            # DETR_CACHE_DIR: the frozen detector's matched queries come from the cache (utils/detr_cache.py); the images
            # are placeholders and no detector forward runs. Identity indices: row i is the query matched to GT object i.
            detr_outputs, detr_indices = detr_cache.detr_outputs(metas, annotations, device)
            detr_loss_dict = {}

        # Whether to only train the DETR, OR to train the MOTIP together:
        if not only_detr:
            _G, _, _N = annotations[0][0]["trajectory_id_labels"].shape
            # Need to prepare for MOTIP:
            _rink_batch = rink_batch_from_metas(metas) if rink_anchor else None
            seq_info = prepare_for_motip(
                detr_outputs=detr_outputs, annotations=annotations, detr_indices=detr_indices,
                with_gt_boxes=id_criterion.use_spatial_loss,
                rink=_rink_batch,
            )
            if rink_anchor:
                _tok_valid = torch.isfinite(seq_info["unknown_rink_xy"]).all(dim=-1)[~seq_info["unknown_masks"]]
                metrics.update(name="rink_valid_frac", value=_tok_valid.float().mean().item() if _tok_valid.numel() else 1.0)
            seq_info = model(seq_info=seq_info, part="trajectory_modeling")
            _embed_recs, _embed_flat = (id_embed_criterion.gather(detr_outputs, detr_indices, annotations, rink=_rink_batch, metas=metas)
                                        if id_embed_criterion is not None else (None, None))
            _dec_out = model(
                seq_info=seq_info,
                part="id_decoder",
                use_decoder_checkpoint=use_decoder_checkpoint,
                id_embed_inputs=_embed_flat,
            )
            id_logits, id_gts, id_masks = _dec_out[:3]
            id_loss = id_criterion(id_logits=id_logits, id_labels=id_gts, id_masks=id_masks)
            if model_without_ddp.record_cue_stats:
                for _k, _v in model_without_ddp.take_cue_stats().items():
                    metrics.update(name=f"cue_{_k}", value=_v)
            id_loss_parts = {}
            if id_criterion.use_spatial_loss:
                # MOTIP2 Spatial ID Loss, added to the CE (the CE itself is unchanged):
                id_loss_parts["id_ce"] = id_loss
                id_loss_parts["id_l1"], id_loss_parts["id_giou"] = id_criterion.spatial_loss(
                    id_logits=id_logits,
                    id_masks=id_masks,
                    slot_labels=seq_info["trajectory_id_labels"],
                    pred_boxes=seq_info["unknown_boxes"],
                    gt_boxes=seq_info["unknown_gt_boxes"],
                )
                id_loss = id_loss \
                    + id_criterion.l1_weight * id_loss_parts["id_l1"] \
                    + id_criterion.giou_weight * id_loss_parts["id_giou"]
            if id_embed_criterion is not None:
                _embed_terms, _embed_stats = id_embed_criterion.loss(_embed_recs, _dec_out[3], device)
                id_embed_total = _embed_terms.pop("total")
                for _k, _v in _embed_terms.items():
                    id_loss_parts[_k] = _v
                for _k, _v in _embed_stats.items():
                    # Same metric names on every rank whatever the batch (a NaN stat, e.g. no hard negative in this
                    # rank's window, is registered empty instead of skipped; see log.Metrics.sync):
                    if _v == _v:
                        metrics.update(name=_k, value=_v)
                    else:
                        metrics.touch(_k)
                if id_embed_log_detr_grad_every > 0 and step % id_embed_log_detr_grad_every == 0 and len(detr_params) > 0:
                    _g = torch.autograd.grad(id_embed_total, detr_params, retain_graph=True, allow_unused=True)
                    _gn = torch.sqrt(sum((g.float() ** 2).sum() for g in _g if g is not None) + 0.0)
                    metrics.update(name="id_embed_detr_grad_norm", value=float(_gn))
                    metrics.update(name="id_embed_detr_grad_params", value=float(sum(g is not None and bool(g.abs().sum() > 0) for g in _g)))
            else:
                id_embed_total = None
            _num_gts_per_frame = max(_num_gts_per_frame, id_gts.shape[-1])
            # print(f"Num of GTs per frame: {_num_gts_per_frame}")
            pass
        else:
            id_loss = None
            id_loss_parts = {}
            id_embed_total = None

        # Backward:
        with accelerator.autocast():
            detr_weight_dict = detr_criterion.weight_dict
            detr_loss = sum(
                (detr_loss_dict[k] * detr_weight_dict[k] for k in detr_loss_dict.keys() if k in detr_weight_dict),
                torch.zeros((), device=device),
            )
            loss = detr_loss + (id_loss if id_loss is not None else 0) * id_criterion.weight
            if id_embed_total is not None:
                loss = loss + id_embed_total
            # Non-finite batch guard, rank-invariant: one all-reduce on every rank every step; if the loss is non-finite on
            # any rank, every rank zero-weights its loss (same backward, same DDP gradient all-reduce) and drops the
            # gradients before the optimizer step, so no rank skips a collective and no NaN reaches the weights.
            nonfinite_skip = nonfinite_loss_on_any_rank(loss, accelerator=accelerator, device=device)
            # Logging losses (a non-finite value only registers its name: same names on every rank, no NaN in the means):
            update_or_touch(metrics, "loss", loss.item())
            update_or_touch(metrics, "detr_loss", detr_loss.item())
            if id_loss is not None:
                update_or_touch(metrics, "id_loss", id_loss.item())
            for k, v in id_loss_parts.items():
                update_or_touch(metrics, k, v.item())
            for k, v in detr_loss_dict.items():
                update_or_touch(metrics, k, v.item())
            # Guard counters (both matchers, events this step) + the skip flag, recorded on every rank every step.
            metrics.update(name="nonfinite_loss_skip", value=float(nonfinite_skip))
            for k, v in take_match_guard_events().items():
                metrics.update(name=k, value=float(v))
            if nonfinite_skip:
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
            if log_switch_stats:
                _switch = empty_switch_stats()
                for _clip in annotations:
                    add_switch_stats(_switch, clip_switch_stats(_clip))
                _switch = summarize_switch_stats(_switch)
                metrics.update(name="switch_swap_rate", value=_switch["swap_rate"])
                metrics.update(name="switch_far_frac", value=_switch["far_frac"])
                metrics.update(name="switch_mean_giou", value=_switch["mean_giou"])
            loss /= accumulate_steps
            accelerator.backward(loss)  # use this line to replace loss.backward()
            if (step + 1) % accumulate_steps == 0:
                if nonfinite_skip:
                    optimizer.zero_grad()     # every rank: the gradients of a skipped batch never reach the weights
                if use_accelerate_clip_norm:
                    if separate_clip_norm:
                        detr_grad_norm = accelerator.clip_grad_norm_(detr_params, max_norm=max_clip_norm) \
                            if len(detr_params) > 0 else torch.zeros((), device=device)
                        other_grad_norm = accelerator.clip_grad_norm_(other_params, max_norm=max_clip_norm)
                    else:
                        detr_grad_norm = other_grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
                else:
                    if separate_clip_norm:
                        accelerator.unscale_gradients()
                        detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, max_clip_norm) \
                            if len(detr_params) > 0 else torch.zeros((), device=device)
                        other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, max_clip_norm)
                    else:
                        accelerator.unscale_gradients()
                        detr_grad_norm = other_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
                # Hack implementation to log grad_norm
                update_or_touch(metrics, "detr_grad_norm", detr_grad_norm.item())
                update_or_touch(metrics, "other_grad_norm", other_grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()

        # Logging:
        tps.update(tps=tps.timestamp() - step_timestamp)
        step_timestamp = tps.timestamp()
        # Logging:
        if step % logging_interval == 0:
            # logger.info(f"[Epoch: {epoch}] [{step}/{total_steps}] [tps: {tps.average:.2f}s]")
            # Get learning rate for current step:
            _lr = optimizer.state_dict()["param_groups"][-1]["lr"]
            # Get the GPU memory usage:
            torch.cuda.synchronize()
            _cuda_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            _cuda_memory = torch.tensor([_cuda_memory], device=device)
            # _cuda_memory_reduce = accelerator.reduce(_cuda_memory, reduction="none")
            _gathered_cuda_memory = accelerator.gather(_cuda_memory)
            _max_cuda_memory = _gathered_cuda_memory.max().item()
            accelerator.wait_for_everyone()
            # Clear some values:
            metrics["lr"].clear()  # clear the learning rate value from last step
            metrics["max_cuda_mem(MB)"].clear()
            # Update them to the metrics:
            metrics.update(name="lr", value=_lr)
            metrics.update(name="max_cuda_mem(MB)", value=_max_cuda_memory)
            # Sync the metrics:
            metrics.sync()
            eta = tps.eta(total_steps=len(dataloader), current_steps=step)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{step}/{len(dataloader)}] "
                    f"[tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )
        # For multi last checkpoints:
        if is_last_epochs and multi_last_checkpoints > 0:
            if (step + 1) == int(math.ceil((len(dataloader) / multi_last_checkpoints) * (current_last_checkpoint_idx + 1))):
                _dir = os.path.join(outputs_dir, "multi_last_checkpoints")
                os.makedirs(_dir, exist_ok=True)
                save_checkpoint(
                    model=model,
                    path=os.path.join(_dir, f"last_checkpoint_{current_last_checkpoint_idx}.pth"),
                    states=states,
                    optimizer=None,
                    scheduler=None,
                    only_detr=only_detr,
                )
                logger.info(
                    log=f"Save the last checkpoint {current_last_checkpoint_idx} at step {step}."
                )
                current_last_checkpoint_idx += 1
        # Update the counters:
        states["global_step"] += 1
    states["start_epoch"] += 1
    return metrics


def _rfdetr_dinov2_lr_decay(name, lr_decay_rate, num_layers):
    # Per-layer LR decay for DINOv2 (mirrors RF-DETR get_dinov2_lr_decay_rate).
    # MOTIP wraps RF-DETR so backbone params are named "detr.backbone.0.encoder.encoder.layer.N...."
    layer_id = num_layers + 1  # non-backbone: decay^0 = 1.0
    if "detr.backbone.0.encoder" in name:
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name:
            try:
                after = name[name.find(".layer."):]
                layer_id = int(after.split(".")[2]) + 1
            except (ValueError, IndexError):
                pass
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _rfdetr_dinov2_wd(name, base_wd):
    # Zero WD for norms/biases/embeddings (mirrors RF-DETR get_dinov2_weight_decay_rate).
    if any(k in name for k in ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")):
        return 0.0
    return base_wd


def get_param_groups(model, config) -> list[dict]:
    def _match_names(_name, _key_names):
        for _k in _key_names:
            if _k in _name:
                return True
        return False

    # RF-DETR: per-layer ViT LR decay matching RF-DETR's own get_param_dict.
    # Gated on DETR_FRAMEWORK so D-DETR training is unaffected.
    if config.get("DETR_FRAMEWORK") == "rf_detr":
        base_lr = config["LR"]
        # lr_encoder defaults to 1.5x base (RF-DETR default ratio: 1.5e-4 / 1e-4)
        lr_encoder = config.get("LR_ENCODER", base_lr * 1.5)
        lr_vit_layer_decay = config.get("LR_VIT_LAYER_DECAY", 0.8)
        lr_component_decay = config.get("LR_COMPONENT_DECAY", 1.0)
        weight_decay = config["WEIGHT_DECAY"]
        # num_layers = out_feature_indexes[-1] + 1; [3,6,9,12] -> 13
        num_layers = config.get("RFDETR_OUT_FEATURE_INDEXES", [3, 6, 9, 12])[-1] + 1

        backbone_enc_key = "detr.backbone.0.encoder"
        decoder_key = "detr.transformer.decoder"
        linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
        linear_proj_scale = config["LR_LINEAR_PROJ_SCALE"]
        dictionary_names = config["LR_DICTIONARY_NAMES"]
        dictionary_scale = config["LR_DICTIONARY_SCALE"]

        backbone_set, decoder_set, linproj_set, dict_set = set(), set(), set(), set()
        param_groups = []

        # 1. Backbone encoder: per-layer decay; lr_scale set so lr_warmup scales correctly
        for n, p in model.named_parameters():
            if backbone_enc_key in n and p.requires_grad:
                layer_decay = _rfdetr_dinov2_lr_decay(n, lr_vit_layer_decay, num_layers)
                lr = lr_encoder * layer_decay * (lr_component_decay ** 2)
                lr_scale = lr / base_lr
                wd = _rfdetr_dinov2_wd(n, weight_decay)
                param_groups.append({"params": p, "lr": lr, "lr_scale": lr_scale, "weight_decay": wd})
                backbone_set.add(n)

        # 2. Decoder: full LR * component_decay
        for n, p in model.named_parameters():
            if decoder_key in n and p.requires_grad:
                lr = base_lr * lr_component_decay
                lr_scale = lr_component_decay
                param_groups.append({"params": p, "lr": lr, "lr_scale": lr_scale})
                decoder_set.add(n)

        # 3. Linear proj params (sampling offsets, reference points, etc.)
        for n, p in model.named_parameters():
            if _match_names(n, linear_proj_names) and n not in backbone_set and n not in decoder_set and p.requires_grad:
                param_groups.append({"params": p, "lr": base_lr * linear_proj_scale, "lr_scale": linear_proj_scale})
                linproj_set.add(n)

        # 4. Dictionary params
        for n, p in model.named_parameters():
            if _match_names(n, dictionary_names) and n not in backbone_set and n not in decoder_set and n not in linproj_set and p.requires_grad:
                param_groups.append({"params": p, "lr": base_lr * dictionary_scale, "lr_scale": dictionary_scale})
                dict_set.add(n)

        # 4b. Spatial-anchor scale: own LR multiplier (a zero-init scale barely moves at a fine-tune LR), no WD.
        anchor_set = set()
        anchor_lr_scale = config.get("LR_ANCHOR_SCALE", 1.0)
        for n, p in model.named_parameters():
            if n.endswith("anchor_scale") and p.requires_grad:
                param_groups.append({"params": p, "lr": base_lr * anchor_lr_scale, "lr_scale": anchor_lr_scale, "weight_decay": 0.0})
                anchor_set.add(n)

        # 4c. Rink anchor (new randomly initialised MLP + zero-init last layer): own LR multiplier; weights keep the
        # base weight decay, biases and the "no position" embedding get none.
        rink_set = set()
        rink_lr_scale = config.get("LR_RINK_ANCHOR_SCALE", 1.0)
        for n, p in model.named_parameters():
            if n.startswith("rink_anchor.") and p.requires_grad:
                group = {"params": p, "lr": base_lr * rink_lr_scale, "lr_scale": rink_lr_scale}
                if n.endswith("bias") or n.endswith("no_position"):
                    group["weight_decay"] = 0.0
                if n.endswith("out_scale"):
                    # RINK_ANCHOR_OUT_SCALE_INIT: the per-channel output scale gets the image anchor's treatment
                    _s = config.get("LR_RINK_ANCHOR_OUT_SCALE", rink_lr_scale)
                    group = {"params": p, "lr": base_lr * _s, "lr_scale": _s, "weight_decay": 0.0}
                param_groups.append(group)
                rink_set.add(n)

        # 4d. Team cue (new MLP + scalar gate + "missing" vector): own LR multiplier; weights keep the base weight decay,
        # biases, the gate and the "missing" vector get none.
        team_set = set()
        team_lr_scale = config.get("LR_TEAM_CUE_SCALE", 1.0)
        for n, p in model.named_parameters():
            if n.startswith("team_cue.") and p.requires_grad:
                group = {"params": p, "lr": base_lr * team_lr_scale, "lr_scale": team_lr_scale}
                if n.endswith("bias") or n.endswith("missing") or n.endswith("gate"):
                    group["weight_decay"] = 0.0
                param_groups.append(group)
                team_set.add(n)

        # 4e. Look-ahead blocks of the ID decoder (zero-init output layers): own LR multiplier; biases, the time-bias table
        # and the future-token ID offset get no weight decay.
        la_set = set()
        la_lr_scale = config.get("LR_LOOKAHEAD_SCALE", 1.0)
        for n, p in model.named_parameters():
            if ".lookahead." in n and p.requires_grad:
                group = {"params": p, "lr": base_lr * la_lr_scale, "lr_scale": la_lr_scale}
                if n.endswith("bias") or n.endswith("time_bias") or n.endswith("future_id_delta"):
                    group["weight_decay"] = 0.0
                param_groups.append(group)
                la_set.add(n)

        # 4f. Rink-distance attention bias tables (zero-init): own LR multiplier, no weight decay.
        rdb_set = set()
        rdb_lr_scale = config.get("LR_RINK_DIST_BIAS_SCALE", 1.0)
        for n, p in model.named_parameters():
            if ".rink_dist_bias." in n and p.requires_grad:
                param_groups.append({"params": p, "lr": base_lr * rdb_lr_scale, "lr_scale": rdb_lr_scale, "weight_decay": 0.0})
                rdb_set.add(n)

        # 4g. Identity-loss projection head (new, randomly initialised): own LR multiplier; biases get no weight decay.
        head_set = set()
        head_lr_scale = config.get("LR_ID_EMBED_HEAD_SCALE", 1.0)
        for n, p in model.named_parameters():
            if n.startswith("id_embed_head.") and p.requires_grad:
                group = {"params": p, "lr": base_lr * head_lr_scale, "lr_scale": head_lr_scale}
                if n.endswith("bias"):
                    group["weight_decay"] = 0.0
                param_groups.append(group)
                head_set.add(n)

        # 5. Remaining params: full base LR
        all_special = backbone_set | decoder_set | linproj_set | dict_set | anchor_set | rink_set | team_set | la_set \
            | rdb_set | head_set
        for n, p in model.named_parameters():
            if n not in all_special and p.requires_grad:
                param_groups.append({"params": p, "lr": base_lr})

        return param_groups

    # Default (D-DETR / non-RF-DETR): original flat backbone scale logic
    backbone_names = config["LR_BACKBONE_NAMES"]
    linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
    dictionary_names = config["LR_DICTIONARY_NAMES"]
    pass
    # Param groups:
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, backbone_names) and p.requires_grad],
            "lr_scale": config["LR_BACKBONE_SCALE"],
            "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, linear_proj_names) and p.requires_grad],
            "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
            "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, dictionary_names) and p.requires_grad],
            "lr_scale": config["LR_DICTIONARY_SCALE"],
            "lr": config["LR"] * config["LR_DICTIONARY_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not _match_names(n, backbone_names)
                       and not _match_names(n, linear_proj_names)
                       and not _match_names(n, dictionary_names)
                       and p.requires_grad],
        }
    ]
    return param_groups


def update_or_touch(metrics, name: str, value: float):
    """metrics.update for a finite value; a non-finite one only registers the name (see log.Metrics.sync)."""
    if math.isfinite(value):
        metrics.update(name=name, value=value)
    else:
        metrics.touch(name)


def nonfinite_loss_on_any_rank(loss, accelerator, device) -> bool:
    """True on every rank if the loss is non-finite on any rank. Called by every rank at every step (one all-reduce)."""
    bad = torch.tensor([0.0 if bool(torch.isfinite(loss.detach()).all()) else 1.0], device=device)
    if accelerator.num_processes > 1:
        bad = accelerator.reduce(bad, reduction="sum")
    return bool(bad.item() > 0)


def cosine_lr_lambda(warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    """LambdaLR factor for epoch n (the scheduler is stepped once per epoch): 1 during warmup, then a cosine to min_lr_ratio."""
    def _cosine_lambda(n):
        epoch_for = n + 1
        if epoch_for <= warmup_epochs:
            return 1.0
        progress = (epoch_for - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(progress, 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return _cosine_lambda


def position_scheduler_after_resume(scheduler, train_states: dict, resume_scheduler: bool, resume_states: bool):
    """Put the per-epoch scheduler where the run starts, after load_checkpoint filled train_states from the checkpoint.

    RESUME_STATES False (a fresh fine-tune from a checkpoint): epoch and step restart at 0 and so does the schedule.
    This reset used to come after the scheduler was stepped start_epoch times with the *checkpoint's* start_epoch, so
    every v1x run (from stage-2 checkpoint_5, start_epoch 6) used the LR of epoch e + 6 at epoch e (2026-11-22 note).
    RESUME_SCHEDULER True (a real resume): the saved scheduler state is loaded, one step moves it to the next epoch.
    RESUME_SCHEDULER False with RESUME_STATES True: stepped start_epoch times, as before.
    """
    if not resume_states:
        train_states["start_epoch"] = 0
        train_states["global_step"] = 0
    if resume_scheduler:
        scheduler.step()
    else:
        for _ in range(0, train_states["start_epoch"]):
            scheduler.step()
    return


def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return


def annotations_to_flatten_detr_targets(annotations: list, device):
    """
    Args:
        annotations: annotations from the dataloader.
        device: move the targets to the device.

    Returns:
        A list of targets for the DETR model supervision, len=(B*T).
    """
    targets = []
    for annotation in annotations:      # scan by batch
        for ann in annotation:          # scan by frame
            targets.append(
                {
                    "boxes": ann["bbox"].to(device),
                    "labels": ann["category"].to(device),
                }
            )
    return targets


def nested_tensor_index_select(nested_tensor: NestedTensor, dim: int, index: torch.Tensor):
    tensors, mask = nested_tensor.decompose()
    _device = tensors.device
    index = index.to(_device)
    selected_tensors = torch.index_select(input=tensors, dim=dim, index=index).contiguous()
    selected_mask = torch.index_select(input=mask, dim=dim, index=index).contiguous()
    return NestedTensor(tensors=selected_tensors, mask=selected_mask)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]


def tensor_dict_cat(tensor_dict1, tensor_dict2, dim=0):
    if tensor_dict1 is None or tensor_dict2 is None:
        assert tensor_dict1 is not None or tensor_dict2 is not None, "One of the tensor dict should be not None."
        return tensor_dict1 if tensor_dict2 is None else tensor_dict2
    else:
        res_tensor_dict = defaultdict()
        for k in tensor_dict1.keys():
            if isinstance(tensor_dict1[k], torch.Tensor):
                res_tensor_dict[k] = torch.cat([tensor_dict1[k], tensor_dict2[k]], dim=dim)
            elif isinstance(tensor_dict1[k], dict):
                res_tensor_dict[k] = tensor_dict_cat(tensor_dict1[k], tensor_dict2[k], dim=dim)
            elif isinstance(tensor_dict1[k], list):
                assert len(tensor_dict1[k]) == len(tensor_dict2[k]), "The list should have the same length."
                res_tensor_dict[k] = [
                    tensor_dict_cat(tensor_dict1[k][_], tensor_dict2[k][_], dim=dim)
                    for _ in range(len(tensor_dict1[k]))
                ]
            else:
                raise ValueError(f"Unsupported type {type(tensor_dict1[k])} in the tensor dict concat.")
        return dict(res_tensor_dict)


def tensor_dict_to_float32(tensor_dict):
    res_tensor_dict = dict()
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            res_tensor_dict[k] = v.float() if v.is_floating_point() else v
        elif isinstance(v, dict):
            res_tensor_dict[k] = tensor_dict_to_float32(v)
        elif isinstance(v, list):
            res_tensor_dict[k] = [tensor_dict_to_float32(_) for _ in v]
        else:
            raise ValueError(f"Unsupported type {type(v)} in the tensor dict float cast.")
    return res_tensor_dict


def tensor_dict_index_select(tensor_dict, index, dim=0):
    res_tensor_dict = defaultdict()
    for k in tensor_dict.keys():
        if isinstance(tensor_dict[k], torch.Tensor):
            res_tensor_dict[k] = torch.index_select(tensor_dict[k], index=index, dim=dim).contiguous()
        elif isinstance(tensor_dict[k], dict):
            res_tensor_dict[k] = tensor_dict_index_select(tensor_dict[k], index=index, dim=dim)
        elif isinstance(tensor_dict[k], list):
            res_tensor_dict[k] = [
                tensor_dict_index_select(tensor_dict[k][_], index=index, dim=dim)
                for _ in range(len(tensor_dict[k]))
            ]
        else:
            raise ValueError(f"Unsupported type {type(tensor_dict[k])} in the tensor dict index select.")
    return dict(res_tensor_dict)


def rink_batch_from_metas(metas) -> dict:
    """The rink anchor's per-frame inputs from a batch's metas (B clips x T frames; see data/rink.py): ``H`` (B, T, 3, 3)
    normalised augmented-image coords -> rink metres (NaN: frame has no homography), ``flip`` (B, T), ``bounds`` (B, T, 2).
    Frames without the keys (a dataset with no rink sidecar) are all-NaN, so their tokens get the "no position" embedding."""
    nan3, nan2 = np.full((3, 3), np.nan), np.full(2, np.nan)
    return {
        "H": torch.as_tensor(np.stack([[m.get("rink_Hn", nan3) for m in clip] for clip in metas]), dtype=torch.float64),
        "flip": torch.as_tensor([[bool(m.get("rink_flip", False)) for m in clip] for clip in metas]),
        "bounds": torch.as_tensor(np.stack([[m.get("rink_bounds", nan2) for m in clip] for clip in metas]), dtype=torch.float64),
    }


def prepare_for_motip(detr_outputs, annotations, detr_indices, with_gt_boxes: bool = False, rink: dict | None = None):
    """``rink`` (from ``rink_batch_from_metas``) also returns ``{trajectory,unknown}_rink_xy`` (B, G, T, N, 2): each token's
    feet in rink metres, projected from the *predicted* box of the detection it was built from with its own frame's
    homography (so training sees the noise inference does); NaN where invalid.

    ``with_gt_boxes`` also returns ``unknown_gt_boxes``: the GT box (cxcywh, normalised) of each slot's
    object in the current frame (not of the detection a switched trajectory token was built from), which the
    Spatial ID Loss compares the matched detections' predicted boxes against."""
    _B, _T = len(annotations), len(annotations[0])
    _G, _, _N = annotations[0][0]["trajectory_id_labels"].shape
    _device = detr_outputs["pred_logits"].device
    _feature_dim = detr_outputs["outputs"].shape[-1]
    # Init corresponding variables:
    trajectory_id_labels = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    trajectory_times = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    trajectory_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=_device)
    trajectory_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=_device)
    trajectory_features = torch.zeros((_B, _G, _T, _N, _feature_dim), dtype=torch.float32, device=_device)
    unknown_id_labels = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    unknown_times = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    unknown_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=_device)
    unknown_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=_device)
    unknown_features = torch.zeros((_B, _G, _T, _N, _feature_dim), dtype=torch.float32, device=_device)
    unknown_gt_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=_device)
    _kit = detr_outputs.get("kit")      # TEAM_CUE: (B*T, Nmax, K) kit descriptors, same row order as the embeddings
    if _kit is not None:
        trajectory_kit = torch.full((_B, _G, _T, _N, _kit.shape[-1]), float("nan"), dtype=torch.float32, device=_device)
        unknown_kit = torch.full((_B, _G, _T, _N, _kit.shape[-1]), float("nan"), dtype=torch.float32, device=_device)
    for b in range(_B):
        for t in range(_T):
            flatten_idx = b * _T + t
            go_back_detr_idxs = torch.argsort(detr_indices[flatten_idx][1])
            detr_output_embeds = detr_outputs["outputs"][flatten_idx][detr_indices[flatten_idx][0][go_back_detr_idxs]]
            detr_boxes = detr_outputs["pred_boxes"][flatten_idx][detr_indices[flatten_idx][0][go_back_detr_idxs]]
            gt_boxes = annotations[b][t]["bbox"].to(_device)
            if _kit is not None:
                detr_kit = _kit[flatten_idx][detr_indices[flatten_idx][0][go_back_detr_idxs]]
            # detr_output_embeds = einops.repeat(detr_output_embeds, "n d -> g n d", g=_G)
            # detr_boxes = einops.repeat(detr_boxes, "n d -> g n d", g=_G)
            for group in range(_G):
                _curr_traj_ann_idxs = annotations[b][t]["trajectory_ann_idxs"][group, 0, :]
                _curr_unk_ann_idxs = annotations[b][t]["unknown_ann_idxs"][group, 0, :]
                _curr_traj_masks = annotations[b][t]["trajectory_id_masks"][group, 0, :]
                _curr_unk_masks = annotations[b][t]["unknown_id_masks"][group, 0, :]
                # Fill the fields:
                trajectory_id_labels[b, group, t] = annotations[b][t]["trajectory_id_labels"][group, 0, :]
                unknown_id_labels[b, group, t] = annotations[b][t]["unknown_id_labels"][group, 0, :]
                trajectory_times[b, group, t] = annotations[b][t]["trajectory_times"][group, 0, :]
                unknown_times[b, group, t] = annotations[b][t]["unknown_times"][group, 0, :]
                trajectory_masks[b, group, t] = _curr_traj_masks
                unknown_masks[b, group, t] = _curr_unk_masks
                trajectory_features[b, group, t, ~_curr_traj_masks] = detr_output_embeds[_curr_traj_ann_idxs[~_curr_traj_masks]]
                unknown_features[b, group, t, ~_curr_unk_masks] = detr_output_embeds[_curr_unk_ann_idxs[~_curr_unk_masks]]
                trajectory_boxes[b, group, t, ~_curr_traj_masks] = detr_boxes[_curr_traj_ann_idxs[~_curr_traj_masks]]
                unknown_boxes[b, group, t, ~_curr_unk_masks] = detr_boxes[_curr_unk_ann_idxs[~_curr_unk_masks]]
                if _kit is not None:
                    trajectory_kit[b, group, t, ~_curr_traj_masks] = detr_kit[_curr_traj_ann_idxs[~_curr_traj_masks]]
                    unknown_kit[b, group, t, ~_curr_unk_masks] = detr_kit[_curr_unk_ann_idxs[~_curr_unk_masks]]
                if with_gt_boxes:
                    unknown_gt_boxes[b, group, t, ~_curr_unk_masks] = gt_boxes[_curr_unk_ann_idxs[~_curr_unk_masks]]
                pass
            pass
    seq_info = {
        "trajectory_id_labels": trajectory_id_labels,
        "trajectory_times": trajectory_times,
        "trajectory_masks": trajectory_masks,
        "trajectory_boxes": trajectory_boxes,
        "trajectory_features": trajectory_features,
        "unknown_id_labels": unknown_id_labels,
        "unknown_times": unknown_times,
        "unknown_masks": unknown_masks,
        "unknown_boxes": unknown_boxes,
        "unknown_features": unknown_features,
    }
    if with_gt_boxes:
        seq_info["unknown_gt_boxes"] = unknown_gt_boxes
    if _kit is not None:
        seq_info["trajectory_kit"] = trajectory_kit
        seq_info["unknown_kit"] = unknown_kit
    if rink is not None:
        from models.motip.rink_anchor import project_feet
        _h = rink["H"].to(_device)[:, None, :, None]            # (B, 1, T, 1, 3, 3)
        _flip = rink["flip"].to(_device)[:, None, :, None]       # (B, 1, T, 1)
        _bounds = rink["bounds"].to(_device)[:, None, :, None]   # (B, 1, T, 1, 2)
        seq_info["trajectory_rink_xy"] = project_feet(trajectory_boxes, _h, _flip, _bounds)
        seq_info["unknown_rink_xy"] = project_feet(unknown_boxes, _h, _flip, _bounds)
    return seq_info


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:   # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:                                   # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    train_engine(config=cfg)
