# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This script is a simplified version of the training script in detectron2/tools.
"""
from functools import partial
import copy
import itertools
import logging
import os
from typing import Any, Dict, List, Set, Union
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    # inference_on_dataset,
    print_csv_format,
    inference_context,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import SemanticSegmentorWithTTA, add_mask_former_config
from mask2former.config import add_freemix_config
from mask2former.utils.events import WandbWriter, setup_wandb
# from omegaconf import OmegaConf
from omegaconf import OmegaConf
from yacs.config import CfgNode as CN

def setup(args):
    """
    Create configs and perform basic setups.
    """
    # fcfg = open(args.config_file)
    # cfg = CN.load_cfg(fcfg)
    # cfg.freeze()

    # cfg = OmegaConf.load(args.config_file)

    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_freemix_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    if not args.eval_only:
        setup_wandb(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former"
    )
    return cfg


def main(args):
    cfg = setup(args)
    # print(cfg)
    if cfg.TRAINER == 'FreeMix_Trainer':
        from trainer.freeseg_trainer import FreeMix_Trainer as Trainer
    else:
        from trainer.freeseg_trainer import FreeSeg_Trainer as Trainer

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if model._region_clip_adapter is not None:
            model._region_clip_adapter.load_state_dict(model.clip_adapter.state_dict())

        if cfg.TEST.AUG.ENABLED:
            res = Trainer.test_with_TTA(cfg, model)
        else:
            res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res


    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
