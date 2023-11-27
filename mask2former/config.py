# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        if ori_value:
            logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")

def add_mask_former_default_config(cfg):
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_our_config(cfg):
    cfg.ORACLE = False
    cfg.PSEUDO = False
    cfg.PSEUDO_WITH_PRIOR = True
    cfg.PSEUDO_REJECT_THRESHOLD = 0.0
    cfg.TEST.SLIDING_WINDOW = False
    cfg.TEST.SLIDING_TILE_SIZE = 224
    cfg.TEST.SLIDING_OVERLAP = 2 / 3.0
    cfg.PSEUDO_FLAG_NAME = "trainable_flag"
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1
    cfg.DATASETS.SAMPLE_PER_CLASS = -1
    cfg.DATASETS.SAMPLE_SEED = 0

    cfg.TEST.OPTIM = CN()
    cfg.TEST.OPTIM.LR = 0.001

    cfg.INPUT.TASK_NAME = ["semantic segmentation."]
    # whether to use dense crf
    cfg.TEST.DENSE_CRF = False
    # embedding head
    cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM = 1024
    cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS = 2
    # clip_adapter
    cfg.MODEL.CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER = "learnable"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = ["a sculpture of a {}."]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.TASK_PROMPT_SHAPE = 8
    cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT = ""
    cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.MASK_FILL = "mean"
    cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO = 1.0
    cfg.MODEL.CLIP_ADAPTER.MASK_THR = 0.5
    cfg.MODEL.CLIP_ADAPTER.MASK_MATTING = False
    cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT = 0.8
    #
    cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER = False
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_LEARNER = "predefined"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = [
        "a photo of a {}."
    ]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_CHECKPOINT = ""


    cfg.MODEL.SEM_SEG_HEAD.EMB_SIZE = 256
    cfg.MODEL.SEM_SEG_HEAD.EMBED_DIM = 2048
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.USE_LAYER_SCALE = True


    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "zero_shot_seg"
    cfg.WANDB.NAME = None

def add_freemix_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    add_mask_former_default_config(cfg)

    cfg.TRAINER = "FreeMix_Trainer"

    cfg.LOADER = CN()
    cfg.LOADER.JOINT = True
    cfg.LOADER.KEY_DATASET = 'potsdam'

    cfg.DATASETS.DETECTIONS_PER_IMAGE= 100
    cfg.DATASETS.SAMPLE_PER_CLASS = -1
    cfg.DATASETS.SAMPLE_SEED = 0

    cfg.MISSION_NAME= ["semantic segmentation."]
    cfg.SCENE_NAMES = [ "autonomous driving scenario", "remote sensing scene"]

    cfg.Camvid = CN()
    cfg.Camvid.Scene_name = "autonomous driving scenario"
    cfg.Camvid.INPUT = CN()
    cfg.Camvid.INPUT.DATASET_MAPPER_NAME = "camvid_mask_former_semantic"
    cfg.Camvid.INPUT.COLOR_AUG_SSD = True
    cfg.Camvid.INPUT.CROP = CN({"ENABLED": True})
    cfg.Camvid.INPUT.CROP.TYPE = "absolute"
    cfg.Camvid.INPUT.CROP.SIZE = [512, 512]
    cfg.Camvid.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.Camvid.INPUT.FORMAT = "BGR"
    cfg.Camvid.INPUT.IMAGE_SIZE = 1024
    cfg.Camvid.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    cfg.Camvid.INPUT.MAX_SCALE = 2.0
    cfg.Camvid.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.Camvid.INPUT.MAX_SIZE_TEST = 2048
    cfg.Camvid.INPUT.MIN_SCALE = 0.1
    cfg.Camvid.INPUT.MIN_SIZE_TEST = 512
    cfg.Camvid.INPUT.MIN_SIZE_TRAIN = (320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960)
    cfg.Camvid.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.Camvid.INPUT.RANDOM_FLIP = "horizontal"
    cfg.Camvid.INPUT.SIZE_DIVISIBILITY = 512

    cfg.Camvid.TEST = CN()
    cfg.Camvid.TEST.AUG = CN({"ENABLED": False})
    cfg.Camvid.TEST.AUG.FLIP = True
    cfg.Camvid.TEST.AUG.MIN_SIZES = (256, 384, 512, 640, 768, 896)
    cfg.Camvid.TEST.AUG.MAX_SIZE = 3584
    cfg.Camvid.TEST.DENSE_CRF = False
    cfg.Camvid.TEST.EVAL_PERIOD = 5000
    cfg.Camvid.TEST.EXPECTED_RESULTS = []
    cfg.Camvid.TEST.KEYPOINT_OKS_SIGMAS = []
    cfg.Camvid.TEST.OPTIM = CN()
    cfg.Camvid.TEST.OPTIM.LR = 0.001
    cfg.Camvid.TEST.PRECISE_BN = CN({"ENABLED": False})
    cfg.Camvid.TEST.PRECISE_BN.NUM_ITER = 200
    cfg.Camvid.TEST.SLIDING_WINDOW = False
    cfg.Camvid.TEST.SLIDING_TILE_SIZE = 224
    cfg.Camvid.TEST.SLIDING_OVERLAP = 2 / 3.0

    cfg.Camvid.DATALOADER = CN()
    cfg.Camvid.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.Camvid.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.Camvid.DATALOADER.NUM_WORKERS = 4
    cfg.Camvid.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.Camvid.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    cfg.Potsdam = CN()
    cfg.Potsdam.Scene_name = "remote sensing scene"
    cfg.Potsdam.INPUT = CN()
    cfg.Potsdam.INPUT.DATASET_MAPPER_NAME = "camvid_mask_former_semantic"
    cfg.Potsdam.INPUT.COLOR_AUG_SSD = True
    cfg.Potsdam.INPUT.CROP = CN({"ENABLED": True})
    cfg.Potsdam.INPUT.CROP.TYPE = "absolute"
    cfg.Potsdam.INPUT.CROP.SIZE = [512, 512]
    cfg.Potsdam.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.Potsdam.INPUT.FORMAT = "BGR"
    cfg.Potsdam.INPUT.IMAGE_SIZE = 1024
    cfg.Potsdam.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    cfg.Potsdam.INPUT.MAX_SCALE = 2.0
    cfg.Potsdam.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.Potsdam.INPUT.MAX_SIZE_TEST = 2048
    cfg.Potsdam.INPUT.MIN_SCALE = 0.1
    cfg.Potsdam.INPUT.MIN_SIZE_TEST = 512
    cfg.Potsdam.INPUT.MIN_SIZE_TRAIN = (320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960)
    cfg.Potsdam.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.Potsdam.INPUT.RANDOM_FLIP = "horizontal"
    cfg.Potsdam.INPUT.SIZE_DIVISIBILITY = 512

    cfg.Potsdam.TEST = CN()
    cfg.Potsdam.TEST.AUG = CN({"ENABLED": False})
    cfg.Potsdam.TEST.AUG.FLIP = True
    cfg.Potsdam.TEST.AUG.MIN_SIZES = (256, 384, 512, 640, 768, 896)
    cfg.Potsdam.TEST.AUG.MAX_SIZE = 3584
    cfg.Potsdam.TEST.DENSE_CRF = False
    cfg.Potsdam.TEST.EVAL_PERIOD = 5000
    cfg.Potsdam.TEST.EXPECTED_RESULTS = []
    cfg.Potsdam.TEST.KEYPOINT_OKS_SIGMAS = []
    cfg.Potsdam.TEST.OPTIM = CN()
    cfg.Potsdam.TEST.OPTIM.LR = 0.001
    cfg.Potsdam.TEST.PRECISE_BN = CN({"ENABLED": False})
    cfg.Potsdam.TEST.PRECISE_BN.NUM_ITER = 200
    cfg.Potsdam.TEST.SLIDING_WINDOW = False
    cfg.Potsdam.TEST.SLIDING_TILE_SIZE = 224
    cfg.Potsdam.TEST.SLIDING_OVERLAP = 2 / 3.0

    cfg.Potsdam.DATALOADER = CN()
    cfg.Potsdam.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.Potsdam.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.Potsdam.DATALOADER.NUM_WORKERS = 4
    cfg.Potsdam.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.Potsdam.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    cfg.GID_15 = CN()
    cfg.GID_15.Scene_name = "remote sensing"
    cfg.GID_15.INPUT = CN()
    cfg.GID_15.INPUT.DATASET_MAPPER_NAME = "GID_15_mask_former_semantic"
    cfg.GID_15.INPUT.COLOR_AUG_SSD = True
    cfg.GID_15.INPUT.CROP = CN({"ENABLED": True})
    cfg.GID_15.INPUT.CROP.TYPE = "absolute"
    cfg.GID_15.INPUT.CROP.SIZE = [512, 512]
    cfg.GID_15.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.GID_15.INPUT.FORMAT = "BGR"
    cfg.GID_15.INPUT.IMAGE_SIZE = 1024
    cfg.GID_15.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    cfg.GID_15.INPUT.MAX_SCALE = 2.0
    cfg.GID_15.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.GID_15.INPUT.MAX_SIZE_TEST = 2048
    cfg.GID_15.INPUT.MIN_SCALE = 0.1
    cfg.GID_15.INPUT.MIN_SIZE_TEST = 512
    cfg.GID_15.INPUT.MIN_SIZE_TRAIN = (320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960)
    cfg.GID_15.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.GID_15.INPUT.RANDOM_FLIP = "horizontal"
    cfg.GID_15.INPUT.SIZE_DIVISIBILITY = 512

    cfg.GID_15.TEST = CN()
    cfg.GID_15.TEST.AUG = CN({"ENABLED": False})
    cfg.GID_15.TEST.AUG.FLIP = True
    cfg.GID_15.TEST.AUG.MIN_SIZES = (256, 384, 512, 640, 768, 896)
    cfg.GID_15.TEST.AUG.MAX_SIZE = 3584
    cfg.GID_15.TEST.DENSE_CRF = False
    cfg.GID_15.TEST.EVAL_PERIOD = 5000
    cfg.GID_15.TEST.EXPECTED_RESULTS = []
    cfg.GID_15.TEST.KEYPOINT_OKS_SIGMAS = []
    cfg.GID_15.TEST.OPTIM = CN()
    cfg.GID_15.TEST.OPTIM.LR = 0.001
    cfg.GID_15.TEST.PRECISE_BN = CN({"ENABLED": False})
    cfg.GID_15.TEST.PRECISE_BN.NUM_ITER = 200
    cfg.GID_15.TEST.SLIDING_WINDOW = False
    cfg.GID_15.TEST.SLIDING_TILE_SIZE = 224
    cfg.GID_15.TEST.SLIDING_OVERLAP = 2 / 3.0

    cfg.GID_15.DATALOADER = CN()
    cfg.GID_15.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.GID_15.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.GID_15.DATALOADER.NUM_WORKERS = 4
    cfg.GID_15.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.GID_15.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    cfg.GID_5 = CN()
    cfg.GID_5.Scene_name = "remote sensing"
    cfg.GID_5.INPUT = CN()
    cfg.GID_5.INPUT.DATASET_MAPPER_NAME = "GID_5_mask_former_semantic"
    cfg.GID_5.INPUT.COLOR_AUG_SSD = True
    cfg.GID_5.INPUT.CROP = CN({"ENABLED": True})
    cfg.GID_5.INPUT.CROP.TYPE = "absolute"
    cfg.GID_5.INPUT.CROP.SIZE = [512, 512]
    cfg.GID_5.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.GID_5.INPUT.FORMAT = "BGR"
    cfg.GID_5.INPUT.IMAGE_SIZE = 1024
    cfg.GID_5.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    cfg.GID_5.INPUT.MAX_SCALE = 2.0
    cfg.GID_5.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.GID_5.INPUT.MAX_SIZE_TEST = 2048
    cfg.GID_5.INPUT.MIN_SCALE = 0.1
    cfg.GID_5.INPUT.MIN_SIZE_TEST = 512
    cfg.GID_5.INPUT.MIN_SIZE_TRAIN = (320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960)
    cfg.GID_5.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.GID_5.INPUT.RANDOM_FLIP = "horizontal"
    cfg.GID_5.INPUT.SIZE_DIVISIBILITY = 512

    cfg.GID_5.TEST = CN()
    cfg.GID_5.TEST.AUG = CN({"ENABLED": False})
    cfg.GID_5.TEST.AUG.FLIP = True
    cfg.GID_5.TEST.AUG.MIN_SIZES = (256, 384, 512, 640, 768, 896)
    cfg.GID_5.TEST.AUG.MAX_SIZE = 3584
    cfg.GID_5.TEST.DENSE_CRF = False
    cfg.GID_5.TEST.EVAL_PERIOD = 5000
    cfg.GID_5.TEST.EXPECTED_RESULTS = []
    cfg.GID_5.TEST.KEYPOINT_OKS_SIGMAS = []
    cfg.GID_5.TEST.OPTIM = CN()
    cfg.GID_5.TEST.OPTIM.LR = 0.001
    cfg.GID_5.TEST.PRECISE_BN = CN({"ENABLED": False})
    cfg.GID_5.TEST.PRECISE_BN.NUM_ITER = 200
    cfg.GID_5.TEST.SLIDING_WINDOW = False
    cfg.GID_5.TEST.SLIDING_TILE_SIZE = 224
    cfg.GID_5.TEST.SLIDING_OVERLAP = 2 / 3.0

    cfg.GID_5.DATALOADER = CN()
    cfg.GID_5.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.GID_5.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.GID_5.DATALOADER.NUM_WORKERS = 4
    cfg.GID_5.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.GID_5.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    cfg.COCO = CN()
    cfg.COCO.Scene_name = "remote sensing"
    cfg.COCO.INPUT = CN()
    cfg.COCO.INPUT.DATASET_MAPPER_NAME = "GID_5_mask_former_semantic"
    cfg.COCO.INPUT.COLOR_AUG_SSD = True
    cfg.COCO.INPUT.CROP = CN({"ENABLED": True})
    cfg.COCO.INPUT.CROP.TYPE = "absolute"
    cfg.COCO.INPUT.CROP.SIZE = [512, 512]
    cfg.COCO.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.COCO.INPUT.FORMAT = "BGR"
    cfg.COCO.INPUT.IMAGE_SIZE = 1024
    cfg.COCO.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    cfg.COCO.INPUT.MAX_SCALE = 2.0
    cfg.COCO.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.COCO.INPUT.MAX_SIZE_TEST = 2048
    cfg.COCO.INPUT.MIN_SCALE = 0.1
    cfg.COCO.INPUT.MIN_SIZE_TEST = 512
    cfg.COCO.INPUT.MIN_SIZE_TRAIN = (320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960)
    cfg.COCO.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.COCO.INPUT.RANDOM_FLIP = "horizontal"
    cfg.COCO.INPUT.SIZE_DIVISIBILITY = 512

    cfg.COCO.TEST = CN()
    cfg.COCO.TEST.AUG = CN({"ENABLED": False})
    cfg.COCO.TEST.AUG.FLIP = True
    cfg.COCO.TEST.AUG.MIN_SIZES = (256, 384, 512, 640, 768, 896)
    cfg.COCO.TEST.AUG.MAX_SIZE = 3584
    cfg.COCO.TEST.DENSE_CRF = False
    cfg.COCO.TEST.EVAL_PERIOD = 5000
    cfg.COCO.TEST.EXPECTED_RESULTS = []
    cfg.COCO.TEST.KEYPOINT_OKS_SIGMAS = []
    cfg.COCO.TEST.OPTIM = CN()
    cfg.COCO.TEST.OPTIM.LR = 0.001
    cfg.COCO.TEST.PRECISE_BN = CN({"ENABLED": False})
    cfg.COCO.TEST.PRECISE_BN.NUM_ITER = 200
    cfg.COCO.TEST.SLIDING_WINDOW = False
    cfg.COCO.TEST.SLIDING_TILE_SIZE = 224
    cfg.COCO.TEST.SLIDING_OVERLAP = 2 / 3.0

    cfg.COCO.DATALOADER = CN()
    cfg.COCO.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.COCO.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.COCO.DATALOADER.NUM_WORKERS = 4
    cfg.COCO.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.COCO.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"


    cfg.ORACLE = False
    cfg.PSEUDO = False
    cfg.PSEUDO_FLAG_NAME = "trainable_flag"
    cfg.PSEUDO_REJECT_THRESHOLD = 0.0
    cfg.PSEUDO_WITH_PRIOR = True

    cfg.SOLVER.TEST_IMS_PER_BATCH = 4

    cfg.SAM_Branch = False


    # embedding head
    cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM = 1024
    cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS = 2
    # clip_adapter
    cfg.MODEL.CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER = "learnable"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = ["a sculpture of a {}."]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.TASK_PROMPT_SHAPE = 8
    cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT = ""
    cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.MASK_FILL = "mean"
    cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO = 1.0
    cfg.MODEL.CLIP_ADAPTER.MASK_THR = 0.5
    cfg.MODEL.CLIP_ADAPTER.MASK_MATTING = False
    cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT = 0.8
    #
    cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER = False
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_LEARNER = "predefined"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = [
        "a photo of a {}."
    ]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_CHECKPOINT = ""

    cfg.MODEL.SEM_SEG_HEAD.EMB_SIZE = 256
    cfg.MODEL.SEM_SEG_HEAD.EMBED_DIM = 2048
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.USE_LAYER_SCALE = True

    cfg.MODEL.SAM = CN()
    cfg.MODEL.SAM.MODEL_TAPE = "vit_t"
    cfg.MODEL.SAM.IMG_RESOLUTION = 512 # same with crop size
    cfg.MODEL.SAM.CLIP_MODEL_NAME =  "ViT-B/16"
    cfg.MODEL.SAM.CLIP_PIXEL_MEAN =  [ 123.675, 116.280, 103.530 ]
    cfg.MODEL.SAM.CLIP_PIXEL_STD =  [ 58.395, 57.120, 57.375 ]

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "zero_shot_seg"
    cfg.WANDB.NAME = None



def add_mask_former_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    add_mask_former_default_config(cfg)
    add_our_config(cfg)

def load_cfg_from_config_files(conf_file):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    with open(conf_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    load_config_dict_to_opt(opt, config_dict)

    return opt