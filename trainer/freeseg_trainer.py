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
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
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
import datetime
from torch import nn
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

# MaskFormer
from mask2former import SemanticSegmentorWithTTA, add_mask_former_config
from mask2former.data import (
    COCOFullTaskNewBaselineDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    MaskFormerBinarySemanticDatasetMapper,
    MaskFormerBinaryFullDatasetMapper,
    ProposalClasificationDatasetMapper,
)

from mask2former.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    dataset_sample_per_class,
    dataset_sample_per_task_class,
)
from mask2former.data.build import build_freemix_detection_train_loader, build_freemix_detection_test_loader
from mask2former.evaluation import (
    GeneralizedSemSegEvaluator,
    GeneralizedPseudoSemSegEvaluator,
    ClassificationEvaluator,
    GeneralizedPanopticEvaluator,
    InstanceSegEvaluator,
    COCOEvaluator,
)
from detectron2.evaluation import SemSegEvaluator

from mask2former.utils.events import WandbWriter
from mask2former.utils.post_process_utils import dense_crf_post_process
import torch.utils.data as torchdata

class JointLoader(torchdata.IterableDataset):
    def __init__(self, loaders, key_dataset):
        dataset_names = []
        for key, loader in loaders.items():
            # name = "{}".format(key.split('_')[0])
            name = key
            setattr(self, name, loader)
            dataset_names += [name]
        self.dataset_names = dataset_names
        self.key_dataset = key_dataset

    def __iter__(self):
        for batch in zip(*[getattr(self, name) for name in self.dataset_names]):
            yield {key: batch[i] for i, key in enumerate(self.dataset_names)}

    def __len__(self):
        return len(getattr(self, self.key_dataset))


class FreeSeg_Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            if cfg.PSEUDO:
                evaluator = partial(
                    GeneralizedPseudoSemSegEvaluator,
                    with_prior=cfg.PSEUDO_WITH_PRIOR,
                    reject_threshold=cfg.PSEUDO_REJECT_THRESHOLD,
                )
            else:
                evaluator = GeneralizedSemSegEvaluator
                # evaluator = SemSegEvaluator
            evaluator_list.append(
                evaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    post_process_func=dense_crf_post_process
                    if cfg.TEST.DENSE_CRF
                    else None,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type == "ade20k_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(GeneralizedPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "classification":
            evaluator_list.append(ClassificationEvaluator(dataset_name))

        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = None
        mapper = None
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":  # here
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_binary_semantic":
            mapper = MaskFormerBinarySemanticDatasetMapper(cfg, True)
            dataset = dataset_sample_per_class(cfg)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_full_binary_semantic":
            mapper = MaskFormerBinaryFullDatasetMapper(cfg, True)
            dataset = dataset_sample_per_task_class(cfg)

        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_full_lsj":
            mapper = COCOFullTaskNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if cfg.ORACLE:
            if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":  # here
                mapper = MaskFormerSemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_binary_semantic":
                mapper = MaskFormerBinarySemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_full_binary_semantic":
                mapper = MaskFormerBinarySemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "propsoal_classification":
                mapper = ProposalClasificationDatasetMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:
        ::
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            WandbWriter(),
        ]

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                            hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def inference_on_dataset(cls, model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.

        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.

                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.

        Returns:
            The return value of `evaluator.evaluate()`
        """
        num_devices = get_world_size()
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length
        if evaluator is None:
            # create a no-op evaluator
            evaluator = DatasetEvaluators([])
        if isinstance(evaluator, abc.MutableSequence):
            evaluator = DatasetEvaluators(evaluator)
        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = cls.inference_on_dataset(model=model, data_loader=data_loader, evaluator=evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class FreeMix_Trainer(FreeSeg_Trainer):
    @classmethod
    def build_train_loader(cls, cfg):
        dataset_names = cfg.DATASETS.TRAIN
        loaders = {}
        for dataset_name in dataset_names:
            cfg = cls.update_dataset_config(cfg, dataset_name)
            mapper_name = cfg.INPUT.DATASET_MAPPER_NAME
            # Semantic segmentation dataset mapper
            if mapper_name == "camvid_mask_former_semantic":
                mapper = MaskFormerSemanticDatasetMapper(cfg, True)
                loaders[dataset_name] = build_freemix_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
            elif mapper_name == "potsdam_mask_former_semantic":
                mapper = MaskFormerSemanticDatasetMapper(cfg, True)
                loaders[dataset_name] = build_freemix_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
            elif mapper_name == "GID_15_mask_former_semantic":
                mapper = MaskFormerSemanticDatasetMapper(cfg, True)
                loaders[dataset_name] = build_freemix_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
            elif mapper_name == "GID_5_mask_former_semantic":
                mapper = MaskFormerSemanticDatasetMapper(cfg, True)
                loaders[dataset_name] = build_freemix_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)

            # Panoptic segmentation dataset mapper
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
                mapper = MaskFormerPanopticDatasetMapper(cfg, True)
                loaders[dataset_name] = build_freemix_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
            # Instance segmentation dataset mapper
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
                mapper = MaskFormerInstanceDatasetMapper(cfg, True)
                loaders[dataset_name] = build_freemix_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_full_lsj":
                mapper = COCOFullTaskNewBaselineDatasetMapper(cfg, True)
                loaders[dataset_name] = build_freemix_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)

        if len(loaders) == 1 and not cfg.LOADER.get('JOINT', False):
            return list(loaders.values())[0]
        else:
            return JointLoader(loaders, key_dataset=cfg.LOADER.get('KEY_DATASET', 'COCO'))

    @classmethod
    def update_dataset_config(cls, cfg, dataset_name):
        if 'camvid' in dataset_name:
            cfg.update(cfg.Camvid)
        if 'potsdam' in dataset_name:
            cfg.update(cfg.Potsdam)
        if 'GID_15' in dataset_name:
            cfg.update(cfg.GID_15)
        if 'GID_5' in dataset_name:
            cfg.update(cfg.GID_5)
        if 'coco' in dataset_name:
            cfg.update(cfg.COCO)
        return cfg

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        cfg = cls.update_dataset_config(cfg, dataset_name)
        if cfg.ORACLE:
            if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":  # here
                mapper = MaskFormerSemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_binary_semantic":
                mapper = MaskFormerBinarySemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_full_binary_semantic":
                mapper = MaskFormerBinarySemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "propsoal_classification":
                mapper = ProposalClasificationDatasetMapper(cfg, False)
        else:
            mapper = None
        return build_freemix_detection_test_loader(cfg, dataset_name, mapper=mapper)
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg["OUTPUT_DIR"], "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            if cfg.PSEUDO:
                evaluator = partial(
                    GeneralizedPseudoSemSegEvaluator,
                    with_prior=cfg.PSEUDO_WITH_PRIOR,
                    reject_threshold=cfg.PSEUDO_REJECT_THRESHOLD,
                )
            else:
                evaluator = GeneralizedSemSegEvaluator
                # evaluator = SemSegEvaluator
            evaluator_list.append(
                evaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type == "ade20k_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(GeneralizedPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "classification":
            evaluator_list.append(ClassificationEvaluator(dataset_name))

        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]


        return DatasetEvaluators(evaluator_list)


