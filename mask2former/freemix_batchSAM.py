# Copyright (c) Facebook, Inc. and its affiliates.
from cgitb import text
import logging
import copy
import random
import os
from typing import Tuple

import torchvision
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.logger import log_first_n
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.transforms import ResizeTransform

from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_prompt_learner,
)
from .modeling.clip_adapter.clip import build_clip_model
from .mask_former_model import MaskFormer
import cv2
from third_party.MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from third_party.MobileSAM.mobile_sam.utils.amg import build_all_layer_point_grids
from third_party.CLIP.clip.model import MaskCLIP

@META_ARCH_REGISTRY.register()
class FreeMix_BatchSAM(MaskFormer):

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            clip_adapter: nn.Module,
            region_clip_adapter: nn.Module = None,
            task_names: list,
            criterion: nn.Module,
            num_queries: int,
            semantic_on: bool,
            instance_on: bool,
            panoptic_on: bool,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            clip_ensemble: bool,
            clip_ensemble_weight: float,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            test_topk_per_image: int,
            sam_branch: bool,
            sam_mask_generator,
            sam,
            img_resolution,
            sam_clip_model_name,
            maskCLIP,
            clip_pixel_mean,
            clip_pixel_std,
            cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            semantic_on=semantic_on,
            instance_on=instance_on,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter

        self._region_clip_adapter = region_clip_adapter

        self.task_names = task_names
        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight
        self.test_topk_per_image = test_topk_per_image
        self.ma_loss = nn.SmoothL1Loss()  # SmoothL1Loss L1Loss L2Loss KLLoss

        self.outdir = cfg.OUTPUT_DIR
        self.sam_branch = sam_branch
        if self.sam_branch:
            self.sam=sam
            self.sam_mask_generator = sam_mask_generator
            # clip_model = build_clip_model(sam_clip_model_name, frozen=False).float()
            # self.myclip_model = clip_model.visual
            self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
            # img resolution
            self.img_resolution = img_resolution
            self.input_point = torch.as_tensor(build_all_layer_point_grids(16, 0, 1)[0] * self.img_resolution,
                                          dtype=torch.int64).cuda()
            self.input_label = torch.tensor([1 for _ in range(self.input_point.shape[0])]).cuda()

            self.maskCLIP = maskCLIP

        self._freeze()

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)

        scene_names = cfg.MISSION_NAME  # 任务
        # scene_names = cfg.SCENE_NAMES # 场景数据源

        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER, scene_names)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER, cfg.MISSION_NAME
            )
            region_clip_adapter = MaskFormerClipAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                mask_thr=0.4,
                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            )

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
        )

        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter
        init_kwargs["task_names"] = cfg.MISSION_NAME
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT  # 0.7
        init_kwargs["test_topk_per_image"] = cfg.DATASETS.DETECTIONS_PER_IMAGE  # 100
        init_kwargs["metadata"] = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        init_kwargs["semantic_on"] = "semantic segmentation." in cfg.MISSION_NAME
        init_kwargs["instance_on"] = "instance segmentation." in cfg.MISSION_NAME
        init_kwargs["panoptic_on"] = "panoptic segmentation." in cfg.MISSION_NAME
        # sam_branch
        init_kwargs["sam_branch"] = cfg.SAM_Branch
        if cfg.SAM_Branch:
            sam_checkpoint = "third_party/MobileSAM/weights/mobile_sam.pt"
            model_type = cfg.MODEL.SAM.MODEL_TAPE
            img_resolution  = cfg.MODEL.SAM.IMG_RESOLUTION
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, custom_img_size=img_resolution)
            sam.to(device=device)
            sam.eval()
            mask_generator = SamAutomaticMaskGenerator(sam)
            init_kwargs["sam_mask_generator"] = mask_generator
            init_kwargs["img_resolution"] = img_resolution
            init_kwargs["sam"] = sam
            maskCLIP = MaskCLIP(input_resolution=224,  # "ViT-B/16" CLIP.visual
                                     patch_size=16,
                                     width=768,
                                     layers=12,
                                     heads=12,
                                     output_dim=512,
                                     )
            # CLIP_state_dict = torch.jit.load('/home/wjy/.cache/clip/ViT-B-16.pt', map_location="cpu")
            # maskCLIP.load_state_dict(CLIP_state_dict.state_dict(), strict=False)
            init_kwargs["maskCLIP"] = maskCLIP


            # build a fine tune clip encoder
            init_kwargs[
                "sam_clip_model_name"] = cfg.MODEL.SAM.CLIP_MODEL_NAME  # 可以不同与 cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME
            init_kwargs["clip_pixel_mean"] = cfg.MODEL.SAM.CLIP_PIXEL_MEAN
            init_kwargs["clip_pixel_std"] = cfg.MODEL.SAM.CLIP_PIXEL_STD

        init_kwargs["cfg"] = cfg

        return init_kwargs

    def _freeze(self):
        frozen_exclude  = ["sam"]
        for name, param in self.named_parameters():
            print('\t', name, param.requires_grad)
        print("="*80)
        for name, param in self.named_parameters():
            if any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False
            if "myclip_model" in name:
                param.requires_grad = False
            if param.requires_grad == True:
                print(name, param.requires_grad)
        print("=" * 80)

    def forward(self, batched_inputs, text_labels=None):
        if self.training:
            losses = {}
            for dataset_name, batched_input in batched_inputs.items():
                scene_name = batched_input[0]["meta"]["scene_name"]
                losses_seg = self.forward_seg(batched_input, dataset_name, text_labels, scene_name=scene_name)
                # print(losses_seg)
                for k in list(losses_seg.keys()):
                    if k in list(losses.keys()):
                        losses[k] = (losses_seg[k] + losses[k]) / len(batched_inputs)  # 平均loss
                        # losses[k] = losses_seg[k] + losses[k] # 求和loss
                    else:
                        losses[k] = losses_seg[k]
            return losses  # 只保存了最后一个batch数据集的loss
        else:
            # 推理时候只推理一个数据集
            dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs][
                0]  # batched_inputs[0]["meta"]["dataset_name"]
            class_names = self.get_class_name_list(dataset_name)
            scene_name = batched_inputs[0]["meta"]["scene_name"]
            return self.forward_seg(batched_inputs, dataset_name, class_names, scene_name=scene_name)

    def forward_seg(self, batched_inputs, dataset_name, text_labels=None, scene_name=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        images = [x["image"].to(self.device) for x in batched_inputs]
        # clip_images
        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # 更换成IP_CLIP模型（maskCLIP）去一次性推理batch带mask的图片
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        clip_images_480 = F.interpolate(clip_images.tensor, size=(480, 480), mode="bilinear", align_corners=False,)
        # clip_images_480 = torch.vstack(clip_images)
        # clip_images_480 = torch.stack(clip_images, dim=0)
        # clip_images = torch.cat(clip_images, dim = 0).unsqueeze(0)
        # clip_images_480 = F.interpolate(clip_images, size=(480, 480), mode="bilinear", align_corners=False,)
        # clip_images_480 = clip_images_480.reshape(len(batched_inputs), -1, 480, 480)

        features = self.backbone(images.tensor)

        if text_labels == None:
            class_names = self.get_class_name_list(dataset_name)
        else:
            class_names = text_labels

        # print(dataset_name)
        # print(class_names)
        if self.training:
            # ================
            # 使用任务prompt
            # ================
            task_name = random.choice(self.task_names)
            text_features = self.clip_adapter.get_text_features(class_names, task_name)
            # ================
            # 使用数据源或场景prompt
            # ================
            # text_features = self.clip_adapter.get_text_features(class_names, scene_name)

            # ================
            # SAM Branch中间提取的候选mask
            # ================
            if self.sam_branch:
                sam_pre_masks = None
                sam_add_num = 100 # 增加到100，训练时间增加一倍，内存也会增加
                # batchify
                # sam_batched_input = [x['image'].cuda() for x in batched_inputs]
                sam_batched_input = [
                    {
                        'image': images.tensor[idx],
                        'point_coords': self.input_point,
                        'point_labels': self.input_label,
                        'original_size': images.image_sizes[idx]
                    } for idx in range(len(images))
                ]
                # LBK propagation
                # with torch.no_grad():
                refined_masks = self.sam.individual_forward(sam_batched_input, multimask_output=True)
                for b in range(len(refined_masks)):
                    sam_masks = refined_masks[b]
                    # sam_masks = F.interpolate(sam_masks.float().unsqueeze(0), size=(640,640), mode='nearest').squeeze(0) # N,128,128
                    if sam_masks.shape[0] < sam_add_num:
                        # append  zero
                        append_len = sam_add_num - sam_masks.shape[0]
                        sam_masks_zeros = torch.zeros((append_len, sam_masks.shape[1], sam_masks.shape[2]),
                                                      device=self.device)
                        sam_masks = torch.vstack([sam_masks, sam_masks_zeros]).unsqueeze(0)
                    elif sam_masks.shape[0] > sam_add_num:
                        sam_masks = sam_masks[:sam_add_num, :, :].unsqueeze(0)
                    else:
                        sam_masks = sam_masks.unsqueeze(0)
                    for i in range(sam_add_num):
                        #     if torch.sum(sam_masks[0][i])== 262144: # 512*512
                        if torch.sum(sam_masks[0][i]) > 131072:  # 大于512*256的分割结果不要，会导致微调clip时提取的image_feature为nan
                            sam_masks[0][i] = torch.zeros((sam_masks.shape[2], sam_masks.shape[3]),
                                                          device=self.device)
                    if sam_pre_masks is None:
                        sam_pre_masks = sam_masks
                    else:
                        sam_pre_masks = torch.cat((sam_masks, sam_pre_masks), dim=0)



                # 更换成IP_CLIP模型（maskCLIP）去一次性推理batch带mask的图片
                image_features = self.maskCLIP(clip_images_480, sam_pre_masks)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)


                clip_cls = self.clip_adapter.get_sim_logits(text_features, image_features)
                sam_pre_masks = F.interpolate(sam_pre_masks.float(), size=(128,128),
                                              mode='nearest')  # 4,100,128,128

            outputs, fused_text_features = self.sem_seg_head(features, text_features)  # text_features没有变化
            outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
            )  # outputs["pred_logits"]作为图像特征, 输出后得到预测类别

            # add sam branch result
            outputs["pred_logits"] = torch.cat((outputs["pred_logits"], clip_cls), dim=1)
            outputs["pred_masks"] = torch.cat((outputs["pred_masks"], sam_pre_masks), dim=1)

            outputs["empty_weight"] = torch.ones(len(class_names), device=outputs["pred_logits"].device)
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i]["empty_weight"] = torch.ones(len(class_names), device=outputs["pred_logits"].device)
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "sem_instances" in batched_inputs[0]:
                targets = self.freemix_prepare_targets(batched_inputs, images)

            # print(len(outputs["pred_masks"]), len(targets))
            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]

            return losses
        else:
            # 使用任务prompt
            task_name = "semantic segmentation."

            # 使用数据源或场景prompt
            # task_name = scene_name

            text_features = self.clip_adapter.get_text_features(class_names, task_name)

            # SAM Branch中间提取的候选mask
            if self.sam_branch:
                sam_pre_masks = None
                sam_add_num = 100
                # sam_batched_input = [x['image'].cuda() for x in batched_inputs]
                # sam_batched_input = [
                #     {
                #         'image': x,
                #         'point_coords': self.input_point,
                #         'point_labels': self.input_label,
                #         'original_size': x.shape[1:]
                #     } for x in sam_batched_input
                # ]
                sam_batched_input = [
                    {
                        'image': images.tensor[idx],
                        'point_coords': self.input_point,
                        'point_labels': self.input_label,
                        'original_size': images.image_sizes[idx]
                    } for idx in range(len(images))
                ]
                # LBK propagation
                refined_masks = self.sam.individual_forward(sam_batched_input, multimask_output=True)
                for b in range(len(refined_masks)):
                    sam_masks = refined_masks[b]
                    # sam_masks = F.interpolate(sam_masks.unsqueeze(0), scale_factor=0.25, mode='nearest').squeeze(0) # N,128,128
                    if sam_masks.shape[0] < sam_add_num:
                        # append  zero
                        append_len = sam_add_num - sam_masks.shape[0]
                        sam_masks_zeros = torch.zeros((append_len, sam_masks.shape[1], sam_masks.shape[2]),
                                                      device=self.device)
                        sam_masks = torch.vstack([sam_masks, sam_masks_zeros]).unsqueeze(0)
                    elif sam_masks.shape[0] > sam_add_num:
                        sam_masks = sam_masks[:sam_add_num, :, :].unsqueeze(0)
                    else:
                        sam_masks = sam_masks.unsqueeze(0)
                    for i in range(sam_add_num):
                        #     if torch.sum(sam_masks[0][i])== 262144: # 512*512
                        if torch.sum(sam_masks[0][i]) > 131072:  # 大于512*256的分割结果不要，会导致微调clip时提取的image_feature为nan
                            sam_masks[0][i] = torch.zeros((sam_masks.shape[2], sam_masks.shape[3]),
                                                          device=self.device)
                    if sam_pre_masks is None:
                        sam_pre_masks = sam_masks
                    else:
                        sam_pre_masks = torch.cat((sam_masks, sam_pre_masks), dim=0)

                # 更换成IP_CLIP模型（maskCLIP）去一次性推理batch带mask的图片
                image_features = self.maskCLIP(clip_images_480, sam_pre_masks)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                clip_cls = self.clip_adapter.get_sim_logits(text_features, image_features)
                sam_pre_masks = F.interpolate(sam_pre_masks.float(), scale_factor=0.25,
                                              mode='nearest')  # 4,100,512,512 => 4,100,128,128

            outputs, fused_text_features = self.sem_seg_head(features, text_features)
            outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
            )

            if self.sam_branch:
                # add sam branch result
                mask_cls_results = torch.cat((outputs["pred_logits"], clip_cls), dim=1)
                mask_pred_results = torch.cat((outputs["pred_masks"], sam_pre_masks), dim=1)
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=True,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(  # 还原分辨率
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)

                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, task_name, dataset_name
                )
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

            # instance segmentation inference
            if self.instance_on:

                task_name = "instance segmentation."

                text_features = self.clip_adapter.get_text_features(class_names, task_name)

                outputs, fused_text_features = self.sem_seg_head(features, text_features)

                outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                    text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
                )

                if self.sam_branch:
                    # add sam branch result
                    mask_cls_results = torch.cat((outputs["pred_logits"], clip_cls), dim=1)
                    mask_pred_results = torch.cat((outputs["pred_masks"], sam_pre_masks), dim=1)
                else:
                    mask_cls_results = outputs["pred_logits"]
                    mask_pred_results = outputs["pred_masks"]

                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )

                for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                )):
                    height = image_size[0]
                    width = image_size[1]
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )
                    image = input_per_image["image"].to(self.device)

                    instance_r = self.instance_inference(
                        mask_cls_result, mask_pred_result, image, class_names, task_name, dataset_name
                    )

                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    # process results
                    if instance_r.pred_masks.shape[0] > 0:
                        cur_device = instance_r.pred_masks.device
                        instance_mask = instance_r.pred_masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        ori_h, ori_w, num_mask = instance_mask.shape[0], instance_mask.shape[1], instance_mask.shape[2]
                        transform = ResizeTransform(ori_h, ori_w, height, width)

                        if num_mask > 3:
                            instance_mask_list = [transform.apply_segmentation(instance_mask[:, :, p1 - 3:p1]) for p1 in
                                                  range(3, num_mask + 1, 3)]
                            if np.mod(num_mask, 3) > 0:
                                mask_last = transform.apply_segmentation(instance_mask[:, :, -np.mod(num_mask, 3):])
                                instance_mask_list.append(mask_last)
                            instance_mask = np.concatenate(instance_mask_list, axis=2)
                        else:
                            instance_mask = transform.apply_segmentation(instance_mask)

                        instance_mask = torch.tensor(instance_mask).permute(2, 0, 1).to(cur_device)
                        instance_r.pred_masks = instance_mask

                        if not instance_r.pred_boxes is None:
                            instance_boxes = instance_r.pred_boxes.tensor
                            x1_coords, x2_coords = instance_boxes[:, :2], instance_boxes[:, 2:]
                            x1_coords = transform.apply_coords(x1_coords)
                            x2_coords = transform.apply_coords(x2_coords)
                            instance_boxes = torch.cat((x1_coords, x2_coords), dim=1)
                            instance_r.pred_boxes = Boxes(instance_boxes)

                    processed_results[i]["instances"] = instance_r

            # panoptic segmentation inference
            if self.panoptic_on:

                task_name = "panoptic segmentation."

                text_features = self.clip_adapter.get_text_features(class_names, task_name)

                outputs, fused_text_features = self.sem_seg_head(features, text_features)

                outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
                    text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
                )

                if self.sam_branch:
                    # add sam branch result
                    mask_cls_results = torch.cat((outputs["pred_logits"], clip_cls), dim=1)
                    mask_pred_results = torch.cat((outputs["pred_masks"], sam_pre_masks), dim=1)
                else:
                    mask_cls_results = outputs["pred_logits"]
                    mask_pred_results = outputs["pred_masks"]

                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )
                for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                )):
                    height = image_size[0]
                    width = image_size[1]
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )
                    image = input_per_image["image"].to(self.device)

                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result, image, class_names, task_name, dataset_name
                    )

                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    # process results
                    cur_device = panoptic_r[0].device
                    panoptic_mask = panoptic_r[0].cpu().numpy().astype(np.uint8)
                    ori_h, ori_w = panoptic_mask.shape[0], panoptic_mask.shape[1]
                    transform = ResizeTransform(ori_h, ori_w, height, width)
                    panoptic_mask = transform.apply_segmentation(panoptic_mask)
                    panoptic_r[0] = torch.tensor(panoptic_mask).to(cur_device)

                    segment_info = panoptic_r[1]
                    cur_seg_ids = list(torch.unique(panoptic_r[0]))
                    segment_info = [seg_info for seg_info in segment_info if seg_info["id"] in cur_seg_ids]
                    panoptic_r[1] = segment_info
                    processed_results[i]["panoptic_seg"] = panoptic_r

            return processed_results

    def semantic_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()

        # get the classification result from clip model
        # print("self.clip_ensemble",  self.clip_ensemble)
        if self.clip_ensemble:

            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, task_name, mask_pred, normalize=True
            )

            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)

            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)

                mask_cls = trained_mask * torch.pow(mask_cls, self.clip_ensemble_weight) * torch.pow(map_back_clip_cls,
                                                                                                     1 - self.clip_ensemble_weight) \
                           + (1 - trained_mask) * torch.pow(mask_cls, 1 - self.clip_ensemble_weight) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight)
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]

        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()

        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, task_name, mask_pred, normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)

            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)

                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                                   1 - trained_mask
                           ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]

        sem_maps = torch.einsum("qc,qhw->chw", mask_cls, mask_pred).argmax(0)

        scores, labels = F.softmax(mask_cls / 0.01, dim=-1).max(-1)
        keep = labels.ne(self.sem_seg_head.num_classes)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return [panoptic_seg, segments_info]
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                pred_class_name = class_names[pred_class]
                isthing = pred_class_name in self.metadata.thing_classes

                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_masks[k] >= 0.5) & (sem_maps == pred_class)
                mask_area = mask.sum().item()

                if original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    if isthing and cur_scores[k] < 0.5:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            panoptic_res = [panoptic_seg, segments_info]
            return panoptic_res

    def instance_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):

        image_size = mask_pred.shape[-2:]
        num_classes = mask_cls.shape[-1]

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, task_name, mask_pred.sigmoid(), normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)

            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)

                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                                   1 - trained_mask
                           ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]

        sem_maps = torch.einsum("qc,qhw->chw", mask_cls, mask_pred.sigmoid()).argmax(0)

        scores = F.softmax(mask_cls / 0.01, dim=-1)[:, :-1]
        scores_per_image, labels_per_image = scores.max(-1)

        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                pred_class_name = class_names[lab]
                keep[i] = pred_class_name in self.metadata.thing_classes

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        class_mask_memory = {}
        keep = torch.zeros_like(scores_per_image).bool()

        for k in range(labels_per_image.shape[0]):

            pred_class = labels_per_image[k]
            original_area = (mask_pred[k] >= 0.5).sum().item()

            mask = (mask_pred[k] >= 0.5) & (sem_maps == pred_class)
            mask_area = mask.sum().item()

            if mask_area > 0 and original_area > 0 and scores_per_image[k] > 0.5:
                if mask_area / original_area > self.overlap_threshold:
                    keep[k] = True

                    if lab in class_mask_memory.keys():
                        class_mask_memory[lab].append(k)
                    else:
                        class_mask_memory[lab] = [k]

        for cls_id, idx_list in class_mask_memory.items():
            mask_area_list = [(mask_pred[i] >= 0.5).sum().item() for i in idx_list]
            max_area = np.max(np.array(mask_area_list))
            max_idx = np.argmax(np.array(mask_area_list))
            union_mask = torch.zeros_like(mask_pred[0]).bool()
            for i, idx in enumerate(idx_list):
                if i != max_idx:
                    union_mask = (union_mask == True) | (mask_pred[idx] >= 0.5)
            union_mask_area = union_mask.sum().item()
            if union_mask_area / max_area > 0.8:
                keep[idx_list[max_idx]] = False

        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]

        result = Instances(image_size)

        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    def freemix_prepare_targets(self, batched_inputs, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets_per_image = batch_per_image["sem_instances"].to(self.device)
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def get_iou(self, pred, target):
        # pred = pred.sigmoid()
        b, c, h, w = pred.shape
        if len(target.shape) != len(pred.shape):
            target = target.unsqueeze(1)
        # assert pred.shape == target.shape
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=(target.shape[-2], target.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

        pred = pred.reshape(b, c, -1)
        target = target.reshape(b, 1, -1)

        # compute the IoU of the foreground
        Iand1 = torch.sum(target * pred, dim=-1)
        Ior1 = torch.sum(target, dim=-1) + torch.sum(pred, dim=-1) - Iand1 + 0.0000001
        IoU1 = Iand1 / Ior1

        return IoU1

    def mynorm(self, embeding):
        assert len(embeding.shape) == 2, embeding.shape
        min_em, _ = torch.min(embeding, dim=-1)
        max_em, _ = torch.max(embeding, dim=-1)
        embeding = (embeding - min_em.unsqueeze(-1)) / ((max_em - min_em + 0.00000001).unsqueeze(-1))
        return embeding

    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter
