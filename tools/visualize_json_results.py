#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from mask2former.data.datasets import register_Camvid_seg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    labels = np.asarray([predictions[i]["category_id"] for i in range(len(predictions))])
    ret.pred_classes = labels
    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in range(len(predictions))]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", help="JSON file produced by the model",
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/camvid/Camvid_mask2former_R50_bs16_20k_semantic/inference/sem_seg_predictions.json")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/camvid/Baseline_Camvid-base_mask2former_R50_bs16_20k/inference/sem_seg_predictions.json")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/camvid/Baseline_Camvid-base_mask2former_R50_bs16_20k_learned_prompt/inference/sem_seg_predictions.json")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/campot/Baseline_CamPot-base_mask2former_R50_bs16_20k_learned_prompt/inference/sem_seg_predictions.json")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/potsdam/Baseline_Potsdam-base_mask2former_R50_bs16_20k/inference/sem_seg_predictions.json")
                        default="/media/data1/wjy/projects/FreeSeg/freeseg_output/potsdam/Potsdam_mask2former_R50_bs16_20k_semantic/inference/sem_seg_predictions.json")
    parser.add_argument("--output", help="output directory",
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/camvid/Camvid_mask2former_R50_bs16_20k_semantic/inference/vis_result")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/camvid/Baseline_Camvid-base_mask2former_R50_bs16_20k/inference/")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/camvid/Baseline_Camvid-base_mask2former_R50_bs16_20k_learned_prompt/inference")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/campot/Baseline_CamPot-base_mask2former_R50_bs16_20k_learned_prompt/inference")
                        # default="/media/data1/wjy/projects/FreeSeg/freeseg_output/potsdam/Baseline_Potsdam-base_mask2former_R50_bs16_20k/inference")
                        default="/media/data1/wjy/projects/FreeSeg/freeseg_output/potsdam/Potsdam_mask2former_R50_bs16_20k_semantic/inference")
    parser.add_argument("--dataset", help="name of the dataset",
                        # default="camvid_sem_seg_test")
                        # default="campot_sem_seg_test")
                        default="potsdam_sem_seg_test")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    # logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["file_name"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)

    args.output = args.output+"/vis_result"
    os.makedirs(args.output, exist_ok=True)
    # exit()

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["file_name"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
