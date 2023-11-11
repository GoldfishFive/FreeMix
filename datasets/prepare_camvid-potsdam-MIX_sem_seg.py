import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmengine
import numpy as np
from PIL import Image


# 类别id转到训练id
full_clsID_to_trID = {
    0 : 0 ,
    1 : 1 ,
    2 : 2 ,
    3 : 3 ,
    4 : 4 ,
    5 : 5 ,
    6 : 6 ,
    7 : 7 ,
    8 : 8 ,
    9 : 9 ,
    10 : 10 ,
    11 : 11 ,
    12 : 12 ,
    13 : 13 ,
    14 : 14 ,
    15 : 15 ,
    16 : 16 ,
    17 : 17 ,
    18 : 18 ,
    19 : 19 ,
    20 : 20 ,
    21 : 21 ,
    22 : 22 ,
    23 : 23 ,
    24 : 24 ,
    25 : 25 ,
    26 : 26 ,
    27 : 27 ,
    28 : 28 ,
    29 : 29 ,
    30 : 30, #转换过后的Camvid
    31 : 31, #转换过后的potsdam
    32 : 32, #转换过后的potsdam
    33 : 33, #转换过后的potsdam
    255: 255
}

novel_clsID = [26, 27, 28, 29, 30, 32, 33] # Camvid_novel_class = 26, 27, 28, 29, 30; Potsdam_novel_class = 26, 32, 33
base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [255]]
novel_clsID_to_trID = {k: i for i, k in enumerate(novel_clsID)}
base_clsID_to_trID = {k: i for i, k in enumerate(base_clsID)}
print(full_clsID_to_trID)
print(base_clsID_to_trID)
print(novel_clsID_to_trID)

# exit()

def convert_to_trainID(
    maskpath, out_mask_dir, is_train, clsID_to_trID=full_clsID_to_trID, suffix=""
):
    """
    将标注标签转换成训练对应的标签，只包含背景0标签的样本不保留
    #参与训练的类才有对应标注，不可见类别全部设置为0
    """
    mask = np.array(Image.open(maskpath))
    mask_copy = np.ones_like(mask, dtype=np.uint8) * 255
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = (
        osp.join(out_mask_dir, "train" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_mask_dir, "val" + suffix, osp.basename(maskpath))
    )
    if len(np.unique(mask_copy)) == 1 and np.unique(mask_copy)[0] == 255:
        return
    Image.fromarray(mask_copy).save(seg_filename, "PNG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Camvid-potsdam annotations to mmsegmentation format"
    )  # noqa
    parser.add_argument("--voc_path", help="voc path")
    parser.add_argument("-o", "--out_dir", help="output path",
                        default="/media/data2/wjy/datasets/CamVid_Potsdam_Mix/")
    parser.add_argument("--nproc", default=16, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    voc_path = args.voc_path
    nproc = args.nproc

    out_dir = args.out_dir or voc_path
    out_mask_dir = osp.join(out_dir, "annotations_detectron2")
    for dir_name in [
        "train",
        "val",
        "train_base",
        "train_novel",
        "val_base",
        "val_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)

    camvid_train = "/media/data2/wjy/datasets/Camvid_fmt-VOC/annotations_detectron2/train"
    camvid_val = "/media/data2/wjy/datasets/Camvid_fmt-VOC/annotations_detectron2/val"
    potsdam_train = "/media/data2/wjy/datasets/Potsdam_fmt-VOC/annotations_map2Camvid/annotations_detectron2/train"
    potsdam_val = "/media/data2/wjy/datasets/Potsdam_fmt-VOC/annotations_map2Camvid/annotations_detectron2/val"

    train_list = [osp.join(camvid_train, f) for f in os.listdir(camvid_train)] + [osp.join(potsdam_train, f) for f in os.listdir(potsdam_train)]
    test_list = [osp.join(camvid_val, f) for f in os.listdir(camvid_val)] + [osp.join(potsdam_val, f) for f in os.listdir(potsdam_val)]

    if args.nproc > 1:
        mmengine.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
            nproc=nproc,
        )
        mmengine.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
            nproc=nproc,
        )
        mmengine.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
            nproc=nproc,
        )
        mmengine.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
            nproc=nproc,
        )
        mmengine.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
            nproc=nproc,
        )
        mmengine.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
            nproc=nproc,
        )
    else:
        mmengine.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
        )
        mmengine.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
        )
        mmengine.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
        )
        mmengine.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
        )
        mmengine.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
        )
        mmengine.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
        )
    print("Done!")


if __name__ == "__main__":
    main()
