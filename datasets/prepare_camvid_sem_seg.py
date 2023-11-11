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
    31: 30,
    30: 255,
    255: 255
}

novel_clsID = [26, 27, 28, 29, 31]
base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [30, 255]]
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
        description="Convert VOC2021 annotations to mmsegmentation format"
    )  # noqa
    parser.add_argument("--voc_path", help="voc path",
                        default="/media/data2/wjy/datasets/Camvid_fmt-VOC/")
    parser.add_argument("-o", "--out_dir", help="output path")
    parser.add_argument("--nproc", default=16, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    voc_path = args.voc_path
    nproc = args.nproc

    out_dir = args.out_dir or voc_path
    # out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, "annotations_detectron2")
    # out_image_dir = osp.join(out_dir, "images_detectron2")
    for dir_name in [
        "train",
        "val",
        "train_base",
        "train_novel",
        "val_base",
        "val_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)
        # if dir_name in ["train", "val"]:
        #     os.makedirs(osp.join(out_image_dir, dir_name), exist_ok=True)

    train_list = [
        osp.join(voc_path, "SegmentationClass", f)
        for f in np.loadtxt(osp.join(voc_path, "ImageSets/train_val.txt"),dtype=np.str_,encoding='utf-8').tolist()
    ]
    test_list = [
        osp.join(voc_path, "SegmentationClass", f)
        for f in np.loadtxt(osp.join(voc_path, "ImageSets/test.txt"),dtype=np.str_,encoding='utf-8').tolist()
    ]

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

def split_Camvid_vocformat_dataset():
    img_root = "/media/data2/wjy/datasets/CamVid/"
    test_list = [
        osp.join(img_root, "SegmentationClass", f)
        for f in np.loadtxt(osp.join(img_root+'/ImageSets/', "val.txt"),dtype=np.str_,encoding='utf-8').tolist()
    ]
    print(len(test_list))
    val_dst = '/media/data2/wjy/datasets/CamVid/annotations_detectron2/val/'
    train_dst = '/media/data2/wjy/datasets/CamVid/annotations_detectron2/train/'
    for  i in test_list:
        src_path = i
        dst_path = val_dst + os.path.basename(i)
        shutil.copy(src_path,dst_path)
    train_list = [
        osp.join(img_root, "SegmentationClass", f)
        for f in np.loadtxt(osp.join(img_root+'/ImageSets/', "train.txt"),dtype=np.str_,encoding='utf-8').tolist()
    ]
    for  i in train_list:
        src_path = i
        dst_path = train_dst + os.path.basename(i)
        shutil.copy(src_path,dst_path)

if __name__ == "__main__":
    main()
    # split_Camvid_vocformat_dataset()
