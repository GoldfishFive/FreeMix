import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmengine
import numpy as np
from PIL import Image


# # 类别id转到训练id
# full_clsID_to_trID = {
#     0:255,
#     1: 0,
#     2: 1,
#     3: 2,
#     4: 3,
#     5: 4,
#     6: 5,
#     255: 255
# }
# full_clsID_to_trID = {
#     0:255,
#     1: 31,
#     2: 4,
#     3: 32,
#     4: 26,
#     5: 5,
#     6: 33,
#     255: 255
# } # mix with camvid
full_clsID_to_trID = {
    0: 255,
    1: 5,
    2: 6,
    3: 7,
    4: 8,
    5: 9,
    6: 10,
    255: 255
} # mix with GID5
base_clsID_to_trID = {
    1: 3,
    2: 4,
    5: 5
}# mix with GID5
novel_clsID_to_trID = {
    3: 2,
    4: 3,
    6: 4
}# mix with GID5
novel_clsID = [3,4,6]
base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [0, 255]]
# novel_clsID_to_trID = {k: i for i, k in enumerate(novel_clsID)}
# base_clsID_to_trID = {k: i for i, k in enumerate(base_clsID)}

print(full_clsID_to_trID)
print(base_clsID_to_trID)
print(novel_clsID_to_trID)


def convert_to_trainID(
    maskpath, out_mask_dir, is_train, clsID_to_trID=full_clsID_to_trID, suffix=""
):
    """
    将标注标签转换成训练对应的标签，只包含背景0标签的样本不保留
    #参与训练的类才有对应标注，不可见类别全部设置为255
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
        description="Convert Potsdam annotations to detectron2 format"
    )  # noqa
    parser.add_argument("--voc_path", help="voc path",
                        default="/media/data2/wjy/datasets/Potsdam_fmt-VOC/")
    parser.add_argument("-o", "--out_dir", help="output path",
                        # default="/media/data2/wjy/datasets/Potsdam_fmt-VOC/")
                        default="/media/data2/wjy/datasets/Potsdam_fmt-VOC/annotations_map2GID5/")
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
    for dir_name in [
        "train",
        "val",
        "train_base",
        "train_novel",
        "val_base",
        "val_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)

    train_list = [
        osp.join("/media/data2/wjy/datasets/Potsdam_fmt-VOC/ann_dir/", 'train', f)
        for f in os.listdir(osp.join("/media/data2/wjy/datasets/Potsdam_fmt-VOC/ann_dir/", "train"))
    ]
    test_list = [
        osp.join("/media/data2/wjy/datasets/Potsdam_fmt-VOC/ann_dir/", 'val', f)
        for f in os.listdir(osp.join("/media/data2/wjy/datasets/Potsdam_fmt-VOC/ann_dir/", "val"))
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



if __name__ == "__main__":
    main()
