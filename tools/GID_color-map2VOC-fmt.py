# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import mmengine
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('--dataset_path',
                        default="/media/data2/wjy/datasets/GID/Large-scale_Classification_5classes/image_RGB", # 最后不要“/”
                        # default="/media/data2/wjy/datasets/GID/Fine_land-cover_Classification_15classes/image_RGB", # 最后不要“/”
                        help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', default="/media/data2/wjy/datasets/GID_5_fmt_VOC/", help='output path')
    # parser.add_argument('-o', '--out_dir', default="/media/data2/wjy/datasets/GID_15_fmt_VOC/", help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args

# for 15 classes
# label_information_of_15_classes_RGB = {
#     "industrial land": [200, 0, 0],
#     "urban residential": [250, 0, 150],
#     "rural residential": [200, 150, 150], # 2 unseen
#     "traffic land": [250, 150, 150],
#     "paddy field": [0, 200, 0],
#     "irrigated land": [150, 250, 0],
#     "dry cropland": [150, 200, 150], # 6 unseen
#     "garden plot": [200, 0, 200],
#     "arbor woodland": [150, 0, 250],
#     "shrub land": [150, 150, 250],
#     "natural grassland": [250, 200, 0],
#     "artificial grassland": [200, 200, 0],
#     "river": [0, 0, 200],
#     "lake": [0, 150, 200],
#     "pond": [0, 200, 250] # 14 unseen
# }
# s = set()
# matmul_map = np.array([5,7,31])
# for k,v in label_information_of_15_classes_RGB.items():
#     s.add(np.matmul(np.asarray(v), matmul_map.reshape(3,1))[0])
# assert len(s) == len(label_information_of_15_classes_RGB)
# color_map =  [v for v in label_information_of_15_classes_RGB.values()] #RGB

# #for 5 classes
label_information_of_5_classes_RGB = {
    "built-up": [255, 0, 0],
    "farmland": [0, 255, 0],
    "forest": [0, 255, 255],
    "meadow": [255, 255, 0], # 草 甸
    "water": [0, 0, 255], # unseen
}
s = set()
matmul_map = np.array([2,3,4])
for k,v in label_information_of_5_classes_RGB.items():
    s.add(np.matmul(np.asarray(v), matmul_map.reshape(3,1))[0])
assert len(s) == len(label_information_of_5_classes_RGB)
color_map =  [v for v in label_information_of_5_classes_RGB.values()] #RGB

print(color_map)
def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    image = mmcv.imread(image_path, channel_order='rgb')

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
        (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
        (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size # 没有重叠
    ymin = y * clip_size #没有重叠
    # xmin = x * stride_size  # 重叠stride_size
    # ymin = y * stride_size # 重叠stride_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
        axis=1)

    if to_label:
        flatten_v = np.matmul(
            image.reshape(-1, c),
            matmul_map.reshape(3, 1))
        # out = np.zeros_like(flatten_v) # 初始化全是0
        out = np.ones_like(flatten_v, np.uint8) * 255 # 初始化全是255
        for idx, class_color in enumerate(color_map): # form 1 to class_num
            value_idx = np.matmul(class_color,
                                  matmul_map.reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                        start_x:end_x] if to_label else image[
                                                        start_y:end_y, start_x:end_x, :]
        idx_i = osp.basename(image_path).split('_')[3][:13]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{idx_i}_{start_x}_{end_x}_{start_y}_{end_y}.png'))


def main():
    args = parse_args()
    # 15 class:  8 for training; 2 for testing
    if "15classes" in args.dataset_path:
        splits = {
            'train': [
                "L1A0001064454", "L1A0001118839", "L1A0001395956", "L1A0000718813",
                "L1A0001378501", "L1A0001471436", "L1A0001517494", "L1A0001787564",
            ],
            'val': [
                "L1A0001680858", "L1A0001821754"
            ]
        }
    # 5 class :  120 for training; 30 for testing
    elif "5classes" in args.dataset_path:
        img_list = sorted(os.listdir(args.dataset_path))
        np.random.seed(42)
        inds = np.random.permutation(np.arange(len(img_list)))
        total_train_examples = int(0.8 * len(img_list))
        train_inds = inds[:total_train_examples]
        test_inds = inds[total_train_examples:]
        splits = {
            "train": [img_list[ind].split("_")[3][:13] for ind in train_inds],
            "vat": [img_list[ind].split("_")[3][:13] for ind in test_inds]
        }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'GID')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmengine.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    src_path_list = glob.glob(os.path.join(dataset_path, '*.tif'))
    # src_path_list = []
    if "15classes" in args.dataset_path:
        # print(os.path.join(os.path.dirname(dataset_path), "label_15classes", '*.tif'))
        # print( glob.glob(os.path.join(os.path.dirname(dataset_path), "label_15classes", '*.tif')))
        src_path_list += glob.glob(os.path.join(os.path.dirname(dataset_path), "label_15classes", '*.tif'))
    else:
        src_path_list += glob.glob(os.path.join(os.path.dirname(dataset_path), "label_5classes", '*.tif'))

    print(src_path_list,len(src_path_list))

    prog_bar = mmengine.ProgressBar(len(src_path_list))
    for i, src_path in enumerate(src_path_list):
        idx_i = osp.basename(src_path).split('_')[3][:13]
        data_type = 'train' if f'{idx_i}' in splits['train'] else 'val'
        if 'label' in src_path:
            dst_dir = osp.join(out_dir, 'ann_dir', data_type)
            clip_big_image(src_path, dst_dir, args, to_label=True)
        else:
            dst_dir = osp.join(out_dir, 'img_dir', data_type)
            clip_big_image(src_path, dst_dir, args, to_label=False)
        prog_bar.update()

    print('Done!')


if __name__ == '__main__':
    main()
    # exit()
