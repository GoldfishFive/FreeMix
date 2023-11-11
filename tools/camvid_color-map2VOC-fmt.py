import os
import shutil
import numpy as np
import cv2
import mmcv

imgRoot = "/media/data1/wjy/projects/mmsegmentation/data/CamVid/ImageSets"

#复制数据，构造txt，划分训练集、验证集和测试集。
for i in ["train","val","test"]:
    imgPath = os.path.join("/media/data1/wjy/projects/dataset/Camvid","camvid_"+i)
    with open(imgRoot+"/"+i+'.txt',"w") as f:
        for j in os.listdir(imgPath):
            img = os.path.join("/media/data1/wjy/projects/mmsegmentation/data/CamVid/JPEGImages", j)
            # label = os.path.join("/media/data1/wjy/projects/mmsegmentation/data/CamVid/SegmentationClass", j)
            print(img)
            # f.write(img+" "+label+'\n')
            f.write(j+'\n')

            shutil.copy(os.path.join(imgPath, j),img)

    for j in os.listdir(os.path.join("/media/data1/wjy/projects/dataset/Camvid","camvid_"+i+'_labels')):
        sour = os.path.join("/media/data1/wjy/projects/dataset/Camvid","camvid_"+i+'_labels', j)
        shutil.copy(sour,os.path.join('/media/data1/wjy/projects/mmsegmentation/data/CamVid/SegmentationClass', j[:-6]+'.png'))

# 将3通道的彩色标签转成单通道的灰度标签
iSAID_palette = \
    {
    0: (64, 128, 64) ,
    1: (192, 0, 128) ,
    2: (0, 128, 192) ,
    3: (0, 128, 64) ,
    4: (128, 0, 0) ,
    5: (64, 0, 128) ,
    6: (64, 0, 192) ,
    7: (192, 128, 64) ,
    8: (192, 192, 128) ,
    9: (64, 64, 128) ,
    10: (128, 0, 192) ,
    11: (192, 0, 64) ,
    12: (128, 128, 64) ,
    13: (192, 0, 192) ,
    14: (128, 64, 64) ,
    15: (64, 192, 128) ,
    16: (64, 64, 0) ,
    17: (128, 64, 128) ,
    18: (128, 128, 192) ,
    19: (0, 0, 192) ,
    20: (192, 128, 128) ,
    21: (128, 128, 128) ,
    22: (64, 128, 192) ,
    23: (0, 0, 64) ,
    24: (0, 64, 64) ,
    25: (192, 64, 128) ,
    26: (128, 128, 0) ,
    27: (192, 128, 192) ,
    28: (64, 0, 64) ,
    29: (192, 192, 0) ,
    30: (0, 0, 0) ,
    31: (64, 192, 0) ,
    }

iSAID_invert_palette = {v: k for k, v in iSAID_palette.items()}

def iSAID_convert_from_color(arr_3d, palette=iSAID_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d
for i in os.listdir('/media/data1/wjy/projects/mmsegmentation/data/CamVid/SegmentationClass'):
    src_path = os.path.join('/media/data1/wjy/projects/mmsegmentation/data/CamVid/SegmentationClass',i)
    print(i)
    label = mmcv.imread(src_path, channel_order='rgb')
    label = iSAID_convert_from_color(label)
    mmcv.imwrite(label, src_path)



# li = os.listdir('/media/data1/wjy/projects/mmsegmentation/data/CamVid/JPEGImages')
# for i in os.listdir("/media/data1/wjy/projects/mmsegmentation/data/CamVid/SegmentationClass"):
#     if i not in li:
#         print(i)
#

# l = [(64, 128, 64), (192, 0, 128), (0, 128, 192), (0, 128, 64), (128, 0, 0),
#     (64, 0, 128), (64, 0, 192), (192, 128, 64), (192, 192, 128),
#                (64, 64, 128), (128, 0, 192), (192, 0, 64), (128, 128, 64),
#                (192, 0, 192), (128, 64, 64), (64, 192, 128), (64, 64, 0),
#                (128, 64, 128), (128, 128, 192), (0, 0, 192), (192, 128, 128),
#                (128, 128, 128), (64, 128, 192), (0, 0, 64), (0, 64, 64),
#                (192, 64, 128), (128, 128, 0), (192, 128, 192), (64, 0, 64),
#                (192, 192, 0), (0, 0, 0), (64, 192, 0)]
# for i in range(len(l)):
#     print(str(i)+":",l[i],",")


# img = cv2.imread('0016E5_07981.png',-1)
# print(img.shape)
# for i in range(32):
#     print("["+str(i)+"],",end="")
#
# exit()
# iSAID_palette = \
#     {
#     0: (64, 128, 64) ,
#     1: (192, 0, 128) ,
#     2: (0, 128, 192) ,
#     3: (0, 128, 64) ,
#     4: (128, 0, 0) ,
#     5: (64, 0, 128) ,
#     6: (64, 0, 192) ,
#     7: (192, 128, 64) ,
#     8: (192, 192, 128) ,
#     9: (64, 64, 128) ,
#     10: (128, 0, 192) ,
#     11: (192, 0, 64) ,
#     12: (128, 128, 64) ,
#     13: (192, 0, 192) ,
#     14: (128, 64, 64) ,
#     15: (64, 192, 128) ,
#     16: (64, 64, 0) ,
#     17: (128, 64, 128) ,
#     18: (128, 128, 192) ,
#     19: (0, 0, 192) ,
#     20: (192, 128, 128) ,
#     21: (128, 128, 128) ,
#     22: (64, 128, 192) ,
#     23: (0, 0, 64) ,
#     24: (0, 64, 64) ,
#     25: (192, 64, 128) ,
#     26: (128, 128, 0) ,
#     27: (192, 128, 192) ,
#     28: (64, 0, 64) ,
#     29: (192, 192, 0) ,
#     30: (0, 0, 0) ,
#     31: (64, 192, 0) ,
#     }
#
# iSAID_invert_palette = {v: k for k, v in iSAID_palette.items()}
#
# def iSAID_convert_from_color(arr_3d, palette=iSAID_invert_palette):
#     """RGB-color encoding to grayscale labels."""
#     arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
#
#     for c, i in palette.items():
#         m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
#         arr_2d[m] = i
#
#     return arr_2d
#
# for i in os.listdir('/media/data1/wjy/projects/mmsegmentation/data/CamVid/SegmentationClass'):
#     src_path = os.path.join('/media/data1/wjy/projects/mmsegmentation/data/CamVid/SegmentationClass',i)
#     print(i)
#     label = mmcv.imread(src_path, channel_order='rgb')
#     label = iSAID_convert_from_color(label)
#     mmcv.imwrite(label, src_path)
