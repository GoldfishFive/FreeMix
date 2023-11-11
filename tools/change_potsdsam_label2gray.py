import cv2
import os


ann_path = "/media/data2/wjy/datasets/Potsdam_fmt-VOC/annotations_detectron2"

for d in os.listdir(ann_path):
    for i in os.listdir(os.path.join(ann_path, d)):
        img_path = os.path.join(ann_path, d,i)
        img = cv2.imread(img_path, -1)
        print(img.shape)
        if len(img.shape) ==3:
            img = img[:,:,0]
            cv2.imwrite(img_path, img)

for d in os.listdir(ann_path):
    for i in os.listdir(os.path.join(ann_path, d)):
        img_path = os.path.join(ann_path, d,i)
        img = cv2.imread(img_path, -1)
        print(img.shape)
