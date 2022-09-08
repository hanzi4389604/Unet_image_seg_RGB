"""
*_* coding:utf-8 *_*
time:            2021/11/10 15:59
author:          丁治
remarks：        备注信息
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2.cv2 as cv2
from torchvision.transforms import transforms
from setting.setting import voc_data_dir, img_size


class MyDataSet(Dataset):
    def __init__(self, data_dir=voc_data_dir):
        self.train_dir = os.path.join(data_dir, 'JPEGImages')
        self.target_dir = os.path.join(data_dir, 'SegmentationClass')
        self.dataset = os.listdir(self.target_dir)
        self.trans = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_name = self.dataset[item]
        train_img = Image.open(os.path.join(self.train_dir, f'{img_name[:-3]}jpg'))  # 训练图片
        target_img = Image.open(os.path.join(self.target_dir, img_name))  # 分割标签

        """ 统一尺寸 """
        w, h = train_img.size
        max_border = max(w, h)
        train_img = train_img.resize((int(img_size*w/max_border), int(img_size*h/max_border)))
        target_img = target_img.resize((int(img_size*w/max_border), int(img_size*h/max_border)))

        """ 创建数据和标签 """
        train_img_back = transforms.ToPILImage()(torch.zeros((3, img_size, img_size)))
        target_img_back = transforms.ToPILImage()(torch.zeros((3, img_size, img_size)))
        train_img_back.paste(train_img)
        target_img_back.paste(target_img)
        return self.trans(train_img_back), self.trans(target_img_back)


if __name__ == '__main__':
    train_dataset = MyDataSet(r'/home/l/20211218 practice/data/unet4/unet_voc-master/dataset/VOCdevkit/VOC2007')
    print(train_dataset[0])
    print(len(train_dataset))
