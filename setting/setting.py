"""
*_* coding:utf-8 *_*
time:            2021/11/10 15:59
author:          丁治
remarks：        备注信息
"""
import os


BASEDIR = os.path.dirname(os.path.dirname(__file__))
voc_data_dir = r'/home/l/20211218 practice/data/20220528_mask_rCNN/unet_voc-master/dataset/VOCdevkit/VOC2007'
img_size = 224
best_params_path = os.path.join(BASEDIR, 'static_resources/save_model/best_params')
last_params_path = os.path.join(BASEDIR, 'static_resources/save_model/last_params')
train_log_dir = os.path.join(BASEDIR, 'static_resources/tensorboard_log/train_log')
is_save_img = True  # 是否保存图片进行查看
