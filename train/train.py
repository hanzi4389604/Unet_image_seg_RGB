"""
*_* coding:utf-8 *_*
time:            2021/11/10 15:59
author:          丁治
remarks：        备注信息
"""
import os

import torch
from net.nets import UNet
from dataset.dataset import MyDataSet
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
import numpy as np
from setting.setting import best_params_path, last_params_path, train_log_dir, is_save_img
from torch.utils.tensorboard import SummaryWriter


batch_size = 4
lr = 0.0001


if __name__ == '__main__':
    summary_writer = SummaryWriter(log_dir=train_log_dir)
    train_dataset = MyDataSet()
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = UNet().cuda()
    if os.path.exists(best_params_path):
        model.load_state_dict(torch.load(best_params_path))
        print('init params load success')
    else:
        print('no params to init')
    loss_fun = BCELoss()
    opt = Adam(model.parameters(), lr=lr)

    epoch = 0
    global_step = 0
    best_params = None
    while True:
        epoch += 1
        epoch_loss = []

        model.train()
        for batch_index, (x, y) in enumerate(train_dataLoader):
            batch_index += 1
            global_step += 1
            x, y = x.cuda(), y.cuda()
            y_pre = model(x)
            loss = loss_fun(y_pre, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'epoch: {epoch}/{batch_index}  loss ==> {loss.item()}')
            summary_writer.add_scalar('loss', loss.item(), global_step=global_step)
            epoch_loss.append(loss.item())

            if is_save_img:
                """ 保存图片查看效果 """
                img_list = []
                for i in range(x.shape[0]):
                    img_list.append(torch.stack((x[i], y[i], y_pre[i]), dim=0))
                img_list = torch.cat(img_list, dim=0)
                img_save_dir = f'img/epoch={epoch}'
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                save_image(img_list, os.path.join(img_save_dir, f'{batch_index}.png'))

        """ 模型保存 """
        if epoch == 1:
            best_params = np.mean(epoch_loss)
        elif np.mean(epoch_loss) < best_params:
            best_params = np.mean(epoch_loss)
            torch.save(model.state_dict(), best_params_path)
            print('best model save success')
        torch.save(model.state_dict(), last_params_path)








