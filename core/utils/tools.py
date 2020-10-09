import os
import shutil
import torch
import logging
import itertools
from glob import glob
from collections import OrderedDict
from sklearn import metrics
import math
import numpy as np
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as transforms
# from .config import *
from copy import deepcopy
import matplotlib.pyplot as plt

# Some default settings
model_savepath = 'checkpoints'
outputs_path = 'outputs'
power = 0.9
cls_thresh = 0.5


def increment_dir(dir, e=None):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    d = sorted(glob(dir + '*'))  # directories
    if len(d):
        n = int(d[-1].split('/')[-1].split('-')[0][-3:]) + 1  # increment
    if e is not None:
        os.makedirs(dir + str(n).zfill(3) + '-' + e, exist_ok=True)
        return dir + str(n).zfill(3) + '-'+e
    else:
        os.makedirs(dir + '-' + str(n).zfill(3), exist_ok=True)
        return dir + str(n).zfill(3) + '-'





def print_result(display_str, result_class, classes):
    num_classes = len(classes)
    display_str = display_str
    for i in range(num_classes):
        if i < num_classes-1:
            display_str += '{} {:.2f}, '.format(classes[i], result_class[i])
        else:
            display_str += '{} {:.2f}'.format(classes[i], result_class[i])
    logging.info(display_str)


def print_thresh_result(display_str, result, thresh, classes):
    display_str = display_str
    for idx in range(len(classes)):
        display_str += classes[idx] + '{'
        for i in range(len(thresh)):
            if i+1 != len(thresh):
                display_str += '{:.2f}({}), '.format(result[idx][i], thresh[i])
            else:
                display_str += '{:.2f}({})'.format(result[idx][i], thresh[i])
        if idx+1 != len(classes):
            display_str += '}; '
        else:
            display_str += '}'
    logging.info(display_str)


def save_checkpoint(net, arch, epoch, base_dir, _best=None, best=0):
    savepath = os.path.join(base_dir, 'checkpoint', arch)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    file_name = os.path.join(savepath, "{}_epoch_{:0>4}".format(arch, epoch) + '.pth')
    torch.save({
        'model': net.state_dict(),
        'epoch': epoch,
    }, file_name)
    remove_flag = False
    if _best:
        best_name = os.path.join(savepath, "{}_best_{}".format(arch, _best) + '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_{}".format(arch, _best) + '.txt'), 'w')
        file.write('arch: {}'.format(arch)+'\n')
        file.write('epoch: {}'.format(epoch)+'\n')
        file.write('best {}: {}'.format(_best, best)+'\n')
        file.close()
    if remove_flag:
        os.remove(file_name)


def load_checkpoint(net, model_path, _sgpu=True):
    state_dict = torch.load(model_path)
    if _sgpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            # print(k)
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            head = k[:7]
            if head != 'module.':
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    logging.info('Finish loading resume network')


def set_requires_grad(net, fixed_layer, _sgpu=True):
    update_flag = {}
    for name, _ in net.named_parameters():
        # print(name)
        update_flag[name] = 0
        for item in fixed_layer:
            if _sgpu:
                if name[:len(item)] == item:
                    # print('hehe')
                    update_flag[name] = 1
            else:
                if name[7:7+len(item)] == item:
                    # print('hehe')
                    update_flag[name] = 1

    for name, param in net.named_parameters():
        # print(name)
        if update_flag[name] == 1:
            param.requires_grad = False
        else:
            param.requires_grad = True


def adjust_learning_rate(optimizer, epoch, epoch_num, initial_lr, reduce_epoch, decay=0.1):
    if reduce_epoch == 'dynamic':
        lr = initial_lr * (1 - math.pow(float(epoch)/float(epoch_num), power))
    else:
        lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sum_flag=True):
        if sum_flag:
            self.val = val
            self.sum += val * n
        else:
            self.val = val / n
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count
