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


def plot_confusion_matrix(cm, classes, save_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, format='tif', dpi=300)
    plt.close()


def get_one_hot(data, num_classes):
    data = data.reshape(-1)
    data = np.eye(num_classes)[data]
    return data


def calculate_auc(y_pred, y_gt, config, show_roc_curve=False):
    num_classes = len(config['Data_CLASSES'])
    '''calculate the mean AUC'''
    y_true_one_hot = get_one_hot(y_gt, num_classes)

    auc_each_class = []
    nan_index = []

    mean_auc = 0

    for index in range(num_classes):

        # tmp_y_pred = y_pred[y_gt == index, :]
        # tmp_y_gt = y_gt_one_hot[y_gt == index, :]
        tmp_y_pred = y_pred
        tmp_y_true = y_true_one_hot

        preds = tmp_y_pred[:, index]
        label = tmp_y_true[:, index]

        fpr, tpr, thresholds = metrics.roc_curve(label, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        if show_roc_curve:
            os.makedirs(os.path.join(config['base_dir'], 'roc'), exist_ok=True)
            if index == 1:
                plt.figure()
                lw = 2
                plt.figure(figsize=(10, 10))
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % auc)  # 假正率为横坐标，真正率为纵坐标做曲线
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(config['base_dir'], 'roc', 'roc.png'), dpi=100)
                plt.close()

        if math.isnan(auc):
            nan_index.append(index)
            auc = 0.0

        auc_each_class.append(auc)
        mean_auc += auc

    mean_auc = float(mean_auc) / float(num_classes)

    return mean_auc, auc_each_class


def calculate_f1(output, target, config):
    num_classes = len(config['Data_CLASSES'])
    '''calculate the f1 score'''
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    y_pred = np.argmax(y_pred, axis=1)
    y_true_one_hot = get_one_hot(y_true, num_classes)
    y_pred_one_hot = get_one_hot(y_pred, num_classes)

    f1_each_class = []
    nan_index = []
    mean_f1 = 0.0

    for index in range(num_classes):

        preds = y_pred_one_hot[:, index]
        label = y_true_one_hot[:, index]

        f1 = metrics.f1_score(label, preds, average='binary')

        if math.isnan(f1):
            nan_index.append(index)
            f1 = 0.0

        f1_each_class.append(f1)
        mean_f1 += f1

    # mean_f1 = metrics.f1_score(y_gt.flatten(), y_pred.flatten(), average='binary')
    mean_f1 = mean_f1 / num_classes

    return mean_f1, f1_each_class


def calculate_bac(output, target, config):
    num_classes = len(config['Data_CLASSES'])
    '''calculate the bac score'''
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    y_pred = np.argmax(y_pred, axis=1)
    y_true_one_hot = get_one_hot(y_true, num_classes)
    y_pred_one_hot = get_one_hot(y_pred, num_classes)

    bac_each_class = []
    nan_index = []
    mean_bac = 0.0

    for index in range(num_classes):

        preds = y_pred_one_hot[:, index]
        label = y_true_one_hot[:, index]

        bac = metrics.balanced_accuracy_score(label, preds)

        if math.isnan(bac):
            nan_index.append(index)
            bac = 0.0

        bac_each_class.append(bac)
        mean_bac += bac

    # mean_f1 = metrics.f1_score(y_gt.flatten(), y_pred.flatten(), average='binary')
    mean_bac = mean_bac / num_classes

    return mean_bac, bac_each_class

def calculate_acc(output, target, config):
    num_classes = len(config['Data_CLASSES'])
    '''calculate the acc score'''
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    y_pred = np.argmax(y_pred, axis=1)
    y_true_one_hot = get_one_hot(y_true, num_classes)
    y_pred_one_hot = get_one_hot(y_pred, num_classes)

    acc_each_class = []
    nan_index = []
    mean_acc = 0.0

    for index in range(num_classes):

        preds = y_pred_one_hot[:, index]
        label = y_true_one_hot[:, index]

        acc = metrics.accuracy_score(label, preds)

        if math.isnan(acc):
            nan_index.append(index)
            acc = 0.0

        acc_each_class.append(acc)
        mean_acc += acc

    # mean_f1 = metrics.f1_score(y_gt.flatten(), y_pred.flatten(), average='binary')
    mean_acc = mean_acc / num_classes

    return mean_acc, acc_each_class


# def accuracy(output, target, config):
#     num_classes = len(config['Data_CLASSES'])
#     """Computes the precision@k for the specified values of k"""
#     y_pred = deepcopy(output)
#     y_true = deepcopy(target)
#     y_pred = np.argmax(y_pred, axis=1)
#     y_pred_one_hot = get_one_hot(y_pred, num_classes)
#     y_true_one_hot = get_one_hot(y_true, num_classes)

#     error_map = np.equal(y_pred_one_hot, y_true_one_hot)
#     acc_each_class = []
#     nan_index = []
#     mean_acc = 0.0
#     bs = float(output.shape[0])

#     for class_index in range(num_classes):
#         class_error_map = error_map[:, class_index]
#         acc = np.sum(class_error_map) / bs
#         if math.isnan(acc):
#             nan_index.append(class_index)
#             acc = 0.0
#         acc_each_class.append(acc)
#         mean_acc += acc

#     mean_acc = mean_acc / num_classes

#     # mean_acc = metrics.balanced_accuracy_score(output, target)
#     # acc = []
#     #
#     # one_hot_output = get_one_hot(output, num_classes)
#     # one_hot_target = get_one_hot(target, num_classes)

#     return mean_acc, acc_each_class


def recall(output, target, config, show_confusion_matrix=False):
    num_classes = len(config['Data_CLASSES'])
    """Computes the precision@k for the specified values of k"""
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    y_pred = np.argmax(y_pred, axis=1)
    classes = config['Data_CLASSES']
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=list(range(7)))
    # y_pred_one_hot = get_one_hot(y_pred, num_classes)
    # y_true_one_hot = get_one_hot(y_true, num_classes)

    if show_confusion_matrix:
        # calc confusion matrix
        save_path = os.path.join(config['base_dir'], 'confusion_matrix', 'smei_label{}.tif'.format(config['label_fold']))
        os.makedirs(os.path.join(config['base_dir'], 'confusion_matrix'), exist_ok=True)
        plot_confusion_matrix(cm, classes, save_path)

    recall_each_class = []
    nan_index = []
    mean_recall = 0.
    counter = num_classes
    # bs = float(output.shape[0])

    for class_index in range(num_classes):
        recall = cm[class_index, class_index] / np.sum(cm[class_index, :])
        if math.isnan(recall):
            nan_index.append(class_index)
            recall = 0.0
            counter -= 1
        recall_each_class.append(recall)
        mean_recall += recall

    mean_recall = mean_recall / counter

    return mean_recall, recall_each_class