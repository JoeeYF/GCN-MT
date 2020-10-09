import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import pickle
from utils.config import *
from itertools import cycle
from sklearn.metrics import roc_curve, auc


def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def arg_parse():
    parser = argparse.ArgumentParser(description='PlotROC')
    parser.add_argument('-cfg', '--config', default='configs/valid/se50_fold0.yaml',
                        type=str, help='load the config file')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    config = yaml.load(open(args.config))
    pkl_path = os.path.join(outputs_path, 'valid_result', config['Valid_Result'])
    pkl_data = read_pkl(pkl_path)
    y_pred = pkl_data['preds']
    y_true = pkl_data['label']
    n_classes = len(config['Data_CLASSES'])
    fold = args.config.split('.')[0].split('/')[-1].split('_')[-1]

    # calc every class ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    colors = cycle(['hotpink', 'darkorange', 'cornflowerblue', 'lawngreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0:8} (area = {1:0.3f})'.format(config['Data_CLASSES'][i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of knee cartilage classification')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outputs_path, 'roc', 'roc_4class_{}.png'.format(fold)), dpi=200)
    # plt.show()


if __name__ == '__main__':
    main()