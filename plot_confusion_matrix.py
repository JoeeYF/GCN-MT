import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import pickle
from utils.config import *
from itertools import cycle
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools


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
    plt.savefig(save_path, format='tif', dpi=100)
    plt.close()


def main():
    args = arg_parse()
    config = yaml.load(open(args.config))
    pkl_path = os.path.join(outputs_path, 'valid_result', config['Valid_Result'])
    pkl_data = read_pkl(pkl_path)
    y_pred = pkl_data['preds']
    y_true = pkl_data['label']
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    classes = config['Data_CLASSES']
    fold = args.config.split('.')[0].split('/')[-1].split('_')[-1]

    # calc confusion matrix
    save_path = os.path.join(outputs_path, 'confusion_matrix', 'cm_4class_{}.tif'.format(fold))
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plot_confusion_matrix(cm, classes, save_path)


if __name__ == '__main__':
    main()
