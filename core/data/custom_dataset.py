import os
import os.path as osp
import torch
import torch.utils.data as data
import numpy as np
import h5py as h5
import time
import pickle
import matplotlib.pyplot as plt
import random
import pandas as pd
import itertools
import cv2
from copy import deepcopy
from scipy.ndimage import filters
from torch.utils.data.sampler import Sampler
import albumentations
label_to_num = {
    'BackGround':        0,
    'femur':             1,
    'femur_cartilage':   2,
    'L_meniscus':        3,
    'R_meniscus':        4,
    'patella':           5,
    'patella_cartilage': 6,
}
num_to_label = {v: k for k, v in label_to_num.items()}


def relabel_dataset(dataset, labels):
    labeled_idxs = dataset.path_df.image.isin(labels)
    labeled_idxs = labeled_idxs[labeled_idxs == True].index.values.tolist()

    assert len(labeled_idxs) == len(labels), 'Labeled num do not match'
    unlabeled_idxs = sorted(set(range(len(dataset.path_df))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (secondary_batch, primary_batch)
            in zip(grouper(secondary_iter, self.secondary_batch_size),
                   grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def get_labeled_mask(masks, label):
    masks_tmp = deepcopy(masks)
    masks[masks == label_to_num['femur_cartilage']] = 9
    # masks[masks == label_to_num['L_meniscus']] = 9
    # masks[masks == label_to_num['R_meniscus']] = 9
    # masks[masks == label_to_num['patella_cartilage']] = 9
    masks = np.where(masks < 8, 0, masks)
    masks[masks == 9] = 1

    # Dilation
    kernel_dila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    masks = cv2.dilate(masks, kernel_dila)

    # if label == 2:
    #     masks_femur = masks_tmp
    #     masks_femur[masks_femur != label_to_num['femur']] = 0
    #     nonzeros = np.nonzero(masks)
    #     try:
    #         x_min = min(nonzeros[1])
    #         x_max = max(nonzeros[1])
    #         y = min(np.concatenate((nonzeros[0][nonzeros[1] == x_min], nonzeros[0][nonzeros[1] == x_max])))
    #         masks_femur[:y, :] = 0
    #         masks[masks_femur == label_to_num['femur']] = 1
    #         masks_used = masks
    #     except ValueError:
    #         masks_used = masks

    # if label == 2:
    #     masks_femur = masks_tmp
    #     masks_femur[masks_femur != label_to_num['femur']] = 0
    #     masks_femur = np.array(masks_femur, np.uint8)
    #     ret, thr = cv2.threshold(masks_femur, 0, 255, 0)
    #     contours, hier = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     cv2.drawContours(masks_femur, contours, -1, (255, 0, 0), 3)
    #     nonzeros = np.nonzero(masks)
    #     try:
    #         x_min = min(nonzeros[1])
    #         x_max = max(nonzeros[1])
    #         y = max(np.concatenate((nonzeros[0][nonzeros[1] == x_min], nonzeros[0][nonzeros[1] == x_max])))
    #         masks_femur[:y, :] = 0
    #         masks_femur = cv2.dilate(masks_femur, kernel_dila)
    #         masks[masks_femur == 255] = 1
    #         masks_used = masks
    #     except:
    #         masks_used = masks
    # else:
    #     masks_used = masks

    return masks


class CustomDataset(data.Dataset):
    def __init__(self, name, data_path, fold_file, folds, transform=None):
        self.name = name
        self.transform = transform

        # Load DataFrame
        self.path_df = pd.read_csv(fold_file)
        self.path_df['ImagePath'] = data_path + 'ISIC2018_Task3_Training_Input/'+self.path_df['image']+'.jpg'
        self.path_df['MaskPath'] = data_path + 'ISIC2018_Task3_Seg/'+self.path_df['image']+'_segmentation.png'

        if folds:
            self.path_df = self.path_df[self.path_df.Fold.isin(folds)]
            self.path_df = self.path_df.reset_index(drop=True)

        pass

    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, index):
        row = self.path_df.iloc[index]
        image = cv2.imread(row['ImagePath'])[:, :, ::-1].astype("float32")
        masks = cv2.imread(row['MaskPath'])
        masks = (masks != 0).astype('int')[:, :, 0]
        label = row['label']

        flags = np.zeros(1, dtype=np.float32)

        if label != 0:
            flags[0] = 1

        # Normalization
        # min_value = np.percentile(image, 0.1)
        # max_value = np.percentile(image, 99.9)
        # image[image > max_value] = max_value
        # image[image < min_value] = min_value  # -outliers
        # norm_image = (image - min_value) / (max_value - min_value)
        # image = np.array([norm_image, norm_image, norm_image]).transpose((1, 2, 0))
        image = image/255.

        ######################################
        # plt.figure()
        # plt.imshow(image)
        # plt.figure()
        # plt.imshow(masks)
        # plt.show()
        ######################################

        if self.transform:
            augmented1 = self.transform(image=image, mask=masks)
            image1 = augmented1['image']
            masks1 = augmented1['mask']

            augmented2 = self.transform(image=image, mask=masks)
            image2 = augmented2['image']
            masks2 = augmented2['mask']

        #####################################
        # plt.figure()
        # plt.imshow(image1)
        # plt.figure()
        # plt.imshow(masks1*255)
        # plt.figure()
        # plt.imshow(image2)
        # plt.figure()
        # plt.imshow(masks2*255)
        # plt.show()
        #####################################

        # masks1 = get_labeled_mask(masks1, label)
        # masks2 = get_labeled_mask(masks2, label)

        #####################################
        # plt.figure()
        # plt.imshow(image1, cmap='gray')
        # plt.figure()
        # plt.imshow(masks1)
        # plt.figure()
        # plt.imshow(image2, cmap='gray')
        # plt.figure()
        # plt.imshow(masks2)
        # plt.show()
        #####################################

        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        label = np.array(label)

        image1 = torch.from_numpy(image1).float()
        masks1 = torch.from_numpy(masks1).float()
        image2 = torch.from_numpy(image2).float()
        masks2 = torch.from_numpy(masks2).float()
        label = torch.from_numpy(label).long()
        flags = torch.from_numpy(flags).float()

        return (image1, masks1), (image2, masks2), label, flags, row.image


class InferDataset(data.Dataset):
    def __init__(self, name, data_path, pickle_file, folds):
        self.name = name

        # Load DataFrame
        with open(pickle_file, 'rb') as f:
            self.df = pickle.load(f)
        self.df['Data_Path'] = data_path + self.df['ID'] + '.nii.gz'

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = sitk.ReadImage(row.Data_Path)
        image = sitk.GetArrayFromImage(img)

        # Normalization
        min_value = np.percentile(image, 0.1)
        max_value = np.percentile(image, 99.9)
        image[image > max_value] = max_value
        image[image < min_value] = min_value  # -outliers
        norm_image = (image - min_value) / (max_value - min_value)
        norm_image = np.transpose(norm_image, (1, 2, 0))

        image = torch.from_numpy(norm_image).float()

        return image, [row.ID, img.GetOrigin(), img.GetSpacing(), img.GetDirection()]


if __name__ == '__main__':
    dataset = CustomDataset('train',
                            '/home/yuanfang/Dataset/ISIC2018/',
                            '/home/yuanfang/DC-MT-SRC/configs/val_index.csv',
                            None,
                            albumentations.Compose([
                                albumentations.Resize(512, 512),
                                albumentations.OneOf([
                                    # albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
                                    albumentations.RandomBrightnessContrast(brightness_limit=0.05,
                                                                            contrast_limit=0.05, p=0.9),
                                ], p=0.5),
                                albumentations.OneOf([
                                    albumentations.Blur(
                                        blur_limit=4, p=1),
                                    # albumentations.MotionBlur(blur_limit=4, p=1),
                                    # albumentations.MedianBlur(
                                    #     blur_limit=3, p=1)
                                ], p=0.5),
                                albumentations.HorizontalFlip(p=0.5),
                                albumentations.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.01,
                                                                rotate_limit=3,
                                                                interpolation=cv2.INTER_LINEAR,
                                                                border_mode=cv2.BORDER_CONSTANT, p=0.5),
                            ]),)
    for i, (input, ema_input, label, flags, name) in enumerate(dataset):
        image1, masks1 = input
        image2, masks2 = ema_input
        print(image1.max(), masks1.max())
        break
