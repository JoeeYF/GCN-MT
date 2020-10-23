import os
import time
import glob
import cv2
import csv
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations
import pandas as pd
from .tools import *
from .metric_utils import *
from .config import *
from .ramps import *
from core.modules.losses.mse_loss import cls_mse_loss, att_mse_loss, relation_mse_loss
from core.modules.feature_queue import FeatureQueue
from core.data.custom_dataset import *

mask_mse_loss_func = att_mse_loss
consistency_criterion_cls = cls_mse_loss
consistency_criterion_att = att_mse_loss
# sigma_loss_func = nn.SmoothL1Loss()

label_list = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
label_to_num = {name: index for index, name in enumerate(label_list)}
num_to_label = {v: k for k, v in label_to_num.items()}
global_step = 0


def get_current_consistency_cls_weight(epoch, config):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return config['consistency_cls'] * sigmoid_rampup(epoch, config['consistency_rampup'], type='cls')


def get_current_consistency_att_weight(epoch, config):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if epoch < 20:
        return 0.0
    else:
        return config['consistency_att'] * sigmoid_rampup(epoch, config['consistency_rampup'], type='att')


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def prepare_net(config, model, GCNModel, _use='train'):
    # img_size = (config['img_size'], config['img_size'])
    def worker_init_fn(worker_id):
        random.seed(config['seed']+worker_id)
    if _use == 'train':
        if config['optim'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=config['weight_decay'])
            gcn_optimizer = torch.optim.Adam(GCNModel.parameters(), config['lr'], weight_decay=config['weight_decay'])

        if config['optim'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), config['lr'], weight_decay=config['weight_decay'])
            gcn_optimizer = torch.optim.RMSprop(GCNModel.parameters(), config['lr'], weight_decay=config['weight_decay'])

        elif config['optim'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), config['lr'], momentum=config['momentum'],
                                        weight_decay=config['weight_decay'], nesterov=config['nesterov'])
            gcn_optimizer = torch.optim.SGD(GCNModel.parameters(), config['lr'],  momentum=config['momentum'],
                                            weight_decay=config['weight_decay'], nesterov=config['nesterov'])

        folds = [fold for fold in range(config['n_fold'])]
        train_dataset = CustomDataset('train', config['DataRoot'], config['TrainFold'], None,
                                      transform=albumentations.Compose([
                                          albumentations.Resize(
                                              config['img_size'], config['img_size']),
                                          albumentations.OneOf([
                                              # albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
                                              albumentations.RandomBrightnessContrast(brightness_limit=0.05,
                                                                                      contrast_limit=0.05, p=0.9),
                                          ], p=0.5),
                                          albumentations.OneOf([
                                              albumentations.Blur(
                                                  blur_limit=4, p=1),
                                              # albumentations.MotionBlur(blur_limit=4, p=1),
                                              #   albumentations.MedianBlur(
                                              #       blur_limit=4, p=1)
                                          ], p=0.5),
                                          albumentations.HorizontalFlip(p=0.5),
                                          albumentations.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.01,
                                                                          rotate_limit=3,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          border_mode=cv2.BORDER_CONSTANT, p=0.5),
                                          #   albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                                      ]),
                                      )

        labeled_df = pd.read_csv(config['TrainFold'])
        labeled_fold = [i for i in config['label_fold']]
        labeled_df = labeled_df[labeled_df.fold_label.isin(labeled_fold)]
        labeled_fold_name = labeled_df.image.tolist()
        labeled_idxs, unlabeled_idxs = relabel_dataset(
            train_dataset, labeled_fold_name)
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, config['batchsize'], config['label_bs'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=config['num_workers'], batch_sampler=batch_sampler, pin_memory=True, worker_init_fn=worker_init_fn)

        # Count different classes num in train dataset
        # all_label = np.array([label for _, _, label, _, _ in train_dataset])
        # class_sample_count = np.array([len(np.where(all_label == t)[0]) for t in np.unique(all_label)])
        # for index in range(len(config['Data_CLASSES'])):
        #     print("Train class {}: Num {}".format(index, class_sample_count[index]))

        valid_dataset = CustomDataset('valid', config['DataRoot'], config['ValidFold'], None,
                                      transform=albumentations.Compose([
                                          albumentations.Resize(
                                              config['img_size'], config['img_size']),
                                      ])
                                      )
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10, shuffle=False,
                                                   num_workers=config['num_workers'], drop_last=False,
                                                   pin_memory=True, worker_init_fn=worker_init_fn)

        # Count different classes num in valid dataset
        # all_label = np.array([label for _, _, label, _, _ in valid_dataset])
        # class_sample_count = np.array([len(np.where(all_label == t)[0]) for t in np.unique(all_label)])
        # for index in range(len(config['Data_CLASSES'])):
        #     print("Valid class {}: Num {}".format(index, class_sample_count[index]))

        return optimizer, gcn_optimizer, train_loader, valid_loader

    elif _use == 'infer':
        infer_dataset = CustomDataset('infer', config['DataRoot'], config['TestFold'], None,
                                      transform=albumentations.Compose([
                                          albumentations.Resize(
                                              config['img_size'], config['img_size']),
                                      ])
                                      )
        infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=10, shuffle=False,
                                                   num_workers=config['num_workers'], drop_last=False,
                                                   pin_memory=True, worker_init_fn=worker_init_fn)

        return infer_loader


def train_net(visualizer, optimizer, gcn_optimizer, train_loader, val_loader, model, config):

    best_metric_dict = {i: 0 for i in ['acc', 'bac', 'auc', 'f1', 'recall', 'tiou', 'tior', 'acc_epoch']}

    cls_criterion = nn.NLLLoss()

    if config['lr_decay'] == None:
        lr_decay = 0.1
    else:
        lr_decay = config['lr_decay']

    for epoch in range(1, config['num_epoch']+1):
        adjust_learning_rate(optimizer, epoch - 1, config['num_epoch'], config['lr'], config['lr_decay_freq'], lr_decay)

        train(visualizer, train_loader, model, optimizer, gcn_optimizer, epoch, config, cls_criterion)

        if epoch % config['valid_freq'] == 0:
            best_metric_dict = valid_net(val_loader, model, config, best_metric_dict, epoch)
            logging.info('Valid-Cls: Best ACC   update to: {:.4f}, from Epoch {}'.format(best_metric_dict['acc'], best_metric_dict['acc_epoch']))
            logging.info('Valid-Cls: Best BAC   update to: {:.4f}'.format(best_metric_dict['bac']))
            logging.info('Valid-Cls: Best AUC   update to: {:.4f}'.format(best_metric_dict['auc']))
            logging.info('Valid-Cls: Best F1    update to: {:.4f}'.format(best_metric_dict['f1']))
            logging.info('Valid-Cls: Best recal update to: {:.4f}'.format(best_metric_dict['recall']))
            logging.info('Valid-Cls: Best TIOU  update to: {:.4f}'.format(best_metric_dict['tiou']))
            logging.info('Valid-Cls: Best TIOR  update to: {:.4f}'.format(best_metric_dict['tior']))


def valid_net(val_loader, model, config, best_metric_dict, epoch):
    result_s, result_t, result_gcn, TIOU, TIOR = valid(val_loader, model, config)
    StudentModel, TeacherModel, GCNModel = model
    m_acc_s, all_acc_s, m_auc_s, all_auc_s, m_recall_s, all_recall_s, m_f1_s, all_f1_s, m_bac_s, all_bac_s = result_s
    m_acc_t, all_acc_t, m_auc_t, all_auc_t, m_recall_t, all_recall_t, m_f1_t, all_f1_t, m_bac_t, all_bac_t = result_t
    m_acc_gcn, all_acc_gcn, m_auc_gcn, all_auc_gcn, m_recall_gcn, all_recall_gcn, m_f1_gcn, all_f1_gcn, m_bac_gcn, all_bac_gcn = result_gcn
    TIOU_s, TIOU_t = TIOU
    TIOR_s, TIOR_t = TIOR

    mTIOU_s = 0.
    mTIOU_t = 0.
    assert TIOU_s.shape[1] == TIOU_t.shape[1], "TIOU dimension error"
    len_TIOU = TIOU_s.shape[1]
    for idx in range(len(config['Data_CLASSES'])):
        mTIOU_s += TIOU_s[idx].sum() / float(len_TIOU)
        mTIOU_t += TIOU_t[idx].sum() / float(len_TIOU)
    mTIOU_s /= float(len(config['Data_CLASSES']))
    mTIOU_t /= float(len(config['Data_CLASSES']))

    mTIOR_s = 0.
    mTIOR_t = 0.
    assert TIOR_s.shape[1] == TIOR_t.shape[1], "TIOR dimension error"
    len_TIOR = TIOR_s.shape[1]
    for idx in range(len(config['Data_CLASSES'])):
        mTIOR_s += TIOR_s[idx].sum() / float(len_TIOR)
        mTIOR_t += TIOR_t[idx].sum() / float(len_TIOR)
    mTIOR_s /= float(len(config['Data_CLASSES']))
    mTIOR_t /= float(len(config['Data_CLASSES']))

    logging.info('[Student Model]')
    logging.info('Valid-Cls: Mean ACC: {:.4f}, Mean BAC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}, Mean recall: {:.4f}, Mean TIoU: {:.4f}, Mean TIoR: {:.4f}'.format(m_acc_s,
                                                                                                                                                                      m_bac_s, m_auc_s, m_f1_s, m_recall_s, mTIOU_s, mTIOR_s))
    print_result('Valid-Cls: ACC for All Classes: ', all_acc_s, config['Data_CLASSES'])
    print_result('Valid-Cls: BAC for All Classes: ', all_bac_s, config['Data_CLASSES'])
    print_result('Valid-Cls: AUC for All Classes: ', all_auc_s, config['Data_CLASSES'])
    print_result('Valid-Cls: F1  for All Classes: ', all_f1_s,  config['Data_CLASSES'])
    print_result('Valid-Cls: recall for All Classes: ', all_recall_s, config['Data_CLASSES'])
    print_thresh_result('Valid-TIoU: ', TIOU_s, thresh_TIOU, config['Data_CLASSES'])
    print_thresh_result('Valid-TIoR: ', TIOR_s, thresh_TIOR, config['Data_CLASSES'])

    logging.info('[Teacher Model]')
    logging.info('Valid-Cls: Mean ACC: {:.4f}, Mean BAC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}, Mean recall: {:.4f}, Mean TIoU: {:.4f}, Mean TIoR: {:.4f}'.format(m_acc_t,
                                                                                                                                                                      m_bac_t, m_auc_t, m_f1_t, m_recall_t, mTIOU_t, mTIOR_t))
    print_result('Valid-Cls: ACC for All Classes: ', all_acc_t, config['Data_CLASSES'])
    print_result('Valid-Cls: BAC for All Classes: ', all_bac_t, config['Data_CLASSES'])
    print_result('Valid-Cls: AUC for All Classes: ', all_auc_t, config['Data_CLASSES'])
    print_result('Valid-Cls: F1  for All Classes: ', all_f1_t, config['Data_CLASSES'])
    print_result('Valid-Cls: recall for All Classes: ', all_recall_t, config['Data_CLASSES'])
    print_thresh_result('Valid-TIoU: ', TIOU_t, thresh_TIOU, config['Data_CLASSES'])
    print_thresh_result('Valid-TIoR: ', TIOR_t, thresh_TIOR, config['Data_CLASSES'])

    logging.info('[GCN Model]')
    logging.info('Valid-Cls: Mean ACC: {:.4f}, Mean BAC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}, Mean recall: {:.4f}'.format(m_acc_gcn, m_bac_gcn, m_auc_gcn, m_f1_gcn, m_recall_gcn))
    print_result('Valid-Cls: ACC for All Classes: ', all_acc_gcn, config['Data_CLASSES'])
    print_result('Valid-Cls: BAC for All Classes: ', all_bac_gcn, config['Data_CLASSES'])
    print_result('Valid-Cls: AUC for All Classes: ', all_auc_gcn, config['Data_CLASSES'])
    print_result('Valid-Cls: F1  for All Classes: ', all_f1_gcn, config['Data_CLASSES'])
    print_result('Valid-Cls: recall for All Classes: ', all_recall_gcn, config['Data_CLASSES'])

    # m_acc = max(m_acc_s, m_acc_t, m_acc_gcn)
    # m_recall = max(m_recall_s, m_recall_t, m_recall_gcn)
    # m_bac = max(m_bac_s, m_bac_t, m_bac_gcn)
    # m_auc = max(m_auc_s, m_auc_t, m_auc_gcn)
    # m_f1 = max(m_f1_s, m_f1_t, m_f1_gcn)
    # m_tiou = max(mTIOU_s, mTIOU_t)
    # m_tior = max(mTIOR_s, mTIOR_t)

    m_acc =  m_acc_gcn
    m_recall =  m_recall_gcn
    m_bac =  m_bac_gcn
    m_auc =  m_auc_gcn
    m_f1 = m_f1_gcn
    m_tiou = max(mTIOU_s, mTIOU_t)
    m_tior = max(mTIOR_s, mTIOR_t)

    if m_acc > best_metric_dict['acc']:
        save_checkpoint(StudentModel, 'S_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='acc', best=m_acc_s)
        save_checkpoint(TeacherModel, 'T_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='acc', best=m_acc_t)
        save_checkpoint(GCNModel, 'G_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='acc', best=m_acc_gcn)
        best_metric_dict['acc'] = m_acc
        best_metric_dict['acc_epoch'] = epoch
    if m_recall >= best_metric_dict['recall']:
        save_checkpoint(StudentModel, 'S_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='recall', best=m_recall_s)
        save_checkpoint(TeacherModel, 'T_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='recall', best=m_recall_t)
        save_checkpoint(GCNModel, 'G_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='recall', best=m_recall_gcn)
        best_metric_dict['recall'] = m_recall
    if m_bac >= best_metric_dict['bac']:
        save_checkpoint(StudentModel, 'S_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='bac', best=m_recall_s)
        save_checkpoint(TeacherModel, 'T_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='bac', best=m_recall_t)
        save_checkpoint(GCNModel, 'G_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='bac', best=m_recall_gcn)
        best_metric_dict['bac'] = m_bac
    if m_auc >= best_metric_dict['auc']:
        save_checkpoint(StudentModel, 'S_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='auc', best=m_auc_s)
        save_checkpoint(TeacherModel, 'T_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='auc', best=m_auc_t)
        save_checkpoint(GCNModel, 'G_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='auc', best=m_auc_gcn)
        best_metric_dict['auc'] = m_auc
    if m_f1 >= best_metric_dict['f1']:
        save_checkpoint(StudentModel, 'S_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='f1', best=m_f1_s)
        save_checkpoint(TeacherModel, 'T_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='f1', best=m_f1_t)
        save_checkpoint(GCNModel, 'G_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='f1', best=m_f1_gcn)
        best_metric_dict['f1'] = m_f1

    if m_tiou >= best_metric_dict['tiou']:
        save_checkpoint(StudentModel, 'S_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='tiou', best=mTIOU_s)
        save_checkpoint(TeacherModel, 'T_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='tiou', best=mTIOU_t)
        best_metric_dict['tiou'] = m_tiou
    if m_tior >= best_metric_dict['tior']:
        save_checkpoint(StudentModel, 'S_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='tior', best=mTIOR_s)
        save_checkpoint(TeacherModel, 'T_fold' + str(config['label_fold']) + '_' + config['arch'], epoch, config['base_dir'], _best='tior', best=mTIOR_t)
        best_metric_dict['tior'] = m_tior

    return best_metric_dict


def train(visualizer, train_loader, model, optimizer, gcn_optimizer, epoch, config, cls_criterion):
    global global_step
    StudentModel, TeacherModel, GCNModel = model
    losses = AverageMeter()
    cls_losses = AverageMeter()
    attmse_losses = AverageMeter()
    attbound_losses = AverageMeter()
    src_losses = AverageMeter()
    consiscls_losses = AverageMeter()
    consisatt_losses = AverageMeter()
    batch_time = AverageMeter()
    cls_accs = AverageMeter()
    cls_accs_gcn = AverageMeter()
    cls_AUCs = AverageMeter()
    cls_F1s = AverageMeter()
    gcn_cls_losses = AverageMeter()

    num_classes = len(config['Data_CLASSES'])

    StudentModel.eval()
    TeacherModel.eval()
    GCNModel.train()
    end = time.time()

    StudentFeatureQueue = FeatureQueue(config, 2)
    TeacherFeatureQueue = FeatureQueue(config, 2)

    for i, (input, ema_input, label, flags, name) in enumerate(train_loader):

        with torch.autograd.set_detect_anomaly(True):
            image1, masks1 = input
            image2, masks2 = ema_input

            im_h = image1.size(2)
            im_w = image1.size(3)
            bs = image1.size(0)
            label_bs = config['label_bs']

            visualizer.reset()
            visual_ret = OrderedDict()
            errors_ret = OrderedDict()

            image1 = image1.cuda()
            masks1 = masks1.cuda()
            image2 = image2.cuda()
            # masks2 = masks2.cuda()
            masks1 = masks1.unsqueeze(1)
            # masks2 = masks2.unsqueeze(1)
            label = label.cuda()
            # flags = flags.cuda()

            visual_ret['input'] = image1
            masks_vis = visual_masks(masks1, im_h, im_w)
            visual_ret['mask'] = masks_vis
            with torch.no_grad():
                output_s, cam_refined_s, feature_s = StudentModel(image1)
                output_t, cam_refined_t, feature_t = TeacherModel(image2)

            # StudentFeatureQueue.enqueue(feature_s, label)
            # TeacherFeatureQueue.enqueue(feature_s, label)

            output_gcn = GCNModel(feature_s, feature_t)

            class_idx = label.cpu().long().numpy()
            for index, idx in enumerate(class_idx):
                tmp1 = cam_refined_s[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                tmp2 = cam_refined_t[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                if index == 0:
                    cam_refined_class_s = tmp1
                    cam_refined_class_t = tmp2
                else:
                    cam_refined_class_s = torch.cat((cam_refined_class_s, tmp1), dim=0)
                    cam_refined_class_t = torch.cat((cam_refined_class_t, tmp2), dim=0)
            cam_refined_s = cam_refined_class_s
            cam_refined_t = cam_refined_class_t

            # Classification
            probe = torch.softmax(output_s, dim=1)
            cls_loss = cls_criterion(torch.log(probe[:label_bs]), label[:label_bs])

            # Attention
            # MSE loss
            mask_loss = mask_mse_loss_func(masks1[:label_bs], cam_refined_s[:label_bs])

            # Bound loss
            # bound_loss = torch.exp(torch.tensor(1) - torch.min(masks1[:label_bs], cam_refined_s[:label_bs]).sum((2, 3)) / torch.clamp(cam_refined_s[:label_bs].sum((2, 3)), min=1e-5))
            bound_loss = torch.tensor(1) - torch.min(masks1[:label_bs], cam_refined_s[:label_bs]).sum((2, 3)) / torch.clamp(cam_refined_s[:label_bs].sum((2, 3)), min=1e-5)
            bound_loss = bound_loss.sum() / bs

            gcams_vis = visual_masks(cam_refined_s.float(), im_h, im_w)
            visual_ret['attention'] = gcams_vis

            # Attention Consistency
            consistency_weight_att = get_current_consistency_att_weight(epoch, config)
            consistency_loss_att = consistency_weight_att * consistency_criterion_att(cam_refined_s[label_bs:], cam_refined_t[label_bs:])

            # Classification Consistency
            consistency_weight_cls = get_current_consistency_cls_weight(epoch, config)
            consistency_loss_cls = consistency_weight_att * consistency_criterion_cls(output_s, output_t)

            # SRC Loss
            consistency_relation_dist = torch.sum(relation_mse_loss(feature_s, feature_t)) / bs
            src_loss = consistency_weight_att * consistency_relation_dist*config['src_weight']

            # GCN Classification
            gcn_probe = torch.softmax(output_gcn, dim=1)
            gcn_cls_loss = cls_criterion(torch.log(gcn_probe[:label_bs]), label[:label_bs])*config['gcn_weight']

            total_loss = gcn_cls_loss

            errors_ret['ClsLoss'] = float(cls_loss)
            errors_ret['AttMseLoss'] = float(mask_loss)
            errors_ret['AttBoundLoss'] = float(bound_loss)
            errors_ret['ConsisClsLoss'] = float(consistency_loss_cls)
            errors_ret['ConsisAttLoss'] = float(consistency_loss_att)
            errors_ret['SRCLoss'] = float(src_loss)
            errors_ret['GCNClsLoss'] = float(gcn_cls_loss)
            errors_ret['Loss'] = float(total_loss)

            losses.update(total_loss.item(), bs)
            cls_losses.update(cls_loss.item(), bs)
            attmse_losses.update(mask_loss.item(), bs)
            attbound_losses.update(bound_loss.item(), bs)
            consiscls_losses.update(consistency_loss_cls.item(), bs)
            consisatt_losses.update(consistency_loss_att.item(), bs)
            src_losses.update(src_loss.item(), bs)
            gcn_cls_losses.update(gcn_cls_loss.item(), bs)

            gcn_optimizer.zero_grad()
            total_loss.backward()
            gcn_optimizer.step()

            global_step += 1
            # update_ema_variables(StudentModel, TeacherModel, config['ema_decay'], global_step)
            # update_ema_variables(GCNStudentModel, GCNTeacherModel, config['ema_decay'], global_step)

            m_acc, _ = calculate_acc(probe.cpu().detach().numpy(), label.cpu().detach().numpy(), config)
            cls_accs.update(m_acc, bs)
            m_acc_gcn, _ = calculate_acc(gcn_probe.cpu().detach().numpy(), label.cpu().detach().numpy(), config)
            cls_accs_gcn.update(m_acc_gcn, bs)
            m_auc, _ = calculate_auc(probe.cpu().detach().numpy(), label.cpu().detach().numpy(), config)
            cls_AUCs.update(m_auc, bs)
            m_f1, _ = calculate_f1(probe.cpu().detach().numpy(), label.cpu().detach().numpy(), config)
            cls_F1s.update(m_f1, bs)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config['print_freq'] == 0:
                logging.info('Epoch: [{}][{}/{}]\t'
                             'Lr: {:.5f} '
                             'ConsistencyWeightAtt: {:.4f} '
                             'ClsAcc: {cls_acc.val:.4f} ({cls_acc.avg:.4f}) '
                             'ClsAccg: {cls_accg.val:.4f} ({cls_accg.avg:.4f}) '
                             'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                             'ClsLoss: {cls_loss.val:.4f} ({cls_loss.avg:.4f}) '
                             'AttMseloss: {attmse_loss.val:.4f} ({attmse_loss.avg:.4f}) '
                             'AttBndLoss: {attbnd_loss.val:.4f} ({attbnd_loss.avg:.4f}) '
                             'ConsisClsLoss: {concls_loss.val:.4f} ({concls_loss.avg:.4f}) '
                             'ConsisAttLoss: {conatt_loss.val:.4f} ({conatt_loss.avg:.4f}) '
                             'SRCLoss: {src_loss.val:.4f} ({src_loss.avg:.4f}) '
                             'GCNClsLoss: {gcn_cls_loss.val:.4f} ({gcn_cls_loss.avg:.4f}) '.format(
                                 epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], consistency_weight_att, cls_acc=cls_accs, cls_accg=cls_accs_gcn, loss=losses, cls_loss=cls_losses, attmse_loss=attmse_losses,
                                 attbnd_loss=attbound_losses, concls_loss=consiscls_losses, conatt_loss=consisatt_losses, src_loss=src_losses, gcn_cls_loss=gcn_cls_losses))

                if config['display_id'] > 0:
                    visualizer.plot_current_losses(epoch, float(
                        i) / float(len(train_loader)), errors_ret)
            if i % config['display_freq'] == 0:
                visualizer.display_current_results(
                    visual_ret, class_idx[0], epoch, save_result=False)


def valid(valid_loader, model, config):
    StudentModel, TeacherModel, GCNModel = model
    batch_time = AverageMeter()
    StudentModel.eval()
    TeacherModel.eval()
    GCNModel.eval()

    num_classes = len(config['Data_CLASSES'])
    counts = np.zeros(num_classes)
    TIOU_s = np.zeros((num_classes, len(thresh_TIOU)))
    TIOR_s = np.zeros((num_classes, len(thresh_TIOR)))
    TIOU_t = np.zeros((num_classes, len(thresh_TIOU)))
    TIOR_t = np.zeros((num_classes, len(thresh_TIOR)))

    with torch.no_grad():
        end = time.time()
        for i, (input, ema_input, label, flags, name) in enumerate(valid_loader):
            image, masks = input

            im_h = image.size(2)
            im_w = image.size(3)
            bs = image.size(0)

            image = image.cuda()
            masks = masks.cuda()
            label = label.cuda()
            masks = masks.unsqueeze(1)

            output_s, cam_refined_s, feature_s = StudentModel(image)
            output_t, cam_refined_t, feature_t = TeacherModel(image)

            # StudentFeatureQueue.enqueue(feature_s, label)
            # TeacherFeatureQueue.enqueue(feature_s, label)

            output_gcn = GCNModel(feature_s, feature_t)

            class_idx = label.cpu().long().numpy()
            for index, idx in enumerate(class_idx):
                tmp_s = cam_refined_s[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                tmp_t = cam_refined_t[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                if index == 0:
                    cam_refined_class_s = tmp_s
                    cam_refined_class_t = tmp_t
                else:
                    cam_refined_class_s = torch.cat((cam_refined_class_s, tmp_s), dim=0)
                    cam_refined_class_t = torch.cat((cam_refined_class_t, tmp_t), dim=0)

            cam_refined_s = cam_refined_class_s
            cam_refined_t = cam_refined_class_t

            probe_s = torch.softmax(output_s, dim=1)
            probe_t = torch.softmax(output_t, dim=1)
            probe_gcn = torch.softmax(output_gcn, dim=1)

            cam_refined_s = cam_refined_s >= cam_thresh
            cam_refined_t = cam_refined_t >= cam_thresh

            batch_iou_s = single_IOU(cam_refined_s[:, 0, :, :], masks[:, 0, :, :])
            batch_ior_s = single_IOR(cam_refined_s[:, 0, :, :], masks[:, 0, :, :])
            batch_iou_t = single_IOU(cam_refined_t[:, 0, :, :], masks[:, 0, :, :])
            batch_ior_t = single_IOR(cam_refined_t[:, 0, :, :], masks[:, 0, :, :])
            # print(TIOU.shape)
            # print(TIOR.shape)

            for j in range(len(thresh_TIOU)):
                if batch_iou_s >= thresh_TIOU[j]:
                    TIOU_s[class_idx, j] += 1
                if batch_iou_t >= thresh_TIOU[j]:
                    TIOU_t[class_idx, j] += 1
            for j in range(len(thresh_TIOR)):
                if batch_ior_s >= thresh_TIOR[j]:
                    TIOR_s[class_idx, j] += 1
                if batch_ior_t >= thresh_TIOR[j]:
                    TIOR_t[class_idx, j] += 1
            counts[class_idx] += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if i % (config['print_freq'] * config['batchsize']) == 0:
                logging.info('Valid: [{}/{}]\t''Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(i, len(valid_loader), batch_time=batch_time))

            if i == 0:
                y_true = label.cpu().detach().numpy()
                y_pred_s = probe_s.cpu().detach().numpy()
                y_pred_t = probe_t.cpu().detach().numpy()
                y_pred_gcn = probe_gcn.cpu().detach().numpy()
            else:
                y_true = np.concatenate((y_true, label.cpu().detach().numpy()), axis=0)
                y_pred_s = np.concatenate((y_pred_s, probe_s.cpu().detach().numpy()), axis=0)
                y_pred_t = np.concatenate((y_pred_t, probe_t.cpu().detach().numpy()), axis=0)
                y_pred_gcn = np.concatenate((y_pred_gcn, probe_gcn.cpu().detach().numpy()), axis=0)

        m_acc_s, all_acc_s = calculate_acc(y_pred_s, y_true, config)
        m_auc_s, all_auc_s = calculate_auc(y_pred_s, y_true, config)
        m_recall_s, all_recall_s = recall(y_pred_s, y_true, config, show_confusion_matrix=True)
        m_f1_s, all_f1_s = calculate_f1(y_pred_s, y_true, config)
        m_bac_s, all_bac_s = calculate_bac(y_pred_s, y_true, config)

        m_acc_t, all_acc_t = calculate_acc(y_pred_t, y_true, config)
        m_auc_t, all_auc_t = calculate_auc(y_pred_t, y_true, config)
        m_recall_t, all_recall_t = recall(y_pred_t, y_true, config)
        m_f1_t, all_f1_t = calculate_f1(y_pred_t, y_true, config)
        m_bac_t, all_bac_t = calculate_bac(y_pred_t, y_true, config)

        m_acc_gcn, all_acc_gcn = calculate_acc(y_pred_gcn, y_true, config)
        m_auc_gcn, all_auc_gcn = calculate_auc(y_pred_gcn, y_true, config)
        m_recall_gcn, all_recall_gcn = recall(y_pred_gcn, y_true, config)
        m_f1_gcn, all_f1_gcn = calculate_f1(y_pred_gcn, y_true, config)
        m_bac_gcn, all_bac_gcn = calculate_bac(y_pred_gcn, y_true, config)

        for idx in range(num_classes):
            for j in range(len(thresh_TIOU)):
                if counts[idx] == 0:
                    TIOU_s[idx, j] = 0.
                    TIOU_t[idx, j] = 0.
                else:
                    TIOU_s[idx, j] = float(TIOU_s[idx, j]) / float(counts[idx])
                    TIOU_t[idx, j] = float(TIOU_t[idx, j]) / float(counts[idx])

        for idx in range(num_classes):
            for j in range(len(thresh_TIOR)):
                if counts[idx] == 0:
                    TIOR_s[idx, j] = 0.
                    TIOR_t[idx, j] = 0.
                else:
                    TIOR_s[idx, j] = float(TIOR_s[idx, j]) / float(counts[idx])
                    TIOR_t[idx, j] = float(TIOR_t[idx, j]) / float(counts[idx])

        return [m_acc_s, all_acc_s, m_auc_s, all_auc_s, m_recall_s, all_recall_s, m_f1_s, all_f1_s, m_bac_s, all_bac_s], \
               [m_acc_t, all_acc_t, m_auc_t, all_auc_t, m_recall_t, all_recall_t, m_f1_t, all_f1_t, m_bac_t, all_bac_t], \
               [m_acc_gcn, all_acc_gcn, m_auc_gcn, all_auc_gcn, m_recall_gcn, all_recall_gcn, m_f1_gcn, all_f1_gcn, m_bac_gcn, all_bac_gcn], \
               [TIOU_s, TIOU_t], \
               [TIOR_s, TIOR_t]


def infer(infer_loader, model, config):
    batch_time = AverageMeter()
    model.eval()

    num_classes = len(config['Data_CLASSES'])
    counts = np.zeros(num_classes)
    TIOU = np.zeros((num_classes, len(thresh_TIOU)))
    TIOR = np.zeros((num_classes, len(thresh_TIOR)))
    all_name = []

    with torch.no_grad():
        end = time.time()
        for i, (input, ema_input, label, flags, name) in enumerate(infer_loader):
            all_name = all_name + list(name)
            image, masks = input

            im_h = image.size(2)
            im_w = image.size(3)
            bs = image.size(0)

            image = image.cuda()
            masks = masks.cuda()
            label = label.cuda()
            masks = masks.unsqueeze(1)

            output, cam_refined, cam, = model(image)
            class_idx = label.cpu().long().numpy()
            for index, idx in enumerate(class_idx):
                tmp = cam_refined[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                if index == 0:
                    cam_refined_class = tmp
                else:
                    cam_refined_class = torch.cat((cam_refined_class, tmp), dim=0)
            cam_refined = cam_refined_class
            probe = torch.softmax(output, dim=1)

            cam_refined = cam_refined >= cam_thresh

            batch_iou = single_IOU(cam_refined[:, 0, :, :], masks[:, 0, :, :])
            batch_ior = single_IOR(cam_refined[:, 0, :, :], masks[:, 0, :, :])
            # print(TIOU.shape)
            # print(TIOR.shape)
            for j in range(len(thresh_TIOU)):
                if batch_iou >= thresh_TIOU[j]:
                    TIOU[class_idx, j] += 1
            for j in range(len(thresh_TIOR)):
                if batch_ior >= thresh_TIOR[j]:
                    TIOR[class_idx, j] += 1
            counts[class_idx] += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if i % (config['print_freq'] * config['batchsize']) == 0:
                logging.info('Infer-Cls: [{}/{}]\t'
                             'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
                                 i, len(infer_loader), batch_time=batch_time))

            if i == 0:
                y_gt = label.cpu().detach().numpy()
                y_pred = probe.cpu().detach().numpy()
            else:
                y_gt = np.concatenate((y_gt, label.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, probe.cpu().detach().numpy()), axis=0)

        m_acc, all_acc = calculate_acc(y_pred, y_gt, config)
        m_auc, all_auc = calculate_auc(y_pred, y_gt, config)
        m_recall, all_recall = recall(y_pred, y_gt, config, show_confusion_matrix=True)
        m_f1, all_f1 = calculate_f1(y_pred, y_gt, config)
        m_bac, all_bac = calculate_bac(y_pred, y_gt, config)

        for idx in range(num_classes):
            for j in range(len(thresh_TIOU)):
                if counts[idx] == 0:
                    TIOU[idx, j] = 0.
                else:
                    TIOU[idx, j] = float(TIOU[idx, j]) / float(counts[idx])

        for idx in range(num_classes):
            for j in range(len(thresh_TIOR)):
                if counts[idx] == 0:
                    TIOR[idx, j] = 0.
                else:
                    TIOR[idx, j] = float(TIOR[idx, j]) / float(counts[idx])

        mTIOU = 0.
        len_TIOU = TIOU.shape[1]
        for idx in range(len(config['Data_CLASSES'])):
            mTIOU += TIOU[idx].sum() / float(len_TIOU)
        mTIOU /= float(len(config['Data_CLASSES']))

        mTIOR = 0.
        len_TIOR = TIOR.shape[1]
        for idx in range(len(config['Data_CLASSES'])):
            mTIOR += TIOR[idx].sum() / float(len_TIOR)
        mTIOR /= float(len(config['Data_CLASSES']))

        # result_dict = {'Path': all_name, 'G0_Pred': y_pred[:, 0], 'G1_Pred': y_pred[:, 1], 'G2_Pred': y_pred[:, 2], 'Label': y_gt}
        # result_df = pd.DataFrame(result_dict)
        # result_df.to_csv('./outputs/results/{}.csv'.format(config['model_name']), index=False)
        logging.info('Infer-Cls: Mean ACC: {:.4f}, Mean BAC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}, Mean recall: {:.4f}, Mean TIoU: {:.4f}, Mean TIoR: {:.4f}'.format(
            m_acc, m_bac, m_auc, m_f1, m_recall, mTIOU, mTIOR))
        print_result('Infer-Cls: ACC for All Classes: ', all_acc, config['Data_CLASSES'])
        print_result('Valid-Cls: BAC for All Classes: ', all_bac, config['Data_CLASSES'])
        print_result('Infer-Cls: AUC for All Classes: ', all_auc, config['Data_CLASSES'])
        print_result('Infer-Cls: F1  for All Classes: ', all_f1, config['Data_CLASSES'])
        print_result('Infer-Cls: recall for All Classes: ', all_recall, config['Data_CLASSES'])
        print_thresh_result('Infer-TIoU: ', TIOU, thresh_TIOU, config['Data_CLASSES'])
        print_thresh_result('Infer-TIoR: ', TIOR, thresh_TIOR, config['Data_CLASSES'])


def single_IOU(pred, target):
    pred_class = pred.data.cpu().contiguous().view(-1)
    target_class = target.data.cpu().contiguous().view(-1)
    pred_inds = pred_class == 1
    target_inds = target_class == 1
    # Cast to long to prevent overflows
    intersection = (pred_inds[target_inds]).long().sum().item()
    union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    iou = float(intersection) / float(max(union, 1))
    return iou


def single_IOR(pred, target):
    pred_class = pred.data.cpu().contiguous().view(-1)
    target_class = target.data.cpu().contiguous().view(-1)
    pred_inds = pred_class == 1
    target_inds = target_class == 1
    # Cast to long to prevent overflows
    intersection = (pred_inds[target_inds]).long().sum().item()
    union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    iou = float(intersection) / float(max(pred_inds.long().sum().item(), 1))
    return iou


def visual_masks(masks, im_h, im_w):
    mask_vis = masks[0, :, :, :].unsqueeze(0).clone()
    mask_one = torch.zeros((1, im_h, im_w)).cuda()
    mask_one = mask_one + mask_vis[:, 0, :, :]
    mask_one[mask_one >= 1] = 1
    vis_mask1 = mask_one.clone()
    vis_mask2 = mask_one.clone()
    vis_mask3 = mask_one.clone()
    vis_mask1[vis_mask1 == 1] = palette[1][0]
    vis_mask2[vis_mask2 == 1] = palette[1][1]
    vis_mask3[vis_mask3 == 1] = palette[1][2]
    vis_mask1 = vis_mask1.unsqueeze(1)
    vis_mask2 = vis_mask2.unsqueeze(1)
    vis_mask3 = vis_mask3.unsqueeze(1)
    vis_mask = torch.cat((vis_mask1, vis_mask2, vis_mask3), 1)
    return vis_mask
