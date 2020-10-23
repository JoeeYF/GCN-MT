import os
import shutil
import argparse
import yaml
import logging
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from glob import glob
from core.utils import net_utils_gcn as net_utils
from core.utils.logging import open_log
from core.utils.tools import load_checkpoint, increment_dir
from core.utils.visualizer import Visualizer
from core.models import AttentionNet
from core.models.GCNModel.GAT import MutilHeadGAT
from core.models.GCNModel.SimpleGCN import SimpleGCN


def arg_parse():
    parser = argparse.ArgumentParser(description='ClsNet')
    parser.add_argument('-cfg', '--config', default='configs/se50_semi.yaml', type=str, help='load the config file')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--use_html', default=True, type=bool, help='Use html')
    parser.add_argument('--stage', default='train', type=str, help='Which stage: train | valid | infer')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    config = yaml.load(open(args.config))
    base_dir = increment_dir(f"/mnt/data/yuanfang/DC-MT-SRC/outputs/{config['base_dir_prev']}_", config['description'])
    config['base_dir'] = base_dir
    shutil.copy(args.config, base_dir)
    os.system(f"cp -r core {config['base_dir']}")

    if args.cuda:
        gpus = ','.join([str(i) for i in config['GPUs']])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if config['deterministic']:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])

    # open log file
    open_log(args, config, 'train')
    logging.info(args)
    logging.info(config)
    visualizer = Visualizer('Mean-Teacher', config, args)

    logging.info(config['Data_CLASSES'])
    logging.info('Using the network: {}'.format(config['arch']))

    # Student net
    logging.info('Building Student Model')
    StudentModel = AttentionNet.build_model(config, ema=False)
    # Teacher net
    logging.info('Building Teacher Model')
    TeacherModel = AttentionNet.build_model(config, ema=True)
    # GCN net
    logging.info('Building GCN Model')
    # GCNModel = MutilHeadGAT(nfeat=config['gcn_feature_dim'], nhid=config['gcn_hidden_dim'], nclass=len(config['Data_CLASSES']), dropout=config['gcn_dropout'], alpha=0.2, nheads=config['gcn_hidden_num'])
    GCNModel = SimpleGCN(nfeat=config['gcn_feature_dim'], nhid=config['gcn_hidden_dim'], nclass=len(config['Data_CLASSES']), dropout=config['gcn_dropout'], alpha=0.2, nheads=config['gcn_hidden_num'])

    if config['Using_pretrained_weights']:
        StudentModel.load_pretrained_weights(load_fc=False)

    if config['Cls']['resume'] != None:
        load_checkpoint(StudentModel, config['Cls']['resume'])

    if args.cuda:
        StudentModel.cuda()
        TeacherModel.cuda()
        GCNModel.cuda()

    optimizer, gcn_optimizer, train_loader, valid_loader = net_utils.prepare_net(config, StudentModel, GCNModel)

    # apex speed up
    # ClsModel, optimizer = amp.initialize(ClsModel, optimizer, opt_level="O1")

    # StudentModel = torch.nn.DataParallel(StudentModel)
    # TeacherModel = torch.nn.DataParallel(TeacherModel)
    # GCNModel = torch.nn.DataParallel(GCNModel)
    load_checkpoint(StudentModel, '/mnt/data/yuanfang/DC-MT-SRC/outputs/DC_MT_000-DC_MT/checkpoint/S_fold[0]_se_resnext50_32x4d/S_fold[0]_se_resnext50_32x4d_best_acc.pth')
    load_checkpoint(TeacherModel, '/mnt/data/yuanfang/DC-MT-SRC/outputs/DC_MT_000-DC_MT/checkpoint/T_fold[0]_se_resnext50_32x4d/T_fold[0]_se_resnext50_32x4d_best_acc.pth')
    model = [StudentModel, TeacherModel, GCNModel]
    net_utils.train_net(visualizer, optimizer, gcn_optimizer, train_loader, valid_loader, model, config)

    # infer
    # logging.info('\n----------------------------INFER----------------------------')
    # logging.info(config['Data_CLASSES'])
    # logging.info('Using the network: {}'.format(config['arch']))
    # logging.info('Building ClsModel')

    # AttentionModel = AttentionNet.build_model(config, ema=False)
    # config['TestModel'] = glob(os.path.join(config['base_dir'], 'checkpoint', 'S*', '*_best_acc.pth'))
    # assert len(config['TestModel']) == 1
    # logging.info('Resuming network: {}'.format(config['TestModel'][0]))
    # load_checkpoint(AttentionModel, config['TestModel'][0])
    # if args.cuda:
    #     AttentionModel.cuda()
    #     cudnn.benchmark = True
    # infer_loader = net_utils.prepare_net(config, AttentionModel, None, 'infer')
    # AttentionModel = torch.nn.DataParallel(AttentionModel)
    # net_utils.infer(infer_loader, AttentionModel, config)

    # AttentionModel = AttentionNet.build_model(config, ema=False)
    # config['TestModel'] = glob(os.path.join(config['base_dir'], 'checkpoint', 'T*', '*_best_acc.pth'))
    # assert len(config['TestModel']) == 1
    # logging.info('Resuming network: {}'.format(config['TestModel'][0]))
    # load_checkpoint(AttentionModel, config['TestModel'][0])
    # if args.cuda:
    #     AttentionModel.cuda()
    #     cudnn.benchmark = True
    # infer_loader = net_utils.prepare_net(config, AttentionModel, None, 'infer')
    # AttentionModel = torch.nn.DataParallel(AttentionModel)
    # net_utils.infer(infer_loader, AttentionModel, config)


if __name__ == '__main__':
    main()
