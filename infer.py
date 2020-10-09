import os
import argparse
import yaml
import logging
import torch
import torch.backends.cudnn as cudnn
from core.utils.logging import open_log
from core.utils.tools import *
from core.models import AttentionNet


def arg_parse():
    parser = argparse.ArgumentParser(
        description='ClsNet')
    parser.add_argument('-cfg', '--config', default='./configs/se50_test_t.yaml',
                        type=str, help='load the config file')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    parser.add_argument('--use_html', default=True,
                        type=bool, help='Use html')
    parser.add_argument('--stage', default='infer',
                        type=str, help='Which stage: train | valid | infer')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    config = yaml.load(open(args.config))
    config['base_dir'] = '/mnt/data/yuanfang/DC-MT-SRC/outputs/'+config['TestModel'].split('/')[6]

    if args.cuda:
        gpus = ','.join([str(i) for i in config['GPUs']])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # open log file
    open_log(args, config, 'infer')
    logging.info(args)
    logging.info(config)
    config['model_name'] = args.config.split('/')[-1].split('.')[0]

    logging.info(config['Data_CLASSES'])
    logging.info('Using the network: {}'.format(config['arch']))

    # set net
    logging.info('Building ClsModel')
    AttentionModel = AttentionNet.build_model(config, ema=False)

    assert config['TestModel'] != ''  # must load a trained model
    logging.info('Resuming network: {}'.format(config['TestModel']))
    load_checkpoint(AttentionModel, config['TestModel'])

    if args.cuda:
        AttentionModel.cuda()
        cudnn.benchmark = True

    from core.utils import net_utils
    infer_loader = net_utils.prepare_net(config, AttentionModel, None,'infer')

    AttentionModel = torch.nn.DataParallel(AttentionModel)

    net_utils.infer(infer_loader, AttentionModel, config)


if __name__ == '__main__':
    main()
