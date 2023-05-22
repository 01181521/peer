import argparse
import logging
import os

import torch

import configs
from scripts import train_hashing
from tools import *

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s: %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S')

torch.backends.cudnn.benchmark = True
configs.default_workers = os.cpu_count()

parser = argparse.ArgumentParser(description='OrthoHash')
parser.add_argument('--nbit', default=64, type=int, help='number of bits')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--epochs', default=150, type=int, help='training epochs')
parser.add_argument('--ds', default='cifar10_en', choices=['cifar10_en', 'cifar100', 'imagenet100', 'nuswide', 'coco','corel','corel_en'],
                    help='dataset')
parser.add_argument('--arch', default='alexnet', choices=['alexnet'], help='backbone name')

# loss related
parser.add_argument('--scale', default=8, type=float, help='scale for cossim')
parser.add_argument('--margin', default=0.3, type=float, help='ortho margin ')
parser.add_argument('--margin-type', default='cos', choices=['cos', 'arc'], help='margin type')
parser.add_argument('--ce', default=1.0, type=float, help='classification scale')
parser.add_argument('--quan', default=0.0, type=float, help='quantization loss scale')
parser.add_argument('--quan-type', default='cs', choices=['cs', 'l1', 'l2'], help='quantization types')
parser.add_argument('--multiclass-loss', default='label_smoothing',
                    choices=['bce', 'imbalance', 'label_smoothing'], help='multiclass loss types')

# codebook generation
parser.add_argument('--codebook-method', default='B', choices=['N', 'B', 'O'], help='N = sign of gaussian; '
                                                                                    'B = bernoulli; '
                                                                                    'O = optimize')

parser.add_argument('--seed', default=59, help='seed number; default: random')

parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()

config = {
    
    "device": torch.device("cuda:0"),
    "bit_list": [32],
    "n_class" : 100,
    "topK":-1,
    "save_path":'/hdd/sxz/1result/vigh/cifar10_en/',
###########################################################
    'arch': args.arch,
    'arch_kwargs': {
        'nbit': args.nbit,
        'nclass': 0,  # will be updated below
        'pretrained': True,
        'freeze_weight': False,
    },
    'batch_size': args.bs,
    'dataset': args.ds,
    'multiclass': args.ds == 'nuswide',
    'dataset_kwargs': {
        'resize': 256 if args.ds in ['nuswide'] else 224,
        'crop': 224,
        'norm': 2,
        'evaluation_protocol': 1,  # only affect cifar10
        'reset': False,
        'separate_multiclass': False,
    },
    'optim': 'adam',
    'optim_kwargs': {
        'lr': args.lr,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'nesterov': False,
        'betas': (0.9, 0.999)
    },
    'epochs': args.epochs,
    'scheduler': 'step',
    'scheduler_kwargs': {
        'step_size': int(args.epochs * 0.8),
        'gamma': 0.1,
        'milestones': '0.5,0.75'
    },
    'save_interval': 0,
    'eval_interval': 5,
    'tag': 'orthohash',
    'seed': args.seed,

    'codebook_generation': args.codebook_method,

    # loss_param
    'ce': args.ce,
    's': args.scale,
    'm': args.margin,
    'm_type': args.margin_type,
    'quan': args.quan,
    'quan_type': args.quan_type,
    'multiclass_loss': args.multiclass_loss,
    'device': args.device
}

config['arch_kwargs']['nclass'] = configs.nclass(config)
config['R'] = configs.R(config)

logdir = (f'logs/{config["arch"]}{config["arch_kwargs"]["nbit"]}_'
          f'{config["dataset"]}_{config["dataset_kwargs"]["evaluation_protocol"]}_'
          f'{config["epochs"]}_'
          f'{config["optim_kwargs"]["lr"]}_'
          f'{config["optim"]}_'
          f'{config["ce"]}')

if config['tag'] != '':
    logdir += f'/{config["tag"]}_{config["seed"]}_'
else:
    logdir += f'/{config["seed"]}_'

# make sure no overwrite problem
count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'

while os.path.isdir(logdir):
    count += 1
    logdir = orig_logdir + f'{count:03d}'

config['logdir'] = logdir

count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'

config = config_dataset(config)
train_hashing.main(config)
