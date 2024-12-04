import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn 
from network.UNet_GAC import UNet_GAC
from trainer_GAC import trainer_GAC

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='LA_MRI', help='dataset_name')
parser.add_argument('--model', type=str,  default='UNet_GAC', help='model_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synthetic_low_contrast': {
            'root_path': 'data/Synthetic/train_low_contrast',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'Synthetic_SD0.1': {
            'root_path': 'data/Synthetic/train_noise/SD0.1',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'Synthetic_SD0.2': {
            'root_path': 'data/Synthetic/train_noise/SD0.2',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'Synthetic_SD0.3': {
            'root_path': 'data/Synthetic/train_noise/SD0.3',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'LA_MRI': {
            'root_path': 'data/LA/train_npz',
            'list_dir': 'lists/lists_LA',
            'num_classes': 2,
        },
        'Liver_CT': {
            'root_path': 'data/Liver/train_npz',
            'list_dir': 'lists/lists_Liver',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False
    args.exp = args.model +"_" + dataset_name + "_" + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, args.model)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    net = UNet_GAC(n_channels=1, n_classes=args.num_classes-1, bilinear=True).cuda()

    trainer_GAC(args, net, snapshot_path)

