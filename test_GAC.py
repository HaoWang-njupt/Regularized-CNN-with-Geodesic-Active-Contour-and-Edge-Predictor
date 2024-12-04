import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import dataset

from network.UNet_GAC import UNet_GAC
from utils_GAC import test_single_image
from collections import OrderedDict
import pandas as pd
 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='LA_MRI', help='dataset_name')
parser.add_argument('--model', type=str,  default='UNet_GAC', help='model_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='lists/lists_Synthetic', help='list dir')

parser.add_argument('--max_epochs', type=int, default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_save', default=True, help='whether to save results during inference')



parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()



def inference(args, model, test_save_path=None):
    db_test = dataset(dataset_name=args.dataset,base_dir=args.root_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    metric_dict = OrderedDict()
    metric_dict['name'] = list()
    metric_dict['Dice'] = list()
    metric_dict['Jaccard'] = list()
    metric_dict['HD95'] = list()
    metric_dict['ASD'] = list()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[1:3]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_image(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name)
        metric_dict['name'].append(case_name)
        metric_dict['Dice'].append(metric_i[0][0])
        metric_dict['Jaccard'].append(metric_i[0][1])
        metric_dict['HD95'].append(metric_i[0][2])
        metric_dict['ASD'].append(metric_i[0][3])
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_jaccard %f mean_hd95 %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],np.mean(metric_i, axis=0)[2],np.mean(metric_i, axis=0)[3]))
    metric_list = metric_list / len(db_test)
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_jaccard = np.mean(metric_list, axis=0)[1]
    mean_hd95 = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    logging.info('Testing performance in best val model: mean_dice : %f mean_jaccard : %f mean_hd95 : %f mean_asd : %f' % (mean_dice, mean_jaccard, mean_hd95, mean_asd))
    metric_csv = pd.DataFrame(metric_dict)
    metric_csv.to_csv(test_save_path + '/metric_'+str(args.max_epochs)+'.csv', index=False)
    return "Testing Finished!"


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


    dataset_config = {
        'Synthetic_low_contrast': {
            'root_path': 'data/Synthetic/test_low_contrast',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'Synthetic_SD0.1': {
            'root_path': 'data/Synthetic/test_noise/SD0.1',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'Synthetic_SD0.2': {
            'root_path': 'data/Synthetic/test_noise/SD0.2',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'Synthetic_SD0.3': {
            'root_path': 'data/Synthetic/test_noise/SD0.3',
            'list_dir': 'lists/lists_Synthetic',
            'num_classes': 2,
        },
        'LA_MRI': {
            'root_path': 'data/LA/test_vol_h5',
            'list_dir': 'lists/lists_LA',
            'num_classes': 2,
        },
        'Liver_CT': {
            'root_path': 'data/Liver/test_vol_h5',
            'list_dir': 'lists/lists_Liver',
            'num_classes': 2,
        },
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False
    # name the same snapshot defined in train script!
    args.exp = args.model +"_" + dataset_name + "_" + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, args.model)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    # net = VNetMultiTask(n_channels=1, n_classes=args.num_classes - 1, normalization='batchnorm',
    #                     has_dropout=True).cuda()
    net = UNet_GAC(n_channels=1, n_classes=args.num_classes - 1, bilinear=True).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_save:
        args.test_save_dir = 'predictions/'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


