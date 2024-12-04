import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from utils_GAC import ICSTM



def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def trainer_GAC(args, model, snapshot_path):
    from datasets.dataset import dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = dataset(dataset_name=args.dataset,base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(num_classes-1)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, SDF_batch, OSDF_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['SDF'], sampled_batch['OSDF']
            image_batch, label_batch, SDF_batch, OSDF_batch = image_batch.cuda(), label_batch.cuda(), SDF_batch.cuda(), OSDF_batch.cuda()
            out_dis, boundary = model(image_batch)
            with torch.no_grad():
                boundary_gt_batch = 1 * (torch.abs(OSDF_batch)<=1.5)
            if epoch_num<=int(max_epoch/2):
                outputs = torch.sigmoid(-1500*out_dis)
            else:
                outputs = ICSTM(out_dis,boundary,10)

            loss_dice = dice_loss(outputs[:,0,:,:], label_batch==1)
            loss_dis = torch.norm(out_dis[:,0,:,:] - SDF_batch, 1)/ torch.numel(out_dis[:,0,:,:])
            loss_boundary = dice_loss(boundary[:, 0, :, :], boundary_gt_batch == 1)
            # loss_boundary = torch.norm(boundary[:, 0, :, :] - boundary_gt_batch[:, 0, :, :], 1) / torch.numel(boundary[:, 0, :, :])
            loss = loss_dice + loss_dis + loss_boundary
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice,iter_num)
            writer.add_scalar('info/loss_dis', loss_dis, iter_num)
            writer.add_scalar('info/loss_boundary', loss_boundary, iter_num)
            logging.info('iteration %d : loss : %f, loss_dice: %f, loss_dis: %f, loss_boundary: %f'  % (iter_num, loss.item(), loss_dice.item(), loss_dis.item(), loss_boundary.item()))
            # logging.info('iteration %d : loss : %f, loss_dice: %f, loss_dis: %f' % (
            # iter_num, loss.item(), loss_dice.item(), loss_dis.item()))
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 5  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"