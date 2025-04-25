# – coding: utf-8 –
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

from glob import glob

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import DAGNet_MultiBox_Loss
from models.dagnet import build_dagnet
from utils.logging import Logger


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='DAGNet training')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--input_size', 
                    default='512',
                    type=str, 
                    help='The input of the network is fixed to 512.')
parser.add_argument('--dataset_root', 
                    default='./data/VOC2007/',
                    type=str,
                    help='Dataset root directory path')
parser.add_argument('--basenet',
                    default= './weights/basenet.pth',
                    type=str,
                    help='Pretrained base model')
parser.add_argument('--batch_size', 
                    default=2, 
                    type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    type=bool,
                    default=False,
                    help='whether to resume training')

parser.add_argument('--num_workers', 
                    default=4, 
                    type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', 
                    default=True, 
                    type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--lr', 
                    '--learning-rate',
                    default=1e-3,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', 
                    default=0.9, 
                    type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', 
                    default=5e-4, 
                    type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', 
                    default=0.1, 
                    type=float,
                    help='Gamma update for SGD')

parser.add_argument('--save_folder', 
                    default='./weights/',
                    type=str,
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

localtime = time.strftime('%Y%m%d-%H%M%S', time.localtime())
sys.stdout = Logger(os.path.join(args.save_folder, f'log_{localtime}.txt'))


def train():
    cfg = voc_dagnet[args.input_size]
    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    dagnet = build_dagnet('train',
                          device,
                          cfg['num_classes'])

    init_iter = 0
    if args.resume:
        files = [file for file in os.listdir(args.save_folder)
                 if file.startswith('last') and file.endswith('pth')]
        last_corruption = False

        if len(files) != 0:
            print(f'Resuming training, loading from last.pth')
            try:
                init_iter = int(files[0].split('.')[0].split('_')[-1])
                dagnet.load_weights(f'{args.save_folder}/{files[0]}')
            except:
                last_corruption = True

        if len(files) == 0 or last_corruption:
            print('Last weight file corrputed, loading from the nearest file...')
            files = [file for file in os.listdir(args.save_folder)
                     if file.endswith('pth') and file.startswith('RefineDet')]
            files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
            files = [files[-1]]
            init_iter = int(files[0].split('_')[-1].split('.')[0])
            dagnet.load_weights(f'{args.save_folder}/{files[0]}')
    else:
        backbone_weights = torch.load(args.basenet)
        print('Loading base network...')

        '''Modify weight'''
        print('Finding lost keys...')
        lost_keys = [x[0] for x in list(dagnet.named_parameters()) + list(dagnet.named_buffers())
                     if x[0] not in list(backbone_weights.keys())]

        print('Finish finding keys, generating lost weight dictionary...')
        lost_weight_dict = {x[0].replace('module.', '').replace('backbone.', ''): x[1]
                            for x in list(dagnet.named_parameters()) + list(dagnet.named_buffers())
                            if x[0] in lost_keys}  # and ('attention' in x[0] or 'Norm' in x[0])}

        print(f'Complete generation, now insert new weights into backbone weight dict...')
        for k, v in lost_weight_dict.items():
            backbone_weights[k] = v
        '''-------------'''

        dagnet.backbone.load_state_dict(backbone_weights, strict=False)

    if not args.resume:
        print('Initializing weights...')
        dagnet.mbd1_loc.apply(weights_init)
        dagnet.mbd1_conf.apply(weights_init)
        dagnet.mbd2_loc.apply(weights_init)
        dagnet.mbd2_conf.apply(weights_init)
        for fa in dagnet.fas:
            fa.apply(weights_init)

    if args.cuda:
        dagnet = torch.nn.DataParallel(dagnet).to(device)
        cudnn.benchmark = True

    optimizer = optim.SGD(dagnet.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    mbd1_criterion = DAGNet_MultiBox_Loss(num_classes=2,
                                          overlap_thresh=0.5,
                                          prior_for_matching=True,
                                          bkg_label=0,
                                          neg_mining=True,
                                          neg_pos=3,
                                          neg_overlap=0.5,
                                          encode_target=False,
                                          use_gpu=args.cuda)

    mbd2_criterion = DAGNet_MultiBox_Loss(num_classes=cfg['num_classes'],
                                          overlap_thresh=0.5,
                                          prior_for_matching=True,
                                          bkg_label=0,
                                          neg_mining=True,
                                          neg_pos=3,
                                          neg_overlap=0.5,
                                          encode_target=False,
                                          use_gpu=args.cuda,
                                          use_mbd1=True)

    dagnet.train()
    mbd1_loc_loss = 0
    mbd1_conf_loss = 0
    mbd2_loc_loss = 0
    mbd2_conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    if init_iter < cfg['lr_steps'][0]:
        pass
    else:
        while init_iter < cfg['lr_steps'][step_index]:
            step_index += 1

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(init_iter, cfg['max_iter']):

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets, img_ids = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets, img_ids = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = images
            targets = [ann for ann in targets]
        
        # forward
        t0 = time.time()
        out = dagnet(images)
        
        # backprop
        optimizer.zero_grad()
        mbd1_loss_l, mbd1_loss_c = mbd1_criterion(out, targets)
        mbd2_loss_l, mbd2_loss_c = mbd2_criterion(out, targets)
        
        mbd1_loss = mbd1_loss_l + mbd1_loss_c
        mbd2_loss = mbd2_loss_l + mbd2_loss_c
        loss = mbd1_loss + mbd2_loss
        loss.backward()

        # torch.nn.utils.clip_grad_norm(dagnet.parameters(), 3)
        optimizer.step()
        t1 = time.time()
        mbd1_loc_loss += mbd1_loss_l.item()
        mbd1_conf_loss += mbd1_loss_c.item()
        mbd2_loc_loss += mbd2_loss_l.item()
        mbd2_conf_loss += mbd2_loss_c.item()

        if iteration % 10 == 0:
            print(
                'iter ' + repr(iteration) + '\tmbd1_L Loss: %.4f\tmbd1_C Loss: %.4f\tmbd2_L Loss: %.4f\tmbd2_C Loss: '
                                            '%.4f\ttotal Loss: %.4f\t' \
                % (mbd1_loss_l.item(), mbd1_loss_c.item(), mbd2_loss_l.item(), mbd2_loss_c.item(), loss.item()),
                end=' ')
            print('timer: %.4f sec.' % (t1 - t0))

        if iteration != 0 and iteration % 2500 == 0:
            print('Saving state, iter:', iteration)
            torch.save(dagnet.module.state_dict(), args.save_folder
                       + '/RefineDet{}_{}_{}.pth'.format(args.input_size, args.dataset,
                                                         repr(iteration)))

            weights = sorted(glob(os.path.join(args.save_folder, 'RefineDet512_VOC_*.pth')))
            if len(weights) > 10:
                print('Removing one weight file.')
                os.remove(random.choice(weights[:5]))

        if iteration % 100 == 0:
            files = [file for file in os.listdir(args.save_folder)
                     if file.startswith('last') and file.endswith('pth')]
            if len(files) == 1:
                os.remove(f'{args.save_folder}/{files[0]}')
            torch.save(dagnet.module.state_dict(),
                       f'{args.save_folder}/last_{repr(iteration)}.pth')

        # If read last weights file, the following error shows up:
        # RuntimeError: [enforce fail at inline_container.cc:222] . file not found:XXXXX
        # That means the last iter's weights file save failed (File corrupted)

    torch.save(dagnet.module.state_dict(), args.save_folder
               + '/RefineDet{}_{}_final.pth'.format(args.input_size, args.dataset))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def get_dict_info(weight: dict, start: int, end: int) -> dict:
    new_dict = {}

    for i, (key, value) in enumerate(weight.items()):
        if i < start:
            continue
        if i == end:
            break
        if start != 0:
            new_dict[key.split('.')[1] + '.' + key.split('.')[-1]] = value
        else:
            new_dict[key] = value

    return new_dict


if __name__ == '__main__':
    train()
