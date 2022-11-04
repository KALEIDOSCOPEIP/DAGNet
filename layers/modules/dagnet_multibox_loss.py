# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import coco as cfg
from ..box_utils import log_sum_exp, refine_match


class DAGNet_MultiBox_Loss(nn.Module):

    def __init__(self,
                 num_classes,
                 overlap_thresh,
                 prior_for_matching,
                 bkg_label,
                 neg_mining,
                 neg_pos,
                 neg_overlap,
                 encode_target,
                 use_gpu=True,
                 theta=0.01,
                 use_mbd1=False):
        super(DAGNet_MultiBox_Loss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.theta = theta
        self.use_mbd1 = use_mbd1

    def forward(self, predictions, targets):
        mbd1_loc_data, mbd1_conf_data, mbd2_loc_data, mbd2_conf_data, priors = predictions
        if self.use_mbd1:
            loc_data, conf_data = mbd2_loc_data, mbd2_conf_data
        else:
            loc_data, conf_data = mbd1_loc_data, mbd1_conf_data
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if num_classes == 2:
                labels = labels >= 0
            defaults = priors.data
            if self.use_mbd1:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                             loc_t, conf_t, idx, self.use_mbd1, num_classes,
                             mbd1_loc_data[idx].data)
            else:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                             loc_t, conf_t, idx, self.use_mbd1, num_classes)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        loc_t.requires_grad = False
        conf_t.requires_grad = False

        if self.use_mbd1:
            P = F.softmax(mbd1_conf_data, 2)
            mbd1_conf_tmp = P[:, :, 1]
            object_score_index = mbd1_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        batch_conf = conf_data.view(-1, self.num_classes)
        a = conf_t.reshape(-1, 1)

        b = batch_conf.gather(1, a)
        c = log_sum_exp(batch_conf)
        loss_c = c - b

        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
