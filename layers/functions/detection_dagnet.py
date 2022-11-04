import torch
from torch.autograd import Function
from ..box_utils import decode, nms, center_size
from data import voc_dagnet as cfg


class Detect_DAGNet(Function):

    def __init__(self,
                 num_classes,
                 bkg_label,
                 top_k,
                 conf_thresh,
                 nms_thresh,
                 objectness_thre,
                 keep_top_k):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = cfg['512']['variance']

    def forward(self,
                mbd1_loc_data,
                mbd1_conf_data,
                mbd2_loc_data,
                mbd2_conf_data,
                prior_data):
        loc_data = mbd2_loc_data
        conf_data = mbd2_conf_data

        mbd1_object_conf = mbd1_conf_data.data[:, :, 1:]
        no_object_index = mbd1_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        for i in range(num):
            default = decode(mbd1_loc_data[i],
                             prior_data,
                             self.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i],
                                   default,
                                   self.variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):

                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes.detach(),
                                 scores.detach(),
                                 self.nms_thresh, self.top_k)

                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        return output
