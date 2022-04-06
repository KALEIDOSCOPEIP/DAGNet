import torch
from torch.autograd import Function
from ..box_utils import decode, nms, center_size
from data.config import voc_dagnet as cfg


class Detect_DAGNet(Function):
    """At test time, Detect is the final layer of DAGNet. Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh,
                 objectness_thre, keep_top_k):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = cfg[str(size)]['variance']

    def forward(self, cr1_loc_data, cr1_conf_data, cr2_loc_data, cr2_conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data = cr2_loc_data
        conf_data = cr2_conf_data

        arm_object_conf = cr1_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        for i in range(num):
            default = decode(cr1_loc_data[i].cuda() if torch.cuda.is_available() else cr1_loc_data[i],
                             prior_data.cuda() if torch.cuda.is_available() else prior_data,
                             self.variance)
            default = center_size(default)

            decoded_boxes = decode(loc_data[i].cuda() if torch.cuda.is_available() else loc_data[i],
                                   default.cuda() if torch.cuda.is_available() else default,
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
