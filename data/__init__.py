from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT

from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    t, i, s = [], [], []
    for sample in batch:
        i.append(sample[0])
        t.append(torch.FloatTensor(sample[1]))
        s.append(sample[2])
    return torch.stack(i, 0), t, s


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x = x - mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
