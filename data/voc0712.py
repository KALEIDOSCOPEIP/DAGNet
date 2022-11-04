"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = ('uav', )

VOC_ROOT = HOME


class VOCAnnotationTransform(object):

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        assert self.class_to_ind is not None
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]

        return res


class VOCDetection(data.Dataset):

    def __init__(self, root,
                 image_sets=(('2007', 'trainval'), ),
                 transform=None,
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.bmp')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, img_id = self.pull_item(index)

        return im, gt, img_id

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        img = cv2.imdecode(np.fromfile(self._imgpath % img_id, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        target = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            if target.shape[0] == 0:
                img, _, _ = self.transform(img, None, None)
                target = np.empty((0, 5), dtype=np.float32)
            else:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            img = img[:, :, (2, 1, 0)]

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, img_id

    def pull_image(self, index):
        img_id = self.ids[index]
        img = cv2.imdecode(np.fromfile(self._imgpath % img_id, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        return img

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
