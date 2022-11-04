from __future__ import print_function

import datetime
import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path as path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import time
import cv2
import argparse
import warnings
warnings.filterwarnings('ignore')

from torch.autograd import Variable
from pathlib import Path

from data import BaseTransform, VOC_CLASSES as labelmap
from models.dagnet import build_dagnet
from models.backbone import backbone_model_convert, whole_model_convert

import torch


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class HiddenOutput:
    def __enter__(self):
        self.original_stdout = sys.__stdout__
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout


parser = argparse.ArgumentParser(description='DAGNet')
parser.add_argument('--cuda',
                    default=True,
                    type=str2bool,
                    help='Whether to use CUDA device.')
parser.add_argument('--image_dir',
                    default='./test_imgs/',
                    type=str,
                    help='Image directory')
parser.add_argument('--txt',
                    type=str,
                    default=None,
                    help='Txt file')
parser.add_argument('--output',
                    default=f'./detection/{datetime.datetime.now().strftime("%Y%m%d")}',
                    type=str,
                    help='Output directory')
parser.add_argument('--weights',
                    default='./weights/test.pth',
                    type=str,
                    help='Trained state_dict file path')
parser.add_argument('--conf_thr',
                    type=float,
                    default=0.5,
                    help='Use cuda in live demo')

args = parser.parse_args()

bbox_color = {
    'uav': (0, 0, 192),
    'GT': (0, 192, 0)}
font_type = cv2.FONT_HERSHEY_SIMPLEX
exts = ['bmp', 'jpg', 'png']


def cv2_demo(net, transform):

    def imgs_dataset(test_txt_path: str = None) -> list:
        assert os.path.exists(test_txt_path), 'Path of txt does not exist.'
        test_ids0 = open(test_txt_path, 'r').readlines()
        test_ids = [id.split('/')[-1].strip() + '.bmp' for id in test_ids0]
        test_ids.sort()
        return test_ids

    def _rectangle(image, lt, rb, color, thickness):
        assert thickness > 0
        bbox = tuple(map(int, (lt[0], lt[1], rb[0], rb[1])))
        for i in range(thickness):
            lt, rb = (bbox[0] - i, bbox[1] - i), \
                     (bbox[2] + i, bbox[3] + i)
            cv2.rectangle(image, lt, rb,
                          color, 1)
        return image

    def predict(frame):
        height, width = frame.shape[:2]

        preframe = transform(frame)[0][:, :, ::-1]
        x = torch.from_numpy(preframe.copy()).permute(2, 0, 1)
        if args.cuda:
            x = x.cuda()
        x = Variable(x.unsqueeze(0))
        y = net(x)

        detections = y.data
        scale = torch.Tensor([1, width, height, width, height])

        for i in range(1, detections.size(1)):

            label = labelmap[i - 1]

            dets = detections[0, i, :]
            mask = dets[:, 0].gt(args.conf_thr).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)

            scores, boxes = dets[:, 0].cpu().numpy(), dets[:, 1:] * scale[1:]
            for score, box in zip(scores, boxes):
                bbox = tuple([round(float(box[0])), round(float(box[1])),
                              round(float(box[2])), round(float(box[3]))])

                line_thickness = 2
                frame = _rectangle(frame,
                                   [bbox[0] + 1, bbox[1] + 1],
                                   [bbox[2], bbox[3] + 1],
                                   bbox_color[label], thickness=line_thickness)

        return frame

    if args.output:
        if os.path.isdir(args.output):
            import shutil
            shutil.rmtree(args.output)
        os.makedirs(args.output)
    else:
        print('Output directory not given.')
        exit()

    if args.txt is None:
        frames = [img for img in os.listdir(args.image_dir) if img.split('.')[-1] in exts]
        frames.sort()
    else:
        frames = imgs_dataset(args.txt)

    for i, frame in enumerate(frames):

        img = cv2.imdecode(np.fromfile(os.path.join(args.image_dir, frame), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        pred_img = predict(img)

        print(f'[{i + 1}/{len(frames)}] img_id: {frame}')
        if args.output:
            cv2.imencode('.png', pred_img)[1].tofile(os.path.join(args.output, f"{Path(frame).stem}.png"))

    print('All images are outputted to the output directory.')


if __name__ == '__main__':

    with HiddenOutput():
        num_classes = len(labelmap) + 1
        device = torch.device('cuda:0' if args.cuda else 'cpu')
        net_train = build_dagnet('train',
                                 device,
                                 num_classes=num_classes)
        net_train.load_state_dict(torch.load(args.weights))
        net_train.eval()

        net_test = build_dagnet('test',
                                device,
                                num_classes=num_classes)
        net_test.backbone = backbone_model_convert(net_train.backbone,
                                                   num_classes=num_classes)
        net = whole_model_convert(net_train, net_test)
        net.eval()
        del net_train, net_test

        if args.cuda:
            net = net.cuda()

    transform = BaseTransform(512, (104, 117, 123))
    cv2_demo(net, transform)
