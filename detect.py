from __future__ import print_function

import os
import sys
from pathlib import Path
from glob import glob

import cv2
import argparse
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from os import path
from data import BaseTransform, VOC_CLASSES as labelmap
from torch.autograd import Variable
from models.dagnet import build_dagnet
from models.backbone import backbone_convert, whole_model_convert

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
warnings.filterwarnings('ignore')


class HiddenOutput():
    """This function omits excessive output from the programme."""
    def __enter__(self):
        self.original_stdout = sys.__stdout__
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout


parser = argparse.ArgumentParser(description='DAGNet')
parser.add_argument('--image_dir',
                    default='demo/*.bmp',
                    type=str,
                    help='Image path. Later will be fed to a glob function.')
parser.add_argument('--output',
                    default='./detection/',
                    type=str,
                    help='Output directory')
parser.add_argument('--weights',
                    default='weights/test.pth',
                    type=str,
                    help='Trained state_dict file path')
parser.add_argument('--input_size',
                    default='512',
                    help='Image input size')
parser.add_argument('--conf_thr',
                    default=0.8,
                    type=float,
                    help='Confidence score')
args = parser.parse_args()

bbox_color = {
    'uav': (0, 0, 192),
}


def _rectangle(image, lt, rb, color, thickness):
    assert thickness > 0, 'Wrong line thickness.'

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
    if torch.cuda.is_available():
        x = x.cuda()
    x = Variable(x.unsqueeze(0))

    y = net(x)

    detections = y.data
    scale = torch.Tensor([1, width, height, width, height])

    for i in range(1, detections.size(1)):

        dets = detections[0, i, :]
        mask = dets[:, 0].gt(args.conf_thr).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)

        scores, boxes = dets[:, 0].cpu().numpy(), dets[:, 1:] * scale[1:]
        cnt = 0
        for score, box in zip(scores, boxes):
            cnt += 1
            bbox = tuple([round(float(box[0])), round(float(box[1])),
                          round(float(box[2])), round(float(box[3]))])

            line_thickness = 2
            frame = _rectangle(frame, bbox[:2], bbox[2:],
                               bbox_color[labelmap[i - 1]], thickness=line_thickness)

        print(f"Detect {cnt} {labelmap[i-1]}{'s' if cnt > 1 else ''}", end='\n')

    return frame


def cv2_demo():
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    else:
        print('No valid output directory is specified. Please check.')
        exit(-1)

    images = glob(args.image_dir, recursive=False)
    for i, imgpath in enumerate(images):
        print(f'[{i+1}/{len(images)}] Image name: {Path(imgpath).name} - ', end='')
        img = cv2.imread(imgpath)
        det_img = predict(img)
        if args.output:
            cv2.imwrite(os.path.join(args.output, Path(imgpath).name), det_img)


if __name__ == '__main__':

    with HiddenOutput():
        num_classes = len(labelmap) + 1
        net_train = build_dagnet('train',
                                 size=int(args.input_size),
                                 num_classes=num_classes)
        net_train.load_state_dict(torch.load(args.weights))
        net_train.eval()

        net_test = build_dagnet('test',
                                size=int(args.input_size),
                                num_classes=num_classes)
        net_test.backbone = backbone_convert(net_train.backbone,
                                             save_path=None)
        net = whole_model_convert(net_train, net_test)
        net.eval()
        del net_train, net_test

    if torch.cuda.is_available():
        net = net.cuda()
    # print(net)
    # exit()
    transform = BaseTransform(net.size, (104, 117, 123))

    cv2_demo()
