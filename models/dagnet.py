from models.backbone import *
from layers import *
from layers.functions.detection_dagnet import *
from data.config import voc_dagnet
import os


class DAGNet(nn.Module):

    def __init__(self,
                 phase,
                 size,
                 base,
                 cr1,
                 cr2,
                 f,
                 num_classes,
                 **kwargs):
        super(DAGNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # index = 1  # Default choosing VOC dataset
        self.cfg = voc_dagnet

        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        self.backbone = base

        self.cr1_loc = nn.ModuleList(cr1[0])
        self.cr1_conf = nn.ModuleList(cr1[1])
        self.cr2_loc = nn.ModuleList(cr2[0])
        self.cr2_conf = nn.ModuleList(cr2[1])
        self.f0 = nn.ModuleList(f[0])
        self.f1 = nn.ModuleList(f[1])
        self.f2 = nn.ModuleList(f[2])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_DAGNet(num_classes=num_classes,
                                        size=self.size,
                                        bkg_label=0,
                                        top_k=1000,
                                        conf_thresh=0.6,
                                        nms_thresh=0.1,
                                        objectness_thre=0.1,
                                        keep_top_k=500)

    def forward(self, x):

        f_source = list()
        cr1_loc = list()
        cr1_conf = list()
        cr2_loc = list()
        cr2_conf = list()

        _, sources = self.backbone(x)

        for (x, l, c) in zip(sources, self.cr1_loc, self.cr1_conf):
            cr1_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            cr1_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        cr1_loc = torch.cat([o.view(o.size(0), -1) for o in cr1_loc], 1)
        cr1_conf = torch.cat([o.view(o.size(0), -1) for o in cr1_conf], 1)

        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(4):
                s = self.f0[(3 - k) * 4 + i](s)
            if k != 0:
                u = p
                u = self.f1[3 - k](u)
                s += u
            for i in range(4):
                s = self.f2[(3 - k) * 4 + i](s)
            p = s
            f_source.append(s)
        f_source.reverse()

        for (x, l, c) in zip(f_source, self.cr2_loc, self.cr2_conf):
            cr2_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            cr2_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        cr2_loc = torch.cat([o.view(o.size(0), -1) for o in cr2_loc], 1)
        cr2_conf = torch.cat([o.view(o.size(0), -1) for o in cr2_conf], 1)

        if self.phase == "test":
            output = self.detect.forward(
                cr1_loc.view(cr1_loc.size(0), -1, 4),
                self.softmax(cr1_conf.view(cr1_conf.size(0), -1,
                                           2)),
                cr2_loc.view(cr2_loc.size(0), -1, 4),
                self.softmax(cr2_conf.view(cr2_conf.size(0), -1,
                                           self.num_classes)),
                self.priors.type(type(x.data))
            )
        else:
            output = (
                cr1_loc.view(cr1_loc.size(0), -1, 4),
                cr1_conf.view(cr1_conf.size(0), -1, 2),
                cr2_loc.view(cr2_loc.size(0), -1, 4),
                cr2_conf.view(cr2_conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
        else:
            print('Only .pth files supported.')

    def align_conv(self):
        pass


def cr1_multibox(backbone, cfg):
    arm_loc_layers = []
    arm_conf_layers = []
    backbone_stages = [backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4]
    for i, o in enumerate(backbone_stages):
        out_channels = o[-1].out_channels
        arm_loc_layers += [nn.Conv2d(out_channels, cfg[i] * 4,
                                     kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(out_channels, cfg[i] * 2,
                                      kernel_size=3, padding=1)]
    return arm_loc_layers, arm_conf_layers


def cr2_multibox(cfg, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    for i in range(0, 4):
        odm_loc_layers += [nn.Conv2d(256, cfg[i] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)


def add_fuse(cfg, vgg_output_channels):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [
            nn.Conv2d(vgg_output_channels[k], cfg[k], 1),
            nn.Conv2d(cfg[k], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [
            # BasicChannelAttention(in_channels=256, reduction=16),
            DCA(in_channels=256, reduction=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [
                nn.ConvTranspose2d(256, 256, 2, 2)
            ]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)


def build_dagnet(phase, size=512, num_classes=21):
    if phase != "test" and phase != "train":
        print("Only support train or test")
        return
    if size != 512:
        print("Only support 512")
        return

    base = TCBBackbone(num_blocks=[4, 6, 16, 1],
                       num_classes=1000)
    cr1 = cr1_multibox(base, [3, 3, 3, 3])
    cr2 = cr2_multibox([3, 3, 3, 3], num_classes)
    f = add_fuse([512, 512, 1024, 512], [128, 256, 512, 2048])
    return DAGNet(phase,
                  size,
                  base,
                  cr1,
                  cr2,
                  f,
                  num_classes)


if __name__ == '__main__':
    net = build_dagnet('train', size=512, num_classes=2)
    a = torch.randn((3, 64, 64)).unsqueeze(0)
    b = net(a)
