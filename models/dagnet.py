import os
from models.backbone import *
from layers import *
from data import voc_dagnet


class RefineDet(nn.Module):

    def __init__(self,
                 phase,
                 device,
                 backbone,
                 mbd1,
                 mbd2,
                 fas,
                 num_classes):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.device = device
        self.cfg = voc_dagnet

        self.priorbox = PriorBox(self.cfg['512'], device)
        with torch.no_grad():
            self.priors = self.priorbox.forward()

        self.backbone = backbone

        self.mbd1_loc = nn.ModuleList(mbd1[0])
        self.mbd1_conf = nn.ModuleList(mbd1[1])
        self.mbd2_loc = nn.ModuleList(mbd2[0])
        self.mbd2_conf = nn.ModuleList(mbd2[1])
        # self.fas0 = nn.ModuleList(fas[0])
        # self.fas1 = nn.ModuleList(fas[1])
        # self.fas2 = nn.ModuleList(fas[2])
        self.fas = nn.Sequential(*fas)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_DAGNet(num_classes=num_classes,
                                        bkg_label=0,
                                        top_k=1000,
                                        conf_thresh=0.6,
                                        nms_thresh=0.1,
                                        objectness_thre=0.1,
                                        keep_top_k=500)

    def forward(self, x):
        fas_source = list()
        mbd1_loc = list()
        mbd1_conf = list()
        mbd2_loc = list()
        mbd2_conf = list()

        _, sources = self.backbone(x)

        for (x, l, c) in zip(sources, self.mbd1_loc, self.mbd1_conf):
            mbd1_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            mbd1_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        mbd1_loc = torch.cat([o.view(o.size(0), -1) for o in mbd1_loc], 1)
        mbd1_conf = torch.cat([o.view(o.size(0), -1) for o in mbd1_conf], 1)

        # p = None
        # for k, v in enumerate(sources[::-1]):
        #     s = v
        #     for i in range(4):
        #         s = self.fas0[(3 - k) * 4 + i](s)
        #     if k != 0:
        #         u = p
        #         u = self.fas1[3 - k](u)
        #         s += u
        #     for i in range(4):
        #         s = self.fas2[(3 - k) * 4 + i](s)
        #     p = s
        #     fas_source.append(s)
        # fas_source.reverse()
        for i, (s, d) in enumerate(zip(sources[:3], sources[1:])):
            fas_source.append(self.fas[i](s, d))
        fas_source.append(self.fas[-1](sources[-1]))

        for (x, l, c) in zip(fas_source, self.mbd2_loc, self.mbd2_conf):
            mbd2_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            mbd2_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        mbd2_loc = torch.cat([o.view(o.size(0), -1) for o in mbd2_loc], 1)
        mbd2_conf = torch.cat([o.view(o.size(0), -1) for o in mbd2_conf], 1)

        if self.phase == "test":
            output = self.detect.forward(
                mbd1_loc.view(mbd1_loc.size(0), -1, 4),
                self.softmax(mbd1_conf.view(mbd1_conf.size(0), -1, 2)),
                mbd2_loc.view(mbd2_loc.size(0), -1, 4),
                self.softmax(mbd2_conf.view(mbd2_conf.size(0), -1, self.num_classes)),
                self.priors
            )
        else:
            output = (
                mbd1_loc.view(mbd1_loc.size(0), -1, 4),
                mbd1_conf.view(mbd1_conf.size(0), -1, 2),
                mbd2_loc.view(mbd2_loc.size(0), -1, 4),
                mbd2_conf.view(mbd2_conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def multibox_decode1(backbone, cfg):
    loc_layers = []
    conf_layers = []
    backbone_stages = [backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4]
    for i, o in enumerate(backbone_stages):
        out_channels = o[-1].out_channels
        loc_layers += [nn.Conv2d(out_channels, cfg[i] * 4,
                                 kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(out_channels, cfg[i] * 2,
                                  kernel_size=3, padding=1)]
    return loc_layers, conf_layers


def multibox_decode2(cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for i in range(0, 4):
        loc_layers += [nn.Conv2d(256, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers


class FeatureAggregator(nn.Module):
    def __init__(self,
                 s_in_channels: int,
                 out_channels: int,
                 last_stage: bool,
                 d_in_channels: int = None):
        super().__init__()
        self.last_stage = last_stage

        self.calibrate_conv1 = nn.Conv2d(s_in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.calibrate_conv2 = nn.Conv2d(out_channels, 256, 3, padding=1)

        if not last_stage:
            assert d_in_channels is not None
            self.deconv = nn.ConvTranspose2d(d_in_channels, 256, 2, 2)
            self.fuse = AttentionAtFusion(256, 3)
        else:
            self.relu2 = nn.ReLU()
            self.calibrate_conv3 = nn.Conv2d(256, 256, 3, padding=1)
            self.relu3 = nn.ReLU()

        self.sca = SCA(256, 4)

    def forward(self,
                s: torch.Tensor,
                d: torch.Tensor = None):

        assert (self.last_stage and d is None) or (not self.last_stage and d is not None), \
            'If initialized as last stage FA, there should not be a deep feature map fed.'
        s_calibrated = self.calibrate_conv2(self.relu1(self.calibrate_conv1(s)))

        if not self.last_stage:
            assert hasattr(self, 'deconv') and hasattr(self, 'fuse'), \
                'If not last stage FA, there should be a de-convlution and a fuse-attention operation.'

            c_upsampled = self.deconv(d)
            assert s_calibrated.shape[2] == c_upsampled.shape[2] and \
                   s_calibrated.shape[3] == c_upsampled.shape[3], \
                'Shallow feature map needs to have the same spatial size of deep feature map.'

            fused = self.fuse(s_calibrated, c_upsampled)
            out = self.sca(fused)
            return out

        else:
            assert hasattr(self, 'relu2') and hasattr(self, 'calibrate_conv3') and hasattr(self, 'relu3'), \
                'If last stage FA, there should only be two consecutive convolutions for the input feature map.'

            s_result = self.relu3(self.calibrate_conv3(self.relu2(s_calibrated)))
            out = self.sca(s_result)
            return out


def build_dagnet(phase,
                 device,
                 num_classes=21):
    if phase != "test" and phase != "train":
        print(f"ERROR: Phase: {phase} not recognized")
        return

    backbone = Backbone(num_blocks=[4, 6, 16, 1],
                        num_classes=num_classes,
                        width_multiplier=[2, 2, 2, 4],
                        override_groups_map=None)
    decode1 = multibox_decode1(backbone, [3, 3, 3, 3])
    decode2 = multibox_decode2([3, 3, 3, 3], num_classes)
    output_channels = [128, 256, 512, 2048]
    fas = [FeatureAggregator(s_in_channels=s_bb_dim,
                             d_in_channels=None if i == 3 else output_channels[i + 1],
                             out_channels=fa_dim,
                             last_stage=True if i == 3 else False)
           for i, (s_bb_dim, fa_dim) in enumerate(zip([128, 256, 512, 2048], [512, 512, 1024, 512]))]
    return RefineDet(phase,
                     device,
                     backbone,
                     decode1,
                     decode2,
                     fas,
                     num_classes)
