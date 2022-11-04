import mmcv.ops
import numpy as np

# from layers.modules.l2norm import L2Norm
# from models.attentions import *
from models.attentions import *


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class TCB(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False):
        super(TCB, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation,
                                         groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def backbone_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), \
               bias.detach().cpu().numpy(),


class Backbone(nn.Module):

    def __init__(self,
                 num_blocks,
                 num_classes=1000,
                 width_multiplier=None,
                 override_groups_map=None,
                 deploy=False):
        super(Backbone, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.size = 512

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = TCB(in_channels=3,
                          out_channels=self.in_planes,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          deploy=self.deploy)
        self.cur_layer_idx = 1

        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.attention1 = CDSA(int(64 * width_multiplier[0]), 4)

        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.attention2 = DCA(int(128 * width_multiplier[1]), 4, 4)

        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.attention3 = DCA(int(256 * width_multiplier[2]), 4, 4)

        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.attention4 = SCA(int(512 * width_multiplier[3]), 4)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(TCB(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                              stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        sources = []
        out = self.stage0(x)

        out = self.stage1(out)
        out = self.attention1(out)
        sources.append(out)

        out = self.stage2(out)
        out = self.attention2(out)
        sources.append(out)

        out = self.stage3(out)
        out = self.attention3(out)
        sources.append(out)

        out = self.stage4(out)
        out = self.attention4(out)
        sources.append(out)

        return None, sources


def whole_model_convert(train_model: torch.nn.Module,
                        deploy_model: torch.nn.Module):
    all_weights = {}
    for name, module in train_model.named_modules():
        if hasattr(module, 'backbone_convert'):
            kernel, bias = module.backbone_convert()
            all_weights[name + '.rbr_reparam.weight'] = torch.from_numpy(kernel)
            all_weights[name + '.rbr_reparam.bias'] = torch.from_numpy(bias)
        else:
            if 'backbone.linear' in name or \
                    'mbd1' in name or \
                    'mbd2' in name or \
                    'fas' in name or \
                    'attention' in name:
                for p_name, p_tensor in module.named_parameters():
                    full_name = name + '.' + p_name
                    if full_name not in all_weights:
                        all_weights[full_name] = p_tensor

                for p_name, p_tensor in module.named_buffers():
                    full_name = name + '.' + p_name
                    if full_name not in all_weights:
                        all_weights[full_name] = p_tensor

    deploy_model.load_state_dict(all_weights)
    return deploy_model


def backbone_model_convert(model: torch.nn.Module,
                           num_classes: int):
    converted_weights = {}

    for i, (name, module) in enumerate(model.named_modules()):

        if hasattr(module, 'backbone_convert'):
            kernel, bias = module.backbone_convert()
            converted_weights[name + '.rbr_reparam.weight'] = kernel
            converted_weights[name + '.rbr_reparam.bias'] = bias
        elif isinstance(module, torch.nn.Linear) or \
                isinstance(module, torch.nn.Conv2d) or \
                isinstance(module, mmcv.ops.DeformConv2d) or \
                isinstance(module, torch.nn.ConvTranspose2d):
            try:
                converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
                converted_weights[name + '.bias'] = module.bias.detach().cpu().numpy()
            except:
                continue
        elif isinstance(module, torch.nn.BatchNorm2d):
            try:
                converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
                converted_weights[name + '.bias'] = module.bias.detach().cpu().numpy()
                converted_weights[name + '.running_mean'] = module.running_mean.detach().cpu().numpy()
                converted_weights[name + '.running_var'] = module.running_var.detach().cpu().numpy()
                converted_weights[name + '.num_batches_tracked'] = module.num_batches_tracked.detach().cpu().numpy()
            except:
                continue
        # elif isinstance(module, L2Norm):
        #     try:
        #         converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
        #         converted_weights[name + '.gamma'] = module.gamma.detach().cpu().numpy()
        #         converted_weights[name + '.eps'] = module.eps.detach().cpu().numpy()
        #     except:
        #         continue

    deploy_model = Backbone(num_blocks=[4, 6, 16, 1],
                            num_classes=num_classes,
                            width_multiplier=[2, 2, 2, 4],
                            override_groups_map=None,
                            deploy=True)

    for i, (name, param) in enumerate(deploy_model.named_parameters()):
        try:
            param.data = torch.from_numpy(converted_weights[name]).float()
        except KeyError as e:
            print(e, f'Could not find key {name} in the dict converted_weights.')

    for i, (name, param) in enumerate(deploy_model.named_buffers()):
        try:
            param.data = torch.from_numpy(converted_weights[name]).float()
        except KeyError as e:
            print(e, f'Could not find key {name} in the dict converted_weights.')

    return deploy_model
