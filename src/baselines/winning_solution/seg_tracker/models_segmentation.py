import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config
import timm
import numpy as np

import effdet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet, BiFpn, get_feature_info
from omegaconf import OmegaConf
from . import models_dla_conv

from collections import OrderedDict
from typing import List, Callable, Optional, Union, Tuple


class HRNetSegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']
        feature_location = cfg['feature_location']
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.upscale_mode = cfg.get('upscale_mode', 'nearest')
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        self.pred_scale = cfg['pred_scale']

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            feature_location=feature_location,
                                            out_indices=(1, 2, 3, 4),
                                            in_chans=input_depth,
                                            pretrained=pretrained)
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name} upscale: {self.upscale_mode} comb outputs: {self.combine_outputs_dim}')
        print(f"Feature channels: {self.backbone_depths}")
        # self.backbone_depths = {
        #     "hrnet_w32": [32, 64, 128, 256],
        #     "hrnet_w48": [48, 96, 192, 384],
        #     "hrnet_w64": [64, 128, 256, 512],
        #     "hrnet_w18": [18, 36, 72, 144],
        #     "hrnet_w18_small_v2": [18, 36, 72, 144],
        # }[base_model_name]
        hrnet_outputs = sum(self.backbone_depths)

        if self.combine_outputs_dim > 0:
            self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
            self.fc_comb = nn.Conv2d(hrnet_outputs, self.combine_outputs_dim,
                                     kernel_size=self.combine_outputs_kernel)
                                     # padding=self.combine_outputs_kernel - 1 // 2)
            hrnet_outputs = self.combine_outputs_dim

        self.fc_cls = nn.Conv2d(hrnet_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        if self.output_binary_mask:
            self.fc_mask = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)

    def freeze_encoder(self):
        """Freeze all backbone parameters regardless of backbone type."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all backbone parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        stages_output = self.base_model(inputs)
        # print([xi.shape for xi in stages_output])

        if self.pred_scale == 4:
            if self.upscale_mode == 'linear':
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[2], scale_factor=4, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[3], scale_factor=8, mode=self.upscale_mode, align_corners=False),
                ]
            else:
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode="nearest"),
                    F.interpolate(stages_output[2], scale_factor=4, mode="nearest"),
                    F.interpolate(stages_output[3], scale_factor=8, mode="nearest"),
                ]
        elif self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),
                stages_output[1],
                F.interpolate(stages_output[2], scale_factor=2, mode="nearest"),
                F.interpolate(stages_output[3], scale_factor=4, mode="nearest"),
            ]
        else:
            raise RuntimeError('Invalid pred_scale')

        x = torch.cat(x, dim=1)
        # print(x.shape)

        if self.combine_outputs_dim > 0:
            # print(x.shape)
            x = F.relu(self.fc_comb(x))
            # print(x.shape)

        cls = self.fc_cls(x)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls=cls,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)

        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HeadConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.conv3 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class HeadConvV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.conv3(x))
        return x


class HRNetSegmentationSeparateHeads(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']
        feature_location = cfg['feature_location']
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.upscale_mode = cfg.get('upscale_mode', 'nearest')
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        self.pred_scale = cfg['pred_scale']

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            feature_location=feature_location,
                                            out_indices=(1, 2, 3, 4),
                                            in_chans=input_depth,
                                            pretrained=pretrained)
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name} upscale: {self.upscale_mode} comb outputs: {self.combine_outputs_dim}')
        print(f"Feature channels: {self.backbone_depths}")
        # self.backbone_depths = {
        #     "hrnet_w32": [32, 64, 128, 256],
        #     "hrnet_w48": [48, 96, 192, 384],
        #     "hrnet_w64": [64, 128, 256, 512],
        #     "hrnet_w18": [18, 36, 72, 144],
        #     "hrnet_w18_small_v2": [18, 36, 72, 144],
        # }[base_model_name]
        hrnet_outputs = sum(self.backbone_depths)

        self.head_cls = HeadConv(hrnet_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(hrnet_outputs, self.combine_outputs_dim)

        hrnet_outputs = self.combine_outputs_dim

        self.fc_size = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_mask = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.freeze_encoder()

    def unfreeze_encoder(self):
        self.base_model.unfreeze_encoder()

    def forward(self, inputs):
        stages_output = self.base_model(inputs)
        # print([xi.shape for xi in stages_output])

        if self.pred_scale == 4:
            if self.upscale_mode == 'linear':
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[2], scale_factor=4, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[3], scale_factor=8, mode=self.upscale_mode, align_corners=False),
                ]
            else:
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode="nearest"),
                    F.interpolate(stages_output[2], scale_factor=4, mode="nearest"),
                    F.interpolate(stages_output[3], scale_factor=8, mode="nearest"),
                ]
        elif self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),
                stages_output[1],
                F.interpolate(stages_output[2], scale_factor=2, mode="nearest"),
                F.interpolate(stages_output[3], scale_factor=4, mode="nearest"),
            ]
        else:
            raise RuntimeError('Invalid pred_scale')

        x = torch.cat(x, dim=1)
        # print(x.shape)

        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        mask = self.fc_mask(x_cls)
        size = self.fc_size(x_reg)
        offset = self.fc_offset(x_reg)
        distance = self.fc_distance(x_reg)
        tracking = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        res = dict(
            cls=mask,
            mask=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=above_horizon
        )

        return res



class EffDetSegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            min_level=2,
            max_level=7,
            num_levels=7-2+1,
            image_size=(512, 512),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_indices=(1, 2, 3, 4),
            backbone_out_indices=(0, 1, 2, 3, 4),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': custom_backbone})
            print('Using custom backbone', custom_backbone)

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs}')

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        if self.output_binary_mask:
            self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)

        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)

    def forward(self, inputs):
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        # print([xi.shape for xi in x])
        x = x[0]

        cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls=cls,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)

        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


class EffDetSegmentation8x(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)
        self.output_binary_mask = cfg.get('output_binary_mask', True)
        self.output_above_horizon = cfg.get('output_above_horizon', True)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            image_size=(512, 512),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            self.custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': self.custom_backbone})
            print('Using custom backbone', self.custom_backbone)
        else:
            self.custom_backbone = None

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs}')

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
        models_dla_conv.fill_fc_weights(self.fc_horizon)

    def freeze_encoder(self):
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        if self.custom_backbone == 'gernet_m':
            self.model.backbone.stem.train()
            self.model.backbone.stem.requires_grad = True
            for param in self.model.backbone.stem.parameters():
                param.requires_grad = True

        # self.model.backbone.conv_stem.train()
        # self.model.backbone.conv_stem.requires_grad = True
        # self.model.backbone.bn1.train()
        # self.model.backbone.bn1.requires_grad = True

    def unfreeze_encoder(self):
        self.model.backbone.requires_grad = True
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        # print([xi.shape for xi in x])
        x = x[0]

        # cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            mask=self.fc_mask(x),
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=self.fc_horizon(x)
        )

        return res


class EffDetSegmentation8xSeparateHeads(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', 256)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            image_size=(512, 512),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            self.custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': self.custom_backbone})
            print('Using custom backbone', self.custom_backbone)
        else:
            self.custom_backbone = None

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs} {self.combine_outputs_dim}')

        self.head_cls = HeadConv(model_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(model_outputs, self.combine_outputs_dim)
        model_outputs = self.combine_outputs_dim

        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
        models_dla_conv.fill_fc_weights(self.fc_horizon)
        models_dla_conv.fill_fc_weights(self.fc_mask)

    def freeze_encoder(self):
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        if self.custom_backbone == 'gernet_m':
            self.model.backbone.stem.train()
            self.model.backbone.stem.requires_grad = True
            for param in self.model.backbone.stem.parameters():
                param.requires_grad = True
        #
        # self.model.backbone.conv_stem.train()
        # self.model.backbone.conv_stem.requires_grad = True
        # self.model.backbone.bn1.train()
        # self.model.backbone.bn1.requires_grad = True

    def unfreeze_encoder(self):
        self.model.backbone.requires_grad = True
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        # print([xi.shape for xi in x])
        x = x[0]

        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        mask = self.fc_mask(x_cls)
        size = self.fc_size(x_reg)
        offset = self.fc_offset(x_reg)
        distance = self.fc_distance(x_reg)
        tracking = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        res = dict(
            mask=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=above_horizon
        )

        return res


class DLA8xSeparateHeads(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']
        self.combine_outputs_dim = cfg['combine_outputs_dim']

        self.output_binary_mask = True
        self.output_above_horizon = True

        self.backbone = timm.create_model(base_model_name,
                                            features_only=True,
                                            out_indices=(1, 2, 3, 4),
                                            in_chans=input_depth,
                                            pretrained=pretrained)
        self.backbone_depths = list(self.backbone.feature_info.channels())

        model_outputs = sum(self.backbone_depths)
        print(f'{base_model_name} {self.backbone_depths}')

        self.head_cls = HeadConv(model_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(model_outputs, self.combine_outputs_dim)
        model_outputs = self.combine_outputs_dim

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
        models_dla_conv.fill_fc_weights(self.fc_horizon)
        models_dla_conv.fill_fc_weights(self.fc_mask)

    def forward(self, inputs):
        stages_output = self.backbone(inputs)
        # print([xi.shape for xi in stages_output])
        # x = x[-1]

        x = [
            F.avg_pool2d(stages_output[0], 4),
            F.avg_pool2d(stages_output[1], 2),
            stages_output[2],
            F.interpolate(stages_output[3], scale_factor=2, mode="nearest"),
        ]
        x = torch.cat(x, dim=1)

        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        mask = self.fc_mask(x_cls)
        size = self.fc_size(x_reg)
        offset = self.fc_offset(x_reg)
        distance = self.fc_distance(x_reg)
        tracking = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        res = dict(
            # cls=mask,
            mask=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=above_horizon
        )

        return res

    def freeze_encoder(self):
        """Freeze all backbone parameters regardless of backbone type."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def tsm(tensor, duration, dilation=1):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    shift_size = size[1] // 8
    mix_tensor, peri_tensor = tensor.split([shift_size, 7 * shift_size], dim=2)
    mix_tensor = mix_tensor[:, (1, 0), :, :]
    return torch.cat((mix_tensor, peri_tensor), dim=2).view(size)


def add_tsm_to_module(obj, duration, dilation=1):
    orig_forward = obj.forward

    def updated_forward(*args, **kwargs):
        a = (tsm(args[0], duration=duration, dilation=dilation), ) + args[1:]
        return orig_forward(*a, **kwargs)

    obj.forward = updated_forward

    return obj


class EffDetTsmSegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = 1  # cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            min_level=2,
            max_level=7,
            num_levels=7-2+1,
            image_size=(512, 512),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_indices=(1, 2, 3, 4),
            backbone_out_indices=(0, 1, 2, 3, 4),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': custom_backbone})
            print('Using custom backbone', custom_backbone)

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs}')

        duration = 1
        if 'resnet' in h['backbone_name']:
            for l in self.model.backbone.layer1:
                add_tsm_to_module(l.conv1, duration)

            for l in self.model.backbone.layer2:
                add_tsm_to_module(l.conv1, duration)

            for l in self.model.backbone.layer3:
                add_tsm_to_module(l.conv1, duration)

            for l in self.model.backbone.layer4:
                add_tsm_to_module(l.conv1, duration)

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

    def forward(self, inputs):
        # inputs: B x T x H x W
        B, T, H, W = inputs.shape

        x = inputs.view(B * T, 1, H, W)
        x = self.model.backbone(x)
        x = self.model.fpn(x)
        # print([xi.shape for xi in x])
        x = x[0]

        # x: B*T x C x H x W
        x = x.view(B, -1, x.shape[1], x.shape[2], x.shape[3])
        x = x[:, 1]  # only the current frame

        cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls=cls,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        return res



class DLASegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        input_depth = cfg['input_frames']

        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        down_ratio = cfg['pred_scale']

        # down_ratio = 8
        last_level = 5
        final_kernel = 1

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.backbone = models_dla_conv.dla34(pretrained=pretrained)
        self.backbone.base_layer = nn.Sequential(
            nn.Conv2d(input_depth, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(inplace=True))

        channels = self.backbone.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = models_dla_conv.DLAUp(self.first_level, channels[self.first_level:], scales)

        out_channel = channels[self.first_level]

        self.ida_up = models_dla_conv.IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        model_outputs = channels[self.first_level]
        print(f'DLASegmentation8x {model_outputs}')

        if self.combine_outputs_dim > 0:
            self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
            self.fc_comb = nn.Conv2d(model_outputs, self.combine_outputs_dim,
                                     kernel_size=self.combine_outputs_kernel)
            model_outputs = self.combine_outputs_dim

        # self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)

        # models_dla_conv.fill_fc_weights(self.fc_cls)
        models_dla_conv.fill_fc_weights(self.fc_size)
        models_dla_conv.fill_fc_weights(self.fc_offset)
        models_dla_conv.fill_fc_weights(self.fc_distance)
        models_dla_conv.fill_fc_weights(self.fc_tracking)
        models_dla_conv.fill_fc_weights(self.fc_mask)

        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
            models_dla_conv.fill_fc_weights(self.fc_horizon)

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        x = y[-1]
        #
        # x = self.dla_up(x)
        # print([xi.shape for xi in x])
        # x = x[0]

        if self.combine_outputs_dim > 0:
            # print(x.shape)
            x = F.relu(self.fc_comb(x))
            # print(x.shape)

        #cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)
        mask = self.fc_mask(x)

        res = dict(
            mask=mask,
            cls=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


class MobileNetV4Segmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name   = cfg['base_model_name']
        input_depth       = cfg['input_frames']
        self.combine_outputs_dim    = cfg.get('combine_outputs_dim', 256)
        self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
        self.output_binary_mask     = cfg.get('output_binary_mask', False)
        self.output_above_horizon   = cfg.get('output_above_horizon', False)
        self.pred_scale             = cfg['pred_scale']

        # MobileNetV4 in timm supports features_only=True with in_chans override.
        # ImageNet weights are averaged across channels when in_chans != 3,
        # so pretrained=True still gives a meaningful starting point.
        self.base_model = timm.create_model(
            base_model_name,
            features_only = True,
            out_indices   = (1, 2, 3, 4),
            in_chans      = input_depth,
            pretrained    = pretrained,
        )
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name} feature channels: {self.backbone_depths}')

        total_channels = sum(self.backbone_depths)

        # Project to a uniform width before the detection heads.
        # MobileNetV4-Small has 960 channels in the final stage which would
        # dominate without this projection step.
        self.fc_comb = nn.Conv2d(
            total_channels,
            self.combine_outputs_dim,
            kernel_size = self.combine_outputs_kernel,
        )
        head_ch = self.combine_outputs_dim

        # Detection heads — identical to HRNetSegmentation
        self.fc_cls      = nn.Conv2d(head_ch, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(head_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(head_ch, 2, kernel_size=1)
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(head_ch, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(head_ch, 1, kernel_size=1)

    def freeze_encoder(self):
        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.base_model.parameters():
            p.requires_grad_(True)

    def forward(self, inputs):
        stages_output = self.base_model(inputs)
        # stages_output: list of 4 tensors at strides [4, 8, 16, 32]

        # Upsample / downsample all stages to pred_scale resolution
        if self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),            # stride 4  -> 8
                stages_output[1],                              # stride 8  (reference)
                F.interpolate(stages_output[2], scale_factor=2, mode='nearest'),  # 16 -> 8
                F.interpolate(stages_output[3], scale_factor=4, mode='nearest'),  # 32 -> 8
            ]
        elif self.pred_scale == 4:
            x = [
                stages_output[0],                              # stride 4  (reference)
                F.interpolate(stages_output[1], scale_factor=2, mode='nearest'),
                F.interpolate(stages_output[2], scale_factor=4, mode='nearest'),
                F.interpolate(stages_output[3], scale_factor=8, mode='nearest'),
            ]
        else:
            raise RuntimeError(f'Invalid pred_scale: {self.pred_scale}')

        x = torch.cat(x, dim=1)
        x = F.relu(self.fc_comb(x))

        cls      = self.fc_cls(x)
        size     = self.fc_size(x)
        offset   = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls      = cls,
            size     = size,
            offset   = offset,
            distance = distance,
            tracking = tracking,
        )

        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)
        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


class EfficientViTSegmentation(nn.Module):
    # Known output channel widths for each EfficientViT-M variant.
    # Output channels of [stage_0, stage_1, stage_2] after patch_embed.
    _EFFICIENTVIT_CHANNELS = {
        'efficientvit_m0': [64,  128, 192],
        'efficientvit_m1': [128, 144, 192],
        'efficientvit_m2': [128, 192, 224],
        'efficientvit_m3': [128, 240, 320],
        'efficientvit_m4': [128, 256, 384],
        'efficientvit_m5': [192, 288, 384],
    }

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name             = cfg['base_model_name']
        input_depth                 = cfg['input_frames']
        self.combine_outputs_dim    = cfg.get('combine_outputs_dim', 256)
        self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
        self.output_binary_mask     = cfg.get('output_binary_mask', False)
        self.output_above_horizon   = cfg.get('output_above_horizon', False)
        self.pred_scale             = cfg['pred_scale']

        # Backbone: two-step loading required for non-default input sizes.
        input_size = cfg.get('input_size', 512)

        # Build architecture at the correct input resolution.
        self.base_model = timm.create_model(
            base_model_name,
            pretrained  = False,        # Weights loaded manually in Step 2.
            in_chans    = input_depth,
            num_classes = 0,            # Remove classifier head.
            img_size    = input_size,
        )

        # Transfer ImageNet weights, skipping shape-mismatched keys.
        # All other weights (conv, BN, linear) are transferred cleanly.
        if pretrained:
            _tmp = timm.create_model(
                base_model_name,
                pretrained  = True,
                num_classes = 0,
            )
            src_sd  = _tmp.state_dict()
            tgt_sd  = self.base_model.state_dict()
            del _tmp

            # Keep only keys whose tensor shapes match exactly in both dicts.
            filtered_sd = {
                k: v for k, v in src_sd.items()
                if k in tgt_sd and v.shape == tgt_sd[k].shape
            }
            skipped = [k for k in src_sd if k not in filtered_sd]

            missing, unexpected = self.base_model.load_state_dict(
                filtered_sd, strict=False
            )

            # Expected skipped keys, log for transparency.
            expected_skips = {'patch_embed.conv1.conv.weight'} | {k for k in skipped if 'attention_bias' in k}
            unexpected_skips = [k for k in skipped if k not in expected_skips]
            if unexpected_skips:
                print(f'[EfficientViTSegmentation] WARNING — unexpected shape '
                      f'mismatches skipped during pretrained load: {unexpected_skips}')
            else:
                print(f'[EfficientViTSegmentation] Pretrained weights loaded. '
                      f'Skipped {len(skipped)} shape-mismatched keys '
                      f'(patch_embed stem + {len(skipped)-1} attention bias tables). '
                      f'These will be learned from scratch.')

        # Look up feature channel widths; fall back to a runtime probe if the
        # exact variant string isn't in the table (e.g. with a weight tag suffix).
        _key = base_model_name.split('.')[0]   # strip any weight-tag suffix
        if _key in self._EFFICIENTVIT_CHANNELS:
            self.backbone_depths = self._EFFICIENTVIT_CHANNELS[_key]
        else:
            self.backbone_depths = None   # resolved after hook registration below

        # Forward hooks, attach to stage_0, stage_1, stage_2.
        # EfficientViT-M2 (MSRA) has 3 stages after patch_embed at strides
        # [16, 32, 64]. We hook all 3 stages (not patch_embed, since stage_0
        # runs at the same resolution with richer features).
        self._feature_maps: List[torch.Tensor] = []

        def _make_hook(self_ref):
            def _hook(module, input, output):
                self_ref._feature_maps.append(output)
            return _hook

        self._hooks = []
        for stage in self.base_model.stages.children():
            h = stage.register_forward_hook(_make_hook(self))
            self._hooks.append(h)

        # Runtime probe for unknown variants — requires hooks to be attached.
        if self.backbone_depths is None:
            with torch.no_grad():
                dummy = torch.zeros(1, input_depth, input_size, input_size)
                self._feature_maps.clear()
                self.base_model(dummy)
                self.backbone_depths = [f.shape[1] for f in self._feature_maps]
                self._feature_maps.clear()

        print(f'{base_model_name} feature channels: {self.backbone_depths}')
        total_channels = sum(self.backbone_depths)

        # Projection conv, collapses the concatenated multi-scale features
        # to a uniform width before the detection heads.
        # Kept at 256 / kernel=1 (same deliberate choice as MobileNetV4;
        # both are lightweight anchors on the complexity curve).
        self.fc_comb = nn.Conv2d(
            total_channels,
            self.combine_outputs_dim,
            kernel_size = self.combine_outputs_kernel,
        )
        head_ch = self.combine_outputs_dim

        # Detection heads, identical to MobileNetV4Segmentation.
        self.fc_cls      = nn.Conv2d(head_ch, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(head_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(head_ch, 2, kernel_size=1)
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(head_ch, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(head_ch, 1, kernel_size=1)

    def freeze_encoder(self):
        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.base_model.parameters():
            p.requires_grad_(True)

    # Native Input Size the Backbone Was Built For.
    # Inputs Larger than this are Automatically Tiled in forward().
    PATCH_SIZE = 512

    def _forward_patch(self, patch: torch.Tensor) -> dict:
        # Clear the capture buffer before each forward pass.
        self._feature_maps.clear()

        # Trigger backbone forward — hooks populate self._feature_maps.
        self.base_model(patch)

        # stages_output: [stage_0, stage_1, stage_2].
        stages_output = list(self._feature_maps)

        # Calculate target dimensions based on input resolution / pred_scale
        target_h = patch.shape[2] // self.pred_scale
        target_w = patch.shape[3] // self.pred_scale

        # Upsample all stages to pred_scale resolution.
        x = [
            F.interpolate(stages_output[0], size=(target_h, target_w),
                          mode='bilinear', align_corners=False),
            F.interpolate(stages_output[1], size=(target_h, target_w),
                          mode='bilinear', align_corners=False),
            F.interpolate(stages_output[2], size=(target_h, target_w),
                          mode='bilinear', align_corners=False),
        ]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc_comb(x))
 
        res = dict(
            cls      = self.fc_cls(x),
            size     = self.fc_size(x),
            offset   = self.fc_offset(x),
            distance = self.fc_distance(x),
            tracking = self.fc_tracking(x),
        )
        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)
        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)
        return res

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        ps = self.PATCH_SIZE
        out_scale = self.pred_scale
 
        # Fast path: input fits in one patch — no tiling overhead.
        if H <= ps and W <= ps:
            return self._forward_patch(inputs)
 
        # Tiled path: pad to multiple of patch_size, tile, stitch.
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(inputs, (0, pad_w, 0, pad_h))
        else:
            x_pad = inputs
 
        pH, pW   = x_pad.shape[2], x_pad.shape[3]
        rows, cols = pH // ps, pW // ps
        ops = ps // out_scale           # output patch size
 
        # Pre-allocate output grids on the same device as inputs.
        def _alloc(ch):
            return torch.zeros(B, ch, rows * ops, cols * ops,
                               device=inputs.device, dtype=inputs.dtype)
 
        # Run one patch to discover output channel counts.
        _probe = self._forward_patch(x_pad[:, :, :ps, :ps])
        grids = {k: _alloc(v.shape[1]) for k, v in _probe.items()}
 
        # Fill in the probe result for the first patch.
        for k, v in _probe.items():
            grids[k][:, :, :ops, :ops] = v
 
        # Process remaining patches.
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue   # already filled from probe
                patch = x_pad[:, :, r*ps:(r+1)*ps, c*ps:(c+1)*ps]
                pred  = self._forward_patch(patch)
                for k, v in pred.items():
                    grids[k][:, :, r*ops:(r+1)*ops, c*ops:(c+1)*ops] = v
 
        # Crop back to the original (unpadded) output dimensions.
        orig_oh = H // out_scale
        orig_ow = W // out_scale
        return {k: v[:, :, :orig_oh, :orig_ow] for k, v in grids.items()}


class EfficientViTB2Segmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name             = cfg['base_model_name']
        input_depth                 = cfg['input_frames']
        self.combine_outputs_dim    = cfg.get('combine_outputs_dim', 256)
        self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
        self.output_binary_mask     = cfg.get('output_binary_mask', False)
        self.output_above_horizon   = cfg.get('output_above_horizon', False)
        self.pred_scale             = cfg['pred_scale']

        self.base_model = timm.create_model(
            base_model_name,
            pretrained   = pretrained,
            features_only = True,
            in_chans     = input_depth,
        )

        # Probe feature channel widths from the model's own feature info.
        self.backbone_depths = [fi['num_chs'] for fi in self.base_model.feature_info]
        print(f'{base_model_name} feature channels: {self.backbone_depths}')
        total_channels = sum(self.backbone_depths)

        # Projection conv - collapses 720 concatenated channels to 256.
        self.fc_comb = nn.Conv2d(
            total_channels,
            self.combine_outputs_dim,
            kernel_size = self.combine_outputs_kernel,
        )
        head_ch = self.combine_outputs_dim

        # Detection heads - identical to MobileNetV4Segmentation.
        self.fc_cls      = nn.Conv2d(head_ch, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(head_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(head_ch, 2, kernel_size=1)
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(head_ch, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(head_ch, 1, kernel_size=1)

    def freeze_encoder(self):
        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.base_model.parameters():
            p.requires_grad_(True)

    def forward(self, inputs):
        stages_output = self.base_model(inputs)

        if self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),                                # stride 4  -> 8
                stages_output[1],                                                 # stride 8  (reference)
                F.interpolate(stages_output[2], scale_factor=2, mode='nearest'),  # stride 16 -> 8
                F.interpolate(stages_output[3], scale_factor=4, mode='nearest'),  # stride 32 -> 8
            ]
        elif self.pred_scale == 4:
            x = [
                stages_output[0],                                                 # stride 4  (reference)
                F.interpolate(stages_output[1], scale_factor=2, mode='nearest'),  # stride 8  -> 4
                F.interpolate(stages_output[2], scale_factor=4, mode='nearest'),  # stride 16 -> 4
                F.interpolate(stages_output[3], scale_factor=8, mode='nearest'),  # stride 32 -> 4
            ]
        else:
            raise RuntimeError(f'Invalid pred_scale: {self.pred_scale}')

        x = torch.cat(x, dim=1)
        x = F.relu(self.fc_comb(x))

        cls      = self.fc_cls(x)
        size     = self.fc_size(x)
        offset   = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls      = cls,
            size     = size,
            offset   = offset,
            distance = distance,
            tracking = tracking,
        )

        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)
        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


def print_summary():
    import pytorch_model_summary
    # model = HRNetSegmentation(cfg={'base_model_name': 'hrnet_w32', 'input_frames': 2, 'feature_location': '', 'upscale_mode': 'nearest', 'combine_outputs_dim': 512})
    # model = EffDetSegmentation(cfg={'base_model_name': 'tf_efficientdet_d3', 'input_frames': 2})
    # model = EffDetTsmSegmentation(cfg={'base_model_name': 'tf_efficientdet_d2', 'input_frames': 2, 'custom_backbone': 'dpn68b'})
    # model = DLASegmentation(cfg={'input_frames': 2})

    # model = EffDetSegmentation8x(cfg={'base_model_name': 'tf_efficientdet_d2', 'input_frames': 2, 'custom_backbone': 'halonet_h1_c4c5'})
    # model = PANSegmentation(cfg={'input_frames': 2, 'custom_backbone': 'resnet50', 'decoder_channels': 128})
    # model = UNetSegmentation(cfg={'input_frames': 2, 'custom_backbone': 'efficientnet-b5', 'decoder_channels': [512, 256, 4, 4, 4]})
    model = DLA8xSeparateHeads(cfg={'base_model_name': 'dla60_res2next', 'input_frames': 2, 'combine_outputs_dim': 256})
    model(torch.zeros((2, 2, 512, 512)))

    print(pytorch_model_summary.summary(model, torch.zeros((2, 2, 512, 512)), max_depth=2, show_hierarchical=False))


if __name__ == "__main__":
    print_summary()
