# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: models_segmentation.py %
# Date Created:            %date_created: Mon Mar 02 09:00:00 2021 %
# CM Version               %version: 1 %
# CM Instance              %instance: 3 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Defines all segmentation model architectures used in the airborne object
# detection and tracking pipeline.
#
# Each class wraps a pretrained backbone (loaded via timm or effdet) with a
# shared set of detection heads for classification, size, offset, distance,
# tracking, and optionally binary mask and above-horizon prediction.
#
# Architectures defined here:
#   - HRNetSegmentation               : HRNet backbone, shared head.
#   - HRNetSegmentationSeparateHeads  : HRNet backbone, dual cls/reg heads.
#   - EffDetSegmentation              : EfficientDet BiFPN, 4x scale.
#   - EffDetSegmentation8x            : EfficientDet BiFPN, 8x scale.
#   - EffDetSegmentation8xSeparateHeads : EfficientDet BiFPN, dual heads.
#   - DLA8xSeparateHeads              : timm DLA backbone, dual heads.
#   - EffDetTsmSegmentation           : EfficientDet + Temporal Shift Module.
#   - DLASegmentation                 : Custom DLA-34 backbone.
#   - MobileNetV4Segmentation         : MobileNetV4 backbone.
#   - EfficientViTSegmentation        : EfficientViT-M backbone (MSRA).
#   - EfficientViTB2Segmentation      : EfficientViT-B2 backbone (MIT).
#   - ResNet34Segmentation            : ResNet-34 backbone.
#
# USAGE INSTRUCTIONS:
#   - Models are instantiated from config dicts by the training and inference
#     entrypoints. Direct instantiation is also supported for profiling.
#
# ------------------------------------------------------------------------------
# -                          L I C E N S I N G                                 -
# ------------------------------------------------------------------------------
# Original work Copyright (c) 2021 Dmytro Poplavskiy.
# Integrated into this project under the terms of the original competition
# open-source requirements. Modifications and integration by Lockheed Martin
# Australia are protected under applicable copyright and intellectual property
# laws.
#
# All rights reserved. No part of this software may be reproduced, distributed,
# modified, or transmitted in any form or by any means without the prior written
# permission of Lockheed Martin Australia.
#
# Unauthorised use, copying, or distribution of this software is strictly
# prohibited.
# ------------------------------------------------------------------------------

import os
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from effdet import get_efficientdet_config, EfficientDet

from . import config
from . import models_dla_conv


# ------------------------------------------------------------------------------
#                           SHARED HEAD MODULES
# ------------------------------------------------------------------------------
class ConvRelu(nn.Module):
    """
    Convenience module: Conv2d --> BatchNorm --> ReLU with auto-padding.

    Padding is computed to preserve spatial dimensions for odd kernel sizes.

    Args:
        in_channels  (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size  (int): Convolution kernel size.
        stride       (int): Convolution stride.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        # Compute Symmetric Padding to Preserve Spatial Size for Odd Kernels.
        pad = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride      = stride,
            padding     = pad,
            bias        = False,
        )

        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HeadConv(nn.Module):
    """
    Three-layer detection head: 1x1 projection --> 3x3 → 3x3.

    All layers use ConvRelu (Conv --> BN --> ReLU). Used as the cls and reg
    heads in dual-head architectures.

    Args:
        in_channels  (int): Input feature channel width.
        out_channels (int): Output feature channel width.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvRelu(in_channels=in_channels,  out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.conv3 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class HeadConvV2(nn.Module):
    """
    Variant detection head: 1x1 --> 3x3 --> 1x1 (No BN on final layer).

    The final conv uses a plain Conv2d + ReLU instead of ConvRelu, giving
    slightly different regularisation behaviour at the output.

    Args:
        in_channels  (int): Input feature channel width.
        out_channels (int): Output feature channel width.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvRelu(in_channels=in_channels,  out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)

        # Final Layer: Plain Conv Without BatchNorm.
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(self.conv3(x))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           HRNET SEGMENTATION
# ------------------------------------------------------------------------------
class HRNetSegmentation(nn.Module):
    """
    HRNet-based segmentation model with a single shared detection head.

    Builds an HRNet backbone (via timm) and fuses multi-scale feature maps
    from four stages into a single tensor before applying parallel 1x1
    convolution heads for each detection target.

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name       : timm model string (e.g. 'hrnet_w32').
            input_frames          : Number of input channels.
            feature_location      : Feature extraction location for timm.
            pred_scale            : Output stride (4 or 8).
            combine_outputs_dim   : (Optional) Bottleneck channel width. -1 disables.
            combine_outputs_kernel: (Optional) Kernel size for bottleneck conv.
            upscale_mode          : (Optional) Interpolation mode ('nearest' or 'linear').
            output_binary_mask    : (Optional) If True, adds a binary mask head.
            output_above_horizon  : (Optional) If True, adds an above-horizon head.
        pretrained (bool): If True, loads ImageNet-pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name          = cfg['base_model_name']
        input_depth              = cfg['input_frames']
        feature_location         = cfg['feature_location']
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.upscale_mode        = cfg.get('upscale_mode', 'nearest')
        self.output_binary_mask  = cfg.get('output_binary_mask', False)
        self.output_above_horizon= cfg.get('output_above_horizon', False)
        self.pred_scale          = cfg['pred_scale']

        # Build HRNet Backbone via timm, Extracting Features from Stages 1-4.
        self.base_model = timm.create_model(
            base_model_name,
            features_only    = True,
            feature_location = feature_location,
            out_indices      = (1, 2, 3, 4),
            in_chans         = input_depth,
            pretrained       = pretrained,
        )
        self.backbone_depths = list(self.base_model.feature_info.channels())

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} upscale: {self.upscale_mode} comb outputs: {self.combine_outputs_dim}')
            print(f"Feature channels: {self.backbone_depths}")

        # Total Channels After Concatenating All Stage Outputs.
        hrnet_outputs = sum(self.backbone_depths)

        # Optional Bottleneck Conv to Reduce Concatenated Channel Width.
        if self.combine_outputs_dim > 0:
            self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
            self.fc_comb = nn.Conv2d(
                hrnet_outputs,
                self.combine_outputs_dim,
                kernel_size = self.combine_outputs_kernel,
            )
            hrnet_outputs = self.combine_outputs_dim

        # Detection Heads. One Per Prediction Target.
        self.fc_cls      = nn.Conv2d(hrnet_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)

        # Optional Heads.
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)


    def freeze_encoder(self):
        """
        Freezes all backbone parameters, preventing them from being updated
        during training. Useful for the initial warm-up phase where only the
        detection heads are trained.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        for param in self.base_model.parameters():
            param.requires_grad = True


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Set the Output Stages for the Model. 
        stages_output = self.base_model(inputs)

        # Align All Stage Outputs to pred_scale Resolution Before Concatenation.
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
                    F.interpolate(stages_output[1], scale_factor=2,  mode='nearest'),
                    F.interpolate(stages_output[2], scale_factor=4,  mode='nearest'),
                    F.interpolate(stages_output[3], scale_factor=8,  mode='nearest'),
                ]
        elif self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),
                stages_output[1],
                F.interpolate(stages_output[2], scale_factor=2, mode='nearest'),
                F.interpolate(stages_output[3], scale_factor=4, mode='nearest'),
            ]
        else:
            raise RuntimeError('Invalid pred_scale')

        # Concatenate Multi-Scale Features Along the Channel Dimension.
        x = torch.cat(x, dim=1)

        # Apply Optional Bottleneck Projection.
        if self.combine_outputs_dim > 0:
            x = F.relu(self.fc_comb(x))

        # Apply Detection Heads.
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

        # Append Optional Head Outputs if Configured.
        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)
        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   HRNET SEGMENTATION — SEPARATE HEADS
# ------------------------------------------------------------------------------
class HRNetSegmentationSeparateHeads(nn.Module):
    """
    HRNet-based segmentation model with separate classification and regression
    heads.

    Identical backbone to HRNetSegmentation but routes features through two
    independent HeadConv modules (head_cls, head_reg) before applying the
    per-target output convolutions. This gives the cls and reg branches
    independent capacity to specialise.

    Args:
        cfg (dict): Model configuration. Expected keys mirror HRNetSegmentation,
            with combine_outputs_dim required (no -1 fallback here).
        pretrained (bool): If True, loads ImageNet-pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name          = cfg['base_model_name']
        input_depth              = cfg['input_frames']
        feature_location         = cfg['feature_location']
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.upscale_mode        = cfg.get('upscale_mode', 'nearest')
        self.output_binary_mask  = cfg.get('output_binary_mask', False)
        self.output_above_horizon= cfg.get('output_above_horizon', False)
        self.pred_scale          = cfg['pred_scale']

        # Build HRNet Backbone via timm, Extracting Features from Stages 1-4.
        self.base_model = timm.create_model(
            base_model_name,
            features_only    = True,
            feature_location = feature_location,
            out_indices      = (1, 2, 3, 4),
            in_chans         = input_depth,
            pretrained       = pretrained,
        )
        self.backbone_depths = list(self.base_model.feature_info.channels())

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} upscale: {self.upscale_mode} comb outputs: {self.combine_outputs_dim}')
            print(f"Feature channels: {self.backbone_depths}")

        # Total Channels After Concatenating All Stage Outputs.
        hrnet_outputs = sum(self.backbone_depths)

        # Separate Classification and Regression Heads.
        self.head_cls = HeadConv(hrnet_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(hrnet_outputs, self.combine_outputs_dim)
        hrnet_outputs = self.combine_outputs_dim

        # Detection Output Convolutions — cls Branch.
        self.fc_mask    = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)

        # Detection Output Convolutions — reg Branch.
        self.fc_size     = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)


    def freeze_encoder(self):
        """
        Freezes all backbone parameters, preventing them from being updated
        during training. Useful for the initial warm-up phase where only the
        detection heads are trained.
        """
        self.base_model.freeze_encoder()


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        self.base_model.unfreeze_encoder()


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Set the Output Stages for the Model. 
        stages_output = self.base_model(inputs)

        # Align All Stage Outputs to pred_scale Resolution Before Concatenation.
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
                    F.interpolate(stages_output[1], scale_factor=2,  mode='nearest'),
                    F.interpolate(stages_output[2], scale_factor=4,  mode='nearest'),
                    F.interpolate(stages_output[3], scale_factor=8,  mode='nearest'),
                ]
        elif self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),
                stages_output[1],
                F.interpolate(stages_output[2], scale_factor=2, mode='nearest'),
                F.interpolate(stages_output[3], scale_factor=4, mode='nearest'),
            ]
        else:
            raise RuntimeError('Invalid pred_scale')

        # Concatenate Multi-Scale Features Along the Channel Dimension.
        x = torch.cat(x, dim=1)

        # Route Through Separate cls and reg Heads.
        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        # Apply Per-Target Output Convolutions.
        mask          = self.fc_mask(x_cls)
        size          = self.fc_size(x_reg)
        offset        = self.fc_offset(x_reg)
        distance      = self.fc_distance(x_reg)
        tracking      = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        return dict(
            cls           = mask,
            mask          = mask,
            size          = size,
            offset        = offset,
            distance      = distance,
            tracking      = tracking,
            above_horizon = above_horizon,
        )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       EFFICIENTDET SEGMENTATION (4x)
# ------------------------------------------------------------------------------
class EffDetSegmentation(nn.Module):
    """
    EfficientDet-based segmentation model with a BiFPN neck at 4x output scale.

    Uses the effdet library to construct an EfficientDet backbone + BiFPN,
    then attaches per-target 1x1 detection heads to the lowest-resolution
    FPN level (level 0).

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name      : effdet config string (e.g. 'tf_efficientdet_d0').
            input_frames         : Number of input channels.
            output_binary_mask   : (Optional) If True, adds a binary mask head.
            output_above_horizon : (Optional) If True, adds an above-horizon head.
            custom_backbone      : (Optional) Override the default backbone name.
        pretrained (bool): If True, loads pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name         = cfg['base_model_name']
        input_depth             = cfg['input_frames']
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon=cfg.get('output_above_horizon', False)

        # Build EfficientDet Config, Overriding Defaults for this Task.
        eff_config = get_efficientdet_config(base_model_name)
        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes         = 2,
            min_level           = 2,
            max_level           = 7,
            num_levels          = 7 - 2 + 1,
            image_size          = (512, 512),
            norm_kwargs         = dict(eps=.001, momentum=.01),
            backbone_indices    = (1, 2, 3, 4),
            backbone_out_indices= (0, 1, 2, 3, 4),
            backbone_args       = dict(in_chans=input_depth),
        ))

        # Optionally Override the Default EfficientNet Backbone.
        if 'custom_backbone' in cfg:
            custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': custom_backbone})
            if os.getenv("DEBUG_MODEL_INIT"):
                print('Using custom backbone', custom_backbone)

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(h)

        # Set Model With Given Parameters.
        self.model    = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} {model_outputs}')

        # Detection Heads: Applied to FPN Level 0 Output.
        self.fc_cls      = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        # Optional Heads.
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(model_outputs, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)

    def forward(self, inputs):
        # Extract Backbone Features then Apply BiFPN.
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        x = x[0]   # Use Lowest-Resolution FPN Level.

        # Scale cls Logits Down to Prevent Overconfident Initialisation.
        cls      = self.fc_cls(x * 0.1)
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
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       EFFICIENTDET SEGMENTATION (8x)
# ------------------------------------------------------------------------------
class EffDetSegmentation8x(nn.Module):
    """
    EfficientDet-based segmentation model at 8x output scale.

    Identical in structure to EffDetSegmentation but operates at a coarser
    output resolution. Binary mask and above-horizon heads are always active
    regardless of config (defaults to True).

    Args:
        cfg (dict): Model configuration. See EffDetSegmentation for keys.
        pretrained (bool): If True, loads pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name          = cfg['base_model_name']
        input_depth              = cfg['input_frames']

        # Both Optional Heads are Enabled by Default for this Variant.
        self.output_binary_mask  = cfg.get('output_binary_mask', True)
        self.output_above_horizon= cfg.get('output_above_horizon', True)

        # Build EfficientDet Config.
        eff_config = get_efficientdet_config(base_model_name)
        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes  = 2,
            image_size   = (512, 512),
            norm_kwargs  = dict(eps=.001, momentum=.01),
            backbone_args= dict(in_chans=input_depth),
        ))

        # Optionally Override the Default EfficientNet Backbone.
        if 'custom_backbone' in cfg:
            self.custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': self.custom_backbone})
            if os.getenv("DEBUG_MODEL_INIT"):
                print('Using custom backbone', self.custom_backbone)
        else:
            self.custom_backbone = None

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(h)

        # Set Model With Given Parameters.
        self.model    = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} {model_outputs}')

        # Detection Heads.
        self.fc_cls      = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_mask     = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon  = nn.Conv2d(model_outputs, 1, kernel_size=1)

        # Initialise Horizon Head Weights to Zero Bias (Standard Fill).
        models_dla_conv.fill_fc_weights(self.fc_horizon)


    def freeze_encoder(self):
        """
        Freezes the EfficientDet backbone.

        For GerNet-M backbones, the stem layers are kept trainable to allow
        the input projection to adapt to non-ImageNet channel counts.
        """
        # Freeze the Backbone Layers.
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Keep GerNet-M Stem Trainable for Input Adaptation.
        if self.custom_backbone == 'gernet_m':
            self.model.backbone.stem.train()
            self.model.backbone.stem.requires_grad = True
            for param in self.model.backbone.stem.parameters():
                param.requires_grad = True


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        self.model.backbone.requires_grad = True
        for param in self.model.backbone.parameters():
            param.requires_grad = True


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Extract Backbone Features then Apply BiFPN.
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        x = x[0]   # Use Lowest-Resolution FPN Level.

        size     = self.fc_size(x)
        offset   = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        return dict(
            mask          = self.fc_mask(x),
            size          = size,
            offset        = offset,
            distance      = distance,
            tracking      = tracking,
            above_horizon = self.fc_horizon(x),
        )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#              EFFICIENTDET SEGMENTATION (8x) — SEPARATE HEADS
# ------------------------------------------------------------------------------
class EffDetSegmentation8xSeparateHeads(nn.Module):
    """
    EfficientDet-based segmentation model at 8x scale with separate cls/reg
    heads.

    Extends EffDetSegmentation8x by routing FPN features through dual HeadConv
    branches before the per-target output convolutions.

    Args:
        cfg (dict): Model configuration. See EffDetSegmentation for keys.
            combine_outputs_dim : Width of the dual head output (default 256).
        pretrained (bool): If True, loads pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name          = cfg['base_model_name']
        input_depth              = cfg['input_frames']
        self.output_binary_mask  = cfg.get('output_binary_mask', False)
        self.output_above_horizon= cfg.get('output_above_horizon', False)
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', 256)

        # Build EfficientDet Config.
        eff_config = get_efficientdet_config(base_model_name)
        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes  = 2,
            image_size   = (512, 512),
            norm_kwargs  = dict(eps=.001, momentum=.01),
            backbone_args= dict(in_chans=input_depth),
        ))

        # Optionally Override the Default EfficientNet Backbone.
        if 'custom_backbone' in cfg:
            self.custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': self.custom_backbone})
            if os.getenv("DEBUG_MODEL_INIT"):
                print('Using custom backbone', self.custom_backbone)
        else:
            self.custom_backbone = None

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(h)

        # Set Model With Given Parameters.
        self.model    = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} {model_outputs} {self.combine_outputs_dim}')

        # Separate Classification and Regression Heads.
        self.head_cls = HeadConv(model_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(model_outputs, self.combine_outputs_dim)
        model_outputs = self.combine_outputs_dim

        # Detection Output Convolutions — reg Branch.
        self.fc_size     = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        # Detection Output Convolutions — cls Branch.
        self.fc_mask    = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)

        # Initialise cls Head Weights to Zero Bias.
        models_dla_conv.fill_fc_weights(self.fc_horizon)
        models_dla_conv.fill_fc_weights(self.fc_mask)


    def freeze_encoder(self):
        """
        Freezes the EfficientDet backbone.

        For GerNet-M backbones, the stem layers are kept trainable to allow
        the input projection to adapt to non-ImageNet channel counts.
        """
        # Freeze the Backbone Layers.
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Keep GerNet-M Stem Trainable for Input Adaptation.
        if self.custom_backbone == 'gernet_m':
            self.model.backbone.stem.train()
            self.model.backbone.stem.requires_grad = True
            for param in self.model.backbone.stem.parameters():
                param.requires_grad = True


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        self.model.backbone.requires_grad = True
        for param in self.model.backbone.parameters():
            param.requires_grad = True


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Extract Backbone Features then Apply BiFPN.
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        x = x[0]   # Use Lowest-Resolution FPN Level.

        # Route Through Separate cls and reg Heads.
        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        # Apply Per-Target Output Convolutions.
        mask          = self.fc_mask(x_cls)
        size          = self.fc_size(x_reg)
        offset        = self.fc_offset(x_reg)
        distance      = self.fc_distance(x_reg)
        tracking      = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        return dict(
            mask          = mask,
            size          = size,
            offset        = offset,
            distance      = distance,
            tracking      = tracking,
            above_horizon = above_horizon,
        )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   DLA (8x) — SEPARATE HEADS
# ------------------------------------------------------------------------------
class DLA8xSeparateHeads(nn.Module):
    """
    timm DLA backbone at 8x output scale with separate cls/reg heads.

    Builds a DLA backbone via timm and fuses four stage outputs at stride 8
    using average pooling and bilinear upsampling before dual HeadConv branches.

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name      : timm DLA model string.
            input_frames         : Number of input channels.
            combine_outputs_dim  : Width of the dual head output (required).
        pretrained (bool): If True, loads ImageNet-pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name          = cfg['base_model_name']
        input_depth              = cfg['input_frames']
        self.combine_outputs_dim = cfg['combine_outputs_dim']

        # Both Optional Heads are Always Active for this Variant.
        self.output_binary_mask  = True
        self.output_above_horizon= True

        # Build timm DLA Backbone, Extracting Features from Stages 1-4.
        self.backbone = timm.create_model(
            base_model_name,
            features_only = True,
            out_indices   = (1, 2, 3, 4),
            in_chans      = input_depth,
            pretrained    = pretrained,
        )
        self.backbone_depths = list(self.backbone.feature_info.channels())

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} {self.backbone_depths}')

        # Total Channels After Concatenating All Stage Outputs.
        model_outputs = sum(self.backbone_depths)

        # Separate Classification and Regression Heads.
        self.head_cls = HeadConv(model_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(model_outputs, self.combine_outputs_dim)
        model_outputs = self.combine_outputs_dim

        # Detection Output Convolutions.
        self.fc_cls      = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_mask     = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon  = nn.Conv2d(model_outputs, 1, kernel_size=1)

        # Initialise cls Head Weights to Zero Bias.
        models_dla_conv.fill_fc_weights(self.fc_horizon)
        models_dla_conv.fill_fc_weights(self.fc_mask)


    def freeze_encoder(self):
        """
        Freezes all backbone parameters, preventing them from being updated
        during training. Useful for the initial warm-up phase where only the
        detection heads are trained.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Set the Output Stages for the Model. 
        stages_output = self.backbone(inputs)

        # Align All Stage Outputs to Stride 8 Resolution.
        x = [
            F.avg_pool2d(stages_output[0], 4),                                # stride 4  -> 8 (via 2x2 pool)
            F.avg_pool2d(stages_output[1], 2),                                # stride 4  -> 8
            stages_output[2],                                                 # stride 8  (reference)
            F.interpolate(stages_output[3], scale_factor=2, mode='nearest'),  # stride 16 -> 8
        ]

        # Concatenate Multi-Scale Features Along the Channel Dimension.
        x = torch.cat(x, dim=1)

        # Route Through Separate cls and reg Heads.
        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        # Apply Per-Target Output Convolutions.
        mask          = self.fc_mask(x_cls)
        size          = self.fc_size(x_reg)
        offset        = self.fc_offset(x_reg)
        distance      = self.fc_distance(x_reg)
        tracking      = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        return dict(
            mask          = mask,
            size          = size,
            offset        = offset,
            distance      = distance,
            tracking      = tracking,
            above_horizon = above_horizon,
        )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   TEMPORAL SHIFT MODULE (TSM) UTILITIES
# ------------------------------------------------------------------------------
def tsm(tensor, duration, dilation=1):
    """
    Applies a Temporal Shift Module (TSM) operation to a batched tensor.

    Shifts a fraction (1/8) of channels from the adjacent frame into the
    current frame's channel space, enabling temporal context without extra
    parameters. Operates on tensors shaped [N*T, C, H, W].

    Args:
        tensor   (torch.Tensor): Input tensor of shape [N*T, C, H, W].
        duration (int)         : Number of temporal frames T per sample.
        dilation (int)         : Temporal dilation (unused, reserved).

    Returns:
        torch.Tensor: Shifted tensor of the same shape as input.
    """
    # Get Size of a Tensor.
    size = tensor.size()

    # Reshape to [N, T, C, H, W] for Temporal Indexing.
    tensor = tensor.view((-1, duration) + size[1:])

    # Split Off the Channels to Be Shifted (1/8 of Total).
    shift_size = size[1] // 8
    mix_tensor, peri_tensor = tensor.split([shift_size, 7 * shift_size], dim=2)

    # Reverse the Temporal Order of the Shifted Channels (frames 0↔1).
    mix_tensor = mix_tensor[:, (1, 0), :, :]

    # Recombine and Flatten Back to [N*T, C, H, W].
    return torch.cat((mix_tensor, peri_tensor), dim=2).view(size)


def add_tsm_to_module(obj, duration, dilation=1):
    """
    Monkey-patches a module's forward method to apply TSM before each call.

    Wraps the existing forward function so that the first positional argument
    (the input tensor) is shifted temporally via tsm() on every forward pass.

    Args:
        obj      (nn.Module): Module to patch in-place.
        duration (int)      : Number of temporal frames per sample.
        dilation (int)      : Temporal dilation passed through to tsm().

    Returns:
        nn.Module: The same module with its forward method replaced.
    """
    # Store Original Forward Function.
    orig_forward = obj.forward

    # Apply TSM to the First Positional Argument (the Input Tensor).
    def updated_forward(*args, **kwargs):
        a = (tsm(args[0], duration=duration, dilation=dilation),) + args[1:]
        return orig_forward(*a, **kwargs)

    # Update the Forward Function.
    obj.forward = updated_forward

    return obj
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#               EFFICIENTDET + TEMPORAL SHIFT MODULE SEGMENTATION
# ------------------------------------------------------------------------------
class EffDetTsmSegmentation(nn.Module):
    """
    EfficientDet segmentation with Temporal Shift Module for multi-frame input.

    Processes a batch of T grayscale frames per sample by reshaping to
    [B*T, 1, H, W], running through a TSM-patched backbone, then selecting
    only the current-frame features (index 1) for detection head application.

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name      : effdet config string.
            custom_backbone      : (Optional) Override backbone name.
        pretrained (bool): If True, loads pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name = cfg['base_model_name']
        input_depth     = 1   # Always Single-Channel; Temporal Handled via TSM.
        eff_config = get_efficientdet_config(base_model_name)

        # Build EfficientDet Config.
        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes         = 2,
            min_level           = 2,
            max_level           = 7,
            num_levels          = 7 - 2 + 1,
            image_size          = (512, 512),
            norm_kwargs         = dict(eps=.001, momentum=.01),
            backbone_indices    = (1, 2, 3, 4),
            backbone_out_indices= (0, 1, 2, 3, 4),
            backbone_args       = dict(in_chans=input_depth),
        ))

        # Optionally Override the Default EfficientNet Backbone.
        if 'custom_backbone' in cfg:
            custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': custom_backbone})
            if os.getenv("DEBUG_MODEL_INIT"):
                print('Using custom backbone', custom_backbone)

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(h)

        # Set Model With Given Parameters.
        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} {model_outputs}')

        # Patch TSM Into Each ResNet Layer's First Conv if Using a ResNet Backbone.
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

        # Detection Heads.
        self.fc_cls      = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # inputs: [B, T, H, W] — Batch of T Grayscale Frames per Sample.
        B, T, H, W = inputs.shape

        # Flatten Temporal Dimension for Backbone Processing.
        x = inputs.view(B * T, 1, H, W)
        x = self.model.backbone(x)
        x = self.model.fpn(x)
        x = x[0]   # Use Lowest-Resolution FPN Level.

        # Unflatten and Select Only the Current Frame Features (Index 1).
        x = x.view(B, -1, x.shape[1], x.shape[2], x.shape[3])
        x = x[:, 1]

        # Scale cls Logits Down to Prevent Overconfident Initialisation.
        cls      = self.fc_cls(x * 0.1)
        size     = self.fc_size(x)
        offset   = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        return dict(
            cls      = cls,
            size     = size,
            offset   = offset,
            distance = distance,
            tracking = tracking,
        )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       DLA-34 SEGMENTATION
# ------------------------------------------------------------------------------
class DLASegmentation(nn.Module):
    """
    Custom DLA-34 backbone segmentation model.

    Uses the project-local models_dla_conv implementation of DLA-34 with a
    DLAUp neck and IDAUp upsampling module, rather than the timm version. The
    base_layer is replaced to accept the configured number of input channels.

    Args:
        cfg (dict): Model configuration. Expected keys:
            input_frames         : Number of input channels.
            pred_scale           : Output stride (down_ratio), typically 8.
            combine_outputs_dim  : (Optional) Bottleneck width. -1 disables.
            combine_outputs_kernel: (Optional) Bottleneck kernel size.
            output_binary_mask   : (Optional) If True, adds a binary mask head.
            output_above_horizon : (Optional) If True, adds an above-horizon head.
        pretrained (bool): If True, loads pretrained DLA-34 backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        input_depth              = cfg['input_frames']
        self.output_binary_mask  = cfg.get('output_binary_mask', False)
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.output_above_horizon= cfg.get('output_above_horizon', False)
        down_ratio               = cfg['pred_scale']

        last_level   = 5
        self.first_level = int(np.log2(down_ratio))
        self.last_level  = last_level

        # Build DLA-34 Backbone and Replace Input Stem for Configured Channels.
        self.backbone = models_dla_conv.dla34(pretrained=pretrained)
        self.backbone.base_layer = nn.Sequential(
            nn.Conv2d(input_depth, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        # Build DLAUp Neck and IDAUp Upsampling Module.
        channels = self.backbone.channels
        scales   = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up  = models_dla_conv.DLAUp(self.first_level, channels[self.first_level:], scales)
        out_channel  = channels[self.first_level]
        self.ida_up  = models_dla_conv.IDAUp(
            out_channel,
            channels[self.first_level:self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
        )

        # Set Model With Given Parameters.
        model_outputs = channels[self.first_level]

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'DLASegmentation8x {model_outputs}')

        # Optional Bottleneck Projection.
        if self.combine_outputs_dim > 0:
            self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
            self.fc_comb = nn.Conv2d(
                model_outputs,
                self.combine_outputs_dim,
                kernel_size = self.combine_outputs_kernel,
            )
            model_outputs = self.combine_outputs_dim

        # Detection Heads: Initialised with Zero Bias Fill.
        self.fc_size     = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_mask     = nn.Conv2d(model_outputs, 1, kernel_size=1)

        models_dla_conv.fill_fc_weights(self.fc_size)
        models_dla_conv.fill_fc_weights(self.fc_offset)
        models_dla_conv.fill_fc_weights(self.fc_distance)
        models_dla_conv.fill_fc_weights(self.fc_tracking)
        models_dla_conv.fill_fc_weights(self.fc_mask)

        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
            models_dla_conv.fill_fc_weights(self.fc_horizon)


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Run DLA-34 Backbone.
        x = self.backbone(inputs)

        # Apply DLAUp Neck.
        x = self.dla_up(x)

        # Collect and Upsample Features via IDAUp.
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        x = y[-1]

        # Apply Optional Bottleneck Projection.
        if self.combine_outputs_dim > 0:
            x = F.relu(self.fc_comb(x))

        size     = self.fc_size(x)
        offset   = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)
        mask     = self.fc_mask(x)

        res = dict(
            mask     = mask,
            cls      = mask,    # cls Aliases mask for Compatibility.
            size     = size,
            offset   = offset,
            distance = distance,
            tracking = tracking,
        )

        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       MOBILENETV4 SEGMENTATION
# ------------------------------------------------------------------------------
class MobileNetV4Segmentation(nn.Module):
    """
    MobileNetV4-based segmentation model.

    Builds a MobileNetV4 backbone via timm (features_only=True) and fuses
    multi-scale outputs at the configured pred_scale before a bottleneck
    projection and shared detection heads.

    ImageNet weights are channel-averaged when input channels differ from 3,
    so pretrained=True still provides a meaningful starting point.

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name      : timm MobileNetV4 model string.
            input_frames         : Number of input channels.
            pred_scale           : Output stride (4 or 8).
            combine_outputs_dim  : Bottleneck channel width (default 256).
            combine_outputs_kernel: Bottleneck kernel size (default 1).
            output_binary_mask   : (Optional) If True, adds a binary mask head.
            output_above_horizon : (Optional) If True, adds an above-horizon head.
        pretrained (bool): If True, loads ImageNet-pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name             = cfg['base_model_name']
        input_depth                 = cfg['input_frames']
        self.combine_outputs_dim    = cfg.get('combine_outputs_dim', 256)
        self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
        self.output_binary_mask     = cfg.get('output_binary_mask', False)
        self.output_above_horizon   = cfg.get('output_above_horizon', False)
        self.pred_scale             = cfg['pred_scale']

        # Build MobileNetV4 Backbone via timm, Extracting Features from Stages 1-4.
        self.base_model = timm.create_model(
            base_model_name,
            features_only = True,
            out_indices   = (1, 2, 3, 4),
            in_chans      = input_depth,
            pretrained    = pretrained,
        )
        self.backbone_depths = list(self.base_model.feature_info.channels())

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} feature channels: {self.backbone_depths}')

        # Total Channels After Concatenating All Stage Outputs.
        # MobileNetV4-Small Has 960 Channels at Stage 4. Projection is Essential.
        total_channels = sum(self.backbone_depths)

        # Bottleneck Projection to a Uniform Width Before Detection Heads.
        self.fc_comb = nn.Conv2d(
            total_channels,
            self.combine_outputs_dim,
            kernel_size = self.combine_outputs_kernel,
        )
        head_ch = self.combine_outputs_dim

        # Detection Heads: Identical Layout to HRNetSegmentation.
        self.fc_cls      = nn.Conv2d(head_ch, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(head_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(head_ch, 2, kernel_size=1)

        # Optional Heads.
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(head_ch, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(head_ch, 1, kernel_size=1)


    def freeze_encoder(self):
        """
        Freezes all backbone parameters, preventing them from being updated
        during training. Useful for the initial warm-up phase where only the
        detection heads are trained.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(False)


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(True)


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Set the Output Stages for the Model. 
        stages_output = self.base_model(inputs)

        # Align All Stage Outputs to pred_scale Resolution Before Concatenation.
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

        # Concatenate, Project, and Apply Detection Heads.
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
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   EFFICIENTVIT-M SEGMENTATION (MSRA)
# ------------------------------------------------------------------------------
class EfficientViTSegmentation(nn.Module):
    """
    EfficientViT-M (MSRA) backbone segmentation model with tiled inference.

    EfficientViT-M operates at a fixed native resolution (PATCH_SIZE = 512).
    For inputs larger than this, the forward method automatically pads and
    tiles the input, runs each patch independently, then stitches the output
    back together. This makes the model applicable to full-resolution flight
    frames without retraining.

    Pretrained ImageNet weights are loaded with shape-mismatch filtering:
    the patch_embed stem and attention bias tables are re-learned from scratch
    (incompatible input channel count and resolution), while all other weights
    are transferred cleanly.

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name      : timm EfficientViT-M model string.
            input_frames         : Number of input channels.
            pred_scale           : Output stride (used within each patch).
            input_size           : Native input resolution (default 512).
            combine_outputs_dim  : Bottleneck channel width (default 256).
            combine_outputs_kernel: Bottleneck kernel size (default 1).
            output_binary_mask   : (Optional) If True, adds a binary mask head.
            output_above_horizon : (Optional) If True, adds an above-horizon head.
        pretrained (bool): If True, transfers compatible ImageNet weights.
    """

    # Known Output Channel Widths for Each EfficientViT-M Variant.
    # Index Order: [stage_0, stage_1, stage_2] After patch_embed.
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

        # Extract Backbone & Architecture Config.
        base_model_name             = cfg['base_model_name']
        input_depth                 = cfg['input_frames']
        self.combine_outputs_dim    = cfg.get('combine_outputs_dim', 256)
        self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
        self.output_binary_mask     = cfg.get('output_binary_mask', False)
        self.output_above_horizon   = cfg.get('output_above_horizon', False)
        self.pred_scale             = cfg['pred_scale']
        input_size                  = cfg.get('input_size', 512)

        # Build Architecture at the Correct Input Resolution.
        # EfficientViT-M Requires img_size at Construction for Attention Bias Tables.
        self.base_model = timm.create_model(
            base_model_name,
            pretrained  = False,        # Weights Loaded Manually in Step 2.
            in_chans    = input_depth,
            num_classes = 0,            # Remove Classifier Head.
            img_size    = input_size,
        )

        # Transfer Compatible ImageNet Weights, Skip Shape Mismatches.
        if pretrained:
            _tmp = timm.create_model(base_model_name, pretrained=True, num_classes=0)
            src_sd = _tmp.state_dict()
            tgt_sd = self.base_model.state_dict()
            del _tmp

            # Keep Only Keys Whose Tensor Shapes Match Exactly in Both Dicts.
            filtered_sd = {
                k: v for k, v in src_sd.items()
                if k in tgt_sd and v.shape == tgt_sd[k].shape
            }
            skipped = [k for k in src_sd if k not in filtered_sd]

            missing, unexpected = self.base_model.load_state_dict(filtered_sd, strict=False)

            # Log Skipped Keys: Patch Embed Stem and Attention Bias Tables are Expected.
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

        # Resolve Feature Channel Widths from Lookup Table or Runtime Probe.
        _key = base_model_name.split('.')[0]   # Strip Any Weight-Tag Suffix.
        if _key in self._EFFICIENTVIT_CHANNELS:
            self.backbone_depths = self._EFFICIENTVIT_CHANNELS[_key]
        else:
            self.backbone_depths = None   # Resolved After Hook Registration Below.

        # Register Forward Hooks on stage_0, stage_1, stage_2 to Capture Features.
        # EfficientViT-M Has 3 Stages After patch_embed at Strides [16, 32, 64].
        self._feature_maps: List[torch.Tensor] = []

        def _make_hook(self_ref):
            def _hook(module, input, output):
                self_ref._feature_maps.append(output)
            return _hook

        self._hooks = []
        for stage in self.base_model.stages.children():
            h = stage.register_forward_hook(_make_hook(self))
            self._hooks.append(h)

        # Runtime Probe for Unknown Variants. Requires Hooks to be Attached.
        if self.backbone_depths is None:
            with torch.no_grad():
                dummy = torch.zeros(1, input_depth, input_size, input_size)
                self._feature_maps.clear()
                self.base_model(dummy)
                self.backbone_depths = [f.shape[1] for f in self._feature_maps]
                self._feature_maps.clear()

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} feature channels: {self.backbone_depths}')

        total_channels = sum(self.backbone_depths)

        # Bottleneck Projection: Collapse Concatenated Multi-Scale Features to
        # a Uniform Width Before Detection Heads.
        self.fc_comb = nn.Conv2d(
            total_channels,
            self.combine_outputs_dim,
            kernel_size = self.combine_outputs_kernel,
        )
        head_ch = self.combine_outputs_dim

        # Detection Heads: Identical Layout to MobileNetV4Segmentation.
        self.fc_cls      = nn.Conv2d(head_ch, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(head_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(head_ch, 2, kernel_size=1)

        # Optional Heads.
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(head_ch, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(head_ch, 1, kernel_size=1)


    def freeze_encoder(self):
        """
        Freezes all backbone parameters, preventing them from being updated
        during training. Useful for the initial warm-up phase where only the
        detection heads are trained.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(False)


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(True)


    # Native Input Size the Backbone Was Built For.
    # Inputs Larger Than This Are Automatically Tiled in forward().
    PATCH_SIZE = 512

    def _forward_patch(self, patch: torch.Tensor) -> dict:
        """
        Runs a single patch through the backbone and detection heads.

        Clears the feature capture buffer, triggers a backbone forward pass
        (populating self._feature_maps via hooks), then aligns and concatenates
        multi-scale features before applying detection heads.

        Args:
            patch (torch.Tensor): Input patch of shape [B, C, PATCH_SIZE, PATCH_SIZE].

        Returns:
            dict: Detection outputs keyed by head name.
        """
        # Clear Capture Buffer Before Each Forward Pass.
        self._feature_maps.clear()

        # Trigger Backbone Forward — Hooks Populate self._feature_maps.
        self.base_model(patch)

        # stages_output: [stage_0, stage_1, stage_2].
        stages_output = list(self._feature_maps)

        # Calculate Target Spatial Dimensions Based on Input / pred_scale.
        target_h = patch.shape[2] // self.pred_scale
        target_w = patch.shape[3] // self.pred_scale

        # Upsample All Stage Outputs to pred_scale Resolution.
        x = [
            F.interpolate(stages_output[0], size=(target_h, target_w), mode='bilinear', align_corners=False),
            F.interpolate(stages_output[1], size=(target_h, target_w), mode='bilinear', align_corners=False),
            F.interpolate(stages_output[2], size=(target_h, target_w), mode='bilinear', align_corners=False),
        ]

        # Concatenate, Project, and Apply Detection Heads.
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
        """
        Runs the model on a full-resolution input, tiling if necessary.

        If the input fits within PATCH_SIZE, it is processed directly.
        Otherwise the input is padded to a multiple of PATCH_SIZE, processed
        tile by tile, and the results are stitched back into a single output
        tensor cropped to the original spatial dimensions.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs keyed by head name, spatial shape [B, ch, H/scale, W/scale].
        """
        B, C, H, W = inputs.shape
        ps        = self.PATCH_SIZE
        out_scale = self.pred_scale

        # Fast Path: Input Fits in One Patch — No Tiling Overhead.
        if H <= ps and W <= ps:
            return self._forward_patch(inputs)

        # Tiled Path: Pad Input to a Multiple of PATCH_SIZE.
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(inputs, (0, pad_w, 0, pad_h))
        else:
            x_pad = inputs

        pH, pW     = x_pad.shape[2], x_pad.shape[3]
        rows, cols = pH // ps, pW // ps
        ops        = ps // out_scale   # Output Patch Size in Pixels.

        # Pre-Allocate Output Grids on the Same Device as Inputs.
        def _alloc(ch):
            return torch.zeros(B, ch, rows * ops, cols * ops,
                               device=inputs.device, dtype=inputs.dtype)

        # Run a Probe Patch to Discover Output Channel Counts.
        _probe = self._forward_patch(x_pad[:, :, :ps, :ps])
        grids  = {k: _alloc(v.shape[1]) for k, v in _probe.items()}

        # Fill in the Probe Result for the First (Top-Left) Patch.
        for k, v in _probe.items():
            grids[k][:, :, :ops, :ops] = v

        # Process All Remaining Patches in Raster Order.
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue   # Already Filled from Probe.
                patch = x_pad[:, :, r*ps:(r+1)*ps, c*ps:(c+1)*ps]
                pred  = self._forward_patch(patch)
                for k, v in pred.items():
                    grids[k][:, :, r*ops:(r+1)*ops, c*ops:(c+1)*ops] = v

        # Crop Back to the Original (Unpadded) Output Dimensions.
        orig_oh = H // out_scale
        orig_ow = W // out_scale
        return {k: v[:, :, :orig_oh, :orig_ow] for k, v in grids.items()}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   EFFICIENTVIT-B2 SEGMENTATION (MIT)
# ------------------------------------------------------------------------------
class EfficientViTB2Segmentation(nn.Module):
    """
    EfficientViT-B2 (MIT) backbone segmentation model.

    Uses the MIT EfficientViT-B2 architecture via timm's features_only API,
    which exposes four feature stages directly without requiring forward hooks.
    This is distinct from the MSRA EfficientViT-M family used in
    EfficientViTSegmentation.

    Feature channel widths are read directly from the model's feature_info
    metadata rather than from a lookup table or runtime probe.

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name      : timm EfficientViT-B2 model string.
            input_frames         : Number of input channels.
            pred_scale           : Output stride (4 or 8).
            combine_outputs_dim  : Bottleneck channel width (default 256).
            combine_outputs_kernel: Bottleneck kernel size (default 1).
            output_binary_mask   : (Optional) If True, adds a binary mask head.
            output_above_horizon : (Optional) If True, adds an above-horizon head.
        pretrained (bool): If True, loads ImageNet-pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name             = cfg['base_model_name']
        input_depth                 = cfg['input_frames']
        self.combine_outputs_dim    = cfg.get('combine_outputs_dim', 256)
        self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
        self.output_binary_mask     = cfg.get('output_binary_mask', False)
        self.output_above_horizon   = cfg.get('output_above_horizon', False)
        self.pred_scale             = cfg['pred_scale']

        # Build EfficientViT-B2 Backbone via timm features_only API.
        self.base_model = timm.create_model(
            base_model_name,
            pretrained    = pretrained,
            features_only = True,
            in_chans      = input_depth,
        )

        # Probe Feature Channel Widths Directly from the Model's Feature Info.
        self.backbone_depths = [fi['num_chs'] for fi in self.base_model.feature_info]

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} feature channels: {self.backbone_depths}')

        # Total Channels After Concatenating All Stage Outputs (720 for B2).
        total_channels = sum(self.backbone_depths)

        # Bottleneck Projection: Collapse 720 Channels to combine_outputs_dim.
        self.fc_comb = nn.Conv2d(
            total_channels,
            self.combine_outputs_dim,
            kernel_size = self.combine_outputs_kernel,
        )
        head_ch = self.combine_outputs_dim

        # Detection Heads: Identical Layout to MobileNetV4Segmentation.
        self.fc_cls      = nn.Conv2d(head_ch, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(head_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(head_ch, 2, kernel_size=1)

        # Optional Heads.
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(head_ch, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(head_ch, 1, kernel_size=1)


    def freeze_encoder(self):
        """
        Freezes all backbone parameters, preventing them from being updated
        during training. Useful for the initial warm-up phase where only the
        detection heads are trained.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(False)


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(True)


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # Set the Output Stages for the Model. 
        stages_output = self.base_model(inputs)

        # Align All Stage Outputs to pred_scale Resolution Before Concatenation.
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

        # Concatenate, Project, and Apply Detection Heads.
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
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       RESNET-34 SEGMENTATION
# ------------------------------------------------------------------------------
class ResNet34Segmentation(nn.Module):
    """
    ResNet-34 backbone segmentation model.

    Builds a ResNet-34 backbone via timm (features_only=True), which exposes
    five feature stages at strides [2, 4, 8, 16, 32] with channel widths
    [64, 64, 128, 256, 512]. All five stages are fused at the configured
    pred_scale resolution before a bottleneck projection and detection heads.

    Args:
        cfg (dict): Model configuration. Expected keys:
            base_model_name      : timm ResNet-34 model string.
            input_frames         : Number of input channels.
            pred_scale           : Output stride (4 or 8).
            combine_outputs_dim  : Bottleneck channel width (default 256).
            combine_outputs_kernel: Bottleneck kernel size (default 1).
            output_binary_mask   : (Optional) If True, adds a binary mask head.
            output_above_horizon : (Optional) If True, adds an above-horizon head.
        pretrained (bool): If True, loads ImageNet-pretrained backbone weights.
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__()

        # Extract Backbone & Architecture Config.
        base_model_name             = cfg['base_model_name']
        input_depth                 = cfg['input_frames']
        self.combine_outputs_dim    = cfg.get('combine_outputs_dim', 256)
        self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
        self.output_binary_mask     = cfg.get('output_binary_mask', False)
        self.output_above_horizon   = cfg.get('output_above_horizon', False)
        self.pred_scale             = cfg['pred_scale']

        # Build ResNet-34 Backbone via timm, Extracting All 5 Feature Stages.
        self.base_model = timm.create_model(
            base_model_name,
            pretrained    = pretrained,
            features_only = True,
            in_chans      = input_depth,
        )

        # Probe Feature Channel Widths from the Model's Own Feature Info.
        # ResNet-34: [64, 64, 128, 256, 512] at Strides [2, 4, 8, 16, 32].
        self.backbone_depths = [fi['num_chs'] for fi in self.base_model.feature_info]

        # Verbose Logging During Model Initialisation if DEBUG_MODEL_INIT is Set.
        if os.getenv("DEBUG_MODEL_INIT"):
            print(f'{base_model_name} feature channels: {self.backbone_depths}')

        # Total Channels After Concatenating All 5 Stage Outputs (1024).
        total_channels = sum(self.backbone_depths)

        # Bottleneck Projection to a Uniform Width Before Detection Heads.
        self.fc_comb = nn.Conv2d(
            total_channels,
            self.combine_outputs_dim,
            kernel_size = self.combine_outputs_kernel,
        )
        head_ch = self.combine_outputs_dim

        # Detection Heads: Identical Layout to MobileNetV4Segmentation.
        self.fc_cls      = nn.Conv2d(head_ch, config.NB_CLASSES, kernel_size=1)
        self.fc_size     = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_offset   = nn.Conv2d(head_ch, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(head_ch, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(head_ch, 2, kernel_size=1)

        # Optional Heads.
        if self.output_binary_mask:
            self.fc_mask    = nn.Conv2d(head_ch, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(head_ch, 1, kernel_size=1)


    def freeze_encoder(self):
        """
        Freezes all backbone parameters, preventing them from being updated
        during training. Useful for the initial warm-up phase where only the
        detection heads are trained.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(False)


    def unfreeze_encoder(self):
        """
        Unfreezes all backbone parameters, allowing end-to-end fine-tuning
        of both the backbone and detection heads.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(True)


    def forward(self, inputs):
        """
        Runs a forward pass through the backbone, fuses multi-scale feature
        maps at the configured pred_scale resolution, and applies all
        detection heads.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            dict: Detection outputs with keys:
                cls           : Classification logits.
                size          : Object size predictions (w, h).
                offset        : Sub-pixel centre offset predictions.
                distance      : Depth/distance predictions.
                tracking      : Inter-frame tracking offset predictions.
                mask          : Binary foreground mask (if output_binary_mask).
                above_horizon : Above-horizon indicator (if output_above_horizon).
        """
        # features_only=True Returns All 5 Stages Directly.
        # f0: stride-2   f1: stride-4   f2: stride-8
        # f3: stride-16  f4: stride-32
        f0, f1, f2, f3, f4 = self.base_model(inputs)

        # Align All Stage Outputs to pred_scale Resolution Before Concatenation.
        if self.pred_scale == 8:
            # Reference is stride-8 (f2) — Target Spatial = H/8 x W/8.
            x = [
                F.avg_pool2d(f0, 4),                                    # stride 2  -> 8
                F.avg_pool2d(f1, 2),                                    # stride 4  -> 8
                f2,                                                     # stride 8  (reference)
                F.interpolate(f3, scale_factor=2, mode='nearest'),      # stride 16 -> 8
                F.interpolate(f4, scale_factor=4, mode='nearest'),      # stride 32 -> 8
            ]
        elif self.pred_scale == 4:
            # Reference is stride-4 (f1) — Target Spatial = H/4 x W/4.
            x = [
                F.avg_pool2d(f0, 2),                                    # stride 2  -> 4
                f1,                                                     # stride 4  (reference)
                F.interpolate(f2, scale_factor=2, mode='nearest'),      # stride 8  -> 4
                F.interpolate(f3, scale_factor=4, mode='nearest'),      # stride 16 -> 4
                F.interpolate(f4, scale_factor=8, mode='nearest'),      # stride 32 -> 4
            ]
        else:
            raise RuntimeError(f'Invalid pred_scale: {self.pred_scale}')

        # Concatenate, Project, and Apply Detection Heads.
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
# ------------------------------------------------------------------------------
