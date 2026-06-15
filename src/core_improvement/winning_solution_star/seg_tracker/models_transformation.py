# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: models_transformation.py %
# Date Created:            %date_created: Tue May 19 19:51:23 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Motion estimation models used by the SegTracker inference pipeline for
# frame-to-frame affine transform prediction. Includes the original two-frame
# ResNet baseline, a Temporal Shift Module (TSM) variant, and an optimised
# cached-inference implementation that reuses backbone activations to reduce
# repeated computation and improve real-time inference throughput.
# 
# ------------------------------------------------------------------------------
# -                          L I C E N S I N G                                 -
# ------------------------------------------------------------------------------
# This software and its contents are the intellectual property of Lockheed 
# Martin Australia and are protected under applicable copyright and intellectual
# property laws.
#
# All rights reserved. No part of this software may be reproduced, distributed, 
# modified, or transmitted in any form or by any means without the prior written
# permission of Lockheed Martin Australia.
#
# Unauthorised use, copying, or distribution of this software is strictly 
# prohibited.
# ------------------------------------------------------------------------------

import timm
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ------------------------------------------------------------------------------
#               TrEstimator (Original Two-Channel ResNet Baseline)
# ------------------------------------------------------------------------------
class TrEstimator(nn.Module):
    """
    Original motion estimation baseline using a two-channel ResNet backbone.

    The previous and current grayscale frames are stacked together along the
    channel dimension and processed jointly by the CNN. The network predicts:

        1. A confidence-weighted heatmap indicating reliable motion regions.
        2. Per-grid-cell spatial offsets between the two frames.
    """
    def __init__(self, cfg, pretrained=True):
        # Initialise the Parent nn.Module.
        super().__init__()
        
        # Backbone Architecture Name from Config.
        base_model_name = cfg['base_model_name']
        input_depth = 2

        # Scalar Controlling Sharpness/Peakedness of the Confidence Heatmap.
        self.weight_scale = cfg['weight_scale']

        # Create the CNN Backbone Using Timm.
        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            in_chans=input_depth,
                                            pretrained=pretrained)

        # Number of Channels in the Final Backbone Feature Map.
        out_ch = self.base_model.feature_info.channels()[-1]

        # 1x1 Convolution Predicting a Single-Channel Confidence Heatmap.
        self.conv_heatmap = nn.Conv2d(out_ch, 1, kernel_size=1, bias=True)

        # 1x1 Convolution Predicting 2D Motion Offsets.
        self.conv_offset = nn.Conv2d(out_ch, 2, kernel_size=1, bias=True)


    def freeze_encoder(self):
        """
        Freeze backbone weights during training. Useful when fine-tuning only 
        the prediction heads.
        """
        self.base_model.freeze_encoder()


    def unfreeze_encoder(self):
        """
        Unfreeze backbone weights so the full network can train.
        """
        self.base_model.unfreeze_encoder()


    def forward(self, prev_frame, cur_frame):
        """
        Run motion estimation on a pair of consecutive grayscale frames.

        Args:
            prev_frame: Previous frame tensor with shape [B, H, W].
            cur_frame: Current frame tensor with shape [B, H, W].

        Returns:
            heatmap: Normalised confidence map with shape [B, 1, H', W'].
            offsets: Predicted motion offsets with shape [B, 2, H', W'].
        """
        # Stack Frames Along the Channel Dimension
        inputs = torch.stack([prev_frame, cur_frame], dim=1)

        # Run the Stacked Frame Pair through the CNN Backbone.
        x = self.base_model(inputs)
        x = x[-1]
        x = x[:, :, 2:-2, 2:-2]
        x_hm = self.conv_heatmap(x)

        # Convert Logits into Positive Confidence Weights.
        m = torch.exp(self.weight_scale * torch.sigmoid(x_hm))

        # Normalise into a Probability-Style Spatial Heatmap Whose Values Sum
        # to 1 Across All Spatial Locations.
        heatmap = m / torch.sum(m, dim=(2, 3), keepdim=True)

        # Predict Per-Cell Motion Offsets.
        offsets = self.conv_offset(x)

        # Return Both Outputs.
        return heatmap, offsets



# ------------------------------------------------------------------------------
#                      TSMConv2d (Zero-Extra-Compute TSM)
# ------------------------------------------------------------------------------
class TSMConv2d(nn.Module):
    """
    Temporal Shift Module (TSM) wrapper around a standard Conv2d layer.

    A small subset of channels are shifted between consecutive frames before
    convolution, allowing temporal information exchange at effectively zero
    additional FLOP cost.

    This implementation is fully traceable by torch.compile and replaces the
    older runtime monkey-patch implementation used in the original baseline.
    """
    def __init__(self, conv: nn.Conv2d, duration: int = 2) -> None:
        # Initialise the Parent nn.Module.
        super().__init__()

        # Wrapped Spatial Convolution Layer.
        self.conv = conv

        # Number of Temporal Frames Per Sample.
        self.duration = duration


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal channel shifting before convolution.

        Args:
            x: Input tensor with shape [B*T, C, H, W].

        Returns:
            torch.Tensor: Output tensor after temporal shifting and convolution.
        """
        # Extract Tensor Dimensions.
        B_T, C, H, W = x.shape
        T = self.duration

        # Only Apply Temporal Shifting When the Batch Contains Multiple Frames Per Sample.
        # Cached Single-Frame Inference Paths Bypass the Shift Entirely.
        if B_T % T == 0 and B_T // T > 0 and T > 1:

            # Recover the Original Batch Dimension.
            B = B_T // T

            # Reshape into Explicit [Batch, Time, Channel, Height, Width].
            x_bt = x.view(B, T, C, H, W)

            # Shift Only a Small Fraction of Channels.
            shift = C // 8

            # Roll the Selected Channels Across the Temporal Dimension.
            shifted = x_bt[:, :, :shift].roll(1, dims=1)

            # Clone the Tensor Before In-Place Modification.
            out = x.clone()
            out_bt = out.view(B, T, C, H, W)

            # Insert the Temporally Shifted Channels.
            out_bt[:, :, :shift] = shifted

            # Flatten Back into [B*T, C, H, W] Before Convolution.
            return self.conv(out_bt.view(B_T, C, H, W))

        # Single-Frame Pass-Through Path.
        return self.conv(x)



# ------------------------------------------------------------------------------
#               TrEstimatorTsm (Original Paired-Frame Model)
# ------------------------------------------------------------------------------
class TrEstimatorTsm(nn.Module):
    """
    Transformation estimator using a ResNet + TSM backbone.

    Processes prev_frame and cur_frame together as a batch of T=2 so the
    TSMConv2d layers can exchange channel information between time steps.

    This class is kept API-compatible with the original.  For inference use
    TrEstimatorTsmCached (below) which halves backbone compute.
    """
    def __init__(self, cfg, pretrained=True):
        # Initialise the Parent nn.Module.
        super().__init__()

        # Backbone Architecture Name from Config.
        base_model_name = cfg['base_model_name']

        # Single-Channel Grayscale Input.
        input_depth = 1

        # Scalar Controlling Sharpness of the Confidence Heatmap.
        self.weight_scale = cfg['weight_scale']

        # Create the CNN Backbone Using timm.
        self.base_model = timm.create_model(
            base_model_name,
            features_only=True,
            in_chans=input_depth,
            pretrained=pretrained
        )

        # Number of Temporal Frames Per Sample.
        duration = 2

        # Replace ResNet conv1 Layers with TSMConv2d Wrappers.
        if 'resnet' in base_model_name:
            for layer in [
                self.base_model.layer1,
                self.base_model.layer2,
                self.base_model.layer3,
                self.base_model.layer4,
            ]:
                for block in layer:
                    block.conv1 = TSMConv2d(block.conv1, duration=duration)

        # Number of Channels in the Final Backbone Feature Map.
        out_ch = self.base_model.feature_info.channels()[-1]

        # 1x1 Convolution Predicting a Confidence Heatmap.
        self.conv_heatmap = nn.Conv2d(out_ch * 2, 1, kernel_size=1, bias=True)

        # 1x1 Convolution Predicting 2D Motion Offsets.
        self.conv_offset = nn.Conv2d(out_ch * 2, 2, kernel_size=1, bias=True)


    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load checkpoints saved before TSMConv2d was introduced.

        Older checkpoints store weights using the original Conv2d naming
        convention. This function transparently remaps those keys so legacy
        checkpoints remain compatible without retraining.
        """
        # Store Remapped Parameter Keys.
        remapped = {}

        # Remap the Keys.
        for k, v in state_dict.items():
            # Remap: conv1.weight -> conv1.conv.weight
            if (
                "conv1.weight" in k
                and "conv1.conv.weight" not in k
                and any(f"layer{i}" in k for i in range(1, 5))
            ):
                k = k.replace("conv1.weight", "conv1.conv.weight")

            # Save Remapped Results.
            remapped[k] = v

        # Load the Remapped State Dictionary.
        return super().load_state_dict(remapped, strict=strict)


    def forward(self, prev_frame, cur_frame):
        """
        Run motion estimation on a pair of consecutive grayscale frames.

        Args:
            prev_frame: Previous frame tensor with shape [B, H, W].
            cur_frame: Current frame tensor with shape [B, H, W].

        Returns:
            heatmap: Normalised confidence map with shape [B, 1, H', W'].
            offsets: Predicted motion offsets with shape [B, 2, H', W'].
        """
        # Stack Frames into a Temporal Batch: [B, T, C, H, W]
        inputs = torch.stack( [prev_frame, cur_frame], dim=1)[:, :, None, :, :]

        # Extract Tensor Dimensions.
        B, T, C, H, W = inputs.shape

        # Flatten into Standard CNN Batch Layout: [B*T, C, H, W]
        x = inputs.view(B * T, C, H, W)

        # Run the Temporal Batch Through the CNN Backbone.
        x = self.base_model(x)

        # Use the Deepest Backbone Feature Map.
        x = x[-1]

        # Concatenate Previous and Current Frame Features Along the Channel Dimension.
        x = x.view(B, -1, x.shape[2], x.shape[3])

        # Remove Border Cells Affected by Padding Artefacts.
        x = x[:, :, 2:-2, 2:-2]

        # Predict Raw Heatmap Logits.
        x_hm = self.conv_heatmap(x)

        # Convert Logits into Positive Confidence Weights.
        m = torch.exp(self.weight_scale * torch.sigmoid(x_hm))

        # Normalise into a Spatial Probability Distribution.
        heatmap = m / torch.sum(m, dim=(2, 3), keepdim=True)

        # Predict Per-Cell Motion Offsets.
        offsets = self.conv_offset(x)

        # Return Both Outputs.
        return heatmap, offsets



# ------------------------------------------------------------------------------
#       TrEstimatorTsmCached (2× faster Inference Via Activation Caching)
# ------------------------------------------------------------------------------
class TrEstimatorTsmCached(nn.Module):
    """
    Optimised inference-only motion estimation model using a ResNet + TSM
    backbone with feature caching.

    Instead of recomputing backbone features for both the previous and current
    frame every timestep, this implementation reuses the previous frame's
    cached activations. Since the current frame at timestep N becomes the
    previous frame at timestep N+1, this removes redundant backbone execution
    and approximately halves inference compute cost.

    The model preserves the exact same output structure and checkpoint
    compatibility as TrEstimatorTsm while improving real-time throughput.
    """
    def __init__(self, cfg, pretrained: bool = True) -> None:
        # Initialise the Parent nn.Module.
        super().__init__()

        # Backbone Architecture Name from Config.
        base_model_name = cfg['base_model_name']

        # Scalar Controlling Sharpness/Peakedness of the Confidence Heatmap.
        self.weight_scale = cfg['weight_scale']

        # Create the Single-Channel CNN Backbone.
        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            in_chans=1,
                                            pretrained=pretrained)

        # Replace Standard ResNet conv1 Layers with TSM Wrappers so Temporal
        # Information can Flow Between Consecutive Frames.
        if 'resnet' in base_model_name:
            for layer in [
                self.base_model.layer1,
                self.base_model.layer2,
                self.base_model.layer3,
                self.base_model.layer4,
            ]:
                for block in layer:
                    block.conv1 = TSMConv2d(block.conv1, duration=2)

        # Number of Channels in the Final Backbone Feature Map.
        out_ch = self.base_model.feature_info.channels()[-1]

        # 1x1 Convolution Predicting a Single-Channel Confidence Heatmap.
        self.conv_heatmap = nn.Conv2d(out_ch * 2, 1, kernel_size=1, bias=True)

        # 1x1 Convolution Predicting 2D Motion Offsets.
        self.conv_offset = nn.Conv2d(out_ch * 2, 2, kernel_size=1, bias=True)

        # Cache Storing Previous-Frame Backbone Activations Between Timesteps.
        self._prev_feat: Optional[torch.Tensor] = None

        # Compile the Backbone Encoding Path Separately.
        self._compiled_encode = torch.compile(self._encode_impl, mode="default")

        # Compile the Prediction Head Separately.
        self._compiled_run_head = torch.compile(self._run_head_impl, mode="max-autotune")


    def _encode_impl(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Encode a Single Grayscale Frame into Backbone Feature Maps.

        Args:
            frame: Input tensor with shape [B, 1, H, W].

        Returns:
            torch.Tensor: Cropped backbone feature map tensor.
        """
        # Run the Frame Through the CNN Backbone.
        feats = self.base_model(frame)

        # Extract the Final Backbone Feature Level.
        f = feats[-1]

        # Remove Border Regions Affected by Padding Artefacts.
        return f[:, :, 2:-2, 2:-2]


    def _run_head_impl(self, prev_feat: torch.Tensor, cur_feat: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Motion Heatmaps and Spatial Offsets from Backbone Features.

        Args:
            prev_feat: Previous-frame feature map tensor.
            cur_feat: Current-frame feature map tensor.

        Returns:
            heatmap: Normalised confidence heatmap tensor.
            offsets: Predicted spatial motion offsets tensor.
        """
        # Concatenate Previous and Current Features Along the Channel Dimension.
        x = torch.cat([prev_feat, cur_feat], dim=1)

        # Predict Raw Heatmap Logits.
        x_hm = self.conv_heatmap(x)

        # Convert Logits into Positive Confidence Weights.
        m = torch.exp(self.weight_scale * torch.sigmoid(x_hm))

        # Normalise into a Probability-Style Spatial Heatmap Whose Values Sum
        # to 1 Across All Spatial Locations.
        heatmap = m / torch.sum(m, dim=(2, 3), keepdim=True)

        # Predict Per-Cell Motion Offsets.
        offsets = self.conv_offset(x)

        # Return Both Outputs.
        return heatmap, offsets


    def reset_cache(self) -> None:
        """
        Clear Cached Backbone Features Between Video Sequences.
        """
        self._prev_feat = None


    @property
    def flops_model(self) -> "TrEstimatorTsm":
        """
        Create a Plain Uncompiled TrEstimatorTsm Instance for FLOPs Analysis.

        FLOPs measurement tools based on torch.jit.trace are incompatible with
        torch.compile wrappers. This helper rebuilds an equivalent vanilla
        nn.Module and copies all weights into it.

        Returns:
            TrEstimatorTsm: Equivalent uncompiled model suitable for complexity analysis.
        """
        # Build an Equivalent Plain TrEstimatorTsm Model.
        cfg = dict(
            base_model_name=self.base_model.default_cfg["architecture"],
            weight_scale=self.weight_scale
        )

        # Initalise the Tracking Estimator.
        plain = TrEstimatorTsm(cfg=cfg, pretrained=False)

        # Reverse-Remap TSMConv2d Weight Keys Back into the Original Format.
        remapped = {}

        for k, v in self.state_dict().items():
            if "conv1.conv.weight" in k:
                k = k.replace("conv1.conv.weight", "conv1.weight")

            remapped[k] = v.cpu()

        # Load Weights into the Plain Model.
        plain.load_state_dict(remapped, strict=True)

        # Switch to Evaluation Mode.
        plain.eval()

        # Return the Plain Model.
        return plain


    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load Checkpoints Saved Before TSMConv2d Wrapping Was Introduced.

        Older checkpoints store standard conv1 weights directly. This remaps
        those keys into the nested TSMConv2d structure automatically so legacy
        checkpoints remain fully compatible.

        Args:
            state_dict: Checkpoint parameter dictionary.
            strict: Whether strict key matching should be enforced.

        Returns:
            Result from nn.Module.load_state_dict().
        """
        # Store Remapped Parameter Keys.
        remapped = {}

        # Remap the Keys.
        for k, v in state_dict.items():
            # Remap: conv1.weight -> conv1.conv.weight
            if (
                "conv1.weight" in k
                and "conv1.conv.weight" not in k
                and any(f"layer{i}" in k for i in range(1, 5))
            ):
                k = k.replace("conv1.weight", "conv1.conv.weight")

            # Save Remapped Results.
            remapped[k] = v

        # Load the Remapped State Dictionary.
        return super().load_state_dict(remapped, strict=strict)


    def forward(self, prev_frame: torch.Tensor, cur_frame:  torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Motion Estimation Using Cached Backbone Activations.

        Args:
            prev_frame: Previous grayscale frame tensor with shape [B, H, W].
            cur_frame: Current grayscale frame tensor with shape [B, H, W].

        Returns:
            heatmap: Normalised confidence map with shape [B, 1, H', W'].
            offsets: Predicted motion offsets with shape [B, 2, H', W'].
        """
        # Encode the Current Frame into Backbone Features.
        cur_feat = self._compiled_encode(cur_frame[:, None, :, :])

        # If no Cached Features Exist Yet, Encode the Previous Frame Normally.
        if self._prev_feat is None:
            prev_feat = self._compiled_encode(prev_frame[:, None, :, :])

        # Otherwise Reuse the Cached Previous-Frame Activations.
        else:
            prev_feat = self._prev_feat

        # Store Current Features for Reuse During the Next Timestep.
        self._prev_feat = cur_feat.detach()

        # Run the Prediction Head Using Previous + Current Features.
        return self._compiled_run_head(prev_feat, cur_feat)
