# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: seg_tracker.py %
# Date Created:            %date_created: Fri May 08 22:31:43 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 2 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# This module implements the core detection and tracking pipeline used in the
# winning-solution ensemble. It includes the SegDetector class, which loads the
# four ensemble segmentation models together with the ResNet34-TSM frame-alignment
# model and provides transformation estimation and object detection utilities.
# The module also provides SegTracker for lightweight single-frame evaluation
# without temporal tracking, and SegTrackerFromOffset, which integrates
# SimpleOffsetTracker to maintain consistent track IDs across video frames.
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

import logging
import warnings
import torch
import numpy as np
import cv2

from . import models_segmentation
from . import models_transformation
from . import offset_grid_to_transform
from . import predict_ensemble

from .prediction_structure import *
from . import tracking


# ------------------------------------------------------------------------------
#                            SEG DETECTOR CLASS
# ------------------------------------------------------------------------------
class SegDetector:
    def __init__(self):
        # Suppress torch.compile / inductor verbosity:
        #   - TypedStorage deprecation warning from select_algorithm.py.
        #   - AUTOTUNE addmm benchmarking tables.
        #   - Autotune Choices Stats JSON lines.
        logging.getLogger("torch._inductor").setLevel(logging.ERROR)
        logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated",
            category=UserWarning,
        )

        # Directory Containing All Ensemble Model Weights.
        models_dir = 'configs/models/core_improvement/winning_solution_star'

        # Set the Device to GPU if Available.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the Frame-Alignment Model Architecture.
        self._model_transform_raw = models_transformation.TrEstimatorTsmCached(
            cfg=dict(base_model_name="resnet34",weight_scale=3,),
            pretrained=False,
        )

        # Load Pretrained Frame-Alignment Model Weights.
        self._model_transform_raw.load_state_dict(
            torch.load(
                f'{models_dir}/030_tr_tsn_rn34_w3_crop_borders_255_0.pth',
                map_location=self.device,
                weights_only=False,
            )["model_state_dict"]
        )

        # Move Frame-Alignment Model to Inference Device.
        self._model_transform_raw = self._model_transform_raw.to(self.device)

        # Set Frame-Alignment Model to Evaluation Mode.
        self._model_transform_raw.eval()

        # Prevent Adding Cache Branch into Compiled Graph & Breaking CUDA Graph Capture.
        self.model_transform = self._model_transform_raw

        # Create HRNet-W32 Segmentation Model Architecture.
        self.model_seg1 = models_segmentation.HRNetSegmentation(
            cfg=dict(
                base_model_name="hrnet_w32",
                input_frames=2,
                feature_location='',
                combine_outputs_dim=512,
                combine_outputs_kernel=1,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8,
            ),
            pretrained=False,
        )

        # Load Pretrained HRNet-W32 Weights.
        self.model_seg1.load_state_dict(
            torch.load(
                f'{models_dir}/120_hrnet32_all.pt',
                map_location=self.device,
                weights_only=False,
            )['model_state_dict'],
            strict=False,
        )

        # Move Model to Inference Device and Set to Evaluation Mode.
        self.model_seg1 = self.model_seg1.to(self.device)
        self.model_seg1.eval()

        # Keep Raw Reference for FLOPs Measurement Before Compiling.
        self._model_seg1_raw = self.model_seg1

        # Compile for Faster Repeated Inference.
        self.model_seg1 = torch.compile(self.model_seg1, mode="max-autotune")

        # Create EfficientDet-D2 Segmentation Model Architecture.
        self.model_seg2 = models_segmentation.EffDetSegmentation8x(
            cfg=dict(
                base_model_name="tf_efficientdet_d2",
                custom_backbone="gernet_m",
                input_frames=2,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8,
            ),
            pretrained=False,
        )

        # Load Pretrained EfficientDet-D2 Weights.
        self.model_seg2.load_state_dict(
            torch.load(
                f'{models_dir}/120_gernet_m_b2_fpm.pt',
                map_location=self.device,
                weights_only=False,
            )['model_state_dict'],
            strict=False,
        )

        # Move Model to Inference Device and Set to Evaluation Mode.
        self.model_seg2 = self.model_seg2.to(self.device)
        self.model_seg2.eval()

        # Keep Raw Reference for FLOPs Measurement Before Compiling.
        self._model_seg2_raw = self.model_seg2

        # Compile for Faster Repeated Inference.
        self.model_seg2 = torch.compile(self.model_seg2, mode="max-autotune")

        # Create DLA-60 Segmentation Model Architecture.
        self.model_dla = models_segmentation.DLA8xSeparateHeads(
            cfg=dict(
                base_model_name="dla60_res2next",
                input_frames=2,
                feature_location='',
                combine_outputs_dim=256,
                combine_outputs_kernel=1,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8,
            ),
            pretrained=False,
        )

        # Load Pretrained DLA-60 Weights.
        self.model_dla.load_state_dict(
            torch.load(
                f'{models_dir}/120_dla60_256_sgd_all.pt',
                map_location=self.device,
                weights_only=False,
            )['model_state_dict'],
            strict=False,
        )

        # Move Model to Inference Device and Set to Evaluation Mode.
        self.model_dla = self.model_dla.to(self.device)
        self.model_dla.eval()

        # Keep Raw Reference for FLOPs Measurement Before Compiling.
        self._model_dla_raw = self.model_dla

        # Compile for Faster Repeated Inference.
        self.model_dla = torch.compile(self.model_dla, mode="max-autotune")

        # Create HRNet-W48 Segmentation Model Architecture.
        self.model_seg3 = models_segmentation.HRNetSegmentation(
            cfg=dict(
                base_model_name="hrnet_w48",
                input_frames=2,
                feature_location='',
                combine_outputs_dim=512,
                combine_outputs_kernel=1,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8,
            ),
            pretrained=False,
        )

        # Load Pretrained HRNet-W48 Weights.
        self.model_seg3.load_state_dict(
            torch.load(
                f'{models_dir}/130_hrnet48_all.pt',
                map_location=self.device,
                weights_only=False,
            )['model_state_dict'],
            strict=False,
        )

        # Move Model to Inference Device and Set to Evaluation Mode.
        self.model_seg3 = self.model_seg3.to(self.device)
        self.model_seg3.eval()

        # Keep Raw Reference for FLOPs Measurement Before Compiling.
        self._model_seg3_raw = self.model_seg3

        # Compile for Faster Repeated Inference.
        self.model_seg3 = torch.compile(self.model_seg3, mode="max-autotune")

        # Frame Counter (Used Externally for Diagnostics).
        self.frames = 0


    def reset_transform_cache(self) -> None:
        """
        Discard the Cached Prev-Frame Features from the Alignment Model.

        MUST be Called at the Start of Every New Flight / Video Sequence.
        Without This, the First Frame of a New Flight Would Use Stale Features
        from the Last Frame of the Previous Flight, Producing a Corrupt
        Transformation Estimate for That Frame.
        """
        self._model_transform_raw.reset_cache()


    def estimate_transformation(self, prev_img, img):
        """
        Estimate (dx, dy, angle) Using the Central 1024x1024 Crop.

        Args:
            prev_img: Previous grayscale frame as a 2D NumPy array.
            img:      Current grayscale frame as a 2D NumPy array.

        Returns:
            Tuple of (dx, dy, angle) representing the estimated motion.
        """
        # Define the Fixed Crop Size Used for Motion Estimation.
        crop_w = 1024
        crop_h = 1024

        # Get Image Dimensions.
        h, w = img.shape

        # Centre the Crop Vertically and Horizontally.
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2

        # Extract Matching Crops from the Current & Previous Frames.
        img_crop      = img     [y0:y0 + crop_h, x0:x0 + crop_w]
        img_crop_prev = prev_img[y0:y0 + crop_h, x0:x0 + crop_w]

        # Convert Cropped Images into PyTorch Tensors & Normalise to [0, 1].
        img_crop_t      = torch.from_numpy(img_crop      / 255.0).float().to(self.device)
        img_crop_prev_t = torch.from_numpy(img_crop_prev / 255.0).float().to(self.device)

        # Run Inference Without Gradient Tracking for Efficiency.
        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                heatmap, offsets = self.model_transform(
                    img_crop_prev_t[None, :], img_crop_t[None, :]
                )

        # Move Model Outputs Back to CPU NumPy Arrays.
        heatmap = heatmap.detach().float().cpu().numpy()
        offsets = offsets.detach().float().cpu().numpy()

        # Create a Regular Grid of Points in the Previous Frame Crop.
        prev_points = np.zeros((2, 32, 32), dtype=np.float32)
        prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
        prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]
        prev_points = prev_points[..., 2:-2, 2:-2]

        # Compute Current Frame Points from Predicted Offsets.
        center_offset = np.array([512.0, 512.0])[:, None]
        cur_points    = prev_points + offsets[0]

        # Estimate (dx, dy, angle) Using Weighted Point Correspondences.
        dx, dy, angle, err = offset_grid_to_transform.offset_grid_to_transform_params(
            prev_frame_points = prev_points.reshape(2, -1) - center_offset,
            cur_frame_points  = cur_points.reshape(2, -1)  - center_offset,
            points_weight     = heatmap[0].reshape(-1) ** 2,
        )

        # Return the Estimated Motion Parameters.
        return dx, dy, angle


    def estimate_transformation_full(self, prev_img, img):
        """
        Estimate the Full Affine Transform Using the 2048x1280 Lower Crop.

        Args:
            prev_img: Previous grayscale frame as a 2D NumPy array.
            img:      Current grayscale frame as a 2D NumPy array.

        Returns:
            np.ndarray: Estimated affine transformation matrix.
        """
        # Define the Fixed Crop Size Used for Full Affine Estimation.
        crop_w = 1024 + 1024
        crop_h = 1024 + 256

        # Compute the Number of Grid Points in Each Dimension.
        crop_w_points = crop_w // 32
        crop_h_points = crop_h // 32

        # Get Image Dimensions.
        h, w = img.shape

        # Fixed Vertical Crop Start (Lower Half of Frame).
        y0 = 512

        # Centre the Crop Horizontally.
        x0 = (w - crop_w) // 2

        # Extract Matching Crops from the Current & Previous Frames.
        img_crop      = img     [y0:y0 + crop_h, x0:x0 + crop_w]
        img_crop_prev = prev_img[y0:y0 + crop_h, x0:x0 + crop_w]

        # Convert Cropped Images into PyTorch Tensors & Normalise to [0, 1].
        img_crop_t      = torch.from_numpy(img_crop      / 255.0).float().to(self.device)
        img_crop_prev_t = torch.from_numpy(img_crop_prev / 255.0).float().to(self.device)

        # Run Inference Without Gradient Tracking for Efficiency.
        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                heatmap, offsets = self.model_transform(
                    img_crop_prev_t[None, :], img_crop_t[None, :]
                )

        # Move Model Outputs Back to CPU NumPy Arrays.
        heatmap = heatmap.detach().float().cpu().numpy()
        offsets = offsets.detach().float().cpu().numpy()

        # Create a Regular Grid of Points in the Previous Frame Crop.
        prev_points = np.zeros((2, crop_h_points, crop_w_points), dtype=np.float32)
        prev_points[0, :, :] = np.arange(16, crop_w, 32)[None, :]
        prev_points[1, :, :] = np.arange(16, crop_h, 32)[:, None]
        prev_points = prev_points[..., 2:-2, 2:-2]

        # Convert Crop-Local Coordinates Back into Full-Image Coordinates.
        points_offset = np.array([x0, y0])[:, None]
        cur_points    = prev_points + offsets[0]

        # Estimate the Global Affine Transform Using Weighted Point Matches.
        tr, err = offset_grid_to_transform.offset_grid_to_transform(
            prev_frame_points = prev_points.reshape(2, -1) + points_offset,
            cur_frame_points  = cur_points.reshape(2, -1)  + points_offset,
            points_weight     = heatmap[0].reshape(-1) ** 2,
        )

        # Return the Estimated Affine Transformation Matrix.
        return tr


    def detect_objects(self, image, prev_image):
        """
        Align the Previous Frame to the Current Frame and Run the Full
        Four-Model Ensemble to Produce Detection Candidates.

        Args:
            image:      Current grayscale frame as a 2D NumPy array.
            prev_image: Previous grayscale frame as a 2D NumPy array.

        Returns:
            Tuple of (detected_objects, prev_tr) where detected_objects is a
            list of detection dicts and prev_tr is the affine transform matrix.
        """
        # Initialise Batch Dimensions and Padding.
        batch_size = 1
        h, w       = image.shape
        padding    = (16 + 32 + 64) // 2

        # Allocate Input Buffer (Two-Frame Padded Array).
        if not hasattr(self, "_detect_buffer"):
            self._detect_buffer = np.zeros(
                (batch_size, 2, h, w + padding * 2), dtype=np.uint8
            )
        X = self._detect_buffer

        # Estimate the Full Affine Transform Between Frames.
        prev_tr = self.estimate_transformation_full(prev_img=prev_image, img=image)

        # Warp the Previous Frame into Alignment with the Current Frame.
        prev_img_aligned = cv2.warpAffine(
            prev_image,
            prev_tr[:2, :],
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
        )

        # Insert Aligned Previous Frame and Current Frame into Input Buffer.
        X[0, 0, :, padding:-padding] = prev_img_aligned
        X[0, 1, :, padding:-padding] = image

        # Run the Four-Model Ensemble to Produce Detection Candidates.
        detected_objects = predict_ensemble.predict_ensemble(
            X                  = X,
            models_full_res    = [self.model_seg1, self.model_seg2],
            models_crops       = [self.model_seg3, self.model_dla],
            full_res_threshold = 0.35,
            x_offset           = -padding,
            device             = self.device,
        )

        return detected_objects, prev_tr
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#               SEG TRACKER (Confidence / Distance Filter Only)
# ------------------------------------------------------------------------------
class SegTracker:
    """
    Lightweight Wrapper Around SegDetector.

    Filters Detections by Distance and Confidence with No Temporal Tracking.
    Used for Single-Frame Evaluation Where Track Continuity is Not Required.
    """
    def __init__(self, detector: SegDetector):
        # Store Reference to the Shared Detector Instance.
        self.detector = detector


    def predict(self, image, prev_image):
        """
        Run Detection and Return Filtered Single-Frame Results.

        Args:
            image:      Current grayscale frame as a 2D NumPy array.
            prev_image: Previous grayscale frame as a 2D NumPy array.

        Returns:
            List of detection dicts, each with a 'track_id' assigned by index.
        """
        # Run the Ensemble Detector.
        detected_objects, prev_tr = self.detector.detect_objects(
            image=image, prev_image=prev_image
        )

        # Filter by Distance and Confidence, Then Assign Sequential Track IDs.
        res = []
        for detected_object in detected_objects:

            # Skip Detections That are Too Far Away.
            if detected_object['distance'] > 800:
                continue

            # Skip Low-Confidence Detections.
            if detected_object['conf'] < 0.6:
                continue

            # Assign a Sequential Track ID.
            detected_object['track_id'] = len(res) + 1
            res.append(detected_object)

        return res
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#          SEG TRACKER FROM OFFSET (Main Inference Tracker / SegPredictor)
# ------------------------------------------------------------------------------
class SegTrackerFromOffset:
    """
    Full Tracking Wrapper Used by SegPredictor.

    Passes Detections Through SimpleOffsetTracker to Assign and Maintain
    Track IDs Across Frames Using Inter-Frame Motion Offsets.
    """
    def __init__(self, detector: SegDetector):
        # Store Reference to the Shared Detector Instance.
        self.detector = detector

        # Initialise the Offset-Based Tracker.
        self.tracker = tracking.SimpleOffsetTracker(
            min_track_size        = 8,
            threshold_to_find     = 0.6,
            threshold_to_continue = 0.6,
            threshold_distance    = 40,
            min_distance          = 1000,
        )

        # Reset the Alignment Cache so the First Frame of This Sequence Does Not
        # Use Stale Features from a Previous Instance or Previous Flight.
        self.detector.reset_transform_cache()


    def predict(self, image, prev_image):
        """
        Run Detection and Return Tracked Results with Stable Track IDs.

        Args:
            image:      Current grayscale frame as a 2D NumPy array.
            prev_image: Previous grayscale frame as a 2D NumPy array.

        Returns:
            List of dicts with cx, cy, w, h, track_id, conf, and offset fields.
        """
        # Run the Ensemble Detector.
        detected_objects, prev_tr = self.detector.detect_objects(
            image=image, prev_image=prev_image
        )

        # Convert Raw Detections into DetectedItem Structures for the Tracker.
        detected_items = [
            DetectedItem(
                cx         = it['cx'] + it['offset'][0],
                cy         = it['cy'] + it['offset'][1],
                w          = it['w'],
                h          = it['h'],
                distance   = it['distance'],
                confidence = it['conf'],
                track_id   = -1,
                dx         = it['tracking'][0],
                dy         = it['tracking'][1],
                item_id    = '',
            )

            # Only Keep Sufficiently Confident Detections.
            for it in detected_objects
            if it['conf'] >= self.tracker.threshold_to_continue
        ]

        # Update Tracker State and Assign Track IDs.
        items_to_submit = self.tracker.process_frame_detections(
            detected_items, prev_tr
        )

        # Format and Return Final Tracked Detections.
        return [
            dict(
                cx       = it.cx,
                cy       = it.cy,
                w        = it.w,
                h        = it.h,
                track_id = it.track_id,
                conf     = it.confidence,
                offset   = [0.0, 0.0],
            )
            for it in items_to_submit
        ]
# ------------------------------------------------------------------------------
