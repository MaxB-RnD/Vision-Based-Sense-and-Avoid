# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: resnet34_test.py %
# Date Created:            %date_created: Tue May 12 12:58:32 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Inference entrypoint for the ResNet34 backbone variant.
# Replaces the 4-model ensemble with a single classic CNN model while keeping
# the identical CenterTrack-style output head and tracking pipeline. Part of
# the complexity vs AFDR experiment (H1). This model provides a well-understood
# convolutional reference point on the complexity curve.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from the Project Root:
#       - python -m src.core_improvement.resnet34.resnet34_test
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

import os
import sys
import cv2
import time
import torch
import numpy as np

from tqdm import tqdm
from typing import List, Dict, Any
from logging_module import LogLevel, log

sys.path.insert(0, os.path.dirname(__file__))
from notebooks.helpers.notebook_helpers import record_fps_stats
from src.baselines.winning_solution.evaluator.airborne_detection import AirbornePredictor

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Path to the Fine-Tuned Model Checkpoint Directory.
MODEL_PATH = os.getenv("MODEL_PATH", "configs/models/core_improvement/resnet34/resnet34_500.pt")

# Minimum Number of Frames a Track Must Span Before Being Committed to Results.
MIN_TRACK_LEN = int(os.getenv("MIN_TRACK_LEN", 30))

# Minimum Confidence Score Required for a Detection to Be Committed to Results.
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.6))

# Detection Confidence Threshold Passed Directly to the Model.
# NOTE: Set Equal to or Lower than MIN_SCORE.
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.6))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                          RESNET34 PREDICTOR CLASS
# ------------------------------------------------------------------------------
class ResNet34Predictor(AirbornePredictor):
    """
    Inference predictor for the ResNet34 backbone.
    Implements inference_setup() and run_model_on_frame() from the
    ModelPredictor template. All other pipeline methods are inherited.
    """
    # These are Required by AirbornePredictor But Not Used in this Adapter.
    training_data_path = None
    test_data_path     = None
    vocabulary_path    = None

    def inference_setup(self) -> None:
        """
        Loads all models into GPU memory before any flights are processed.
        Called once automatically before inference begins.
        """
        # Import Modules Required Only During Inference Setup.
        # Keeping these Imports Local can Reduce Startup/Import Overhead.
        from src.baselines.winning_solution.seg_tracker.prediction_structure import (
            DetectedItem
        )

        from src.baselines.winning_solution.seg_tracker import (
            models_segmentation,
            models_transformation,
            offset_grid_to_transform,
            seg_prediction_to_items,
            tracking,
        )

        # Set the Device to GPU if Available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(LogLevel.INFO, "resnet34:inference_setup", "Device: %s", self.device)

        # Configuration Dictionary for the ResNet34 Segmentation Model.
        seg_cfg = {
            "base_model_name"       : "resnet34",
            "input_frames"          : 2,
            "combine_outputs_dim"   : 256,
            "combine_outputs_kernel": 1,
            "output_binary_mask"    : True,
            "output_above_horizon"  : True,
            "pred_scale"            : 8,
        }

        # Create the Segmentation Model Architecture.
        self.model_seg = models_segmentation.ResNet34Segmentation(
            cfg=seg_cfg, pretrained=False
        )

        # Check if the Pretrained Checkpoint Exists.
        if os.path.exists(MODEL_PATH):
            # Load Checkpoint Onto the Selected Device.
            ckpt  = torch.load(MODEL_PATH, map_location=self.device,
                               weights_only=False)

            # Extract State Dictionary.
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

            # Load Weights into the Model.
            self.model_seg.load_state_dict(state)
            log(LogLevel.INFO, "resnet34:inference_setup",
                "Segmentation Weights Loaded: %s", MODEL_PATH)

        else:
            # Warn if Checkpoint is Missing. Model Will Run with Random/Untrained Weights.
            log(LogLevel.WARNING, "resnet34:inference_setup",
                "Checkpoint Not Found: %s. Using Random Weights.", MODEL_PATH)

        # Move Model Onto Inference Device.
        self.model_seg = self.model_seg.to(self.device)

        # Set Model to Evaluation Mode.
        self.model_seg.eval()

        # Directory Containing Transformation Model Weights.
        models_dir = "configs/models/baselines/winning_solution"

        # Create Transform Estimation Model.
        self.model_transform = models_transformation.TrEstimatorTsm(
            cfg=dict(base_model_name="resnet34", weight_scale=3), pretrained=False
        )

        # Load Pretrained Transformation Model Weights.
        self.model_transform.load_state_dict(
            torch.load(
                f"{models_dir}/030_tr_tsn_rn34_w3_crop_borders_255_0.pth",
                map_location=self.device, weights_only=False
            )["model_state_dict"]
        )

        # Move Transform Model to Inference Device.
        self.model_transform = self.model_transform.to(self.device)

        # Set Transform Model to Evaluation Mode.
        self.model_transform.eval()
        log(LogLevel.INFO, "resnet34:inference_setup", "Transform Model Loaded.")

        # Tracker Used to Maintain Object Identities Between Frames.
        self._offset_tracker = tracking.SimpleOffsetTracker(
            min_track_size        = 8,
            threshold_to_find     = 0.6,
            threshold_to_continue = 0.6,
            threshold_distance    = 40,
            min_distance          = 1000,
        )

        # Store Frame Processing Times for Performance Monitoring.
        self.frame_times: List[float] = []
        log(LogLevel.INFO, "resnet34:inference_setup", "ResNet34Predictor Ready.")

        # Store Helper Modules/Functions as Instance Members for Reuse.
        self._offset_grid_to_transform = offset_grid_to_transform
        self._seg_prediction_to_items  = seg_prediction_to_items
        self._tracking                 = tracking
        self._DetectedItem             = DetectedItem


    def get_complexity_targets(self) -> List[dict]:
        """
        Declares the models and input shapes to be measured by
        measure_predictor_complexity() in notebook_helpers.py. Returns one
        entry per model.

        Returns:
            List[dict]: Each entry has keys:
                model        : torch.nn.Module to profile.
                input_shape  : (B, C, H, W) dummy input shape.
                label        : Human-readable name for logging.
        """
        return [
            {
                "model"       : self.model_seg,
                "input_shape" : (1, 2, 2048, 2432),
                "label"       : "ResNet34 (segmentation)",
            },
            {
                "model"       : self.model_transform,
                "input_shape"    : None,
                "input_override" : (
                    torch.zeros(1, 1024, 1024),  # prev_frame.
                    torch.zeros(1, 1024, 1024),  # cur_frame.
                ),
                "label"       : "ResNet34-TSM Transform (model_transform)",
            },
        ]


    def run_model_on_frame(
        self,
        frame     : Any,
        prev_frame: Any,
        frame_name: str,
        flight_id : str
    ) -> List[Dict]:
        """
        Run the model on a single frame and return detected objects.

        Args:
            frame: Current frame as a numpy array (BGR format).
            prev_frame: Previous frame, or None if this is the first frame.
            frame_name: Name of the current frame file.
            flight_id: Identifier for the flight sequence.

        Returns:
            List[Dict]: A list of detections, where each detection contains:
                - track_id: Unique identifier for the object.
                - bbox: Bounding box coordinates.
                - confidence: Detection confidence score (0–1).
                - class_name: Label of the detected object.
        """
        # Cannot Estimate Motion or Perform Temporal Inference Without a Previous Frame.
        if prev_frame is None:
            return []

        # Convert Frames to Grayscale.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame

        # Start Frame Processing Timer.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Estimate Frame-to-Frame Transform.
        prev_tr = self._estimate_transform(prev_gray, frame_gray)

        # Build Two-Frame Model Input.
        h, w = frame_gray.shape
        padding = (16 + 32 + 64) // 2   # = 56

        # Allocate Input Tensor.
        X = np.zeros((1, 2, h, w + padding * 2), dtype=np.uint8)

        # Warp the Previous Frame into Alignment with the Current Frame.
        prev_aligned = cv2.warpAffine(prev_gray, prev_tr[:2, :], dsize=(w, h),
                                       flags=cv2.INTER_LINEAR)

        # Insert Aligned Previous Frame into Model Input.
        X[0, 0, :, padding:-padding] = prev_aligned

        # Insert Current Frame into Model Input.
        X[0, 1, :, padding:-padding] = frame_gray

        # Convert Input Array into Normalised PyTorch Tensor.
        X_t = torch.from_numpy(X).float().to(self.device) / 255.0

        # Disable Gradients During Inference for Efficiency.
        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                # Run Segmentation Model Forward Pass.
                pred = self.model_seg(X_t)

        # Extract Model Outputs.
        mask = torch.sigmoid(pred["mask"][0, 0].float()).cpu().numpy()
        offset = pred["offset"][0].float().cpu().numpy()
        size = pred["size"][0].float().cpu().numpy()
        tracking_out = pred["tracking"][0].float().cpu().numpy()
        distance = pred["distance"][0, 0].float().cpu().numpy()
        above_horizon = pred["above_horizon"][0, 0].float().cpu().numpy()

        # Remove Border Artefacts.
        border_x = padding // 8 + 4
        border_y = 4

        # Zero Out Unreliable Border Regions.
        mask[:border_y, :]      = 0.0
        mask[-border_y - 4:, :] = 0.0
        mask[:, :border_x]      = 0.0
        mask[:, -border_x:]     = 0.0

        # Convert Heatmaps into Detection Candidates.
        detected = self._seg_prediction_to_items.pred_to_items(
            comb_pred   = mask,
            offset = offset,
            size = size,
            tracking = tracking_out,
            distance = distance,
            above_horizon  = above_horizon,
            conf_threshold = DETECTION_THRESHOLD,
            pred_scale = 8,
            x_offset = -padding,
        )

        # Multi-Frame Object Tracking.
        detected_items = [
            self._DetectedItem(
                cx = it["cx"] + it["offset"][0],
                cy = it["cy"] + it["offset"][1],
                w = it["w"],
                h = it["h"],
                distance = it["distance"],
                confidence = it["conf"],
                track_id = -1,
                dx = it["tracking"][0],
                dy = it["tracking"][1],
                item_id = "",
            )

            # Only Keep Sufficiently Confident Detections.
            for it in detected
            if it["conf"] >= self._offset_tracker.threshold_to_continue
        ]

        # Update Tracker State & Assign Track IDs.
        items_to_submit = self._offset_tracker.process_frame_detections(
            detected_items, prev_tr
        )

        # Stop Frame Processing Timer.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.frame_times.append(time.perf_counter() - t0)

        # Format Final Output.
        return [
            {
                "track_id"  : item.track_id,
                "bbox"      : [item.cx - item.w/2, item.cy - item.h/2,
                               item.cx + item.w/2, item.cy + item.h/2],
                "confidence": float(item.confidence),
                "class_name": "airborne",
            }
            for item in items_to_submit
        ]


    def _estimate_transform(self, prev_img: np.ndarray,
                             img: np.ndarray) -> np.ndarray:
        """
        Estimate the affine transform between two consecutive frames using
        the ResNet34-TSM motion estimation model.

        Args:
            prev_img (np.ndarray):
                Previous grayscale image/frame as a 2D NumPy array.
            img (np.ndarray):
                Current grayscale image/frame as a 2D NumPy array.

        Returns:
            np.ndarray:
                Estimated affine transformation matrix mapping points from the
                previous frame into the current frame coordinate system.
        """
        # Define the Fixed Crop Size Used for Motion Estimation.
        crop_w, crop_h = 2048, 1280

        # Get Image Dimensions.
        h, w = img.shape

        # Fixed Vertical Crop Start.
        y0 = 512

        # Centre the Crop Horizontally.
        x0 = (w - crop_w) // 2

        # Number of Tracking Points in the Model Output Grid.
        cw_pts = crop_w // 32
        ch_pts = crop_h // 32

        # Extract Matching Crops from the Current & Previous Frames.
        crop      = img[y0:y0+crop_h, x0:x0+crop_w]
        crop_prev = prev_img[y0:y0+crop_h, x0:x0+crop_w]

        # Convert Cropped Images into PyTorch Tensors & Normalise to [0, 1].
        t_cur  = torch.from_numpy(crop / 255.0).float().to(self.device)
        t_prev = torch.from_numpy(crop_prev / 255.0).float().to(self.device)

        # Run Inference Without Gradient Tracking for Efficiency.
        with torch.no_grad():
            # Enable Automatic Mixed Precision for Faster Inference on GPU.
            with torch.amp.autocast(self.device.type):
                heatmap, offsets = self.model_transform(t_prev[None], t_cur[None])

        # Move Model Outputs Back to CPU NumPy Arrays.
        heatmap = heatmap.detach().float().cpu().numpy()
        offsets = offsets.detach().float().cpu().numpy()

        # Create a Regular Grid of Points in the Previous Frame Crop.
        prev_pts = np.zeros((2, ch_pts, cw_pts), dtype=np.float32)
        prev_pts[0] = np.arange(16, crop_w, 32)[None, :]
        prev_pts[1] = np.arange(16, crop_h, 32)[:, None]
        prev_pts = prev_pts[..., 2:-2, 2:-2]

        # Convert Crop-Local Coordinates Back into Full-Image Coordinates.
        pts_offset = np.array([x0, y0])[:, None]
        cur_pts = prev_pts + offsets[0]

        # Estimate the Global Affine Transform Using Weighted Point Matches.
        tr, _ = self._offset_grid_to_transform.offset_grid_to_transform(
            prev_frame_points = prev_pts.reshape(2, -1) + pts_offset,
            cur_frame_points  = cur_pts.reshape(2, -1) + pts_offset,
            points_weight     = heatmap[0].reshape(-1) ** 2,
        )

        # Return the Estimated Affine Transformation Matrix.
        return tr


    # --------------------------------------------------------------------------
    #           DO NOT MODIFY THE BELOW CODE (RESULTS PIPELINING)
    # --------------------------------------------------------------------------
    def flight_started(self) -> None:
        """
        Resets all per-flight state. Called automatically at the start of
        each flight by inference(). Do not call manually.

        Clears result buffers, visited-frame records, and track-length
        counters. Also reinitialises the tracker so that track IDs and
        internal state do not bleed across flights.
        """
        # Reset Per-Flight State Dictionaries.
        self.track_id_results : Dict = {}
        self.visited_frame    : Dict = {}
        self.track_len_so_far : Dict = {}

        # Reset the Per-Flight Track ID Counter for Untracked Models.
        # Safe to Leave in Even if the Model has its Own Tracker.
        self._next_track_id : int = 1

        # Reset Optional Offset Tracker State if this Model Uses One.
        # Prevents Frame History & Track IDs from Leaking Between Flights.
        if hasattr(self, "_offset_tracker"):
            self._offset_tracker.prev_frames_items = [[]]
            self._offset_tracker.track_ids_in_use  = set()


    def get_all_frame_images(self, flight_id: str) -> List[str]:
        """
        Returns a sorted list of image filenames for the given flight.

        Scans the flight folder under test_data_path and collects all files
        (excluding subdirectories) in sorted order.

        Args:
            flight_id (str): The flight folder name.

        Returns:
            List[str]: Sorted list of image filenames within the flight folder.
        """
        # Import Extra Libraries for this Function.
        from os import listdir
        from os.path import isfile, join

        # Initalise Variables.
        flight_folder = join(str(self.test_data_path), flight_id)

        # Return All Image Files from the Flight Folder.
        return sorted(
            f for f in listdir(flight_folder)
            if isfile(join(flight_folder, f))
        )


    def _proxy_register(
        self,
        class_name : str,
        track_id   : int,
        bbox       : List[float],
        confidence : float,
        img_name   : str
    ) -> None:
        """
        Wraps register_object_and_location with MIN_TRACK_LEN and MIN_SCORE
        filtering before committing a detection to the results.

        A detection is only forwarded if:
          - Its track has been observed for more than MIN_TRACK_LEN frames.
          - Its confidence score meets or exceeds MIN_SCORE.
          - Its track ID has not already been registered for this frame.

        Args:
            class_name (str): Object class label, e.g. "airborne".
            track_id (int): Unique ID for the tracked object.
            bbox (List[float]): Bounding box [x1, y1, x2, y2].
            confidence (float): Detection confidence score (0-1).
            img_name (str): Image filename this detection belongs to.

        Raises:
            Exception: If the same track_id is registered twice for img_name.
        """
        # Initialise Track Length Counter if this is a New Track ID.
        if track_id not in self.track_len_so_far:
            self.track_len_so_far[track_id] = 0
        self.track_len_so_far[track_id] += 1

        # Skip Detections from Tracks that Have Not Yet Reached MIN_TRACK_LEN.
        if self.track_len_so_far[track_id] <= MIN_TRACK_LEN:
            return

        # Skip Detections Below the Minimum Confidence Threshold.
        if confidence < MIN_SCORE:
            return

        # Guard Against Duplicate Track IDs within the Same Frame.
        if img_name not in self.visited_frame:
            self.visited_frame[img_name] = []
        if track_id in self.visited_frame[img_name]:
            raise Exception(
                f"Two Detections in Frame '{img_name}' Share track_id={track_id}."
            )
        self.visited_frame[img_name].append(track_id)

        # Commit the Detection to the Results.
        self.register_object_and_location(class_name, track_id, bbox, confidence, img_name)


    def inference(self, flight_id: str) -> None:
        """
        Runs the segmentation-based detection and tracking pipeline over all
        frames in a single flight.

        Iterates frames in order then calls the tracker and registers qualifying
        detections. Called automatically by AirbornePredictor.evaluation() for
        each flight.

        Do not call manually.

        Args:
            flight_id (str): The flight folder name to run inference on.

        Returns:
            None:
        """
        # Reset Per-Flight State.
        self.flight_started()
        prev_frame = None

        # Apply to all Frames in the Sequence.
        for frame_name in tqdm(self.get_all_frame_images(flight_id), desc=f"Flight {flight_id}"):
            frame_path = self.get_frame_image_location(flight_id, frame_name)
            frame = cv2.imread(str(frame_path))

            # Skip Empty Frames.
            if frame is None:
                log(LogLevel.WARNING, "format_output:inference",
                     "Could Not Read Frame '%s', Skipping.", frame_path)
                prev_frame = frame
                continue

            # Run the Model on the Current & Previous Frame.
            detections = self.run_model_on_frame(
                frame=frame,
                prev_frame=prev_frame,
                frame_name=frame_name,
                flight_id=flight_id
            )

            # Update the Frames.
            prev_frame = frame

            # Register Each Detection through the Filter/Format Layer.
            for det in detections:
                self._proxy_register(
                    class_name = det.get("class_name", "airborne"),
                    track_id = int(det["track_id"]),
                    bbox = det["bbox"],
                    confidence = float(det["confidence"]),
                    img_name = frame_name
                )


    def run(self) -> None:
        """
        Extends the parent run() to save FPS statistics to fps_stats.json
        inside the run directory after all flights have been processed.

        This ensures record_fps_stats() is called exactly once, after the
        full frame_times list has been populated across every flight.
        """
        # Run the Standard Evaluation Pipeline.
        super().run()

        # Save FPS Stats if any Frames were Timed.
        if self.frame_times:
            record_fps_stats(
                frame_times_sec = self.frame_times,
                run_dir         = self.get_results_directory(),
                warmup_frames   = 10,
            )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = ResNet34Predictor()
    predictor.run()
# ------------------------------------------------------------------------------
