# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: drone_test.py %
# Date Created:            %date_created: Tue Apr 28 12:21:34 2026 %
# CM Version               %version: 2 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Evaluation entrypoint for the YOLOv11 drone detector.
# Implements DronePredictor, which wraps the Ultralytics YOLOv11 inference
# pipeline inside the AirbornePredictor evaluation framework.
#
# Model was trained on the DUT Anti-Drone dataset at 1240px resolution.
# Single class: drone. All detections are registered as "airborne" for
# compatibility with the evaluation metric.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from the Project Root:
#       - python -m src.baselines.drone_detector.drone_test
#
#   2. Optional Environment Variables:
#       - MODEL_PATH=/path/to/checkpoint.pt
#           Overrides the default segmentation model checkpoint path.
#
#       - MIN_TRACK_LEN=0
#           Minimum track length before detections are committed to results.
#
#       - MIN_SCORE=0.4
#           Minimum confidence score required for final output detections.
#
#       - DETECTION_THRESHOLD=0.4
#           Detection threshold passed directly to the segmentation model.
#           Should generally be <= MIN_SCORE.
#
#       - INFERENCE_IMG_SIZE=1280
#           Image size passed to the YOLO model for inference. This must match 
#           the resolution used during training.
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
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "configs/models/baselines/drone_detector/anti-uav-bw-optimised.pt")

# Minimum Number of Frames a Track Must Span Before Being Committed to Results.
# NOTE: This Model has No Tracker. MIN_TRACK_LEN is Set to 0 so Detections are Not Suppressed.
MIN_TRACK_LEN = int(os.getenv("MIN_TRACK_LEN", 0))

# Minimum Confidence Score Required for a Detection to Be Committed to Results.
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.4))

# Detection Confidence Threshold Passed Directly to the YOLO Model.
# NOTE: Set Equal to or Lower than MIN_SCORE.
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.4))

# Image Size Passed to YOLO Inference.
# NOTE: Must Match the Resolution Used During Training.
INFERENCE_IMG_SIZE = int(os.getenv("INFERENCE_IMG_SIZE", 1280))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         DRONE PREDICTOR CLASS
# ------------------------------------------------------------------------------
class DronePredictor(AirbornePredictor):
    """
    YOLOv11 drone detector for airborne object evaluation.

    Extends AirbornePredictor to plug the Ultralytics YOLOv11 model trained
    on the DUT Anti-Drone dataset into the AICrowd evaluation pipeline.
    Handles per-flight frame iteration, MIN_TRACK_LEN / MIN_SCORE filtering,
    and result.json serialisation.

    The model detects a single class (drone). All detections are registered
    as class_name "airborne" since the evaluation metric requires this label.

    NOTE: This model produces independent per-frame detections with no temporal
    tracking. MIN_TRACK_LEN defaults to 0 for this reason. Each detection
    receives a unique incrementing track_id scoped to the current flight.
    """

    # These are Required by AirbornePredictor But Not Used in this Adapter.
    training_data_path = None
    test_data_path     = None
    vocabulary_path    = None

    def inference_setup(self) -> None:
        """
        Loads the YOLOv11 model weights into memory.

        Called once before any flights are processed. Reads weights from
        MODEL_PATH. The Ultralytics YOLO class handles device placement
        automatically.
        """
        # Import Libraries Needed for this Function. 
        from ultralytics import YOLO

        # Check the Local Model Checkpoint Exists.
        log(LogLevel.INFO, "drone_test:inference_setup",
            "Loading Model from '%s'.", MODEL_PATH)

        # Load YOLOv11 Weights. Ultralytics Handles Device Placement.
        self.model = YOLO(MODEL_PATH)

        # Accumulates Per-Frame Wall-Clock Inference Times Across All Flights.
        self.frame_times: List[float] = []

        # Print Success Message. 
        log(LogLevel.INFO, "drone_test:inference_setup",
            "Model Loaded Successfully.")
    

    def get_complexity_targets(self) -> List[dict]:
        """
        Declares the YOLO model and input shape to be measured by
        measure_predictor_complexity().

        Returns:
            List[dict]: Each entry has keys:
            model        : torch.nn.Module to profile.
            input_shape  : (B, C, H, W) dummy input shape.
            label        : Human-readable name for logging.
        """
        return [
            {
                "model": self.model.model,
                "input_shape": (1, 3, INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE),
                "label": "YOLOv11",
            }
        ]


    def run_model_on_frame(
        self,
        frame     : Any,
        prev_frame: Any,
        frame_name: str,
        flight_id : str
    ) -> List[Dict]:
        """
        Runs YOLOv11 inference on a single frame and returns detections.

        The model is stateless with respect to previous frames — prev_frame
        is accepted for interface compatibility but is not used.

        Args:
            frame      : Current frame as a numpy BGR array (from cv2.imread).
            prev_frame : Previous frame — unused by this model.
            frame_name : Filename of the current frame (for logging only).
            flight_id  : Flight folder name (for logging only).

        Returns:
            List[Dict]: One dict per detection, each with keys:
                - "track_id"   (int)  : Unique ID, incremented per detection.
                - "bbox"       (list) : [x1, y1, x2, y2] in pixel coordinates.
                - "confidence" (float): Score in range (0, 1).
                - "class_name" (str)  : Always "airborne".
        """
        # Start Timing FPS.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()

        # Run YOLO Inference on the Current Frame.
        # verbose=False Suppresses Per-Frame Console Output from Ultralytics.
        results = self.model(
            frame,
            imgsz=INFERENCE_IMG_SIZE,
            conf=DETECTION_THRESHOLD,
            verbose=False
        )

        # End Timing FPS.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.frame_times.append(time.perf_counter() - _t0)

        # Create Detection List.
        detections = []

        # Build a List of Detections with Required Information for Each Object.
        for box in results[0].boxes:
            # Extract Bounding Box as [x1, y1, x2, y2] Corner Coordinates.
            bbox = box.xyxy[0].tolist()

            # Extract Confidence Score.
            confidence = float(box.conf.item())

            # Assign a Unique Track ID for This Detection Within This Flight.
            track_id = self._next_track_id
            self._next_track_id += 1

            # Append the Detection Information to the List of Detections.
            detections.append({
                "track_id"  : track_id,
                "bbox"      : bbox,         # [x1, y1, x2, y2]
                "confidence": confidence,
                "class_name": "airborne"
            })

        # Return the List of Detections.
        return detections


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
        Runs the YOLO detection pipeline over all frames in a single flight.
        Called automatically by AirbornePredictor.evaluation() for each flight.
        Do not call manually.

        Args:
            flight_id (str): The flight folder name to run inference on.
        """
        # Reset Per-Flight State.
        self.flight_started()
        prev_frame = None

        # Apply to all Frames in the Sequence.
        for frame_name in tqdm(self.get_all_frame_images(flight_id), desc=f"Flight {flight_id}"):
            frame_path = self.get_frame_image_location(flight_id, frame_name)
            frame = cv2.imread(str(frame_path))

            # Skip Unreadable Frames.
            if frame is None:
                log(LogLevel.INFO, "drone_test:inference", 
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

            # Register Each Detection through the Filter Layer.
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
    predictor = DronePredictor()
    predictor.run()
# ------------------------------------------------------------------------------
