# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: format_output.py %
# Date Created:            %date_created: Mon Apr 20 13:48:31 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Universal adapter template for plugging any detection model into the AICrowd 
# Airborne Object Tracking evaluation pipeline (AirbornePredictor).
#
# Users Only Need to Implement Two Methods:
#   1. inference_setup()     --> Load & Initialise the Model.
#   2. run_model_on_frame()  --> Perform Inference on a Single Frame.
#
# This abstraction allows different detection models (with or without tracking)
# to be plugged into the evaluation pipeline with minimal changes, while
# ensuring consistent output formatting and evaluation behaviour.
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

from tqdm import tqdm
from typing import List, Dict, Any
from logging_module import LogLevel, log

sys.path.insert(0, os.path.dirname(__file__))
from src.baselines.winning_solution.evaluator.airborne_detection import AirbornePredictor

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Minimum Number of Frames a Track Must Span Before Being Committed to Results.
MIN_TRACK_LEN = int(os.getenv("MIN_TRACK_LEN", 30))

# Minimum Confidence Score Required for a Detection to Be Committed to Results.
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.985))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                      TEMPLATE MODEL PREDICTOR CLASS
# ------------------------------------------------------------------------------
class ModelPredictor(AirbornePredictor):
    """
    Universal adapter that connects any detection model to the AICrowd
    AirbornePredictor evaluation pipeline.

    To use this for a new model:
      1. Implement inference_setup() to load the model.
      2. Implement run_model_on_frame() to return detections for one frame.

    The flight iteration, JSON writing, filtering, and format conversion are
    all handled here.
    """
    # These are Required by AirbornePredictor But Not Used in this Adapter.
    training_data_path = None
    test_data_path     = None
    vocabulary_path    = None

    def inference_setup(self) -> None:
        """
        Initialises the segmentation detector and tracker.

        Called once before any flights are processed. Sets up a shared
        SegDetector instance and an initial SegTrackerFromOffset, which
        will be reinitialised per flight in flight_started().
        """
        # Import Libraries Needed for this Function. 
        import torch

        # Set the Device to GPU if Available. 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # [EDIT] Load the New Model Here. Delete this Placeholder.
        raise NotImplementedError(
            "inference_setup() Must Be Implemented. "
            "Load the Model into self.model (And Any Processors Needed)."
        )


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
        # [EDIT] Replace this with the Model's Inference Logic.
        raise NotImplementedError(
            "run_model_on_frame() Must Be Implemented. "
            "Return a List of Detection Dicts."
        )


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
                log(LogLevel.INFO, "format_output:inference", 
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
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = ModelPredictor()
    predictor.run()
# ------------------------------------------------------------------------------
