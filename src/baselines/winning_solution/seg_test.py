# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: seg_test.py %
# Date Created:            %date_created: Mon Mar 02 09:00:00 2021 %
# CM Version               %version: 1 %
# CM Instance              %instance: 3 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Winning solution entrypoint for the AICrowd Airborne Object Tracking
# Challenge (2021). Implements the SegPredictor class which uses a segmentation-
# based detection and tracking pipeline (SegTracker) to identify airborne
# objects in sequential flight frames.
#
# Original solution by Dmytro Poplavskiy (Dmytro.Poplavskiy@gmail.com).
# Full write-up:
#   https://www.aicrowd.com/challenges/airborne-object-tracking-challenge
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from Project Root:
#       - python -m src.baselines.winning_solution.seg_test
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
import sys
import cv2
import re
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.patches

sys.path.insert(0, os.path.dirname(__file__))
from src.baselines.winning_solution.evaluator.airborne_detection import AirbornePredictor
from src.baselines.winning_solution.seg_tracker.seg_tracker import SegDetector, SegTrackerFromOffset

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Minimum Number of Frames a Track Must Span Before Being Committed to Results.
MIN_TRACK_LEN = int(os.getenv("MIN_TRACK_LEN", 30))

# Minimum Confidence Score Required for a Detection to Be Committed to Results.
MIN_SCORE     = float(os.getenv("MIN_SCORE", 0.985))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           SEG PREDICTOR
# ------------------------------------------------------------------------------
class SegPredictor(AirbornePredictor):
    """
    Segmentation-based airborne object predictor.

    Extends AirbornePredictor to implement the winning pipeline from the
    AICrowd Airborne Object Tracking Challenge (2021), using SegDetector and
    SegTrackerFromOffset for per-flight detection and tracking.
    """
    # Set Class Variables
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
        self.detector       = SegDetector()
        self.tracker        = SegTrackerFromOffset(detector=self.detector)
        self.visualize_pred = False


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
        # Initalise Variables.
        frames        = []
        flight_folder = join(str(self.test_data_path), flight_id)

        # Collect All Image Files from the Flight Folder.
        for frame in sorted(listdir(flight_folder)):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)

        # Return the List of Frames. 
        return frames


    def flight_started(self) -> None:
        """
        Resets all per-flight tracking state before processing a new flight.

        Clears result buffers, visited-frame records, and track-length
        counters. Also reinitialises the tracker so that track IDs and
        internal state do not bleed across flights.
        """
        # Reset Per-Flight State Dictionaries.
        self.track_id_results : Dict = {}
        self.visited_frame    : Dict = {}
        self.track_len_so_far : Dict = {}

        # Set Outout Dataset Path. 
        self.inference_output_path

        # Reinitialise Tracker for Each New Flight.
        self.tracker = SegTrackerFromOffset(detector=self.detector)


    def proxy_register_object_and_location(
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

        # Guard Against Duplicate Track IDs Within the Same Frame.
        if img_name not in self.visited_frame:
            self.visited_frame[img_name] = []
        if track_id in self.visited_frame[img_name]:
            raise Exception(
                "Two entities within the same frame {} have the same track id".format(img_name)
            )
        self.visited_frame[img_name].append(track_id)

        # Commit the Detection to the Results.
        self.register_object_and_location(class_name, track_id, bbox, confidence, img_name)


    def inference(self, flight_id: str) -> None:
        """
        Runs the segmentation-based detection and tracking pipeline over all
        frames in a single flight.

        Iterates frames in order, skipping the first (no previous frame for
        differencing), then calls the tracker and registers qualifying
        detections. Optionally renders bounding boxes via matplotlib when
        visualize_pred is enabled.

        Args:
            flight_id (str): The flight folder name to run inference on.

        Returns:
            None:
        """
        # Reset Per-Flight State.
        self.flight_started()
        prev_frame = None

        # Apply to all Frames in the Sequence.
        for frame_image in tqdm(self.get_all_frame_images(flight_id)):
            frame_image_path = self.get_frame_image_location(flight_id, frame_image)
            frame = cv2.imread(str(frame_image_path), cv2.IMREAD_GRAYSCALE)

            # Skip the First Frame - the Tracker Requires a Previous Frame.
            if prev_frame is None:
                prev_frame = frame
                continue

            # Run Tracker on the Current and Previous Frame.
            results    = self.tracker.predict(image=frame, prev_image=prev_frame)
            prev_frame = frame
            class_name = "airborne"

            # Optionally Set Up Matplotlib Visualisation for This Frame.
            ax = None
            if self.visualize_pred and len(results):
                plt.imshow(frame)
                ax = plt.gca()

            # Process Each Detection Result.
            for res in results:
                track_id   = int(res["track_id"])
                confidence = float(res["conf"])

                # Convert Centre + Offset to Absolute Corner Coordinates.
                cx = float(res["cx"]) + float(res["offset"][0])
                cy = float(res["cy"]) + float(res["offset"][1])
                w  = float(res["w"])
                h  = float(res["h"])
                bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

                # Register the Detection.
                self.register_object_and_location(class_name, track_id, bbox, confidence, frame_image)

                # Draw BBox Overlay if Visualisation is Enabled.
                if self.visualize_pred and ax is not None:
                    rect = matplotlib.patches.Rectangle(
                        bbox[:2], w, h, linewidth=1, edgecolor="b", facecolor="none"
                    )
                    ax.add_patch(rect)
                    plt.text(bbox[0], bbox[1], f"{int(confidence * 100)}", c="yellow")

            # Display the Visualised Frame.
            if self.visualize_pred and len(results):
                plt.show()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    submission = SegPredictor()
    submission.run()
# ------------------------------------------------------------------------------
