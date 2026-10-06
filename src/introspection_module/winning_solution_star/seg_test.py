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
#       - python -m src.introspection_module.winning_solution_star.seg_test
#
#   2. Optional Environment Variables:
#       - DEBUG_MODEL_INIT=1
#           Enables verbose/debug logging during model initialisation.
#
#       - MIN_TRACK_LEN=30
#           Minimum track length before detections are committed to results.
#
#       - MIN_SCORE=0.6
#           Minimum confidence score required for final output detections.
#
#       - STAGE_TIMING=1
#           Enables detailed stage timing and profiling across pipeline models.
#
#       - COMPLEXITY_CROP_COUNT=6
#           Upper bound multiplier for Stage-2 per-crop FLOPs computation.
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
import time
import torch
 
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from typing import List, Dict

import matplotlib.patches
import matplotlib.pyplot as plt
 
sys.path.insert(0, os.path.dirname(__file__))
from src.introspection_module.utils import warmup_models as wm
from notebooks.helpers.notebook_helpers import record_fps_stats
from src.introspection_module.utils.timing_profiling import StageTimer, full_frame_input_shape
from src.introspection_module.winning_solution_star.evaluator.airborne_detection import AirbornePredictor
from src.introspection_module.winning_solution_star.seg_tracker.seg_tracker import SegDetector, SegTrackerFromOffset

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Minimum Number of Frames a Track Must Span Before Being Committed to Results.
MIN_TRACK_LEN = int(os.getenv("MIN_TRACK_LEN", 30))

# Minimum Confidence Score Required for a Detection to Be Committed to Results.
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.6))

# Stage-2 FLOPs are Multiplied by this Variable. (6 = Max Crops Possible Per Frame).
COMPLEXITY_CROP_COUNT = int(os.getenv("COMPLEXITY_CROP_COUNT", 6))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           SEG PREDICTOR
# ------------------------------------------------------------------------------
class WinningSolutionPredictor(AirbornePredictor):
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

    def __init__(self) -> None:
        """
        Initialises the predictor instance and applies timeout extensions.

        Warm-up compilation and cold model compilation can take extended
        periods. This lifts the base predictor setup and flight runtime limits.
        """
        super().__init__()
        wm.apply_timeout_buffer(self)


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

        # Accumulates Per-Frame Wall-Clock Inference Times Across All Flights.
        self.frame_times: List[float] = []
        self.timer = StageTimer(enabled=os.getenv("STAGE_TIMING") == "1")
        self.detector.timer = self.timer

        # Run Model Compilation and Input Shape Warm-Up.
        self.warmup()


    def warmup(self) -> None:
        """
        Compiles and exercises all dynamic tensor shapes before inference.

        Warms up full-frame shapes and crop batch sizes (1 to 6) through the
        underlying ensemble, clears alignment caches by reinstantiating the
        tracker, and exports a warm-up timing report to the output directory.
        """
        # Check If Warm-Up Is Enabled.
        if not wm.enabled():
            print("[warm-up] disabled (WARMUP=0).", flush=True)
            return

        # Start the Warm-Up Timer & Create a New Warm-Up Report.
        t0 = time.perf_counter()
        report = wm.new_report("winning_solution_ensemble")

        # Warm Up the Ensemble Using the Full Frame Shape.
        wm.warmup_ensemble(self.detector, wm.resolve_full_shape(self.test_data_path), report)

        # Create a Fresh Tracker (Its Constructor also Clears the Alignment Cache).
        self.tracker = SegTrackerFromOffset(detector=self.detector)

        # Save the Warm-Up Report & Timing.
        wm.finish_report(report, os.path.join(self.get_results_directory(), "warmup_report.json"), t0)


    def set_timer(self, timer: StageTimer) -> None:
        """
        Assigns an external StageTimer to the predictor and underlying detector.

        Args:
            timer (StageTimer): Timer instance to use for profiling stages.
        """
        self.timer = timer
        self.detector.timer = timer

    
    def get_complexity_targets(self) -> List[dict]:
        """
        Declares the models and input configurations for FLOP/complexity profiling.

        Stage 1 (seg1, seg2) runs on the full padded frame. Stage 2 (seg3, dla)
        runs on 512x512 crops (batch 1-6) and is profiled per crop, multiplied by
        COMPLEXITY_CROP_COUNT. The alignment model runs on a 2048x1280 crop.

        Returns:
            List[dict]: Specifications of models, input shapes, multipliers,
                input overrides, and logging labels.
        """
        # Determine the Dimensions of the Full Image. 
        full = full_frame_input_shape(self.test_data_path)

        # Return the Model Shape Results. 
        return [
            {
                "model"       : self.detector._model_seg1_raw,
                "input_shape" : full,
                "label"       : "HRNet32 (model_seg1)",
            },
            {
                "model"       : self.detector._model_seg2_raw,
                "input_shape" : full,
                "label"       : "GerNet-M BiFPN (model_seg2)",
            },
            {
                "model"       : self.detector._model_seg3_raw,
                "input_shape" : (1, 2, 512, 512),
                "count"       : COMPLEXITY_CROP_COUNT,
                "label"       : "HRNet48 (model_seg3) per crop",
            },
            {
                "model"       : self.detector._model_dla_raw,
                "input_shape" : (1, 2, 512, 512),
                "count"       : COMPLEXITY_CROP_COUNT,
                "label"       : "DLA60-Res2Next (model_dla) per crop",
            },
            {
                "model"          : self.detector.model_transform.flops_model,
                "input_shape"    : None,
                "input_override" : (torch.zeros(1, 1280, 2048), torch.zeros(1, 1280, 2048)),
                "label"          : "ResNet34-TSM Transform (model_transform)",
            },
        ]


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

            # Start Timing FPS.
            self.timer.begin_frame()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

            # Run Tracker on the Current & Previous Frame.
            results = self.tracker.predict(image=frame, prev_image=prev_frame)

            # End Timing FPS.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _total = time.perf_counter() - _t0
            self.frame_times.append(_total)
            self.timer.end_frame(_total)
            
            # Update the Previous Frame & Assign Class Name. 
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
                self.proxy_register_object_and_location(class_name, track_id, bbox, confidence, frame_image)

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

        # Save Stage Profiling Metrics to JSON if Enabled.
        if self.timer.enabled and self.timer.n_frames:
            self.timer.save(
                os.path.join(self.get_results_directory(), "stage_timing.json"),
                warmup_frames=10,
                extra=dict(
                    pipeline="winning_solution_ensemble",
                    model_map={"stage1/model0": "HRNet32 (seg1)", 
                               "stage1/model1": "GerNet-M (seg2)",
                               "stage2/model0": "HRNet48 (seg3)", 
                               "stage2/model1": "DLA60 (dla)"
                    },
                ),
            )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    submission = WinningSolutionPredictor()
    submission.run()
# ------------------------------------------------------------------------------
