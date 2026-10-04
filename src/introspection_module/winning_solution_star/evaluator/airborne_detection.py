# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: seg_test.py %
# Date Created:            %date_created: Mon Mar 02 09:00:00 2021 %
# CM Version               %version: 1 %
# CM Instance              %instance: 2 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Evaluation framework and base predictor class for airborne object tracking.
#
# Provides the AirbornePredictor abstract base class, managing dataset flight
# discovery, per-flight image iteration, runtime execution limits, diagnostic
# introspection logging, and final JSON result serialisation.
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

import json
import traceback
import os
import signal

from typing import List, Optional, Any
from contextlib import contextmanager
from os import listdir
from os.path import isfile, join

from src.introspection_module.winning_solution_star.evaluator import aicrowd_helpers


# ------------------------------------------------------------------------------
#                        TIMEOUT HANDLING UTILITIES
# ------------------------------------------------------------------------------
class TimeoutException(Exception):
    """
    Raised when an inference stage or flight exceeds its allotted runtime limit.
    """
    # No Additional Exception Behaviour is Required.
    pass


@contextmanager
def time_limit(seconds: int):
    """
    Enforces a strict execution time limit using POSIX signal alarms.

    Args:
        seconds (int): Maximum duration permitted for code block execution.

    Raises:
        TimeoutException: If the block exceeds the specified second limit.
    """
    # Define the Function Called When the Timeout Alarm Fires.
    def signal_handler(signum, frame):
        # Raise a Timeout Exception to Stop the Current Operation.
        raise TimeoutException("Prediction timed out!")

    # Register the Timeout Signal Handler.
    signal.signal(signal.SIGALRM, signal_handler)

    # Start the Timeout Alarm.
    signal.alarm(seconds)

    # Run the Protected Code Block.
    try:
        yield

    # Always Disable the Timeout Alarm After the Code Block Finishes.
    finally:
        signal.alarm(0)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       AIRBORNE PREDICTOR BASE CLASS
# ------------------------------------------------------------------------------
class AirbornePredictor:
    """
    Abstract base evaluation class for airborne object detection and tracking.

    Manages dataset directories, evaluation execution flow, per-flight timing,
    introspection and detection logging, and submission result generation.
    """

    def __init__(self) -> None:
        """
        Initialises predictor parameters, dataset paths, and timeout ceilings.
        """
        # Dataset & Filesystem Storage Paths.
        self.test_data_path        = os.getenv("TEST_DATASET_PATH", os.path.join(os.getcwd(), "data/corruption_dataset/part1/Images"))
        self.inference_output_path = self.get_results_directory()

        # Operational Execution Timeout Settings (in Seconds).
        self.inference_setup_timeout  = int(os.getenv("INFERENCE_SETUP_TIMEOUT_SECONDS", "600"))
        self.inference_flight_timeout = int(os.getenv("INFERENCE_PER_FLIGHT_TIMEOUT_SECONDS", "600"))

        # Optional Filter for Partial Dataset Flight Evaluations.
        self.partial_run = os.getenv("PARTIAL_RUN_FLIGHTS", None)

        # Internal Results Buffers and Runtime Tracking Identifiers.
        self.results                = []
        self.current_flight_results = []
        self.current_img_name       = None
        self.track_id_seq           = 0


    def register_object_and_location(
        self,
        class_name: str,
        track_id: int,
        bbox: List[float],
        confidence: float,
        img_name: Optional[str],
        source: str = 'DETECTED',
        warning: bool = False,
        introspection_score: Optional[float] = None,
        introspection_threshold: Optional[float] = None,
        **extra_introspection_fields: Any
    ) -> None:
        """
        Registers a tracked object detection and its location.

        Args:
            class_name (str): The class label of the detected object.
            track_id (int): Unique identifier for the tracked airborne object.
            bbox (List[float]): Bounding box in the format (x, y, w, h).
            confidence (float): Detection confidence score in the range (0, 1).
            img_name (Optional[str]): Name of the image this detection belongs to.
                If None, the current image name is used.
            source (str): Detection source, typically 'DETECTED' or 'COASTED'. Defaults to 'DETECTED'.
            warning (bool): Indicates whether this detection should be flagged as an
                introspection warning. Defaults to False.
            introspection_score (Optional[float]): Continuous decision signal from the active
                introspection strategy (e.g., confidence, EWMA, coast count) for this track/frame.
            introspection_threshold (Optional[float]): Threshold introspection_score was
                compared against.
            **extra_introspection_fields (Any): Additional strategy-specific diagnostic fields
                written straight through to the results.
        """
        # Ensure the Confidence Score is a Valid Probability.
        assert 0 < confidence < 1

        # Every Detection Must Belong to a Valid Track.
        assert track_id is not None

        # Fall Back to Currently Active Frame Name if Unspecified.
        if img_name is None:
            img_name = self.current_img_name

        # Create the Detection Record in the Expected Output Format.
        detection_record = {
            "track_id": track_id,
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
            "n": class_name,
            "s": confidence,
            "source": source,
            "warning": warning,
            "introspection_score": introspection_score,
            "introspection_threshold": introspection_threshold,
        }

        # Merge in Any Strategy-Specific Extra Diagnostic Fields.
        detection_record.update(extra_introspection_fields)

        # Create the Result Entry for the Current Image.
        result = {
            "detections": [detection_record],
            "img_name": img_name,
        }

        # Store the Detection in the Overall Results List.
        self.results.append(result)

        # Also Store it in the Current Flight's Results.
        self.current_flight_results.append(result)


    def get_all_flight_ids(self) -> List[str]:
        """
        Discovers and returns all valid flight directory identifiers.

        Returns:
            List[str]: Folder names of flights present in the test dataset.
        """
        # Initialise the Optional List of Flights to Process.
        valid_flight_ids = None

        # Check Whether a Partial Flight Run was Requested.
        if self.partial_run:
            # Convert the Comma-Separated Flight List into a Python List.
            valid_flight_ids = self.partial_run.split(',')

        # Create a List for the Discovered Flight Identifiers.
        flight_ids = []

        # Inspect Every Entry in the Dataset Directory.
        for folder in listdir(self.test_data_path):
            # Keep Only Directory Entries and Apply the Optional Flight Filter.
            if not isfile(join(self.test_data_path, folder)):
                if valid_flight_ids is None or folder in valid_flight_ids:
                    # Add the Valid Flight Directory to the Results.
                    flight_ids.append(folder)

        # Return All Discovered Flight Identifiers.
        return flight_ids


    def get_all_frame_images(self, flight_id: str) -> List[str]:
        """
        Collects all image filenames for a specific flight sequence.

        Args:
            flight_id (str): Directory name of the flight folder.

        Returns:
            List[str]: Unsorted list of image file names within the flight folder.
        """
        # Create a List for the Discovered Frame Filenames.
        frames = []

        # Build the Full Path to the Flight Directory.
        flight_folder = join(self.test_data_path, flight_id)

        # Inspect Every Entry in the Flight Directory.
        for frame in listdir(flight_folder):
            # Keep Only Files and Exclude Subdirectories.
            if isfile(join(flight_folder, frame)):
                # Add the Frame Filename to the List.
                frames.append(frame)
        
        # Return All Frame Filenames.
        return frames


    def get_frame_image_location(self, flight_id: str, frame_id: str) -> str:
        """
        Constructs the absolute filesystem path to an individual frame image.

        Args:
            flight_id (str): Flight sequence identifier.
            frame_id (str): Image filename.

        Returns:
            str: Resolved filepath to the target image.
        """
        # Return the Full Path to the Requested Frame.
        return join(self.test_data_path, flight_id, frame_id)


    def get_flight_folder_location(self, flight_id: str) -> str:
        """
        Constructs the filesystem directory path for a flight sequence.

        Args:
            flight_id (str): Flight sequence identifier.

        Returns:
            str: Resolved filepath to the flight folder.
        """
        # Return the Full Path to the Requested Flight Directory.
        return join(self.test_data_path, flight_id)


    def evaluation(self) -> None:
        """
        Executes the end-to-end evaluation harness across all flight sequences.

        Initialises setup routines with timeout protection, loops across each
        flight, executes inference pipelines, and persists intermediate and
        final predictions.
        """
        # Notify AIcrowd that Evaluation Execution has Started.
        aicrowd_helpers.execution_start()

        # Run the Evaluation Setup with Timeout Protection.
        try:
            with time_limit(self.inference_setup_timeout):
                # Load Models and Perform One-Time Inference Setup.
                self.inference_setup()

        # Continue When the Subclass Does Not Implement Setup.
        except NotImplementedError:
            print("inference_setup doesn't exist for this run, skipping...")

        # Notify AIcrowd that Evaluation Execution is Running.
        aicrowd_helpers.execution_running()

        # Discover All Flights Available for Evaluation.
        flights = self.get_all_flight_ids()

        # Process Each Flight Under Its Configured Timeout.
        for flight in flights:
            # Run Inference for the Current Flight.
            with time_limit(self.inference_flight_timeout):
                self.inference(flight)
            
            # Save the Current Flight Results.
            self.save_results(flight)

        # Save the Complete Aggregate Results.
        self.save_results()

        # Notify AIcrowd that Evaluation Completed Successfully.
        aicrowd_helpers.execution_success()


    def run(self) -> None:
        """
        Main entrypoint initiating evaluation with comprehensive error handling.
        """
        # Attempt to Run the Complete Evaluation.
        try:
            self.evaluation()

        # Handle Any Unexpected Evaluation Errors.
        except Exception as e:
            # Capture the Full Exception Traceback.
            error = traceback.format_exc()

            # Print the Error Traceback.
            print(error)

            # Report the Error to AIcrowd.
            aicrowd_helpers.execution_error(error)
            
            # Re-Raise the Error When Running Outside the Grading Environment.
            if not aicrowd_helpers.is_grading():
                raise e


    def inference_setup(self) -> None:
        """
        Performs one-time pre-evaluation setup and model loading into memory.

        Subclasses must implement this method.
        """
        # Require Subclasses to Provide Their Own Setup Implementation.
        raise NotImplementedError


    def inference(self, flight_id: str) -> None:
        """
        Executes object detection and tracking for all frames in a flight.

        Subclasses must implement this method.

        Args:
            flight_id (str): Target flight folder name to evaluate.
        """
        # Require Subclasses to Provide Their Own Inference Implementation.
        raise NotImplementedError


    def get_results_directory(self, flight_id: Optional[str] = None) -> str:
        """
        Resolves and creates destination directories for prediction output files.

        Args:
            flight_id (Optional[str]): Optional sub-directory scoped to a specific flight.

        Returns:
            str: Resolved output directory path.
        """
        # Get the Root Inference Output Directory from the Environment.
        root_directory = os.getenv("INFERENCE_OUTPUT_PATH", os.path.join(os.getcwd(), "data/results/"))

        # Get the Current Dataset Run Identifier.
        run_id = os.getenv("DATASET_ENV", "run0")

        # Build the Main Results Directory.
        results_directory = os.path.join(root_directory, run_id)

        # Add the Flight Identifier When Saving Per-Flight Results.
        if flight_id is not None:
            results_directory = os.path.join(results_directory, flight_id)

        # Create the Results Directory When It Does Not Exist.
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # Return the Resolved Results Directory.
        return results_directory


    def save_results(self, flight_id: Optional[str] = None) -> None:
        """
        Serialises recorded predictions to JSON in nested flight directories.

        Args:
            flight_id (Optional[str]): Flight identifier for per-flight outputs,
                or None to write overall results.
        """
        # Use the Overall Results When Saving the Final Submission.
        if flight_id is None:
            submission = self.results

        # Otherwise Use the Current Flight Results.
        else:
            submission = self.current_flight_results

        # Build the Output Path for the Current Results.
        output_path = os.path.join(self.get_results_directory(flight_id), "result.json")

        # Open the Output JSON File for Writing.
        with open(output_path, 'w') as fp:
            # Serialise the Results as JSON.
            json.dump(submission, fp)

        # Clear the Per-Flight Results Before Processing the Next Flight.
        self.current_flight_results = []
# ------------------------------------------------------------------------------
