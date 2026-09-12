# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: hrnetw32_introspective_test.py %
# Date Created:            %date_created: Tue Jul 30 17:45:43 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 3 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Implements the introspective version of the HRNet-W32 Tracking-to-Detect
# (T2D) predictor. This predictor extends the existing T2D pipeline by 
# introducing an introspection module that evaluates the state of each active 
# track and determines whether it should be escalated to a heavyweight ensemble 
# model. The underlying tracking, detection, and coasting behaviour provided by 
# the T2D predictor remains unchanged. The design allows different introspection 
# strategies to be evaluated while preserving the original T2D pipeline as the 
# baseline. This enables direct comparison of performance and computational 
# overhead between standard T2D inference and introspection-driven inference.
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

import time
import torch

from typing import Any, Dict, List
from logging_module import LogLevel, log

from src.introspection_module.hrnetw32_introspective.hrnetw32_track_to_detect_test import HRNetW32TrackToDetectPredictor
from src.introspection_module.hrnetw32_introspective.introspection_modules import IntrospectionModule, NullIntrospectionModule


# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Number of recent observations retained for each track. These values are
# used by the introspection module when making escalation decisions.
TRACK_HISTORY_LEN = 10
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#     HRNET-W32 TRACK-TO-DETECT INTROSPECTIVE PREDICTOR CLASS
# ------------------------------------------------------------------------------
class HRNetW32IntrospectivePredictor(HRNetW32TrackToDetectPredictor):
    """
    Extends the HRNet-W32 Tracking-to-Detect Predictor with an Introspection
    Module.

    The predictor maintains a short history for each active track and provides
    this information to the selected introspection strategy. If the strategy
    determines that a track requires additional processing, the track can be
    escalated to the heavyweight ensemble model while leaving the remainder of
    the T2D pipeline unchanged.
    """
    # Set the Default Introspection Strategy.
    introspection_module: IntrospectionModule = NullIntrospectionModule()


    def inference_setup(self) -> None:
        """
        Initialises the predictor and any resources required for introspection.

        The standard T2D pipeline is initialised through the base class. If the
        selected introspection strategy is capable of requesting escalation, the
        ensemble detector is also loaded.

        Returns:
            None
        """
        # Initialise the Standard T2D Predictor.
        super().inference_setup()
        print()

        # Store the Recent History for Each Active Track.
        self.track_history: Dict[int, Dict[str, List]] = {}

        # Initialise Counters Used for Reporting.
        self.num_escalations: int = 0
        self.num_frames_processed: int = 0

        ### TODO: Restore Conditional Ensemble Loading.
        self._ensemble_detector = None

        # Load the Ensemble Detector Only When it may be Required.
        log(LogLevel.INFO, "hrnetw32_introspective:inference_setup",
            "Introspection Module 'none' Selected. Ensemble NOT Loaded.")

        # Log the Selected Introspection Strategy.
        log(LogLevel.INFO, "hrnetw32_introspective:inference_setup",
            "HRNetW32IntrospectivePredictor Ready (Introspection: %s).",
            self.introspection_module.name)


    def flight_started(self) -> None:
        """
        Resets per-flight state before processing a new flight.

        This includes clearing the stored track history and resetting the selected
        introspection strategy.

        Returns:
            None
        """
        # Reset the Base Predictor State.
        super().flight_started()

        # Clear the Stored Track History.
        self.track_history = {}

        # Reset the Active Introspection Strategy.
        self.introspection_module.reset()


    def _update_track_history(self, track_id: int, confidence: float, source: str, coast_count: int, position: "tuple[float, float]") -> None:
        """
        Updates the stored history for an active track. Only the most recent 
        TRACK_HISTORY_LEN observations are retained.

        Args:
            track_id (int): Unique identifier for the active track.
            confidence (float): Confidence score assigned to the track.
            source (str): Source of the detection (e.g. DETECTED or COASTED).
            coast_count (int): Number of consecutive coasted frames.
            position (tuple[float, float]): Current (cx, cy) bbox-centre position
                                            of the detection, in image coordinates.

        Returns:
            None
        """
        # Create a History Entry for New Tracks.
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'confidence_history': [],
                'source_history': [],
                'coast_count_history': [],
                'position_history': [],
            }

        # Record the Latest Track Information.
        history = self.track_history[track_id]
        history['confidence_history'].append(float(confidence))
        history['source_history'].append(source)
        history['coast_count_history'].append(int(coast_count))
        history['position_history'].append((float(position[0]), float(position[1])))

        # Retain Only the Most Recent Observations.
        for key in ('confidence_history', 'source_history', 'coast_count_history', 'position_history'):
            history[key] = history[key][-TRACK_HISTORY_LEN:]


    def _build_track_state(self, track_id: int, coast_count: int) -> Dict:
        """
        Builds the track state passed to the introspection strategy. The returned
        dictionary contains the recent history and current status of the 
        specified track.

        Args:
            track_id (int): Unique identifier of the track.
            coast_count (int): Current number of consecutive coasted frames.

        Returns:
            Dict: Dictionary containing the track history and current state used
                  by the introspection module.
        """
        # Retrieve Stored History for the Track, or Use an Empty History if it is New.
        history = self.track_history.get(track_id, {
            'confidence_history': [], 'source_history': [], 'coast_count_history': [],
            'position_history': [],
        })

        # Package Track Information into the Format Expected by the Introspection Module.
        return {
            'track_id': track_id,
            'confidence_history': history['confidence_history'],
            'source_history': history['source_history'],
            'coast_count_history': history['coast_count_history'],
            'coast_count': coast_count,
            'position_history': history['position_history'],
            'position': history['position_history'][-1] if history['position_history'] else None,
        }


    def _run_ensemble_on_track(self, det: Dict, frame: Any, prev_frame: Any) -> Dict:
        """
        Runs the ensemble detector for an escalated track. Currently this method
        acts as a placeholder and returns the original detection unchanged while
        recording that an escalation occurred.

        Args:
            det (Dict): Detection associated with the current track.
            frame (Any): Current video frame.
            prev_frame (Any): Previous video frame.

        Returns:
            Dict: Detection result. Currently identical to the input detection.
        """
        # Record that an Escalation Request Occurred.
        self.num_escalations += 1

        ### TODO: No Ensemble Call Happens Here Right Now.
        #         self._ensemble_detector is always None (see inference_setup()),
        #         so there's nothing to call yet.
        

        # Return the Original Detection Until Ensemble Processing is Implemented.
        return det
    

    def run_model_on_frame(self, frame: Any, prev_frame: Any, frame_name: str, flight_id: str) -> List[Dict]:
        """
        Processes a single frame using the T2D pipeline and evaluates each track
        using the selected introspection strategy. If the introspection module 
        requests escalation, the corresponding track is passed to the ensemble
        processing stage. Frame timing is also updated to include the additional
        introspection overhead.

        Args:
            frame (Any): Current video frame.
            prev_frame (Any): Previous video frame.
            frame_name (str): Name or identifier of the current frame.
            flight_id (str): Identifier of the current flight.

        Returns:
            List[Dict]: Detection results for the current frame.
        """
        # Skip Processing when No Previous Frame is Available.
        if prev_frame is None:
            return []

        # Start Timing Before the Complete Inference Pipeline.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Run the Standard T2D Detection, Tracking, and Coasting Pipeline.
        detections = HRNetW32TrackToDetectPredictor.run_model_on_frame(
            self, frame, prev_frame, frame_name, flight_id
        )
        self.num_frames_processed += 1

        # Evaluate Each Detection Using the Selected Introspection Strategy.
        updated_detections = []
        for det in detections:
            track_id = det.get("track_id", -1)
            confidence = det.get("confidence", 0.0)
            source = det.get("source", "DETECTED")
            coast_count = det.get("coast_count", 0)

            # Ignore Detections Without a Valid Track Assignment.
            if track_id < 0:
                updated_detections.append(det)
                continue

            # Extract the Bbox-Centre Position.
            x1, y1, x2, y2 = det["bbox"]
            position = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            # Update Track History and Build the Introspection Input.
            self._update_track_history(track_id, confidence, source, coast_count, position)
            track_state = self._build_track_state(track_id, coast_count)

            # Read the Continuous Score & Threshold Behind this Frame's Escalation Decision.
            introspection_score, introspection_threshold = self.introspection_module.get_score_and_threshold(track_state)
            det["introspection_score"] = introspection_score
            det["introspection_threshold"] = introspection_threshold

            # Merge in any Strategy-Specific Extra Diagnostics. Empty for Strategies 
            # that Don't Override get_extra_diagnostics().
            det.update(self.introspection_module.get_extra_diagnostics(track_state))

            # Escalate the Track if Requested by the Introspection Strategy.      
            if self.introspection_module.should_escalate(track_state):
                # Log the Flag for Debugging/Traceability.
                log(LogLevel.DEBUG, "hrnetw32_introspective:run_model_on_frame",
                    "Escalation Triggered --> Flight: %s Frame: %s Track: %s (coast_count = %d)",
                    flight_id, frame_name, track_id, coast_count)

                # Record that a Flag was Raised.
                det = self._run_ensemble_on_track(det, frame, prev_frame)

                # Mark this Detection as Having Been Warned About.
                det["warning"] = True
            else:
                # No Flag Raised for this Track on this Frame.
                det["warning"] = False

            # Append the Detection to this Frame's Output.
            updated_detections.append(det)

        # Update Timing to Include Introspection and Escalation Overhead.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.frame_times:
            self.frame_times[-1] = time.perf_counter() - t0

        # Return the Updated Detections.
        return updated_detections


    def run(self) -> None:
        """
        Runs inference over the configured dataset and reports summary statistics.

        In addition to the standard predictor metrics, the total number of
        escalations and the escalation rate are reported.

        Returns:
            None
        """
        # Run the Standard T2D Inference Pipeline.
        super().run()

        # Calculate and Report the Frequency of Escalation Requests.
        if self.num_frames_processed:
            escalation_rate = self.num_escalations / self.num_frames_processed

            # Log the Final Introspection Statistics.
            log(LogLevel.INFO, "hrnetw32_introspective:run",
                "Escalations: %d / %d Frame-Track Checks (%.4f Rate) --> Introspection: %s",
                self.num_escalations, self.num_frames_processed, escalation_rate,
                self.introspection_module.name)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = HRNetW32IntrospectivePredictor()
    predictor.run()
# ------------------------------------------------------------------------------
