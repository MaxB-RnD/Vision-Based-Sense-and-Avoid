# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: introspection_modules.py %
# Date Created:            %date_created: Tue Jul 30 17:35:45 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 3 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# An introspection module examines the recent state of an individual track
# and determines whether that track should be processed by the heavyweight
# ensemble model instead of the lightweight HRNet-W32 model for the current
# frame. Each strategy operates using information already available from the 
# tracker, such as confidence history, track age, and coast state. The module 
# does not modify the underlying detection models or perform any training; it
# simply provides a run-time decision on whether additional processing is
# required.
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

from typing import Dict, List

import numpy as np


# ------------------------------------------------------------------------------
#                         INTROSPECTION MODULE BASE
# ------------------------------------------------------------------------------
class IntrospectionModule:
    """
    Base Class for All Introspection Strategies.

    Each implementation determines whether a track should be escalated to the
    heavyweight ensemble model for the current frame. The predictor performs
    the escalation itself, while the introspection module is responsible only
    for making the decision.
    """
    # Filesystem-Safe Identifier Used in Logs & Run Directory Naming.
    name: str = "base"

    def should_escalate(self, track_state: Dict) -> bool:
        """
        Decide Whether a Track Should be Escalated to the Ensemble Model.

        Args:
            track_state (Dict): Per-track signals accumulated so far this flight. 
                                Expected keys (populated by the predictor, not 
                                all strategies need to use all of them)

        Returns:
            bool: True if this track should be escalated for the current frame, 
                  False to continue using the lightweight backbone.
        """
        # Send Clean Error Message to the Log.
        raise NotImplementedError


    def get_score_and_threshold(self, track_state: Dict):
        """
        Returns this Strategy's Continuous Decision Signal & the Threshold it 
        is Being Compared Against for the Current Frame.

        This is Separate from should_escalate() so that the Underlying Score
        (Not Just the Final True/False Decision) can be Written Through to
        result.json for Downstream Visualisation/Debugging. Every Concrete
        Strategy Should Override this so that 'introspection_score' and
        'introspection_threshold' are Populated in the Final Results.

        Args:
            track_state (Dict): Per-track signals accumulated so far this
                                flight. Same input as should_escalate().

        Returns:
            Tuple[Optional[float], Optional[float]]: (score, threshold). Either 
                may be None if the Strategy has Not Yet Accumulated Enough 
                History to Report a Value for this Track/Frame.
        """
        # Base Class has No Score to Report. Subclasses Should Override.
        return None, None


    def get_extra_diagnostics(self, track_state: Dict) -> Dict:
        """
        Optional Additional Per-Track Diagnostic Values to Write Through to
        result.json, for Strategies whose Escalation Decision Combines More
        than One Signal (e.g. an OR of Two Independent Checks) where a Single
        Score/Threshold Pair Doesn't Tell the Whole Story.

        Args:
            track_state (Dict): Per-track signals accumulated so far this
                                flight. Same input as should_escalate().

        Returns:
            Dict: Extra {field_name: value} Pairs to Merge into the Detection
                  Record. Empty by Default -- Most Strategies Don't Need this.
        """
        # No Extra Diagnostics by Default.
        return {}


    def reset(self) -> None:
        """
        Resets any Internal State Held by the Introspection Module.

        Called at the beginning of each flight to ensure that information from
        previous flights is not carried over. Stateless implementations do not
        need to override this method.
        """
        # A Clean Pass Through for Resetting.
        pass
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#             MODULE 0 -- NULL INTROSPECTION (CONTROL CONDITION)
# ------------------------------------------------------------------------------
class NullIntrospectionModule(IntrospectionModule):
    """
    Introspection Strategy that Never Requests Escalation.

    This module acts as the control condition for experiments by disabling
    introspection entirely. It allows the behaviour of the introspective
    predictor to be compared directly with the standard HRNet-W32 predictor,
    ensuring that any observed differences are due to the introspection logic
    rather than the surrounding evaluation framework.
    """
    # Identifier Used to Select this Strategy within the Evaluation Framework.
    name = "none"

    # Always Returns False so that No Tracks are Escalated to the Ensemble Model.
    def should_escalate(self, track_state: Dict) -> bool:
        return False


    # No Decision Signal to Report -- this is the Control Condition.
    def get_score_and_threshold(self, track_state: Dict):
        return None, None
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   MODULE 1 -- SIMPLE CONFIDENCE THRESHOLD
# ------------------------------------------------------------------------------
class ConfidenceThresholdIntrospectionModule(IntrospectionModule):
    """
    Escalates a Track when its Confidence Remains Below a Specified Threshold
    for a Fixed Number of Consecutive Frames.

    This strategy provides a simple baseline for evaluating the effectiveness
    of introspection. It assumes that consistently low confidence indicates
    increased uncertainty, making the track a suitable candidate for
    reprocessing with the more computationally expensive ensemble model.
    """
    # Identifier Used to Select this Strategy within the Evaluation Framework.
    name = "confidence_threshold"

    def __init__(self, threshold: float = 0.70, window: int = 3):
        # Minimum Confidence Required Before Escalation is Considered.
        self.threshold = threshold

        # Number of Consecutive Frames that Must Satisfy the Threshold
        # Condition Before Escalation is Triggered.
        self.window = window


    def should_escalate(self, track_state: Dict) -> bool: 
        # Retrieve the Confidence Values Recorded for this Track.
        history: List[float] = track_state.get('confidence_history', [])

        # Wait Until Enough Confidence Samples have been Collected Before
        # Evaluating the Escalation Condition.
        if len(history) < self.window:
            return False

        # Examine Only the Most Recent Confidence Values Defined by the
        # Window Size.
        recent_history = history[-self.window:]

        # Escalate Only if Every Confidence Value in the Selected Window
        # Remains Below the Threshold.
        return all(c < self.threshold for c in recent_history)


    def get_score_and_threshold(self, track_state: Dict):
        """
        Reports the Track's Most Recent Confidence Value Against the Fixed
        Threshold this Strategy Escalates On.

        Note this is a Single-Frame Reading, not the Windowed all()
        Condition should_escalate() Actually Applies -- it's the Clearest
        Single Number to Show Alongside the Escalation Decision.
        """
        # Retrieve the Confidence Values Recorded for this Track.
        history: List[float] = track_state.get('confidence_history', [])

        # Nothing Recorded Yet for this Track.
        if not history:
            return None, self.threshold

        # Report the Most Recent Confidence Reading Against the Threshold.
        return history[-1], self.threshold
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   MODULE 2 -- ROLLING CONFIDENCE TREND / EWMA
# ------------------------------------------------------------------------------
class RollingConfidenceTrendIntrospectionModule(IntrospectionModule):
    """
    Escalates a Track Based on the TREND of its Confidence Over Recent Frames, 
    Rather than a Single-Frame Threshold Check.

    Combines two related signals as an OR condition -- either alone is 
    sufficient to raise a flag, since they catch different failure shapes:

      1. EWMA below threshold: An Exponentially Weighted Moving Average of 
         confidence has dropped below a fixed level. EWMA requires SUSTAINED low 
         confidence to cross the line -- one noisy low frame is smoothed out, a
         genuine multi-frame decline is not.

      2. Negative slope beyond threshold: A linear trend fit over the recent 
         confidence window has a sufficiently negative slope. This can fire even 
         if the EWMA/absolute confidence level hasn't dropped far yet -- it's 
         reacting to the DIRECTION of change, not the current value.
    """
    # Identifier Used to Select this Strategy within the Evaluation Framework.
    name = "rolling_confidence_trend"

    def __init__(self, ewma_threshold: float = 0.70, ewma_alpha: float = 0.30,
                 slope_threshold: float = -0.05, min_history: int = 3):
        # Set EWMA Threshold.
        self.ewma_threshold = ewma_threshold

        # Set EWMA Smoothing Factor.
        self.ewma_alpha = ewma_alpha

        # Set Slope Threshold.
        self.slope_threshold = slope_threshold

        # Set Minimum Number of Frames of History Required Before Either Signal
        # is Evaluated.
        self.min_history = min_history


    def _compute_ewma(self, history: List[float]) -> float:
        # Initialise the EWMA with the Oldest Confidence Value in the History.
        ewma = history[0]

        # Update the Running EWMA. Newer Confidence Values Receive Greater
        # Weight, While Older Values Gradually Decay in Influence.
        for value in history[1:]:
            ewma = self.ewma_alpha * value + (1 - self.ewma_alpha) * ewma

        # Return the Final Smoothed Confidence Estimate.
        return ewma


    def _compute_slope(self, history: List[float]) -> float:
        # Number of Confidence Samples in the Current History Window.
        n = len(history)

        # Use Sequential Frame Indices as the Independent Variable (x).
        x = list(range(n))

        # Compute the Mean Frame Index and Mean Confidence.
        x_mean = sum(x) / n
        y_mean = sum(history) / n

        # Compute the Variance of the Frame Indices.
        denom = sum((xi - x_mean) ** 2 for xi in x)

        # Guard Against Division by Zero if the Variance is Zero.
        if denom == 0:
            return 0.0

        # Compute the Covariance Between Frame Index and Confidence.
        numer = sum((x[i] - x_mean) * (history[i] - y_mean) for i in range(n))

        # Return the Best-Fit Slope.
        return numer / denom


    def should_escalate(self, track_state: Dict) -> bool:
        # Retrieve the Confidence Values Recorded for this Track.
        history: List[float] = track_state.get('confidence_history', [])

        # Wait Until Enough History has Accumulated Before Evaluating
        # Either Signal.
        if len(history) < self.min_history:
            return False

        # Signal 1: Smoothed Confidence has Dropped Below Threshold.
        ewma = self._compute_ewma(history)
        if ewma < self.ewma_threshold:
            return True

        # Signal 2: Confidence is Actively Trending Downward, Regardless
        # of Current Absolute Level.
        slope = self._compute_slope(history)
        if slope < self.slope_threshold:
            return True

        # Neither Signal Fired. No Escalation.
        return False


    def get_score_and_threshold(self, track_state: Dict):
        """
        Reports the EWMA-Smoothed Confidence as the Canonical Score, Since it's
        the More 'Confidence-Like' of the Two Signals this Strategy Combines.
        The Slope Signal is Reported Separately via get_extra_diagnostics(),
        Since a Single Score/Threshold Pair Can't Represent an OR of Two
        Independent Checks Without Being Misleading (e.g. Escalation can Fire
        on Slope Alone While EWMA Sits Comfortably Above its Threshold).
        """
        # Retrieve the Confidence Values Recorded for this Track.
        history: List[float] = track_state.get('confidence_history', [])

        # Not Enough History Yet to Report a Meaningful EWMA.
        if len(history) < self.min_history:
            return None, self.ewma_threshold

        # Report the Smoothed Confidence Estimate Against its Threshold.
        return self._compute_ewma(history), self.ewma_threshold


    def get_extra_diagnostics(self, track_state: Dict) -> Dict:
        """
        Reports the Confidence Trend Slope Alongside its Own Threshold, Since
        Escalation can be Triggered by Slope Alone Even when EWMA (the
        Primary Score) has Not Crossed its Threshold.
        """
        # Retrieve the Confidence Values Recorded for this Track.
        history: List[float] = track_state.get('confidence_history', [])

        # Not Enough History Yet to Report a Meaningful Slope.
        if len(history) < self.min_history:
            return {'introspection_slope': None, 'introspection_slope_threshold': self.slope_threshold}

        # Report the Confidence Trend Slope Against its Threshold.
        return {
            'introspection_slope': self._compute_slope(history),
            'introspection_slope_threshold': self.slope_threshold,
        }
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#               MODULE 3 -- CONSECUTIVE-COAST-COUNT THRESHOLD
# ------------------------------------------------------------------------------
class CoastCountIntrospectionModule(IntrospectionModule):
    """
    Escalates a Track Once it has been Propagated by Motion Prediction
    (Coasting) for More than a Specified Number of Consecutive Frames
    Without a Real Detection.
    """
    # Identifier Used to Select this Strategy within the Evaluation Framework.
    name = "coast_count_threshold"

    def __init__(self, threshold: int = 3):
        # Number of Consecutive Coasted Frames Required Before Escalation.
        self.threshold = threshold


    def should_escalate(self, track_state: Dict) -> bool:
        # Coast Count is Already Maintained by the T2D Tracker Itself. Read 
        # Directly, No History Reconstruction Needed.
        coast_count = track_state.get('coast_count', 0)

        # Return True Once the Number of Consecutive Coasted Frames Meets or
        # Exceeds the Configured Escalation Threshold.
        return coast_count >= self.threshold


    def get_score_and_threshold(self, track_state: Dict):
        """
        Reports the Track's Current Consecutive Coast Count Against the
        Configured Escalation Threshold.
        """
        # Coast Count is Read Directly from Track State, Same as should_escalate().
        return float(track_state.get('coast_count', 0)), float(self.threshold)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                MODULE 4 -- KALMAN FILTER NIS (LEARNING-FREE)
# ------------------------------------------------------------------------------
# 95% Chi-Squared Critical Value for 2 Degrees of Freedom.
CHI2_CRITICAL_DF2_95 = -2.0 * np.log(1.0 - 0.95)


class KalmanNISIntrospectionModule(IntrospectionModule):
    """
    Escalates a Track when its Observed Position Becomes Statistically Inconsistent
    with a Constant-Velocity Kalman Filter's Predicted Position. The Filter Tracks
    [cx, cy, vx, vy] and Computes the Normalised Innovation Squared (NIS) from the
    Position Innovation, Escalating when NIS Exceeds the 95% Chi-Squared Critical
    Value for 2 Degrees of Freedom.
    """
    # Identifier Used to Select this Strategy within the Evaluation Framework.
    name = "kalman_nis"

    def __init__(self, chi2_threshold: float = CHI2_CRITICAL_DF2_95,
                 process_noise: float = 1.0, measurement_noise: float = 4.0,
                 initial_position_var: float = 100.0, initial_velocity_var: float = 1000.0):
        # NIS Threshold Above which a Track is Considered Statistically Inconsistent.
        self.chi2_threshold = chi2_threshold

        # Continuous White-Noise-Acceleration Spectral Density. Larger Values Make
        # the Filter More Tolerant of Genuine Manoeuvring, at the Cost of Being Less 
        # Sensitive to Real Anomalies.
        self.process_noise = process_noise

        # Assumed Variance of the Detector's Own Position Noise (pixels^2).
        self.measurement_noise = measurement_noise

        # Initial State Uncertainty Used the First Time a Track is Observed.
        self.initial_position_var = initial_position_var
        self.initial_velocity_var = initial_velocity_var

        # Per-Track Kalman Filter State, Keyed by track_id. Reset Every Flight.
        self._filters: Dict[int, Dict] = {}


    def reset(self) -> None:
        """
        Clears all Per-Track Kalman Filter State.
        """
        # Set Filters to an Empty Dictionary. 
        self._filters = {}


    @staticmethod
    def _build_matrices(process_noise: float, measurement_noise: float):
        """
        Builds the Constant-Velocity Kalman Filter Matrices for a Single
        Discrete Time Step (dt=1, Matching the Fixed-FPS Frame Spacing this
        Pipeline Assumes Elsewhere, e.g. SimpleOffsetTracker's Per-Frame dx/dy).
        """
        # State Transition: Position Advances by Velocity Each Frame.
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Measurement Model: Only Position is Directly Observed.
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Discretised White-Noise-Acceleration Process Noise (dt=1).
        q = process_noise
        Q = np.array([
            [q / 4, 0.0,   q / 2, 0.0],
            [0.0,   q / 4, 0.0,   q / 2],
            [q / 2, 0.0,   q,     0.0],
            [0.0,   q / 2, 0.0,   q],
        ])

        # Isotropic Measurement Noise on (cx, cy).
        R = np.eye(2) * measurement_noise

        # Return the State-Transition, Measurement, Process-Noise & Measurement-Noise Matrices.
        return F, H, Q, R


    def _step(self, track_state: Dict):
        """
        Advances this Track's Kalman Filter by Exactly One Frame and Returns
        the NIS for that Frame, or None if no Prediction Exists Yet to Compare
        Against (the Track's First Observation), or if this Frame was Coasted.

        Args:
            track_state (Dict): Per-track signals for this frame, including
                                'track_id', 'position', and 'source_history'.

        Returns:
            Optional[float]: This frame's NIS, or None if it could not be
                             computed (first observation, missing position,
                             or a coasted frame).
        """
        # Extract the Track ID, Current Position & Source from the Track State.
        track_id = track_state.get('track_id', -1)
        position = track_state.get('position')
        source_history: List = track_state.get('source_history', [])
        source = source_history[-1] if source_history else 'DETECTED'

        # Nothing to Do Without a Valid Track or a Position to Compare Against.
        if track_id < 0 or position is None:
            return None

        # Get the Cache Key Associated with this Track ID. 
        cache_key = id(track_state)

        # Retrieve the Existing Kalman Filter State for this Track, if One Exists.
        filt = self._filters.get(track_id)

        # Already Stepped this Exact Frame for this Track. Return the Cached Result.
        if filt is not None and filt.get('last_track_state_id') == cache_key:
            return filt['last_nis']

        # Convert the Current Position Measurement into a NumPy Floating-Point Array.
        z = np.array(position, dtype=np.float64)

        # First Observation for this Track.
        if filt is None:
            x0 = np.array([z[0], z[1], 0.0, 0.0])
            P0 = np.diag([self.initial_position_var, self.initial_position_var, self.initial_velocity_var, self.initial_velocity_var,])
            self._filters[track_id] = {'x': x0, 'P': P0, 'last_track_state_id': cache_key, 'last_nis': None,}
            return None

        # Build the State-Transition, Measurement, Process-Noise & Measurement-Noise Matrices.
        F, H, Q, R = self._build_matrices(self.process_noise, self.measurement_noise)

        # Predict. Always Advances Regardless of Whether this Frame is Detected or Coasted.
        x_pred = F @ filt['x']
        P_pred = F @ filt['P'] @ F.T + Q

        # Coasted Frames: No Genuine Measurement Exists to Update Against. Persist the 
        # Predicted (Not Updated) State & Report No NIS.
        if source != 'DETECTED':
            filt['x'] = x_pred
            filt['P'] = P_pred
            filt['last_track_state_id'] = cache_key
            filt['last_nis'] = None
            return None

        # Innovation & its Covariance.
        innovation = z - H @ x_pred
        S = H @ P_pred @ H.T + R

        # Normalised Innovation Squared.
        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)

        # Update (Standard Kalman Gain Correction).
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ innovation
        P_upd = (np.eye(4) - K @ H) @ P_pred

        # Persist State for the Next Frame and Cache this Frame's Result.
        filt['x'] = x_upd
        filt['P'] = P_upd
        filt['last_track_state_id'] = cache_key
        filt['last_nis'] = nis

        # Return the NIS Score.
        return nis


    def should_escalate(self, track_state: Dict) -> bool:
        # No NIS Available Yet (First Observation for this Track).
        nis = self._step(track_state)
        if nis is None:
            return False

        # Escalate Once the Observed Position Becomes Statistically Inconsistent.
        return nis > self.chi2_threshold


    def get_score_and_threshold(self, track_state: Dict):
        """
        Reports the Track's Current-Frame NIS Against the Chi-Squared Critical
        Value it is Compared Against.
        """
        # Use the Memoised Filter Step so this Matches the Same Frame without 
        # Advancing the Filter Independently.
        nis = self._step(track_state)

        # Return the Score Alongside the Threshold Used for Comparison.
        return nis, self.chi2_threshold
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       REGISTRY OF AVAILABLE MODULES
# ------------------------------------------------------------------------------
INTROSPECTION_MODULE_MAP = {
    0: NullIntrospectionModule,                    # No Introspection.
    1: ConfidenceThresholdIntrospectionModule,     # Simple Confidence Threshold.
    2: RollingConfidenceTrendIntrospectionModule,  # Rolling Confidence Trend / EWMA.
    3: CoastCountIntrospectionModule,              # Consecutive-Coast-Count Threshold.
    4: KalmanNISIntrospectionModule,               # Kalman Filter NIS.
}
# ------------------------------------------------------------------------------
