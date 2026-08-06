# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: introspection_module_evaluator.py %
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

    def __init__(self, threshold: float = 0.70, window: int = 1):
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

    def __init__(self, threshold: int = 2):
        # Number of Consecutive Coasted Frames Required Before Escalation.
        self.threshold = threshold


    def should_escalate(self, track_state: Dict) -> bool:
        # Coast Count is Already Maintained by the T2D Tracker Itself. Read 
        # Directly, No History Reconstruction Needed.
        coast_count = track_state.get('coast_count', 0)

        # Return True Once the Number of Consecutive Coasted Frames Meets or
        # Exceeds the Configured Escalation Threshold.
        return coast_count >= self.threshold
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       REGISTRY OF AVAILABLE MODULES
# ------------------------------------------------------------------------------
INTROSPECTION_MODULE_MAP = {
    0: NullIntrospectionModule,                    # No Introspection.
    1: ConfidenceThresholdIntrospectionModule,     # Simple Confidence Threshold.
    2: RollingConfidenceTrendIntrospectionModule,  # Rolling Confidence Trend / EWMA.
    3: CoastCountIntrospectionModule,              # Consecutive-Coast-Count Threshold.
}
# ------------------------------------------------------------------------------
