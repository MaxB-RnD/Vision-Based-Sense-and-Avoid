# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: severity_scheduler.py %
# Date Created:            %date_created: Wed Aug 12 17:55:34 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# FIUX MEEEEEEE
# Generates per-frame corruption severity schedules for a flight sequence.
#
# A schedule is a list of floats, one per frame, each in [0, 1], describing
# how strongly the corruption in corruptions.py should be applied to that
# frame. Frames before the onset frame are clean (severity 0.0). From the
# onset frame, severity ramps up according to the chosen shape until it
# reaches peak_severity, then holds at peak_severity for the remainder of the
# track. This is deliberately kept separate from the corruption functions
# themselves -- swapping ramp shape, onset frame, or peak severity never
# requires touching corruptions.py, and adding a new ramp SHAPE never
# requires touching a new corruption FUNCTION.
#
# NOTE ON THE TWO NOTIONS OF "ONSET":
#   The onset_frame parameter here is the DESIGNED onset -- the frame where
#   you started ramping up corruption. This is NOT the same as the Failure
#   Onset Frame defined in your metrics (the retrospective, threshold-based
#   definition computed from CEDR/FTPS after running the real pipeline). The
#   two won't line up exactly, because the detector's response to increasing
#   corruption is not guaranteed to be linear even if the corruption ramp is.
#   Log both in the manifest so you can compare them later -- that comparison
#   is itself a useful sanity check on the Failure Onset Frame methodology.
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

from typing import List

# ------------------------------------------------------------------------------
#                           RAMP SHAPE FUNCTIONS
# ------------------------------------------------------------------------------
def _linearRamp(num_frames: int, onset_frame: int, ramp_length: int,
                peak_severity: float) -> List[float]:
    """
    Builds a linear severity ramp: 0 before onset, linearly increasing across
    ramp_length frames, then held flat at peak_severity.

    Args:
        num_frames (int): Total number of frames in the track.
        onset_frame (int): Frame index where the ramp begins.
        ramp_length (int): Number of frames over which severity increases
                           from 0 to peak_severity.
        peak_severity (float): Severity value once the ramp completes.

    Returns:
        List[float]: Per-frame severity values, length num_frames.
    """
    # Build the Schedule Frame-by-Frame.
    schedule = []
    for i in range(num_frames):
        if i < onset_frame:
            # Before Onset -- Clean.
            s = 0.0
        elif ramp_length <= 0 or i >= onset_frame + ramp_length:
            # Ramp Complete (or Instantaneous Ramp) -- Hold at Peak.
            s = peak_severity
        else:
            # Within the Ramp -- Linearly Interpolate.
            s = peak_severity * (i - onset_frame) / float(ramp_length)
        schedule.append(s)

    # Return the Completed Schedule.
    return schedule


def _sigmoidRamp(num_frames: int, onset_frame: int, ramp_length: int,
                 peak_severity: float) -> List[float]:
    """
    Builds a sigmoid (S-curve) severity ramp, centred on the midpoint of the
    ramp window. Slower to start and finish than a linear ramp, steepest in
    the middle -- included as an extension point, not used by default.

    Args:
        num_frames (int): Total number of frames in the track.
        onset_frame (int): Frame index where the ramp begins.
        ramp_length (int): Width of the ramp window in frames.
        peak_severity (float): Severity value once the ramp completes.

    Returns:
        List[float]: Per-frame severity values, length num_frames.
    """
    # Import Locally to Keep the Module Dependency-Light for the Default Case.
    import math

    # Guard Against a Degenerate (Zero-Length) Ramp Window.
    if ramp_length <= 0:
        return _linearRamp(num_frames, onset_frame, ramp_length, peak_severity)

    # Centre & Steepness of the Sigmoid, Derived from the Ramp Window Size.
    midpoint = onset_frame + ramp_length / 2.0
    steepness = 8.0 / ramp_length  # Tuned so ~99% of the Transition Fits in ramp_length.

    # Build the Schedule Frame-by-Frame.
    schedule = []
    for i in range(num_frames):
        if i < onset_frame:
            # Before Onset -- Clean.
            s = 0.0
        else:
            # Sigmoid Curve Centred on the Ramp Midpoint.
            s = peak_severity / (1.0 + math.exp(-steepness * (i - midpoint)))
        schedule.append(s)

    # Return the Completed Schedule.
    return schedule
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               SHAPE REGISTRY
# ------------------------------------------------------------------------------
# Maps a Filesystem-Safe Shape Name to its Corresponding Ramp Function. Add New
# Shapes Here -- the Main Build Script Only Ever Looks Values up by Name.
RAMP_SHAPE_REGISTRY = {
    "linear":  _linearRamp,
    "sigmoid": _sigmoidRamp,
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           PUBLIC ENTRY POINT
# ------------------------------------------------------------------------------
def buildSeveritySchedule(num_frames: int, onset_frame: int, ramp_length: int,
                          peak_severity: float = 1.0, shape: str = "linear") -> List[float]:
    """
    Builds a per-frame severity schedule for a track.

    Args:
        num_frames (int): Total number of frames in the track.
        onset_frame (int): Frame index (0-indexed) where corruption begins.
        ramp_length (int): Number of frames over which severity ramps from
                           0 to peak_severity.
        peak_severity (float): Severity value once the ramp completes.
                               Defaults to 1.0 (worst-case corruption).
        shape (str): Ramp shape name, must exist in RAMP_SHAPE_REGISTRY.
                    Defaults to 'linear'.

    Returns:
        List[float]: Per-frame severity values, length num_frames.

    Raises:
        ValueError: If the requested shape is not in RAMP_SHAPE_REGISTRY, or
                   onset_frame falls outside the track.
    """
    # Validate the Requested Shape Exists.
    if shape not in RAMP_SHAPE_REGISTRY:
        raise ValueError(
            "Unknown Ramp Shape '%s'. Available: %s" % (shape, list(RAMP_SHAPE_REGISTRY.keys()))
        )

    # Validate the Onset Frame Falls Within the Track.
    if not (0 <= onset_frame < num_frames):
        raise ValueError(
            "onset_frame (%d) Must Fall Within [0, %d)." % (onset_frame, num_frames)
        )

    # Delegate to the Selected Ramp Function.
    ramp_fn = RAMP_SHAPE_REGISTRY[shape]
    return ramp_fn(num_frames, onset_frame, ramp_length, peak_severity)
# ------------------------------------------------------------------------------
