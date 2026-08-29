# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: dataset_quality_metrics.py %
# Date Created:            %date_created: Sat Aug 29 16:18:24 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Model-agnostic dataset quality metrics for the corruption test set. All
# metrics are computed directly from image pixels and ground-truth bounding
# boxes, without running the detector, tracker, or any introspection module.
# This provides a consistent, model-independent measure of dataset quality and
# corruption severity. The metrics characterise the dataset itself rather than
# the performance of any particular model. Individual metric definitions and 
# their rationale are documented in the corresponding function docstrings below.
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

import cv2
import numpy as np

from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Exclude Pixels Immediately Surrounding the Target from Background Statistics.
GUARD_BAND_PX = 5

# Background Region Size Relative to the Target Bounding Box.
BACKGROUND_MARGIN_RATIO = 2.0

# Minimum Background Standard Deviation Used when Computing SCR.
MIN_BACKGROUND_STD = 1.0

# Minimum Valid SCR Coverage Required for Reliable Ramp Monotonicity.
MIN_RAMP_COVERAGE_FRACTION = 0.5
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                        BOUNDING BOX / IMAGE HELPERS
# ------------------------------------------------------------------------------
def resolveBoundingBox(bb: Optional[object]) -> Optional[Tuple[int, int, int, int]]:
    """
    Resolves a BoundingBox object into pixel corner coordinates.

    Args:
        bb: BoundingBox object, or None.

    Returns:
        The bounding box as (x1, y1, x2, y2), or None if invalid or incomplete.
    """
    # Guard Against a Missing BoundingBox.
    if bb is None:
        return None

    # Resolve Each Field, Accepting Either Naming Convention.
    left   = getattr(bb, 'left',   None) or getattr(bb, 'x', None)
    top    = getattr(bb, 'top',    None) or getattr(bb, 'y', None)
    width  = getattr(bb, 'width',  None) or getattr(bb, 'w', None)
    height = getattr(bb, 'height', None) or getattr(bb, 'h', None)

    # Bail Out if Any Required Field is Missing.
    if None in (left, top, width, height):
        return None

    # Convert to Integer Corner Coordinates.
    x1 = int(left)
    y1 = int(top)
    x2 = int(left + width)
    y2 = int(top + height)

    # Return the Resolved BBox.
    return (x1, y1, x2, y2)


def getPrimaryBoundingBox(frame: object) -> Optional[Tuple[int, int, int, int]]:
    """
    Selects the largest-area ground-truth bounding box in a frame.

    The largest detected object is used as the primary target for dataset
    quality metrics.

    Args:
        frame: AOT frame object containing detected object locations.

    Returns:
        The largest valid bounding box as (x1, y1, x2, y2), or None if no valid
        detections are present.
    """
    # Retrieve Detected Object Locations, Defaulting to Empty if Absent.
    object_locations = getattr(frame, 'detected_object_locations', {}) or {}

    # Track the Largest BBox Seen so Far.
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_area = -1.0

    # Scan Every Detected Object in the Frame.
    for obj in object_locations.values():
        # Resolve the Object's Bounding Box into Pixel Corner Coordinates.
        bbox = resolveBoundingBox(getattr(obj, 'bb', None))

        # Ignore Objects with Missing or Invalid Bounding Boxes.
        if bbox is None:
            continue

        # Calculate the Bounding-Box Area to Identify the Primary Target.
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        # Keep This BBox if it's the Largest Seen so Far.
        if area > best_area:
            best_area = area
            best_bbox = bbox

    # Return the Largest Valid BBox, or None if Nothing Was Found.
    return best_bbox


def toGrayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to single-channel grayscale.

    Args:
        image: Input grayscale, BGR, or BGRA image.

    Returns:
        Single-channel grayscale image.
    """
    # Already Single-Channel. Return Unchanged.
    if image.ndim == 2:
        return image

    # Single-Channel Wrapped in a Trailing Dimension.
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    # Multi-Channel. Convert via OpenCV.
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       TARGET / BACKGROUND EXTRACTION
# ------------------------------------------------------------------------------
def _extractTargetAndBackground(gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extracts the target region and surrounding background mask.

    Args:
        gray: Single-channel grayscale image.
        bbox: Target bounding box as (x1, y1, x2, y2).

    Returns:
        The target region, surrounding background window, and background mask, or
        None if the bounding box is invalid.
    """
    # Get the Image Dimensions & Unpack the Target Bounding-Box Coordinates.
    h_img, w_img = gray.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clip the Target BBox to the Image Bounds.
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    # Bail Out if the Clipped BBox is Degenerate.
    if x2 <= x1 or y2 <= y1:
        return None

    # Extract the Target Region.
    target_region = gray[y1:y2, x1:x2]

    # Compute the Outer (Background) Window Bounds, Scaled to BBox Size.
    bb_w, bb_h = x2 - x1, y2 - y1
    margin_x = int(bb_w * BACKGROUND_MARGIN_RATIO) + GUARD_BAND_PX
    margin_y = int(bb_h * BACKGROUND_MARGIN_RATIO) + GUARD_BAND_PX

    ox1 = max(0, x1 - margin_x)
    oy1 = max(0, y1 - margin_y)
    ox2 = min(w_img, x2 + margin_x)
    oy2 = min(h_img, y2 + margin_y)

    outer_window = gray[oy1:oy2, ox1:ox2]
    if outer_window.size == 0:
        return None

    # Compute the Guard Band Bounds (Target + Buffer), Relative to Outer Window.
    gx1 = max(0, x1 - GUARD_BAND_PX)
    gy1 = max(0, y1 - GUARD_BAND_PX)
    gx2 = min(w_img, x2 + GUARD_BAND_PX)
    gy2 = min(h_img, y2 + GUARD_BAND_PX)

    rel_gx1 = max(0, gx1 - ox1)
    rel_gy1 = max(0, gy1 - oy1)
    rel_gx2 = min(outer_window.shape[1], gx2 - ox1)
    rel_gy2 = min(outer_window.shape[0], gy2 - oy1)

    # Build the Background Mask. True Everywhere Except the Guard Band.
    background_mask = np.ones(outer_window.shape, dtype=bool)
    background_mask[rel_gy1:rel_gy2, rel_gx1:rel_gx2] = False

    # Return the Extracted Regions.
    return target_region, outer_window, background_mask
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#             METRIC 1 -- LOCAL SIGNAL-TO-CLUTTER RATIO (SCR)
# ------------------------------------------------------------------------------
def computeLocalSCR(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
    """
    Computes local Signal-to-Clutter Ratio for the target region.

    SCR measures how distinct the target is from its immediate background using
    the difference between their mean intensities relative to background
    variation. Higher values indicate a more visually separable target, while
    lower values indicate reduced local detectability. The metric is
    model-agnostic and is computed directly from image statistics.

    Args:
        image (np.ndarray): Frame to evaluate (grayscale or BGR).
        bbox (Tuple[int, int, int, int]): Target (x1, y1, x2, y2).

    Returns:
        The SCR value, or None if the target/background regions cannot be
        extracted.
    """
    # Convert to Grayscale for Consistent Intensity Statistics.
    gray = toGrayscale(image)

    # Extract the Target and Background Regions.
    extracted = _extractTargetAndBackground(gray, bbox)
    if extracted is None:
        return None
    target_region, outer_window, background_mask = extracted

    # Gather the Background Pixels via the Mask.
    background_pixels = outer_window[background_mask]
    if background_pixels.size < 10:
        return None

    # Compute Target & Background Statistics.
    target_mean = float(np.mean(target_region))
    background_mean = float(np.mean(background_pixels))
    background_std = max(float(np.std(background_pixels)), MIN_BACKGROUND_STD)

    # Return the Local SCR.
    return abs(target_mean - background_mean) / background_std
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#             METRIC 2 -- LOCAL BACKGROUND COMPLEXITY (CLUTTER)
# ------------------------------------------------------------------------------
def computeBackgroundComplexity(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
    """
    Computes local texture/edge energy in the background annulus around the
    target, via the variance of the Laplacian.

    Background complexity is measured using the variance of the Laplacian on
    the clean frame. Higher values indicate a more textured or structurally
    complex background, helping distinguish target signal loss from difficulty
    caused by naturally cluttered scenes. The metric is independent of any
    corruption applied later.

    Args:
        image: Clean (uncorrupted) frame to evaluate.
        bbox: Target bounding box as (x1, y1, x2, y2).

    Returns:
        The background Laplacian variance, or None if the target/background
        regions cannot be extracted.
    """
    # Convert to Grayscale for Consistent Edge Statistics.
    gray = toGrayscale(image)

    # Extract the Target and Background Regions.
    extracted = _extractTargetAndBackground(gray, bbox)
    if extracted is None:
        return None
    _, outer_window, background_mask = extracted

    # Compute the Laplacian Across the Full Outer Window, then Mask to Background.
    laplacian = cv2.Laplacian(outer_window, cv2.CV_64F)
    background_laplacian = laplacian[background_mask]

    if background_laplacian.size < 10:
        return None

    # Return the Variance of the Background Laplacian.
    return float(np.var(background_laplacian))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#               METRIC 3 -- CROSS-CORRUPTION CALIBRATION (PSNR)
# ------------------------------------------------------------------------------
def computePSNR(reference_image: np.ndarray, test_image: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio between a clean reference frame and
    its corrupted counterpart.

    PSNR measures the overall pixel-level degradation relative to the clean
    frame. Higher values indicate less degradation, while lower values
    indicate greater degradation. It provides a common reference for checking
    whether severity levels produce comparable overall degradation across
    different corruption types.

    Args:
        reference_image: Clean (uncorrupted) frame.
        test_image: Corrupted frame to compare against it.

    Returns:
        PSNR in dB, or float('inf') for pixel-identical images.
    """
    # Convert Both Images to Grayscale for a Consistent Comparison.
    ref = toGrayscale(reference_image).astype(np.float64)
    test = toGrayscale(test_image).astype(np.float64)

    # Compute Mean Squared Error Between the Two Frames.
    mse = float(np.mean((ref - test) ** 2))

    # Guard Against a Zero-MSE (Identical Frame) Case.
    if mse < 1e-10:
        return float('inf')

    # Return the Standard PSNR Formula for 8-bit Imagery.
    return 20.0 * np.log10(255.0 / np.sqrt(mse))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   METRIC 4 -- RAMP MONOTONICITY / SMOOTHNESS
# ------------------------------------------------------------------------------
def computeRampCoverage(scr_curve: List[Optional[float]], onset_frame: int, ramp_length: int) -> Optional[float]:
    """
    Computes the fraction of the ramp window with a valid SCR reading.

    Ramp coverage indicates how much of the ramp contains usable SCR data.
    Higher coverage means the corresponding scr_monotonicity value is more
    reliable, while low coverage can make monotonicity sensitive to missing or
    clustered target detections.

    Args:
        scr_curve: Per-frame SCR values for the full flight.
        onset_frame: Frame index where the ramp begins.
        ramp_length: Number of frames in the ramp.

    Returns:
        The fraction of valid SCR readings in [0, 1], or None if the ramp
        window is empty or outside the available frame range.
    """
    # Bound the Ramp Window to the Curve's Actual Length.
    ramp_end = min(onset_frame + ramp_length, len(scr_curve))
    window_length = ramp_end - onset_frame

    # Guard Against a Degenerate Window.
    if window_length <= 0:
        return None

    # Count Frames within the Window that Have a Valid SCR Reading.
    valid_count = sum(1 for v in scr_curve[onset_frame:ramp_end] if v is not None)

    # Return the Coverage Fraction.
    return valid_count / float(window_length)


def computeRampMonotonicity(scr_curve: List[Optional[float]], onset_frame: int, ramp_length: int) -> Optional[float]:
    """
    Measures how cleanly the target's SCR decreases across the ramp window.

    The metric is the Pearson correlation between frame index and SCR, with
    valid readings restricted to the ramp window. Values near -1 indicate a
    smooth decline in SCR as severity increases, while values near 0 indicate
    a less consistent degradation. The metric is only computed when sufficient
    SCR coverage is available; see computeRampCoverage.

    Args:
        scr_curve: Per-frame SCR values for the full flight, with None where
                   no valid target bounding box was available.
        onset_frame: Frame index where the ramp begins.
        ramp_length: Number of frames in the ramp.

    Returns:
        The Pearson correlation coefficient in [-1, 1], or None if fewer than
        three valid SCR values are available or ramp coverage is insufficient.
    """
    # Check Ramp Coverage First. Below Threshold, the Value isn't Trustworthy.
    coverage = computeRampCoverage(scr_curve, onset_frame, ramp_length)
    if coverage is None or coverage < MIN_RAMP_COVERAGE_FRACTION:
        return None

    # Collect Valid (Frame Index, SCR) Pairs within the Ramp Window.
    ramp_end = min(onset_frame + ramp_length, len(scr_curve))
    indices, values = [], []

    for i in range(onset_frame, ramp_end):
        if scr_curve[i] is not None:
            indices.append(i)
            values.append(scr_curve[i])

    # Need at Least 3 Points for a Meaningful Correlation.
    if len(values) < 3:
        return None

    # Return the Pearson Correlation Between Frame Index and SCR.
    return float(np.corrcoef(indices, values)[0, 1])
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                METRIC 5 -- DATASET-WIDE COVERAGE / SPREAD
# ------------------------------------------------------------------------------
def _summariseValues(values: List[float]) -> Optional[Dict]:
    """
    Computes summary statistics for a list of numeric values.

    Args:
        values: Numeric values to summarise.

    Returns:
        Summary statistics, or None if the list is empty.
    """
    # Guard Against an Empty Input List.
    if not values:
        return None

    # Compute Summary Statistics via NumPy.
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "min":   float(np.min(arr)),
        "p25":   float(np.percentile(arr, 25)),
        "p50":   float(np.percentile(arr, 50)),
        "p75":   float(np.percentile(arr, 75)),
        "max":   float(np.max(arr)),
    }


def summariseDatasetQuality(manifest_entries: List[Dict]) -> Dict:
    """
    Summarises dataset-wide coverage and spread across all corrupted flights.

    The summary aggregates distribution statistics for the per-flight quality
    metrics across the corrupted dataset. It shows the overall range and spread
    of difficulty, helping identify whether the dataset provides a useful
    distribution of easy, medium, and hard cases rather than clustering around
    a single difficulty level.

    Args:
        manifest_entries: Per-flight manifest entries containing the available
            quality metrics.

    Returns:
        Summary statistics for each quality metric across the dataset.
    """
    # Pull Out Each Metric Across All Flights, Skipping Missing Values.
    terminal_scrs   = [e["terminal_scr"] for e in manifest_entries if e.get("terminal_scr") is not None]
    complexities    = [e["background_complexity"] for e in manifest_entries if e.get("background_complexity") is not None]
    monotonicities  = [e["scr_monotonicity"] for e in manifest_entries if e.get("scr_monotonicity") is not None]
    coverages       = [e["scr_ramp_coverage"] for e in manifest_entries if e.get("scr_ramp_coverage") is not None]
    onsets          = [e["onset_frame"] for e in manifest_entries if e.get("onset_frame") is not None]
    psnr_at_peaks   = [e["psnr_at_peak"] for e in manifest_entries if e.get("psnr_at_peak") not in (None, float('inf'))]

    # Return the Combined Summary.
    return {
        "terminal_scr":          _summariseValues(terminal_scrs),
        "background_complexity": _summariseValues(complexities),
        "scr_monotonicity":      _summariseValues(monotonicities),
        "scr_ramp_coverage":     _summariseValues(coverages),
        "onset_frame":           _summariseValues(onsets),
        "psnr_at_peak":          _summariseValues(psnr_at_peaks),
    }
# ------------------------------------------------------------------------------
