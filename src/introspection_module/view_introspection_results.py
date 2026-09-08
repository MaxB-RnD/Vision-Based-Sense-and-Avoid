# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: view_introspection_results.py %
# Date Created:            %date_created: Sun Aug 25 23:54:08 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 5 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Helper module for generating visualisations of introspection module results.
#
# Provides notebook-callable functionality for creating annotated videos of
# flight sequences, allowing predicted object locations, tracking behaviour,
# confidence information, and introspection warnings to be visually inspected
# across the duration of a flight.
#
# Also provides functionality for combining multiple introspection
# visualisations into a single side-by-side comparison video, supporting
# qualitative comparison of different introspection modules or results.
#
# The visualisations are intended to assist with the evaluation, interpretation,
# and debugging of introspection module behaviour by providing a clear visual
# representation of predictions and detected warnings over the corresponding
# flight imagery.
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
import json
import subprocess
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from logging_module import LogLevel, log
from src.data.corruption_dataset.dataset_quality_metrics import getPrimaryBoundingBox
from src.data.corruption_dataset.severity_scheduler import buildSeveritySchedule

# ------------------------------------------------------------------------------
#                        CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Paths & Environment Variables. 
DEFAULT_DATASET_ROOT = 'data/corruption_dataset'

# Video Settings.
VIDEO_FPS = 10

# Bounding Box (BGR).
BB_COLOUR_DETECTED = (0, 0, 211)     # Red.
BB_COLOUR_WARNING  = (211, 0, 211)   # Magenta.
BB_COLOUR_GT       = (0, 211, 0)     # Green.
BB_THICKNESS = 4

# Bounding Box Labels.
BB_LABEL_FONT_SCALE = 1.0
BB_LABEL_THICKNESS  = 2
BB_LABEL_LINE_GAP   = 12

# Status Banner.
BANNER_HEIGHT = 80
BANNER_FONT_SCALE = 1.0
BANNER_TEXT_THICKNESS = 2
BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_TEXT_COLOUR = (255, 255, 255)
BANNER_COLUMN_GAP = 20 
BANNER_COLOUR_WARNING    = (211, 0, 211)  # Magenta.
BANNER_COLOUR_DETECTION  = (0, 0, 211)    # Red.
BANNER_COLOUR_EMPTY      = (60, 60, 60)   # Grey.
BANNER_COLOUR_TRACK_LOST = (0, 121, 251)  # Orange.

# Fixed-Width Status Slot.
_STATUS_STRINGS = ["No Object", "TRACK LOST / GT TARGET PRESENT", "DETECTION", "INTROSPECTION WARNING"]
_STATUS_WIDTHS = [
    cv2.getTextSize(s, BANNER_FONT, BANNER_FONT_SCALE, BANNER_TEXT_THICKNESS)[0][0]
    for s in _STATUS_STRINGS
]
BANNER_STATUS_SLOT_WIDTH = max(_STATUS_WIDTHS) + 20

# Fixed-Width Severity Slot.
BANNER_SEVERITY_FORMAT = "|  Severity: %.2f  |"
BANNER_SEVERITY_SLOT_WIDTH = cv2.getTextSize(
    BANNER_SEVERITY_FORMAT % 0.0, BANNER_FONT, BANNER_FONT_SCALE, BANNER_TEXT_THICKNESS
)[0][0] + 20

# Warning Lead-Time Display (Top Left). 
AWLT_FONT = cv2.FONT_HERSHEY_SIMPLEX
AWLT_FONT_SCALE = 0.8
AWLT_TEXT_THICKNESS = 2
AWLT_TEXT_COLOUR = (255, 255, 255)
AWLT_BG_COLOUR = (60, 60, 60)
AWLT_MARGIN = 12
AWLT_LINE_GAP = 8

# True-Onset Marker Text Colour.
ONSET_TEXT_COLOUR = (0, 246, 255)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       FLIGHT & RESULTS RESOLUTION
# ------------------------------------------------------------------------------
def resolveFlightImagePaths(flight_id: str, dataset_root: str = DEFAULT_DATASET_ROOT) -> List[str]:
    """
    Finds every frame image on disk for the given flight, sorted in
    chronological order (image filenames are zero-padded numeric timestamps,
    so a plain alphabetical sort is chronological).

    Args:
        flight_id (str): Flight ID string, prefixed with its partition
                         (e.g. 'part2e9c4568409604a6183d24726c0ebf696').
        dataset_root (str): Root directory containing '{part}/Images/...'.

    Returns:
        List[str]: Sorted list of absolute/relative paths to this flight's
                   frame PNGs.

    Raises:
        FileNotFoundError: If no image directory / no PNGs can be found for
                           this flight under any of the candidate locations.
    """
    # Determine which Dataset Partition the Flight Belongs To.
    if flight_id.startswith('part1'):
        part, bare_id = 'part1', flight_id[len('part1'):]
    elif flight_id.startswith('part2'):
        part, bare_id = 'part2', flight_id[len('part2'):]
    elif flight_id.startswith('part3'):
        part, bare_id = 'part3', flight_id[len('part3'):]
    else:
        part, bare_id = 'part1', flight_id

    # Build the Possible Flight Image Directories.
    candidate_dirs = [
        os.path.join(dataset_root, part, 'Images', bare_id),
        os.path.join(dataset_root, part, 'Images', flight_id),
    ]

    # Search Each Candidate Directory for PNG Frames.
    for candidate_dir in candidate_dirs:
        # Sort Frames Chronologically by their Timestamp Filenames.
        image_paths = sorted(glob(os.path.join(candidate_dir, '*.png')))

        # Print Success Message if Images are Found.
        if image_paths:
            log(LogLevel.INFO, "introspection_visualiser:resolveFlightImagePaths",
                "Found %d Frames for Flight %s in '%s'.", len(image_paths), flight_id, candidate_dir)
            return image_paths

    # Raise an Error if no Flight Images Were Found.
    raise FileNotFoundError(
        "Could Not Find any Frame Images for Flight '%s' under '%s'. Tried: %s"
        % (flight_id, dataset_root, candidate_dirs)
    )


def loadIntrospectionResults(result_json_path: str) -> Tuple[Dict[str, List[dict]], Optional[int], Optional[float], Optional[str]]:
    """
    Loads a per-flight result.json and indexes it by img_name for fast lookup.

    Expected format is a flat list of frame entries:
        [{"img_name": "...png", "detections": [{"track_id", "x", "y", "w", "h",
                                                  "n", "s", "source", "warning",
                                                  "introspection_score",
                                                  "introspection_threshold",
                                                  "failure_onset_frame",
                                                  "failure_onset_img_name",
                                                  "warning_img_name",
                                                  "warning_lead_time_seconds"}]}, ...]

    NOTE: Use 'failure_onset_img_name' (returned separately below) to place the 
    onset marker against the dense raw frame list used for video generation.

    Args:
        result_json_path (str): Path to the flight's result.json file.

    Returns:
        A tuple containing the detections by image name, failure onset frame,
        warning lead time in seconds, and onset image name.
    """
    # Check that the Results File Exists.
    if not os.path.exists(result_json_path):
        log(LogLevel.WARNING, "introspection_visualiser:loadIntrospectionResults",
            "Results File Not Found: %s. All Frames will Show as 'No Object'.", result_json_path)
        return {}, None, None, None

    # Load the Raw Results from JSON.
    with open(result_json_path, 'r') as f:
        raw_entries = json.load(f)

    # Index Results by Image Name for Fast Frame Lookup.
    lookup: Dict[str, List[dict]] = {}

    # Process Each Frame Entry from the Results File.
    for entry in raw_entries:
        img_name = entry.get('img_name', '')
        lookup.setdefault(img_name, []).extend(entry.get('detections', []))

    # Start with No Failure Onset Information, Image Name, or Warning Lead Time.
    onset_frame: Optional[int] = None
    onset_img_name: Optional[str] = None
    warning_lead_time: Optional[float] = None

    # Check all Detections Across the Flight.
    for detections in lookup.values():
        # Check Each Detection in the Current Frame.
        for det in detections:
            # Get the Failure Onset Frame for this Detection.
            det_onset = det.get('failure_onset_frame')

            # Only Use Detections that have a Recorded Onset Frame.
            if det_onset is not None and (onset_frame is None or det_onset < onset_frame):
                # Store the Earliest Failure Onset Frame.
                onset_frame = det_onset

                # Store the Image Name for the Earliest Onset.
                onset_img_name = det.get('failure_onset_img_name')

                # Store the Warning Lead Time for the Same Detection.
                warning_lead_time = det.get('warning_lead_time_seconds')

    # Display the Number of Frames with Results.
    log(LogLevel.INFO, "introspection_visualiser:loadIntrospectionResults",
        "Loaded Results for %d Frames from '%s'. Onset Frame: %s (%s).",
        len(lookup), result_json_path, onset_frame, onset_img_name)

    # Return Lookup Dictionary, Plus the Derived Onset Frame, Lead Time & Onset Image Name.
    return lookup, onset_frame, warning_lead_time, onset_img_name


def loadGroundTruthForFlight(dataset, flight_id: str) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    """
    Loads the ground-truth target bbox for every frame of a flight, indexed
    by image filename to match the same key used for predicted results.

    Args:
        dataset: Loaded AOT Dataset object.
        flight_id (str): Full flight ID (e.g. 'part1abc123...').

    Returns:
        Dict[str, Optional[Tuple[int, int, int, int]]]: Mapping of image
            filename -> (x1, y1, x2, y2) ground-truth bbox, or None for
            frames with no ground-truth target.
    """
    # Load the Flight's Metadata & Frames.
    try:
        flight = dataset.get_flight(flight_id)
        frames = flight.frames
    except Exception as e:
        # Return an Empty Lookup if the Flight Cannot be Loaded.
        log(LogLevel.WARNING, "introspection_visualiser:loadGroundTruthForFlight",
            "Could Not Load Ground Truth for Flight %s: %s. GT Overlay will be Unavailable.",
            flight_id, str(e))
        return {}

    # Create a Lookup from Image fFlename to Ground-Truth Bbox.
    gt_lookup: Dict[str, Optional[Tuple[int, int, int, int]]] = {}

    # Process Each Frame in the Flight.
    for frame_id, frame in frames.items():
        # Use the Image Filename to Match the Prediction Results.
        try:
            image_filename = os.path.basename(frame.image_path())
        except Exception:
            # Fall Back to the Frame ID if the Image Path is Unavailable.
            image_filename = frame_id

        # Store the Ground-Truth Bbox for this Frame.
        gt_lookup[image_filename] = getPrimaryBoundingBox(frame)

    # Count the Frames that Contain a Ground-Truth Target.
    present_count = sum(1 for bbox in gt_lookup.values() if bbox is not None)

    # Log the Number of Frames with a Ground-Truth Target.
    log(LogLevel.INFO, "introspection_visualiser:loadGroundTruthForFlight",
        "Loaded Ground Truth for Flight %s: %d / %d Frames have a Target.",
        flight_id, present_count, len(gt_lookup))

    # Return the Completed Ground-Truth Lookup.
    return gt_lookup
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    ANNOTATION HELPERS (Boxes + Banner)
# ------------------------------------------------------------------------------
def _fitTextScale(text: str, font: int, max_width: int, base_scale: float,
                  thickness: int, min_scale: float = 0.45) -> Tuple[float, int]:
    """
    Shrinks the font scale for a piece of text until it fits within max_width, 
    so long strings ever run off the edge of the frame instead of silently being
    clipped by OpenCV.

    Args:
        text (str): Text to be measured/drawn.
        font (int): OpenCV font face.
        max_width (int): Maximum allowed pixel width for the rendered text.
        base_scale (float): Preferred/starting font scale.
        thickness (int): Text thickness used for measurement.
        min_scale (float): Floor below which the scale will not shrink further.

    Returns:
        Tuple[float, int]: (scale to use, rendered width at that scale).
    """
    # Start at the Preferred Scale and Measure the Text.
    scale = base_scale
    (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)

    # Shrink the Scale in Small Steps Until the Text Fits, or the Floor is Hit.
    while text_width > max_width and scale > min_scale:
        scale = round(scale - 0.05, 2)
        (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)

    # Return the Final Scale and its Rendered Width.
    return scale, text_width


def drawPredictedBoxes(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Draws predicted bounding boxes and labels for all detections in a frame.

    Bounding box coordinates are read from the result fields 'x', 'y', 'w',
    and 'h', which contain the top-left and bottom-right coordinates
    respectively. Box colour indicates the detection status, with warnings
    shown in the warning colour.

    Args:
        image (np.ndarray): OpenCV image array to draw onto (modified in place).
        detections (List[dict]): This frame's list of detection dicts.

    Returns:
        np.ndarray: Annotated image.
    """
    # Set the Font Used for Box Labels.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw Each Predicted Detection.
    for det in detections:
        # Read the Bounding Box Coordinates.
        x1, y1, x2, y2 = det.get('x'), det.get('y'), det.get('w'), det.get('h')

        # Skip Detections with Missing Box Coordinates.
        if None in (x1, y1, x2, y2):
            continue

        # Convert Coordinates to Integers for OpenCV.
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Read the Warning Status.
        warning = bool(det.get('warning', False))

        # Set the Box Colour Based on the Warning Status.
        colour = BB_COLOUR_WARNING if warning else BB_COLOUR_DETECTED

        # Draw the Bounding Box.
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, BB_THICKNESS)

        # Read the Track ID.
        track_id = det.get('track_id', '?')

        # Read the Continuous Introspection Score for this Track/Frame.
        introspection_score = det.get('introspection_score', None)
        introspection_str = ("%.2f" % introspection_score) if introspection_score is not None else "N/A"

        # Read the Alarm Threshold the Introspection Score is Compared Against.
        introspection_threshold = det.get('introspection_threshold', None)
        threshold_str = ("%.2f" % introspection_threshold) if introspection_threshold is not None else "N/A"

        # Create the Three-Line Box Label.
        line1 = "ID: " + str(track_id)
        line2 = "Score: " + introspection_str
        line3 = "Threshold: " + threshold_str

        # Calculate the Label Dimensions.
        (w1, h1), _ = cv2.getTextSize(line1, font, BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)
        (w2, h2), _ = cv2.getTextSize(line2, font, BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)
        (w3, h3), _ = cv2.getTextSize(line3, font, BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)

        # Position the Label Below the Bounding Box.
        label_x = x1
        line1_y = y2 + h1 + 4
        line2_y = line1_y + h2 + BB_LABEL_LINE_GAP
        line3_y = line2_y + h3 + BB_LABEL_LINE_GAP

        # Draw the Label Background, Sized to Fit All Three Lines.
        bg_x1, bg_y1 = label_x - 2, y2 + 2
        bg_x2, bg_y2 = label_x + max(w1, w2, w3) + 4, line3_y + 4
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), colour, cv2.FILLED)

        # Draw the Track ID, Introspection Score, and Alarm Threshold.
        cv2.putText(image, line1, (label_x, line1_y), font, BB_LABEL_FONT_SCALE,
                    (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)
        cv2.putText(image, line2, (label_x, line2_y), font, BB_LABEL_FONT_SCALE,
                    (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)
        cv2.putText(image, line3, (label_x, line3_y), font, BB_LABEL_FONT_SCALE,
                    (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)

    # Return the Marked-Up Frame. 
    return image


def drawGroundTruthBox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Draws the ground-truth target bounding box, in a colour distinct from any
    predicted box, so ground truth and prediction can be visually compared at
    a glance.

    Args:
        image (np.ndarray): OpenCV image array to draw onto (modified in place).
        bbox (Tuple[int, int, int, int]): Ground-truth (x1, y1, x2, y2).

    Returns:
        np.ndarray: Annotated image.
    """
    # Unpack & Draw the Ground-Truth Box.
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), BB_COLOUR_GT, BB_THICKNESS)

    # Label the Box so it isn't Confused with a Prediction.
    label = "GT"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)
    label_x, label_y = x1, max(y1 - 8, label_h + 4)

    # Draw the Label Text & Background.
    cv2.rectangle(image, (label_x - 2, label_y - label_h - 4),
                 (label_x + label_w + 4, label_y + 4), BB_COLOUR_GT, cv2.FILLED)
    cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
               BB_LABEL_FONT_SCALE, (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)

    # Return the Marked-Up Frame.
    return image


def addWarningLeadTimeOverlay(image: np.ndarray, warning_lead_time: Optional[float]) -> np.ndarray:
    """
    Draws the Warning Lead Time in a grey box, top-left corner of the frame.

    Args:
        image (np.ndarray): OpenCV image to annotate.
        warning_lead_time (Optional[float]): Warning lead time in seconds for 
                                             this flight/run.

    Returns:
        np.ndarray: Annotated image.
    """
    # Build the Display Text.
    text = ("Warning Lead Time: %.2fs" % warning_lead_time) if warning_lead_time is not None \
        else "Warning Lead Time: N/A"

    # Measure the Text to Size its Background Plate.
    (text_w, text_h), baseline = cv2.getTextSize(text, AWLT_FONT, AWLT_FONT_SCALE, AWLT_TEXT_THICKNESS)

    # Position in the Top-Left Corner, With a Small Margin.
    x1, y1 = AWLT_MARGIN, AWLT_MARGIN
    x2 = x1 + text_w + 2 * AWLT_MARGIN
    y2 = y1 + text_h + baseline + 2 * AWLT_MARGIN

    # Draw a Solid Background Plate so the Text Stays Legible Over any Underlying Imagery.
    cv2.rectangle(image, (x1, y1), (x2, y2), AWLT_BG_COLOUR, cv2.FILLED)

    # Draw the Text.
    text_x, text_y = x1 + AWLT_MARGIN, y1 + AWLT_MARGIN + text_h
    cv2.putText(image, text, (text_x, text_y), AWLT_FONT, AWLT_FONT_SCALE,
                AWLT_TEXT_COLOUR, AWLT_TEXT_THICKNESS, cv2.LINE_AA)

    # Return the Marked-Up Frame.
    return image


def addOnsetOverlay(image: np.ndarray) -> np.ndarray:
    """
    Draws a "TRUE FAILURE ONSET FRAME" box, top-right corner, in the same
    green used for the ground-truth box. 
    
    Only ever called from the true onset frame onwardand persists for the rest 
    of the video once shown.

    Args:
        image (np.ndarray): OpenCV image to annotate.

    Returns:
        np.ndarray: Annotated image.
    """
    # Build the Display Text.
    text = "TRUE FAILURE ONSET FRAME"

    # Measure the Text to Size its Background Plate.
    (text_w, text_h), baseline = cv2.getTextSize(text, AWLT_FONT, AWLT_FONT_SCALE, AWLT_TEXT_THICKNESS)

    # Position in the Top-Right Corner, With a Small Margin.
    h, w = image.shape[:2]
    x2, y1 = w - AWLT_MARGIN, AWLT_MARGIN
    x1 = x2 - text_w - 2 * AWLT_MARGIN
    y2 = y1 + text_h + baseline + 2 * AWLT_MARGIN

    # Draw a Solid Background Plate.
    cv2.rectangle(image, (x1, y1), (x2, y2), AWLT_BG_COLOUR, cv2.FILLED)

    # Draw the Text in Ground-Truth Green.
    text_x, text_y = x1 + AWLT_MARGIN, y1 + AWLT_MARGIN + text_h
    cv2.putText(image, text, (text_x, text_y), AWLT_FONT, AWLT_FONT_SCALE,
                ONSET_TEXT_COLOUR, AWLT_TEXT_THICKNESS, cv2.LINE_AA)

    # Return the Marked-Up Frame.
    return image


def addIntrospectionBanner(image: np.ndarray, frame_num: int, total_frames: int,
                           detections: List[dict], flight_id: str, module_name: str,
                           gt_present: bool = False, severity: Optional[float] = None) -> np.ndarray:
    """
    Adds a coloured status banner to the bottom of the image: grey for no
    ground-truth target (nothing to detect), orange for a ground-truth
    target present but no prediction (a genuine tracking failure), red for
    an active prediction (detected or coasted), and magenta for an 
    introspection warning.

    Args:
        image (np.ndarray): OpenCV image to annotate.
        frame_num (int): Current frame number (1-indexed for display).
        total_frames (int): Total number of frames in the flight.
        detections (List[dict]): This frame's list of detection dicts.
        flight_id (str): Flight ID string for display.
        module_name (str): Name of the introspection module.
        gt_present (bool): Whether a ground-truth target exists in this frame.
        severity (Optional[float]): This frame's corruption severity in [0, 1], 
                                    if available. 

    Returns:
        np.ndarray: Image with banner appended at the bottom.
    """
    # Get the Image Dimensions.
    h, w = image.shape[:2]

    # Create the Empty Banner.
    banner = np.zeros((BANNER_HEIGHT, w, 3), dtype=np.uint8)

    # Check the Detection Status for the Frame.
    has_detection = bool(detections)
    has_warning = any(bool(d.get('warning', False)) for d in detections)

    # Set the Banner Colour and Status.
    if has_warning:
        colour, status_str = BANNER_COLOUR_WARNING, "INTROSPECTION WARNING"
    elif has_detection:
        colour, status_str = BANNER_COLOUR_DETECTION, "DETECTION"
    elif gt_present:
        colour, status_str = BANNER_COLOUR_TRACK_LOST, "TRACK LOST / GT TARGET PRESENT"
    else:
        colour, status_str = BANNER_COLOUR_EMPTY, "No Object"

    # Apply the Status Colour to the Banner.
    banner[:] = colour

    # Baseline Y Position for All Banner Text.
    baseline_y = 50

    # Column 1: Frame Counter.
    digit_width = len(str(total_frames))
    frame_text = "Frame %0*d / %d  |" % (digit_width, frame_num, total_frames)
    frame_col_x = 20
    cv2.putText(banner, frame_text, (frame_col_x, baseline_y), BANNER_FONT,
               BANNER_FONT_SCALE, BANNER_TEXT_COLOUR, BANNER_TEXT_THICKNESS, cv2.LINE_AA)
    frame_text_width = cv2.getTextSize(frame_text, BANNER_FONT, BANNER_FONT_SCALE,
                                       BANNER_TEXT_THICKNESS)[0][0]

    # Column 2: Status.
    status_slot_x = frame_col_x + frame_text_width + BANNER_COLUMN_GAP
    status_text_width = cv2.getTextSize(status_str, BANNER_FONT, BANNER_FONT_SCALE,
                                        BANNER_TEXT_THICKNESS)[0][0]
    status_x = status_slot_x + max(0, (BANNER_STATUS_SLOT_WIDTH - status_text_width) // 2)
    cv2.putText(banner, status_str, (status_x, baseline_y), BANNER_FONT,
               BANNER_FONT_SCALE, BANNER_TEXT_COLOUR, BANNER_TEXT_THICKNESS, cv2.LINE_AA)

    # Column 3: Severity.
    severity_slot_x = status_slot_x + BANNER_STATUS_SLOT_WIDTH + BANNER_COLUMN_GAP
    if severity is not None:
        severity_text = BANNER_SEVERITY_FORMAT % severity
        cv2.putText(banner, severity_text, (severity_slot_x, baseline_y), BANNER_FONT,
                   BANNER_FONT_SCALE, BANNER_TEXT_COLOUR, BANNER_TEXT_THICKNESS, cv2.LINE_AA)

    # Column 4: Module Name & Flight ID.
    trailer_x = severity_slot_x + BANNER_SEVERITY_SLOT_WIDTH + BANNER_COLUMN_GAP
    trailer_text = "%s  |  %s" % (module_name, flight_id)
    trailer_max_width = max(w - trailer_x - 10, 10)
    trailer_scale, _ = _fitTextScale(trailer_text, BANNER_FONT, trailer_max_width,
                                     BANNER_FONT_SCALE, BANNER_TEXT_THICKNESS)
    cv2.putText(banner, trailer_text, (trailer_x, baseline_y), BANNER_FONT,
                trailer_scale, (255, 255, 255), BANNER_TEXT_THICKNESS, cv2.LINE_AA)

    # Append the Banner to the Bottom of the Image.
    return np.vstack([image, banner])
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           VIDEO GENERATOR
# ------------------------------------------------------------------------------
def generateIntrospectionVideo(flight_id: str, result_json_path: str, module_name: str,
                               output_path: str, dataset_root: str = DEFAULT_DATASET_ROOT,
                               video_fps: int = VIDEO_FPS, dataset=None,
                               manifest_entry: Optional[Dict] = None,
                               warning_lead_time: Optional[float] = None,
                               onset_frame: Optional[int] = None,
                               onset_img_name: Optional[str] = None) -> None:
    """
    Generates an annotated video covering every frame of a flight, with the
    introspection module's predicted results overlaid where available.

    Args:
        flight_id (str): Flight ID string, prefixed with its partition
                         (e.g. 'part2e9c4568409604a6183d24726c0ebf696').
        result_json_path (str): Path to this flight's result.json.
        module_name (str): Name of the introspection module, shown in the banner.
        output_path (str): Full path (including filename) to save the .mp4 to.
        dataset_root (str): Root directory containing '{part}/Images/...'.
                            Defaults to DEFAULT_DATASET_ROOT.
        video_fps (int): Output video frame rate. Defaults to VIDEO_FPS.
        dataset: Optional loaded AOT Dataset object.
        manifest_entry (Optional[Dict]): This flight's entry from corruption_manifest.json.
        warning_lead_time (Optional[float]): Explicit override for this flight's warning 
                                             lead time in seconds. Normally left as None,
                                             in which case it is read directly from
                                             result.json (see loadIntrospectionResults()).
        onset_frame (Optional[int]): Explicit override for the 1-indexed frame number of
                                     the true failure onset for this flight.
        onset_img_name (Optional[str]): Explicit override for the filename of the true
                                        failure onset frame.

    Returns:
        None:
    """
    # Locate Every Frame Image for the Flight.
    image_paths = resolveFlightImagePaths(flight_id, dataset_root)
    total_frames = len(image_paths)

    # Load the Predicted Results for the Flight, Including the Onset Frame & Warning Lead Time.
    results_lookup, results_onset_frame, results_warning_lead_time, results_onset_img_name = loadIntrospectionResults(result_json_path)
    if onset_frame is None:
        onset_frame = results_onset_frame
    if onset_img_name is None:
        onset_img_name = results_onset_img_name
    if warning_lead_time is None:
        warning_lead_time = results_warning_lead_time

    # Find the Onset Frame in the Dense Raw Image List.
    onset_dense_idx: Optional[int] = None

    # Only Look for the Dense Frame when an Onset Image is Available.
    if onset_img_name:
        # Map Each Raw Image Name to its Dense Frame Index.
        img_name_to_dense_idx = {os.path.basename(p): i for i, p in enumerate(image_paths)}

        # Find the Dense Index for the Onset Image.
        onset_dense_idx = img_name_to_dense_idx.get(onset_img_name)

        # Warn if the Onset Image Cannot Be Found in the Raw Frames.
        if onset_dense_idx is None:
            log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
                "Onset Image '%s' Not Found Among this Flight's Raw Frames -- "
                "Falling Back to the (Potentially Misaligned) Sparse Onset Frame Index.",
                onset_img_name)

    # Load Ground Truth, if a Dataset was Supplied.
    gt_lookup: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
    if dataset is not None:
        gt_lookup = loadGroundTruthForFlight(dataset, flight_id)

    # Build the Per-Frame Severity Schedule, if a Manifest Entry was Supplied.
    severity_schedule: Optional[List[float]] = None
    if manifest_entry is not None:
        # Generate the Severity Values Across all Frames.
        try:
            severity_schedule = buildSeveritySchedule(
                num_frames=total_frames,
                onset_frame=manifest_entry['onset_frame'],
                ramp_length=manifest_entry['ramp_length'],
                peak_severity=manifest_entry.get('peak_severity', 1.0),
                shape=manifest_entry.get('ramp_shape', 'linear'),
            )
        except Exception as e:
            # Continue Without Severity Values if the Schedule Cannot be Built.
            log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
                "Could Not Build Severity Schedule for Flight %s: %s. Severity will not be Shown.",
                flight_id, str(e))

    # Ensure the Output Directory Exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Print Process Commenced Message. 
    log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
        "Generating '%s' Introspection Video for Flight %s  (%d Frames).",
        module_name, flight_id, total_frames)

    # Initialise Variables & Counters.
    writer = None
    frames_written = 0

    # Process Every Frame in the Flight.
    for frame_idx, image_path in enumerate(image_paths):
        # Load the Current Frame.
        image = cv2.imread(image_path)

        # Print Warning if Frame Not Found.
        if image is None:
            log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
                "Could Not Load Image '%s'. Skipping.", image_path)
            continue

        # Look Up Predictions, Ground Truth, and Severity for the Current Frame.
        img_name = os.path.basename(image_path)
        detections = results_lookup.get(img_name, [])
        gt_bbox = gt_lookup.get(img_name)
        severity = severity_schedule[frame_idx] if severity_schedule is not None else None

        # Initialise the VideoWriter from the First Valid Frame.
        if writer is None:
            h, w = image.shape[:2]
            video_h = h + BANNER_HEIGHT
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (w, video_h))

            # Print Error Message if Cannot Write to Video.
            if not writer.isOpened():
                log(LogLevel.ERROR, "introspection_visualiser:generateIntrospectionVideo",
                    "Failed to Open VideoWriter for '%s'.", output_path)
                return

            # Print Progress Message.
            log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
                "Writing %d Frames at %dx%d  %d fps.", total_frames, w, h, video_fps)

        # Draw the Ground-Truth Box First, so Predicted Boxes are Drawn on Top.
        if gt_bbox is not None:
            image = drawGroundTruthBox(image, gt_bbox)

        # Draw Predicted Bounding Boxes when Detections are Present.
        if detections:
            image = drawPredictedBoxes(image, detections)

        # Add the Top-Left Warning Lead Time Overlay.
        image = addWarningLeadTimeOverlay(image, warning_lead_time)

        # Add the Top-Right True-Onset Overlay, from the Onset Frame Onward.
        if onset_dense_idx is not None:
            if frame_idx >= onset_dense_idx:
                image = addOnsetOverlay(image)
        elif onset_frame is not None and (frame_idx + 1) >= onset_frame:
            image = addOnsetOverlay(image)

        # Add the Introspection Status Banner.
        image = addIntrospectionBanner(image, frame_idx + 1, total_frames,
                                       detections, flight_id, module_name,
                                       gt_present=(gt_bbox is not None), severity=severity)

        # Write the Annotated Frame to the Video.
        writer.write(image)
        frames_written += 1

    # Stop if No Frames Could Be Written.
    if writer is None:
        log(LogLevel.ERROR, "introspection_visualiser:generateIntrospectionVideo",
            "No Frames Could Be Loaded for Flight '%s'.", flight_id)
        return

    # Finalise the Raw Video.
    writer.release()
    log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
        "Raw Video Written (%d / %d Frames). Re-encoding to H.264.", frames_written, total_frames)

    # Warn if Any Frames Could Not Be Written.
    if frames_written < total_frames:
        log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
            "%d Frame(s) were Skipped due to Image Load Failures — Video is Incomplete.",
            total_frames - frames_written)

    # Re-Encode to H.264 for Compatibility.
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    result = subprocess.run(
        ["ffmpeg",
            "-y",
            "-i", output_path,
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "23",
            h264_path,
            "-loglevel", "error",
        ],
        check=False,
    )
    ret = result.returncode

    # Replace the Raw Video with the H.264 Version.
    if ret == 0 and os.path.exists(h264_path):
        os.remove(output_path)
        os.rename(h264_path, output_path)
        log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
            "Video Re-encoded to H.264 and Saved to '%s'.", output_path)

    # Keep the Raw Video if Re-Encoding Fails.
    else:
        log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
            "H.264 Re-encode Failed. Raw mp4v Video Kept at '%s'.", output_path)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    SIDE-BY-SIDE VIDEO SPLICER
# ------------------------------------------------------------------------------
def spliceVideosSideBySide(video_paths: List[str], output_path: str,
                           labels: Optional[List[str]] = None) -> None:
    """
    Places two or more videos side by side into a single comparison video.

    Videos are composited frame-by-frame using OpenCV, allowing different video
    heights and durations to be handled consistently. Shorter videos remain
    frozen on their final frame until the longest video finishes.

    Labels are added directly to each video pane using OpenCV. ffmpeg is used
    only to re-encode the completed video to H.264.

    Args:
        video_paths (List[str]): Paths to the mp4s to place side by side,
                                 in left-to-right order. Needs at least 2.
        output_path (str): Full path (including filename) to save the
                           combined .mp4.
        labels (Optional[List[str]]): One label per video, in the same order
                                      as video_paths. Defaults to each
                                      video's filename without extension.

    Returns:
        None
    """
    # Require at Least Two Input Videos.
    if len(video_paths) < 2:
        log(LogLevel.ERROR, "introspection_visualiser:spliceVideosSideBySide",
            "Need at Least 2 Videos to Splice Side by Side, Got %d.",
            len(video_paths))
        return

    # Check that Every Input Video Exists.
    for path in video_paths:
        if not os.path.exists(path):
            log(LogLevel.ERROR, "introspection_visualiser:spliceVideosSideBySide",
                "Video Not Found: '%s'.", path)
            return

    # Ensure the Output Directory Exists.
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Open Every Input Video.
    captures = [cv2.VideoCapture(path) for path in video_paths]

    # Check That Every Input Video Opened Successfully.
    for path, cap in zip(video_paths, captures):
        if not cap.isOpened():
            log(LogLevel.ERROR, "introspection_visualiser:spliceVideosSideBySide",
                "Could Not Open Input Video: '%s'.", path)

            # Release Any Videos That Were Opened Successfully.
            for opened_cap in captures:
                opened_cap.release()

            # Exit Function. Video Splicing Failed. 
            return

    # Get the Common Output Height and Frame Rate.
    target_height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captures[0].get(cv2.CAP_PROP_FPS)

    # Use the Default Frame Rate if the Video's FPS Is Invalid.
    if not fps or fps <= 0:
        fps = VIDEO_FPS

    # Stop if the First Video's Height Could Not Be Determined.
    if target_height <= 0:
        log(LogLevel.ERROR, "introspection_visualiser:spliceVideosSideBySide",
            "Could Not Determine Height of First Video '%s'.",
            video_paths[0])

        # Release All Open Videos Before Returning.
        for cap in captures:
            cap.release()

        # Exit Function. Video Splicing Failed. 
        return

    # Get the Number of Frames in Each Video.
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]

    # Use the Longest Video to Determine the Output Length.
    max_frames = max(frame_counts)

    # Print Log Message with Details Gathered from Input Videos. 
    log(LogLevel.INFO, "introspection_visualiser:spliceVideosSideBySide",
        "Splicing %d Videos Side by Side. Target Height: %d, FPS: %.2f, "
        "Output Frames: %d.",
        len(video_paths), target_height, fps, max_frames)

    # Read the First Frame from Each Video.
    last_frames: List[Optional[np.ndarray]] = []

    # Read the First Frame from Each Video.
    for index, cap in enumerate(captures):
        ok, frame = cap.read()

        # Stop if the First Frame Cannot Be Read.
        if not ok or frame is None:
            log(LogLevel.ERROR, "introspection_visualiser:spliceVideosSideBySide",
                "Could Not Read First Frame from Video '%s'.",
                video_paths[index])

            # Release All Open Videos Before Returning.
            for opened_cap in captures:
                opened_cap.release()

            # Exit Function. Video Splicing Failed. 
            return

        # Store the Frame for Shorter Video Freezing.
        last_frames.append(frame)

        # Reset the Video to the First Frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Calculate the Width of Each Resized Video Pane.
    pane_widths = []

    # Maintain the Original Aspect Ratio for Every Frame.
    for frame in last_frames:
        frame_height, frame_width = frame.shape[:2]
        pane_width = int(round(frame_width * target_height / frame_height))
        pane_widths.append(pane_width)

    # Calculate the Total Output Width.
    output_width = sum(pane_widths)

    # Write the Intermediate mp4v Video.
    raw_output_path = output_path.replace('.mp4', '_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(
        raw_output_path,
        fourcc,
        fps,
        (output_width, target_height)
    )

    # Stop if the VideoWriter Could Not Be Opened.
    if not writer.isOpened():
        log(LogLevel.ERROR, "introspection_visualiser:spliceVideosSideBySide",
            "Failed to Open VideoWriter for '%s'.", raw_output_path)

        # Release All Open Videos Before Returning.
        for cap in captures:
            cap.release()

        # Exit Function. Video Splicing Failed. 
        return

    # Composite Every Frame from Each Input Video.
    frames_written = 0

    # Process Every Frame Until the Longest Video Ends.
    for frame_idx in range(max_frames):
        panes = []

        # Read and Prepare the Current Frame from Each Video.
        for index, cap in enumerate(captures):
            ok, frame = cap.read()

            # Update the Last Frame when a New Frame is Available.
            if ok and frame is not None:
                last_frames[index] = frame

            # Stop Using This Video if No Frame Is Available.
            elif last_frames[index] is None:
                log(LogLevel.WARNING,
                    "introspection_visualiser:spliceVideosSideBySide",
                    "Video '%s' Has No Frame Available at Frame %d.", video_paths[index], frame_idx + 1)
                continue

            # Use the Last Frame when the Video Has Ended.
            frame = last_frames[index].copy()

            # Resize the Pane to the Common Height.
            frame_height, frame_width = frame.shape[:2]

            # Resize the Pane to the Common Height.
            if frame_height != target_height:
                target_width = int(round(frame_width * target_height / frame_height))

                # Maintain the Pane's Original Aspect Ratio.
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Add the Prepared Pane to the Current Frame.
            panes.append(frame)

        # Skip the Frame if No Panes are Available.
        if not panes:
            continue

        # Stack All Video Panes Horizontally.
        combined = np.hstack(panes)

        # Ensure the Combined Frame Matches the Output Dimensions.
        if combined.shape[1] != output_width or combined.shape[0] != target_height:
            combined = cv2.resize(combined, (output_width, target_height), interpolation=cv2.INTER_AREA)

        # Write the Combined Frame.
        writer.write(combined)
        frames_written += 1

    # Release the Video Writer and Input Videos.
    writer.release()

    # Release All Input Video Captures.
    for cap in captures:
        cap.release()

    # Confirm the Intermediate Video Was Written.
    log(LogLevel.INFO, "introspection_visualiser:spliceVideosSideBySide",
        "Raw Side-by-Side Video Written (%d Frames) -> '%s'.",
        frames_written, raw_output_path)

    # Stop if No Frames Were Written.
    if frames_written == 0:
        log(LogLevel.ERROR, "introspection_visualiser:spliceVideosSideBySide",
            "No Frames Were Written. Removing Raw Output '%s'.", raw_output_path)

        # Remove the Empty Output File if It Exists.
        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)

        # Return. No File Produced. 
        return

    # Re-Encode the Completed Video to H.264 Using ffmpeg.
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    ret = subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i', raw_output_path,
            '-vcodec', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-loglevel', 'error',
            h264_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Replace the Raw Video with the H.264 Version.
    if ret.returncode == 0 and os.path.exists(h264_path):
        # Remove an Existing Output Video.
        if os.path.exists(output_path):
            os.remove(output_path)

        # Remove the Intermediate mp4v Video & Save Final Copy. 
        os.rename(h264_path, output_path)
        os.remove(raw_output_path)

        # Print Success Message. 
        log(LogLevel.INFO, "introspection_visualiser:spliceVideosSideBySide",
            "Side-by-Side Video Re-encoded to H.264 and Saved to '%s'.",
            output_path)

    else:
        # Keep the Raw Video if Re-Encoding Fails.
        log(LogLevel.WARNING, "introspection_visualiser:spliceVideosSideBySide",
            "H.264 Re-encode Failed. Raw mp4v Video Kept at '%s'. ffmpeg Error: %s",
            raw_output_path, ret.stderr.strip())
# ------------------------------------------------------------------------------
