# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: introspection_visualiser.py %
# Date Created:            %date_created: Tue Aug 25 00:00:00 2026 %
# CM Version               %version: 2 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Notebook-callable helper that renders an annotated video of a full flight
# sequence with an introspection module's predicted results overlaid.
#
# Intended to be imported directly into introspection_module.ipynb alongside
# the other notebooks/helpers/* imports, and called per-flight, e.g.:
#
#   generateIntrospectionVideo(
#       flight_id       = "part2e9c4568409604a6183d24726c0ebf696",
#       result_json_path = os.path.join(RESULTS_DIR, run_name,
#                                        "part2e9c4568409604a6183d24726c0ebf696", "result.json"),
#       module_name     = "Rolling Confidence Trend",
#       output_path     = os.path.join(VISUALISER_DIR,
#                                       "part2e9c4568409604a6183d24726c0ebf696.mp4"),
#   )
#
# FRAME COVERAGE — result.json is SPARSE: it only contains an entry for a
# given img_name if the tracker produced a detection or coasted track for
# that frame. It is NOT the list of frames to render. This module always
# iterates the flight's FULL image set on disk (found via flight_id, no need
# to already have a `flight`/`dataset` object in scope) and treats
# result.json purely as an overlay — any frame with no matching entry is
# rendered as "No Object" rather than being dropped. This guarantees the
# output video covers every frame in the flight, in order.
#
# FLIGHT DISCOVERY — flight_id is expected to be prefixed with its partition
# ('part1' / 'part2' / 'part3', e.g. 'part2e9c45...'). The image directory is
# located as '{DEFAULT_DATASET_ROOT}/{part}/Images/{bare_id}/*.png'. If your
# corrupted/eval images live somewhere else, pass dataset_root= explicitly.
# ------------------------------------------------------------------------------

import os
import json
from glob import glob
from typing import Dict, List, Optional

import cv2
import numpy as np

from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                        CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Default Root Under which '{part}/Images/{bare_id}/*.png' is Expected to Live.
# Matches DATASET_ROOT as Set in introspection_module.ipynb Section 4. Override
# per-call via the dataset_root= argument if your images live elsewhere (e.g.
# the per-run symlinked EVAL_DATA_DIR from Section 6.1.1).
DEFAULT_DATASET_ROOT = 'data/corruption_dataset'

VIDEO_FPS = 10

# Bounding Box Colours (BGR).
BB_COLOUR_DETECTED = (0, 0, 211)      # Red    — Active Detection.
BB_COLOUR_COASTED  = (0, 140, 255)    # Orange — Coasted / Predicted Track (No Fresh Detection).
BB_COLOUR_WARNING  = (211, 0, 211)    # Magenta — Introspection Module Flagged this Track.
BB_THICKNESS = 4

BB_LABEL_FONT_SCALE = 1.0
BB_LABEL_THICKNESS  = 2
BB_LABEL_LINE_GAP   = 12

# Status Banner.
BANNER_HEIGHT = 80
BANNER_FONT_SCALE = 1.0
BANNER_COLOUR_WARNING   = (211, 0, 211)  # Magenta — At Least One Track Flagged.
BANNER_COLOUR_COASTED   = (0, 140, 255)  # Orange — Coasted / Predicted Track.
BANNER_COLOUR_DETECTION = (0, 0, 211)    # Red     — Detection(s), No Warning.
BANNER_COLOUR_EMPTY     = (60, 60, 60)   # Grey    — No Object Predicted.
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

    # Try the Standard '{part}/Images/{bare_id}' Layout, then Fall Back to
    # Using the Full flight_id as the Folder Name in Case Naming Differs.
    candidate_dirs = [
        os.path.join(dataset_root, part, 'Images', bare_id),
        os.path.join(dataset_root, part, 'Images', flight_id),
    ]

    for candidate_dir in candidate_dirs:
        image_paths = sorted(glob(os.path.join(candidate_dir, '*.png')))
        if image_paths:
            log(LogLevel.INFO, "introspection_visualiser:resolveFlightImagePaths",
                "Found %d Frames for Flight %s in '%s'.", len(image_paths), flight_id, candidate_dir)
            return image_paths

    # No Images Found under Any Candidate Path — Fail Loudly.
    raise FileNotFoundError(
        "Could Not Find any Frame Images for Flight '%s' under '%s'. Tried: %s"
        % (flight_id, dataset_root, candidate_dirs)
    )


def loadIntrospectionResults(result_json_path: str) -> Dict[str, List[dict]]:
    """
    Loads a per-flight result.json and indexes it by img_name for fast lookup.

    Expected format is a flat list of frame entries:
        [{"img_name": "...png", "detections": [{"track_id", "x", "y", "w", "h",
                                                  "n", "s", "source", "warning"}]}, ...]

    Args:
        result_json_path (str): Path to the flight's result.json file.

    Returns:
        Dict[str, List[dict]]: Mapping of img_name -> that frame's detections.
                               Empty dict if the file is missing or fails to load.
    """
    if not os.path.exists(result_json_path):
        log(LogLevel.WARNING, "introspection_visualiser:loadIntrospectionResults",
            "Results File Not Found: %s. All Frames will Show as 'No Object'.", result_json_path)
        return {}

    with open(result_json_path, 'r') as f:
        raw_entries = json.load(f)

    lookup: Dict[str, List[dict]] = {entry.get('img_name', ''): entry.get('detections', [])
                                     for entry in raw_entries}

    log(LogLevel.INFO, "introspection_visualiser:loadIntrospectionResults",
        "Loaded Results for %d Frames from '%s'.", len(lookup), result_json_path)

    return lookup
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    ANNOTATION HELPERS (Boxes + Banner)
# ------------------------------------------------------------------------------
def drawPredictedBoxes(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Draws predicted bounding boxes and labels for all detections in a frame.

    Box coordinates come from the result.json format: 'x'/'y' are the
    top-left corner, 'w'/'h' are width/height. Colour indicates track source
    (DETECTED vs COASTED), overridden to BB_COLOUR_WARNING if the
    introspection module flagged that track for this frame.

    Args:
        image (np.ndarray): OpenCV image array to draw onto (modified in place).
        detections (List[dict]): This frame's list of detection dicts.

    Returns:
        np.ndarray: Annotated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det.get('x'), det.get('y'), det.get('w'), det.get('h')
        if None in (x1, y1, x2, y2):
            continue

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        source  = det.get('source', 'DETECTED')
        warning = bool(det.get('warning', False))

        if warning:
            colour = BB_COLOUR_WARNING
        elif source == 'COASTED':
            colour = BB_COLOUR_COASTED
        else:
            colour = BB_COLOUR_DETECTED

        cv2.rectangle(image, (x1, y1), (x2, y2), colour, BB_THICKNESS)

        # Line 1 — Track ID. Line 2 — Confidence Score (Stands in for GT's
        # Distance Label — the Prediction Format has no Range/Distance Field).
        # Source (DETECTED/COASTED) & Warning Status are Shown in the Banner
        # Instead, Keeping the Box Label Clean.
        track_id = det.get('track_id', '?')
        score = det.get('s', None)
        score_str = ("%.2f" % score) if score is not None else "N/A"

        line1 = "ID: " + str(track_id)
        line2 = score_str

        (w1, h1), _ = cv2.getTextSize(line1, font, BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)
        (w2, h2), _ = cv2.getTextSize(line2, font, BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)

        label_x = x1
        line1_y = y2 + h1 + 4
        line2_y = line1_y + h2 + BB_LABEL_LINE_GAP

        bg_x1, bg_y1 = label_x - 2, y2 + 2
        bg_x2, bg_y2 = label_x + max(w1, w2) + 4, line2_y + 4

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), colour, cv2.FILLED)
        cv2.putText(image, line1, (label_x, line1_y), font, BB_LABEL_FONT_SCALE,
                    (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)
        cv2.putText(image, line2, (label_x, line2_y), font, BB_LABEL_FONT_SCALE,
                    (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)

    return image


def addIntrospectionBanner(image: np.ndarray, frame_num: int, total_frames: int,
                           detections: List[dict], flight_id: str, module_name: str) -> np.ndarray:
    """
    Adds a coloured status banner to the bottom of the image, escalating from
    grey (no predicted object) to red (detection) to magenta (introspection
    warning fired on at least one track this frame).

    Args:
        image (np.ndarray): OpenCV image to annotate.
        frame_num (int): Current frame number (1-indexed for display).
        total_frames (int): Total number of frames in the flight.
        detections (List[dict]): This frame's list of detection dicts.
        flight_id (str): Flight ID string for display.
        module_name (str): Name of the introspection module that produced
                           these results, shown in the banner for context.

    Returns:
        np.ndarray: Image with banner appended at the bottom.
    """
    h, w = image.shape[:2]
    banner = np.zeros((BANNER_HEIGHT, w, 3), dtype=np.uint8)

    has_detection = bool(detections)
    has_warning = any(bool(d.get('warning', False)) for d in detections)
    has_coasted = any(d.get('source', 'DETECTED') == 'COASTED' for d in detections)

    if has_warning:
        colour, status_str = BANNER_COLOUR_WARNING, "INTROSPECTION WARNING"
    elif has_coasted:
        colour, status_str = BB_COLOUR_COASTED, "COASTED"
    elif has_detection:
        colour, status_str = BANNER_COLOUR_DETECTION, "DETECTION"
    else:
        colour, status_str = BANNER_COLOUR_EMPTY, "No Object"


    banner[:] = colour

    label = "Frame %d / %d  |  %s  |  %s  |  %s" % (
        frame_num, total_frames, status_str, module_name, flight_id
    )
    cv2.putText(banner, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                BANNER_FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)

    return np.vstack([image, banner])
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           VIDEO GENERATOR
# ------------------------------------------------------------------------------
def generateIntrospectionVideo(flight_id: str, result_json_path: str, module_name: str,
                               output_path: str, dataset_root: str = DEFAULT_DATASET_ROOT,
                               video_fps: int = VIDEO_FPS) -> None:
    """
    Generates an annotated video covering EVERY frame of the given flight,
    overlaying the introspection module's predicted results where available.

    Locates the flight's images itself (no `flight`/`dataset` object needed),
    so any flight_id from part1/part2/part3 can be passed directly.

    Args:
        flight_id (str): Flight ID string, prefixed with its partition
                         (e.g. 'part2e9c4568409604a6183d24726c0ebf696').
        result_json_path (str): Path to this flight's result.json.
        module_name (str): Name of the introspection module, shown in the banner.
        output_path (str): Full path (including filename) to save the .mp4 to.
        dataset_root (str): Root directory containing '{part}/Images/...'.
                            Defaults to DEFAULT_DATASET_ROOT.
        video_fps (int): Output video frame rate. Defaults to VIDEO_FPS.

    Returns:
        None:
    """
    # Locate Every Frame Image for this Flight (Raises if None Found).
    image_paths = resolveFlightImagePaths(flight_id, dataset_root)
    total_frames = len(image_paths)

    # Load this Flight's Predicted Results (Sparse — Overlay Only).
    results_lookup = loadIntrospectionResults(result_json_path)

    # Ensure the Output Directory Exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
        "Generating '%s' Introspection Video for Flight %s  (%d Frames).",
        module_name, flight_id, total_frames)

    writer = None
    frames_written = 0

    # Iterate EVERY Frame Found on Disk — Not Just Frames with Predictions.
    for frame_idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)  # Already BGR — No Colour Conversion Needed.
        if image is None:
            log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
                "Could Not Load Image '%s'. Skipping.", image_path)
            continue

        # Look Up this Frame's Predicted Detections by Filename (Empty if None).
        img_name = os.path.basename(image_path)
        detections = results_lookup.get(img_name, [])

        # Initialise VideoWriter using the First Valid Frame.
        if writer is None:
            h, w = image.shape[:2]
            video_h = h + BANNER_HEIGHT
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (w, video_h))

            if not writer.isOpened():
                log(LogLevel.ERROR, "introspection_visualiser:generateIntrospectionVideo",
                    "Failed to Open VideoWriter for '%s'.", output_path)
                return

            log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
                "Writing %d Frames at %dx%d  %d fps.", total_frames, w, h, video_fps)

        if detections:
            image = drawPredictedBoxes(image, detections)

        image = addIntrospectionBanner(image, frame_idx + 1, total_frames,
                                       detections, flight_id, module_name)
        writer.write(image)
        frames_written += 1

    if writer is None:
        log(LogLevel.ERROR, "introspection_visualiser:generateIntrospectionVideo",
            "No Frames Could Be Loaded for Flight '%s'.", flight_id)
        return

    writer.release()
    log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
        "Raw Video Written (%d / %d Frames). Re-encoding to H.264.", frames_written, total_frames)

    if frames_written < total_frames:
        log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
            "%d Frame(s) were Skipped due to Image Load Failures — Video is Incomplete.",
            total_frames - frames_written)

    # Re-Encode to H.264 for Compatibility.
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    ret = os.system(
        'ffmpeg -y -i "%s" -vcodec libx264 -preset fast -crf 23 "%s" -loglevel error'
        % (output_path, h264_path)
    )

    if ret == 0 and os.path.exists(h264_path):
        os.remove(output_path)
        os.rename(h264_path, output_path)
        log(LogLevel.INFO, "introspection_visualiser:generateIntrospectionVideo",
            "Video Re-encoded to H.264 and Saved to '%s'.", output_path)
    else:
        log(LogLevel.WARNING, "introspection_visualiser:generateIntrospectionVideo",
            "H.264 Re-encode Failed. Raw mp4v Video Kept at '%s'.", output_path)
# ------------------------------------------------------------------------------