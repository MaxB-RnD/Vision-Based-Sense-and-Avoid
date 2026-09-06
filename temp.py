# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: view_introspection_results.py %
# Date Created:            %date_created: Sun Aug 25 23:54:08 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 2 %
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
from typing import Dict, List, Optional

import cv2
import numpy as np

from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                        CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Paths & Environment Variables. 
DEFAULT_DATASET_ROOT = 'data/corruption_dataset'

# Video Settings.
VIDEO_FPS = 10

# Bounding Box (BGR).
BB_COLOUR_DETECTED = (0, 0, 211)     # Red.
BB_COLOUR_COASTED  = (0, 140, 255)   # Orange.
BB_COLOUR_WARNING  = (211, 0, 211)   # Magenta.
BB_THICKNESS = 4

# Bounding Box Labels.
BB_LABEL_FONT_SCALE = 1.0
BB_LABEL_THICKNESS  = 2
BB_LABEL_LINE_GAP   = 12

# Status Banner.
BANNER_HEIGHT = 80
BANNER_FONT_SCALE = 1.0
BANNER_COLOUR_WARNING   = (211, 0, 211)  # Magenta.
BANNER_COLOUR_COASTED   = (0, 140, 255)  # Orange.
BANNER_COLOUR_DETECTION = (0, 0, 211)    # Red.
BANNER_COLOUR_EMPTY     = (60, 60, 60)   # Grey.
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


def loadIntrospectionResults(result_json_path: str) -> Dict[str, List[dict]]:
    """
    Loads a per-flight result.json and indexes it by img_name for fast lookup.

    Expected format is a flat list of frame entries:
        [{"img_name": "...png", "detections": [{"track_id", "x", "y", "w", "h",
                                                  "n", "s", "source", "warning"}]}, ...]

    Args:
        result_json_path (str): Path to the flight's result.json file.

    Returns:
        Dict[str, List[dict]]: Mapping of img_name --> that frame's detections.
                               Empty dict if the file is missing or fails to load.
    """
    # Check that the Results File Exists.
    if not os.path.exists(result_json_path):
        log(LogLevel.WARNING, "introspection_visualiser:loadIntrospectionResults",
            "Results File Not Found: %s. All Frames will Show as 'No Object'.", result_json_path)
        return {}

    # Load the Raw Results from JSON.
    with open(result_json_path, 'r') as f:
        raw_entries = json.load(f)

    # Index Results by Image Name for Fast Frame Lookup.
    lookup: Dict[str, List[dict]] = {entry.get('img_name', ''): entry.get('detections', [])
                                     for entry in raw_entries}

    # Display the Number of Frames with Results.
    log(LogLevel.INFO, "introspection_visualiser:loadIntrospectionResults",
        "Loaded Results for %d Frames from '%s'.", len(lookup), result_json_path)

    # Return Lookup Dictionary.
    return lookup
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    ANNOTATION HELPERS (Boxes + Banner)
# ------------------------------------------------------------------------------
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

        # Read the Detection Source and Warning Status.
        source  = det.get('source', 'DETECTED')
        warning = bool(det.get('warning', False))

        # Set the Box Colour Based on the Detection Status.
        if warning:
            colour = BB_COLOUR_WARNING
        elif source == 'COASTED':
            colour = BB_COLOUR_COASTED
        else:
            colour = BB_COLOUR_DETECTED

        # Draw the Bounding Box.
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, BB_THICKNESS)

        # Read the Track ID and Confidence Score.
        track_id = det.get('track_id', '?')
        score = det.get('s', None)
        score_str = ("%.2f" % score) if score is not None else "N/A"

        # Create the Two-Line Box Label.
        line1 = "ID: " + str(track_id)
        line2 = score_str

        # Calculate the Label Dimensions.
        (w1, h1), _ = cv2.getTextSize(line1, font, BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)
        (w2, h2), _ = cv2.getTextSize(line2, font, BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)

        # Position the Label Below the Bounding Box.
        label_x = x1
        line1_y = y2 + h1 + 4
        line2_y = line1_y + h2 + BB_LABEL_LINE_GAP

        # Draw the Label Background.
        bg_x1, bg_y1 = label_x - 2, y2 + 2
        bg_x2, bg_y2 = label_x + max(w1, w2) + 4, line2_y + 4
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), colour, cv2.FILLED)

        # Draw the Track ID and Confidence Score.
        cv2.putText(image, line1, (label_x, line1_y), font, BB_LABEL_FONT_SCALE,
                    (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)
        cv2.putText(image, line2, (label_x, line2_y), font, BB_LABEL_FONT_SCALE,
                    (255, 255, 255), BB_LABEL_THICKNESS, cv2.LINE_AA)

    # Return the Marked-Up Frame. 
    return image


def addIntrospectionBanner(image: np.ndarray, frame_num: int, total_frames: int,
                           detections: List[dict], flight_id: str, module_name: str) -> np.ndarray:
    """
    Adds a coloured status banner to the bottom of the image, escalating from
    grey (no predicted object) to orange (coasted) to red (detection) to 
    magenta (introspection warning fired on at least one track this frame).

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
    # Get the Image Dimensions.
    h, w = image.shape[:2]

    # Create the Empty Banner.
    banner = np.zeros((BANNER_HEIGHT, w, 3), dtype=np.uint8)

    # Check the Detection Status for the Frame.
    has_detection = bool(detections)
    has_warning = any(bool(d.get('warning', False)) for d in detections)
    has_coasted = any(d.get('source', 'DETECTED') == 'COASTED' for d in detections)

    # Set the Banner Colour and Status.
    if has_warning:
        colour, status_str = BANNER_COLOUR_WARNING, "INTROSPECTION WARNING"
    elif has_coasted:
        colour, status_str = BB_COLOUR_COASTED, "COASTED"
    elif has_detection:
        colour, status_str = BANNER_COLOUR_DETECTION, "DETECTION"
    else:
        colour, status_str = BANNER_COLOUR_EMPTY, "No Object"

    # Apply the Status Colour to the Banner.
    banner[:] = colour

    # Create the Banner Label.
    label = "Frame %d / %d  |  %s  |  %s  |  %s" % (
        frame_num, total_frames, status_str, module_name, flight_id
    )

    # Draw the Label onto the Banner.
    cv2.putText(banner, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                BANNER_FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)

    # Append the Banner to the Bottom of the Image.
    return np.vstack([image, banner])
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           VIDEO GENERATOR
# ------------------------------------------------------------------------------
def generateIntrospectionVideo(flight_id: str, result_json_path: str, module_name: str,
                               output_path: str, dataset_root: str = DEFAULT_DATASET_ROOT,
                               video_fps: int = VIDEO_FPS) -> None:
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

    Returns:
        None:
    """
    # Locate Every Frame Image for the Flight.
    image_paths = resolveFlightImagePaths(flight_id, dataset_root)
    total_frames = len(image_paths)

    # Load the Predicted Results for the Flight.
    results_lookup = loadIntrospectionResults(result_json_path)

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

        # Look Up Predictions for the Current Frame.
        img_name = os.path.basename(image_path)
        detections = results_lookup.get(img_name, [])

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

        # Draw Predicted Bounding Boxes when Detections are Present.
        if detections:
            image = drawPredictedBoxes(image, detections)

        # Add the Introspection Status Banner.
        image = addIntrospectionBanner(image, frame_idx + 1, total_frames,
                                       detections, flight_id, module_name)

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
