# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: view_data.py %
# Date Created:            %date_created: Sun Mar 08 16:58:43 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Visualises downloaded AOT flight sequences from the selected flights list.
# Supports Two Modes:
#   FRAME MODE  — Opens an OpenCV window and steps through every frame in the
#                 flight. Bounding boxes are drawn on frames that contain a
#                 detected airborne object. A status banner indicates whether
#                 the current frame has a detection or not.
#   VIDEO MODE  — Generates an annotated .mp4 for the selected flight using
#                 the AOT library and saves it to 'data/part{N}/Videos/{id}.mp4'
#
# USAGE INSTRUCTIONS:
#   1. Ensure Flight Images have Been Downloaded Via 'download_data.py'.
#   2. Set the Path to the Selected Flight(s) ID Text File.
#   3. Run the Script from Project Root:
#       - python -m src.view.view_data
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
import sys

import cv2
import numpy as np
from typing import Any

from logging_module import LogLevel, log
from src.data.browse_data import dataLoader
from src.data.download_data import loadFlightIds

# ------------------------------------------------------------------------------
#                        CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Paths & Environment Variables. 
SELECTED_FLIGHTS_PATH = 'data/selected_flights.txt'
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts/truetype/dejavu'

# Video Settings.
VIDEO_FPS = 10

# Bounding Box (BGR).
BB_COLOUR_DETECTION = (0, 0, 211)
BB_THICKNESS = 4

# Bounding Box Labels.
BB_LABEL_FONT_SCALE = 1.0
BB_LABEL_THICKNESS  = 2
BB_LABEL_LINE_GAP   = 12

# Status Banner.
BANNER_HEIGHT = 80
BANNER_FONT_SCALE = 1.1
BANNER_COLOUR_DETECTION = (0, 0, 211)
BANNER_COLOUR_EMPTY     = (60, 60, 60)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         BOUNDING BOX HELPERS
# ------------------------------------------------------------------------------
def drawBoundingBoxes(image: np.ndarray, object_locations: dict) -> np.ndarray:
    """
    Draws bounding boxes and labels for all detected objects, onto the provided 
    image.

    Bounding box coordinates are sourced from 'AirborneObjectLocation.bb', which
    exposes the box as an object with left/top/width/height attributes.

    Args:
        image (np.ndarray): OpenCV image array to draw onto (modified in place).
        object_locations (dict): Mapping of object_id -> AirborneObjectLocation.

    Returns:
        np.ndarray: Annotated image.
    """
    # Use OpenCV's Default Simple Font for All Annotations.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Iterate through Each Detected Object in the Current Frame.
    for obj_id, obj in object_locations.items():
        # Retrieve the BBox Structure Attached to the Object.
        bb = getattr(obj, 'bb', None)

        # Skip Objects that have No BBox Data.
        if bb is None:
            continue

        # Resolve BBox Coordinates. Supports (l,t,w,h) & (x,y,w,h)
        left   = getattr(bb, 'left',   None) or getattr(bb, 'x', None)
        top    = getattr(bb, 'top',    None) or getattr(bb, 'y', None)
        width  = getattr(bb, 'width',  None) or getattr(bb, 'w', None)
        height = getattr(bb, 'height', None) or getattr(bb, 'h', None)

        # Skip Incomplete BBoxs.
        if None in (left, top, width, height):
            continue

        # Tell Type Checker these are Guaranteed to Exist.
        assert left is not None and top is not None
        assert width is not None and height is not None

        # Convert BBox Co-ordinates to Integer Pixel Positions.
        x1 = int(left)           # Left Edge of BBox.
        y1 = int(top)            # Top Edge.
        x2 = int(left + width)   # Right Edge.
        y2 = int(top  + height)  # Bottom Edge.

        # Draw BBox Rectangle.
        cv2.rectangle(image, (x1, y1), (x2, y2), 
                      BB_COLOUR_DETECTION, BB_THICKNESS)

        # Retrieve Object's Estimated Distance from the Camera.
        distance = getattr(obj, 'range_distance_m', None)
        if distance is not None and distance == distance: 
            # Format the Distance Label.
            distance_str = "%dm" % int(distance)
        else:
            # If Value Doesn't Exist OR is NaN Display Unkown.
            distance_str = "Unknown"

        # Measure Both Label Lines to Size the Background Correctly.
        (w1, h1), _ = cv2.getTextSize(obj_id, font, 
                                      BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)
        (w2, h2), _ = cv2.getTextSize(distance_str, font, 
                                      BB_LABEL_FONT_SCALE, BB_LABEL_THICKNESS)

        # Position Labels Below the BBox.
        label_x = x1                          # Align Text with Left Side of Box.
        line1_y = y2 + h1 + 4                 # First Label Line (Object ID).
        line2_y = line1_y + h2 + BB_LABEL_LINE_GAP  # Second Line (Distance).

        # Draw a Background Rectangle Behind Both Lines for Legibility.
        bg_x1 = label_x - 2                 # Left Edge.
        bg_y1 = y2 + 2                      # Top Edge.
        bg_x2 = label_x + max(w1, w2) + 4   # Right Edge.
        bg_y2 = line2_y + 4                 # Bottom Edge.

        # Draw the Filled Background Rectangle.
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                      BB_COLOUR_DETECTION, cv2.FILLED)

        # Draw Line 1 — Object ID.
        cv2.putText(image, obj_id, (label_x, line1_y),
                    font, BB_LABEL_FONT_SCALE, (255, 255, 255),
                    BB_LABEL_THICKNESS, cv2.LINE_AA)

        # Draw Line 2 — Distance.
        cv2.putText(image, distance_str, (label_x, line2_y),
                    font, BB_LABEL_FONT_SCALE, (255, 255, 255),
                    BB_LABEL_THICKNESS, cv2.LINE_AA)

    # Return the Annotated Image (Same Object, Modified in Place).
    return image


def addStatusBanner(image: np.ndarray, frame_num: int, total_frames: int,
                    has_detection: bool, flight_id: str) -> np.ndarray:
    """
    Adds a coloured status banner to the bottom of the image.

    The banner colour is BANNER_COLOUR_DETECTION when a detection is present and
    BANNER_COLOUR_EMPTY when no airborne object is in the frame, providing a 
    quick visual cue as you step through frames.

    Args:
        image (np.ndarray): OpenCV image to annotate.
        frame_num (int): Current frame number (1-indexed for display).
        total_frames (int): Total number of frames in the flight.
        has_detection (bool): Whether the current frame has a detected object.
        flight_id (str): Flight ID string for display.

    Returns:
        np.ndarray: Image with banner appended at the bottom.
    """
    # Get Original Image Dimensions.
    h, w = image.shape[:2]

    # Create an Empty Banner Image (Frame Width & Fixed Height).
    banner = np.zeros((BANNER_HEIGHT, w, 3), dtype=np.uint8)

    # Set Banner Colour Based on Detection Presence.
    colour = BANNER_COLOUR_DETECTION if has_detection else BANNER_COLOUR_EMPTY
    banner[:] = colour

    # Create Status Text.
    status_str = "DETECTION" if has_detection else "No Object"

    # Format Label for Diplay.
    label = "Frame %d / %d   |   %s   |   %s" % (frame_num, total_frames, 
                                                 status_str, flight_id)

    # Draw Label Text on Banner
    cv2.putText(banner, label, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                BANNER_FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)

    # Stack Original Image On Top of Banner & Return Annotated Frame.
    return np.vstack([image, banner])
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                            FRAME VIEWER
# ------------------------------------------------------------------------------
def viewFlight(flight: Any, flight_id: str) -> None:
    """
    Opens an OpenCV window and steps through every frame in the provided flight.

    All frames are shown. Frames with detected objects have bounding boxes
    drawn and a BANNER_COLOUR_DETECTION status banner at the bottom. Frames 
    without detections show a BANNER_COLOUR_EMPTY banner instead.

    Args:
        flight (object): AOT flight object from dataset.get_flight().
        flight_id (str): Flight ID string for display in the banner.

    Returns:
        None:
    """
    # Retrieve Frame Dictionary from the Flight Object.
    frames = flight.frames
    
    # Convert Frame IDs to a List to Iterate in Order.
    frame_ids = list(frames.keys())

    # Total Number of Frames in this Flight.
    total_frames = len(frame_ids)

    # Display Viewer Startup Information.
    log(LogLevel.INFO, "view_data:viewFlight",
        "Opening Frame Viewer for Flight %s  (%d Frames).", flight_id, total_frames)
    
    # Print Viewer Controls to the Terminal.
    print("Viewer Controls:\n   [Any Key] Next Frame  |  [Q] Quit Flight.")

    # Create an OpenCV Window for Displaying Frames.
    window_name = "AOT Viewer — %s" % flight_id
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    # Iterate through each Frame in the Flight.
    for frame_idx, frame_id in enumerate(frame_ids):
        # Retrieve Frame Object from the Flight Frame Dictionary.
        frame = frames[frame_id]

        # Extract Detected Object Information for this Frame.
        object_locations = frame.detected_object_locations

        # Determine if any Detections are Present.
        has_detection   = bool(object_locations)

        try:
            # Load the Frame Image as a NumPy Array via the AOT Library.
            image = frame.image(type='numpy')
            image = image.copy()
        except Exception as e:
            # Skip Frame if the Image Cannot be Loaded.
            log(LogLevel.WARNING, "view_data:viewFlight",
                "Could Not Load Image for Frame %s: %s. Skipping.", frame_id, str(e))
            continue

        # Convert to BGR for OpenCV if Needed.
        if image.ndim == 2:
            # Greyscale — Convert to 3-channel BGR.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB — Convert to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA — Convert to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Draw BBoxes for all Detected Airborne Objects.
        if has_detection:
            image = drawBoundingBoxes(image, object_locations)

        # Add Status Banner at the Bottom of the Frame.
        image = addStatusBanner(image, frame_idx + 1, total_frames,
                                has_detection, flight_id)
        cv2.imshow(window_name, image)

        # Wait Indefinitely for a Key Press.
        key = cv2.waitKey(0) & 0xFF

        # If User Presses 'q', Exit the Viewer Early.
        if key == ord('q'):
            log(LogLevel.INFO, "view_data:viewFlight", "Quit Requested. Closing Viewer.")
            break
    
    # Close the OpenCV Window when Finished.
    cv2.destroyWindow(window_name)

    # Display Viewer Shutdown to Viewer.
    log(LogLevel.INFO, "view_data:viewFlight",
        "Frame Viewer Closed for Flight %s.", flight_id)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           VIDEO GENERATOR
# ------------------------------------------------------------------------------
def generateVideo(flight: Any, flight_id: str) -> None:
    """
    Generates an annotated video for the given flight using OpenCV VideoWriter.

    Loops all frames in order, draws bounding boxes and labels on detections
    using the same drawBoundingBoxes() pipeline as the frame viewer, then
    writes each frame to an .mp4 at VIDEO_FPS.

    Output is saved to data/part{N}/Videos/{flight_hash}.mp4.

    Args:
        flight (Any): AOT flight object from dataset.get_flight().
        flight_id (str): Flight ID string (e.g. 'part1abc123...').

    Returns:
        None:
    """
    # Determine which Dataset Partition the Flight Belongs To.
    if flight_id.startswith('part1'):
        part = 'part1'
    elif flight_id.startswith('part2'):
        part = 'part2'
    elif flight_id.startswith('part3'):
        part = 'part3'
    else:
        part = 'part1'

    # Remove the Partition Prefix.
    bare_id = flight_id[len(part):]

    # Build the Output File Path (Ensure Directory Exists).
    output_dir = os.path.join('data', part, 'Videos')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, bare_id + '.mp4')

    # Display Video Generation Commenced Message.
    log(LogLevel.INFO, "view_data:generateVideo",
        "Generating Annotated Video for Flight %s.", flight_id)

    # Get the Dictionary of Frames from the Flight Object.
    frames = flight.frames
    frame_ids = list(frames.keys())
    total_frames = len(frame_ids)

    # VideoWriter is Created AFTER Frame Dimensions are Known. 
    writer = None

    # Loop through Every Frame in the Flight.
    for frame_idx, frame_id in enumerate(frame_ids):
        # Retrieve the Frame Object & Check if it has any Detections.
        frame            = frames[frame_id]
        object_locations = frame.detected_object_locations
        has_detection    = bool(object_locations)

        try:
            # Load the Image as a NumPy Array.
            image = frame.image(type='numpy')
            image = image.copy()
        except Exception as e:
            # If the Frame Cannot be Loaded, Skip it.
            log(LogLevel.WARNING, "view_data:generateVideo",
                "Could Not Load Frame %s: %s. Skipping.", frame_id, str(e))
            continue

        # Convert to BGR for OpenCV if Needed.
        if image.ndim == 2:
            # Greyscale — Convert to 3-channel BGR.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB — Convert to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA — Convert to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Initialise VideoWriter using the First Valid Frame.
        if writer is None:
            # Extract Image Height & Width.
            h, w = image.shape[:2]
            
            # Account for the Status Banner Added to Every Frame.
            video_h = h + BANNER_HEIGHT
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore[attr-defined]
            writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (w, video_h))

            # Check if Writer Opened Successfully.
            if not writer.isOpened():
                # Display Error & Cancel if CV2 Can't Open. 
                log(LogLevel.ERROR, "view_data:generateVideo",
                    "Failed to Open VideoWriter for '%s'.", output_path)
                return
            
            # Display Video Configuration Message to User. 
            log(LogLevel.INFO, "view_data:generateVideo",
                "Writing %d Frames at %dx%d  %d fps.",
                total_frames, w, h, VIDEO_FPS)

        # Draw BBoxes and Status Banner.
        if has_detection:
            image = drawBoundingBoxes(image, object_locations)

        # Add Frame Counter & Detection Status Banner.
        image = addStatusBanner(image, frame_idx + 1, total_frames,
                                has_detection, flight_id)

        # Write the Annotated Frame to the Video File.
        writer.write(image)

    # Finalise Video After Processing all Frames.
    if writer is not None:
        # Release the VideoWriter Resource & Display Message to User. 
        writer.release()
        log(LogLevel.INFO, "view_data:generateVideo",
            "Raw Video Written. Re-encoding to H.264.")

        # Use FFmpeg to Convert the Video to H.264 for Compatibility.
        h264_path = output_path.replace('.mp4', '_h264.mp4')
        ret = os.system(
            'ffmpeg -y -i "%s" -vcodec libx264 -preset fast -crf 23 "%s" -loglevel error'
            % (output_path, h264_path)
        )

        # If Re-Encoding Succeeded, Replace the Original File
        if ret == 0 and os.path.exists(h264_path):
            # Remove Original mp4v & Replace with H.264 File. 
            os.remove(output_path)
            os.rename(h264_path, output_path)
            log(LogLevel.INFO, "view_data:generateVideo",
                "Video Re-encoded to H.264 and Saved to '%s'.", output_path)
        else:
            # If Conversion Fails, Keep the Original Video.
            log(LogLevel.WARNING, "view_data:generateVideo",
                "H.264 Re-encode Failed. Raw mp4v Video Kept at '%s'.", output_path)
    else:
        # No Frames were Successfully Loaded.
        log(LogLevel.ERROR, "view_data:generateVideo",
            "No Frames Could Be Loaded for Flight '%s'.", flight_id)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                  MAIN
# ------------------------------------------------------------------------------
def main():
    """
    The main function for running the data visualiser interface.
    """
    # Print a Blank Line for Cleaner Console Output Spacing.
    print()

    # Load the Selected Flight IDs from the .txt File.
    flight_ids = loadFlightIds(SELECTED_FLIGHTS_PATH)
    total = len(flight_ids)

    # Load Dataset Metadata.
    dataset = dataLoader(download=False)

    # Iterate through Each Flight ID with its Index.
    for idx, flight_id in enumerate(flight_ids):
        # Display Progress Information to the User.
        print()
        log(LogLevel.INFO, "view_data:main",
            "Flight %d/%d  (%s)", idx + 1, total, flight_id)

        try:
            # Verify the Flight ID Exists in the Loaded Metadata.
            flight = dataset.get_flight(flight_id)
        except Exception:
            # If Lookup Fails, Treat it as a Missing Flight.
            flight = None

        # If the Flight was Not Found in the Dataset Metadata, Skip It.
        if flight is None:
            log(LogLevel.WARNING, "view_data:main",
                "Flight %s Not Found in Metadata. Skipping.", flight_id)
            continue

        # Ask the User What to Do With This Flight.
        while True:
            # Display Instructions to the User. 
            print("Flight %d — Viewer Actions:" % (idx + 1))
            print("  [F] View Frames   |  [V] Generate Video  |  [S] Skip  |  [Q] Quit")
            try:
                # Prompt the User for Input.
                cmd = input("  >> ").strip().lower()
            except EOFError:
                # Treat Input Interruption as a Quit Command.
                cmd = 'q'

            if cmd == 'f':
                # Launch the Interactive Frame Viewer.
                viewFlight(flight, flight_id)
                break
            elif cmd == 'v':
                # Render the Video from Flight Frames.
                generateVideo(flight, flight_id)
                break
            elif cmd == 's':
                # Skip this Flight & Move to the Next One.
                log(LogLevel.INFO, "view_data:main", "Skipping Flight %s.", flight_id)
                break
            elif cmd == 'q':
                # User Requested to Quit the Entire Viewing Process.
                log(LogLevel.INFO, "view_data:main", "Quit Requested by User.\n")
                sys.exit(0)
            else:
                # Any Other Input Re-Asks the Question.
                log(LogLevel.WARNING, "view_data:main",
                    "Invalid Input. Please Enter F, V, S, or Q.")

    # No More Flights to Display.
    print()
    log(LogLevel.INFO, "view_data:main", "All Flights Processed.\n")
# ------------------------------------------------------------------------------


# Call the Main Function
if __name__ == "__main__":
    main()
