# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: visualise_detections.py %
# Date Created:            %date_created: Mon Mar 23 11:58:32 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Randomly samples frames from the inference results CSV and overlays both the
# model's predicted bounding boxes (blue) and the ground truth bounding boxes
# (green) for visual comparison.
#
# Useful for quickly diagnosing whether the model is detecting objects in the
# wrong location (spatial mismatch) vs. not detecting them at all.
#
# USAGE INSTRUCTIONS:
#   1. Set Configuration Variables Below.
#   2. Run the Script from Project Root:
#       - python -m src.baseline.visualise_detections
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
import random
import cv2
import numpy as np
import pandas as pd

from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Number of Random Frames to Sample and Display.
NUM_SAMPLES = 10

# Bounding Box Colours (BGR).
COLOUR_PRED = (211, 0, 0)    # Blue  - Model Prediction.
COLOUR_GT   = (0, 211, 0)    # Green - Ground Truth.
BB_THICKNESS = 3

# Font Settings.
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.7
FONT_THICKNESS = 2
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              DRAWING HELPERS
# ------------------------------------------------------------------------------
def cropToGtRegion(image: np.ndarray, gt_rows: pd.DataFrame, pad: int = 250) -> np.ndarray:
    """
    Crops the image to the region containing all GT bounding boxes,
    with optional padding.

    Args:
        image (np.ndarray): Input image.
        gt_rows (pd.DataFrame): Ground truth rows for this frame.
        pad (int): Padding (in pixels) around the GT region.

    Returns:
        np.ndarray: Cropped image.
    """
    # Nothing to Crop.
    if len(gt_rows) == 0:
        return image  

    # Get all GT Bounds.
    x1 = gt_rows['gt_left'].min()
    y1 = gt_rows['gt_top'].min()
    x2 = gt_rows['gt_right'].max()
    y2 = gt_rows['gt_bottom'].max()

    # Add Padding.
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    # Clamp to Image Bounds.
    h, w = image.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    # Return Cropped Image.
    return image[y1:y2, x1:x2]


def drawPredBox(image: np.ndarray, row: pd.Series) -> np.ndarray:
    """
    Draws a predicted bounding box from a results CSV row onto the image.

    The results CSV stores bboxes as (x1, y1, x2, y2) corner format.

    Args:
        image (np.ndarray): OpenCV image to annotate.
        row (pd.Series): A single row from the results DataFrame.

    Returns:
        np.ndarray: Annotated image.
    """
    # Extract Centre-Format Bbox.
    cx = float(row['x'])
    cy = float(row['y'])
    w  = float(row['w'])
    h  = float(row['h'])

    # Convert to Corner Format
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    # Draw Predicted BBox in Blue.
    cv2.rectangle(image, (x1, y1), (x2, y2), COLOUR_PRED, BB_THICKNESS)

    # Draw Label with Confidence Score.
    label = "pred %.2f" % float(row['s'])
    cv2.putText(image, label, (x1, max(y1 - 6, 10)),
                FONT, FONT_SCALE, COLOUR_PRED, FONT_THICKNESS, cv2.LINE_AA)

    # Return the Annotated Image.
    return image


def drawGtBoxes(image: np.ndarray, gt_rows: pd.DataFrame) -> np.ndarray:
    """
    Draws all ground truth bounding boxes from the groundtruth CSV onto the
    image.

    Args:
        image (np.ndarray): OpenCV image to annotate.
        gt_rows (pd.DataFrame): Rows from groundtruth CSV matching this frame.

    Returns:
        np.ndarray: Annotated image.
    """
    # Create Variables for Storing Results. 
    for _, gt_row in gt_rows.iterrows():
        x1 = int(gt_row['gt_left'])
        y1 = int(gt_row['gt_top'])
        x2 = int(gt_row['gt_right'])
        y2 = int(gt_row['gt_bottom'])

        # Draw GT BBox in Green.
        cv2.rectangle(image, (x1, y1), (x2, y2), COLOUR_GT, BB_THICKNESS)

        # Draw Label.
        cv2.putText(image, "gt", (x1, max(y1 - 6, 10)),
                    FONT, FONT_SCALE, COLOUR_GT, FONT_THICKNESS, cv2.LINE_AA)
    
    # Return the Annotated Image.
    return image
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                   MAIN
# ------------------------------------------------------------------------------
def main(results_csv=None, groundtruth_csv=None, images_dir=None, output_dir=None):
    """
    Randomly samples frames from an inference results CSV and generates
    annotated visualisations comparing predicted and ground truth bounding boxes.

    For each sampled frame, the function overlays:
        - Predicted bounding boxes (blue)
        - Ground truth bounding boxes (green)

    The resulting annotated images are saved to the specified output directory.

    Args:
        results_csv (str): Path to the inference results CSV file.
        groundtruth_csv (str): Path to the ground truth CSV file.
        images_dir (str): Root directory containing the dataset images, 
                          structured by flight ID.
        output_dir (str): Directory where annotated visualisation images will be saved.

    Returns:
        None:
            This function saves annotated image files to disk as a side effect.
    """
    # Print Program Started Message.
    print()
    log(LogLevel.INFO, "visualise_detections:main", "Loading Results CSV: %s", results_csv)

    # Load the Inference Results CSV.
    df = pd.read_csv(results_csv)

    # Drop Any Rows with Missing Image Names.
    df = df.dropna(subset=['img_name'])

    # Print Load Message.
    log(LogLevel.INFO, "visualise_detections:main",
        "Loaded %d Detections across %d Unique Frames.",
        len(df), df['img_name'].nunique())

    # Load Ground Truth Directly from CSV.
    log(LogLevel.INFO, "visualise_detections:main",
        "Loading Ground Truth CSV: %s", groundtruth_csv)
    gt_df = pd.read_csv(groundtruth_csv)

    # Drop Rows with No BBox.
    gt_df = gt_df.dropna(subset=['gt_left', 'gt_top', 'gt_right', 'gt_bottom'])

    # Print Load Message.
    log(LogLevel.INFO, "visualise_detections:main",
        "Loaded %d Ground Truth BBoxes across %d Unique Frames.\n",
        len(gt_df), gt_df['img_name'].nunique())

    # Build a Flight ID Lookup from GT CSV.
    img_to_flight = dict(zip(gt_df['img_name'], gt_df['flight_id']))

    # Get Unique Frame Names from Results and Randomly Sample.
    all_img_names = df['img_name'].unique().tolist()
    sample_size   = min(NUM_SAMPLES, len(all_img_names))
    sampled_names = random.sample(all_img_names, sample_size)

    # Load Sampling Message.
    log(LogLevel.INFO, "visualise_detections:main",
        "Sampled %d Frames for Visualisation.\n", sample_size)

    # Ensure Output Directory Exists.
    os.makedirs(output_dir, exist_ok=True)

    # Process Each Sampled Frame.
    for i, img_name in enumerate(sampled_names):
        print("Processing Frame %d/%d  (%s)" % (i + 1, sample_size, img_name))

        # Look Up Flight ID from GT CSV.
        flight_id = img_to_flight.get(img_name)

        # Check For Incomplete Data.
        if flight_id is None:
            log(LogLevel.WARNING, "visualise_detections:main",
                "Frame %s Not Found in Ground Truth CSV. Skipping.", img_name)
            continue

        # Load the Image from Disk.
        image_path = None
        for prefix in ['', 'part1', 'part2', 'part3']:
            candidate = os.path.join(images_dir, prefix + flight_id, img_name)
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        # Skip Missing Frames.
        if image_path is None:
            log(LogLevel.WARNING, "visualise_detections:main",
                "Could Not Find Image on Disk for Frame %s. Skipping.", img_name)
            continue
        
        # Read Image Once Found.
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Print Error Message on Fail.
        if image is None:
            log(LogLevel.ERROR, "visualise_detections:main",
                "cv2.imread Failed for %s. Skipping.", image_path)
            continue

        # Convert Greyscale to BGR for Coloured Annotations.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Retrieve All Predicted Boxes for this Frame from the Results CSV.
        pred_rows = df[df['img_name'] == img_name]
        has_pred  = len(pred_rows) > 0

        # Draw Each Predicted BBox onto the Image.
        for _, row in pred_rows.iterrows():
            image = drawPredBox(image, row)

        # Retrieve Ground Truth Boxes from GT CSV.
        gt_rows = gt_df[gt_df['img_name'] == img_name]
        has_gt  = len(gt_rows) > 0

        # Draw GT BBoxes onto the Image.
        if has_gt:
            image = drawGtBoxes(image, gt_rows)

        # Crop to GT Region.
        if has_gt:
            image = cropToGtRegion(image, gt_rows)

        # Save the Annotated Frame to the Output Directory.
        out_path = os.path.join(output_dir, "frame_%03d_%s" % (i + 1, img_name))
        cv2.imwrite(out_path, image)

    # Print Success Message. 
    print()
    log(LogLevel.INFO, "visualise_detections:main",
        "Done. %d Annotated Frames Saved to '%s'.\n", sample_size, output_dir)
# ------------------------------------------------------------------------------


# Call the Main Function.
if __name__ == "__main__":
    main()
    