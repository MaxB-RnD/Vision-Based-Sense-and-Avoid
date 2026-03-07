# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: browse_data.py %
# Date Created:            %date_created: Fri Mar 06 22:48:34 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Browses AOT dataset ground truth metadata entry by entry WITHOUT downloading
# any images. Use this to explore the dataset and decide hardness thresholds
# before committing to any image downloads.
#
# USAGE INSTRUCTIONS:
#   1. Set Global Variables: 
#       - No. of Runs to View at Once.
#       - Whether to Download Partial or Full Dataset.
#   2. Run the Script from Project Root:    
#       - python -m src.data.browse_data
#
# ------------------------------------------------------------------------------
# -                          L I C E N S I N G                                 -
# ------------------------------------------------------------------------------
# This software and its contents are the intellectual property of
# Lockheed Martin Australia and are protected under applicable copyright
# and intellectual property laws.
#
# All rights reserved. No part of this software may be reproduced,
# distributed, modified, or transmitted in any form or by any means
# without the prior written permission of Lockheed Martin Australia.
#
# Unauthorised use, copying, or distribution of this software is
# strictly prohibited.
# ------------------------------------------------------------------------------

import os
import shutil

import sys
from typing import Optional

from core.dataset import Dataset
from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                        CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Number of Entries to Display Before Pausing for User Input.
PAUSE_EVERY_N = 1    

# If True, Loads Only Flights Present in the Valid Encounters File (~500GB Subset).
# If False, Loads All Flights in the Full Dataset (~11TB).
PARTIAL = True  
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              DATA DOWNLOAD
# ------------------------------------------------------------------------------
def dataLoader(download: bool = True) -> Dataset:
    """
    Loads AOT dataset metadata from all three S3 parts without downloading images.

    The PARTIAL flag is set on initialisation and inherited by all subsequent
    add() calls. Parts 2 and 3 cannot have a different partial setting to part 1.

    Args:
        download (bool): If True, downloads metadata from S3 if not found locally.
                         If False, uses only locally cached files.

    Returns:
        Dataset: Loaded dataset object containing all valid flights across 
                 parts 1, 2 and 3.

    Raises:
        SystemExit: If the dataset fails to load for any reason.
    """
    # Display Start Message with Clean Spacing Buffer. 
    print()
    log(LogLevel.INFO, "browse_data:dataLoader", 
        "Loading Dataset Metadata (No Images Will Be Downloaded).\n")
    
    # Check if a Fresh Download Has Been Requested.
    if download and os.path.exists("data"):
        # Delete the Data Folder. ATO will Automatically ReDownload. 
        shutil.rmtree("data")
    else:
        # Do Nothing. ATO Only Loads when all Files still Exist. 
        ...

    try:
        # Initialise Part 1 of the Data.
        dataset = Dataset(
            local_path='data/part1',
            s3_path='s3://airborne-obj-detection-challenge-training/part1/',
            prefix='part1',
            partial=PARTIAL        
        )
        # Add Part 2 of the Data. The partial argument is inherited from above.
        dataset.add(
            local_path='data/part2',
            s3_path='s3://airborne-obj-detection-challenge-training/part2/',
            prefix='part2'        
        )
        # Add Part 3 of the Data. The partial argument is inherited from above.
        dataset.add(
            local_path='data/part3',
            s3_path='s3://airborne-obj-detection-challenge-training/part3/',
            prefix='part3'
        )
        # Display Success Message with Clean Spacing Buffer. 
        print()
        log(LogLevel.INFO, "browse_data:dataLoader", 
            "Dataset Metadata Loaded Successfully.")

    except Exception as e:
        # Display Error Message with Clean Spacing Buffer. 
        print()
        log(LogLevel.ERROR, "browse_data:dataLoader", 
            "Failed to Load Dataset Metadata: %s.\n", str(e))
        
        # Exit Program with Error Status. 
        sys.exit(1)

    # Return Populated Datset. 
    return dataset
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                        HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def bboxArea(bb: Optional[list]) -> Optional[float]:
    """
    Computes bounding box area from a [left, top, width, height] bounding box.

    Args:
        bb (Optional[list]): Bounding box as [left, top, width, height], or None.

    Returns:
        Optional[float]: Area in pixels squared, or None if the input is invalid.
    """
    # Check the Input List is Complete. 
    if bb and len(bb) == 4:
        # Return Area of the Bounding Box.
        return round(bb[2] * bb[3], 2)
    
    # If the Input List is Invalid Return None. 
    return None


def displayEntry(flight_id: str, frame_id: str, obj: object) -> None:
    """
    Displays the logged metadata for a single detected airborne object.

    Prints flight, frame, object ID, bounding box, distance, and horizon.

    Args:
        flight_id (str): ID of the parent flight sequence.
        frame_id  (str): ID of the frame within the flight.
        obj       (object): Detected airborne object from the dataset.

    Returns:
        None:
    """
    # Extract All Available Metadata Fields from the Object.
    bb       = getattr(obj, 'bb', None)
    distance = getattr(obj, 'range_distance_m', float('nan'))
    horizon  = getattr(obj, 'is_above_horizon', 'N/A')
    obj_id   = getattr(obj, 'id', 'unknown')
    area     = bboxArea(bb)

    # Convert Numeric Horizon Code to a Readable String.
    horizon_lookup = {1: 'Above', -1: 'Below', 0: 'Unclear'}

    # Default to Raw Value if Unrecognised.
    horizon_str = horizon_lookup.get(horizon, str(horizon)) if isinstance(horizon, int) else str(horizon)

    # Print the Log to the Terminal.
    print("Flight    : %s" % flight_id)
    print("Frame     : %s" % frame_id)
    print("Object ID : %s" % obj_id)
    print("BBox      : %s  (area: %s px²)" % (str(bb), str(area)))
    print("Distance  : %s m" % str(distance))
    print("Horizon   : %s" % horizon_str)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                        MAIN BROWSING LOOP
# ------------------------------------------------------------------------------
# Ask the User Whether to Re-Download the Metadata or Use Local Copy.
while True:
    print("\nDo you Wish to Download the AOT Metadata? (Y/N)")
    download = input("  >> ").strip().lower()

    if download == 'y':
        # Download Fresh Metadata from S3.
        dataset = dataLoader(download=True)
        break
    elif download == 'n':
        # Use Locally Cached Metadata.
        dataset = dataLoader(download=False)
        break
    else:
        # Invalid Input. Re-Ask the Question.
        log(LogLevel.WARNING, "browse_data:main", 
            "Invalid Input. Please Enter Y or N.")


flight_ids    = dataset.get_flight_ids()
total_flights = len(flight_ids)

log(LogLevel.INFO, "browse_data:main", "Found %d Flights in Metadata.", total_flights)
print("Browse Data Using Controls: [Enter] Next  |  [s] Skip Flight  |  [q] Quit\n")

entry_count = 0
stop        = False

for flight_idx, flight_id in enumerate(flight_ids):
    if stop:
        break

    flight = dataset.get_flight(flight_id)
    frames = flight.frames

    log(LogLevel.INFO, "browse_data:main", "Flight %d/%d  (%s)  |  %d Frames",
        flight_idx + 1, total_flights, flight_id, len(frames))

    skip_flight = False

    for frame_id, frame in frames.items():
        if stop or skip_flight:
            break

        objects = frame.detected_objects

        if not objects:
            log(LogLevel.WARNING, "browse_data:main", "Frame %s has No Airborne Objects, Skipping.", frame_id)
            continue

        for obj in objects:
            displayEntry(flight_id, frame_id, obj)
            entry_count += 1

            if entry_count % PAUSE_EVERY_N == 0:
                try:
                    cmd = input("  >> ").strip().lower()
                except EOFError:
                    cmd = 'q'

                if cmd == 'q':
                    log(LogLevel.INFO, "browse_data:main", "Quit Requested by User.")
                    stop = True
                    break
                elif cmd == 's':
                    log(LogLevel.WARNING, "browse_data:main", "Skipping Flight %s.", flight_id)
                    skip_flight = True
                    break
                # anything else (including Enter) → continue

log(LogLevel.INFO, "browse_data:main", "Browsing Complete. Total Entries Viewed: %d.\n", entry_count)
# ------------------------------------------------------------------------------
