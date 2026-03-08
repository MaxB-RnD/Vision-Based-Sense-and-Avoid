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
#       - Path to the Flight ID Storage File. 
#
#   2. Run the Script from Project Root:    
#       - python -m src.data.browse_data
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
import shutil

import sys
from typing import Optional

from core.dataset import Dataset
from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Number of Entries to Display Before Pausing for User Input.
PAUSE_EVERY_N = 1    

# If True, Loads Only Flights Present in the Valid Encounters File (~500GB Subset).
# If False, Loads All Flights in the Full Dataset (~11TB).
PARTIAL = True  

# Path to the File Where Selected Flight IDs are Saved.
SELECTED_FLIGHTS_PATH = 'data/selected_flights.txt'
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               DATA DOWNLOAD
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
        "Loading Dataset Metadata.\n")
    
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
#                               HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def bboxArea(bb: Optional[object]) -> Optional[float]:
    """
    Computes bounding box area from an AirborneObjectLocation.BoundingBox object.

    Args:
        bb (Optional[object]): BoundingBox object with .width and .height attributes, or None.

    Returns:
        Optional[float]: Area in pixels squared, or None if the input is invalid.
    """    
    # If the Input Object is Invalid Return None. 
    if bb is None:
        return None
    
    # Get Width & Height from the Bounding Box Object. 
    width  = getattr(bb, 'w', None) or getattr(bb, 'width',  None)
    height = getattr(bb, 'h', None) or getattr(bb, 'height', None)

    # Check the Input Object is Complete. 
    if width is not None and height is not None:
        # Return Area of the Bounding Box.
        return width * height
    
    # If the Ouput Object is Invalid Return None. 
    return None


def displayEntry(flight_id: str, frame_id: str, obj_id: str, obj: object) -> None:
    """
    Displays the logged metadata for a single detected airborne object.

    Prints flight, frame, object ID, bounding box, distance, and horizon.

    Args:
        flight_id (str):   ID of the parent flight sequence.
        frame_id  (str):   ID of the frame within the flight.
        obj_id    (str):   Object identifier key (e.g. 'Airplane1', 'Bird2').
        obj       (object): AirborneObjectLocation from detected_object_locations.

    Returns:
        None:
    """
    # Extract All Available Metadata Fields from the Object.
    bb       = getattr(obj, 'bb', None)
    distance = getattr(obj, 'range_distance_m', float('nan'))
    horizon  = getattr(obj, 'is_above_horizon', 'N/A')
    
    # Format Area Safely.
    area = bboxArea(bb)
    area_str = f"{area:.2f}" if area is not None else "N/A"

    # Convert Numeric Horizon Code to a Readable String.
    horizon_lookup = {1: 'Above', -1: 'Below', 0: 'Unclear'}

    # Default to Raw Value if Unrecognised.
    horizon_str = horizon_lookup.get(horizon, str(horizon)) if isinstance(horizon, int) else str(horizon)

    # Print the Log to the Terminal.
    print("Flight    : %s" % flight_id)
    print("Frame     : %s" % frame_id)
    print("Object ID : %s" % obj_id)
    print("BBox      : Area: %s px²" % area_str)
    print("Distance  : %.2f m" % distance)
    print("Horizon   : %s" % horizon_str)


def saveFlightId(flight_id: str) -> None:
    """
    Appends a flight ID to the selected flights file if not already present.

    If the file does not yet exist, it is created with a description header
    before the first ID is written. Reads existing contents first to prevent
    duplicate entries.

    Args:
        flight_id (str): Flight ID to save (e.g. 'part1abc123...').

    Returns:
        None:
    """
    # Fixed Header for the File, Describing the Purpose of this .txt File.
    HEADER = (
        "--------------------------------------------------------------------------------\n"
        "-                           D E S C R I P T I O N                              -\n"
        "--------------------------------------------------------------------------------\n"
        "\n"
        "This txt file contains the selected flight IDs appended from the data browser.\n"
        "These IDs can be used to create custom datasets according to predetermined\n"
        "criteria. This allows harder or more feature-specific datasets to be made for\n"
        "testing the robustness of different models and finding any weaknesses in them.\n"
        "\n"
        "--------------------------------------------------------------------------------\n"
        "\n"
        "# FLIGHT IDs\n"
    )

    # Create File with Header if it Does Not Yet Exist.
    if not os.path.exists(SELECTED_FLIGHTS_PATH):
        # Write the Fixed Header to the File. 
        with open(SELECTED_FLIGHTS_PATH, 'w') as f:
            f.write(HEADER)
        
        # Initialise an Empty Set for Already Saved Flight IDs.
        existing = set()

    else:
        # Read Only Lines Below the Sentinel to Avoid Matching Header Text.
        with open(SELECTED_FLIGHTS_PATH, 'r') as f:
            lines = f.readlines()
        
        # Find the Sentinel Line that Marks the Start of Flight IDs.
        sentinel_idx = next(
            (i for i, l in enumerate(lines) if l.strip() == '# FLIGHT IDs'), None
        )

        # Take all Lines After the Sentinel as the Existing Flight IDs.
        id_lines = lines[sentinel_idx + 1:] if sentinel_idx is not None else lines

        # Clean Lines & Store in a Set for Quick Lookup.
        existing = {l.strip() for l in id_lines if l.strip()}

    # Check if the Flight ID is Already in the File.
    if flight_id in existing:
        log(LogLevel.WARNING, "browse_data:saveFlightId",
            "Flight %s Already in Selected List. Skipping.", flight_id)
        return

    # Append the New Flight ID Below the Header.
    with open(SELECTED_FLIGHTS_PATH, 'a') as f:
        f.write(flight_id + '\n')

    # Print Success Message to Terminal.
    log(LogLevel.INFO, "browse_data:saveFlightId",
        "Flight %s Saved to %s  (%d Total).",
        flight_id, SELECTED_FLIGHTS_PATH, len(existing) + 1)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                             MAIN BROWSING LOOP
# ------------------------------------------------------------------------------
def main(): 
    """
    The main function for running the data browsing interface.
    """
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


    # Retrieve the List of Flight IDs from the Dataset.
    flight_ids    = dataset.get_flight_ids()
    total_flights = len(flight_ids)

    # Display How Many Flights were Discovered in the Metadata.
    log(LogLevel.INFO, "browse_data:main", "Found %d Flights in Metadata.", total_flights)

    # Display Browsing Controls to the User.
    print("Controls: [Enter] Next  |  [s] Skip Flight  |  [a] Save Flight  |  [q] Quit\n")

    # Set View Counter & Stop Flag. 
    entry_count = 0
    stop = False


    # Iterate through Each Flight in the Dataset.
    for flight_idx, flight_id in enumerate(flight_ids):
        # If the User Requested to Stop Browsing, Exit the Loop.
        if stop:
            break
        
        # Load the Flight Object & Retrive all Associated Frames.
        flight = dataset.get_flight(flight_id)
        frames = flight.frames

        # Display Information about the Current Flight Being Inspected.
        log(LogLevel.INFO, "browse_data:main", "Flight %d/%d  (%s)  |  %d Frames",
            flight_idx + 1, total_flights, flight_id, len(frames))

        # Set Flag for Skipping Remainder of Current Flight.
        skip_flight = False

        # Iterate through Each Frame in the Current Flight.
        for frame_id, frame in frames.items():
            # Exit Frame Loop Conditions.
            if stop or skip_flight:
                break

            # Grab the Full Ground-Truth Metadata Needed.
            object_locations = frame.detected_object_locations

            # If the Frame Contains No Airborne Objects, Skip It.
            if not object_locations:
                log(LogLevel.WARNING, "browse_data:main",
                    "Frame %s has No Airborne Objects, Skipping.", frame_id)
                continue
            

            # Iterate through Each Airborne Object Detected in the Frame.
            for obj_id, obj in object_locations.items():
                # Display this Dataset Entry to the User.
                displayEntry(flight_id, frame_id, obj_id, obj)
                entry_count += 1

                # Pause Every N Entries so the User can Interact.
                if entry_count % PAUSE_EVERY_N == 0:
                    # Keep Prompting Until a Valid Command is Entered.
                    while True:
                        try:
                            # Take User Input. 
                            cmd = input("  >> ").strip().lower()
                        except EOFError:
                            # Treat Input Interruption as a Quit Command.
                            cmd = 'q'

                        if cmd == 'q':
                            # User Requested to Quit the Entire Browsing Process.
                            log(LogLevel.INFO, "browse_data:main", "Quit Requested by User.")
                            stop = True
                            break
                        elif cmd == 's':
                            # Skip the Rest of this Flight & Move to the Next One.
                            log(LogLevel.WARNING, "browse_data:main",
                                "Skipping Flight %s.", flight_id)
                            skip_flight = True
                            break
                        elif cmd == 'a':
                            # Save the Parent Flight ID to the Selected List.
                            saveFlightId(flight_id)
                            break
                        else:
                            # Enter (or any other input) Continues to the Next Entry.
                            break


    # Once Browsing Finishes Display Summary.
    log(LogLevel.INFO, "browse_data:main",
        "Browsing Complete. Total Entries Viewed: %d.\n", entry_count)
# ------------------------------------------------------------------------------


# Call the Main Function
if __name__ == "__main__":
    main()
