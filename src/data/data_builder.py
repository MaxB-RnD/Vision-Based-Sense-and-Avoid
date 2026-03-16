# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: data_builder.py %
# Date Created:            %date_created: Fri Mar 15 17:11:21 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Automatically builds a custom hard dataset by scanning all AOT ground truth
# metadata and applying configurable filter criteria to isolate the most
# challenging flight sequences. Matching flight IDs are saved to a .txt file
# for use with download_data.py.
#
# FILTER CRITERIA (toggled via CONFIGURATION VARIABLES):
#   - Horizon Position : Below or Unclear Horizon Classifications.
#   - Multiple Objects : Frames Containing More than One Detected Object.
#   - Small Objects    : Bounding Box Area Below a Configurable Threshold.
#
# USAGE INSTRUCTIONS:
#   1. Set Global Variables:
#       - Path to the Hard Dataset Flight ID Storage File.
#       - Filter Thresholds for Each Criteria.
#
#   2. Run the Script from Project Root:
#       - python -m src.data.data_builder
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

from logging_module import LogLevel, log
from src.data.browse_data import dataLoader, saveFlightId, bboxArea

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Path to the File Where Hard Dataset Flight IDs are Saved.
SELECTED_FLIGHTS_PATH = 'data/hard_dataset_flights.txt'

# Select True to Include Flights where Any Object is Below Horizon or Unclear.
FILTER_HORIZON = True

# Include Flights where Any Frame has More than 1 Detected Object.
FILTER_MULTIPLE_OBJECTS = True

# Include Flights where Any Frame has No Detected Object.
FILTER_NO_OBJECTS = True

# Include Flights where Any Object Bbox Area is Below this Threshold (px²).
FILTER_SMALL_OBJECTS = True
SMALL_OBJECT_THRESHOLD = 37.0
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           DATA REFINING HELPERS
# ------------------------------------------------------------------------------
def isBelowOrUnclearHorizon(obj: object) -> bool:
    """
    Returns True if the detected object is below or unclear relative to the horizon.

    Horizon codes sourced from the AOT dataset:
        1  = Above Horizon.
        0  = Unclear.
        -1  = Below Horizon.

    Args:
        obj (object): AirborneObjectLocation from detected_object_locations.

    Returns:
        bool: True if the object is below or unclear, False if above.
    """
    # Retrieve Horizon Attribute. Return None if it Does Not Exist.
    horizon = getattr(obj, 'is_above_horizon', None)

    # Return True for Below (-1) or Unclear (0), False for Above (1) or Unknown.
    return isinstance(horizon, int) and horizon in (-1, 0)


def hasMultipleObjects(frame: object) -> bool:
    """
    Returns True if the frame contains more than one detected airborne object,
    regardless of whether they are planned or unplanned encounters.

    Args:
        frame (object): AOT frame object from flight.frames.

    Returns:
        bool: True if two or more objects are detected, False otherwise.
    """
    # Retrieve Detected Object Locations from the Frame. 
    # Return empty dict if Attribute Does Not Exist.
    object_locations = getattr(frame, 'detected_object_locations', {})

    # Return True if More than One Object is Present in the Frame.
    return len(object_locations) > 1


def hasNoObjects(frame: object) -> bool:
    """
    Returns True if the sequence contains no airborne object in at least one 
    frame.

    Args:
        frame (object): AOT frame object from flight.frames.

    Returns:
        bool: True if no objects are detected, False otherwise.
    """
    # Retrieve Detected Object Locations from the Frame. 
    # Return empty dict if Attribute Does Not Exist.
    object_locations = getattr(frame, 'detected_object_locations', {})

    # Return True if No Object is Present in the Frame.
    return len(object_locations) == 0


def isSmallObject(obj: object) -> bool:
    """
    Returns True if the detected object's bounding box area is below
    SMALL_OBJECT_THRESHOLD.

    Args:
        obj (object): AirborneObjectLocation from detected_object_locations.

    Returns:
        bool: True if the bounding box area is below the threshold.
    """
    # Calculate Bbox Area. Pass None if it Does Not Exist.
    area = bboxArea(getattr(obj, 'bb', None))

    # Return True if Area is Valid and Below the Threshold.
    return area is not None and area < SMALL_OBJECT_THRESHOLD
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                             MAIN BUILDING LOOP
# ------------------------------------------------------------------------------
def main():
    """
    The main function for building a custom hard dataset.
    """
    # Load Dataset Metadata.
    dataset = dataLoader(download=False)

    # Retrieve the List of All Flight IDs from the Dataset.
    flight_ids = dataset.get_flight_ids()
    total_flights = len(flight_ids)

    # Display How Many Flights were Discovered in the Metadata.
    print()
    log(LogLevel.INFO, "data_builder:main", 
        "Scanning %d Flights for Hard Dataset Criteria.\n", total_flights)

    # Set Counters for the Final Summary Report.
    flights_saved = 0
    flights_skipped = 0
    average_length = 0

    # Iterate through Every Flight in the Dataset.
    for flight_idx, flight_id in enumerate(flight_ids):
        # Display Scan Progress to the User.
        print("Checking Flight %d/%d  (%s)" % (flight_idx + 1, total_flights, flight_id))

        # Load the Flight Object & Retrieve All Associated Frames.
        flight = dataset.get_flight(flight_id)
        frames = flight.frames

        # Set Flag to Track Whether this Flight Qualifies for the Hard Dataset.
        qualifies = False

        # Iterate through Each Frame in the Current Flight.
        for frame_id, frame in frames.items():
            # Grab the Full Ground-Truth Metadata Needed.
            object_locations = frame.detected_object_locations
            
            # Check Multiple Objects Filter at Frame Level.
            if FILTER_MULTIPLE_OBJECTS and hasMultipleObjects(frame):
                qualifies = True
                break

            # Check No Objects Filter at Frame Level.
            if FILTER_NO_OBJECTS and hasNoObjects(frame):
                qualifies = True
                break

            # Check Object-Level Filters.
            for obj_id, obj in object_locations.items():
                # Checks if Object is Below OR Unclear Relative to the Horizon.
                if FILTER_HORIZON and isBelowOrUnclearHorizon(obj):
                    qualifies = True
                    break

                # Checks if the Object is Considered Small Based on Bbox Area.
                if FILTER_SMALL_OBJECTS and isSmallObject(obj):
                    qualifies = True
                    break
             
            # Checks if the Flight has Already Met Any Filter Condition.
            if qualifies:
                break

        # Save the Flight ID if it Met the Criteria.
        if qualifies:
            saveFlightId(flight_id, SELECTED_FLIGHTS_PATH)
            flights_saved += 1
            average_length += len(frames)
        else:
            flights_skipped += 1
    
    # Calculate the Average Sequence Length.
    if flights_saved > 0:
        average_length = average_length / (flights_saved * 10.0)
    else:
        average_length = 0

    # Display Final Summary.
    print()
    log(LogLevel.INFO, "data_builder:main",
        "Scan Complete.  Saved: %d  |  Skipped: %d  |  Average Sequence Length: %.2fs.\n", 
        flights_saved, flights_skipped, average_length)
# ------------------------------------------------------------------------------


# Call the Main Function
if __name__ == "__main__":
    main()
