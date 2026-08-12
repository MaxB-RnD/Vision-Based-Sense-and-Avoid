# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: download_data.py %
# Date Created:            %date_created: Sat Mar 07 18:58:21 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 2 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Downloads full flight image sequences for all flight IDs stored in the
# selected flights .txt file produced by browse_data.py. Each flight ID is 
# looked up in the loaded dataset and its images are downloaded from S3 via 
# flight.download(). Flights whose images are already present locally are 
# skipped automatically by the AOT library.
#
# USAGE INSTRUCTIONS:
#   1. Set Global Variables:
#       - Path to the Flight ID Storage File.
#
#   2. Run the Script from Project Root:
#       - python -m src.data.download_data
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
from typing import List

from core.dataset import Dataset
from src.data.original_dataset.browse_data import dataLoader
from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                        CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Path to the .txt File Containing the Selected Flight IDs.
SELECTED_FLIGHTS_PATH = 'configs/datasets/selected_flights.txt'
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         LOAD SELECTED FLIGHT IDS
# ------------------------------------------------------------------------------
def loadFlightIds(path: str) -> List[str]:
    """
    Reads selected flight IDs from the provided .txt file.

    Expects the file to contain a '# FLIGHT IDs' sentinel line, below which
    each non-empty line is treated as a flight ID. Lines above the sentinel 
    are ignored entirely.

    Args:
        path (str): Path to the selected flights .txt file.

    Returns:
        List[str]: Ordered list of flight IDs to download.

    Raises:
        SystemExit: If the file is missing or contains no valid flight IDs.
    """
    # Confirm the File Exists Before Attempting to Read It.
    if not os.path.exists(path):
        # If the File is Missing, Display an Error & Terminate the Program.
        log(LogLevel.ERROR, "download_data:loadFlightIds",
            "Selected Flights File Not Found: '%s'. "
            "Run browse_data.py and Save at Least One Flight with [a] First.\n", path)
        sys.exit(1)

    # Open the File in Read Mode & Load All Lines into Memory.
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find the Sentinel Line.
    sentinel_idx = next((i for i, l in enumerate(lines) if l.strip() == '# FLIGHT IDs'), None)

    # If the Sentinel Line Exists, Only Read the Lines Below it.
    id_lines = lines[sentinel_idx + 1:] if sentinel_idx is not None else lines

    # Clean the Extracted Lines
    flight_ids = [l.strip() for l in id_lines if l.strip()]

    # Exit Early if the List is Empty.
    if not flight_ids:
        # If No Flight IDs were Found, Log an Error & Exit the Program.
        log(LogLevel.ERROR, "download_data:loadFlightIds",
            "No Flight IDs Found in '%s'. "
            "Run browse_data.py and Save at Least One Flight with [a] First.\n", path)
        sys.exit(1)

    # Display a Summary of No. Successfully Loaded Flight IDs.
    log(LogLevel.INFO, "download_data:loadFlightIds",
        "Loaded %d Flight ID(s) from '%s'.", len(flight_ids), path)

    # Return the Flight IDs
    return flight_ids
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               DOWNLOAD FLIGHTS
# ------------------------------------------------------------------------------
def estimateDownloadSize(dataset: Dataset, flight_ids: List[str]) -> float:
    """
    Estimates total download size based on frame count across all flights.

    Uses the metadata already loaded in the dataset object to count frames
    per flight. Assumes ~3MB per frame based on AOT's 2448x2048 resolution.

    Args:
        dataset (Dataset): Loaded dataset object.
        flight_ids (List[str]): Flight IDs to estimate.

    Returns:
        float: Estimated total size in GB.
    """
    # Approximate Storage Size of a Single Frame in Bytes.
    BYTES_PER_FRAME = 2.7 * 1024 * 1024  # ~3MB per Frame at Full Resolution.

    # Running Total of Frames Across All Requested Flights.
    total_frames = 0

    # Iterate through Each Requested Flight ID.
    for flight_id in flight_ids:
        try:
            # Retrieve the Flight Metadata Object from the Dataset.
            flight = dataset.get_flight(flight_id)
            total_frames += len(flight.frames)
        except Exception:
            # If the Flight Metadata Cannot be Retrieved, Fall Back to the
            # Average Flight Length.
            total_frames += 1200  # ~120s at 10fps.
    
    # Convert Total Bytes to Gigabytes before Returning.
    return (total_frames * BYTES_PER_FRAME) / (1024 ** 3)


def downloadFlights(dataset: Dataset, flight_ids: List[str]) -> None:
    """
    Downloads full image sequences for each flight ID in the provided list.

    If a flight ID is not found in the loaded dataset metadata it is logged as 
    a WARNING and skipped. The AOT library handles skipping flights whose images
    are already present locally, so re-running this script is safe.

    Args:
        dataset    (Dataset):   Loaded dataset object.
        flight_ids (List[str]): Flight IDs to download.

    Returns:
        None:
    """
    # Total Number of Flights Requested for Download.
    total = len(flight_ids)

    # Counters Used for the Final Summary Report.
    failed = 0
    skipped = 0
    succeeded = 0

    # Display a Message at the Start of the Download Process.
    print()
    log(LogLevel.INFO, "download_data:downloadFlights",
        "Starting Download of %d Flight(s).", total)

    # Iterate through Each Flight ID with its Index.
    for idx, flight_id in enumerate(flight_ids):
        # Display Progress Information to the User.
        print("Flight %d/%d  %s" % (idx + 1, total, flight_id))

        # Skip the Flight if Already Downloaded.
        if os.path.exists("data/original_dataset/%s/Images/%s" %(flight_id[:5], flight_id)):
            print("            Already Downloaded. Skipping.")
            skipped += 1
            continue
                    
        try:
            # Verify the Flight ID Exists in the Loaded Metadata.
            flight = dataset.get_flight(flight_id)
        except Exception:
            # If Lookup Fails, Treat it as a Missing Flight.
            flight = None

        # If the Flight was Not Found in the Dataset Metadata, Skip It.
        if flight is None:
            # Print Skip Message & Increment Count.
            print("            Not Found in Dataset Metadata. Skipping.")
            skipped += 1
            continue

        try:
            # The AOT Library Handles Downloading the Image Sequence.
            flight.download()
            print("            Downloaded Successfully.")
            succeeded += 1
        except Exception as e:
            # If the Download Fails, Print Error Message & Increase Count.
            print("            Failed to Download Flight: %s." % (str(e)))
            failed += 1

    # Print Final Summary.
    print()
    log(LogLevel.INFO, "download_data:downloadFlights",
        "Download Complete.  Succeeded: %d  |  Skipped: %d  |  Failed: %d.\n",
        succeeded, skipped, failed)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                  MAIN
# ------------------------------------------------------------------------------
def main(selected_flights_path: str = SELECTED_FLIGHTS_PATH):
    """
    The main function for running the data downloader interface.
    """
    # Print a Blank Line for Cleaner Console Output Spacing.
    print()

    # Load the Selected Flight IDs from the .txt File.
    flight_ids = loadFlightIds(selected_flights_path)

    # Load Dataset Metadata.
    dataset = dataLoader(download=False)

    # Print a Blank Line for Cleaner Console Output Spacing.
    print()

    # Display Message Showing No. Flights to Download.
    log(LogLevel.INFO,  "download_data:main",
        "About to Download Images for %d Flight(s).", len(flight_ids))
    
    # Estimate Download Size from Frame Counts.
    estimated_gb = estimateDownloadSize(dataset, flight_ids)

    # Warn the User About the Estimated Total Download Size.
    log(LogLevel.WARNING, "download_data:main",
        "Estimated Download Size: ~%.2f GB  (%d Flight(s)).", estimated_gb, len(flight_ids))

    # Ask the User for Confirmation Before Starting the Download.
    while True:
        # Prompt the User for Input.
        print("\nProceed with Download? (Y/N)")
        confirm = input("  >> ").strip().lower()

        if confirm == 'y':
            # User Confirmed the Download, Exit the Loop & Continue.
            break
        elif confirm == 'n':
            # User Cancelled the Operation, Exit Safely.
            log(LogLevel.INFO, "download_data:main", "Download Cancelled by User.\n")
            sys.exit(0)
        else:
            # Handle Invalid Input.
            log(LogLevel.WARNING, "download_data:main", 
                "Invalid Input. Please Enter Y or N.")

    # Download Images for Each Selected Flight.
    downloadFlights(dataset, flight_ids)
# ------------------------------------------------------------------------------


# Call the Main Function
if __name__ == "__main__":
    main()
