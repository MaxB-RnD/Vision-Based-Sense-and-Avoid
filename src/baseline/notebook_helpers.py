# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: notebook_helpers.py %
# Date Created:            %date_created: Mon Mar 23 21:38:21 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Helper functions for the custom local baseline inference notebook.
# Provides utilities for:
#   - Parsing flight ID .txt files produced by data_builder.py.
#   - Resolving flight IDs to locally downloaded folder paths.
#   - Managing auto-incrementing run output directories.
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
import re
import json

from pathlib import Path
from typing import Optional
from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                           FLIGHT ID PARSING
# ------------------------------------------------------------------------------
def parse_flight_txt(txt_path: str) -> list:
    """
    Parses a flight ID .txt file produced by data_builder.py and returns a
    list of (part, flight_hash) tuples for every valid flight entry.

    Flight IDs in the file are formatted with the part prefix fused directly
    onto the hash:

      part100bb96a5a68f4fa5bc5c5dc66ce314d2
      part2dea1a03fdfe54981b103569bd20a5e95

    Lines that form part of the header block or do not begin with a recognised
    part prefix are silently ignored.

    Args:
        txt_path (str): Path to the flight-ID .txt file.

    Returns:
        list: A list of (part, flight_hash) tuples, e.g.:
              [("part1", "00bb96a5a68f4fa5bc5c5dc66ce314d2"), ...]
    """
    # Store Parsed (part, flight_hash) Pairs.
    entries = []

    # Open the Flight ID File for Reading.
    with open(txt_path) as f:
        # Remove Leading/Trailing Whitespace & Newline Characters.
        for line in f:
            line = line.strip()

            # Check Each Known Part Prefix and Extract the Trailing Hash.
            for part in ["part1", "part2", "part3"]:
                # Ensure the Line Starts with the Part Prefix & Contains a Hash After it.
                if line.startswith(part) and len(line) > len(part):
                    flight_hash = line[len(part):]
                    entries.append((part, flight_hash))
                    break
                
    # Return All Valid Parsed Entries.
    return entries
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           RUN DIRECTORY MANAGEMENT
# ------------------------------------------------------------------------------
def get_next_run_dir(base_dir: str) -> str:
    """
    Returns the name of the next available auto-incrementing run directory
    under base_dir, without creating it.

    Directories are named run0, run1, run2, ... The returned name is the
    first index whose path does not yet exist on disk.

    Args:
        base_dir (str): Path to the parent directory that holds run folders.

    Returns:
        str: The next available run directory name, e.g. "run3".
    """
    # Start from Run Index 0.
    i = 0

    # Increment Until a Run Index that Does Not Yet Exist is Found.
    while os.path.exists(os.path.join(base_dir, f"run{i}")):
        i += 1

    # Return the First Available Run Directory Name.
    return f"run{i}"


def get_last_run(results_dir: str) -> Optional[str]:
    """
    Returns the name of the most recently created run directory under
    results_dir, determined by the highest numeric suffix.

    Args:
        results_dir (str): Path to the directory containing run folders.

    Returns:
        Optional(str): The name of the highest-numbered run folder (e.g. "run4"),
                    or None if no run folders are found.
    """
    # List to Store Tuples for Valid Run Directories.
    runs = []

    # Iterate Over all Entries in the Results Directory.
    for name in os.listdir(results_dir):
        # Check if the Folder Name Matches the 'runN' Pattern.
        match = re.match(r"run(\d+)", name)
        if match:
            runs.append((int(match.group(1)), name))

    # Return None if No Run Folders Exist Yet.
    if not runs:
        return None

    # Return the Run Folder with the Highest Numeric Index.
    return max(runs, key=lambda x: x[0])[1]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           GROUND TRUTH LOADING
# ------------------------------------------------------------------------------
def load_combined_gt(parts: list, local_flights: dict, project_root: str) -> dict:
    """
    Loads and merges the groundtruth.json file from each part's ImageSets/
    folder, filtering down to only the flights that are present on disk.
 
    Each part's groundtruth.json contains entries for every flight registered
    in that part of the AOT dataset, including flights that have not been
    downloaded. Entries whose key is not associated with a local flight are
    silently skipped - this is expected behaviour, not an error.
  
    Args:
        parts (list): List of part names to load, e.g. ["part1", "part2", "part3"].
        local_flights (dict): The local_flights dict returned by resolve_local_flights(), 
                              keyed by folder name with each value containing 'part' 
                              and 'flight_hash'.
        project_root  (str): Absolute path to the project root directory.
 
    Returns:
        dict: A merged ground truth dict in the standard AOT format:
              { "samples": { flight_hash: <annotations>, ... } }
              containing only the flights present on disk.
    """
    # Initialise the Combined GT Dict. & Tracking Variables.
    combined = {"samples": {}}
    local_hashes_by_part = {}
    total_in_file = 0
    total_kept = 0
 
    # Build a Lookup of Flight Hashes per Part to Avoid Duplicates.
    for info in local_flights.values():
        part = info["part"]
        local_hashes_by_part.setdefault(part, set()).add(info["flight_hash"])
  
    # Iterate through Each Part Specified.
    for part in parts:
        # Path to the Part's Ground Truth JSON.
        gt_path = Path(project_root) / "data" / part / "ImageSets" / "groundtruth.json"
 
        # Skip if the Ground Truth File is Missing for this Part.
        if not gt_path.exists():
            log(LogLevel.WARNING, "notebook_helpers:load_gt", "Not Found, Skipping: %s", str(gt_path))
            continue
 
        # Load the Ground Truth JSON for this Part.
        with open(gt_path) as f:
            gt = json.load(f)
 
        # Get the Set of Local Flight Hashes for this Part.
        part_hashes = local_hashes_by_part.get(part, set())
 
        # Merge Only the Flights Whose Hash Matches a Local Flight in This Part.
        for flight_key, data in gt["samples"].items():
            # Count Every Flight Read from File.
            total_in_file += 1

            # If Key and Hash are Found Merge.
            if flight_key in part_hashes:
                combined["samples"][flight_key] = data
                total_kept += 1

    # Print Final Summary Message.
    print()
    log(LogLevel.INFO, "notebook_helpers:load_gt",
        "%d kept / %d total.", total_kept, total_in_file)

    # Return the Combined Filtered Ground Truth.
    return combined
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           FLIGHT RESOLUTION
# ------------------------------------------------------------------------------
def resolve_local_flights(txt_path: str, project_root: str) -> dict:
    """
    Cross-references the flight IDs listed in txt_path against what is
    actually present on disk under project_root/data/<part>/Images/.

    Only flights that are both listed in the .txt file AND have a matching
    folder on disk are included in the returned dict. Flights listed in the
    .txt that have not been downloaded are logged and skipped.

    Args:
        txt_path (str): Path to the flight-ID .txt file.
        project_root (str): Absolute path to the project root directory.

    Returns:
        dict: A dict keyed by flight folder name, where each value contains:
              {
                  "images_dir"  : Path,   # absolute path to the Images sub-folder
                  "part"        : str,    # e.g. "part1"
                  "flight_hash" : str,    # the hash portion of the flight ID
              }
    """
    # Dict & List to Store Flights that Exist/Missing.
    found   = {}
    missing = []

    # Parse Each Tuple from the .txt File.
    for part, flight_hash in parse_flight_txt(txt_path):
        # Path to the Images Folder for this Part.
        images_root = Path(project_root) / "data" / part / "Images"

        # Skip Immediately if the Entire Part Images Folder is Absent.
        if not images_root.exists():
            missing.append(f"{part}/{flight_hash}")
            continue

        # Search for Any Sub-Folder Whose Name Contains the Flight Hash.
        matches = [
            d for d in images_root.iterdir()
            if d.is_dir() and flight_hash in d.name
        ]

        if matches:
            # Use the First Match (there should only ever be one per hash).
            folder = matches[0]

            # Store Metadata for this Flight in the Found Dict.
            found[folder.name] = {
                "images_dir"  : folder,
                "part"        : part,
                "flight_hash" : flight_hash,
            }
        else:
            # No Matching Folder Found, Add to Missing List.
            missing.append(f"{part}/{flight_hash}")

    # Log Summary of Found Vs. Mssing Flights.
    log(LogLevel.INFO, "notebook_helpers:resolve_flights",
        f"{len(found)} Flights on Disk. {len(missing)} Not Found. "
        f"(Run the Download Cell to Fetch Missing Flights).")

    # Return Dict of Flights that Exist Locally.
    return found
# ------------------------------------------------------------------------------
