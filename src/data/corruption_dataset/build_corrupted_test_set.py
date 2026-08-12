# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: build_corrupted_test_set.py %
# Date Created:            %date_created: Wed Aug 12 17:56:48 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# FIUX MEEEEEEE
# Builds the gradually-degrading introspection test set by applying continuous,
# ramping image corruptions to a portion of the flights in test.txt (the
# split_dataset.py test split -- unseen during training/validation). This
# script is NON-DESTRUCTIVE: original images under 'data/' are only ever read,
# never modified or deleted. Corrupted frames are written to a completely
# separate output tree (OUTPUT_DATA_ROOT below), mirroring the original
# data/<part>/Images/<flight_id>/ layout so it can be dropped straight into
# your existing dataset-loading code by pointing at a different data root.
#
# This is intentionally scoped to record-keeping + image generation only:
#   - hard_dataset_flights.txt / data_builder.py are NOT touched -- that's a
#     separate part of the project and is left alone.
#   - No metrics are computed here. Point your existing evaluation code at
#     the flight ID lists this script produces.
#
# OUTPUTS:
#   1. One flight ID .txt file per corruption type (same header/sentinel
#      format as test.txt / selected_flights.txt), e.g.:
#         configs/datasets/introspection_corruption/iso_noise_flights.txt
#         configs/datasets/introspection_corruption/defocus_blur_flights.txt
#         configs/datasets/introspection_corruption/color_quantization_flights.txt
#         configs/datasets/introspection_corruption/clean_flights.txt
#   2. A single corruption_manifest.json recording, per corrupted flight, the
#      corruption type, onset frame, ramp length/shape, peak severity, and
#      total frame count -- everything needed later to compute your Failure
#      Onset Frame ground truth and validate AWLT.
#   3. The corrupted image sequences themselves, under OUTPUT_DATA_ROOT, each
#      corruption type nested as OUTPUT_DATA_ROOT/<name>/data/<part>/Images/
#      -- structured this way on purpose, see "RUNNING INFERENCE" below.
#
# RUNNING INFERENCE ON THE CORRUPTED DATA:
#   notebooks/helpers/notebook_helpers.resolve_local_flights(txt_path,
#   project_root) builds its search path as
#   Path(project_root) / "data" / part / "Images", so it can be pointed at
#   the corrupted tree with a ONE-LINE change at the call site -- nothing in
#   notebook_helpers.py or the rest of the notebook needs editing:
#
#       corrupted_root = os.path.join(PROJECT_ROOT, "data_corrupted", "iso_noise")
#       local_flights = resolve_local_flights(
#           "configs/datasets/introspection_corruption/iso_noise_flights.txt",
#           corrupted_root,
#       )
#
#   To confirm at run time that it actually resolved to the corrupted images
#   and not the originals (without touching or deleting anything under the
#   real data/ folder):
#       - Print info["images_dir"] for a few flights straight after calling
#         resolve_local_flights -- it should show a path under
#         data_corrupted/<corruption_name>/data/..., not data/....
#       - After the notebook symlinks each flight into EVAL_DATA_DIR, run
#         `ls -la` (or os.readlink()) on one of those symlinks -- the target
#         it points to is definitive proof of which tree is being read.
#       - Optionally, checksum a post-onset frame (from the manifest) against
#         the same-named file under the real data/ folder -- they should
#         differ, since severity > 0 always changes pixel values.
#
# USAGE INSTRUCTIONS:
#   1. Review CONFIGURATION VARIABLES below, especially OUTPUT_DATA_ROOT.
#   2. Run the Script from Project Root:
#       - python -m src.data.corruption_dataset.build_corrupted_test_set
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
import random

import cv2
import numpy as np
from typing import Dict, List, Optional

from logging_module import LogLevel, log
from src.data.original_dataset.browse_data import dataLoader
from src.data.corruption_dataset.corruptions import CORRUPTION_REGISTRY
from src.data.corruption_dataset.severity_scheduler import buildSeveritySchedule

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Path to the Existing test.txt Split (Unseen During Training/Validation).
INPUT_TEST_FLIGHTS_PATH = 'configs/datasets/training_subsets/test.txt'

# Directory Where the Per-Corruption-Type Flight ID Lists & Manifest are Written.
OUTPUT_RECORDS_DIR = 'configs/datasets/corruption_subsets'

# NON-DESTRUCTIVE OUTPUT ROOT. Corrupted Images are Written Here, Mirroring the
# Original data/<part>/Images/<flight_id>/ Layout. Original 'data/' is Only
# Ever Read, Never Written to or Deleted.
OUTPUT_DATA_ROOT = 'data_corrupted'

# Random Seed. Kept Consistent with split_dataset.py's Default for Traceability.
SEED = 42

# Fraction of test.txt Flights to Corrupt. Remainder Stay Completely Clean.
CORRUPTED_FRACTION = 0.5

# Corruption Types to Include in this Run. Must Match Keys in
# corruptions.CORRUPTION_REGISTRY. Extend by Adding New Keys There First.
ACTIVE_CORRUPTIONS = ['iso_noise', 'defocus_blur', 'colour_quantisation']

# Ramp Configuration. See severity_scheduler.py for Shape Definitions.
RAMP_SHAPE = 'linear'
RAMP_LENGTH_FRAMES = 30     # ~3s at 10fps -- Tune Based on Desired Lead-Up Length.
PEAK_SEVERITY = 1.0

# Onset Frame is Chosen Randomly within this Fraction Range of Each Track's
# Length, Leaving Clean Context Before it and a Post-Ramp "Failure" Region After.
ONSET_FRACTION_RANGE = (0.30, 0.65)

# Tracks Shorter than this (in Frames) are Skipped for Corruption (too Short
# for a Meaningful Onset + Ramp + Post-Ramp Region) and Fall Back to Clean.
MIN_TRACK_LENGTH_FOR_CORRUPTION = RAMP_LENGTH_FRAMES + 20
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               DATA PARSING
# ------------------------------------------------------------------------------
def parseFlightIds(filepath: str) -> List[str]:
    """
    Reads flight IDs from a dataset text file (same format as test.txt).

    Only lines beginning with 'part' are considered valid flight IDs. All
    header, comment, or malformed lines are ignored. Mirrors
    split_dataset.parse_flight_ids for consistency with the rest of the repo.

    Args:
        filepath (str): Path to the dataset file.

    Returns:
        List[str]: List of valid flight IDs.
    """
    # Open File & Read All Lines.
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Filter Only Valid Flight ID Lines.
    flight_ids = [l.strip() for l in lines if l.strip().startswith('part')]

    # Return the Flight IDs.
    return flight_ids
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           FLIGHT -> CORRUPTION ASSIGNMENT
# ------------------------------------------------------------------------------
def assignCorruptions(flight_ids: List[str], corruption_names: List[str],
                      corrupted_fraction: float, seed: int) -> Dict[str, Optional[str]]:
    """
    Randomly assigns each flight ID to a corruption type, or to None (clean).

    Splits flight_ids into a corrupted pool (corrupted_fraction) and a clean
    pool (remainder), then distributes the corrupted pool round-robin across
    corruption_names so each corruption type gets a roughly equal share.

    Args:
        flight_ids (List[str]): All candidate flight IDs (from test.txt).
        corruption_names (List[str]): Corruption type keys to distribute across.
        corrupted_fraction (float): Fraction of flight_ids to corrupt.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, Optional[str]]: Mapping of flight_id -> corruption_name, or
                                  flight_id -> None for clean flights.
    """
    # Create a Shuffled Copy so the Original List Order is Preserved Elsewhere.
    shuffled = flight_ids[:]
    random.seed(seed)
    random.shuffle(shuffled)

    # Split into Corrupted & Clean Pools.
    n_corrupted = round(len(shuffled) * corrupted_fraction)
    corrupted_pool = shuffled[:n_corrupted]
    clean_pool = shuffled[n_corrupted:]

    # Distribute the Corrupted Pool Round-Robin Across Corruption Types.
    assignment: Dict[str, Optional[str]] = {}
    for idx, flight_id in enumerate(corrupted_pool):
        corruption_name = corruption_names[idx % len(corruption_names)]
        assignment[flight_id] = corruption_name

    # Mark the Remainder as Clean.
    for flight_id in clean_pool:
        assignment[flight_id] = None

    # Return the Completed Assignment.
    return assignment
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           IMAGE I/O HELPERS
# ------------------------------------------------------------------------------
def resolveFrameFilename(frame_id: str) -> str:
    """
    Resolves the output filename for a given frame ID.

    NOTE: This assumes frame_id does not already include a file extension and
    that images should be written as PNG (matching the AOT dataset's original
    lossless PNG format). If your Frame objects expose an explicit image path
    or filename attribute, prefer that instead -- adjust this function to use
    it so the corrupted output filenames line up exactly with the originals.

    Args:
        frame_id (str): Frame identifier as used as the key in flight.frames.

    Returns:
        str: Filename (with extension) to use for the corrupted output.
    """
    # Convert Frame ID to a String.
    frame_id = str(frame_id)

    # Use the Frame ID Directly if it Already Has an Image Extension.
    if frame_id.lower().endswith(('.png', '.jpg', '.jpeg')):
        return frame_id

    # Otherwise Default to PNG to Match the AOT Dataset's Original Format.
    return frame_id + '.png'


def partPrefixFromFlightId(flight_id: str) -> str:
    """
    Extracts the dataset part prefix (e.g. 'part1') from a flight ID.

    Mirrors the convention already used in download_data.py, where
    flight_id[:5] resolves to the part directory name.

    Args:
        flight_id (str): Full flight ID (e.g. 'part1abc123...').

    Returns:
        str: Part prefix (e.g. 'part1').
    """
    # Slice the First 5 Characters, Consistent with download_data.py.
    return flight_id[:5]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           SINGLE FLIGHT CORRUPTION
# ------------------------------------------------------------------------------
def corruptFlight(dataset, flight_id: str, corruption_name: str) -> Optional[Dict]:
    """
    Applies a ramping corruption to every frame of a single flight and writes
    the result to OUTPUT_DATA_ROOT, without modifying the original images.

    Args:
        dataset: Loaded AOT Dataset object.
        flight_id (str): Flight ID to corrupt.
        corruption_name (str): Key into corruptions.CORRUPTION_REGISTRY.

    Returns:
        Optional[Dict]: Manifest entry for this flight, or None if the flight
                        could not be processed (missing metadata, too short,
                        or a frame failed to load).
    """
    # Look up the Corruption Function from the Registry.
    corrupt_fn = CORRUPTION_REGISTRY[corruption_name]

    # Retrieve the Flight Object & its Frames.
    try:
        flight = dataset.get_flight(flight_id)
        frames = flight.frames
    except Exception as e:
        log(LogLevel.WARNING, "build_corrupted_test_set:corruptFlight",
            "Could Not Load Metadata for Flight %s: %s. Skipping.", flight_id, str(e))
        return None

    frame_ids = list(frames.keys())
    num_frames = len(frame_ids)

    # Skip Flights too Short for a Meaningful Onset + Ramp + Post-Ramp Region.
    if num_frames < MIN_TRACK_LENGTH_FOR_CORRUPTION:
        log(LogLevel.WARNING, "build_corrupted_test_set:corruptFlight",
            "Flight %s has Only %d Frames (< %d). Skipping Corruption, Treating as Clean.",
            flight_id, num_frames, MIN_TRACK_LENGTH_FOR_CORRUPTION)
        return None

    # Choose the Onset Frame Randomly within the Configured Fraction Range.
    low_frac, high_frac = ONSET_FRACTION_RANGE
    onset_frame = random.randint(
        int(num_frames * low_frac),
        max(int(num_frames * low_frac), int(num_frames * high_frac) - RAMP_LENGTH_FRAMES)
    )
    onset_frame = max(0, min(onset_frame, num_frames - 1))

    # Build the Per-Frame Severity Schedule for this Track.
    severity_schedule = buildSeveritySchedule(
        num_frames=num_frames,
        onset_frame=onset_frame,
        ramp_length=RAMP_LENGTH_FRAMES,
        peak_severity=PEAK_SEVERITY,
        shape=RAMP_SHAPE,
    )

    # Resolve & Create the Output Directory for this Flight.
    #
    # IMPORTANT: notebook_helpers.resolve_local_flights(txt_path, project_root)
    # builds its search path as Path(project_root) / "data" / part / "Images",
    # so a "data" folder is nested one level inside each corruption type's
    # output directory. This lets you point resolve_local_flights at
    # OUTPUT_DATA_ROOT/<corruption_name> as if it were a normal project root,
    # with zero changes to notebook_helpers.py itself -- see the module
    # docstring for the exact call.
    part = partPrefixFromFlightId(flight_id)
    output_dir = os.path.join(OUTPUT_DATA_ROOT, corruption_name, 'data', part, 'Images', flight_id)
    os.makedirs(output_dir, exist_ok=True)

    # Apply the Corruption Frame-by-Frame According to the Severity Schedule.
    frames_written = 0
    for i, frame_id in enumerate(frame_ids):
        severity = severity_schedule[i]

        try:
            # Load the Original Frame as a NumPy Array (Read-Only Access).
            image = frames[frame_id].image(type='numpy')
        except Exception as e:
            log(LogLevel.WARNING, "build_corrupted_test_set:corruptFlight",
                "Could Not Load Frame %s in Flight %s: %s. Skipping Frame.",
                frame_id, flight_id, str(e))
            continue

        # Apply the Corruption Function at this Frame's Severity Level.
        corrupted_image = corrupt_fn(image, severity)

        # Write the Corrupted Frame to the Non-Destructive Output Tree.
        output_filename = resolveFrameFilename(frame_id)
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, corrupted_image)
        frames_written += 1

    # Guard Against a Flight where No Frames Could be Written.
    if frames_written == 0:
        log(LogLevel.ERROR, "build_corrupted_test_set:corruptFlight",
            "No Frames Could be Written for Flight %s. Skipping.", flight_id)
        return None

    # Build & Return the Manifest Entry for this Flight.
    return {
        "flight_id":        flight_id,
        "corruption_type":  corruption_name,
        "num_frames":       num_frames,
        "frames_written":   frames_written,
        "onset_frame":      onset_frame,
        "ramp_shape":       RAMP_SHAPE,
        "ramp_length":      RAMP_LENGTH_FRAMES,
        "peak_severity":    PEAK_SEVERITY,
        "output_dir":       output_dir,
    }
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           RECORD FILE WRITING
# ------------------------------------------------------------------------------
def writeFlightListFile(flight_ids: List[str], filepath: str, description: str) -> None:
    """
    Writes a flight ID list to disk in the same header/sentinel format used
    by test.txt and selected_flights.txt, so it's a drop-in for your existing
    loadFlightIds-style readers.

    Args:
        flight_ids (List[str]): Flight IDs to write.
        filepath (str): Output file path.
        description (str): Short description written into the file header.

    Returns:
        None:
    """
    # Ensure the Output Directory Exists.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Construct the File Header, Matching the Existing Convention.
    header = (
        "--------------------------------------------------------------------------------\n"
        "-                           D E S C R I P T I O N                              -\n"
        "--------------------------------------------------------------------------------\n"
        "\n"
        + description + "\n"
        "\n"
        "This file is auto-generated by build_corrupted_test_set.py. DO NOT EDIT IT\n"
        "MANUALLY. Re-run the script to regenerate.\n"
        "\n"
        "Seed: %d  |  Total: %d\n"
        "\n"
        "--------------------------------------------------------------------------------\n"
        "\n"
        "# FLIGHT IDs\n"
    ) % (SEED, len(flight_ids))

    # Write Header & Flight IDs.
    with open(filepath, 'w') as f:
        f.write(header)
        f.write('\n'.join(flight_ids) + ('\n' if flight_ids else ''))

    # Print Info Message.
    log(LogLevel.INFO, "build_corrupted_test_set:writeFlightListFile",
        "Written: %s (%d flights)", filepath, len(flight_ids))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               MAIN EXECUTION
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Entry point for building the corrupted introspection test set.
    """
    # Print a Blank Line for Cleaner Console Output Spacing.
    print()

    # Validate the Requested Corruptions Exist in the Registry.
    for name in ACTIVE_CORRUPTIONS:
        if name not in CORRUPTION_REGISTRY:
            raise ValueError(
                "ACTIVE_CORRUPTIONS Contains Unknown Key '%s'. Available: %s"
                % (name, list(CORRUPTION_REGISTRY.keys()))
            )

    # Load the test.txt Flight IDs.
    log(LogLevel.INFO, "build_corrupted_test_set:main",
        "Reading: %s", INPUT_TEST_FLIGHTS_PATH)
    flight_ids = parseFlightIds(INPUT_TEST_FLIGHTS_PATH)
    log(LogLevel.INFO, "build_corrupted_test_set:main",
        "Found: %d flight IDs in test.txt", len(flight_ids))

    # Assign Each Flight to a Corruption Type, or to Clean.
    assignment = assignCorruptions(flight_ids, ACTIVE_CORRUPTIONS, CORRUPTED_FRACTION, SEED)

    # Load Dataset Metadata (Locally Cached Only -- No Fresh Download Triggered).
    dataset = dataLoader(download=False)

    # Process Each Flight According to its Assignment, Collecting Manifest Entries.
    manifest_entries = []
    corrupted_by_type: Dict[str, List[str]] = {name: [] for name in ACTIVE_CORRUPTIONS}
    clean_flights: List[str] = []

    total = len(flight_ids)
    for idx, flight_id in enumerate(flight_ids):
        corruption_name = assignment[flight_id]

        if corruption_name is None:
            # This Flight was Assigned to the Clean Pool.
            print("Flight %d/%d  %s  -->  clean" % (idx + 1, total, flight_id))
            clean_flights.append(flight_id)
            continue

        print("Flight %d/%d  %s  -->  %s" % (idx + 1, total, flight_id, corruption_name))
        entry = corruptFlight(dataset, flight_id, corruption_name)

        if entry is None:
            # Corruption Failed or Track was too Short -- Fall Back to Clean.
            clean_flights.append(flight_id)
            continue

        # Record the Successful Corruption.
        manifest_entries.append(entry)
        corrupted_by_type[corruption_name].append(flight_id)

    # Write One Flight ID List per Corruption Type.
    print()
    for corruption_name, ids in corrupted_by_type.items():
        writeFlightListFile(
            ids,
            filepath=os.path.join(OUTPUT_RECORDS_DIR, "%s_flights.txt" % corruption_name),
            description=(
                "Flight IDs from test.txt that were corrupted with '%s' for the\n"
                "introspection gradual-degradation test set." % corruption_name
            ),
        )

    # Write the Clean Flight ID List.
    writeFlightListFile(
        clean_flights,
        filepath=os.path.join(OUTPUT_RECORDS_DIR, "clean_flights.txt"),
        description=(
            "Flight IDs from test.txt that were left uncorrupted (the 'good' half\n"
            "of the introspection gradual-degradation test set)."
        ),
    )

    # Write the Manifest with Per-Flight Corruption Parameters.
    manifest_path = os.path.join(OUTPUT_RECORDS_DIR, "corruption_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump({
            "seed": SEED,
            "corrupted_fraction": CORRUPTED_FRACTION,
            "active_corruptions": ACTIVE_CORRUPTIONS,
            "ramp_shape": RAMP_SHAPE,
            "ramp_length_frames": RAMP_LENGTH_FRAMES,
            "peak_severity": PEAK_SEVERITY,
            "onset_fraction_range": list(ONSET_FRACTION_RANGE),
            "flights": manifest_entries,
        }, f, indent=2)

    log(LogLevel.INFO, "build_corrupted_test_set:main",
        "Manifest Written: %s (%d corrupted flights)", manifest_path, len(manifest_entries))

    # Display Final Summary.
    print()
    log(LogLevel.INFO, "build_corrupted_test_set:main",
        "Done.  Clean: %d  |  Corrupted: %d  (%s)  |  Total: %d.\n",
        len(clean_flights), len(manifest_entries),
        ", ".join("%s: %d" % (k, len(v)) for k, v in corrupted_by_type.items()),
        total)
# ------------------------------------------------------------------------------


# Call the Main Function
if __name__ == "__main__":
    main()
