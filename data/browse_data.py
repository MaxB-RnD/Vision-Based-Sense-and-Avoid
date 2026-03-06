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

import sys
import os

# Allow imports from repo root (core module and logging_module)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dataset import Dataset
from logging_module import *

# ── Configuration ───────────────────────────────────────────────────────────

PAUSE_EVERY_N = 1      # Pause for input after every N entries (1 = one-by-one)
PARTIAL       = True   # True = metadata only, no image downloads

# ── Initialise dataset (downloads ground truth JSONs only) ───────────────────

log(LogLevel.INFO, "browse_data:main", "Loading Dataset Metadata (No Images Will Be Downloaded).")

try:
    dataset = Dataset(
        local_path='../data/part1',
        s3_path='s3://airborne-obj-detection-challenge-training/part1/',
        prefix='part1',
        partial=PARTIAL
    )
    dataset.add(
        local_path='../data/part2',
        s3_path='s3://airborne-obj-detection-challenge-training/part2/',
        prefix='part2',
        partial=PARTIAL
    )
    dataset.add(
        local_path='../data/part3',
        s3_path='s3://airborne-obj-detection-challenge-training/part3/',
        prefix='part3',
        partial=PARTIAL
    )
    log(LogLevel.INFO, "browse_data:main", "Dataset Metadata Loaded Successfully.")
except Exception as e:
    log(LogLevel.ERROR, "browse_data:main", "Failed to Load Dataset Metadata: %s.", str(e))
    sys.exit(1)

# ── Helper ───────────────────────────────────────────────────────────────────

def bbox_area(bb):
    """Return pixel area from [left, top, width, height]."""
    if bb and len(bb) == 4:
        return round(bb[2] * bb[3], 2)
    return None


def log_entry(flight_id, frame_id, obj):
    """Log a single entry's metadata using the project logger."""
    bb          = getattr(obj, 'bb', None)
    distance    = getattr(obj, 'range_distance_m', float('nan'))
    horizon     = getattr(obj, 'is_above_horizon', 'N/A')
    obj_id      = getattr(obj, 'id', 'unknown')
    area        = bbox_area(bb)

    horizon_str = {1: 'Above', -1: 'Below', 0: 'Unclear'}.get(horizon, str(horizon))

    log(LogLevel.INFO,  "browse_data:log_entry", "Flight   : %s", flight_id)
    log(LogLevel.INFO,  "browse_data:log_entry", "Frame    : %s", frame_id)
    log(LogLevel.INFO,  "browse_data:log_entry", "Object ID: %s", obj_id)
    log(LogLevel.DEBUG, "browse_data:log_entry", "BBox     : %s  (area: %s px²)", str(bb), str(area))
    log(LogLevel.DEBUG, "browse_data:log_entry", "Distance : %s m", str(distance))
    log(LogLevel.DEBUG, "browse_data:log_entry", "Horizon  : %s", horizon_str)


# ── Main browsing loop ───────────────────────────────────────────────────────

flight_ids    = dataset.get_flight_ids()
total_flights = len(flight_ids)

log(LogLevel.INFO, "browse_data:main", "Found %d Flights in Metadata.", total_flights)
log(LogLevel.INFO, "browse_data:main", "Controls: [Enter] Next  |  [s] Skip Flight  |  [q] Quit")

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
            log(LogLevel.DEBUG_FULL, "browse_data:main", "Frame %s has No Airborne Objects, Skipping.", frame_id)
            continue

        for obj in objects:
            log_entry(flight_id, frame_id, obj)
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
                    log(LogLevel.INFO, "browse_data:main", "Skipping Flight %s.", flight_id)
                    skip_flight = True
                    break
                # anything else (including Enter) → continue

log(LogLevel.INFO, "browse_data:main", "Browsing Complete. Total Entries Viewed: %d.", entry_count)
