# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: timing_profiling.py %
# Date Created:            %date_created: Sun Oct 03 22:31:43 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Performance profiling and execution timing utilities for the introspection
# detection and tracking pipelines.
#
# Provides the StageTimer utility for CUDA-synchronised, per-frame and per-stage
# latency measurement, distinguishing between warm inference runs, first-seen
# cold/JIT-compilation frames, and sub-stage component breakdowns. Also includes
# input geometry shape resolution utilities for profiling model FLOPs.
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
import time

from collections import Counter, defaultdict
from contextlib import nullcontext
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch

# ------------------------------------------------------------------------------
#                           CONFIGURATION CONSTANTS
# ------------------------------------------------------------------------------
# Define the Horizontal Symmetric Padding Used By Both Predictor Pipelines.
PADDING = 56  # (16 + 32 + 64) // 2.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                        CUDA SYNCHRONISATION HELPER
# ------------------------------------------------------------------------------
def _sync() -> None:
    """
    Blocks host execution until all queued CUDA device operations complete.

    Guarantees accurate wall-clock stage timing when profiling asynchronous
    GPU kernels. Has no effect when running purely on CPU.
    """
    # Check Whether CUDA is Available.
    if torch.cuda.is_available():
        # Wait for All Queued CUDA Operations to Complete.
        torch.cuda.synchronize()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                          STAGE CONTEXT MANAGER
# ------------------------------------------------------------------------------
class _Stage:
    """
    Context manager that times an isolated pipeline stage with GPU synchronisation.

    Automatically handles pre- and post-execution synchronisation barriers and
    registers the elapsed interval with the owning StageTimer instance upon exit.
    """
    
    def __init__(self, owner: "StageTimer", name: str, also: Optional[str] = None) -> None:
        """
        Initialises the stage timing context.

        Args:
            owner (StageTimer): The parent StageTimer collecting profile metrics.
            name (str): Identifier for the primary stage being timed.
            also (Optional[str]): Optional secondary or hierarchical category label.
        """
        # Store the Parent StageTimer Instance.
        self.o = owner

        # Store the Primary Stage Name.
        self.n = name

        # Store the Optional Secondary Stage Name.
        self.also = also

        # Initialise the Start Time.
        self.t0 = 0.0


    def __enter__(self) -> "_Stage":
        """
        Applies a CUDA synchronisation barrier and captures the initial timestamp.

        Returns:
            _Stage: Active context instance.
        """
        # Synchronise CUDA Before Starting the Timer.
        _sync()

        # Record the Initial High-Resolution Timestamp.
        self.t0 = time.perf_counter()

        # Return the Active Stage Context.
        return self


    def __exit__(self, et, ev, tb) -> bool:
        """
        Applies a post-stage synchronisation barrier and records elapsed latency.

        Args:
            et: Exception type, if an exception occurred within the context.
            ev: Exception value.
            tb: Exception traceback.

        Returns:
            bool: Always False so exceptions propagate normally.
        """
        # Synchronise CUDA Before Stopping the Timer.
        _sync()
        
        # Calculate the Elapsed Stage Duration.
        dt = time.perf_counter() - self.t0
        
        # Only Record Timing Data when No Exception Occurred.
        if et is None:
            # Record the Primary Stage Timing.
            self.o._record(self.n, dt)

            # Record the Secondary Stage Timing when Configured.
            if self.also:
                self.o._record(self.also, dt)
        
        # Return False so any Exception is Propagated.
        return False
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         STAGE TIMER PROFILER CLASS
# ------------------------------------------------------------------------------
class StageTimer:
    """
    Per-stage GPU-synced timer and execution profiler.

    Names without '/' are top-level stages (disjoint, they should sum to the
    frame total). Names with '/' are children or extra breakdowns and are
    excluded from that sum. When disabled, overhead reduces to a nullcontext.
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialises the StageTimer data structures and tracking state.

        Args:
            enabled (bool): Flag toggling active stage profiling.
        """
        self.enabled     = enabled
        self.n_frames    = 0
        self.cur         = None
        self.frames      = {}       # frame_idx -> {stage_name: seconds}
        self.totals      = {}       # frame_idx -> total seconds
        self.meta        = {}       # frame_idx -> {n_crops, n_candidates}
        self.first_seen  = {}       # stage_name -> frame_idx it first appeared in
        self.cold_frames = set()


    def stage(self, name: str, also: Optional[str] = None):
        """
        Returns a context manager scoping a named profiling stage.

        Args:
            name (str): Name of the stage to measure.
            also (Optional[str]): Secondary metric label for conditional grouping.

        Returns:
            _Stage or nullcontext: Timing context manager if enabled.
        """
        # Return the Active Stage Timer when Profiling is Enabled.
        if self.enabled:
            return _Stage(self, name, also)

        # Return a Null Context when Profiling is Disabled.
        return nullcontext()


    def begin_frame(self) -> None:
        """
        Increments the frame sequence counter and activates tracking for the current frame.
        """
        # Check Whether Profiling is Enabled.
        if self.enabled:

            # Increment the Frame Counter.
            self.n_frames += 1

            # Set the Current Frame Identifier.
            self.cur = self.n_frames


    def end_frame(self, total: float) -> None:
        """
        Commits the total wall-clock execution time for the current frame.

        Args:
            total (float): Total frame execution time in seconds.
        """
        # Store the Total Time when a Current Frame is Active.
        if self.enabled and self.cur is not None:
            self.totals[self.cur] = total


    def amend_last_total(self, total: float) -> None:
        """
        Updates or overwrites the recorded total time for the active frame.

        Args:
            total (float): Revised frame execution time in seconds.
        """
        # Update the Current Frame Total.
        self.end_frame(total)


    def set_meta(self, **kw: Any) -> None:
        """
        Attaches custom metadata key-value attributes to the active frame record.

        Args:
            **kw: Arbitrary metadata fields (e.g., n_crops=4, n_candidates=12).
        """
        # Check Whether Profiling is Enabled and a Frame is Active.
        if self.enabled and self.cur is not None:
            # Add the Metadata to the Current Frame.
            self.meta.setdefault(self.cur, {}).update(kw)


    def add(self, name: str, dt: float) -> None:
        """
        Directly accumulates a duration to a named stage on the active frame.

        Args:
            name (str): Profiling stage name.
            dt (float): Duration in seconds to add.
        """
        # Check Whether Profiling is Enabled.
        if self.enabled:
            # Record the Additional Stage Duration.
            self._record(name, dt)


    def _record(self, name: str, dt: float) -> None:
        """
        Internal accumulator tracking duration and flagging cold compilation frames.

        Args:
            name (str): Identifier of the stage.
            dt (float): Duration in seconds.
        """
        # Retrieve the Current Frame Identifier.
        f = self.cur

        # Stop if there is No Active Frame.
        if f is None:
            return

        # Get or Create the Timing Dictionary for the Current Frame.
        d = self.frames.setdefault(f, {})

        # Add the New Duration to the Existing Stage Time.
        d[name] = d.get(name, 0.0) + dt

        # Mark the First Encounter of a Stage as a Cold Or Compile Frame.
        if name not in self.first_seen:
            # Store the First Frame where the Stage was Seen.
            self.first_seen[name] = f
            
            # Mark the Frame as a Cold Frame.
            self.cold_frames.add(f)


    @staticmethod
    def _stats(v: list) -> Dict[str, Any]:
        """
        Computes summary latency statistics in milliseconds from raw second values.

        Args:
            v (list): Sequence of durations in seconds.

        Returns:
            Dict[str, Any]: Dictionary containing count, mean, median, p95, and max in ms.
        """
        # Convert Seconds to Milliseconds.
        a = np.asarray(v, dtype=np.float64) * 1e3

        # Return the Calculated Timing Statistics.
        return dict(
            n         = int(a.size),
            mean_ms   = float(a.mean()),
            median_ms = float(np.median(a)),
            p95_ms    = float(np.percentile(a, 95)),
            max_ms    = float(a.max()),
        )


    def summary(self, warmup_frames: int = 10) -> Dict[str, Any]:
        """
        Aggregates frame timings and stage breakdowns across all evaluated frames.

        Filters out initial warm-up frames as well as cold frames where runtime
        JIT compilations took place, providing stable steady-state latency metrics.

        Args:
            warmup_frames (int): Number of initial frames to discard as burn-in.

        Returns:
            Dict[str, Any]: Comprehensive profile metrics and breakdown dictionary.
        """
        # Exclude Initial Burn-in Frames & any Frames Incurring Cold Compiles.
        kept = [
            f for f in sorted(self.totals)
            if f > warmup_frames and f not in self.cold_frames
        ]

        # Collect all Recorded Stage Names.
        names = sorted({n for d in self.frames.values() for n in d})

        # Calculate Latency Statistics for Each Recorded Stage.
        stages = {}

        # Process Each Recorded Stage.
        for n in names:
            # Collect Valid Timing Values for the Current Stage.
            vals = [self.frames[f][n] for f in kept if n in self.frames.get(f, {})]
            
            # Calculate Statistics when Timing Data Exists.
            if vals:
                stages[n] = self._stats(vals)

        # Identify Top-Level Stages that are Not Sub-Stages.
        top = [n for n in names if "/" not in n]

        # Calculate Unaccounted Frame Time.
        un = [
            self.totals[f] - sum(self.frames.get(f, {}).get(n, 0.0) for n in top)
            for f in kept
        ]

        # Calculate the Total Time Across Kept Frames.
        tot_kept = sum(self.totals[f] for f in kept)

        # Initialise Unaccounted Timing Statistics.
        unacc = None

        # Calculate Unaccounted Timing when Valid Frames Exist.
        if kept:
            unacc = dict(
                mean_ms           = float(np.mean(un) * 1e3),
                median_ms         = float(np.median(un) * 1e3),
                max_ms            = float(np.max(un) * 1e3),
                fraction_of_total = float(sum(un) / tot_kept) if tot_kept > 0 else 0.0,
            )

        # Aggregate Inference Runtimes Grouped by Batch or Crop Count.
        all_counts = Counter(
            m["n_crops"] for f, m in self.meta.items()
            if "n_crops" in m and f in self.totals
        )

        # Initialise Frame Groups by Crop Count.
        groups = defaultdict(list)

        # Add Each Kept Frame to its Crop Count Group.
        for f in kept:
            # Check Whether Crop Count Metadata Exists.
            if "n_crops" in self.meta.get(f, {}):

                # Add The Frame to the Matching Crop Count Group.
                groups[self.meta[f]["n_crops"]].append(f)

        # Initialise Crop Count Summary Results.
        by_n = {}

        # Process Each Observed Crop Count.
        for n in sorted(all_counts):
            # Retrieve Frames Belonging to the Current Crop Count.
            fs = groups.get(n, [])

            # Store Total & Kept Frame Counts.
            row = dict(frames_all=int(all_counts[n]), frames_kept=len(fs))

            # Calculate Timing Statistics when Kept Frames Exist.
            if fs:
                # Calculate Mean Total Frame Time.
                row["mean_frame_total_ms"] = float(np.mean([self.totals[f] for f in fs]) * 1e3)

                # Collect Stage-2 Timing Values.
                s2 = [self.frames[f]["stage2"] for f in fs if "stage2" in self.frames.get(f, {})]

                # Calculate Mean Stage-2 Time.
                row["mean_stage2_ms"] = float(np.mean(s2) * 1e3) if s2 else 0.0

            # Store the Crop Count Group Using a String Key.
            by_n[str(n)] = row

        # Calculate the Total Number of Frames with Crop Metadata.
        total_frames_with_n = sum(all_counts.values())
        
        # Calculate the Average Number of Crops Per Frame.
        mean_crops = (
            sum(n * c for n, c in all_counts.items()) / total_frames_with_n
            if total_frames_with_n else None
        )

        # Count Frames where Candidate Count Exceeded the Crop Limit.
        over_cap = sum(
            1 for f, m in self.meta.items()
            if f in self.totals and m.get("n_candidates", 0) > m.get("n_crops", 0)
        )

        # Return the Complete Profiling Summary.
        return dict(
            warmup_frames                   = warmup_frames,
            n_frames_total                  = len(self.totals),
            n_frames_kept                   = len(kept),
            n_cold_frames                   = len(self.cold_frames),
            stages                          = stages,
            unaccounted                     = unacc,
            frame_total                     = self._stats([self.totals[f] for f in kept]) if kept else None,
            by_crop_count                   = by_n,
            crop_count_frames_total         = int(total_frames_with_n),
            mean_crops_per_frame            = mean_crops,
            frames_with_candidates_over_cap = int(over_cap),
        )


    def save(self, path: str, warmup_frames: int = 10, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Serialises the aggregated profiling summary into a formatted JSON report.

        Args:
            path (str): Destination file path for the output JSON document.
            warmup_frames (int): Number of initial burn-in frames to exclude.
            extra (Optional[Dict[str, Any]]): Extra metadata fields to inject into root.

        Returns:
            Dict[str, Any]: Full JSON payload saved to disk.
        """
        # Create the Output Dictionary with any Additional Metadata.
        out = dict(extra or {})

        # Store the GPU Name or CPU Identifier.
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

        # Add the Profiling Summary to the Output.
        out.update(self.summary(warmup_frames))

        # Ensure Directory Structure Exists Before Writing.
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Open the Output File for Writing.
        with open(path, "w") as f:
            # Write the Profiling Data as Formatted JSON.
            json.dump(out, f, indent=2)

        # Return the Complete Output Dictionary.
        return out
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                      INPUT GEOMETRY PROFILING HELPERS
# ------------------------------------------------------------------------------
def full_frame_input_shape(test_data_path: str, padding: int = PADDING) -> Tuple[int, int, int, int]:
    """
    Infers the full-frame model tensor shape from the first available flight image.

    Inspects the dataset directory structure, loads the initial frame in grayscale,
    and constructs the expected Stage-1 input dimensions (B, C, H, W + 2 * padding).

    Args:
        test_data_path (str): Root path containing flight image subdirectories.
        padding (int): Symmetric horizontal padding in pixels.

    Returns:
        Tuple[int, int, int, int]: Model input shape tuple: (1, 2, H, W + 2 * padding).

    Raises:
        FileNotFoundError: If the folder contains no flight subdirectories or frames.
    """
    # Convert the Input Path to a String.
    root = str(test_data_path)

    # Find all Flight Directories & Sort Them.
    flights = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    # Raise an Error if no Flight Directories Exist.
    if not flights:
        raise FileNotFoundError(f"No flight folders under {root}")

    # Select the First Sorted Flight Directory.
    fdir = os.path.join(root, flights[0])

    # Find & Sort all Files within the Flight Directory.
    frames = sorted(f for f in os.listdir(fdir) if os.path.isfile(os.path.join(fdir, f)))

    # Raise an Error if No Frames are Available.
    if not frames:
        raise FileNotFoundError(f"No image frames found in {fdir}")

    # Read the First Frame in Grayscale.
    img = cv2.imread(os.path.join(fdir, frames[0]), cv2.IMREAD_GRAYSCALE)

    # Raise an Error if the Image Could Not be Read.
    if img is None:
        raise FileNotFoundError(f"Could not read {os.path.join(fdir, frames[0])}")

    # Extract the Image Height & Width.
    h, w = img.shape

    # Return the Expected Full-Frame Model Input Shape.
    return (1, 2, h, w + 2 * padding)
# ------------------------------------------------------------------------------
