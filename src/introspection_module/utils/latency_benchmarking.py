# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: idle_latency_bench.py %
# Date Created:            %date_created: Sun Oct 03 23:45:12 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# First-heavy-after-idle latency benchmarking harness.
#
# Profiles invocation latency when waking heavy ensemble models after variable
# idle periods while keeping both pipelines memory-resident.
#
# Workload Definitions:
#   - Light: HRNet-W32 Tracking-to-Detect (T2D) single-backbone execution.
#   - Heavy: Full multi-model ensemble (predict_ensemble) operating directly
#            on T2D's input buffer under a shared-alignment configuration.
#
# Evaluation Stages:
#   1. Steady State: Interleaved light + heavy passes across consecutive frames
#      to establish steady-state baseline latencies by Stage-2 crop count.
#   2. Light-Only Frame Gaps: Heavy models remain idle while GPU stays active.
#   3. Wall-Clock Sleeps: Complete hardware inactivity testing GPU power/clock
#      downscaling effects and potential CUDA graph/cache desynchronisation.
#
# USAGE INSTRUCTIONS:
#   1. Run Benchmark from Project Root:
#       - python -m src.introspection_module.idle_latency_bench
#
#   2. Optional Environment Variables:
#       - BENCH_FLIGHT: Flight ID to process for frame sequences.
#       - BENCH_STEADY_FRAMES: Number of warmup frames for baseline estimation.
#       - BENCH_GAPS: Comma-separated frame gap intervals (e.g. "5,30,100").
#       - BENCH_SLEEPS: Comma-separated wall-clock sleep durations in seconds.
#       - BENCH_REPEATS: Number of repetitions per frame-gap trial.
#       - BENCH_SLEEP_REPEATS: Number of repetitions per sleep trial.
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
import json
import time
import subprocess
from typing import Tuple, Optional

import cv2
import numpy as np
import torch

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Default Target Flight Identifier for Frame Iteration.
FLIGHT = os.getenv("BENCH_FLIGHT", "part1036edfd6ece44c37b8ca9562fd701682")

# Number of Continuous Interleaved Frames to Establish Steady Baseline.
STEADY = int(os.getenv("BENCH_STEADY_FRAMES", "150"))

# Gap Intervals in Number of Light-Only Frames Before Heavy Invocation.
GAPS = [int(x) for x in os.getenv("BENCH_GAPS", "5,30,100").split(",")]

# Wall-Clock Hardware Sleep Intervals in Seconds Before Heavy Invocation.
SLEEPS = [int(x) for x in os.getenv("BENCH_SLEEPS", "10,60").split(",")]

# Repetition Counts Per Evaluation Condition.
REPEATS       = int(os.getenv("BENCH_REPEATS", "4"))
SLEEP_REPEATS = int(os.getenv("BENCH_SLEEP_REPEATS", "3"))

# Destination Directory for Benchmark Output Reports.
OUT = "outputs/results/introspection_module/step4_checks"

# Symmetric Horizontal Padding Applied to Input Buffer (16 + 32 + 64) // 2.
PAD = 56
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                        CUDA SYNCHRONISATION HELPERS
# ------------------------------------------------------------------------------
def sync() -> None:
    """
    Enforces a strict CUDA host-device execution barrier.
    """
    # Check Whether CUDA is Available on the Host.
    if torch.cuda.is_available():
        # Wait for All Pending GPU Operations to Complete.
        torch.cuda.synchronize()


def sm_clock() -> Tuple[Optional[int], Optional[int]]:
    """
    Queries current and maximum Streaming Multiprocessor (SM) clock speeds.

    Returns:
        Tuple[Optional[int], Optional[int]]: (current_sm_mhz, max_sm_mhz) or
        (None, None) if nvidia-smi is unavailable.
    """
    # Try Querying GPU Clocks Using nvidia-smi.
    try:
        # Execute the nvidia-smi Query Subprocess.
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().split("\n")[0]

        # Parse Current & Maximum SM Clock Frequencies.
        cur, mx = [int(x) for x in out.split(",")]

        # Return the Measured Clock Frequencies.
        return cur, mx

    # Return None When nvidia-smi Fails or Times Out.
    except Exception:
        return None, None
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       BENCHMARK EXECUTION PIPELINE
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Executes the idle-to-heavy latency benchmarking suite.
    """
    # Set the Default Environment Run Identifier.
    os.environ.setdefault("DATASET_ENV", "step4_latency")

    # Import Predictors & Evaluation Modules Locally.
    from src.introspection_module.hrnetw32_introspective.hrnetw32_track_to_detect_test import (
        HRNetW32TrackToDetectPredictor,
    )
    from src.introspection_module.winning_solution_star.seg_test import WinningSolutionPredictor
    from src.introspection_module.winning_solution_star.seg_tracker import predict_ensemble as pe

    # Initialise the Lightweight T2D Predictor.
    t2d = HRNetW32TrackToDetectPredictor()
    t2d.inference_setup()

    # Initialise the Heavy Ensemble Predictor.
    heavy_pred = WinningSolutionPredictor()
    heavy_pred.inference_setup()

    # Extract the Underlying Detector Instance.
    det = heavy_pred.detector

    # Discover and Sort All Available Flight Images.
    names = t2d.get_all_frame_images(FLIGHT)

    # Reset Predictor Tracking State for the Flight.
    t2d.flight_started()

    # Initialise Frame Iteration State Container.
    state = dict(i=0, prev=None)


    def light() -> float:
        """
        Executes a single-frame forward pass using the lightweight T2D predictor.

        Returns:
            float: Wall-clock execution time in seconds.
        """
        # Get the Current Frame Index.
        i = state["i"]

        # Stop When Reaching the End of the Flight Sequence.
        if i >= len(names):
            raise StopIteration

        # Read the Current Grayscale Flight Image.
        frame = cv2.imread(str(t2d.get_frame_image_location(FLIGHT, names[i])))

        # Synchronise Before Starting the Timer.
        sync()

        # Record the Start Timestamp.
        t0 = time.perf_counter()

        # Execute the Lightweight Forward Pass on the Frame.
        t2d.run_model_on_frame(frame, state["prev"], names[i], FLIGHT)

        # Synchronise Before Recording the End Timestamp.
        sync()

        # Calculate Elapsed Inference Latency.
        dt = time.perf_counter() - t0

        # Update Frame History & Advance the Frame Index.
        state["prev"], state["i"] = frame, i + 1

        # Return Elapsed Duration.
        return dt


    def heavy() -> Tuple[float, int]:
        """
        Executes the full ensemble detector directly on T2D's input buffer.

        Returns:
            Tuple[float, int]: (execution_time_seconds, crop_count).
        """
        # Synchronise Before Starting the Timer.
        sync()

        # Record the Start Timestamp.
        t0 = time.perf_counter()

        # Execute Two-Stage Ensemble Inference on the Shared Input Buffer.
        objs = pe.predict_ensemble(
            X                  = t2d._input_buffer,
            models_full_res    = [det.model_seg1, det.model_seg2],
            models_crops       = [det.model_seg3, det.model_dla],
            full_res_threshold = 0.35,
            x_offset           = -PAD,
            device             = det.device,
        )

        # Synchronise Before Recording the End Timestamp.
        sync()

        # Return Elapsed Latency and Number of Evaluated Crops.
        return time.perf_counter() - t0, len(objs)

    # Warm Frame 0 (Requires Previous Frame for Warping Alignment).
    light()


    # Initialise Storage for Steady-State Measurement Tuples.
    steady = []

    # Run Interleaved Steady-State Frames.
    for _ in range(STEADY):
        # Execute the Light Model Pass.
        l = light()
    
        # Execute the Heavy Ensemble Pass.
        h, n = heavy()

        # Store Crop Count, Heavy Latency, and Light Latency.
        steady.append((n, h, l))

    # Discard First 10 Frames to Allow GPU Power/Clocks to Stabilise.
    steady = steady[10:]

    # Dictionary Mapping Crop Counts to Baseline Latency Statistics.
    base = {}

    # Compute Median and P95 Latency for Each Unique Crop Count.
    for n in sorted({s[0] for s in steady}):
        # Filter Heavy Durations Matching Current Crop Count.
        hs = [s[1] for s in steady if s[0] == n]
    
        # Calculate Statistics for the Current Crop Count.
        base[n] = dict(
            n         = len(hs),
            median_s  = float(np.median(hs)),
            p95_s     = float(np.percentile(hs, 95)),
        )
    
    # Calculate Median Light Execution Latency Across Steady Runs.
    light_med = float(np.median([s[2] for s in steady]))

    # Query Active SM Clocks Under Heavy GPU Load.
    clk_busy, clk_max = sm_clock()

    # Print Formatted Steady Baseline Summary Metrics.
    print(
        "steady baseline (heavy ms by crop count): "
        + ", ".join(f"n={n}: {b['median_s']*1e3:.1f} (N={b['n']})" for n, b in base.items())
    )
    print(f"steady light median {light_med*1e3:.1f} ms; SM clock busy {clk_busy}/{clk_max} MHz")


    # Initialise Storage for All Experimental Idle Trials.
    trials = []


    def trial(kind: str, amount: int) -> None:
        """
        Executes an idle condition followed by a single heavy measurement.

        Args:
            kind (str): Type of idle interval ('frames' or 'sleep_s').
            amount (int): Magnitude of the idle gap (frame count or seconds).
        """
        # Handle Active-GPU Frame Gaps.
        if kind == "frames":
            # Execute Intermediate Light-Only Passes.
            for _ in range(amount):
                light()

            # SM Clock Remains Active; Set Clock to None.
            clk = None

        # Handle Full Inactivity Hardware Sleep Intervals.
        else:
            # Synchronise All Device Operations Before Sleeping.
            sync()
            
            # Sleep the Host Process for the Prescribed Duration.
            time.sleep(amount)
            
            # Query GPU SM Clock Immediately Upon Wakeup.
            clk, _ = sm_clock()

        # Execute Post-Idle Light Forward Pass.
        l = light()
        
        # Execute Post-Idle Heavy Forward Pass.
        h, n = heavy()

        # Store Trial Parameters & Measured Runtimes.
        trials.append(
            dict(
                kind                = kind,
                amount              = amount,
                light_s             = l,
                heavy_s             = h,
                n_crops             = n,
                sm_clock_after_idle = clk,
            )
        )

    # Execute Light-Only Frame Gap Trials.
    for rep in range(REPEATS):
        for g in GAPS:
            if state["i"] + g + 2 > len(names):
                break
            trial("frames", g)

    # Execute Hardware Inactivity Sleep Trials.
    for rep in range(SLEEP_REPEATS):
        for s in SLEEPS:
            if state["i"] + 3 > len(names):
                break
            trial("sleep_s", s)


    # Process Performance Ratios and Diagnostic Flags for Each Trial.
    for t in trials:
        # Retrieve Baseline Entry for the Observed Crop Count.
        b = base.get(t["n_crops"])
        # Check Whether Sufficient Baseline Observations Exist.
        have = b is not None and b["n"] >= 5

        # Record Ratios & Statistics.
        t["baseline_heavy_s"] = b["median_s"] if have else None
        t["heavy_excess_s"]   = t["heavy_s"] - b["median_s"] if have else None
        t["heavy_ratio"]      = t["heavy_s"] / b["median_s"] if have else None
        t["light_ratio"]      = t["light_s"] / light_med
        t["stall"]            = bool(
            (have and t["heavy_s"] > 5 * b["median_s"] and t["heavy_s"] > 0.3)
            or (not have and t["heavy_s"] > 1.0)
        )

    # Print Table Header for Idle Latency Comparison.
    print("\nfirst heavy call after idle (vs steady state at the same crop count):")
    print(
        f"{'gap':>14s} {'N':>3s} {'med ratio':>10s} {'max ratio':>10s} {'med excess ms':>14s} "
        f"{'max excess ms':>14s} {'light ratio med':>16s} {'stalls':>7s}"
    )

    # Group Evaluation Trials by Idle Condition.
    groups = {}
    for t in trials:
        groups.setdefault((t["kind"], t["amount"]), []).append(t)

    # Initialise Storage for Condition Summary Records.
    summary = []

    # Compute Statistical Aggregations for Each Idle Condition.
    for (kind, amount), ts in groups.items():
        # Collect Valid Ratios and  Statisitcs.
        rs = [t["heavy_ratio"] for t in ts if t["heavy_ratio"] is not None]
        ex = [t["heavy_excess_s"] * 1e3 for t in ts if t["heavy_excess_s"] is not None]
        lr = [t["light_ratio"] for t in ts]
        st = sum(t["stall"] for t in ts)

        # Build Descriptive Condition Label.
        label = f"{amount} frames" if kind == "frames" else f"{amount} s sleep"

        # Construct Summary Dictionary for the Condition.
        row = dict(
            gap                 = label,
            n                   = len(ts),
            median_ratio        = float(np.median(rs)) if rs else None,
            max_ratio           = float(np.max(rs)) if rs else None,
            median_excess_ms    = float(np.median(ex)) if ex else None,
            max_excess_ms       = float(np.max(ex)) if ex else None,
            median_light_ratio  = float(np.median(lr)),
            stalls              = int(st),
        )

        # Store Summary Row.
        summary.append(row)

        # Formatter Helper for Floating Point Outputs.
        f = lambda v, p: "n/a" if v is None else f"{v:.{p}f}"

        # Print Formatted Metric Row.
        print(
            f"{label:>14s} {len(ts):>3d} {f(row['median_ratio'], 2):>10s} {f(row['max_ratio'], 2):>10s} "
            f"{f(row['median_excess_ms'], 1):>14s} {f(row['max_excess_ms'], 1):>14s} "
            f"{row['median_light_ratio']:>16.2f} {st:>7d}"
        )

    # Compute Total Number of Stalls Encountered Across All Trials.
    stalls = sum(t["stall"] for t in trials)


    # Ensure Destination Directory Exists.
    os.makedirs(OUT, exist_ok=True)

    # Write Complete Performance Log and Trial Breakdown to JSON.
    with open(os.path.join(OUT, "idle_latency.json"), "w") as fh:
        # Serialise JSON Report.
        json.dump(
            dict(
                gpu                   = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                flight                = FLIGHT,
                sm_clock_busy         = clk_busy,
                sm_clock_max          = clk_max,
                steady_baseline       = {str(k): v for k, v in base.items()},
                steady_light_median_s = light_med,
                summary               = summary,
                trials                = trials,
            ),
            fh,
            indent=2,
        )

    # Print Final Verification Results and Exit Status.
    print(f"\n[latency] {len(trials)} trials, {stalls} stall(s) (>5x baseline and >300 ms). -> {OUT}/idle_latency.json")
    print("[latency] " + ("NO COMPILE STALLS" if stalls == 0 else "STALLS FOUND: warm-up is incomplete"))

    # Return Failure Code if Any Stalls Were Detected.
    sys.exit(1 if stalls else 0)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------
