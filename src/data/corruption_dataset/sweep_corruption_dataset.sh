#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: sweep_corruption_dataset.sh %
# Date Created:            %date_created: Sat Aug 29 23:12:45 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 2 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Runs build_corrupted_test_set.py either once using its default configuration 
# or across every parameter combination defined in the sweep arrays below. Each 
# run removes the previously generated dataset and records, then creates a fresh
# dataset and saves its parameters, generated records, and build log under a 
# dedicated runN directory. The original dataset is never modified. The first 
# sweep combination should normally use corrupted-fraction=0.0 to provide an 
# uncorrupted baseline for comparison with the remaining runs.

# USAGE:
#   1. Edit the SWEEP ARRAYS below to define the parameter grid.
#
#   2. Make the Script Executable:
#         chmod +x src/data/corruption_dataset/sweep_corruption_dataset.sh
#
#   3. Run the Full Sweep from the Project Root:
#         nohup bash src/data/corruption_dataset/sweep_corruption_dataset.sh --sweep > outputs/logs/corruption_dataset/sweep.log 2>&1 &
#
#      Or a Single Default Run (no Sweep Arrays Used):
#         nohup bash src/data/corruption_dataset/sweep_corruption_dataset.sh > outputs/logs/corruption_dataset/single_run.log 2>&1 &
#
#   4. Then Monitor from Terminal with:
#         tail -f outputs/logs/corruption_dataset/sweep.log
#         (or the Per-Run Log Listed at the Start of Each Run Below)
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

# Treat Unset Variables as Errors (Helps Catch Typos Early).
set -u

# ------------------------------------------------------------------------------
#                                 CONFIGURATION
# ------------------------------------------------------------------------------
# Generated Dataset & Record Locations.
DATA_OUTPUT_ROOT="data/corruption_dataset"
RECORDS_OUTPUT_DIR="configs/datasets/corruption_subsets"

# Per-Run Results & Logs.
RESULTS_BASE_DIR="outputs/results/corruption_dataset"
LOG_DIR="outputs/logs/corruption_dataset"

# Python Module Used to Build Each Dataset.
BUILD_MODULE="src.data.corruption_dataset.build_corrupted_test_set"
# ------------------------------------------------------------------------------


# --------------------------------------------------------------------------
#                              SWEEP ARRAYS
# --------------------------------------------------------------------------
# Run Once to Collect the Uncorrupted Baseline Metrics.
RUN_BASELINE=true

# Corrupt All Test Flights for Maximum Statistical Power During the Sweep.
SWEEP_CORRUPTED_FRACTION="1.0"

# Parameters Varied by the Sweep.
RAMP_SHAPES=("linear" "sigmoid")
RAMP_LENGTHS=("10" "20" "30" "50" "80" "160")

# Parameters Held Fixed Across All Runs.
ACTIVE_CORRUPTIONS="iso_noise defocus_blur colour_quantisation near_focus far_focus low_light"
PEAK_SEVERITY="1.0"
SEED="42"
# --------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           RUN NUMBERING HELPER
# ------------------------------------------------------------------------------
getNextRunDir() {
    # Make Base Results Directory. 
    local base_dir="$1"
    mkdir -p "$base_dir"

    # Track the Highest Existing Run Number.
    local max_num=0
    local dir_name run_num

    # Check Each Existing Run Directory.
    for dir_path in "$base_dir"/run*; do
        [ -d "$dir_path" ] || continue

        # Extract the Directory Name & Validate the runN Format.
        dir_name=$(basename "$dir_path")
        if [[ "$dir_name" =~ ^run([0-9]+)$ ]]; then

            # Extract the Numeric Run ID.
            run_num="${BASH_REMATCH[1]}"

            # Keep the Highest Run Number Found.
            if (( run_num > max_num )); then
                max_num=$run_num
            fi
        fi
    done

    # Return the Next Available Run ID.
    echo "run$((max_num + 1))"
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           SINGLE RUN EXECUTION
# ------------------------------------------------------------------------------
runOnce() {
    local extra_flags=("$@")

    # Remove Outputs from the Previous Run.
    echo "Removing Previous Generated Dataset: $DATA_OUTPUT_ROOT"
    rm -rf "$DATA_OUTPUT_ROOT"

    echo "Removing Previous Generated Records: $RECORDS_OUTPUT_DIR"
    rm -rf "$RECORDS_OUTPUT_DIR"

    # Create a New Run Directory & Log File.
    local run_name
    run_name=$(getNextRunDir "$RESULTS_BASE_DIR")

    local run_dir="$RESULTS_BASE_DIR/$run_name"
    mkdir -p "$run_dir"

    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/${run_name}.log"

    # Print the Run Details to the Terminal.
    echo "============================================================"
    echo " Starting $run_name at $(date)."
    echo " Flags: ${extra_flags[*]:-(none -- using the hardcoded defaults in build_corrupted_test_set.py)}"
    echo " Log:   $log_file"
    echo "============================================================"

    # Save the Parameters Before Starting so Failed Runs are Still Traceable.
    python3 -c "
import json, sys
flags = sys.argv[1:]
with open('$run_dir/run_params.json', 'w') as f:
    json.dump({'run_name': '$run_name', 'flags': flags}, f, indent=2)
" "${extra_flags[@]}"

    # Run the Builder & Redirect All Output to the Run Log.
    python3 -m "$BUILD_MODULE" \
        --output-data-root "$DATA_OUTPUT_ROOT" \
        --output-records-dir "$RECORDS_OUTPUT_DIR" \
        "${extra_flags[@]}" \
        > "$log_file" 2>&1
    local status=$?

    # Stop this Run if the Builder Failed.
    if [ $status -ne 0 ]; then
        echo " $run_name FAILED with Exit Code $status at $(date). Check $log_file for Details."
        cp "$log_file" "$run_dir/build.log" 2>/dev/null
        return 1
    fi

    # Copy Generated Records & the Build Log into the Run Directory.
    cp -r "$RECORDS_OUTPUT_DIR"/. "$run_dir"/ 2>/dev/null
    cp "$log_file" "$run_dir/build.log"

    # Report the Completed Run.
    echo " Finished $run_name at $(date). Results Saved to: $run_dir"
    echo

    return 0
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               MAIN EXECUTION
# ------------------------------------------------------------------------------
# Check Whether the Script was Started in Sweep Mode.
SWEEP_MODE=false
for arg in "$@"; do
    if [ "$arg" == "--sweep" ]; then
        SWEEP_MODE=true
    fi
done

# Run Once with the Builder's Default Configuration when No Sweep is Requested.
if [ "$SWEEP_MODE" = false ]; then
    echo "No --sweep Flag Supplied. Running Once with build_corrupted_test_set.py's Own Defaults."
    runOnce
    echo "============================================================"
    echo " Single Run Complete at $(date)."
    echo "============================================================"
    exit $?
fi

# Calculate the Number of Configurations in the Sweep Grid.
GRID_SIZE=$(( ${#RAMP_SHAPES[@]} * ${#RAMP_LENGTHS[@]} ))

# Add the Baseline Run if Enabled.
TOTAL_RUNS=$GRID_SIZE
if [ "$RUN_BASELINE" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
fi

# Print the Total Number of Runs Before Starting the Sweep.
echo "Sweep Mode. ${#RAMP_SHAPES[@]} Ramp-Shape(s) x ${#RAMP_LENGTHS[@]} Ramp-Length(s) = $GRID_SIZE Grid Run(s), Plus Baseline: $RUN_BASELINE. Total: $TOTAL_RUNS Run(s)."
echo

# Track How Many Sweep Runs Fail.
FAILED_RUNS=0

# Run the Uncorrupted Baseline Once Before the Sweep.
if [ "$RUN_BASELINE" = true ]; then
    runOnce \
        --corrupted-fraction "0.0" \
        --seed "$SEED" \
        --active-corruptions $ACTIVE_CORRUPTIONS

    # Record the Failure but Continue with the Sweep.
    if [ $? -ne 0 ]; then
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo " Continuing to the Sweep Grid Despite the Baseline Failure Above."
    fi
fi

# Iterate Over Every Combination in the Parameter Grid.
for shape in "${RAMP_SHAPES[@]}"; do
    for length in "${RAMP_LENGTHS[@]}"; do
        # Run the Current Parameter Combination.
        runOnce \
            --corrupted-fraction "$SWEEP_CORRUPTED_FRACTION" \
            --ramp-shape "$shape" \
            --ramp-length-frames "$length" \
            --peak-severity "$PEAK_SEVERITY" \
            --seed "$SEED" \
            --active-corruptions $ACTIVE_CORRUPTIONS
        
        # Record the Failure & Continue with the Next Combination.
        if [ $? -ne 0 ]; then
            FAILED_RUNS=$((FAILED_RUNS + 1))
            echo " Continuing to the Next Combination Despite the Failure Above."
        fi
    done
done

# Print the Final Sweep Summary.
echo "============================================================"
echo " Sweep Complete at $(date). Failed Runs: $FAILED_RUNS / $TOTAL_RUNS."
echo " Results Under: $RESULTS_BASE_DIR"
echo "============================================================"
# ------------------------------------------------------------------------------
