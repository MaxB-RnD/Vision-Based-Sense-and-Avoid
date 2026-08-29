#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: sweep_corruption_dataset.sh %
# Date Created:            %date_created: Sat Aug 29 23:12:45 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
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


# --------------------------------------------------------------------------
#                              SWEEP ARRAYS
# --------------------------------------------------------------------------
# Parameters Varied by the Sweep.
CORRUPTED_FRACTIONS=("0.0" "0.5" "1.0")
RAMP_SHAPES=("linear")
RAMP_LENGTHS=("15")

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
    echo "Removing previous generated dataset: $DATA_OUTPUT_ROOT"
    rm -rf "$DATA_OUTPUT_ROOT"

    echo "Removing previous generated records: $RECORDS_OUTPUT_DIR"
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
        echo " $run_name FAILED with exit code $status at $(date). Check $log_file for details."
        cp "$log_file" "$run_dir/build.log" 2>/dev/null
        return 1
    fi

    # Copy Generated Records & the Build Log into the Run Directory.
    cp -r "$RECORDS_OUTPUT_DIR"/. "$run_dir"/ 2>/dev/null
    cp "$log_file" "$run_dir/build.log"

    # Report the Completed Run.
    echo " Finished $run_name at $(date). Results saved to: $run_dir"
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
    echo "No --sweep flag supplied. Running once with build_corrupted_test_set.py's own defaults."
    runOnce
    echo "============================================================"
    echo " Single run complete at $(date)."
    echo "============================================================"
    exit $?
fi

# Report the Total Number of Parameter Combinations.
echo "Sweep mode. Grid size: ${#CORRUPTED_FRACTIONS[@]} corrupted-fraction(s) x " \
     "${#RAMP_SHAPES[@]} ramp-shape(s) x ${#RAMP_LENGTHS[@]} ramp-length(s) = " \
     "$(( ${#CORRUPTED_FRACTIONS[@]} * ${#RAMP_SHAPES[@]} * ${#RAMP_LENGTHS[@]} )) run(s)."
echo

# Track How Many Sweep Runs Fail.
FAILED_RUNS=0

# Iterate Over Every Combination in the Parameter Grid.
for fraction in "${CORRUPTED_FRACTIONS[@]}"; do
    for shape in "${RAMP_SHAPES[@]}"; do
        for length in "${RAMP_LENGTHS[@]}"; do
            # Run the Current Parameter Combination.
            runOnce \
                --corrupted-fraction "$fraction" \
                --ramp-shape "$shape" \
                --ramp-length-frames "$length" \
                --peak-severity "$PEAK_SEVERITY" \
                --seed "$SEED" \
                --active-corruptions $ACTIVE_CORRUPTIONS
            
            # Record the Failure & Continue with the Next Combination.
            if [ $? -ne 0 ]; then
                FAILED_RUNS=$((FAILED_RUNS + 1))
                echo " Continuing to the next combination despite the failure above."
            fi
        done
    done
done

# Print the Final Sweep Summary.
echo "============================================================"
echo " Sweep complete at $(date). Failed runs: $FAILED_RUNS."
echo " Results under: $RESULTS_BASE_DIR"
echo "============================================================"
# ------------------------------------------------------------------------------
