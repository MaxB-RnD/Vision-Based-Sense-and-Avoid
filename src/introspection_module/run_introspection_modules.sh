#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: run_introspection_modules.sh %
# Date Created:            %date_created: Tue Sep 15 20:20:14 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Designed to Seamlessly Run Introspection Across All Introspection Modules from    
# the Terminal Without Manual Intervention. Supports both running inference on a 
# given set of test flights using fixed module configurations and sweeping 
# hyperparameter configurations across Modules 1-4 to identify the 
# best-performing settings, with Module 0 (Control) retained as the baseline. 
# The script is designed to manage the evaluation process end-to-end, 
# automatically progressing through the required configurations and runs so that
# long-running evaluations can be left unattended rather than manually monitored
# or restarted.
#
# USAGE:
#   1. Make the Script Executable (Once Only):
#         chmod +x src/introspection_module/run_introspection_modules.sh
#
#   2. Preview the Run Plan & Estimated Wall Time Without Touching a GPU. Always
#      Do This First as a Full Sweep Spans Many Hours:
#         ./src/introspection_module/run_introspection_modules.sh --sweep --dry-run
#
#   3. Run the Validation Parameter Sweep:
#         ./src/introspection_module/run_introspection_modules.sh --sweep
#
#   4. Then Monitor from Terminal with:
#         tail -f outputs/logs/introspection_module/parameter_sweep/parameter_sweep.log
#         (or tail -f the Per-Config Log Listed at Each Step Above)
#
#   5. Choose a Shared Target FAR and Write selected_params.json. Follow the
#      Instructions in select_introspection_configs.py. Inspect the Achievable
#      Operating Curves First, then Select Against a Target That Sits Inside the
#      Band Every Module Can Actually Reach.
#
#   6. Run the Selected Configurations Once per Module on the Test Split:
#         ./src/introspection_module/run_introspection_modules.sh --all
#
#   7. Then Monitor from Terminal with:
#         tail -f outputs/logs/introspection_module/selected_params/selected_params.log
#         (or tail -f the Per-Module Log Listed at Each Step Above)
#
#
# MODES (Mutually Exclusive, Exactly One Required):
#   --sweep              Validation Hyperparameter Sweep (Modules 1-4).
#   --all                Selected-Configuration Test Pass (Modules 0-4).
#
#
# OPTIONS:
#   --module N           Restrict the Pass to a Single Module Index (0-4).
#                        Useful for Staging a Long Sweep Across Sessions.
#   --split NAME         Override the Dataset Split. Defaults to 'validation'
#                        for --sweep and 'test' for --all. Must Match a File
#                        at configs/datasets/training_subsets/<NAME>.txt.
#   --dry-run            Print the Run Plan & Exit Without Executing Anything.
#   --minutes-per-run M  Per-Run Estimate Used for Dry-Run Timing (Default 120).
#   --foreground         Do Not Detach Under nohup. Run Attached to the Terminal.
#   --help               Show this Usage Block.
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

# Preserve the Caller's Arguments Verbatim so the Background Re-Exec Below can Replay Them Exactly.
ORIGINAL_ARGS=("$@")


# ------------------------------------------------------------------------------
#                              LOGGING HELPERS
# ------------------------------------------------------------------------------
logMessage() {
    local level="$1"
    local tag="$2"
    shift 2
    echo "[$level] run_introspection_modules:$tag -- $*"
}

logWarning() {
    logMessage "WARNING" "$@"
}

logError() {
    logMessage "ERROR" "$@"
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              ARGUMENT PARSING
# ------------------------------------------------------------------------------
# Initialise Command-Line Parameters & their Default Values.
MODE=""
SPLIT=""
MODULE_FILTER=""
DRY_RUN=0
FOREGROUND=0
MINUTES_PER_RUN=120

# Display the Script Usage Instructions.
usage() {
    sed -n '/^# USAGE:/,/^# ---/p' "$0" | sed 's/^# \{0,1\}//'
}

# Parse & Apply All Command-Line Arguments.
while [ $# -gt 0 ]; do
    case "$1" in
        --sweep)
            if [ -n "$MODE" ]; then logError "args" "Specify Only One of --sweep / --all."; exit 2; fi
            MODE="sweep"; shift ;;
        --all)
            if [ -n "$MODE" ]; then logError "args" "Specify Only One of --sweep / --all."; exit 2; fi
            MODE="all"; shift ;;
        --module)
            MODULE_FILTER="${2:-}"; shift 2 ;;
        --split)
            SPLIT="${2:-}"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --foreground)
            FOREGROUND=1; shift ;;
        --minutes-per-run)
            MINUTES_PER_RUN="${2:-}"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            logError "args" "Unknown Argument: $1"; echo; usage; exit 2 ;;
    esac
done

# Exactly One Mode is Required -- Refuse to Guess.
if [ -z "$MODE" ]; then
    logError "args" "No Mode Selected. Pass Exactly One of --sweep or --all."
    echo
    usage
    exit 2
fi

# Validate the Module Filter if One was Supplied.
if [ -n "$MODULE_FILTER" ]; then
    case "$MODULE_FILTER" in
        0|1|2|3|4) ;;
        *) logError "args" "--module Must be an Integer 0-4 (Got '$MODULE_FILTER')."; exit 2 ;;
    esac
fi

# Default the Split per Mode Unless Explicitly Overridden.
if [ -z "$SPLIT" ]; then
    if [ "$MODE" = "sweep" ]; then SPLIT="validation"; else SPLIT="test"; fi
fi
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                 CONFIGURATION
# ------------------------------------------------------------------------------
# Must Match the module_list Order in introspection_module_evaluator.py.
MODULE_NAMES=("none" "confidence_threshold" "rolling_confidence_trend" "coast_count_threshold" "kalman_filter_nis")

# Consolidated Parameter-Sweep Results, Read by select_introspection_configs.py.
PARAMETER_SWEEP_CSV="outputs/results/introspection_module/parameter_sweep.csv"

# Selected Parameters, Read by --all.
SELECTED_PARAMS_JSON="outputs/results/introspection_module/selected_params.json"

# Per-Config / Per-Module Logs, with Separate Directories for Each Mode.
if [ "$MODE" = "sweep" ]; then
    LOG_DIR="outputs/logs/introspection_module/parameter_sweep"
    MASTER_LOG="$LOG_DIR/parameter_sweep.log"
else
    LOG_DIR="outputs/logs/introspection_module/selected_params"
    MASTER_LOG="$LOG_DIR/selected_params.log"
fi
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                        BACKGROUND (NOHUP) RE-EXEC
# ------------------------------------------------------------------------------
# A Full Pass is Many Hours Long, so Detaching is the Default Rather than 
# Something the Caller has to Remember.
if [ "$DRY_RUN" -eq 0 ] && [ "$FOREGROUND" -eq 0 ] && [ "${INTROSPECTION_RUNNER_DETACHED:-0}" != "1" ]; then
    # Verify Selected Parameters Exist Before Starting a Test Pass.
    if [ "$MODE" = "all" ] && [ ! -f "$SELECTED_PARAMS_JSON" ]; then
        logError "preflight" "Selected Parameters Not Found at $SELECTED_PARAMS_JSON."
        logError "preflight" "Run --sweep then select_introspection_configs.py First."
        exit 1
    fi

    # Ensure the Master Log Directory Exists.
    mkdir -p "$LOG_DIR"

    # Mark the Child Process as Detached to Prevent Recursive Re-Exec.
    export INTROSPECTION_RUNNER_DETACHED=1

    # Re-Execute the Pass Under nohup with --foreground & Redirect Output.
    nohup bash "$0" "${ORIGINAL_ARGS[@]}" --foreground > "$MASTER_LOG" 2>&1 &

    # Report the Background Run & Master Log Location to the Caller.
    echo ""
    echo "Running --$MODE in the Background. Log: $MASTER_LOG"
    echo ""
    exit 0
fi
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                SWEEP GRIDS
# ------------------------------------------------------------------------------
# Confidence Threshold Values and Consecutive Low-Confidence Frame Window Values.
CONF_THRESH_VALUES=("0.65" "0.75" "0.85" "0.95")
CONF_WINDOW_VALUES=("1" "2" "3")

# EWMA Threshold Values and Slope Threshold Values Used by the Two Detection Signals.
ROLLING_EWMA_THRESH_VALUES=("0.65" "0.75" "0.85" "0.95")
ROLLING_SLOPE_THRESH_VALUES=("-0.02" "-0.05" "-0.10" "-0.20")

# Sentinel Values Used to Disable One Signal While Sweeping the Other.
ROLLING_EWMA_DISABLED="0.0"
ROLLING_SLOPE_DISABLED="-1000000000.0"

# Coast Count Threshold Values Defining the Number of Consecutive Coasted Frames Required.
COAST_THRESHOLD_VALUES=("1" "2" "3")

# Chi-Squared Threshold Values Used to Determine When the NIS Indicates an Inconsistent Track.
KALMAN_CHI2_VALUES=("1.386" "2.408" "3.219" "4.605" "5.991" "7.378" "9.210")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           CONFIG RUNNER HELPER
# ------------------------------------------------------------------------------
# Track the Total Number of Configurations Run & the Number That Failed.
CONFIG_COUNT=0
FAILED_CONFIGS=0

# Run a Single Evaluator Configuration & Handle Logging, Parameters & Failures.
runIntrospectionConfig() {
    # Hyperparameter Overrides Passed to the Evaluator as INTROSPECTION_PARAM_* Pairs.
    local module_index="$1"; shift
    local module_name="$1"; shift
    local log_tag="$1"; shift

    # Capture Any Hyperparameter Overrides Passed to the Evaluator.
    local param_assignments=("$@")

    # Increment the Configuration Counter & Build Its Log File Path.
    CONFIG_COUNT=$((CONFIG_COUNT + 1))
    local log_file="$LOG_DIR/${module_name}_${log_tag}.log"

    # Format the Configuration Parameters for Display in the Run Plan.
    local param_display="${param_assignments[*]:-}"
    if [ -z "${param_display// /}" ]; then
        param_display="(module defaults)"
    fi

    # Display the Current Configuration in the Run Plan.
    echo "  [$CONFIG_COUNT] $module_name :: $param_display"

    # Skip Execution During a Dry Run.
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi

    # Display the Log File Used by the Current Configuration.
    echo "        Log: $log_file"

    # Export the Configuration Parameters for the Evaluator Process.
    local assignment
    for assignment in "${param_assignments[@]:-}"; do
        [ -n "$assignment" ] && export "$assignment"
    done

    # Set the Dataset Split for the Evaluator.
    export INTROSPECTION_DATASET_SPLIT="$SPLIT"

    # Run the Evaluator for the Current Configuration.
    python -m src.introspection_module.introspection_module_evaluator "$module_index" > "$log_file" 2>&1
    local status=$?

    # Record Failed Configurations Without Aborting the Overall Pass.
    if [ $status -ne 0 ]; then
        logError "runIntrospectionConfig" \
            "Config $CONFIG_COUNT ($module_name :: $param_display) Failed with Exit $status. See $log_file."
        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
    fi

    # Remove Configuration Parameters Before Running the Next Configuration.
    for assignment in "${param_assignments[@]:-}"; do
        [ -n "$assignment" ] && unset "${assignment%%=*}"
    done

    # Return the Evaluator's Exit Status to the Calling Function.
    return $status
}

# True if the Given Module Index Should Run Under the Current --module Filter.
moduleSelected() {
    [ -z "$MODULE_FILTER" ] || [ "$MODULE_FILTER" = "$1" ]
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               PARAMETER SWEEP
# ------------------------------------------------------------------------------
runSweep() {
    # Prepare the Sweep Results CSV and Export Its Path to the Evaluator.
    if [ "$DRY_RUN" -eq 0 ]; then
        mkdir -p "$(dirname "$PARAMETER_SWEEP_CSV")"
 
        # Remove Previous Sweep Results Before Starting a New Sweep.
        rm -f "$PARAMETER_SWEEP_CSV"
        export INTROSPECTION_SWEEP_RESULTS_CSV="$PARAMETER_SWEEP_CSV"
    fi

    # Sweep Confidence Threshold and Consecutive Low-Confidence Frame Window Values.
    if moduleSelected 1; then
        echo "--- Module 1: Confidence Threshold ($(( ${#CONF_THRESH_VALUES[@]} * ${#CONF_WINDOW_VALUES[@]} )) configs) ---"
        for threshold in "${CONF_THRESH_VALUES[@]}"; do
            for window in "${CONF_WINDOW_VALUES[@]}"; do
                runIntrospectionConfig 1 "confidence_threshold" "t${threshold}_w${window}" \
                    "INTROSPECTION_PARAM_THRESHOLD=$threshold" \
                    "INTROSPECTION_PARAM_WINDOW=$window"
            done
        done
        echo
    fi

    # Sweep EWMA Threshold Values with the Slope Signal Disabled.
    if moduleSelected 2; then
        echo "--- Module 2a: Rolling Trend, EWMA Arm (${#ROLLING_EWMA_THRESH_VALUES[@]} configs, Slope Disabled) ---"
        for ewma_threshold in "${ROLLING_EWMA_THRESH_VALUES[@]}"; do
            runIntrospectionConfig 2 "rolling_confidence_trend" "ewma${ewma_threshold}" \
                "INTROSPECTION_PARAM_EWMA_THRESHOLD=$ewma_threshold" \
                "INTROSPECTION_PARAM_SLOPE_THRESHOLD=$ROLLING_SLOPE_DISABLED"
        done
        echo

        # Sweep Slope Threshold Values with the EWMA Signal Disabled.
        echo "--- Module 2b: Rolling Trend, Slope Arm (${#ROLLING_SLOPE_THRESH_VALUES[@]} configs, EWMA Disabled) ---"
        for slope_threshold in "${ROLLING_SLOPE_THRESH_VALUES[@]}"; do
            runIntrospectionConfig 2 "rolling_confidence_trend" "slope${slope_threshold}" \
                "INTROSPECTION_PARAM_EWMA_THRESHOLD=$ROLLING_EWMA_DISABLED" \
                "INTROSPECTION_PARAM_SLOPE_THRESHOLD=$slope_threshold"
        done
        echo
    fi

    # Sweep the Consecutive Coast Count Required Before Escalation.
    if moduleSelected 3; then
        echo "--- Module 3: Coast-Count Threshold (${#COAST_THRESHOLD_VALUES[@]} configs) ---"
        for threshold in "${COAST_THRESHOLD_VALUES[@]}"; do
            runIntrospectionConfig 3 "coast_count_threshold" "t${threshold}" \
                "INTROSPECTION_PARAM_THRESHOLD=$threshold"
        done
        echo
    fi

    # Sweep NIS Chi-Squared Threshold Values.
    if moduleSelected 4; then
        echo "--- Module 4: Kalman Filter NIS (${#KALMAN_CHI2_VALUES[@]} configs) ---"
        for chi2_threshold in "${KALMAN_CHI2_VALUES[@]}"; do
            runIntrospectionConfig 4 "kalman_filter_nis" "chi2${chi2_threshold}" \
                "INTROSPECTION_PARAM_CHI2_THRESHOLD=$chi2_threshold"
        done
        echo
    fi

    # Report When Module 0 is Explicitly Requested Since It Has Nothing to Sweep.
    if moduleSelected 0 && [ -n "$MODULE_FILTER" ]; then
        logWarning "runSweep" "Module 0 (none) Has No Hyperparameters and is Not Swept. Nothing to Do."
        echo
    fi
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   SELECTED-CONFIGURATION INFERENCE PASS
# ------------------------------------------------------------------------------
runSelected() {
    # Prevent the Test Pass from Writing to the Parameter-Sweep CSV.
    unset INTROSPECTION_SWEEP_RESULTS_CSV 2>/dev/null || true

    # Require Selected Parameters Before Starting the Test Pass.
    if [ ! -f "$SELECTED_PARAMS_JSON" ]; then
        logError "runSelected" "Selected Parameters Not Found at $SELECTED_PARAMS_JSON."
        logError "runSelected" "Run --sweep then select_introspection_configs.py First."
        exit 1
    fi

    # Track the Current Module & Its Selected Parameter Assignments.
    local i name assignment
    local -a selected_assignments

    # Process Modules 0-4 Using Their Selected Configurations.
    for i in 0 1 2 3 4; do
        # Extract Module Name.
        name="${MODULE_NAMES[$i]}"

        # Skip Modules Excluded by the Optional --module Filter.
        moduleSelected "$i" || continue

        # Reset Parameter Assignments Before Loading the Current Module.
        selected_assignments=()

        # Module 0 Uses Its Default Configuration Because It Has No Parameters.
        if [ "$name" != "none" ]; then
            # Load the Module's Selected Parameters as INTROSPECTION_PARAM_* Pairs.
            while IFS= read -r assignment; do
                [ -n "$assignment" ] && selected_assignments+=("$assignment")
            done < <(python3 -c "
import json
with open('$SELECTED_PARAMS_JSON') as f:
    selected = json.load(f)
params = selected['modules'].get('$name', {}).get('params', {})
for k, v in params.items():
    print(f'INTROSPECTION_PARAM_{k.upper()}={v}')
")

            # Skip Modules Without a Selected Configuration Rather Than Using Defaults.
            if [ ${#selected_assignments[@]} -eq 0 ]; then
                echo "--- Module $i ($name) ---"
                logWarning "runSelected" \
                    "Module '$name' Has No Entry in $SELECTED_PARAMS_JSON. Skipping -- Running on Class Defaults Would Misreport an Untuned Configuration."
                echo
                FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                continue
            fi
        fi

        # Run the Current Module Once Using Its Selected Configuration.
        echo "--- Module $i ($name) on ${SPLIT^^} Split ---"
        runIntrospectionConfig "$i" "$name" "${SPLIT}" "${selected_assignments[@]:-}"
        echo
    done
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               MAIN EXECUTION
# ------------------------------------------------------------------------------
# Prepare the Log Directory Before Starting the Selected Pass.
if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$LOG_DIR"
fi

# Display the Current Run Mode, Dataset, Module Filter, Outputs, & Logs.
echo "============================================================"
if [ "$DRY_RUN" -eq 1 ]; then
    echo " DRY RUN -- No Inference Will Be Executed."
else
    echo " Starting Introspection Pass at $(date)."
fi
echo " Mode          : --$MODE"
echo " Dataset Split : $SPLIT"
if [ -n "$MODULE_FILTER" ]; then
    echo " Module Filter : $MODULE_FILTER (${MODULE_NAMES[$MODULE_FILTER]})"
fi
if [ "$MODE" = "sweep" ]; then
    echo " Results CSV   : $PARAMETER_SWEEP_CSV"
else
    echo " Selected Prms : $SELECTED_PARAMS_JSON"
fi
echo " Logs          : $LOG_DIR"
echo "============================================================"
echo

if [ "$MODE" = "sweep" ]; then
    runSweep
else
    runSelected
fi
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                             FINAL OUTPUT SUMMARY
# ------------------------------------------------------------------------------
echo "============================================================"
if [ "$DRY_RUN" -eq 1 ]; then
    TOTAL_MINUTES=$((CONFIG_COUNT * MINUTES_PER_RUN))
    echo " DRY RUN Complete. Planned Runs: $CONFIG_COUNT."
    echo " Estimated Wall Time at ${MINUTES_PER_RUN} min/run:"
    echo "   $TOTAL_MINUTES Minutes  ~=  $((TOTAL_MINUTES / 60)) Hours  ~=  $((TOTAL_MINUTES / 1440)) Day(s)."
    echo " Re-Run Without --dry-run to Execute."
else
    echo " Pass Complete at $(date). Failed: $FAILED_CONFIGS / $CONFIG_COUNT."
    if [ "$MODE" = "sweep" ]; then
        echo " Results Written to: $PARAMETER_SWEEP_CSV"
    else
        echo " Results Under: outputs/results/introspection_module/<module_name>/$SPLIT/"
    fi
fi
echo "============================================================"
# ------------------------------------------------------------------------------
