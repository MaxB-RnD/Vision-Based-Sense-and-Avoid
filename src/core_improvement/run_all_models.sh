#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: run_all_models.sh %
# Date Created:            %date_created: Wed Jul 22 17:08:22 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Runs core_improvement_evaluator.py Sequentially for all 7 Backbone Models
# (0-6), Waiting for Each to Fully Finish Before Starting the Next. This
# Avoids GPU Contention and File Clashes Between Runs When Testing Multiple
# Backbones Overnight.
#
# USAGE:
#   1. Make the Script Executable:
#         chmod +x src/core_improvement/run_all_models.sh
#
#   2. Run the Script from the Project Root:
#         nohup bash src/core_improvement/run_all_models.sh > outputs/logs/core_improvement/run_all_models.log 2>&1 &
#
#   3. Then Monitor from Terminal with:
#            tail -f outputs/logs/core_improvement/run_all_models.log
#            (or tail -f the Per-Model Log Listed Below for a Specific Model)
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
# Directory Where all Per-Model Evaluation Logs will be Written.
LOG_DIR="outputs/logs/core_improvement"

# Create Logging Directory if it Doesn't Exist.
mkdir -p "$LOG_DIR"

# Human-Readable Names for Each Model Index, Used for Logging (Must Match
# the model_list Order in core_improvement_evaluator.py).
MODEL_NAMES=("winning_solution_star" "mobilenetv4" "efficientvitm2" "resnet34" "efficientvitb2" "hrnetw32" "hrnetw32_track_to_detect")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              EXECUTION SETUP
# ------------------------------------------------------------------------------
# Run the Evaluation Loop for Every Model, One at a Time.
for i in 0 1 2 3 4 5 6; do
    # Resolve the Model's Name & Dedicated Log File for This Run.
    NAME="${MODEL_NAMES[$i]}"
    LOG_FILE="$LOG_DIR/${NAME}_evaluation.log"

    echo "============================================================"
    echo " Starting Model $i ($NAME) at $(date)."
    echo " Log: $LOG_FILE."
    echo "============================================================"

    # Run in the Foreground of this Script (No Trailing &), so the Loop
    # Blocks Here Until this Model's Run Fully Completes.
    python -m src.core_improvement.core_improvement_evaluator "$i" > "$LOG_FILE" 2>&1

    # Capture the Exit Code of the Evaluation Run.
    STATUS=$?

    # Safely Handle Failure Cases.
    if [ $STATUS -ne 0 ]; then
        echo " Model $i ($NAME) FAILED with Exit Code $STATUS at $(date)."
        echo " Check $LOG_FILE for Details."
        echo " Stopping Here! Remaining Models Were Not Run."
        exit 1
    fi

    echo " Finished Model $i ($NAME) at $(date)."
    echo
done
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                   SUMMARY
# ------------------------------------------------------------------------------
# Print Final Confirmation Once all 7 Models Have Completed Successfully.
echo "============================================================"
echo " All 7 Models Completed Successfully at $(date)."
echo "============================================================"
# ------------------------------------------------------------------------------
