#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: run_all_modules.sh %
# Date Created:            %date_created: Thurs Aug 7 00:31:29 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Runs introspection_module_evaluator.py Sequentially for all Introspection
# Modules, Waiting for Each to Fully Finish Before Starting the Next. This
# Avoids GPU Contention and File Clashes Between Runs When Testing Multiple
# Backbones Overnight.
#
# USAGE:
#   1. Make the Script Executable:
#         chmod +x src/introspection_module/run_all_modules.sh
#
#   2. Run the Script from the Project Root:
#         nohup bash src/introspection_module/run_all_modules.sh > outputs/logs/introspection_module/run_all_modules.log 2>&1 &
#
#   3. Then Monitor from Terminal with:
#            tail -f outputs/logs/introspection_module/run_all_modules.log
#            (or tail -f the Per-Module Log Listed Below for a Specific Module)
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
# Directory Where all Per-Module Evaluation Logs will be Written.
LOG_DIR="outputs/logs/introspection_module"

# Create Logging Directory if it Doesn't Exist.
mkdir -p "$LOG_DIR"

# Human-Readable Names for Each Module Index, Used for Logging (Must Match the 
# module_list Order in introspection_module_evaluator.py).
MODEL_NAMES=("none" "confidence_threshold" "rolling_confidence_trend" "coast_count_threshold")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              EXECUTION SETUP
# ------------------------------------------------------------------------------
# Run the Evaluation Loop for Every Module, One at a Time.
for i in 0 1 2 3; do
    # Resolve the Module's Name & Dedicated Log File for This Run.
    NAME="${MODEL_NAMES[$i]}"
    LOG_FILE="$LOG_DIR/${NAME}_evaluation.log"

    echo "============================================================"
    echo " Starting Module $i ($NAME) at $(date)."
    echo " Log: $LOG_FILE."
    echo "============================================================"

    # Run in the Foreground of this Script (No Trailing &), so the Loop
    # Blocks Here Until this Module's Run Fully Completes.
    python -m src.introspection_module.introspection_module_evaluator "$i" > "$LOG_FILE" 2>&1

    # Capture the Exit Code of the Evaluation Run.
    STATUS=$?

    # Safely Handle Failure Cases.
    if [ $STATUS -ne 0 ]; then
        echo " Module $i ($NAME) FAILED with Exit Code $STATUS at $(date)."
        echo " Check $LOG_FILE for Details."
        echo " Stopping Here! Remaining Modules Were Not Run."
        exit 1
    fi

    echo " Finished Module $i ($NAME) at $(date)."
    echo
done
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                   SUMMARY
# ------------------------------------------------------------------------------
# Print Final Confirmation Once all 7 Modules Have Completed Successfully.
echo "============================================================"
echo " All 4 Modules Completed Successfully at $(date)."
echo "============================================================"
# ------------------------------------------------------------------------------
