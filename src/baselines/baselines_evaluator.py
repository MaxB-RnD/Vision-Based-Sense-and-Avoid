#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: inference_evaluator.sh %
# Date Created:            %date_created: Tue May 20 23:49:23 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Sequential Inference Evaluation Runner for all baseline and improved models.
# This script executes the full evaluation pipeline for each model one at a 
# time, using notebook_to_terminal.py as the core execution engine.
#
# USAGE:
#   1. Run: chmod +x src/evaluation/inference_evaluator.sh
#   2. nohup bash src/evaluation/inference_evaluator.sh > outputs/logs/master_evaluator_runner.log 2>&1 &
#   3. Then monitor from terminal with:
#            tail -f logs/master_evaluator_runner.log OR watch W&B directly.
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

# Exit Immediately if Any Command Fails.
set -e  

# Resolve Project Root Relative to This Script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Notebook Helper Directory.
NOTEBOOK_DIR="$PROJECT_ROOT/notebooks/helpers"

# Logs Directory.
LOG_DIR="$PROJECT_ROOT/outputs/logs/core_improvement"

# Create Logs Directory if Missing.
mkdir -p "$LOG_DIR"

# Activate Virtual Environment.
source "$PROJECT_ROOT/.venv/bin/activate"

# Model Names.
MODEL_NAMES=(
    "WinningSolution"
    "TransformerDetector"
    "DroneDetector"
    "MobileNetV4"
    "EfficientViTM2"
    "ResNet34"
    "EfficientViTB2"
    "HRNetW32"
)

# Move into the Directory Expected by notebook_to_terminal.py.
cd "$NOTEBOOK_DIR"

# Run Sequential Evaluations.
for i in "${!MODEL_NAMES[@]}"; do

    # Skip Models 1 & 2.
    if [[ "$i" -eq 1 || "$i" -eq 2 ]]; then
        echo "[runner] Skipping model index $i (${MODEL_NAMES[$i]})"
        continue
    fi

    # Collect Model Name & Make Log File. 
    MODEL_NAME="${MODEL_NAMES[$i]}"
    LOG_FILE="$LOG_DIR/${MODEL_NAME}_evaluation.log"

    echo "============================================================"
    echo "[runner] Starting: $MODEL_NAME"
    echo "[runner] Log File : $LOG_FILE"
    echo "============================================================"

    # Call the Inference Script for Current Model.
    if python3 notebook_to_terminal.py "$i" > "$LOG_FILE" 2>&1; then
        echo "[runner] SUCCESS: $MODEL_NAME"
    else
        echo "[runner] FAILED : $MODEL_NAME"
    fi
    echo
done

echo "============================================================"
echo "[runner] All Evaluations Complete."
echo "============================================================"
