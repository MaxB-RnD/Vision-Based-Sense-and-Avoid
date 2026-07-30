#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: train_winning_solution.sh %
# Date Created:            %date_created: Tue May 05 19:34:12 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 2 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Trains all four models of the winning solution ensemble sequentially on a
# single GPU. Each model is trained independently and saves to its own
# auto-incrementing run directory under outputs/model_training/winning_solution_star/
#
# MODELS TRAINED (in order of increasing training time):
#   1. HRNet32
#   2. GerNet-M BiFPN
#   3. DLA60
#   4. HRNet48
#
# USAGE:
#   1. Run: chmod +x src/introspection_module/winning_solution_star/train_winning_solution.sh
#   2. nohup src/introspection_module/winning_solution_star/train_winning_solution.sh > outputs/logs/model_training/train_winning_solution_star.log 2>&1 &
#   3. Then monitor from terminal with:
#            tail -f outputs/logs/model_training/train_winning_solution_star.log OR watch W&B directly.
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

# ------------------------------------------------------------------------------
#                                 CONFIGURATION
# ------------------------------------------------------------------------------
TRAIN_FLIGHTS="configs/datasets/training_subsets/train.txt"
VAL_FLIGHTS="configs/datasets/training_subsets/val.txt"
OUTPUT_DIR="outputs/results/model_training/winning_solution_star"
WANDB_PROJECT="Vision Based Sense & Avoid"
SEED=42
LOG_DIR="outputs/logs/model_training"

# Experiment Config Names (Relative to configs/models/ Directory).
EXPERIMENTS=(
    "winning_solution_star/HRNetW32_Complete"
    "winning_solution_star/GerNetB2_Complete"
    "winning_solution_star/DLA60_Complete"
    "winning_solution_star/HRNetW48_Complete"
)

# Human-Readable Names for Logging.
NAMES=(
    "HRNet32"
    "GerNet-M BiFPN"
    "DLA60"
    "HRNet48"
)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                            DATA TRANSFORMS SETUP
# ------------------------------------------------------------------------------
# Create Logging Directory if it Doesn't Exist.
mkdir -p "$LOG_DIR"

# Create a Log File for Training Data Transforms.
LOG_FILE="$LOG_DIR/frame_transforms_train.log"

# Run Data Transforms Script on Training Data (Python Module Entry Point).
python -m src.introspection_module.winning_solution_star.seg_tracker.generate_frame_transforms \
    --flights-file "$TRAIN_FLIGHTS" \
    --skip-existing \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Safely Handle Failure Cases.
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Data Transforms Failed with Exit Code $EXIT_CODE."
    echo "Check $LOG_FILE for Details."
    exit $EXIT_CODE
fi

# Create a Log File for Validation Data Transforms.
LOG_FILE="$LOG_DIR/frame_transforms_valid.log"

# Run Data Transforms Script on Validation (Python Module Entry Point).
python -m src.introspection_module.winning_solution_star.seg_tracker.generate_frame_transforms \
    --flights-file "$VAL_FLIGHTS" \
    --skip-existing \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Safely Handle Failure Cases.
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Data Transforms Failed with Exit Code $EXIT_CODE."
    echo "Check $LOG_FILE for Details."
    exit $EXIT_CODE
fi
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              EXECUTION SETUP
# ------------------------------------------------------------------------------
# Record Models Running & Start Time.
TOTAL=${#EXPERIMENTS[@]}
START_TIME=$(date +%s)

echo "========================================================================"
echo "  Winning Solution Star — Full Ensemble Retraining"
echo "  Models    : $TOTAL"
echo "  Train set : $TRAIN_FLIGHTS"
echo "  Val set   : $VAL_FLIGHTS"
echo "  Output    : $OUTPUT_DIR"
echo "  Started   : $(date)"
echo "========================================================================"
echo ""

# Run the Training Loop for Every Model.
for i in "${!EXPERIMENTS[@]}"; do
    # Make Clean Log & Run History for Each Model.
    EXP="${EXPERIMENTS[$i]}"
    NAME="${NAMES[$i]}"
    MODEL_NUM=$((i + 1))
    LOG_FILE="$LOG_DIR/train_${NAME// /_}.log"

    echo "------------------------------------------------------------------------"
    echo "  [$MODEL_NUM/$TOTAL] Starting: $NAME"
    echo "  Config  : $EXP"
    echo "  Log     : $LOG_FILE"
    echo "  Time    : $(date)"
    echo "------------------------------------------------------------------------"

    # Run Training Script (Python Module Entry Point).
    python -m src.introspection_module.winning_solution_star.seg_tracker.train train "$EXP" \
        --train-flights "$TRAIN_FLIGHTS" \
        --val-flights   "$VAL_FLIGHTS" \
        --seed          "$SEED" \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --output-dir    "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    # Safely Handle Failure Cases.
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: $NAME Failed with Exit Code $EXIT_CODE."
        echo "Check $LOG_FILE for Details."
        echo "Remaining Models Will NOT Be Trained."
        exit $EXIT_CODE
    fi

    echo ""
    echo "  [$MODEL_NUM/$TOTAL] $NAME Complete. $(date)"
    echo ""
done

# Record End Time & Print Final Summary.
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 3600 ))

echo "========================================================================"
echo "  All $TOTAL Models Trained Successfully."
echo "  Total time : ~${ELAPSED} Hours"
echo "  Finished   : $(date)"
echo "  Results    : $OUTPUT_DIR"
echo "========================================================================"
# ------------------------------------------------------------------------------
