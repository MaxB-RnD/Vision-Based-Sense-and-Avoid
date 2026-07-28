# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: core_improvement_evaluator.py %
# Date Created:            %date_created: Tue Mar 24 15:18:38 2026 %
# CM Version               %version: 3 %
# CM Instance              %instance: 3 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Backbone evaluation script for the AICrowd Airborne Object Tracking pipeline.
# This script runs a full evaluation of multiple detection backbones using a
# shared inference and tracking framework. It is designed to provide a fair
# comparison between models under identical dataset and evaluation conditions.
#
# The pipeline executes dataset resolution and filtering, model initialisation
# for the selected backbone, inference over all flight sequences, metric
# computation using official evaluation scripts, and generation of a final
# benchmark summary including AFDR, FPPI, and EDR scores. The backbone used for 
# evaluation is selected via a command-line argument.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from this Folder:
#         nohup python -m src.core_improvement.core_improvement_evaluator select_model > outputs/logs/core_improvement/core_improvement_evaluation.log 2>&1 &
#
#   2. Optional Environment Variables:
#       - SELECTED_MODEL = (0-6)
#            0 = Winning Solution (Optimised)
#            1 = MobileNetV4
#            2 = EfficientViT-M2
#            3 = ResNet34
#            4 = EfficientViT-B2
#            5 = HRNet-W32
#            6 = HRNet-W32 (Track-to-Detect)
#
#   3. Then monitor from terminal with:
#            tail -f outputs/logs/core_improvement/core_improvement_evaluation.log
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

# Import Required Libraries.
import os
import sys

import json
import torch
import random
import subprocess
from glob import glob

# Because Life, Universe & Everything!
random.seed(42)

# Set Working Directory to Project Root.
PROJECT_ROOT = os.getcwd()

# Add Project Root to Path.
sys.path.insert(0, PROJECT_ROOT)

# Add Core Metric Folders to System Path.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))

# Import Local Project Funtions.
from logging_module import LogLevel, log
from notebooks.helpers.notebook_helpers import (
    get_next_run_dir, load_combined_gt, resolve_local_flights,
    convert_results_to_metrics_format, measure_predictor_complexity,
    build_extended_summary
)
 
# Confirm Project Root.
log(LogLevel.INFO, "core_improvement_evaluator:imports", "Project Root Set: %s", PROJECT_ROOT)


# ------------------------------------------------------------------------------
#                                   MAIN
# ------------------------------------------------------------------------------
def main() -> None:
    # Choose Which Model to Run
    # --------------------------------------------------
    # 0 -> Winning Solution Optimised (SegTracker Ensemble: HRNet32, HRNet48, GerNet-M + BiFPN, DLA60).
    # 1 -> MobileNetV4 (Lightweight Detection Backbone).
    # 2 -> EfficientVit-M2 (Medium-Weight Detection Backbone).
    # 3 -> ResNet34 (Medium-Weight Detection Backbone).
    # 4 -> EfficientVit-B2 (Heavy-Weight Detection Backbone).
    # 5 -> HRNet-W32 (Heavy-Weight Detection Backbone).
    # 6 -> HRNet-W32 Track-to-Detect (Heavy-Weight Detection Backbone With Coasted Tracking).
    model_list = ["winning_solution_star", "mobilenetv4", "efficientvitm2", "resnet34", "efficientvitb2", "hrnetw32", "hrnetw32_track_to_detect"]

    # Allow Model Selection from Command Line.
    if len(sys.argv) > 1:
        SELECTED_MODEL = int(sys.argv[1])
    else:
        SELECTED_MODEL = 0

    # Evaluation Filtering.
    MIN_SCORE = 0.60
    DETECTION_THRESHOLD = 0.60
    MIN_TRACK_LEN = 30

    # Set Path to Desired Evaluation Dataset.
    DATASET_PATH = "configs/datasets/training_subsets/test.txt"
    DATASET_PARTS = ["part1", "part2", "part3"]

    # Set Results Variables (Auto-Incrementing Run Directory).
    RESULTS_DIR = "outputs/results/core_improvement/" + model_list[SELECTED_MODEL]
    run_name = get_next_run_dir(RESULTS_DIR)

    # Evaluation Output Folders.
    EVAL_GT_DIR = os.path.join(RESULTS_DIR, run_name, "gt")
    EVAL_RESULT_DIR  = os.path.join(RESULTS_DIR, run_name, "result")
    EVAL_SUMMARY_DIR = os.path.join(RESULTS_DIR, run_name, "summaries")

    # Print Summary Statement.
    log(LogLevel.INFO, "core_improvement_evaluator:config", "Configuration Loaded.")
    print(f"  Model Selected : {model_list[SELECTED_MODEL]}")
    print(f"  Dataset Path   : {DATASET_PATH}")
    print(f"  Dataset Parts  : {DATASET_PARTS}")
    print(f"  Run Name       : {run_name}")
    print(f"  Results Path   : {RESULTS_DIR}")
    print(f"  Min Score      : {MIN_SCORE}")
    print(f"  Min Track Len  : {MIN_TRACK_LEN}")
    
    # Find all Available Sequences from the Provided Dataset.
    local_flights = resolve_local_flights(DATASET_PATH, PROJECT_ROOT)

    # Add Spacing in Console Output for Readability.
    print()

    # Log How Many Flights were Found & are Ready for Inference.
    log(LogLevel.INFO, "core_improvement_evaluator:preprocessing", "Flights Ready for Inference: %d", len(local_flights))

    # Preview the First 5 Flights to Confirm Dataset is Correct.
    for name, info in list(local_flights.items())[:5]:
        # Count No. Images in the Flight Directory.
        n_imgs = len(list(info["images_dir"].glob("*.png")))

        # Print Flight Part, Name, & No. Images.
        print(f"  [{info['part']}]  {name}  ({n_imgs} images)")

    # If there are More than 5 Flights, Log How Many Additional Ones Exist.
    if len(local_flights) > 5:
        log(LogLevel.DEBUG, "core_improvement_evaluator:preprocessing", " ... and %d more.", len(local_flights) - 5)

    # Load & Filter Ground Truth to Include Local Flights Only. 
    combined_gt = load_combined_gt(DATASET_PARTS, local_flights, PROJECT_ROOT)
    log(LogLevel.INFO, "core_improvement_evaluator:preprocessing", "Flights in Filtered Ground Truth: %d", len(combined_gt['samples']))

    # Build Set of Image Filenames That Physically Exist on Disk.
    all_images = {name for f in local_flights.values() for name in f["image_names"]}

    # Cull GT Entities to Only Those With a Corresponding Image on Disk.
    before = sum(len(flight["entities"]) for flight in combined_gt["samples"].values())

    for flight_hash, flight_data in combined_gt["samples"].items():
        flight_data["entities"] = [
            e for e in flight_data["entities"]
            if e["img_name"] in all_images
        ]

    after = sum(len(flight["entities"]) for flight in combined_gt["samples"].values())

    # Report How Many Images were Removed. 
    log(LogLevel.INFO, "core_improvement_evaluator:preprocessing",
        "GT Culled: %d -> %d entities (%d removed for missing images).", before, after, before - after)

    # Save the Filtered Ground Truth to the Evaluation Folder.
    os.makedirs(EVAL_GT_DIR, exist_ok=True)
    gt_out_path = os.path.join(EVAL_GT_DIR, "groundtruth.json")

    # Copy the Ground Truth Values to the .json.
    with open(gt_out_path, "w") as f:
        json.dump(combined_gt, f)

    # Print Success Message.
    log(LogLevel.INFO, "core_improvement_evaluator:preprocessing", "Ground Truth Saved: %s", gt_out_path)

    # Create a Directory where all Evaluation Flight Data will be Stored.
    EVAL_DATA_DIR = os.path.join(RESULTS_DIR, run_name, "data")
    os.makedirs(EVAL_DATA_DIR, exist_ok=True)

    # Link Each Local Flight's Image Folder into the Evaluation Directory.
    for folder_name, info in local_flights.items():
        link_path = os.path.join(EVAL_DATA_DIR, folder_name)
        if not os.path.exists(link_path):
            os.symlink(info["images_dir"], link_path)

    # Set Environment Variables that AirbornePredictor and SelectedModelPredictor Read.
    os.environ["TEST_DATASET_PATH"] = EVAL_DATA_DIR
    os.environ["INFERENCE_OUTPUT_PATH"] = RESULTS_DIR
    os.environ["DATASET_ENV"] = run_name
    os.environ["MIN_TRACK_LEN"] = str(MIN_TRACK_LEN)
    os.environ["MIN_SCORE"] = str(MIN_SCORE)
    os.environ["DETECTION_THRESHOLD"] = str(DETECTION_THRESHOLD)

    # Limit Inference to Only the Specified Flights.
    os.environ["PARTIAL_RUN_FLIGHTS"] = ",".join(local_flights.keys())

    # Print Summary Statement.
    log(LogLevel.INFO, "core_improvement_evaluator:inference", "Environment Configured.")
    print(f"  Test Data Path     : {EVAL_DATA_DIR}")
    print(f"  Flights to Process : {len(local_flights)}")
    print(f"  MIN_SCORE          : {MIN_SCORE}")
    print(f"  MIN_TRACK_LEN      : {MIN_TRACK_LEN}")

    # Import Baseline Classes.
    from src.core_improvement.resnet34.resnet34_test import ResNet34Predictor
    from src.core_improvement.hrnetw32.hrnet_w32_test import HRNetW32Predictor
    from src.core_improvement.mobilenetv4.mobilenet_test import MobileNetV4Predictor
    from src.core_improvement.winning_solution_star.seg_test import WinningSolutionPredictor
    from src.core_improvement.efficientvitm2.efficientvit_m2_test import EfficientViTM2Predictor
    from src.core_improvement.efficientvitb2.efficientvit_b2_test import EfficientViTB2Predictor
    from src.core_improvement.hrnetw32_track_to_detect.hrnetw32_track_to_detect_test import HRNetW32TrackToDetectPredictor

    # Check GPU is CUDA Enabled.
    if torch.cuda.is_available():
        log(LogLevel.INFO, "core_improvement_evaluator:device_check", "CUDA-Enabled GPU is Available.")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}.")
        print(f"  CUDA Version: {torch.version.cuda}.")
    else:
        log(LogLevel.WARNING, "core_improvement_evaluator:device_check", "No CUDA-Enabled GPU Found, Using CPU Instead.")

    # Map Model Selection to Predictor Classes.
    MODEL_MAP = {
        0: WinningSolutionPredictor,       # Optimised Baseline.
        1: MobileNetV4Predictor,           # MobileNet v4.
        2: EfficientViTM2Predictor,        # EfficientVit m2.
        3: ResNet34Predictor,              # ResNet34.
        4: EfficientViTB2Predictor,        # EfficientVit b2.
        5: HRNetW32Predictor,              # HRNet-W32.
        6: HRNetW32TrackToDetectPredictor, # HRNet-W32 Track-to-Detect.
    }

    # Validate Selection.
    if SELECTED_MODEL not in MODEL_MAP:
        log(LogLevel.ERROR, "core_improvement:inference", "Invalid SELECTED_MODEL: %s.", SELECTED_MODEL)
        raise ValueError("Please Select a Valid Model Index.")

    # Initialise Predictor.
    predictor = MODEL_MAP[SELECTED_MODEL]()

    # Load Models Before Measuring Complexity.
    predictor.inference_setup()

    # Measure the Current Model's Complexity.
    print()
    complexity = measure_predictor_complexity(predictor)

    # Run Inference Over all Flights.
    print()
    predictor.run()

    # Print Summary Statement.
    log(LogLevel.INFO, "core_improvement:inference", "Inference Complete.")
    print(f"  Raw Results Saved to: {os.path.join(RESULTS_DIR, run_name)}\n")

    # Set Results Directories. 
    RAW_RESULTS_PATH = os.path.join(RESULTS_DIR, run_name, "result.json")
    METRICS_RESULTS_PATH = os.path.join(EVAL_RESULT_DIR, "result.json")

    # Ensure Output Directory Exists.
    os.makedirs(EVAL_RESULT_DIR, exist_ok=True)

    # Copy the Results File.
    convert_results_to_metrics_format(RAW_RESULTS_PATH, METRICS_RESULTS_PATH)

    # Print Summary Statement.
    log(LogLevel.INFO, "core_improvement:inference", "Results Copied to Metrics Directory.")
    print(f"  Metrics-Ready Results Stored at : {METRICS_RESULTS_PATH}")

    # Locate the Competition Metrics Script.
    metrics_script = os.path.join(PROJECT_ROOT, "core/metrics/run_airborne_metrics.py")

    # Call the Competition Metrics Script.
    subprocess.run(
        [
            "python3", metrics_script,
            "--dataset-folder",   EVAL_GT_DIR,
            "--results-folder",   EVAL_RESULT_DIR,
            "--summaries-folder", EVAL_SUMMARY_DIR,
            "--min-score", str(MIN_SCORE),
            "--min-track-len", str(MIN_TRACK_LEN),
        ],
        check=True  # Raises CalledProcessError if the script exits with a non-zero code.
    )

    # Print Message on Completion. 
    log(LogLevel.INFO, "core_improvement_evaluator:evaluation", "Metrics Complete.")

    # Find the Summary JSON in the Summaries Directory.
    summary_files = glob(os.path.join(EVAL_SUMMARY_DIR, "*.json"))

    # Check that the Summary Files Exist.
    if not summary_files:
        log(LogLevel.ERROR, "core_improvement_evaluator:summary", "No Summary Files Found! Run Inference First.")
    else:
        # Build Summary DataFrame.
        summary_df = build_extended_summary(
            summary_json_path = summary_files[0],
            run_dir           = os.path.join(RESULTS_DIR, run_name),
            complexity        = complexity,
            output_csv_path   = os.path.join(EVAL_SUMMARY_DIR, "core_improvement_evaluator_benchmark_summary.csv"),
        )
    
        # Print Summary to Console.
        print("\n" + "=" * 60)
        print(" NEW BACKBONE MODEL - FULL HARD DATASET BENCHMARK SUMMARY")
        print("=" * 60)
        print(summary_df.set_index("Metric").to_string())
# ------------------------------------------------------------------------------


# Call the Main Function.
if __name__ == "__main__":
    main()
