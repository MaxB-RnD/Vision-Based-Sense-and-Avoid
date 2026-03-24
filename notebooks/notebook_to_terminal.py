# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: notebook_to_terminal.py %
# Date Created:            %date_created: Tue Mar 24 15:18:38 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Executes all code cells in a Jupyter Notebook (.ipynb) sequentially from the
# terminal, sharing a single execution context across all cells.
#
# Each cell's source is printed before execution so progress can be followed.
# If any cell raises an exception the script halts immediately and reports
# the failing cell number along with the full traceback.
#
# Non-code cells (markdown, raw, etc.) are silently skipped.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from Project Root:
#       - python notebooks/notebook_to_terminal.py
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
import random
import subprocess

from glob import glob

import pandas as pd

# Because Life, Universe & Everything!
random.seed(42)

# Set Working Directory to Project Root.
os.chdir("..") 
PROJECT_ROOT = os.getcwd()

# Add Project Root to Path.
sys.path.insert(0, PROJECT_ROOT)

# Add Core Metric Folders to System Path.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))

# Import Local Project Funtions.
from logging_module import LogLevel, log
from src.baseline.seg_test import SegPredictor
from src.data.download_data import main as downloadData
from notebooks.notebook_helpers import get_next_run_dir, load_combined_gt, resolve_local_flights

# Confirm Project Root.
log(LogLevel.INFO, "baseline:imports", "Project Root Set: %s", PROJECT_ROOT)

# ------------------------------------------------------------------------------
#                                   MAIN
# ------------------------------------------------------------------------------
def main() -> None:
  # Set Path to Desired Evaluation Dataset.
  DATASET_PATH = "configs/datasets/hard_dataset_flights.txt"
  DATASET_PARTS = ["part1", "part2", "part3"]

  # Set Results Variables (Auto-Incrementing Run Directory).
  RESULTS_DIR = "outputs/results/baseline/"
  run_name = get_next_run_dir(RESULTS_DIR)

  # Evaluation Filtering.
  MIN_SCORE = 0.0
  MIN_TRACK_LEN = 0

  # Evaluation Output Folders.
  EVAL_GT_DIR = os.path.join(RESULTS_DIR, run_name, "gt")
  EVAL_RESULT_DIR  = os.path.join(RESULTS_DIR, run_name, "result")
  EVAL_SUMMARY_DIR = os.path.join(RESULTS_DIR, run_name, "summaries")

  # Print Summary Statement.
  log(LogLevel.INFO, "baseline:config", "Configuration Loaded.")
  print(f"  Dataset Path  : {DATASET_PATH}")
  print(f"  Dataset Parts : {DATASET_PARTS}")
  print(f"  Run Name      : {run_name}")
  print(f"  Results Path  : {RESULTS_DIR}")
  print(f"  Min Score     : {MIN_SCORE}")
  print(f"  Min Track Len : {MIN_TRACK_LEN}")
    
  # Find all Available Sequences from the Provided Dataset.
  local_flights = resolve_local_flights(DATASET_PATH, PROJECT_ROOT)

  # Add Spacing in Console Output for Readability.
  print()

  # Log How Many Flights were Found & are Ready for Inference.
  log(LogLevel.INFO, "baseline:preprocessing", "Flights Ready for Inference: %d", len(local_flights))

  # Preview the First 5 Flights to Confirm Dataset is Correct.
  for name, info in list(local_flights.items())[:5]:
      # Count No. Images in the Flight Directory.
      n_imgs = len(list(info["images_dir"].glob("*.png")))

      # Print Flight Part, Name, & No. Images.
      print(f"  [{info['part']}]  {name}  ({n_imgs} images)")

  # If there are More than 5 Flights, Log How Many Additional Ones Exist.
  if len(local_flights) > 5:
      log(LogLevel.DEBUG, "baseline:preprocessing", " ... and %d more.", len(local_flights) - 5)

  # Load & Filter Ground Truth to Include Local Flights Only. 
  combined_gt = load_combined_gt(DATASET_PARTS, local_flights, PROJECT_ROOT)
  log(LogLevel.INFO, "baseline:preprocessing", "Flights in Filtered Ground Truth: %d", len(combined_gt['samples']))

  # Save the Filtered Ground Truth to the Evaluation Folder.
  os.makedirs(EVAL_GT_DIR, exist_ok=True)
  gt_out_path = os.path.join(EVAL_GT_DIR, "groundtruth.json")

  # Copy the Ground Truth Values to the .json.
  with open(gt_out_path, "w") as f:
      json.dump(combined_gt, f)

  # Print Success Message.
  log(LogLevel.INFO, "baseline:preprocessing", "Ground Truth Saved: %s", gt_out_path)

  # Create a Directory where all Evaluation Flight Data will be Stored.
  EVAL_DATA_DIR = os.path.join(RESULTS_DIR, run_name, "data")
  os.makedirs(EVAL_DATA_DIR, exist_ok=True)

  # Link Each Local Flight's Image Folder into the Evaluation Directory.
  for folder_name, info in local_flights.items():
      link_path = os.path.join(EVAL_DATA_DIR, folder_name)
      if not os.path.exists(link_path):
          os.symlink(info["images_dir"], link_path)

  # Set Environment Variables that AirbornePredictor and SegPredictor Read.
  os.environ["TEST_DATASET_PATH"] = EVAL_DATA_DIR
  os.environ["INFERENCE_OUTPUT_PATH"] = RESULTS_DIR
  os.environ["DATASET_ENV"] = run_name
  os.environ["MIN_TRACK_LEN"] = str(MIN_TRACK_LEN)
  os.environ["MIN_SCORE"] = str(MIN_SCORE)

  # Limit Inference to Only the Specified Flights.
  os.environ["PARTIAL_RUN_FLIGHTS"] = ",".join(local_flights.keys())

  # Print Summary Statement.
  log(LogLevel.INFO, "baseline:inference", "Environment Configured.")
  print(f"  Test Data Path     : {EVAL_DATA_DIR}")
  print(f"  Flights to Process : {len(local_flights)}")
  print(f"  MIN_SCORE          : {MIN_SCORE}")
  print(f"  MIN_TRACK_LEN      : {MIN_TRACK_LEN}")

  # Initialise Predictor (Reads Env Variables).
  predictor = SegPredictor()

  # Run Inference Over all Flights.
  predictor.run()

  # Print Summary Statement.
  log(LogLevel.INFO, "baseline:inference", "Inference Complete.")
  print(f"  Raw Results Saved to: {os.path.join(RESULTS_DIR, run_name)}")

  # Set Results Directories. 
  RAW_RESULTS_PATH = os.path.join(RESULTS_DIR, run_name, "result.json")
  METRICS_RESULTS_PATH = os.path.join(EVAL_RESULT_DIR, "result.json")

  # Load Raw Inference Results.
  with open(RAW_RESULTS_PATH) as f:
      flight_results = json.load(f)

  # Convert Each Detection from Corner Format to Centre Format.
  for entry in flight_results:
      for det in entry.get("detections", []):
          # Extract Corner Coordinates.
          x1, y1, x2, y2 = det["x"], det["y"], det["w"], det["h"]

          # Compute Width & Height.
          w = x2 - x1
          h = y2 - y1

          # Convert to Center Coordinates.
          det["x"] = x1 + w / 2
          det["y"] = y1 + h / 2
          det["w"] = w
          det["h"] = h

  # Ensure Output Directory Exists.
  os.makedirs(EVAL_RESULT_DIR, exist_ok=True)

  # Save Converted Results in Metrics-Compatible Format.
  with open(METRICS_RESULTS_PATH, "w") as f:
      json.dump(flight_results, f)

  # Print Summary Statement.
  log(LogLevel.INFO, "baseline:inference", "Results Converted and Saved.")
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
      ],
      check=True  # Raises CalledProcessError if the script exits with a non-zero code.
  )

  # Print Message on Completion. 
  log(LogLevel.INFO, "baseline:evaluation", "Metrics Complete.")
# ------------------------------------------------------------------------------


# Call the Main Function.
if __name__ == "__main__":
    main()