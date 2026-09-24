# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: notebook_helpers.py %
# Date Created:            %date_created: Mon Mar 23 21:38:21 2026 %
# CM Version               %version: 2 %
# CM Instance              %instance: 3 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Helper functions for the custom local baseline inference notebook.
# Provides utilities for:
#   - Parsing flight ID .txt files produced by data_builder.py.
#   - Resolving flight IDs to locally downloaded folder paths.
#   - Managing auto-incrementing run output directories.
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
import re
import json

import torch
import numpy as np
import pandas as pd

from glob import glob
from fvcore.nn import FlopCountAnalysis, parameter_count

from pathlib import Path
from typing import Optional
from IPython.display import display
from logging_module import LogLevel, log

sys.path.insert(0, os.path.dirname(__file__))
from src.baselines.winning_solution.evaluator.airborne_detection import AirbornePredictor

# ------------------------------------------------------------------------------
#                           FLIGHT ID PARSING
# ------------------------------------------------------------------------------
def parse_flight_txt(txt_path: str) -> list:
    """
    Parses a flight ID .txt file produced by data_builder.py and returns a
    list of (part, flight_hash) tuples for every valid flight entry.

    Flight IDs in the file are formatted with the part prefix fused directly
    onto the hash:

      part100bb96a5a68f4fa5bc5c5dc66ce314d2
      part2dea1a03fdfe54981b103569bd20a5e95

    Lines that form part of the header block or do not begin with a recognised
    part prefix are silently ignored.

    Args:
        txt_path (str): Path to the flight-ID .txt file.

    Returns:
        list: A list of (part, flight_hash) tuples, e.g.:
              [("part1", "00bb96a5a68f4fa5bc5c5dc66ce314d2"), ...]
    """
    # Store Parsed (part, flight_hash) Pairs.
    entries = []

    # Open the Flight ID File for Reading.
    with open(txt_path) as f:
        # Remove Leading/Trailing Whitespace & Newline Characters.
        for line in f:
            line = line.strip()

            # Check Each Known Part Prefix and Extract the Trailing Hash.
            for part in ["part1", "part2", "part3"]:
                # Ensure the Line Starts with the Part Prefix & Contains a Hash After it.
                if line.startswith(part) and len(line) > len(part):
                    flight_hash = line[len(part):]
                    entries.append((part, flight_hash))
                    break
                
    # Return All Valid Parsed Entries.
    return entries
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           RUN DIRECTORY MANAGEMENT
# ------------------------------------------------------------------------------
def get_next_run_dir(base_dir: str) -> str:
    """
    Returns the name of the next available auto-incrementing run directory
    under base_dir, without creating it.

    Directories are named run0, run1, run2, ... The returned name is the
    first index whose path does not yet exist on disk.

    Args:
        base_dir (str): Path to the parent directory that holds run folders.

    Returns:
        str: The next available run directory name, e.g. "run3".
    """
    # Start from Run Index 0.
    i = 0

    # Increment Until a Run Index that Does Not Yet Exist is Found.
    while os.path.exists(os.path.join(base_dir, f"run{i}")):
        i += 1

    # Return the First Available Run Directory Name.
    return f"run{i}"


def get_last_run(results_dir: str) -> Optional[str]:
    """
    Returns the name of the most recently created run directory under
    results_dir, determined by the highest numeric suffix.

    Args:
        results_dir (str): Path to the directory containing run folders.

    Returns:
        Optional(str): The name of the highest-numbered run folder (e.g. "run4"),
                    or None if no run folders are found.
    """
    # List to Store Tuples for Valid Run Directories.
    runs = []

    # Iterate Over all Entries in the Results Directory.
    for name in os.listdir(results_dir):
        # Check if the Folder Name Matches the 'runN' Pattern.
        match = re.match(r"run(\d+)", name)
        if match:
            runs.append((int(match.group(1)), name))

    # Return None if No Run Folders Exist Yet.
    if not runs:
        return None

    # Return the Run Folder with the Highest Numeric Index.
    return max(runs, key=lambda x: x[0])[1]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           GROUND TRUTH LOADING
# ------------------------------------------------------------------------------
def load_combined_gt(parts: list, local_flights: dict, dataset_root: str) -> dict:
    """
    Loads and merges the groundtruth.json file from each part's ImageSets/
    folder, filtering down to only the flights that are present on disk.
 
    Each part's groundtruth.json contains entries for every flight registered
    in that part of the AOT dataset, including flights that have not been
    downloaded. Entries whose key is not associated with a local flight are
    silently skipped - this is expected behaviour, not an error.
  
    Args:
        parts (list): List of part names to load, e.g. ["part1", "part2", "part3"].
        local_flights (dict): The local_flights dict returned by resolve_local_flights(), 
                              keyed by folder name with each value containing 'part' 
                              and 'flight_hash'.
        dataset_root  (str): Absolute path to the root of the data directory.
 
    Returns:
        dict: A merged ground truth dict in the standard AOT format:
              { "samples": { flight_hash: <annotations>, ... } }
              containing only the flights present on disk.
    """
    # Initialise the Combined GT Dict. & Tracking Variables.
    combined = {"samples": {}}
    local_hashes_by_part = {}
    total_in_file = 0
    total_kept = 0
 
    # Build a Lookup of Flight Hashes per Part to Avoid Duplicates.
    for info in local_flights.values():
        part = info["part"]
        local_hashes_by_part.setdefault(part, set()).add(info["flight_hash"])
  
    # Iterate through Each Part Specified.
    for part in parts:
        # Path to the Part's Ground Truth JSON.
        gt_path = Path(dataset_root) / part / "ImageSets" / "groundtruth.json"
 
        # Skip if the Ground Truth File is Missing for this Part.
        if not gt_path.exists():
            log(LogLevel.WARNING, "notebook_helpers:load_gt", "Not Found, Skipping: %s", str(gt_path))
            continue
 
        # Load the Ground Truth JSON for this Part.
        with open(gt_path) as f:
            gt = json.load(f)
 
        # Get the Set of Local Flight Hashes for this Part.
        part_hashes = local_hashes_by_part.get(part, set())
 
        # Merge Only the Flights Whose Hash Matches a Local Flight in This Part.
        for flight_key, data in gt["samples"].items():
            # Count Every Flight Read from File.
            total_in_file += 1

            # If Key and Hash are Found Merge.
            if flight_key in part_hashes:
                combined["samples"][flight_key] = data
                total_kept += 1

    # Print Final Summary Message.
    print()
    log(LogLevel.INFO, "notebook_helpers:load_gt",
        "%d kept / %d total.", total_kept, total_in_file)

    # Return the Combined Filtered Ground Truth.
    return combined
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           FLIGHT RESOLUTION
# ------------------------------------------------------------------------------
def resolve_local_flights(txt_path: str, dataset_root: str) -> dict:
    """
    Cross-references the flight IDs listed in txt_path against what is
    actually present on disk under dataset_root/<part>/Images/.

    Only flights that are both listed in the .txt file AND have a matching
    folder on disk are included in the returned dict. Flights listed in the
    .txt that have not been downloaded are logged and skipped.

    Args:
        txt_path (str): Path to the flight-ID .txt file.
        dataset_root (str): Absolute path to the root of the data directory.

    Returns:
        dict: A dict keyed by flight folder name, where each value contains:
              {
                  "images_dir"  : Path,   # absolute path to the Images sub-folder
                  "part"        : str,    # e.g. "part1"
                  "flight_hash" : str,    # the hash portion of the flight ID
              }
    """
    # Dict & List to Store Flights that Exist/Missing.
    found   = {}
    missing = []

    # Parse Each Tuple from the .txt File.
    for part, flight_hash in parse_flight_txt(txt_path):
        # Path to the Images Folder for this Part.
        images_root = Path(dataset_root) / part / "Images"

        # Skip Immediately if the Entire Part Images Folder is Absent.
        if not images_root.exists():
            missing.append(f"{part}/{flight_hash}")
            continue

        # Search for Any Sub-Folder Whose Name Contains the Flight Hash.
        matches = [
            d for d in images_root.iterdir()
            if d.is_dir() and flight_hash in d.name
        ]

        if matches:
            # Use the First Match (there should only ever be one per hash).
            folder = matches[0]

            # Collect Only Image Files That Physically Exist on Disk.
            image_names = {
                f.name for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
            }

            # Store Metadata for this Flight in the Found Dict.
            found[folder.name] = {
                "images_dir"  : folder,
                "part"        : part,
                "flight_hash" : flight_hash,
                "image_names" : image_names,
            }
        else:
            # No Matching Folder Found, Add to Missing List.
            missing.append(f"{part}/{flight_hash}")

    # Log Summary of Found Vs. Missing Flights.
    log(LogLevel.INFO, "notebook_helpers:resolve_flights",
        f"{len(found)} Flights on Disk. {len(missing)} Not Found. "
        f"(Run the Download Cell to Fetch Missing Flights).")

    # Return Dict of Flights that Exist Locally.
    return found
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       INFERENCE RESULTS BBOX CONVERSION
# ------------------------------------------------------------------------------
def convert_results_to_metrics_format(raw_path: str, out_path: str, input_format="corner"):
    """
    Converts inference output bounding boxes from CORNER format
    to CENTRE format for compatibility with the metrics script.

    Our inference pipeline stores bounding boxes as:
        (x1, y1, x2, y2)  -> incorrectly placed in fields (x, y, w, h)

    The metrics script expects:
        (cx, cy, w, h) where:
            cx = centre x-coordinate
            cy = centre y-coordinate
            w  = width
            h  = height

    This function rewrites each detection in-place to match the
    expected centre-based format.

    Args:
        raw_path (str): Path to the JSON file containing raw inference results.
        out_path (str): Path where the converted JSON will be saved.

    Returns:
        None: Writes the converted results to disk.
    """
    # Load Raw Inference Results from Disk.
    with open(raw_path) as f:
        data = json.load(f)

    # Iterate Through Each Flight Entry.
    for flight in data:

        # Iterate Through Each Detection in the Flight.
        for det in flight["detections"]:

            if input_format == "corner":
                # Extract Stored Values (Actually x1, y1, x2, y2).
                x1, y1, x2, y2 = det["x"], det["y"], det["w"], det["h"]

                # Compute Width and Height from Corner Coordinates.
                w = x2 - x1
                h = y2 - y1

                # Convert to Centre Coordinates.
                cx = x1 + w / 2
                cy = y1 + h / 2

                # Overwrite Detection with Correct Format.
                det["x"] = cx
                det["y"] = cy
                det["w"] = w
                det["h"] = h

    # Write Converted Results Back to Disk.
    with open(out_path, "w") as f:
        json.dump(data, f)

    # Log Completion of Conversion Step.
    log(LogLevel.INFO, "notebook_helpers:convert_results",
        f"Centre-Formated Result Conversion Complete.")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       MODEL COMPLEXITY MEASUREMENT
# ------------------------------------------------------------------------------
def measure_model_complexity(
    model        : "torch.nn.Module",
    input_shape  : "tuple | None" = (1, 2, 2048, 2448),
    input_override: "tuple | None" = None,
    device       : str = "cuda",
    label        : str = "",
) -> dict:
    """
    Measures the parameter count and FLOPs (GFLOPs) for a given model.

    Uses fvcore's FlopCountAnalysis for FLOPs. This counts multiply-add
    operations (MACs) and reports them as FLOPs (1 MAC = 2 FLOPs), which
    is the standard convention for comparing model complexity in the
    literature.

    Args:
        model (torch.nn.Module): A torch.nn.Module to profile.
        input_shape (tuple): The (B, C, H, W) input tensor shape to use for the
                             FLOPs analysis.
        input_override (tuple): Optional tuple of tensors to pass directly as the
                                model's forward arguments. Use for models that take
                                more than one input tensor. Overrides input_shape.
        device (str): "cuda" or "cpu". FLOPs analysis runs on CPU regardless
                       but model is moved for the dummy forward pass.
        label (str): Optional name for logging output.

    Returns:
        dict with keys:
            total_params (int): Total number of parameters.
            trainable_params (int): Number of trainable parameters.
            gflops (float): GFLOPs for one forward pass at input_shape.
            gmacs (float): GMACs (= gflops / 2).
    """
    # Set Model to Evaluation Mode (Disables Dropout, etc.).
    model.eval()

    # Count Parameters.
    param_counts = parameter_count(model)
    total_params = int(param_counts[""])

    # Count Only Trainable Parameters.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set fallback value for GFLOPs & GMACs.
    gflops = -1.0
    gmacs = -1.0

    # Check if an Input Shape OR Input Override is Provided.
    if input_shape is not None or input_override is not None:
        try:
            # Create Dummy Input Tensor.
            if input_override is not None:
                # Use Provided Input Override.
                dummy = input_override
            else:
                # Fill with Zeros if Nothing Provided.
                dummy = torch.zeros(input_shape)

            # Initialise FLOPs Counter Using fvcore.
            flop_counter = FlopCountAnalysis(model.cpu(), dummy)

            # Disable Warnings for Unsupported Operations & Unused Models.
            flop_counter.unsupported_ops_warnings(False)
            flop_counter.uncalled_modules_warnings(False)

            # Compute Total FLOPs (Counts MACs * 2 Internally).
            total_flops = flop_counter.total()
            gflops = round(total_flops / 1e9, 3) # Convert FLOPs to GFLOPs.
            gmacs  = round(gflops / 2.0, 3)      # Convert GFLOPs to GMACs.

        except Exception as e:
            # Create Tag for Storing the Model Name.
            tag = f"[{label}] " if label else ""

            # Log Error if FLOPs Computation Fails.
            log(LogLevel.WARNING, "notebook_helpers:measure_complexity",
                "%s FLOPs Count Failed (%s). GFLOPS set to -1.", tag, str(e))

    # Move Model Back to Original Device.
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()

    # Create Tag for Storing the Model Name.
    tag = f"[{label}] " if label else ""
    
    # Log Summary of Results.
    log(LogLevel.INFO, "notebook_helpers:measure_complexity",
        "%sParams: %s total / %s trainable | GFLOPs: %.3f | GMACs: %.3f",
        tag, f"{total_params:,}", f"{trainable_params:,}", gflops, gmacs)

    # Return Summary Statistics. 
    return {
        "total_params"    : total_params,
        "trainable_params": trainable_params,
        "gflops"          : gflops,
        "gmacs"           : gmacs,
    }


def measure_predictor_complexity(predictor: "AirbornePredictor") -> dict:
    """
    Measures the total complexity of any predictor by calling its
    get_complexity_targets() method. This is the function the notebook should 
    call.

    Args:
        predictor (AirbornePredictor): Any AirbornePredictor subclass that
                                       implements get_complexity_targets().

    Returns:
        dict: Summed total_params, trainable_params, gflops, gmacs across
              all targets. If any target fails FLOPs counting, gflops and
              gmacs are set to -1.0 for the whole result.
    """
    # Ensure the Predictor Exposes the Required Interface.
    if not hasattr(predictor, "get_complexity_targets"):
        raise AttributeError(
            f"{type(predictor).__name__} Does Not Implement get_complexity_targets(). "
            f"Add this Method to the Currently Selected Predictor."
        )

    # Retrieve All Model "targets" (Sub-Models or Components to Evaluate).
    targets = predictor.get_complexity_targets()

    # Initialise Running Totals for All Complexity Metrics.
    totals = {
        "total_params"    : 0,    # All Parameters (trainable + frozen).
        "trainable_params": 0,    # Only Parameters Updated During Training.
        "gflops"          : 0.0,  # Floating Point Operations (in Billions).
        "gmacs"           : 0.0,  # Multiply-Accumulate Ops (in Billions).
    }

    # Track Whether FLOPs Computation Succeeds for All Targets.
    flops_valid = True

    # Iterate Over Each Target Model/Component.
    for target in targets:
        # Compute Complexity for the Current Target.
        result = measure_model_complexity(
            model          = target["model"],               # Required: Model Instance.
            input_shape    = target.get("input_shape"),     # Optional: Shape-Based Input.
            input_override = target.get("input_override"),  # Optional: Custom Input Tensor.
            device         = target.get("device", "cuda"),  # Default to GPU if Unspecified.
            label          = target.get("label", ""),       # Optional: Logging/Debug Label.
        )

        # Accumulate Parameter Counts (Always Safe to Sum).
        totals["total_params"]     += result["total_params"]
        totals["trainable_params"] += result["trainable_params"]

        # If Any Model Fails FLOPs Calculation, Mark Entire Result as Invalid.
        if result["gflops"] == -1.0:
            flops_valid = False

        # Only Accumulate FLOPs/MACs if All Previous Computations are Valid.
        if flops_valid:
            totals["gflops"] += result["gflops"]
            totals["gmacs"]  += result["gmacs"]
    
    # Finalise FLOPs/MACs Results.
    if not flops_valid:
        # If Any Target Failed, Invalidate the Aggregate FLOPs/MACs.
        totals["gflops"] = -1.0
        totals["gmacs"]  = -1.0
    else:
        # Otherwise, Round for Cleaner Reporting.
        totals["gflops"] = round(totals["gflops"], 3)
        totals["gmacs"]  = round(totals["gmacs"],  3)

    # Log a Summary for Visibility in Notebooks/Logs.
    log(LogLevel.INFO, "notebook_helpers:measure_predictor_complexity",
        "Total - Params: %s | GFLOPs: %.3f | GMACs: %.3f",
        f"{totals['total_params']:,}", totals["gflops"], totals["gmacs"])

    # Return Aggregated Complexity Metrics.
    return totals
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           FPS STATS RECORDING
# ------------------------------------------------------------------------------
def record_fps_stats(
    frame_times_sec : list,
    run_dir         : str,
    warmup_frames   : int = 10,
) -> dict:
    """
    Computes FPS statistics from a list of per-frame inference times (seconds)
    and saves them to a JSON file inside run_dir.

    The first `warmup_frames` entries are excluded from all statistics to avoid
    GPU JIT compilation and CUDA warm-up artefacts skewing results.

    Args:
        frame_times_sec (list): List of per-frame wall-clock times in seconds.
                                Measure with time.perf_counter() bracketed by
                                torch.cuda.synchronize() calls.
        run_dir (str): Path to the current run directory — fps_stats.json
                       will be written here.
        warmup_frames (int): Number of initial frames to discard before computing
                             statistics. Default 10.

    Returns:
        dict with keys:
            fps_mean  (float): Average across all runs.
            fps_std   (float): Standard deviation across all runs. 
            fps_min   (float): Slowest sequence recorded.
            fps_max   (float): Fastest sequence recorded.
            fps_p95   (float): 95th percentile FPS (robust upper bound).
            n_frames  (int): Number of frames included after warmup.
    """
    # Remove Warmup Frames to Avoid GPU/JIT Bias.
    times = frame_times_sec[warmup_frames:]

     # Check if Any Valid Frames Remain.
    if len(times) == 0:
        # Log Warning if No Usable Data.
        log(LogLevel.ERROR, "notebook_helpers:record_fps",
            "No Frames Remaining After Warmup Discard. Returning Zeros.")
        # Return Zeroed Statistics Safely.
        return {
            "fps_mean": 0.0, "fps_std": 0.0, "fps_min":  0.0, 
            "fps_max": 0.0, "fps_p95":  0.0, "n_frames": 0,
        }

    # Convert Each Frame Time into FPS.
    fps_values = [1.0 / t for t in times if t > 0]
    fps_arr = np.array(fps_values, dtype=np.float64)

    # Build Statistics Dictionary.
    stats = {
        "fps_mean" : round(float(np.mean(fps_arr)), 2),
        "fps_std"  : round(float(np.std(fps_arr)), 2),
        "fps_min"  : round(float(np.min(fps_arr)), 2),
        "fps_max"  : round(float(np.max(fps_arr)), 2),
        "fps_p95"  : round(float(np.percentile(fps_arr, 95)), 2),
        "n_frames" : len(fps_values),
    }

    # Ensure Run Directory Exists Before Writing File.
    os.makedirs(run_dir, exist_ok=True)

    # Define Output JSON File Path.
    fps_path = os.path.join(run_dir, "fps_stats.json")

    # Open File in Write Mode.
    with open(fps_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Log Final Summary Statistics.
    print()
    log(LogLevel.INFO, "notebook_helpers:record_fps",
        "FPS - Mean: %.1f | Std: %.1f | Min: %.1f | Max: %.1f | P95: %.1f (%d frames)\n",
        stats["fps_mean"], stats["fps_std"],
        stats["fps_min"],  stats["fps_max"],
        stats["fps_p95"],  stats["n_frames"])
    
    # Return Computed Statistics Dictionary.
    return stats


def load_fps_stats(run_dir: str) -> dict:
    """
    Loads the fps_stats.json written by record_fps_stats() for a given run.

    Args:
        run_dir (str): Path to the run directory containing fps_stats.json.

    Returns:
        dict with fps_mean, fps_std, fps_min, fps_max, fps_p95, n_frames,
        or a dict of zeros if the file does not exist.
    """
    # Construct Expected File Path.
    fps_path = os.path.join(run_dir, "fps_stats.json")

    # Check if Stats File Exists.
    if not os.path.exists(fps_path):
        log(LogLevel.ERROR, "notebook_helpers:load_fps",
            "fps_stats.json not found at: %s\n", fps_path)
        # Return Safe Default Values.
        return {
            "fps_mean": 0.0, "fps_std": 0.0, "fps_min":  0.0, 
            "fps_max": 0.0, "fps_p95":  0.0, "n_frames": 0,
        }
    
    # Open Existing JSON File for Reading.
    with open(fps_path) as f:
        return json.load(f)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                            ORIGINAL SUMMARY METRICS
# ------------------------------------------------------------------------------
def build_summary(
    summary_json_path : str,
    run_dir           : str,
    complexity        : dict,
    output_csv_path   : str,
) -> "pd.DataFrame":
    """
    Builds the full benchmark summary DataFrame using the orginal competition metrics.

    Args:
        summary_json_path (str): Path to the metrics summary JSON produced by
                                 run_airborne_metrics.py.
        run_dir (str): Path to the run directory (used to load fps_stats.json).
        complexity (str): Dict returned by measure_model_complexity().
        output_csv_path (str): Full path where the CSV should be saved.

    Returns:
        pd.DataFrame with Metric / Value / Status columns, saved to CSV.
    """
    # Load Detection Metrics JSON.
    with open(summary_json_path) as f:
        summary = json.load(f)

    # Extract Detection Metrics.
    fppi         = summary["fppi"]
    hfar         = summary["far"]
    afdr         = summary["fl_dr_in_range"]
    det_aircraft = summary["det_aircraft_in_range"]
    tot_aircraft = summary["tot_aircraft_in_range"]
    edr_all      = summary["Detection"]["Encounters"]["300"]["All"]["dr"]
    edr_above    = summary["Detection"]["Encounters"]["300"]["Above Horizon"]["dr"]
    edr_below    = summary["Detection"]["Encounters"]["300"]["Below Horizon"]["dr"]

    # Compare to the Thresholds to Set Status.
    fppi_status = "PASS (≤ 0.0005)" if fppi <= 0.0005 else f"FAIL (> 0.0005)"
    hfar_status = "PASS (≤ 0.5)"    if hfar <= 0.5    else f"FAIL (> 0.5)"

    # Load FPS Stats (Written by record_fps_stats During Inference).
    fps = load_fps_stats(run_dir)

    # Build Unified Rows.
    rows = [
        # Detection Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "AFDR (Airborne Flight Detection Rate)",       "Value": f"{afdr:.4f}",                                                                "Status": ""},
        {"Metric": "FPPI (False Positives Per Image)",            "Value": f"{fppi:.6f}",                                                                "Status": fppi_status},
        {"Metric": "HFAR (Horizontal False Alarm Rate)",          "Value": f"{hfar:.4f}",                                                                "Status": hfar_status},
        {"Metric": "Total Aircraft Detected",                     "Value": f"{det_aircraft}",                                                            "Status": ""},
        {"Metric": "Total Aircraft in Dataset",                   "Value": f"{tot_aircraft}",                                                            "Status": ""},

        # Tracking Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "EDR - All Scenarios",                         "Value": f"{edr_all:.4f}",                                                             "Status": ""},
        {"Metric": "EDR - Above Horizon",                         "Value": f"{edr_above:.4f}",                                                           "Status": ""},
        {"Metric": "EDR - Below Horizon",                         "Value": f"{edr_below:.4f}",                                                           "Status": ""},

        # FPS Statistics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "FPS Mean",                                    "Value": f"{fps['fps_mean']:.2f}",                                                     "Status": ""},
        {"Metric": "FPS Std Dev",                                 "Value": f"{fps['fps_std']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS Min",                                     "Value": f"{fps['fps_min']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS Max",                                     "Value": f"{fps['fps_max']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS P95",                                     "Value": f"{fps['fps_p95']:.2f}",                                                      "Status": ""},
        {"Metric": "Frames Measured",                             "Value": f"{fps['n_frames']}",                                                         "Status": ""},

        # Model Complexity
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "Parameters (Total)",                          "Value": f"{complexity['total_params']:,}",                                            "Status": ""},
        {"Metric": "Parameters (Trainable)",                      "Value": f"{complexity['trainable_params']:,}",                                        "Status": ""},
        {"Metric": "GFLOPs (per frame)",                          "Value": f"{complexity['gflops']:.3f}",                                                "Status": ""},
        {"Metric": "GMACs (per frame)",                           "Value": f"{complexity['gmacs']:.3f}",                                                 "Status": ""},
    ]

    # Save the Summary to a Dataframe. 
    summary_df = pd.DataFrame(rows)

    # Save the Summary Statistics to a CSV.
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    summary_df.to_csv(output_csv_path, index=False)
    log(LogLevel.INFO, "notebook_helpers:build_summary",
        "Summary Saved to: %s", output_csv_path)

    # Return the Dataframe. 
    return summary_df
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   EXTENDED SUMMARY WITH CEDR & FTPS
# ------------------------------------------------------------------------------
def build_extended_summary(
    summary_json_path : str,
    run_dir           : str,
    complexity        : dict,
    output_csv_path   : str,
) -> "pd.DataFrame":
    """
    Builds the full benchmark summary DataFrame, extending the standard
    detection metrics with more accurate tracking metrics CEDR & FTPS.

    Args:
        summary_json_path (str): Path to the metrics summary JSON produced by
                                 run_airborne_metrics.py.
        run_dir (str): Path to the run directory (used to load fps_stats.json).
        complexity (str): Dict returned by measure_model_complexity().
        output_csv_path (str): Full path where the CSV should be saved.

    Returns:
        pd.DataFrame with Metric / Value / Status columns, saved to CSV.
    """
    # Load Detection Metrics JSON.
    with open(summary_json_path) as f:
        summary = json.load(f)

    # Extract Detection Metrics.
    fppi         = summary["fppi"]
    hfar         = summary["far"]
    afdr         = summary["fl_dr_in_range"]
    det_aircraft = summary["det_aircraft_in_range"]
    tot_aircraft = summary["tot_aircraft_in_range"]
    edr_all      = summary["Detection"]["Encounters"]["300"]["All"]["dr"]
    edr_above    = summary["Detection"]["Encounters"]["300"]["Above Horizon"]["dr"]
    edr_below    = summary["Detection"]["Encounters"]["300"]["Below Horizon"]["dr"]

    # CEDR & FTPS --> Use .get() Rather than Direct Indexing, Since Older summary.json Files Won't Have this Key.
    cedr_mean     = summary.get("cedr_mean")
    cedr_n        = summary.get("cedr_num_qualifying_encounters")
    ftps_rates    = summary.get("ftps_rate_by_threshold", {})
    ftps_counts   = summary.get("ftps_count_by_threshold", {})
    ftps_averaged = summary.get("ftps_averaged")
    ftps_eligible = summary.get("ftps_num_eligible_tracks")

    # Load FPS Stats (Written by record_fps_stats During Inference).
    fps = load_fps_stats(run_dir)

    # Compare to the Thresholds to Set Status.
    fppi_status = "PASS (≤ 0.0005)" if fppi <= 0.0005 else f"FAIL (> 0.0005)"
    hfar_status = "PASS (≤ 0.5)"    if hfar <= 0.5    else f"FAIL (> 0.5)"

    # Build Unified Rows.
    rows = [
        # Detection Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "AFDR (Airborne Flight Detection Rate)",       "Value": f"{afdr:.4f}",                                                                "Status": ""},
        {"Metric": "FPPI (False Positives Per Image)",            "Value": f"{fppi:.6f}",                                                                "Status": fppi_status},
        {"Metric": "HFAR (Horizontal False Alarm Rate)",          "Value": f"{hfar:.4f}",                                                                "Status": hfar_status},
        {"Metric": "Total Aircraft Detected",                     "Value": f"{det_aircraft}",                                                            "Status": ""},
        {"Metric": "Total Aircraft in Dataset",                   "Value": f"{tot_aircraft}",                                                            "Status": ""},

        # Tracking Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "EDR - All Scenarios",                         "Value": f"{edr_all:.4f}",                                                             "Status": ""},
        {"Metric": "EDR - Above Horizon",                         "Value": f"{edr_above:.4f}",                                                           "Status": ""},
        {"Metric": "EDR - Below Horizon",                         "Value": f"{edr_below:.4f}",                                                           "Status": ""},
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "CEDR (Continuous Encounter Detection Rate)",  "Value": f"{cedr_mean:.4f}" if cedr_mean is not None else "N/A",                       "Status": f"N = {cedr_n}" if cedr_n is not None else ""},
        {"Metric": "FTPS@50 (False Tracks Per Sequence)",         "Value": f"{ftps_rates.get('50'):.4f}" if ftps_rates.get('50') is not None else "N/A", "Status": f"N = {ftps_counts.get('50')}/{ftps_eligible} Tracks" if ftps_eligible is not None else ""},
        {"Metric": "FTPS@80 (False Tracks Per Sequence)",         "Value": f"{ftps_rates.get('80'):.4f}" if ftps_rates.get('80') is not None else "N/A", "Status": f"N = {ftps_counts.get('80')}/{ftps_eligible} Tracks" if ftps_eligible is not None else ""},
        {"Metric": "FTPS@95 (False Tracks Per Sequence)",         "Value": f"{ftps_rates.get('95'):.4f}" if ftps_rates.get('95') is not None else "N/A", "Status": f"N = {ftps_counts.get('95')}/{ftps_eligible} tracks" if ftps_eligible is not None else ""},
        {"Metric": "FTPS avg(50:95)",                             "Value": f"{ftps_averaged:.4f}" if ftps_averaged is not None else "N/A",               "Status": f"N = {ftps_eligible} Eligible Tracks" if ftps_eligible is not None else ""},

        # FPS Statistics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "FPS Mean",                                    "Value": f"{fps['fps_mean']:.2f}",                                                     "Status": ""},
        {"Metric": "FPS Std Dev",                                 "Value": f"{fps['fps_std']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS Min",                                     "Value": f"{fps['fps_min']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS Max",                                     "Value": f"{fps['fps_max']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS P95",                                     "Value": f"{fps['fps_p95']:.2f}",                                                      "Status": ""},
        {"Metric": "Frames Measured",                             "Value": f"{fps['n_frames']}",                                                         "Status": ""},

        # Model Complexity
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "Parameters (Total)",                          "Value": f"{complexity['total_params']:,}",                                            "Status": ""},
        {"Metric": "Parameters (Trainable)",                      "Value": f"{complexity['trainable_params']:,}",                                        "Status": ""},
        {"Metric": "GFLOPs (per frame)",                          "Value": f"{complexity['gflops']:.3f}",                                                "Status": ""},
        {"Metric": "GMACs (per frame)",                           "Value": f"{complexity['gmacs']:.3f}",                                                 "Status": ""},
    ]

    # Save the Summary to a Dataframe. 
    summary_df = pd.DataFrame(rows)

    # Save the Summary Statistics to a CSV.
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    summary_df.to_csv(output_csv_path, index=False)
    log(LogLevel.INFO, "notebook_helpers:build_extended_summary",
        "Extended Summary Saved to: %s", output_csv_path)

    # Return the Dataframe. 
    return summary_df
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                        TRIMMED INTROSPECTION SUMMARY
# ------------------------------------------------------------------------------
def build_introspection_summary(
    summary_json_path : str,
    run_dir            : str,
    output_csv_path    : str,
) -> "pd.DataFrame":
    """
    Builds a benchmark summary containing only the metrics relevant to
    the introspection module evaluation, omitting metrics outside its
    evaluation scope.

    Args:
        summary_json_path (str): Path to the metrics summary JSON produced
                                 by run_airborne_metrics.py.
        run_dir (str): Path to the run directory (used to load fps_stats.json).
        output_csv_path (str): Full path where the trimmed CSV should be saved.

    Returns:
        pd.DataFrame with Metric / Value / Status columns, saved to CSV.
    """
    # Load Detection Metrics JSON.
    with open(summary_json_path) as f:
        summary = json.load(f)

    # Extract Only the Original Metrics This Trimmed Summary Reports.
    afdr         = summary["fl_dr_in_range"]
    det_aircraft = summary["det_aircraft_in_range"]
    tot_aircraft = summary["tot_aircraft_in_range"]

    # CEDR & FTPS -- Use .get() Rather than Direct Indexing, Since Older
    # summary.json Files Won't Have these Keys.
    cedr_mean     = summary.get("cedr_mean")
    cedr_n        = summary.get("cedr_num_qualifying_encounters")
    ftps_rates    = summary.get("ftps_rate_by_threshold", {})
    ftps_counts   = summary.get("ftps_count_by_threshold", {})
    ftps_averaged = summary.get("ftps_averaged")
    ftps_eligible = summary.get("ftps_num_eligible_tracks")

    # Extract Introspection Metrics -- Use .get() Rather than Direct Indexing, 
    # Since Older summary.json Files Won't Have these Keys.
    awlt_median = summary.get("awlt_median")
    awlt_q25 = summary.get("awlt_q25")
    awlt_q75 = summary.get("awlt_q75")
    awlt_min = summary.get("awlt_min")
    awlt_max = summary.get("awlt_max")
    num_warned = summary.get("num_warned_in_advance")
    fnr = summary.get("false_negative_rate")
    far = summary.get("false_alarm_rate")
    num_correct = summary.get("num_correct_tracks")
    mean_iou_slope = summary.get("mean_iou_slope")
    pct_drifting = summary.get("pct_tracks_drifting")
    num_drift_tracks = summary.get("num_eligible_drift_tracks")

    # Per-Failure-Type Breakdown Metrics for Introspection -- Use .get() Rather
    # than Direct Indexing, Since Older summary.json Files Won't Have these Keys.
    awlt_median_by_type = summary.get("awlt_median_by_failure_type", {})
    num_false_alarms    = summary.get("num_false_alarms")
    fnr_by_type         = summary.get("false_negative_rate_by_failure_type", {})
    num_failed_by_type  = summary.get("num_failed_tracks_by_failure_type", {})

    def _format_by_type_status(value_by_type, count_by_type, value_fmt):
        """Build a status string showing each metric by failure type and sample count."""
        # No Breakdown Available: Return an Empty Status String.
        if not value_by_type:
            return ""

        # Create a Store for Seperate Error Types.
        parts = []

        # Format the Divergence & Premature-Loss Results Using Consistent Labels.
        for failure_type, label in (("sustained", "Divergence"), ("early_termination", "Premature Loss")):
            v = value_by_type.get(failure_type)
            n = count_by_type.get(failure_type)

            # Show N/A when a Metric is Unavailable for that Failure Type.
            v_str = value_fmt(v) if v is not None else "N/A"

            # Include the Number of Failed Tracks when Available.
            n_str = f" (N={n})" if n is not None else ""

            # Add the Failure Type with its Label. 
            parts.append(f"{label}: {v_str}{n_str}")

        # Combine Both Failure Types into a Single Display String.
        return " | ".join(parts)

    # Format AWLT to 2 Decimal Places for Display.
    awlt_median_status = _format_by_type_status(awlt_median_by_type, num_failed_by_type, lambda v: f"{v:.2f}")

    # Format the AWLT Spread as "Q25-Q75".
    awlt_spread_value = f"{awlt_q25:.2f}-{awlt_q75:.2f}" if awlt_q25 is not None else "N/A"

    # Create an Empty List to Store Additional AWLT Spread Information.
    awlt_spread_notes = []

    # Add the AWLT Range to the Notes when Both the Minimum & Maximum Values are Available.
    if awlt_min is not None and awlt_max is not None:
        awlt_spread_notes.append(f"Range: {awlt_min:.2f}-{awlt_max:.2f}")

    # Add the Number of Warned Tracks to the Notes when the Count is Available.
    if num_warned is not None:
        awlt_spread_notes.append(f"(N = {num_warned})")

    # Combine All AWLT Spread Notes into a Single String.
    awlt_spread_status = " ".join(awlt_spread_notes)

    # Format FNR to 4 Decimal Places for Display.
    fnr_status  = _format_by_type_status(fnr_by_type, num_failed_by_type, lambda v: f"{v:.4f}")

    # Load FPS Stats (Written by record_fps_stats During Inference).
    fps = load_fps_stats(run_dir)

    # Build the Trimmed Row Set.
    rows = [
        # Introspection Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "AWLT Median (s)",                             "Value": f"{awlt_median:.2f}" if awlt_median is not None else "N/A",                  "Status": awlt_median_status},
        {"Metric": "AWLT IQR (s)",                                "Value": awlt_spread_value,                                                           "Status": awlt_spread_status},
        {"Metric": "False Negative Rate",                         "Value": f"{fnr:.4f}" if fnr is not None else "N/A",                                  "Status": fnr_status},
        {"Metric": "False Alarm Rate (FAR)",                      "Value": f"{far:.4f}" if far is not None else "N/A",                                  "Status": f"{num_false_alarms} of {num_correct} Correct Tracks Warned" if (num_false_alarms is not None and num_correct is not None) else ""},
        {"Metric": "Positional Divergence (Mean IoU Slope)",      "Value": f"{mean_iou_slope:.6f}" if mean_iou_slope is not None else "N/A",            "Status": f"N = {num_drift_tracks} Tracks, {pct_drifting:.1%} Drifting" if num_drift_tracks is not None else ""},

        # Detection Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "AFDR (Airborne Flight Detection Rate)",       "Value": f"{afdr:.4f}",                                                                "Status": ""},
        {"Metric": "Total Aircraft Detected",                     "Value": f"{det_aircraft}",                                                            "Status": ""},
        {"Metric": "Total Aircraft in Dataset",                   "Value": f"{tot_aircraft}",                                                            "Status": ""},

        # Tracking Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "CEDR (Continuous Encounter Detection Rate)",  "Value": f"{cedr_mean:.4f}" if cedr_mean is not None else "N/A",                       "Status": f"N = {cedr_n}" if cedr_n is not None else ""},
        {"Metric": "FTPS@50 (False Tracks Per Sequence)",         "Value": f"{ftps_rates.get('50'):.4f}" if ftps_rates.get('50') is not None else "N/A", "Status": f"N = {ftps_counts.get('50')}/{ftps_eligible} Tracks" if ftps_eligible is not None else ""},
        {"Metric": "FTPS@80 (False Tracks Per Sequence)",         "Value": f"{ftps_rates.get('80'):.4f}" if ftps_rates.get('80') is not None else "N/A", "Status": f"N = {ftps_counts.get('80')}/{ftps_eligible} Tracks" if ftps_eligible is not None else ""},
        {"Metric": "FTPS@95 (False Tracks Per Sequence)",         "Value": f"{ftps_rates.get('95'):.4f}" if ftps_rates.get('95') is not None else "N/A", "Status": f"N = {ftps_counts.get('95')}/{ftps_eligible} tracks" if ftps_eligible is not None else ""},
        {"Metric": "FTPS avg(50:95)",                             "Value": f"{ftps_averaged:.4f}" if ftps_averaged is not None else "N/A",               "Status": f"N = {ftps_eligible} Eligible Tracks" if ftps_eligible is not None else ""},

        # FPS Statistics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "FPS Mean",                                    "Value": f"{fps['fps_mean']:.2f}",                                                     "Status": ""},
        {"Metric": "FPS Std Dev",                                 "Value": f"{fps['fps_std']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS Min",                                     "Value": f"{fps['fps_min']:.2f}",                                                      "Status": ""},
        {"Metric": "FPS Max",                                     "Value": f"{fps['fps_max']:.2f}",                                                      "Status": ""},
        {"Metric": "Frames Measured",                             "Value": f"{fps['n_frames']}",                                                         "Status": ""},
    ]

    # Save the Summary to a DataFrame.
    summary_df = pd.DataFrame(rows)

    # Save the Summary Statistics to a CSV.
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    summary_df.to_csv(output_csv_path, index=False)
    log(LogLevel.INFO, "notebook_helpers:build_introspection_summary",
        "Introspection Summary Saved to: %s", output_csv_path)

    # Return the DataFrame.
    return summary_df
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       CLEAN NOTEBOOK SUMMARY TABLE
# ------------------------------------------------------------------------------
def display_styled_summary(summary_df: "pd.DataFrame") -> None:
    """
    Prints a Section Header & Displays a Summary DataFrame with Notebook
    Styling Applied.

    Args:
        summary_df (pd.DataFrame): Summary DataFrame to display, expected to
                                   have Metric / Value / Status columns.

    Returns:
        None: Displays the styled table directly.
    """
    # Style the DataFrame for Cleaner Notebook Display.
    styled_summary = (
        summary_df.style
        .hide(axis="index")                                  # Drop the 0,1,2... Index Column.
        .set_table_styles([
            {"selector": "th",                                # Header Cells.
             "props": [("text-align", "center"),
                       ("background-color", "#2c3e50"),
                       ("color", "white"),
                       ("font-weight", "bold"),
                       ("padding", "6px 12px")]},
            {"selector": "td",                                # Body Cells.
             "props": [("text-align", "center"),
                       ("padding", "4px 12px")]},
            {"selector": "table",
             "props": [("border-collapse", "collapse"),
                       ("width", "100%")]},
        ])
        .set_properties(subset=["Metric"], **{"text-align": "left", "font-weight": "500"})
    )

    # Display the Styled Table.
    display(styled_summary)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       SUBSET RESULTS COLLECTION
# ------------------------------------------------------------------------------
def prepare_subset(
    subset_txt     : str,
    run_dir        : str,
    run_gt_dir     : str,
    dataset_root   : str,
) -> tuple[str, str, str, str]:
    """
    Stitches per-flight result.json files for a subset of flights, converts
    them to metrics format, and filters the groundtruth to match.

    Args:
        subset_txt (str): Path to the .txt file listing subset flight IDs.
        run_dir (str): Path to the run directory (e.g. outputs/results/baseline/run0).
        run_gt_dir (str): Path to the run's gt/ dir (e.g. outputs/results/baseline/run0/gt).
        dataset_root (str): Absolute path to the root of the dataset directory.

    Returns:
        tuple: (subset_name, subset_result_dir, subset_gt_dir, subset_summary_dir)
    """
    #  Derive Subset Name and Build Output Directories.
    subset_name = os.path.splitext(os.path.basename(subset_txt))[0]

    # Build Directory Structure for this Subset.
    subset_base        = os.path.join(run_dir, "subsets", subset_name)
    subset_result_dir  = os.path.join(subset_base, "result")
    subset_gt_dir      = os.path.join(subset_base, "gt")
    subset_summary_dir = os.path.join(subset_base, "summaries")

    # Ensure all Required Directories Exist.
    os.makedirs(subset_result_dir,  exist_ok=True)
    os.makedirs(subset_gt_dir,      exist_ok=True)
    os.makedirs(subset_summary_dir, exist_ok=True)

    # Log Subset Preparation Start.
    log(LogLevel.INFO, "notebook_helpers:prepare_subset", "Preparing Subset: %s\n", subset_name)

    # Resolve Flight IDs.
    subset_flights  = resolve_local_flights(subset_txt, dataset_root)
    subset_ids_full = set(subset_flights.keys())
    subset_ids_hash = {re.sub(r'^part\d+', '', fid) for fid in subset_ids_full}

    # Log How Many Flights are in the Subset.
    log(LogLevel.INFO, "notebook_helpers:prepare_subset", "Subset Contains %d Flights.\n", len(subset_ids_full))

    # Stitch Per-Flight Results Files Together.
    combined_results = []
    missing_flights  = []

    # Look through all Flight ID's.
    for flight_id in subset_ids_full:
        # Build Path to Each Flight's Result.json.
        flight_result_path = os.path.join(run_dir, flight_id, "result.json")

        # If Result File Doesn't Exist, Track & Skip.
        if not os.path.exists(flight_result_path):
            missing_flights.append(flight_id)
            continue
        
        # Load & Append this Flight's Results.
        with open(flight_result_path) as f:
            combined_results.extend(json.load(f))
    
    # Log Warning if Any Flights were Missing Results.
    if missing_flights:
        log(LogLevel.WARNING, "notebook_helpers:prepare_subset",
            "%d / %d Flights Missing Results - Skipped.", len(missing_flights), len(subset_ids_full))

    # Save Combined Results into a Single File.
    with open(os.path.join(subset_result_dir, "result.json"), "w") as f:
        json.dump(combined_results, f)

    # Log How Many Entries were Stitched Together.
    log(LogLevel.INFO, "notebook_helpers:prepare_subset",
        "Stitched %d Entries from %d Flights.\n", len(combined_results), len(subset_ids_full) - len(missing_flights))

    # Convert Stitched Results to Metrics-Ready Format.
    result_path = os.path.join(subset_result_dir, "result.json")
    convert_results_to_metrics_format(result_path, result_path)

    # Log Conversion Completion.
    log(LogLevel.INFO, "notebook_helpers:prepare_subset", "Results Converted to Metrics Format.\n")

    # Filter groundtruth.json to Subset Flights Only.
    with open(os.path.join(run_gt_dir, "groundtruth.json")) as f:
        full_gt = json.load(f)

    # Keep Only Samples whose IDs Match this Subset (Using Hash IDs).
    subset_gt = {
        "samples": {
            flight_id: data
            for flight_id, data in full_gt["samples"].items()
            if any(flight_id.endswith(hash_id) for hash_id in subset_ids_hash)
        }
    }

    # Save Filtered Ground Truth to Subset Directory.
    with open(os.path.join(subset_gt_dir, "groundtruth.json"), "w") as f:
        json.dump(subset_gt, f)

    # Log How Many GT Samples were Kept vs Total.
    log(LogLevel.INFO, "notebook_helpers:prepare_subset",
        "Filtered GT: %d / %d Samples.\n", len(subset_gt["samples"]), len(full_gt.get("samples", {})))

    # Return Key Paths & Identifiers for Downstream Evaluation.
    return subset_name, subset_result_dir, subset_gt_dir, subset_summary_dir


def load_subset_summary(subset_summary_dir: str, subset_name: str) -> pd.DataFrame | None:
    """
    Loads the metrics summary JSON produced by the competition script and
    returns a styled DataFrame ready for display.

    Args:
        subset_summary_dir (str): Path to the summaries folder for this subset.
        subset_name (str): Name of the subset (used for the CSV filename).

    Returns:
        pd.DataFrame or None if no summary file was found.
    """
    # Find the Summary JSON in the Summaries Directory.
    summary_files = glob(os.path.join(subset_summary_dir, "*.json"))

    # Check that the Summary Files Exist.
    if not summary_files:
        log(LogLevel.ERROR, "notebook_helpers:load_subset_summary", "No Summary Files Found.")
        return None
    else:
        # Load the Summary JSON.
        with open(summary_files[0]) as f:
            summary = json.load(f)

    # Extract Key Metrics from Summary JSON.
    fppi         = summary["fppi"]
    hfar         = summary["far"]
    afdr         = summary["fl_dr_in_range"]
    det_aircraft = summary["det_aircraft_in_range"]
    tot_aircraft = summary["tot_aircraft_in_range"]
    edr_all      = summary["Detection"]["Encounters"]["300"]["All"]["dr"]
    edr_above    = summary["Detection"]["Encounters"]["300"]["Above Horizon"]["dr"]
    edr_below    = summary["Detection"]["Encounters"]["300"]["Below Horizon"]["dr"]
    
    # Rank Metrics Against Thresholds.
    fppi_status = "PASS (≤ 0.0005)" if fppi <= 0.0005 else f"FAIL ({fppi:.6f})"
    hfar_status = "PASS (≤ 0.5)"    if hfar <= 0.5    else f"FAIL ({hfar:.4f})"

    # Build Summary DataFrame.
    summary_df = pd.DataFrame([
        # Detection Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "AFDR (Airborne Flight Detection Rate)",       "Value": f"{afdr:.4f}",                                                                "Status": ""},
        {"Metric": "FPPI (False Positives Per Image)",            "Value": f"{fppi:.6f}",                                                                "Status": fppi_status},
        {"Metric": "HFAR (Horizontal False Alarm Rate)",          "Value": f"{hfar:.4f}",                                                                "Status": hfar_status},
        {"Metric": "Total Aircraft Detected",                     "Value": f"{det_aircraft}",                                                            "Status": ""},
        {"Metric": "Total Aircraft in Dataset",                   "Value": f"{tot_aircraft}",                                                            "Status": ""},

        # Tracking Metrics.
        {"Metric": "", "Value": "", "Status": ""},
        {"Metric": "EDR - All Scenarios",                         "Value": f"{edr_all:.4f}",                                                             "Status": ""},
        {"Metric": "EDR - Above Horizon",                         "Value": f"{edr_above:.4f}",                                                           "Status": ""},
        {"Metric": "EDR - Below Horizon",                         "Value": f"{edr_below:.4f}",                                                           "Status": ""},
    ])

    # Save Aggregated Summary to CSV.
    output_path = os.path.join(subset_summary_dir, f"{subset_name}_summary.csv")
    summary_df.to_csv(output_path, index=False)
    log(LogLevel.INFO, "notebook_helpers:load_subset_summary", "Summary Saved to: %s\n", output_path)

    # Return the Data Frame.
    return summary_df
# ------------------------------------------------------------------------------
