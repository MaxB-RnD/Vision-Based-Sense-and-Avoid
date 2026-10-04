# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: warmup_models.py %
# Date Created:            %date_created: Sun Oct 03 23:21:45 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Warm-up routine for all torch.compile'd input shapes and dynamic batches across
# the ensemble and single-backbone predictor pipelines.
#
# Pre-executes compilation passes, kernel autotuning, and CUDA graph captures
# before any flight sequence begins so that JIT-compilation stalls do not distort
# operational inference times. Dummy tensors are used as compilation depends
# strictly on tensor shapes, data types, batch dimensions, and precision/autocast
# context rather than image pixel content.
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

import gc
import json
import os
import time
from typing import Tuple, List, Dict, Any, Optional

import torch

from src.introspection_module.utils.timing_profiling import full_frame_input_shape

# ------------------------------------------------------------------------------
#                           CONFIGURATION CONSTANTS
# ------------------------------------------------------------------------------
# Execution Timeouts Lifted to Accommodate Cold Graph Compilation.
SETUP_TIMEOUT_S = int(os.getenv("WARMUP_SETUP_TIMEOUT_SECONDS", 21600))   # 6 hours
FLIGHT_TIMEOUT_S = int(os.getenv("WARMUP_FLIGHT_TIMEOUT_SECONDS", 14400)) # 4 hours

# Minimum Repetitions Per Shape: Compile, Graph Capture, Replay.
REPS = int(os.getenv("WARMUP_REPS", 3))

# Stage-2 Batch Dimension Bounds and Crop Geometry.
MAX_CROPS = 6    # Matches max_nb_objects.
CROP      = 512

# Model and Stage Profile Identifiers.
LBL_TRANSFORM_ENS = "transform (2048x1280)"
LBL_TRANSFORM_T2D = "transform (1024x1024)"
LBL_SEG1          = "seg1 HRNet32 full"
LBL_SEG2          = "seg2 GerNet-M full"
LBL_T2D           = "HRNet-W32 full fp16"
CROP_MODELS       = ("seg3 HRNet48", "dla DLA60")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         LABEL AND CONFIG HELPERS
# ------------------------------------------------------------------------------
def crop_label(model: str, n: int) -> str:
    """
    Constructs a formatted profiling label for a Stage-2 crop batch.

    Args:
        model (str): Name of the crop-refinement model.
        n (int): Batch size representing the number of candidate crops.

    Returns:
        str: Human-readable stage identifier.
    """
    # Return the Model Name & Crop Batch Size as a Label.
    return f"{model} crops n={n}"


def ensemble_labels() -> List[str]:
    """
    Generates all model and batch labels present in the ensemble pipeline.

    Returns:
        List[str]: Combined list of Stage-1 and Stage-2 model labels.
    """
    # Return the Transform, Full-Resolution & Crop Model Labels.
    return (
        [LBL_TRANSFORM_ENS, LBL_SEG1, LBL_SEG2]
        + [
            crop_label(m, n)
            for n in range(1, MAX_CROPS + 1)
            for m in CROP_MODELS
        ]
    )


def t2d_labels() -> List[str]:
    """
    Generates all model labels present in the tracking-to-detect pipeline.

    Returns:
        List[str]: List of model labels evaluated during T2D inference.
    """
    # Return the T2D Transform & Segmentation Model Labels.
    return [LBL_TRANSFORM_T2D, LBL_T2D]


def enabled() -> bool:
    """
    Checks whether model warm-up is enabled via environment variables.

    Returns:
        bool: True if WARMUP is not explicitly set to '0'.
    """
    # Return False When WARMUP is Explicitly Disabled.
    return os.getenv("WARMUP", "1") != "0"


def apply_timeout_buffer(predictor: Any) -> None:
    """
    Extends predictor timeout thresholds to avoid premature task aborts.

    Cold compilations and autotuning passes can take significant time;
    lifts inference setup and per-flight timeout ceilings.

    Args:
        predictor: AirbornePredictor instance.
    """
    # Increase the Setup Timeout When the Existing Value is Lower.
    predictor.inference_setup_timeout  = max(predictor.inference_setup_timeout,  SETUP_TIMEOUT_S)

    # Increase the Flight Timeout When the Existing Value is Lower.
    predictor.inference_flight_timeout = max(predictor.inference_flight_timeout, FLIGHT_TIMEOUT_S)


def configure_compile_limits(limit: int = 64) -> None:
    """
    Expands TorchDynamo compilation cache boundaries.

    Prevents Dynamo from silently falling back to eager evaluation when
    multiple input shapes are cached in sequence.

    Args:
        limit (int): Base threshold for cache and recompile limits.
    """
    # Import the TorchDynamo Configuration Module.
    import torch._dynamo

    # Get the TorchDynamo Configuration Object.
    cfg = torch._dynamo.config

    # Update the Standard Cache and Recompile Limits.
    for name in ("cache_size_limit", "recompile_limit"):
        # Check Whether the Configuration Value Exists.
        if hasattr(cfg, name):
            # Try to Increase the Limit Without Reducing the Existing Value.
            try:
                setattr(cfg, name, max(int(getattr(cfg, name)), limit))

            # Ignore Configuration Errors.
            except Exception:
                pass

    # Update the Accumulated Cache and Recompile Limits.
    for name in ("accumulated_cache_size_limit", "accumulated_recompile_limit"):
        # Check Whether the Configuration Value Exists.
        if hasattr(cfg, name):
            # Try to Increase the Accumulated Limit.
            try:
                setattr(cfg, name, max(int(getattr(cfg, name)), limit * 8))

            # Ignore Configuration Errors.
            except Exception:
                pass


def unique_graphs() -> Optional[int]:
    """
    Retrieves the total count of unique TorchDynamo compiled graphs.

    Returns:
        Optional[int]: Number of compiled graphs, or None if unavailable.
    """
    # Try to Read the TorchDynamo Compilation Counters.
    try:
        from torch._dynamo.utils import counters

        # Return the Number of Unique Compiled Graph.
        return int(counters["stats"]["unique_graphs"])

    # Return None When the Counter is Unavailable.
    except Exception:
        return None


def resolve_full_shape(test_data_path: str) -> Tuple[int, int, int, int]:
    """
    Resolves the Stage-1 full-frame input tensor shape.

    Reads from the WARMUP_FULL_SHAPE environment override if specified;
    otherwise infers dimensions from the first available flight frame.

    Args:
        test_data_path (str): Path to the test flight datasets.

    Returns:
        Tuple[int, int, int, int]: Shape tuple (B, C, H, W).
    """
    # Read the Optional Shape Override from the Environment.
    env = os.getenv("WARMUP_FULL_SHAPE")

    # Use the Environment Shape When One is Provided.
    if env:
        # Convert the Comma-Separated Shape into Integers.
        return tuple(int(x) for x in env.split(","))

    # Otherwise Infer the Shape from the Dataset.
    return full_frame_input_shape(test_data_path)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         DEVICE EXECUTION HARNESS
# ------------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    """
    Synchronises host execution with the CUDA device stream.

    Args:
        device (torch.device): Device on which operations are executing.
    """
    # Check Whether the Current Device is CUDA.
    if device.type == "cuda":
        # Wait for All CUDA Operations to Complete.
        torch.cuda.synchronize()


def _run(model: torch.nn.Module, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> float:
    """
    Executes a single forward pass with synthetic random tensors.

    Args:
        model (torch.nn.Module): Target model to exercise.
        shape (Tuple[int, ...]): Tensor dimensions.
        dtype (torch.dtype): Data type of the input tensor.
        device (torch.device): Device on which inference is performed.

    Returns:
        float: Elapsed execution time in seconds.
    """
    # Create a Random Input Tensor with the Requested Shape & Data Type.
    x = torch.rand(shape, device=device, dtype=dtype)

    # Synchronise Before Starting the Timer.
    _sync(device)

    # Record the Start Time.
    t0 = time.perf_counter()

    # Run the Model Without Gradient Tracking.
    with torch.no_grad():
        # Enable Automatic Mixed Precision for the Active Device
        with torch.amp.autocast(device.type):
            # Execute the Model Forward Pass.
            out = model(x)

    # Synchronise Before Recording the End Time.
    _sync(device)

    # Calculate the Elapsed Execution Time.
    dt = time.perf_counter() - t0

    # Release the Output and Input Tensors.
    del out, x

    # Return the Measured Execution Time.
    return dt


def probe_shape(model: torch.nn.Module, shape: Tuple[int, ...], dtype: torch.dtype,
                device: torch.device, repeats: int = 5) -> Tuple[float, float]:
    """
    Measures the initial compilation overhead versus steady-state latency.

    Args:
        model (torch.nn.Module): Target model to benchmark.
        shape (Tuple[int, ...]): Tensor dimensions.
        dtype (torch.dtype): Tensor data type.
        device (torch.device): Computation device.
        repeats (int): Number of post-compilation iterations.

    Returns:
        Tuple[float, float]: (first_call_seconds, median_steady_state_seconds).
    """
    # Run the First Pass to Capture Compilation Overhead.
    first = _run(model, shape, dtype, device)

    # Run Additional Passes to Measure Steady-State Performance.
    rest  = sorted(_run(model, shape, dtype, device) for _ in range(repeats))

    # Return the First-Call Time & Median Steady-State Time.
    return first, rest[len(rest) // 2]


def warm_model(model: torch.nn.Module, shape: Tuple[int, ...], dtype: torch.dtype,
               device: torch.device, label: str, report: Dict[str, Any], reps: Optional[int] = None) -> None:
    """
    Repeatedly executes a model over a specific tensor shape to compile graphs.

    Args:
        model (torch.nn.Module): Model to warm up.
        shape (Tuple[int, ...]): Input tensor shape.
        dtype (torch.dtype): Tensor precision.
        device (torch.device): Execution device.
        label (str): Logging label for the stage.
        report (Dict[str, Any]): Warm-up tracking dictionary.
        reps (Optional[int]): Number of repetitions.
    """
    # Use the Provided Repetition Count or the Default Value.
    reps = reps or REPS

    # Print the Shape Being Compiled.
    print(f"  [warm-up] {label:30s} {tuple(shape)} compiling (may take minutes) ...", flush=True)

    # Execute the Model for the Requested Number of Repetitions.
    times = [_run(model, shape, dtype, device) for _ in range(reps)]

    # Store the Warm-Up Results in the Report.
    report["entries"].append(
        dict(
            label   = label,
            shape   = list(shape),
            dtype   = str(dtype).replace("torch.", ""),
            times_s = [round(t, 4) for t in times],
        )
    )

    # Print the Execution Times for Each Repetition.
    print(f"  [warm-up] {label:30s} reps (s): " + " ".join(f"{t:.2f}" for t in times), flush=True)


def warm_transform(model_raw: torch.nn.Module, h: int, w: int, device: torch.device,
                   label: str, report: Dict[str, Any], reps: int = 3) -> None:
    """
    Exercises the eager frame alignment model across cache paths and resets state.

    Args:
        model_raw (torch.nn.Module): Underlying transform model instance.
        h (int): Height of the crop window.
        w (int): Width of the crop window.
        device (torch.device): Execution device.
        label (str): Logging label.
        report (Dict[str, Any]): Warm-up tracking dictionary.
        reps (int): Number of warmup repetitions.
    """
    # Create a List for the Recorded Execution Times.
    times = []

    # Run the Transform Model for Each Warm-Up Repetition.
    for _ in range(reps):
        # Create the First Random Input Tensor.
        a = torch.rand(1, h, w, device=device)

        # Create the Second Random Input Tensor.
        b = torch.rand(1, h, w, device=device)

        # Synchronise Before Starting the Timer.
        _sync(device)

        # Record the Start Time.
        t0 = time.perf_counter()

        # Run the Transform Model Without Gradient Tracking.
        with torch.no_grad():
            # Enable Automatic Mixed Precision.
            with torch.amp.autocast(device.type):
                # Execute the Transform Model.
                out = model_raw(a, b)

        # Synchronise Before Recording the End Time.
        _sync(device)

        # Store the Execution Time.
        times.append(time.perf_counter() - t0)

        # Release the Temporary Tensors.
        del out, a, b

    # Reset the Internal Feature Cache.
    model_raw.reset_cache()

    # Store the Warm-Up Results in the Report.
    report["entries"].append(
        dict(
            label   = label,
            shape   = [1, h, w],
            dtype   = "float32",
            times_s = [round(t, 4) for t in times],
        )
    )

    # Print the Execution Times for Each Repetition.
    print(f"  [warm-up] {label:30s} reps (s): " + " ".join(f"{t:.2f}" for t in times), flush=True)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                      PIPELINE COMPILATION ROUTINES
# ------------------------------------------------------------------------------
def warmup_ensemble(detector: Any, full_shape: Tuple[int, int, int, int], report: Dict[str, Any]) -> None:
    """
    Warms all components and batch dimensions in the ensemble pipeline.

    Compiles the alignment network, Stage-1 full-frame models, and Stage-2
    crop models across batch counts 1 through MAX_CROPS. Clears runtime caches
    and triggers garbage collection upon completion.

    Args:
        detector: SegDetector instance containing the model ensemble.
        full_shape (Tuple[int, int, int, int]): Full frame input shape.
        report (Dict[str, Any]): Warm-up tracking dictionary.
    """
    # Get the Device Used by the Detector.
    dev = detector.device

    # Stop When the Detector is Running on CPU.
    if dev.type != "cuda":
        print("[warm-up] CPU device: nothing to warm.", flush=True)
        return

    # Increase TorchDynamo Compilation Limits.
    configure_compile_limits()

    # Print the Ensemble Warm-Up Configuration.
    print(f"[warm-up] ensemble: full frame {tuple(full_shape)}, crop batches 1..{MAX_CROPS}, reps={REPS}", flush=True)

    # Warm the Frame Alignment Model.
    warm_transform(detector._model_transform_raw, 1280, 2048, dev, LBL_TRANSFORM_ENS, report)

    # Warm the First & Second Stage-1 Full-Frame Model.
    warm_model(detector.model_seg1, full_shape, torch.float32, dev, LBL_SEG1, report)
    warm_model(detector.model_seg2, full_shape, torch.float32, dev, LBL_SEG2, report)

    # Warm the Stage-2 Crop Models for Every Supported Batch Size.
    for n in range(1, MAX_CROPS + 1):
        # Warm Each Crop Model for the Current Batch Size.
        for name, m in zip(CROP_MODELS, (detector.model_seg3, detector.model_dla)):
            # Compile the Current Crop Model & Batch Shape.
            warm_model(m, (n, 2, CROP, CROP), torch.float32, dev, crop_label(name, n), report)

    # Reset the Alignment Cache After Warm-Up.
    detector.reset_transform_cache()

    # Synchronise CUDA Before Finishing.
    torch.cuda.synchronize()

    # Run Garbage Collection.
    gc.collect()


def warmup_t2d(pred: Any, full_shape: Tuple[int, int, int, int], report: Dict[str, Any]) -> None:
    """
    Warms all models in the tracking-to-detect single-backbone pipeline.

    Compiles the transformation estimator and half-precision HRNet-W32
    segmentation backbone, resetting tracking and alignment caches afterwards.

    Args:
        pred: HRNetW32TrackToDetectPredictor instance.
        full_shape (Tuple[int, int, int, int]): Full frame input shape.
        report (Dict[str, Any]): Warm-up tracking dictionary.
    """
    # Get the Device Used by the Predictor.
    dev = pred.device

    # Stop When the Predictor is Running on CPU.
    if dev.type != "cuda":
        print("[warm-up] CPU device: nothing to warm.", flush=True)
        return

    # Increase TorchDynamo Compilation Limits.
    configure_compile_limits()
    print(f"[warm-up] T2D: full frame {tuple(full_shape)} fp16, reps={REPS}", flush=True)

    # Warm Alignment and Half-Precision Segmentation Models.
    warm_transform(pred._model_transform_raw, 1024, 1024, dev, LBL_TRANSFORM_T2D, report)
    warm_model(pred.model_seg, full_shape, torch.float16, dev, LBL_T2D, report)

    # Reset Predictor Alignment and Active Track State.
    pred._model_transform_raw.reset_cache()
    pred.active_tracks = {}

    # Synchronise CUDA Before Finishing.
    torch.cuda.synchronize()

    # Run Garbage Collection.
    gc.collect()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         REPORT SERIALISATION
# ------------------------------------------------------------------------------
def new_report(pipeline: str) -> Dict[str, Any]:
    """
    Initialises an execution timing dictionary for the warm-up session.

    Args:
        pipeline (str): Identifier of the active pipeline.

    Returns:
        Dict[str, Any]: Blank report structure with hardware/software metadata.
    """
    # Return the Initial Report Structure.
    return dict(
        pipeline           = pipeline,
        gpu                = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        torch              = torch.__version__,
        reps               = REPS,
        inductor_cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR"),
        triton_cache_dir   = os.getenv("TRITON_CACHE_DIR"),
        entries            = [],
    )


def finish_report(report: Dict[str, Any], path: str, t_start: float) -> None:
    """
    Finalises and serialises the warm-up performance log to JSON.

    Args:
        report (Dict[str, Any]): Active report dictionary.
        path (str): Target filesystem path for the output file.
        t_start (float): Perf counter timestamp captured at warm-up initiation.
    """
    # Calculate & Store the Total Warm-Up Duration.
    report["total_s"] = round(time.perf_counter() - t_start, 1)

    # Store the Number of Unique Compiled Graphs.
    report["unique_graphs_after"] = unique_graphs()

    # Ensure the Output Directory Exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Open the Output File for Writing.
    with open(path, "w") as f:
        # Write the Report as Formatted JSON.
        json.dump(report, f, indent=2)

    # Print the Final Warm-Up Duration & Output Path.
    print(f"[warm-up] done in {report['total_s']} s -> {path}", flush=True)
# ------------------------------------------------------------------------------
