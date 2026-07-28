# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: generate_frame_transforms.py %
# Date Created:            %date_created: Wed May 6 15:35:45 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Generates frame-to-frame camera motion transforms for all flights in the AOT
# dataset using a pre-trained ResNet34-TSM transformation estimator.
#
# For each flight:
#   - Loads sequential frames
#   - Applies centre cropping (default 1024x1024)
#   - Estimates relative motion between consecutive frames
#   - Outputs per-frame transform parameters:
#         dx, dy, rotation angle, and estimation error
#
# Results are saved as:
#   - .pkl (fast loading for training pipelines)
#   - .csv (human-readable inspection/debugging)
#
# USAGE INSTRUCTIONS:
#   1. Run from project root:
#        python -m src.baselines.winning_solution.seg_tracker.generate_frame_transforms
#
#   2. Optional arguments:
#        --parts 1 2 3
#        --flights-file configs/datasets/training_subsets/train.txt
#        --skip-existing
#        --batch-size 4
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
import argparse
from typing import List, Optional

import mpipe
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import src.core_improvement.winning_solution_star.seg_tracker.config as config
import src.core_improvement.winning_solution_star.seg_tracker.models_transformation as models_transformation

from src.core_improvement.winning_solution_star.seg_tracker.load_images_pipe import load_img, decode_img, limited_pipe
from src.core_improvement.winning_solution_star.seg_tracker.offset_grid_to_transform import offset_grid_to_transform_params

# ------------------------------------------------------------------------------
#                           TRANSFORM ESTIMATION
# ------------------------------------------------------------------------------
def load_transform_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the pre-trained TrEstimatorTsm model from a checkpoint file.
    This is the ResNet34-TSM model used to estimate camera motion between frames.
    """
    # Instantiate the Model Architecture (No Weights Yet).
    model = models_transformation.TrEstimatorTsm(
        cfg=dict(
            base_model_name="resnet34",
            weight_scale=3,
        ),
        pretrained=False,
    )

    # Load Checkpoint (Maps Tensors to Correct Device).
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load Model Weights.
    model.load_state_dict(checkpoint["model_state_dict"])
        
    # Move Model to Device (GPU/CPU).
    model = model.to(device)
    
    # Set to Eval Mode (Disables Dropout, Batchnorm Updates).
    model.eval()
    print(f"  Transform model loaded from: {model_path}")
    return model


def estimate_transforms_for_batch(
    model: torch.nn.Module,
    img_crops: List[np.ndarray],
    device: torch.device,
) -> List[List[float]]:
    """
    Runs the transform model on a batch of consecutive image crops.
    Returns a list of [dx, dy, angle, error] for each consecutive pair.

    img_crops: list of N grayscale images — produces N-1 transform estimates.
    """
    # Prev Points Grid (32x32 at Stride 32, Trimmed by 2 on Each Side = 28x28).
    prev_points = np.zeros((2, 32, 32), dtype=np.float32)
    prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
    prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]
    prev_points = prev_points[..., 2:-2, 2:-2]  # trim to 28x28

    # Center Coordinates Around Image Center (for Rotation Math).
    center_offset = np.array([512.0, 512.0])[:, None]

    # Stack Image Crops into Tensor & Normalise to [0,1].
    all_frames = torch.from_numpy(
        np.stack(img_crops, axis=0) / 255.0
    ).float().to(device)

    # Split into Consecutive Frame Pairs.
    cur_frame  = all_frames[1:]
    prev_frame = all_frames[:-1]

    # Inference (No Gradients, Mixed Precision for Speed).
    with torch.no_grad():
        with torch.amp.autocast(device.type):
            heatmap, offsets = model(prev_frame, cur_frame)

    # Move Results to CPU Numpy.
    heatmap = heatmap.detach().float().cpu().numpy()
    offsets = offsets.detach().float().cpu().numpy()

    results = []

    # Process Each Pair in Batch.
    for i in range(heatmap.shape[0]):
        # Apply Predicted Offsets to Grid Points.
        cur_points = prev_points + offsets[i]

        # Convert Grid Motion into Global Transform Parameters.
        dx, dy, angle, err = offset_grid_to_transform_params(
            prev_frame_points=prev_points.reshape(2, -1) - center_offset,
            cur_frame_points=cur_points.reshape(2, -1) - center_offset,
            points_weight=heatmap[i].reshape(-1) ** 2,
        )
        results.append([dx, dy, angle, err])

    return results


def generate_transforms_for_flight(
    flight_id: str,
    part: str,
    df_flight: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str,
    batch_size: int = 4,
    crop_size: int = 1024,
) -> str:
    """
    Generates and saves the frame transform pkl for a single flight.
    """
    # Extract Numeric Part (e.g. "part1" --> "1").
    part_num = part.replace('part', '')

    # Build Full File Paths for All Frames in this Flight.
    file_names = [
        f'{config.DATA_DIR}/{part}/Images/{flight_id}/{fn[:-4]}.{config.IMG_FORMAT}'
        for fn in df_flight.img_name
    ]

    # Set-Up Async Image Loading Pipeline.
    load_stage = mpipe.OrderedStage(load_img, 1)
    decode_stage = mpipe.OrderedStage(decode_img, 8)
    load_stage.link(decode_stage)
    pipe = mpipe.Pipeline(load_stage)

    # Fallback Image (Used if Load Fails).
    last_img = np.zeros((crop_size, crop_size), dtype=np.uint8)
    load_failed = []
    frame_transforms = [[0.0, 0.0, 0.0, 0.0]]  # First Frame has No Previous.
    img_crops = []

    # Iterate through Pipeline.
    for fn, img in limited_pipe(pipe, file_names, max_queue=32):
        failed = img is None
        load_failed.append(failed)

        # If Load Fails, Reuse Last Image.
        if img is None:
            img = last_img
        last_img = img

        # Centre Crop to crop_size x crop_size.
        h, w = img.shape
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        img_crop = img[y0:y0 + crop_size, x0:x0 + crop_size]
        img_crops.append(img_crop)

        # Process Batch Once We Have Enough Frames.
        if len(img_crops) >= batch_size + 1:
            batch_results = estimate_transforms_for_batch(model, img_crops, device)
            frame_transforms += batch_results
            img_crops = [img_crops[-1]]  # Keep Last Frame as Context for Next Batch.

    # Process Any Remaining Frames.
    if len(img_crops) > 1:
        batch_results = estimate_transforms_for_batch(model, img_crops, device)
        frame_transforms += batch_results

    # Signal Pipeline to Shut Down.
    pipe.put(None)  

    # Build Output DataFrame.
    frame_transforms_arr = np.array(frame_transforms)

    # Ensure Length Matches Number of Frames.
    n_frames = len(df_flight)
    n_tr = len(frame_transforms_arr)

    if n_tr < n_frames:
        pad = np.zeros((n_frames - n_tr, 4))
        frame_transforms_arr = np.vstack([frame_transforms_arr, pad])
    elif n_tr > n_frames:
        frame_transforms_arr = frame_transforms_arr[:n_frames]

    # Attach Metadata & Transform Outputs.
    df_out = df_flight.copy()
    df_out['load_failed'] = load_failed[:n_frames] + [False] * max(0, n_frames - len(load_failed))
    df_out['dx'] = frame_transforms_arr[:, 0]
    df_out['dy'] = frame_transforms_arr[:, 1]
    df_out['angle'] = frame_transforms_arr[:, 2]
    df_out['error'] = frame_transforms_arr[:, 3]

    # Save Results.
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f'{flight_id}.pkl')
    csv_path = os.path.join(output_dir, f'{flight_id}.csv')
    df_out.to_pickle(pkl_path)
    df_out.to_csv(csv_path, index=False)

    return pkl_path
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               MAIN
# ------------------------------------------------------------------------------
def load_flight_ids_from_file(filepath: str) -> List[str]:
    """Reads flight IDs from a split txt file (train.txt / val.txt / test.txt)."""
    # Open the File for Reading.
    with open(filepath) as f:
        lines = f.readlines()
    
    # Keep Only Valid Lines (Starting with 'part').
    return [l.strip() for l in lines if l.strip().startswith('part')]


def run(
    parts: List[int],
    model_path: str,
    output_base: str,
    flights_file: Optional[str],
    skip_existing: bool,
    batch_size: int,
    device: torch.device,
):
    # Print Header for Esy Reading.
    print(f"\n{'='*60}")
    print(f"  Frame Transform Generation")
    print(f"  Parts      : {parts}")
    print(f"  Output     : {output_base}")
    print(f"  Model      : {model_path}")
    print(f"  Device     : {device}")
    print(f"  Skip exist : {skip_existing}")
    if flights_file:
        print(f"  Flights    : {flights_file}")
    print(f"{'='*60}\n")

    # Load Model Once (Reuse for All Flights).
    model = load_transform_model(model_path, device)

    # Optional Filtering by Split File.
    allowed_flight_ids = None
    if flights_file:
        raw_ids = load_flight_ids_from_file(flights_file)
        allowed_flight_ids = set()

        # Remove "partX" Prefix --> Keep Only flight_id.
        for fid in raw_ids:
            raw = fid[5:] if fid.startswith(('part1', 'part2', 'part3')) else fid
            allowed_flight_ids.add(raw)
        print(f"Restricting to {len(allowed_flight_ids)} flights from {flights_file}\n")

    # Count Totals.
    total_flights   = 0
    skipped_flights = 0
    failed_flights  = []

    # Loop Over Dataset Parts.
    for part_num in parts:
        part    = f'part{part_num}'
        gt_path = f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv'
        out_dir = os.path.join(output_base, part)

        # Skip Missing Parts.
        if not os.path.exists(gt_path):
            print(f"WARNING: groundtruth.csv not found for {part}: {gt_path} — skipping.")
            continue
    
        df_gt = pd.read_csv(gt_path)
        
        # Unique Flights in this Part.
        flight_ids = list(sorted(df_gt['flight_id'].unique()))

        # Apply Flight Filter if Provided.
        if allowed_flight_ids is not None:
            flight_ids = [f for f in flight_ids if f in allowed_flight_ids]

        print(f"\n{part}: {len(flight_ids)} flights to process")
        os.makedirs(out_dir, exist_ok=True)

        # Process Each Flight.
        for flight_id in tqdm(flight_ids, desc=f"  {part}"):
            total_flights += 1
            pkl_path = os.path.join(out_dir, f'{flight_id}.pkl')

            # Skip Already Processed Flights.
            if skip_existing and os.path.exists(pkl_path):
                skipped_flights += 1
                continue

            try:
                # Extract Per-Flight Frame List.
                df_flight = (
                    df_gt[df_gt.flight_id == flight_id][['img_name', 'frame']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

                # Generate Transforms.
                generate_transforms_for_flight(
                    flight_id  = flight_id,
                    part       = part,
                    df_flight  = df_flight,
                    model      = model,
                    device     = device,
                    output_dir = out_dir,
                    batch_size = batch_size,
                )

            except Exception as e:
                print(f"\n  ERROR: {part}/{flight_id}: {e}")
                failed_flights.append(f"{part}/{flight_id}")
                continue

    # Summary Print Statement. 
    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Total processed : {total_flights - skipped_flights}")
    print(f"  Skipped (exist) : {skipped_flights}")
    print(f"  Failed          : {len(failed_flights)}")

    if failed_flights:
        print(f"  Failed flights  :")
        for f in failed_flights:
            print(f"    {f}")

    print(f"  Transforms saved to: {output_base}")
    print(f"{'='*60}\n")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate frame-to-frame alignment transforms for all flights."
    )
    parser.add_argument(
        "--parts", type=int, nargs="+", default=[1, 2, 3],
        help="Which dataset parts to process (default: 1 2 3).",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="configs/models/core_improvement/ResNet34_Transforms_Adjusted.pt",
        help="Path to pre-trained transform model weights.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Base output directory. Default: data/frame_transforms/",
    )
    parser.add_argument(
        "--flights-file", type=str, default=None,
        help="Optional path to a split txt file (train.txt etc.) to restrict "
             "processing to specific flights. Useful for testing.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip flights that already have a .pkl file. Safe to re-run.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of consecutive frames per forward pass (default: 4).",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (not recommended — very slow).",
    )

    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    output_base = args.output_dir or os.path.join(config.DATA_DIR, "frame_transforms")

    run(
        parts          = args.parts,
        model_path     = args.model_path,
        output_base    = output_base,
        flights_file   = args.flights_file,
        skip_existing  = args.skip_existing,
        batch_size     = args.batch_size,
        device         = device,
    )
# ------------------------------------------------------------------------------
