# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: predict_ensemble.py %
# Date Created:            %date_created: Mon Mar 02 09:00:00 2021 %
# CM Version               %version: 1 %
# CM Instance              %instance: 2 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Implements the two-stage ensemble inference pipeline for airborne object
# detection. In Stage 1, full-resolution models generate candidate detection
# items and motion offsets across the padded frame. In Stage 2, candidate crops
# are extracted, batched, and refined using high-resolution crop models to rescore
# detection confidence. Supports modular stage-level profiling via StageTimer.
#
# Original solution by Dmytro Poplavskiy (Dmytro.Poplavskiy@gmail.com).
# Integrated and adapted for the introspection module pipeline.
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

from dataclasses import dataclass
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from contextlib import nullcontext

from . import common_utils
from . import config
from . import seg_prediction_to_items

# ------------------------------------------------------------------------------
#                         DATA STRUCTURE DEFINITIONS
# ------------------------------------------------------------------------------
@dataclass
class PredictionCrop:
    """
    Data container representing a candidate crop extracted for Stage-2 rescoring.

    Attributes:
        crop_offset_x (int): Horizontal origin coordinate within the full frame.
        crop_offset_y (int): Vertical origin coordinate within the full frame.
        X (torch.Tensor): Padded two-frame tensor slice for this crop region.
        pred_masks (List[np.ndarray]): Stage-2 segmentation probability masks.
        detected_object (Dict): Candidate detection metadata dictionary.
    """
    # Define the Horizontal Crop Offset.
    crop_offset_x: int

    # Define the Vertical Crop Offset.
    crop_offset_y: int

    # Define the Padded Tensor Crop.
    X: torch.Tensor

    # Define the Stage-2 Prediction Masks.
    pred_masks: List[np.ndarray]

    # Define the Detected Object Metadata.
    detected_object: Dict
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         PROFILING HELPER UTILITIES
# ------------------------------------------------------------------------------
def _stage(timer, name: str, also: str = None):
    """
    Wraps execution within a profiling stage context if a timer is configured.

    Args:
        timer: StageTimer instance or None.
        name (str): Primary profiling stage name.
        also (str, optional): Secondary or sub-category profiling stage name.

    Returns:
        ContextManager: Active stage context manager or nullcontext if disabled.
    """
    # Return the Profiling Stage when a Timer is Available.
    return timer.stage(name, also=also) if timer is not None else nullcontext()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         ENSEMBLE INFERENCE PIPELINE
# ------------------------------------------------------------------------------
def predict_ensemble(
    X: np.ndarray,
    models_full_res: List[torch.nn.Module],
    models_crops: List[torch.nn.Module],
    full_res_threshold: float,
    x_offset: int,
    device: torch.device,
    timer = None
) -> List[Dict]:
    """
    Executes the two-stage ensemble detection and rescoring pipeline.

    Stage 1 runs full-resolution segmentation models to produce candidate
    airborne object detections. Stage 2 extracts local crops around up to 6
    top candidate objects, batches them through crop-refinement models, and
    recomputes confidence scores.

    Args:
        X (np.ndarray): Padded two-frame input array of shape (B, 2, H, W).
        models_full_res (List[torch.nn.Module]): Stage-1 full-resolution models.
        models_crops (List[torch.nn.Module]): Stage-2 crop-refinement models.
        full_res_threshold (float): Confidence threshold for Stage-1 candidates.
        x_offset (int): Negative padding applied along the horizontal axis.
        device (torch.device): Device on which inference is performed.
        timer (optional): StageTimer instance for tracking stage runtimes.

    Returns:
        List[Dict]: Final list of detected objects with rescored confidence values.
    """
    # Initialise The Prediction Results List.
    predictions_full_res = []

    # Disable Gradient Calculation During Inference.
    torch.set_grad_enabled(False)

    # STAGE 1: Full-Resolution Detection & Candidate Extraction
    with _stage(timer, "stage1"):
        # Upload And Normalise the Full Frame Tensor to the Device.
        with _stage(timer, "stage1/upload"):
            X = torch.from_numpy(X).float().to(device) / 255.0

        # Run Forward Pass for Each Full-Resolution Model in Ensemble.
        for i, model in enumerate(models_full_res):
            # Start Profiling for the Current Full-Resolution Model.
            with _stage(timer, f"stage1/model{i}"):
                # Enable Automatic Mixed Precision.
                with torch.amp.autocast(device.type):
                    # Run The Model Forward Pass.
                    pred = model(X)

                # Convert Predicted Logits to Probabilities via Sigmoid.
                pred['mask'] = torch.sigmoid(pred['mask'][0].float())

                # Convert All Prediction Values to CPU NumPy Arrays.
                for k in list(pred.keys()):
                    pred[k] = pred[k][0].float().cpu().detach().numpy()

            # Store the Current Model Predictions.
            predictions_full_res.append(pred)

        # Post-Process Full-Resolution Predictions & Extract Candidates.
        with _stage(timer, "stage1/postproc"):
            # Extract the Prediction Masks from each Full-Resolution Model.
            predictions_masks_full_res = [pred['mask'] for pred in predictions_full_res]
            comb_pred = np.mean(predictions_masks_full_res, axis=0)

            # Mask Out Border Margin Artefacts.
            border_x = -x_offset // 8 + 4
            border_y = 4
            comb_pred[:border_y, :] = 0.0
            comb_pred[-border_y-4:, :] = 0.0
            comb_pred[:, :border_x] = 0.0
            comb_pred[:, -border_x:] = 0.0

            # Extract Initial Detection Candidates from Blended Probability Mask.
            pred = predictions_full_res[0]
            detected_objects_full_res = seg_prediction_to_items.pred_to_items(
                comb_pred      = comb_pred,
                offset         = pred['offset'],
                size           = pred['size'],
                tracking       = pred['tracking'],
                distance       = pred['distance'][0],
                above_horizon  = pred['above_horizon'][0],
                conf_threshold = full_res_threshold,
                pred_scale     = 8,
                x_offset       = x_offset,
            )

    # Count the Number of Detection Candidates.
    n_candidates = len(detected_objects_full_res)

    # Return Early If No Candidates or Crop Models are Available.
    if n_candidates == 0 or len(models_crops) == 0:
        # Store Candidate & Crop Counts in the Timer Metadata.
        if timer is not None:
            timer.set_meta(n_candidates=n_candidates, n_crops=0)

        # Return the Full-Resolution Detection Results.
        return detected_objects_full_res

    # Define the Maximum Number of Stage-2 Crop Objects.
    max_nb_objects = 6

    # Limit The Candidates to the Maximum Number of Stage-2 Crops.
    detected_objects_full_res = detected_objects_full_res[:max_nb_objects]

    # Count the Number of Crops to Process.
    n_crops = len(detected_objects_full_res)

    # Store Candidate & Crop Counts in the Timer Metadata.
    if timer is not None:
        timer.set_meta(n_candidates=n_candidates, n_crops=n_crops)


    # STAGE 2: Crop Extraction, Batch Refinement & Rescoring
    h, w = X.shape[-2:]
    crop_size = 512
    res = []

    # Start Profiling for Stage 2.
    with _stage(timer, "stage2", also=f"stage2_by_n/{n_crops}"):
        # Initialise the Crop List.
        crops = []

        # Extract Centred 512x512 Crops Around Candidate Objects.
        with _stage(timer, "stage2/crops_build"):
            # Process Each Full-Resolution Candidate Object.
            for obj in detected_objects_full_res:
                # Clamp the Candidate X & Y Coordinates to the Crop Bounds.
                cx = int(np.clip(obj['cx'], crop_size / 2, w - crop_size / 2))
                cy = int(np.clip(obj['cy'], crop_size / 2, h - crop_size / 2))

                # Align Crop Offsets to 32-Pixel Stride Boundaries.
                crop_offset_x = (cx - crop_size // 2) // 32 * 32
                crop_offset_y = (cy - crop_size // 2) // 32 * 32

                # Extract the Crop Tensor from the Full Frame.
                X_crop = X[:, :, crop_offset_y:crop_offset_y+crop_size, crop_offset_x:crop_offset_x+crop_size]

                # Create & Store the Prediction Crop Object.
                crops.append(
                    PredictionCrop(
                        X               = X_crop,
                        crop_offset_x   = crop_offset_x,
                        crop_offset_y   = crop_offset_y,
                        detected_object = obj,
                        pred_masks      = [],
                    )
                )

            # Combine All Crop Tensors into a Single Batch.
            X_crop_combined = torch.cat([c.X for c in crops], dim=0)

        # Run Forward Pass for Each Crop Model Across Combined Batch.
        for i, model in enumerate(models_crops):
            # Start Profiling for the Current Crop Model.
            with _stage(timer, f"stage2/model{i}"):
                # Enable Automatic Mixed Precision.
                with torch.amp.autocast(device.type):
                    # Run The Crop Model Forward Pass.
                    pred = model(X_crop_combined)

                # Convert Crop Logits to Probabilities.
                pred_mask = torch.sigmoid(pred['mask'].float()).cpu().detach().numpy()

                # Distribute Each Prediction Mask to its Corresponding Crop.
                for j, crop in enumerate(crops):
                    crop.pred_masks.append(pred_mask[j, 0])

                # Release the Crop Prediction Tensor.
                del pred


        # Rescore Candidates by Combining Stage 1 & Stage 2 Confidence.
        with _stage(timer, "stage2/rescore"):
            # Process Each Candidate Crop for Rescoring.
            for crop in crops:
                # Retrieve the Horizontal & Vertical Crop Offset.
                crop_offset_x = crop.crop_offset_x
                crop_offset_y = crop.crop_offset_y

                # Slice Stage-1 Mask Corresponding to Crop Window and Blend with Stage 2.
                predictions_masks_crops = [
                    p[crop_offset_y//8: (crop_offset_y+crop_size)//8, crop_offset_x//8: (crop_offset_x+crop_size)//8]
                    for p in predictions_masks_full_res
                ] + crop.pred_masks

                # Average all Stage-1 & Stage-2 Prediction Masks.
                comb_pred_crop = np.mean(predictions_masks_crops, axis=0)

                # Calculate the Candidate X & Y Coordinate within the Crop.
                crop_x = np.round((crop.detected_object['cx'] - crop_offset_x - x_offset) / 8 - 0.5)
                crop_y = np.round((crop.detected_object['cy'] - crop_offset_y) / 8 - 0.5)

                # Clamp the Candidate X & Y Coordinate to Valid Prediction Bounds.
                crop_x = int(np.clip(crop_x, 1, comb_pred_crop.shape[-1] - 2))
                crop_y = int(np.clip(crop_y, 1, comb_pred_crop.shape[-2] - 2))

                # Update Detection Confidence with Local Peak Probability.
                conf_updated = np.max(comb_pred_crop[crop_y-1:crop_y+2, crop_x-1:crop_x+2])

                # Store the Updated Detection Confidence.
                crop.detected_object['conf'] = conf_updated

                # Add the Rescored Detection to the Results.
                res.append(crop.detected_object)

    # Return the Results. 
    return res
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         DIAGNOSTIC VERIFICATION ROUTINE
# ------------------------------------------------------------------------------
def check_predict_ensemble() -> None:
    """
    Executes a standalone test flight sequence to verify ensemble prediction.
    """
    # Define The Dataset Partition.
    part = 'part1'

    # Define The Flight Identifier.
    flight_id = '027b2fd1f91d4fdfbf5ceffc3c4280b7'

    # Disable CuDNN Deterministic Mode.
    torch.backends.cudnn.deterministic = False

    # Enable CuDNN Benchmark Mode.
    torch.backends.cudnn.benchmark = True

    # Import The Model Construction Function.
    from introspection_module.winning_solution_star.seg_tracker.train import build_model

    # Configure Model Ensemble Checkpoints.
    models_names = [
        ('100_hrnet32_fpm', 2220),
        ('100_gernet_m_b2_fpm', 2220),
        ('100_edet_b5_fpm', 1560),
    ]

    # Initialise The Model List.
    models = []

    # Load Each Model & Its Checkpoint.
    for model_name, epoch in models_names:
        # Build the Model from tts Configuration.
        model = build_model(common_utils.load_config_data(model_name))

        # Move the Model to the CUDA Device.
        model = model.cuda()

        # Set the Model to Evaluation Mode.
        model.eval()

        # Define the Checkpoint Directory.
        checkpoints_dir = f"../output/checkpoints/{model_name}/0"

        # Print the Checkpoint Path.
        print(f"{checkpoints_dir}/{epoch:03}.pt")

        # Load the Model Checkpoint.
        checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")

        # Load the Checkpoint Weights into the Model.
        model.load_state_dict(checkpoint["model_state_dict"])

        # Add the Loaded Model to the Ensemble.
        models.append(model)

    # Load the Ground Truth Records.
    df = pd.read_csv(f'../data/{part}/ImageSets/groundtruth.csv')

    # Load Precomputed Frame Transforms.
    frame_transforms = pd.read_pickle(f'{config.DATA_DIR}/frame_transforms/{part}/{flight_id}.pkl')

    # Filter Ground Truth Records For the Selected Flight.
    df_flight = df[df.flight_id == flight_id][['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
    
    # Extract the Frame Numbers.
    frame_numbers = df_flight.frame.values
    
    # Build the Full Image File Paths.
    file_names = [f'{config.DATA_DIR}/{part}/Images/{flight_id}/{fn[:-4]}.jpg' for fn in df_flight.img_name]

    # Initialise the Previous Image.
    prev_image = None

    # Process Each Frame in the Flight.
    for i, frame_num in enumerate(tqdm(frame_numbers)):
        # Load the Current Frame as a Grayscale Image.
        image = cv2.imread(file_names[i], cv2.IMREAD_GRAYSCALE)

        # Skip First Frame (Requires Previous Frame for Alignment).
        if prev_image is None:
            prev_image = image
            continue

        # Retrieve the Current Frame Transforms & Translations.
        tr = frame_transforms.iloc[i]
        dx = tr['dx']
        dy = tr['dy']
        angle = tr['angle']

        # Define the Input Batch Size.
        batch_size = 1

        # Retrieve the Current Image Dimensions.
        h, w = image.shape

        # Calculate the Required Horizontal Padding.
        padding = (16 + 32 + 64) // 2

        # Initialise the Two-Frame Input Array
        X = np.zeros((batch_size, 2, h, w + padding * 2), dtype=np.uint8)

        # Build the Affine Alignment Transform.
        prev_tr = common_utils.build_geom_transform(
            dst_w         = w,
            dst_h         = h,
            src_center_x  = w / 2 + dx,
            src_center_y  = h / 2 + dy,
            scale_x       = 1.0,
            scale_y       = 1.0,
            angle         = angle,
            return_params = True,
        )

        # Warp Previous Frame to Current Frame Coordinates.
        prev_img_aligned = cv2.warpAffine(
            prev_image,
            prev_tr[:2, :],
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
        )

        # Populate the Input Array with the Aligned & Current Frames.
        X[0, 0, :, padding:-padding] = prev_img_aligned
        X[0, 1, :, padding:-padding] = image

        # Update the Previous Image For the Next Iteration.
        prev_image = image

        # Execute the Two-Stage Ensemble Prediction.
        detected_objects = predict_ensemble(
            X                  = X,
            models_full_res    = models[:2],
            models_crops       = models[2:],
            full_res_threshold = 0.5,
            x_offset           = -padding,
            device             = torch.device('cuda'),
        )

        # Skip Frames where No Objects were Detected.
        if len(detected_objects) == 0:
            continue
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    check_predict_ensemble()
# ------------------------------------------------------------------------------
