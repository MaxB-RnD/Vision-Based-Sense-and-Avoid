# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: train_transformation.py %
# Date Created:            %date_created: Fri Jun 12 21:44:23 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Training, validation, export, and diagnostic entrypoint for the ResNet34-TSM
# frame-to-frame transformation estimation model.
#
# USAGE:
#   1. Train using the command:
#        nohup python -m src.core_improvement.winning_solution_star.seg_tracker.train_transformation \
#            train winning_solution_star/ResNet34_Transforms_Adjusted \
#            > outputs/logs/model_training/train_transformation.log 2>&1 &
#
#   2. Then monitor from terminal with:
#        tail -f outputs/logs/model_training/train_transformation.log
#
#   3. Optional Environment Variables:
#        --fold   INT    Dataset fold index (default: 0).
#        --epoch  INT    Epoch to continue from or export (default: -1).
#        --sample INT    Reserved for future use.
#        --part   STR    Dataset partition for offset prediction (e.g. 'part1').
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

# Standard Library Imports.
import argparse
import os
import random
import re
from pathlib import Path

# Third-Party Imports.
import cv2
import madgrad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pytorch / Deep Learning Imports.
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project-Specific Imports.
import src.core_improvement.winning_solution_star.seg_tracker.config as config
from src.core_improvement.winning_solution_star.seg_tracker.load_images_pipe import *
from src.core_improvement.winning_solution_star.seg_tracker.common_utils import DotDict
import src.core_improvement.winning_solution_star.seg_tracker.common_utils as common_utils
import src.core_improvement.winning_solution_star.seg_tracker.dataset_transform as dataset_transform
import src.core_improvement.winning_solution_star.seg_tracker.models_transformation as models_transformation
import src.core_improvement.winning_solution_star.seg_tracker.offset_grid_to_transform as offset_grid_to_transform


# ------------------------------------------------------------------------------
#                              REPRODUCIBILITY
# ------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """
    Sets the global random seed for full reproducibility across Python, NumPy,
    and PyTorch (CPU and CUDA).

    Args:
        seed (int): Integer seed value to apply globally.

    Returns:
        None
    """
    # Python / OS Level Determinism.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy Deterministic Behaviour.
    np.random.seed(seed)

    # Torch CPU + GPU Seeds.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # NOTE: Benchmark ON for Performance, Deterministic OFF for Speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Set the Global Seed.
seed_everything(seed=42)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def get_next_run_dir(base_dir: str) -> tuple:
    """
    Creates and returns a new numbered run directory under base_dir.
    Run directories are named run0, run1, run2, … and each contains the
    standard subdirectories: checkpoints/, tensorboard/, and oof/.

    Args:
        base_dir (str): Root directory under which run directories are created.

    Returns:
        tuple: (run_dir: Path, run_id: int) — the new run directory path and
               its integer index.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Find the Highest Existing Run Index.
    runs = []
    for p in base.iterdir():
        if p.is_dir():
            m = re.match(r"run(\d+)", p.name)
            if m:
                runs.append(int(m.group(1)))

    next_run = 0 if not runs else max(runs) + 1
    run_dir  = base / f"run{next_run}"

    # Create Standard Run Subdirectories.
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "oof").mkdir(parents=True, exist_ok=True)

    return run_dir, next_run


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiates the transformation estimation model from the experiment config.
    The model class is resolved dynamically from the models_transformation module
    using the 'model_cls' key in the config.

    Args:
        cfg (dict): Full experiment config dictionary loaded from the YAML file.

    Returns:
        nn.Module: Instantiated transformation model with pretrained backbone.
    """
    # Extract Model Configuration Block.
    model_params = cfg['model_params']

    # Dynamically Resolve and Instantiate the Model Class.
    model: nn.Module = models_transformation.__dict__[model_params['model_cls']](
        cfg=model_params, pretrained=True
    )

    # Return the Constructed Model.
    return model
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                  TRAINING
# ------------------------------------------------------------------------------
def train(experiment_name: str, fold: int, continue_epoch: int = -1) -> None:
    """
    Trains the ResNet34-TSM transformation estimation model for a given
    experiment configuration and dataset fold. Supports resuming from a
    saved checkpoint via continue_epoch.

    The model is trained to predict per-pixel optical flow offsets between
    consecutive frame crops, supervised by OpenCV-estimated ground truth
    transformations (or synthetically generated transformations at the ratio
    specified by synthetic_img_ratio in the dataset config).

    Args:
        experiment_name (str): Experiment config name mapping to a YAML file
                               under the experiments/ directory.
        fold (int): Dataset fold index. Training uses all parts except fold+1;
                    validation uses part fold+1.
        continue_epoch (int): Epoch to resume training from. Pass -1 to start
                              a fresh run. Default: -1.

    Returns:
        None
    """
    # Load the Config File.
    cfg          = common_utils.load_config_data(experiment_name)
    model_params = DotDict(cfg["model_params"])
    train_params = DotDict(cfg["train_params"])

    # Set Output Directories Using a New Numbered Run Directory.
    base_out        = Path("outputs/results/model_training/transforms")
    run_dir, run_id = get_next_run_dir(base_out)
    cache_dir       = str(base_out / "data_cache")
    checkpoints_dir = str(run_dir / "checkpoints")
    tensorboard_dir = str(run_dir / "tensorboard")
    oof_dir         = str(run_dir / "oof")

    os.makedirs(cache_dir,       exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(oof_dir,         exist_ok=True)
    print(f"\n Starting run{run_id} --> {run_dir}\n")
    print("\n", experiment_name, "\n")

    # Initialise TensorBoard Logger and Mixed Precision Scaler.
    logger = SummaryWriter(log_dir=tensorboard_dir)
    scaler = torch.amp.GradScaler('cuda')

    # Create Training Dataset (All Parts Except the Validation Fold).
    dataset_train = dataset_transform.DatasetTransform(
        stage=dataset_transform.DatasetTransform.STAGE_TRAIN,
        fold=fold,
        cfg_data=cfg['dataset_params'],
        return_torch_tensors=True,
        small_subset=False,
        cache_dir=cache_dir
    )

    # Create Validation Dataset (The Held-Out Fold Only).
    dataset_valid = dataset_transform.DatasetTransform(
        stage=dataset_transform.DatasetTransform.STAGE_VALID,
        fold=fold,
        cfg_data=cfg['dataset_params'],
        return_torch_tensors=True,
        small_subset=False,
        cache_dir=cache_dir
    )

    # Extract Sampling Parameters from Config.
    batch_size           = cfg['train_data_loader']['batch_size']
    nb_samples_per_epoch = cfg['train_params']['nb_samples_per_epoch']
    nb_samples_valid     = cfg['train_params']['nb_samples_valid']

    # RandomSampler with Replacement Gives a Fixed-Size Epoch Regardless of
    # Actual Dataset Size, Which is Important for Consistent LR Scheduling.
    data_loaders = {
        "train": DataLoader(
            dataset_train,
            num_workers=cfg['train_data_loader']['num_workers'],
            shuffle=False,
            batch_size=batch_size,
            sampler=torch.utils.data.RandomSampler(
                dataset_train,
                num_samples=nb_samples_per_epoch,
                replacement=True
            )
        ),
        "val": DataLoader(
            dataset_valid,
            num_workers=cfg['val_data_loader']['num_workers'],
            shuffle=False,
            batch_size=cfg['val_data_loader']['batch_size'],
            sampler=torch.utils.data.RandomSampler(
                dataset_valid,
                num_samples=nb_samples_valid,
                replacement=True
            )
        ),
    }

    # Make the Model.
    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.train()

    # Set-up the Optimiser.
    initial_lr = float(train_params.initial_lr)
    if train_params.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "madgrad":
        optimizer = madgrad.MADGRAD(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True
        )
    else:
        raise RuntimeError("Invalid optimiser: " + train_params.optimizer)

    # Set-up the Scheduler.
    nb_epochs = train_params.nb_epochs
    if train_params.scheduler == "steps":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_params.optimiser_milestones,
            gamma=0.2,
            last_epoch=continue_epoch
        )
    elif train_params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = common_utils.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_params.scheduler_period,
            T_mult=train_params.get('scheduler_t_mult', 1),
            eta_min=initial_lr / 1000.0,
            last_epoch=-1,
            first_epoch_lr_scale=0.01
        )
        # Fast-Forward Scheduler State to Match the Resume Epoch.
        for i in range(continue_epoch + 1):
            scheduler.step()
    else:
        raise RuntimeError("Invalid scheduler name: " + train_params.scheduler)

    # Resume from Checkpoint Logic.
    if continue_epoch > -1:
        ckpt_path = f"{checkpoints_dir}/{continue_epoch:03}.pt"
        print(ckpt_path)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    grad_clip_value = train_params.get("grad_clip", 8.0)
    print("grad clip:", grad_clip_value)
    print(
        f"Num training samples: {len(dataset_train)} "
        f"validation samples: {len(dataset_valid)}"
    )

    # Weighted MSE Loss: Each Point's Error is Weighted by the Predicted Heatmap.
    # This Encourages the Model to Place High Confidence Only Where it is Accurate.
    cr_mse = torch.nn.MSELoss(reduction='none')

    # Run the Epoch Loop.
    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        for phase in ["train", "val"]:
            # Switch Model Between Training and Evaluation Modes.
            model.train(phase == "train")

            # Initialise Per-Epoch Loss Accumulators.
            epoch_loss          = common_utils.AverageMeter()
            epoch_metric        = common_utils.AverageMeter()
            epoch_loss_separate = defaultdict(common_utils.AverageMeter)

            data_loader = data_loaders[phase]
            optimizer.zero_grad()

            data_iter = tqdm(data_loader)

            # Run the Batch Loop.
            for data in data_iter:
                with torch.set_grad_enabled(phase == "train"):
                    # Load Frame Pair and Corresponding Point Grids onto GPU.
                    cur_frame   = data['cur_img'].cuda()
                    prev_frame  = data['prev_img'].cuda()
                    cur_points  = data['cur_points_grid'].cuda()
                    prev_points = data['prev_points_grid'].cuda()

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        # Forward Pass: Predict Heatmap and Per-Point Flow Offsets.
                        heatmap, offsets = model(prev_frame, cur_frame)

                        # Ground Truth Offsets: Difference Between Corresponding
                        # Point Positions. Trimmed by 2 on Each Side to Match the
                        # Model's Output Spatial Size After Border Trimming.
                        gt_offsets = (cur_points - prev_points)[..., 2:-2, 2:-2]

                        # Weighted MSE Loss: Confidence-Weighted Per-Point Error.
                        loss = (cr_mse(offsets, gt_offsets) * heatmap).sum(
                            dim=(1, 2, 3)
                        ).mean()

                    epoch_metric.update(0, batch_size)

                    if phase == "train":
                        scaler.scale(loss).backward()

                        # Unscale Before Clipping so Threshold is in Real Gradient Units.
                        scaler.unscale_(optimizer)
                        if grad_clip_value > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_value
                            )

                        # scaler.step() Skips the Update if Gradients Contain Infs/NaNs.
                        scaler.step(optimizer)
                        scaler.update()

                    epoch_loss.update(loss.detach().item(), batch_size)
                    data_iter.set_description(
                        f"{epoch_num} {phase[0]} Loss {epoch_loss.avg:1.4f}"
                    )

            # Print the Logging Information.
            if epoch_num > 0:
                logger.add_scalar(f"loss_{phase}", epoch_loss.avg, epoch_num)
                if phase == "train":
                    logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_num)
                logger.flush()

            # Checkpoint (Train Phase Only).
            if phase == "train":
                scheduler.step()

                # Save at Regular Intervals and at the Final Epoch.
                if (epoch_num % train_params.save_period == 0) or (epoch_num == nb_epochs):
                    torch.save(
                        {
                            "epoch"               : epoch_num,
                            "model_state_dict"    : model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        f"{checkpoints_dir}/{epoch_num:03}.pt",
                    )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              DIAGNOSTIC CHECK
# ------------------------------------------------------------------------------
def check(experiment_name: str, fold: int, epoch: int = -1) -> None:
    """
    Runs a diagnostic inference pass over the validation set and prints
    per-sample offset and angle estimates alongside ground truth values.
    Visualises high-loss samples (loss >= 20) using matplotlib to help
    identify failure modes.

    Args:
        experiment_name (str): Experiment config name mapping to a YAML file
                               under the experiments/ directory.
        fold (int): Dataset fold index used to select the validation partition.
        epoch (int): Checkpoint epoch to load for evaluation. Default: -1.

    Returns:
        None
    """
    # Load Config and Override Synthetic Ratio for Real-Data Evaluation.
    cfg             = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"../output/checkpoints/{experiment_name}/{fold}"
    print("\n", experiment_name, "\n")

    # Force All Samples to Real Frame Pairs — No Synthetic Transforms.
    cfg['dataset_params']['synthetic_img_ratio'] = 0.0

    # Load the Validation Dataset.
    dataset_valid = dataset_transform.DatasetTransform(
        stage=dataset_transform.DatasetTransform.STAGE_VALID,
        fold=fold,
        cfg_data=cfg['dataset_params'],
        return_torch_tensors=True,
        small_subset=False
    )

    # Use a Fixed 1024-Sample Subset for Consistent Diagnostic Runtime.
    data_loader = DataLoader(
        dataset_valid,
        num_workers=2,
        shuffle=False,
        batch_size=1,
        sampler=torch.utils.data.RandomSampler(
            dataset_valid, num_samples=1024, replacement=True
        )
    )

    # Load the Model Checkpoint.
    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()
    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    cr_mse = torch.nn.MSELoss(reduction='none')

    # Run the Diagnostic Loop.
    data_iter = tqdm(data_loader)
    for data in data_iter:
        with torch.set_grad_enabled(False):
            # Load Frame Pair and Point Grids onto GPU.
            # Points are Pre-Trimmed to Match the Model's Trimmed Output Grid.
            cur_frame   = data['cur_img'].cuda()
            prev_frame  = data['prev_img'].cuda()
            cur_points  = data['cur_points_grid'][..., 2:-2, 2:-2].cuda()
            prev_points = data['prev_points_grid'][..., 2:-2, 2:-2].cuda()

            # Forward Pass.
            heatmap, offsets = model(prev_frame, cur_frame)

            # Ground Truth Offsets from Pre-Trimmed Point Grid.
            gt_offsets = (cur_points - prev_points)

            # Weighted MSE Loss for This Sample.
            loss = (cr_mse(offsets, gt_offsets) * heatmap).sum(
                dim=(1, 2, 3)
            ).mean()

            # Print Summary Statistics for This Sample.
            print(f'loss {float(loss):0.1f}')
            print(f'heatmap sum {float(heatmap[0].sum()):0.1f}')
            print(
                f'mean gt offset '
                f'{float(gt_offsets[0, 0].mean()):0.1f} '
                f'{float(gt_offsets[0, 1].mean()):0.1f}'
            )
            print(
                f'mean pr offset '
                f'{float(offsets[0, 0].mean()):0.1f} '
                f'{float(offsets[0, 1].mean()):0.1f}'
            )

            # Confidence-Weighted Mean Offset (More Meaningful than Raw Mean).
            weighted_offset = (offsets[0] * heatmap[0]).sum(dim=(1, 2))
            print(
                f'mean pw offset '
                f'{float(weighted_offset[0]):0.1f} '
                f'{float(weighted_offset[1]):0.1f}'
            )

            # Image Centre Used to Zero-Centre Points for Transform Estimation.
            center_offset   = np.array([512.0, 512.0])[:, None]
            cur_points_pred = (prev_points + offsets).detach().cpu().numpy()

            # Estimate Transform with Linear Heatmap Weighting.
            dx, dy, angle, err = offset_grid_to_transform.offset_grid_to_transform_params(
                prev_frame_points=prev_points.cpu().numpy().reshape(2, -1) - center_offset,
                cur_frame_points=cur_points_pred.reshape(2, -1) - center_offset,
                points_weight=heatmap[0].detach().cpu().numpy().reshape(-1)
            )

            # Estimate Transform with Squared Heatmap Weighting (More Robust).
            dx2, dy2, angle2, err2 = offset_grid_to_transform.offset_grid_to_transform_params(
                prev_frame_points=prev_points.cpu().numpy().reshape(2, -1) - center_offset,
                cur_frame_points=cur_points_pred.reshape(2, -1) - center_offset,
                points_weight=heatmap[0].detach().cpu().numpy().reshape(-1) ** 2
            )

            # Estimate Full Affine Transform Matrix (Used at Inference Time).
            tr, _ = offset_grid_to_transform.offset_grid_to_transform(
                prev_frame_points=prev_points.cpu().numpy().reshape(2, -1),
                cur_frame_points=cur_points_pred.reshape(2, -1),
                points_weight=heatmap[0].detach().cpu().numpy().reshape(-1) ** 2
            )

            print(f'Estimated angle {angle:0.2f}  offset {dx:0.1f},{dy:0.1f}  err {err:0.2f}')
            print(f'Estimated angle {angle2:0.2f} offset {dx2:0.1f},{dy2:0.1f} err {err2:0.2f} (sq. weights)')
            print(
                f"GT        angle {float(data['angle'][0]):0.2f} "
                f"offset {float(data['dx'][0]):0.1f},{float(data['dy'][0]):0.1f} "
                f"scale {float(data['scale'][0]):0.3f} "
                f"synthetic: {data['is_synthetic'][0]}"
            )

            # Skip Visualisation for Low-Loss Samples. Only Show Failures.
            if float(loss) < 20:
                continue

            # Visualise Any High-Loss Sample.
            fig, ax = plt.subplots(2, 3)
            ax[0, 0].imshow(data['cur_img'][0].numpy())
            ax[1, 0].imshow(
                (heatmap[0, 0] / heatmap[0, 0].max()).detach().cpu().numpy()
            )
            ax[0, 1].imshow(gt_offsets[0, 0].cpu().numpy())
            ax[0, 2].imshow(gt_offsets[0, 1].cpu().numpy())
            ax[1, 1].imshow(offsets[0, 0].detach().cpu().numpy())
            ax[1, 2].imshow(offsets[0, 1].detach().cpu().numpy())
            plt.figure()

            cur_img  = data['cur_img'][0].numpy()
            prev_img = data['prev_img'][0].numpy()
            h, w     = cur_img.shape

            # Warp Previous Frame Using Squared-Weight Estimated Transform.
            prev_tr2 = common_utils.build_geom_transform(
                dst_w=w, dst_h=h,
                src_center_x=w / 2 + dx2,
                src_center_y=h / 2 + dy2,
                scale_x=1.0, scale_y=1.0,
                angle=angle2,
                return_params=True
            )
            prev_img_aligned2 = cv2.warpAffine(
                prev_img, prev_tr2[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
            )

            # Red=Current, Green=AlignedPrev, Yellow=Well-Aligned Overlap.
            plt.figure()
            plt.imshow(np.stack([cur_img, prev_img_aligned2, cur_img], axis=2))

            # Warp Previous Frame Using Full Affine Matrix.
            prev_img_aligned = cv2.warpAffine(
                prev_img, tr[:2, :],
                dsize=(cur_img.shape[1], cur_img.shape[0]),
                flags=cv2.INTER_LINEAR
            )
            plt.figure()
            plt.imshow(np.stack([cur_img, prev_img_aligned, cur_img], axis=2))
            plt.show()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               MODEL EXPORT
# ------------------------------------------------------------------------------
def export_model(experiment_name: str, fold: int, epoch: int) -> None:
    """
    Exports a trained checkpoint to a minimal inference-only .pth file
    containing only the model state dict. Removes optimiser and scheduler
    state to produce a clean, compact artifact suitable for deployment.

    Output is saved to: ../output/models/{experiment_name}_{epoch}_{fold}.pth

    Args:
        experiment_name (str): Experiment config name mapping to a YAML file
                               under the experiments/ directory.
        fold (int): Dataset fold index used during training.
        epoch (int): Checkpoint epoch to export.

    Returns:
        None
    """
    # Load Config and Rebuild Model Architecture.
    cfg             = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"../output/checkpoints/{experiment_name}/{fold}"
    dst_dir         = '../output/models'
    os.makedirs(dst_dir, exist_ok=True)

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()

    # Load Trained Weights from Checkpoint.
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Save Only the Model Weights. No Optimiser or Scheduler State.
    torch.save(
        {"model_state_dict": model.state_dict()},
        f'{dst_dir}/{experiment_name}_{epoch}_{fold}.pth'
    )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                          BATCH OFFSET PREDICTION
# ------------------------------------------------------------------------------
def find_transforms(model: nn.Module, img_crops: list) -> list:
    """
    Runs the transformation model over a batch of consecutive 1024x1024 image
    crops and returns the estimated affine parameters for each consecutive
    frame pair.

    The model predicts a flow confidence heatmap and per-point flow offsets on
    a 32x32 grid. These are converted to [dx, dy, angle, error] via weighted
    least-squares fitting, using squared heatmap values as weights to
    down-weight uncertain flow estimates.

    Args:
        model (nn.Module): Loaded transformation model on CUDA in eval mode.
        img_crops (list): List of grayscale uint8 numpy arrays, each of shape
                          (1024, 1024). Must contain at least 2 frames.
                          The last frame of each batch should be carried
                          forward as context for the next batch call.

    Returns:
        list: List of [dx, dy, angle, err] per consecutive frame pair.
              Length is len(img_crops) - 1. dx and dy are in pixels,
              angle is in degrees, err is the residual fitting error.
    """
    # Stack Crops into a Batch Tensor and Normalise to [0, 1].
    all_frames = torch.from_numpy(
        np.stack(img_crops, axis=0) / 255.0
    ).float().cuda()

    # Split into Consecutive Current and Previous Frame Pairs.
    cur_frame  = all_frames[1:]
    prev_frame = all_frames[:-1]

    # Build the Fixed 32x32 Point Grid. Trimmed by 2 on Each Side to Match
    # the Model's Border-Trimmed Output Spatial Size.
    prev_points = np.zeros((2, 32, 32), dtype=np.float32)
    prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
    prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]
    prev_points = prev_points[..., 2:-2, 2:-2]

    # Image Centre Used to Zero-Centre Points for Transform Parameter Fitting.
    center_offset = np.array([512.0, 512.0])[:, None]

    # Run Inference Over All Pairs in the Batch.
    heatmap, offsets = model(prev_frame, cur_frame)
    heatmap = heatmap.detach().cpu().numpy()
    offsets = offsets.detach().cpu().numpy()

    res = []
    for i in range(heatmap.shape[0]):
        # Apply Predicted Offsets to Get Estimated Current Frame Point Positions.
        cur_points = prev_points + offsets[i]

        # Fit dx, dy, angle Using Squared Heatmap Weights for Robustness.
        dx, dy, angle, err = dataset_transform.offset_grid_to_transform_params(
            prev_frame_points=prev_points.reshape(2, -1) - center_offset,
            cur_frame_points=cur_points.reshape(2, -1) - center_offset,
            points_weight=heatmap[i].reshape(-1) ** 2
        )
        res.append([dx, dy, angle, err])

    return res


def predict_dataset_offsets(part: str = 'part1') -> None:
    """
    Runs the trained transformation model over every flight in the given
    dataset partition and saves the per-frame transformation parameters
    (dx, dy, angle, error) as both .pkl and .csv files.

    Output files are saved to:
        {DATA_DIR}/frame_transforms/{part}/{flight_id}.pkl
        {DATA_DIR}/frame_transforms/{part}/{flight_id}.csv

    Args:
        part (str): Dataset partition name to process (e.g. 'part1', 'part2',
                    'part3'). Must match a subdirectory under DATA_DIR.

    Returns:
        None
    """
    # Load the Model from Config.
    experiment_name = config.TRANSFORM_MODEL
    fold            = config.TRANSFORM_MODEL_FOLD
    cfg             = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"../output/checkpoints/{experiment_name}/{fold}"
    print("\n", experiment_name, "\n")

    # Disable Gradient Computation for Full Inference Pass.
    torch.set_grad_enabled(False)

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    # Load the Designated Export Epoch Checkpoint.
    ckpt_path = f"{checkpoints_dir}/{config.TRANSFORM_MODEL_EPOCH:03}.pt"
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load the Flight Index.
    df         = pd.read_csv(f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv')
    flight_ids = list(sorted(df['flight_id'].unique()))

    # Set Up Async Image Loading Pipeline.
    load_img_stage   = mpipe.OrderedStage(load_img, 1)
    decode_img_stage = mpipe.OrderedStage(decode_img, 16)
    load_img_stage.link(decode_img_stage)
    pipe = mpipe.Pipeline(load_img_stage)

    # Fixed Crop Dimensions Matching the Model's Training Input Size.
    crop_w, crop_h = 1024, 1024
    batch_size     = 4

    # Run the Per-Flight Prediction Loop.
    for flight_id in tqdm(flight_ids):
        # Build Ordered List of Frame Paths for this Flight.
        df_flight = df[df.flight_id == flight_id][
            ['img_name', 'frame']
        ].drop_duplicates().reset_index(drop=True)

        file_names = [
            f'{config.DATA_DIR}/{part}/Images/{flight_id}/'
            f'{fn[:-4]}.{config.IMG_FORMAT}'
            for fn in df_flight.img_name
        ]

        # Fallback Frame for Load Failures (Black Frame).
        last_img    = np.zeros((1024, 1024), dtype=np.uint8)
        load_failed = []

        # First Frame Has No Previous Frame; Pad with a Zero Transform.
        frame_transforms = [[0, 0, 0, 0]]

        # Buffer Holding the Current Batch of Crops.
        img_crops = []

        for fn, img in tqdm(limited_pipe(pipe, file_names, 32), total=len(file_names)):
            # Track Whether This Frame Loaded Successfully.
            load_failed.append(img is None)

            # Substitute the Last Valid Frame on Load Failure to Avoid Crashes.
            if img is None:
                img = last_img
            last_img = img

            # Extract the Centred 1024x1024 Crop.
            h, w = img.shape
            y0   = (h - crop_h) // 2
            x0   = (w - crop_w) // 2
            img_crops.append(img[y0:y0 + crop_h, x0:x0 + crop_w])

            # Process the Batch Once Full.
            if len(img_crops) >= batch_size:
                frame_transforms += find_transforms(model, img_crops)
                # Carry the Last Frame Forward as Context for the Next Batch.
                img_crops = [img_crops[-1]]

        # Process Any Remaining Frames After the Main Loop.
        if len(img_crops) > 1:
            frame_transforms += find_transforms(model, img_crops)

        del img_crops

        # Save the Results.
        dst_dir = f'{config.DATA_DIR}/frame_transforms/{part}'
        os.makedirs(dst_dir, exist_ok=True)

        frame_transforms         = np.array(frame_transforms)
        df_flight['load_failed'] = load_failed
        df_flight['dx']          = frame_transforms[:, 0]
        df_flight['dy']          = frame_transforms[:, 1]
        df_flight['angle']       = frame_transforms[:, 2]
        df_flight['error']       = frame_transforms[:, 3]

        df_flight.to_pickle(f'{dst_dir}/{flight_id}.pkl')
        df_flight.to_csv(f'{dst_dir}/{flight_id}.csv', index=False)

    # Signal the Pipeline to Shut Down Cleanly.
    pipe.put(None)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                            STANDALONE ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate the ResNet34-TSM transformation model."
    )
    parser.add_argument("action",     type=str, default="check",
                        choices=["train", "check", "export_model",
                                 "predict_dataset_offsets"],
                        help="Action to perform.")
    parser.add_argument("experiment", type=str, default="",
                        help="Experiment name (YAML file in experiments/).")
    parser.add_argument("--fold",     type=int, default=0,
                        help="Dataset fold index (default: 0).")
    parser.add_argument("--epoch",    type=int, default=-1,
                        help="Epoch to resume from, check, or export "
                             "(default: -1 = start fresh).")
    parser.add_argument("--sample",   type=int, default=-1,
                        help="Reserved for future use.")
    parser.add_argument("--part",     type=str, default='',
                        help="Dataset partition for offset prediction "
                             "(e.g. 'part1').")
    args = parser.parse_args()

    # Normalise Experiment Name. Strip 'experiments/' Prefix and '.yaml' Suffix.
    experiment_name = common_utils.normalize_experiment_name(args.experiment)

    if args.action == "train":
        train(
            experiment_name=experiment_name,
            continue_epoch=args.epoch,
            fold=args.fold
        )

    if args.action == "check":
        check(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if args.action == "export_model":
        export_model(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if args.action == "predict_dataset_offsets":
        predict_dataset_offsets(args.part)
# ------------------------------------------------------------------------------
