# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: train.py %
# Date Created:            %date_created: Tue May 05 16:21:12 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# This script trains the Winning Solution segmentation model for the Vision-Based
# Sense and Avoid system. It supports configurable backbones via YAML experiments,
# dataset filtering using flight ID splits, false positive mining, checkpointing,
# and optional experiment tracking with TensorBoard and Weights & Biases.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from Project Root:
#       - python -m src.baselines.winning_solution.seg_tracker.train train <arguments>
#
#   2. Optional Arguments:
#       - --fold             : Cross-validation fold index (default: 0).
#       - --epoch            : Resume training from epoch (-1 = fresh run).
#       - --seed             : Random seed for reproducibility (default: 42).
#       - --train-flights    : Path to train.txt flight list.
#       - --val-flights      : Path to val.txt flight list.
#       - --output-dir       : Base output directory for runs.
#       - --wandb            : Enable Weights & Biases logging.
#       - --wandb-project    : W&B project name.
#       - --run-dir          : Run directory (export mode).
#       - --checkpoint       : Checkpoint name for export (default: best.pt).
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

# ------------------------------------------------------------------------------
# STANDARD LIBRARY IMPORTS
# ------------------------------------------------------------------------------
import argparse
import copy
import os
import random

from collections import defaultdict
from typing import Iterator, List

# ------------------------------------------------------------------------------
# THIRD-PARTY IMPORTS
# ------------------------------------------------------------------------------
import numpy as np
import madgrad
from tqdm import tqdm

# ------------------------------------------------------------------------------
# PYTORCH / DEEP LEARNING IMPORTS
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ------------------------------------------------------------------------------
# PROJECT-SPECIFIC IMPORTS
# ------------------------------------------------------------------------------\
from logging_module import LogLevel, log
import src.baselines.winning_solution.seg_tracker.focal_loss as focal_loss
import src.baselines.winning_solution.seg_tracker.common_utils as common_utils
import src.baselines.winning_solution.seg_tracker.dataset_tracking as dataset_tracking
import src.baselines.winning_solution.seg_tracker.models_segmentation as models_segmentation
import src.baselines.winning_solution.seg_tracker.seg_prediction_to_items as seg_prediction_to_items

from src.baselines.winning_solution.seg_tracker.common_utils import (
    DotDict,
    timeit_context
)

from src.baselines.winning_solution.seg_tracker.check_frame_level_prediction import (
    Box,
    DetectedItem,
    extend_bounding_boxes,
    calc_iou
)

# ------------------------------------------------------------------------------
#                           OPTIONAL W&B INITIALISATION
# ------------------------------------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               UTILITY FUNCTIONS
# ------------------------------------------------------------------------------
def seed_everything(seed: int):
    """
    Sets global random seed state for reproducibility across Python RNG, NumPy, 
    and PyTorch (CPU + CUDA).
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


def get_next_run_dir(base_dir: str) -> str:
    """
    Returns next available run directory without creating it. Directory format:
    run0, run1, run2, ...
    """
    # Initialise Run Counter.
    i = 0

    # Increment Until Free Directory is Found.
    while True:
        candidate = os.path.join(base_dir, f"run{i}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def load_flight_ids(flights_file: str) -> List[str]:
    """
    Loads flight IDs from split file (train.txt / val.txt). Returns only lines
    that start with 'part', the actual flight ID format.

    This correctly ignores all header text, dashes, comments and blank lines
    regardless of whether they start with # or not.
    """
    # Read Raw File.
    with open(flights_file) as f:
        lines = f.readlines()

    # Filter Only to Lines Begining with 'part'.
    ids = [l.strip() for l in lines if l.strip().startswith('part')]

    # Check Case where .txt File Contains NO IDs.
    if not ids:
        raise ValueError(
            f"No flight IDs found in {flights_file}. "
            f"Expected lines starting with 'part1', 'part2', or 'part3'."
        )
    
    # Return Valid IDs.
    return ids


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiates segmentation model from config dictionary. Model class is 
    dynamically resolved from models_segmentation module.
    """
    # Extract Model Configuration Block.
    model_params = cfg['model_params']

    # Dynamically Resolve Model Class.
    model: nn.Module = models_segmentation.__dict__[model_params['model_cls']](
        cfg=model_params,
        pretrained=True
    )

    # Return the Constructed Model. 
    return model
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               SAMPLING STRATEGIES
# ------------------------------------------------------------------------------
class TrackPreferredSampler(torch.utils.data.Sampler):
    """
    Sampling strategy that biases batches toward tracking-rich frames.
    """
    def __init__(self, dataset: dataset_tracking.TrackingDataset, num_samples: int, tracking_samples_share: float = 0.25):
        # Initalise Base Torch Sampler.
        super().__init__(dataset)

        # Store Variables within Class. 
        self.data_source = dataset
        self.num_samples = num_samples
        self.tracking_samples_share = tracking_samples_share

    def __iter__(self) -> Iterator[int]:
        # Fixed Block Sampling for Stability.
        block_size = 64

        # Generate Samples in Blocks.
        for _ in range(self.num_samples // block_size):
            # Split Block into Tracking Vs Random Samples.
            nb_tracking = round(block_size * self.tracking_samples_share)

            # Select Tracking-heavy Frames.
            samples_tracking = np.random.choice(
                self.data_source.frame_nums_with_match_items,
                nb_tracking
            )

            # Random Fallback Samples.
            samples_other = np.random.randint(
                0, len(self.data_source.frames),
                block_size - nb_tracking
            )

            # Combine + Shuffle.
            all_samples = np.concatenate([samples_tracking, samples_other])
            np.random.shuffle(all_samples)

            yield from all_samples

    def __len__(self) -> int:
        return self.num_samples


class TrackPreferredSamplerWithFP(torch.utils.data.Sampler):
    """
    Extended sampler that includes false-positive mining samples.
    """
    def __init__(self, dataset: dataset_tracking.TrackingDataset, num_samples: int, fp_samples, tracking_samples_share: float = 0.25, fp_samples_share: float = 0.25):
        # Initalise Base Torch Sampler.
        super().__init__(dataset)

        # Store Variables within Class. 
        self.data_source      = dataset
        self.num_samples      = num_samples
        self.tracking_share   = tracking_samples_share
        self.fp_share         = fp_samples_share
        self.fp_samples_p     = np.array([s[0] + 0.1 for s in fp_samples])
        self.fp_samples_p    /= self.fp_samples_p.sum()
        self.fp_samples_idx   = np.array([s[1] for s in fp_samples])

    def __iter__(self) -> Iterator[int]:
        # Fixed Block Size for Structured Sampling.
        block_size = 64

        # Generate Samples in Fixed-Size Chunks.
        for _ in range(self.num_samples // block_size):
            # Number of Tracking & Hard-Negative Samples Per Block.
            nb_tracking = round(block_size * self.tracking_share)
            nb_fp       = round(block_size * self.fp_share)

            # Tracking Positive Samples.
            samples_tracking = np.random.choice(
                self.data_source.frame_nums_with_match_items, nb_tracking)
            
            # Tracking False Positive Samples.
            samples_fp = np.random.choice(
                self.fp_samples_idx, nb_fp, p=self.fp_samples_p)
            
            # Random Background Samples.
            samples_other = np.random.randint(
                0, len(self.data_source.frames), block_size - nb_tracking - nb_fp)
            
            # Merge All Sample Types into Single Block.
            all_samples = np.concatenate([samples_tracking, samples_fp, samples_other])

            # Shuffle to Remove Structural Bias Within Block.
            np.random.shuffle(all_samples)

            # Yeild Individual Indicies to Dataloader.
            yield from all_samples

    def __len__(self) -> int:
        # Total Number of Samples Returned Per Epoch.
        return self.num_samples


class SkipNSampler(torch.utils.data.Sampler):
    """
    Deterministic subsampling used for FP mining (every Nth frame).
    """
    def __init__(self, dataset: dataset_tracking.TrackingDataset, skip_step: int):
        # Initalise Base Torch Sampler.
        super().__init__(dataset)

        # Random Offset so we Don't Always Start Sampling from Index 0.
        self.samples = list(range(np.random.randint(skip_step), len(dataset), skip_step))

    def __iter__(self) -> Iterator[int]:
        # Yield Precomputed Indices Sequentially.
        yield from self.samples

    def __len__(self) -> int:
        # Total Number of Samples Returned Per Epoch.
        return len(self.samples)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           IMAGE COMBINING
# ------------------------------------------------------------------------------
def combine_images(data: dict, input_frames: int) -> torch.Tensor:
    """
    Constructs a multi-frame input tensor by stacking the current frame
    together with its preceding aligned frames.
    """
    # Current Frame (t) & Most Recent Previous Aligned Frame (t-1).
    image = data['image'].float().cuda()
    image_prev0 = data['prev_image_aligned0'].float().cuda()

    # Initialise Frame Stack with Most Recent Temporal Context.
    images = [image_prev0, image]

    # Append Remaining Historical Frames (t-2, t-3, ...).
    for img_idx in range(1, input_frames - 1):
        images.append(data[f'prev_image_aligned{img_idx}'].float().cuda())

    # Stack Along Channel/Frame Dimension -> (B, T, C, H, W).
    return torch.stack(images, dim=1)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       FALSE POSITIVE MINING
# ------------------------------------------------------------------------------
def find_fp_samples(model: nn.Module,
                    dataset: dataset_tracking.TrackingDataset,
                    step: int,
                    batch_size: int = 8,
                    input_frames: int = 2) -> list:
    """
    Runs the model on every nth frame, collects false positive detections.
    Returns list of (max_conf, frame_idx, detections) tuples for use in
    TrackPreferredSamplerWithFP.
    """
    # Sample Every Nth Frame (with random offset).
    sampler     = SkipNSampler(dataset, step)

    # DataLoader Used Only for Inference (No Shuffling, Deterministic Order).
    data_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                             batch_size=batch_size, sampler=sampler)
    
    # Set Model to Evaluation Mode.
    model.eval()
    res = []

    # Inference Loop Over Sampled Frames.
    for data in tqdm(data_loader, desc="FP Mining"):
        with torch.no_grad():
            # Build Temporal Input (Current + Previous Frames).
            images = combine_images(data, input_frames)

            # Mixed Precision Inference for Speed.
            with torch.cuda.amp.autocast():
                pred = model(images)
            
            # Convert Logits to Probabilities for Mask Head.
            pred['mask'] = torch.sigmoid(pred['mask'])

            # Move All Outputs to CPU for Post-Processing.
            pred = {k: pred[k].cpu().detach().numpy() for k in pred.keys()}

            # Process Each Item in the Batch Independently.
            for i in range(images.shape[0]):
                # Set x & y offsets.
                x_offset = float(data['crop_x'][i])
                y_offset = float(data['crop_y'][i])

                # Predicted Segmentation Heatmap for Current Sample.
                comb_pred = pred['mask'][i, 0]

                # Convert Model Outputs into Structured Detections.
                detected = seg_prediction_to_items.pred_to_items(
                    comb_pred=comb_pred, offset=pred['offset'][i],
                    size=pred['size'][i], tracking=pred['tracking'][i],
                    distance=pred['distance'][i, 0],
                    above_horizon=pred['above_horizon'][i, 0],
                    conf_threshold=0.12, pred_scale=8.0,
                    x_offset=x_offset, y_offset=y_offset
                )

                # Ground Truth Retrieval for IoU Comparison.
                item_idx = int(data['idx'][i])
                gt_items = [extend_bounding_boxes(copy.copy(it))
                            for it in dataset.frames[item_idx].items]

                # Convert Predictions into Same Format as GT for IoU Matching.
                pred_items = [
                    extend_bounding_boxes(DetectedItem(
                        item_id='airborne', cx=it['cx'], cy=it['cy'],
                        w=it['w'], h=it['h'], confidence=it['conf'],
                        distance=it['distance'], dx=0, dy=0
                    ))
                    for it in detected
                ]

                # Keep Predictions that DO NOT Overlap with Any GT (IoU Threshold).
                fp = [
                    (p.confidence, p.cx, p.cy)
                    for p in pred_items
                    if not any(calc_iou(p, g) > 0.1 for g in gt_items)
                ]

                # Store Highest-Confidence FP Per Frame.
                if fp:
                    res.append((max(f[0] for f in fp), item_idx, tuple(fp)))
    
    # Set Model to Train Mode & Return Results.
    model.train()
    return res
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           CHECKPOINT HELPERS
# ------------------------------------------------------------------------------
def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optimizer, scheduler, extra: dict = None):
    """
    Saves full training state to disk.

    Includes:
        - model weights
        - optimizer state
        - scheduler state
        - current epoch
        - optional extra metadata (losses, metrics, etc.)
    """
    # Core Training State Required for Resume.
    payload = {
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    # Optional Metadata (e.g. train_loss, best_epoch, custom logs).
    if extra:
        payload.update(extra)
    
    # Persist Checkpoint to Disk.
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer=None, scheduler=None) -> int:
    """
    Loads checkpoint and restores training state. Only model weights are 
    strictly required. Optimiser and scheduler states are restored if provided.
    """
    # Load Checkpoint to GPU (Assumes Training on CUDA).
    checkpoint = torch.load(path, map_location='cuda')

    # Restore Model Weights.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore Optimiser State if Available.
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Restore Scheduler State if Available.
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Return Last Completed Epoch (or -1 if Not Present).
    return checkpoint.get("epoch", -1)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               MAIN TRAIN FUNCTION
# ------------------------------------------------------------------------------
def train(experiment_name: str,
          fold: int,
          continue_epoch: int      = -1,
          train_flights_file: str  = None,
          val_flights_file: str    = None,
          output_base_dir: str     = "outputs/model_training/winning_solution",
          use_wandb: bool          = False,
          wandb_project: str       = "airborne-detection"):
    
    # Load Configuration & Split into Model/Train Params.
    cfg = common_utils.load_config_data(experiment_name)
    model_params = DotDict(cfg["model_params"])
    train_params = DotDict(cfg["train_params"])
 
    # Setup Output Directories for this Run.
    run_dir = get_next_run_dir(output_base_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tensorboard")
    fp_cache_dir = os.path.join(run_dir, "fp_samples")

    # Make Sure Directories Exist.
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(fp_cache_dir, exist_ok=True)
    
    # Print Summary Statisitcs.
    print(f"\n{'='*60}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Run dir    : {run_dir}")
    print(f"  Fold       : {fold}")
    print(f"  Resume     : epoch {continue_epoch if continue_epoch > -1 else 'N/A'}")
    print(f"{'='*60}\n")
 
    # Optional Weights & Biases Logging.
    wb_run = None
    if use_wandb and WANDB_AVAILABLE:
        # Check for Entity Variables.
        wb_entity = os.environ.get("WANDB_ENTITY", None)

        # Initialise Weights & Biases Run Profile.
        wb_run = wandb.init(
            project = wandb_project,
            entity = wb_entity,
            name = f"{experiment_name.split('/')[-1]}_fold{fold}",
            config = {**cfg, "run_dir": run_dir,
                        "train_flights": train_flights_file,
                        "val_flights":   val_flights_file},
            resume = "never",
        )

        # Print Summary Message. 
        print()
        log(LogLevel.INFO, "train:train", "Weights & Biases Run: %s.\n", wb_run.url)
    
    elif use_wandb and not WANDB_AVAILABLE:
        # Print Warning if Weights & Biases is Not Available.
        print()
        log(LogLevel.WARNING, "train:train", 
            "--wandb requested but wandb is not installed. Falling back to " \
            "TensorBoard only. Run: pip install wandb")
    
    # Update the Logger. 
    logger = SummaryWriter(log_dir=tb_dir)
 
    # Load Optional Flight Splits.
    train_flight_ids = load_flight_ids(train_flights_file) \
        if train_flights_file else None
    val_flight_ids   = load_flight_ids(val_flights_file) \
        if val_flights_file else None
    
    # Print Success Messages when Flights Loads. 
    if train_flight_ids:
        log(LogLevel.INFO, "train:train", 
            "Training on %d flights from %s\n", len(train_flight_ids), train_flights_file)
    if val_flight_ids:
        log(LogLevel.INFO, "train:train", 
            "Validating on %d flights from %s\n", len(val_flight_ids), val_flights_file)
 
    # Training Dataset (Augmented, Cropped).
    dataset_train = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_TRAIN,
        cfg_data=cfg,
        return_torch_tensors=True,
        small_subset=False,
        flight_ids=train_flight_ids,
    )
 
    # Full-Resolution Dataset for FP Mining (No Crop Augmentation).
    cfg_full = copy.deepcopy(cfg)
    cfg_full['dataset_params']['crop_with_plane_ratio'] = 0
    cfg_full['dataset_params']['crop_size'] = (2432, 2048)
    
    # Make a Dataset Object for Training. 
    dataset_full = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_TRAIN,
        cfg_data=cfg_full,
        return_torch_tensors=True,
        small_subset=False,
        flight_ids=train_flight_ids,
    )

    # Prep Dataset for Validation.
    dataset_full.is_training = False
 
    # Validation Dataset (No Augmentation, Deterministic Crops).
    dataset_val = None
    loader_val  = None

    if val_flight_ids:
        # Set-Up the Base Config & Croppig Sizes.
        cfg_val = copy.deepcopy(cfg)
        cfg_val['dataset_params']['crop_with_plane_ratio'] = 0
        cfg_val['dataset_params']['crop_size'] = (512, 512)

        # Create Validation Dataset.
        dataset_val = dataset_tracking.TrackingDataset(
            stage=dataset_tracking.BaseDataset.STAGE_VALID,
            cfg_data=cfg_val,
            return_torch_tensors=True,
            small_subset=False,
            flight_ids=val_flight_ids,
        )

        # Disable Training Behaviour (No Augmentation, Deterministic Outputs).
        dataset_val.is_training = False

        # DataLoader for Validation.
        loader_val = DataLoader(
            dataset_val,
            num_workers=cfg['val_data_loader']['num_workers'],
            shuffle=False,
            batch_size=cfg['val_data_loader']['batch_size'],
            sampler=torch.utils.data.RandomSampler(
                dataset_val,
                replacement=True,
                num_samples=cfg['train_params'].get('nb_samples_valid', 1024),
            ),
        )

        # Print Dataset Summary for Sanity Checking
        log(LogLevel.INFO, "train:train", 
            "Validation Dataset: %d frames, %d with tracked objects", 
            len(dataset_val.frames), len(dataset_val.frame_nums_with_match_items))
 
    # Training Hyperparameters.
    batch_size              = cfg['train_data_loader']['batch_size']
    nb_samples_per_epoch    = train_params.nb_samples_per_epoch
    mask_loss_name          = cfg['train_params'].get('mask_loss', 'bce')
    tracking_samples_share  = cfg['train_params'].get('tracking_samples_share', 0.25)
    fp_samples_share        = cfg['train_params'].get('fp_samples_share', 0.25)
    fpm_batch_size          = cfg['train_params'].get('fpm_batch_size', 4)
    fpm_epochs              = cfg['train_params'].get('fpm_epochs', {})
    nb_epochs               = train_params.nb_epochs
    grad_clip_value         = train_params.get("grad_clip", 8.0)
    freeze_backbone_steps   = train_params.get("freeze_backbone_steps", 0)
 
    # Loss Functions.
    cr_mse  = nn.MSELoss(reduction='none')
    cr_mae  = nn.L1Loss(reduction='none')
    cr_bce  = nn.BCEWithLogitsLoss(reduction='none')
    cr_fl   = focal_loss.FocalLossV2(reduction='none')
    mask_loss = cr_fl if mask_loss_name == 'fl' else cr_bce
 
    if 'loss_scale' in train_params:
        loss_items_scale = train_params['loss_scale']
    else:
        loss_items_scale = {
            'cls': 2000, 'size': 1, 'offset': 0.2,
            'distance': 1, 'tracking': 1, 'above_horizon': 1
        }
 
    # Model Set-Up.
    model = build_model(cfg)
    model = model.cuda()
    model.train()

    # Print Dataset Summary for Sanity Checking.
    print(f"Model: {model.__class__.__name__}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
 
    # Set-Up Optimiser.
    initial_lr = float(train_params.initial_lr)
    optimizer_name = train_params.optimizer
    if optimizer_name == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif optimizer_name == "madgrad":
        optimizer = madgrad.MADGRAD(model.parameters(), lr=initial_lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError(f"Invalid optimiser: {optimizer_name}")
 
    # Set-Up the Learning Rate Scheduler.
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
        for i in range(continue_epoch + 1):
            scheduler.step()
    else:
        raise RuntimeError(f"Invalid scheduler: {train_params.scheduler}")
 
    scaler = torch.cuda.amp.GradScaler()
 
    # Resume the Previous Run.
    if continue_epoch > -1:
        resume_path = os.path.join(ckpt_dir, f"{continue_epoch:04d}.pt")
        if not os.path.exists(resume_path):
            # Also check old naming convention
            resume_path = os.path.join(ckpt_dir, f"{continue_epoch:03d}.pt")
        loaded_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"Resumed from epoch {loaded_epoch}: {resume_path}")
 
    # Initial Dataloader (Before FP Mining).
    data_loader = DataLoader(
        dataset_train,
        num_workers=cfg['train_data_loader']['num_workers'],
        shuffle=False,
        batch_size=batch_size,
        sampler=TrackPreferredSampler(
            dataset_train,
            num_samples=nb_samples_per_epoch,
            tracking_samples_share=tracking_samples_share,
        ),
    )
 
    # Epochs at which to Save Numbered Checkpoints.
    checkpoints_to_save = {250, 370, 535, 770, 1100, 1535, 2220}
 
    # Training State Tracking.
    best_val_loss  = float('inf')
    best_val_epoch = -1
    
    print()
    log(LogLevel.INFO, "train:train", 
    "Starting training from epoch %d to %d. \nBackbone Freeze Steps: %d." \
    " \nGrad Clip: %d", (continue_epoch + 1), nb_epochs, freeze_backbone_steps, grad_clip_value)
 
    # Epoch Loop.
    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        # Set Model to Train Mode.
        model.train()

        # Backbone Warmup Freeze
        if freeze_backbone_steps > 0:
            if epoch_num == 0:
                model.freeze_encoder()
                log(LogLevel.INFO, "train:train", "Backbone Frozen for Warmup.\n")
            if epoch_num == freeze_backbone_steps:
                model.unfreeze_encoder()
                log(LogLevel.INFO, "train:train", "Backbone Unfrozen.\n")

        # Calculate the Loss.
        epoch_loss = common_utils.AverageMeter()
        epoch_loss_separate = defaultdict(common_utils.AverageMeter)

        # Set the Optimiser.
        optimizer.zero_grad()
        data_iter = tqdm(data_loader, desc=f"Epoch {epoch_num}")

        # Batch Loop for the Epoch.
        for data in data_iter:
            with torch.set_grad_enabled(True):
                images = combine_images(data, model_params.input_frames)

                label_cls              = data['cls'].float().cuda()
                label_cls_planned      = data['cls_planned'].float().cuda()
                reg_mask               = data['reg_mask'].float().cuda()
                reg_offset             = data['reg_offset'].float().cuda()
                reg_size               = data['reg_size'].float().cuda()
                reg_tracking           = data['reg_tracking'].float().cuda()
                reg_distance           = data['reg_distance'].float().cuda()
                reg_above_horizon      = data['reg_above_horizon'].float().cuda()
                reg_tracking_mask      = data['reg_tracking_mask'].float().cuda()
                reg_distance_mask      = data['reg_distance_mask'].float().cuda()
                reg_offset_mask        = data['reg_offset_mask'].float().cuda()
                reg_above_horizon_mask = data['reg_above_horizon_mask'].float().cuda()
 
                optimizer.zero_grad()
 
                with torch.cuda.amp.autocast():
                    pred = model(images)
 
                    # Heatmap Loss: Pixel-wise Min of All-Airborne Vs Planned-Only.
                    loss_cls_all     = mask_loss(pred['mask'], 1.0 - label_cls[:, :1])
                    loss_cls_planned = mask_loss(pred['mask'], 1.0 - label_cls_planned[:, :1])
                    loss_cls = (torch.minimum(loss_cls_all, loss_cls_planned).mean()
                                if epoch_num > 75 else loss_cls_all.mean())
 
                    # Regression Losses.
                    reg_mask_4d = reg_mask[:, None, :, :]
                    mask_size   = reg_mask_4d.sum() + 0.1
                    loss_size   = cr_mse(
                        pred['size'] * reg_mask_4d,
                        reg_size * reg_mask_4d).sum() / mask_size
 
                    reg_offset_mask_4d = reg_offset_mask[:, None, :, :]
                    loss_offset = cr_mse(
                        pred['offset'] * reg_offset_mask_4d,
                        reg_offset * reg_offset_mask_4d
                    ).sum() / (reg_offset_mask_4d.sum() + 0.1)
 
                    loss_distance = cr_mse(
                        pred['distance'][:, 0] * reg_distance_mask,
                        reg_distance * reg_distance_mask
                    ).sum() / (reg_distance_mask.sum() + 0.1)
 
                    reg_tracking_mask_4d = reg_tracking_mask[:, None, :, :]
                    loss_tracking = cr_mae(
                        pred['tracking'] * reg_tracking_mask_4d,
                        reg_tracking * reg_tracking_mask_4d
                    ).sum() / (reg_tracking_mask.sum() + 0.1)
 
                    losses = [
                        ('cls',      loss_cls),
                        ('size',     loss_size),
                        ('offset',   loss_offset),
                        ('distance', loss_distance),
                        ('tracking', loss_tracking),
                    ]
 
                    if model_params.get('output_above_horizon', False):
                        ah_mask_size = reg_above_horizon_mask.sum()
                        if ah_mask_size > 0:
                            loss_horizon = cr_bce(
                                pred['above_horizon'][reg_above_horizon_mask > 0],
                                reg_above_horizon[reg_above_horizon_mask > 0]
                            ).mean()
                        else:
                            loss_horizon = torch.zeros(1).float().cuda().squeeze()
                        losses.append(('above_horizon', loss_horizon))
 
                    loss = sum(loss_items_scale[n] * v for n, v in losses)
 
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_value > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                scaler.step(optimizer)
                scaler.update()
 
                batch_size_actual = images.shape[0]
                epoch_loss.update(loss.detach().item(), batch_size_actual)
                for loss_name, loss_val in losses:
                    epoch_loss_separate[loss_name].update(
                        loss_val.detach().item(), batch_size_actual)
 
                data_iter.set_description(
                    f"E{epoch_num} loss={epoch_loss.avg:.4f} "
                    + " ".join(
                        f"{n}={v.avg * loss_items_scale[n]:.3f}"
                        for n, v in epoch_loss_separate.items()
                    )
                )
                del loss, losses
 
        scheduler.step()
 
        # Validation Loop.
        val_loss_avg = None
        if loader_val is not None:
            model.eval()
            val_loss_meter = common_utils.AverageMeter()
            val_loss_sep   = defaultdict(common_utils.AverageMeter)
 
            with torch.no_grad():
                for data in tqdm(loader_val, desc=f"  Val E{epoch_num}", leave=False):
                    images = combine_images(data, model_params.input_frames)
 
                    label_cls              = data['cls'].float().cuda()
                    label_cls_planned      = data['cls_planned'].float().cuda()
                    reg_mask               = data['reg_mask'].float().cuda()
                    reg_offset             = data['reg_offset'].float().cuda()
                    reg_size               = data['reg_size'].float().cuda()
                    reg_tracking           = data['reg_tracking'].float().cuda()
                    reg_distance           = data['reg_distance'].float().cuda()
                    reg_above_horizon      = data['reg_above_horizon'].float().cuda()
                    reg_tracking_mask      = data['reg_tracking_mask'].float().cuda()
                    reg_distance_mask      = data['reg_distance_mask'].float().cuda()
                    reg_offset_mask        = data['reg_offset_mask'].float().cuda()
                    reg_above_horizon_mask = data['reg_above_horizon_mask'].float().cuda()
 
                    with torch.cuda.amp.autocast():
                        pred = model(images)
 
                        loss_cls_all     = mask_loss(pred['mask'], 1.0 - label_cls[:, :1])
                        loss_cls_planned = mask_loss(pred['mask'], 1.0 - label_cls_planned[:, :1])
                        loss_cls = (torch.minimum(loss_cls_all, loss_cls_planned).mean()
                                    if epoch_num > 75 else loss_cls_all.mean())
 
                        reg_mask_4d = reg_mask[:, None, :, :]
                        mask_size   = reg_mask_4d.sum() + 0.1
                        loss_size   = cr_mse(pred['size'] * reg_mask_4d,
                                             reg_size * reg_mask_4d).sum() / mask_size
 
                        reg_offset_mask_4d = reg_offset_mask[:, None, :, :]
                        loss_offset = cr_mse(pred['offset'] * reg_offset_mask_4d,
                                             reg_offset * reg_offset_mask_4d
                                             ).sum() / (reg_offset_mask_4d.sum() + 0.1)
 
                        loss_distance = cr_mse(pred['distance'][:, 0] * reg_distance_mask,
                                               reg_distance * reg_distance_mask
                                               ).sum() / (reg_distance_mask.sum() + 0.1)
 
                        reg_tracking_mask_4d = reg_tracking_mask[:, None, :, :]
                        loss_tracking = cr_mae(pred['tracking'] * reg_tracking_mask_4d,
                                               reg_tracking * reg_tracking_mask_4d
                                               ).sum() / (reg_tracking_mask.sum() + 0.1)
 
                        val_losses = [
                            ('cls',      loss_cls),
                            ('size',     loss_size),
                            ('offset',   loss_offset),
                            ('distance', loss_distance),
                            ('tracking', loss_tracking),
                        ]
 
                        if model_params.get('output_above_horizon', False):
                            ah_mask_size = reg_above_horizon_mask.sum()
                            if ah_mask_size > 0:
                                loss_horizon = cr_bce(
                                    pred['above_horizon'][reg_above_horizon_mask > 0],
                                    reg_above_horizon[reg_above_horizon_mask > 0]
                                ).mean()
                            else:
                                loss_horizon = torch.zeros(1).float().cuda().squeeze()
                            val_losses.append(('above_horizon', loss_horizon))
 
                        val_loss = sum(loss_items_scale[n] * v for n, v in val_losses)
 
                    bs = images.shape[0]
                    val_loss_meter.update(val_loss.item(), bs)
                    for loss_name, loss_val in val_losses:
                        val_loss_sep[loss_name].update(loss_val.item(), bs)
 
            val_loss_avg = val_loss_meter.avg
            model.train()
            print(f"  Val loss: {val_loss_avg:.4f}  "
                  + " ".join(f"{n}={v.avg * loss_items_scale[n]:.3f}"
                             for n, v in val_loss_sep.items()))
 
        # Update Logger.
        log_dict = {
            "train/loss_total": epoch_loss.avg,
            "train/lr"        : optimizer.param_groups[0]["lr"],
        }
        for name, meter in epoch_loss_separate.items():
            log_dict[f"train/loss_{name}"] = meter.avg
 
        if val_loss_avg is not None:
            log_dict["val/loss_total"] = val_loss_avg
            for name, meter in val_loss_sep.items():
                log_dict[f"val/loss_{name}"] = meter.avg
 
        logger.add_scalar("loss_train",  epoch_loss.avg,                    epoch_num)
        logger.add_scalar("lr",          optimizer.param_groups[0]["lr"],   epoch_num)
        for name, meter in epoch_loss_separate.items():
            logger.add_scalar(f"loss_{name}_train", meter.avg, epoch_num)
        if val_loss_avg is not None:
            logger.add_scalar("loss_val", val_loss_avg, epoch_num)
            for name, meter in val_loss_sep.items():
                logger.add_scalar(f"loss_{name}_val", meter.avg, epoch_num)
        logger.flush()
 
        if wb_run is not None:
            wb_run.log(log_dict, step=epoch_num)
 
        # Save Numbered Checkpoint.
        if epoch_num in checkpoints_to_save:
            ckpt_path = os.path.join(ckpt_dir, f"{epoch_num:04d}.pt")
            save_checkpoint(ckpt_path, epoch_num, model, optimizer, scheduler,
                            extra={"train_loss": epoch_loss.avg})
            print()
            log(LogLevel.INFO, "train:train", "Saved Checkpoint: %s\n", ckpt_path)
 
        # Save Last Weights (Overwrites Every Epoch).
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(last_path, epoch_num, model, optimizer, scheduler,
                        extra={"train_loss": epoch_loss.avg})
 
        # Save Best Weights Based on Validation Loss.
        monitor_loss  = val_loss_avg if val_loss_avg is not None else epoch_loss.avg
        monitor_label = "val" if val_loss_avg is not None else "train"
 
        if monitor_loss < best_val_loss:
            best_val_loss  = monitor_loss
            best_val_epoch = epoch_num
            best_path      = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, epoch_num, model, optimizer, scheduler,
                            extra={f"{monitor_label}_loss": best_val_loss,
                                   "best_epoch": best_val_epoch})
            log(LogLevel.INFO, "train:train", "New Best %s_loss %s at epoch %d. " / 
                "Saved to best.pt\n", monitor_label, best_val_loss, best_val_epoch)

            if wb_run is not None:
                wb_run.summary[f"best_{monitor_label}_loss"] = best_val_loss
                wb_run.summary["best_epoch"]                 = best_val_epoch
 
        # False Positive Mining.
        if epoch_num in fpm_epochs:
            fp_path = os.path.join(fp_cache_dir, f"fp_samples_{epoch_num:04d}.pt")
            if os.path.exists(fp_path):
                fp_samples = torch.load(fp_path)
                print(f"  Loaded Cached FP Samples: {fp_path}")
            else:
                print(f"\n  Running FP Mining at Epoch {epoch_num}...")
                fp_samples = find_fp_samples(
                    model, dataset=dataset_full,
                    step=fpm_epochs[epoch_num],
                    batch_size=fpm_batch_size,
                    input_frames=model_params.input_frames
                )
                torch.save(fp_samples, fp_path)
 
            print(f"  FP Samples Found: {len(fp_samples)}")
 
            if wb_run is not None:
                wb_run.log({"fp_mining/n_samples": len(fp_samples)}, step=epoch_num)
 
            dataset_train.set_fpm_samples(fp_samples)
            data_loader = DataLoader(
                dataset_train,
                num_workers=cfg['train_data_loader']['num_workers'],
                shuffle=False,
                batch_size=batch_size,
                sampler=TrackPreferredSamplerWithFP(
                    dataset_train,
                    num_samples=nb_samples_per_epoch,
                    fp_samples=fp_samples,
                    tracking_samples_share=tracking_samples_share,
                    fp_samples_share=fp_samples_share,
                )
            )
 
    # Done. 
    logger.close()
    if wb_run is not None:
        wb_run.finish()
 
    print(f"\nTraining complete.")
    print(f"  Best epoch : {best_val_epoch}  (val_loss={best_val_loss:.5f})  -> {ckpt_dir}/best.pt")
    print(f"  Last epoch : {nb_epochs}                                       -> {ckpt_dir}/last.pt")
    print(f"  Run dir    : {run_dir}")
# ------------------------------------------------------------------------------

 
# ------------------------------------------------------------------------------
#                               EXPORT MODEL
# ------------------------------------------------------------------------------
def export_model(experiment_name: str, run_dir: str, checkpoint_name: str = "best.pt"):
    """
    Exports a trained model checkpoint into a minimal inference format. Only the 
    model weights are saved (no optimiser/scheduler state).
    """
    # Load Model Configuration Used During Training.
    cfg = common_utils.load_config_data(experiment_name)

    # Resolve Checkpoint Path.
    ckpt_path = os.path.join(run_dir, "checkpoints", checkpoint_name)

    # Rebuild Model Architecture.
    model = build_model(cfg)

    # Load Trained Weights.
    model = model.cuda()
    
    # Switch to Inference Mode.
    epoch = load_checkpoint(ckpt_path, model)
    model.eval()
    
    # Create Export Directory if Needed.
    dst_dir = os.path.join(run_dir, "exported_models")
    os.makedirs(dst_dir, exist_ok=True)

    # Final Export Path (Keeps Experiment + Checkpoint Identity).
    out_path = os.path.join(dst_dir, f"{experiment_name}_{checkpoint_name}")
    
    # Save Only Model Weights (Clean Inference Artifact).
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print()
    log(LogLevel.INFO, "train:export_model", "Exported: %s (epoch %d).\n", out_path, epoch)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Improve GPU Performance (Non-Deterministic But Faster).
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

    # Argument Parsing.
    parser = argparse.ArgumentParser(
        description="Train winning solution segmentation model.")

    # Core Mode Selection.
    parser.add_argument("action", type=str,
                        choices=["train", "export"],
                        help="Action to perform.")
    parser.add_argument("experiment", type=str,
                        help="Experiment name (YAML file in experiments/).")
    
    # Training Options.
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1,
                        help="Resume from this epoch (-1 = start fresh).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Must match split_dataset.py seed.")

    # Dataset Configuration.
    parser.add_argument("--train-flights", type=str, default=None,
                        help="Path to train.txt from split_dataset.py.")
    parser.add_argument("--val-flights",   type=str, default=None,
                        help="Path to val.txt from split_dataset.py.")

    # Output Configuration.
    parser.add_argument("--output-dir",  type=str,
                        default="outputs/model_training/winning_solution",
                        help="Base output directory. A run{N} subdir is created.")

    # Logging (Weights & Biases).
    parser.add_argument("--wandb",         action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str,
                        default="Vision Based Sense & Avoid",
                        help="W&B project name.")

    # Export Options.
    parser.add_argument("--run-dir",    type=str, default=None,
                        help="Run directory for export action.")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint filename for export action.")

    # Parse CLI Arguments.
    args = parser.parse_args()

    # Set Reproducibility Seed.
    seed_everything(args.seed)

    # Normalise Experiment Name (Ensures Consistent Config Resolution).
    experiment_name = common_utils.normalize_experiment_name(args.experiment)

    # Run Train Mode.
    if args.action == "train":
        train(
            experiment_name    = experiment_name,
            fold               = args.fold,
            continue_epoch     = args.epoch,
            train_flights_file = args.train_flights,
            val_flights_file   = args.val_flights,
            output_base_dir    = args.output_dir,
            use_wandb          = args.wandb,
            wandb_project      = args.wandb_project,
        )

    # Run Export Mode.
    elif args.action == "export":
        if not args.run_dir:
            parser.error("--run-dir is required for the export action.")
        export_model(
            experiment_name = experiment_name,
            run_dir         = args.run_dir,
            checkpoint_name = args.checkpoint,
        )
# ------------------------------------------------------------------------------
