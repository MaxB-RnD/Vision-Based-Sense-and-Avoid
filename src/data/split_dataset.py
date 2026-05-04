# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: split_dataset.py %
# Date Created:            %date_created: Mon May 04 20:06:21 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# This script reads a dataset file (e.g. hard_dataset_flights.txt) and randomly
# splits the contained flight IDs into training, validation, and test sets.
# By default, the split ratio is 70% training, 15% validation, and 15% test.
#
# A fixed random seed (default: 42) is used to ensure reproducibility, meaning
# that the same input and seed will always produce identical splits. The script
# guarantees that there is no overlap between splits, and that the distribution
# of flight parts (e.g. part1/part2/part3) remains naturally balanced due to
# random shuffling.
#
# USAGE INSTRUCTIONS:
#   1. Ensure the Input Dataset File Exists:
#       - Default: hard_dataset_flights.txt
#
#   2. Run the Script from Project Root:
#       - python -m src.data.split_dataset
#
#   3. Optional Arguments:
#       - --input <path>        : Input dataset file
#       - --output-dir <path>   : Directory for split outputs
#       - --seed <int>          : Random seed (default: 42)
#       - --train-ratio <float> : Training split ratio
#       - --val-ratio <float>   : Validation split ratio
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

import random
import argparse
from pathlib import Path
from collections import Counter
from logging_module import LogLevel, log

# ------------------------------------------------------------------------------
#                               DATA PARSING
# ------------------------------------------------------------------------------
def parse_flight_ids(filepath: str) -> list[str]:
    """
    Reads flight IDs from a dataset text file.

    Only lines beginning with 'part' are considered valid flight IDs.
    All header, comment, or malformed lines are ignored.

    Args:
        filepath (str): Path to the dataset file.

    Returns:
        list[str]: List of valid flight IDs.
    """
    # Open File & Read All Lines.
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Filter Only Valid Flight ID Lines.
    flight_ids = [l.strip() for l in lines if l.strip().startswith('part')]

    # Return the Flight IDs.
    return flight_ids
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               DATA SPLITTING
# ------------------------------------------------------------------------------
def split(flight_ids: list[str], 
          seed: int, 
          train_ratio: float = 0.70,
          val_ratio: float = 0.15) -> tuple[list, list, list]:
    """
    Shuffles and splits flight IDs into train / validation / test sets.

    Args:
        flight_ids (list[str]): List of input flight IDs.
        seed (int): Random seed for reproducibility.
        train_ratio (float): Fraction allocated to training set.
        val_ratio (float): Fraction allocated to validation set.

    Returns:
        tuple[list, list, list]: Train, validation, and test splits.
    """
    # Create a Copy to Avoid Modifying Original List.
    shuffled = flight_ids[:]

    # Apply Deterministic Shuffle.
    random.seed(seed)
    random.shuffle(shuffled)

    # Compute Split Sizes.
    n = len(shuffled)
    n_test  = round(n * (1.0 - train_ratio - val_ratio))
    n_val   = round(n * val_ratio)
    n_train = n - n_val - n_test

    # Slice into Respective Splits.
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    # Return the Dataset Splits.
    return train, val, test
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               VERIFICATION
# ------------------------------------------------------------------------------
def verify_no_overlap(train, val, test):
    """
    Ensures there is no overlap between dataset splits. Raises an assertion 
    error if any flight ID appears in more than one split.

    Args:
        train (list[str]): List of flight IDs assigned to the training set.
        val   (list[str]): List of flight IDs assigned to the validation set.
        test  (list[str]): List of flight IDs assigned to the test set.

    Returns:
        None:
    """
    # Run Checks on Datasets.
    assert len(set(train) & set(val))  == 0, "OVERLAP: train and val share flights!"
    assert len(set(train) & set(test)) == 0, "OVERLAP: train and test share flights!"
    assert len(set(val)   & set(test)) == 0, "OVERLAP: val and test share flights!"

    # Print Success Message. 
    log(LogLevel.INFO, "split_dataset:verify_no_overlap", 
        "Overlap Check Passed. No Flights Appear in More than One Split.")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               REPORTING
# ------------------------------------------------------------------------------
def print_stats(train, val, test):
    """
    Displays summary statistics for the dataset splits. Includes total counts, 
    percentage distribution, and part balance.

    Args:
        train (list[str]): List of flight IDs assigned to the training set.
        val   (list[str]): List of flight IDs assigned to the validation set.
        test  (list[str]): List of flight IDs assigned to the test set.

    Returns:
        None:
    """
    # Compute Total Number of Flights Across All Splits.
    n = len(train) + len(val) + len(test)

    # Display High-Level Split Summary.
    print(f"\n{'='*52}")
    print(f"  Total Flights : {n}")
    print(f"  Train         : {len(train):>4}  ({len(train)/n*100:.1f}%)")
    print(f"  Val           : {len(val):>4}  ({len(val)/n*100:.1f}%)")
    print(f"  Test          : {len(test):>4}  ({len(test)/n*100:.1f}%)")
    print(f"{'='*52}")
    print()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               FILE OUTPUT
# ------------------------------------------------------------------------------
def write_split(flight_ids: list[str], filepath: Path, seed: int,
                total: int, train_n: int, val_n: int, test_n: int):
    """
    Writes a dataset split to disk with provenance metadata.

    Args:
        flight_ids (list[str]): IDs to write.
        filepath (Path): Output file path.
        seed (int): Random seed used.
        total (int): Total number of samples.
        train_n (int): Training set size.
        val_n (int): Validation set size.
        test_n (int): Test set size.
    
    Returns:
        None:
    """
    # Construct File Header.
    header = (
        f"# Auto-Generated by split_dataset.py\n"
        f"# Seed: {seed}  |  Total: {total}  |  "
        f"Train: {train_n}  Val: {val_n}  Test: {test_n}\n"
        f"# DO NOT EDIT Manually. Regenerate Via Script\n\n"
    )

    # Write Header & Flight IDs.
    with open(filepath, 'w') as f:
        f.write(header)
        f.write('\n'.join(flight_ids) + '\n')

    # Print Info Message.
    log(LogLevel.INFO, "split_dataset:verify_no_overlap", 
        "Written: %s (%d flights)", filepath, len(flight_ids))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    """
    Entry point for dataset splitting workflow.
    """
    # Parse Command-Line Arguments.
    parser = argparse.ArgumentParser(description='Split hard_dataset_flights.txt')
    parser.add_argument('--input',      default='configs/datasets/hard_dataset_flights.txt',
                        help='Path to hard_dataset_flights.txt')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to write train/val/test.txt')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                        help='Fraction for training (default: 0.70)')
    parser.add_argument('--val-ratio',   type=float, default=0.15,
                        help='Fraction for validation (default: 0.15)')
    args = parser.parse_args()

    # Ensure Output Directory Exists.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Dataset.
    print()
    log(LogLevel.INFO, "split_dataset:main", 
        "Reading: %s", args.input)
    flight_ids = parse_flight_ids(args.input)
    log(LogLevel.INFO, "split_dataset:main", 
        "Found: %d flight IDs", len(flight_ids))

    # Perform Split.
    train, val, test = split(
        flight_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Validate Splits.
    verify_no_overlap(train, val, test)

    # Display Statistics.
    print_stats(train, val, test)

    # Write Output Files.
    n = len(flight_ids)
    for name, ids in [('train', train), ('val', val), ('test', test)]:
        write_split(
            ids,
            filepath=output_dir / f'{name}.txt',
            seed=args.seed,
            total=n,
            train_n=len(train),
            val_n=len(val),
            test_n=len(test),
        )

    print()
    log(LogLevel.INFO, "split_dataset:main", 
        "Done. Record Seed for Reproducibility.\n")
# ------------------------------------------------------------------------------


# Call the Main Function
if __name__ == "__main__":
    main()
