# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: select_introspection_configs.py %
# Date Created:            %date_created: Tue Sep 15 20:07:45 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Reads the consolidated parameter-sweep CSV produced by 
# run_introspection_modules.sh --sweep, which contains hyperparameter 
# configurations alongside their corresponding false alarm rate (FAR) and false 
# negative rate (FNR) metrics on the validation split. For each introspection 
# module, this script identifies and selects the single configuration whose FAR 
# is closest to a specified --target-far, using FNR to break any ties. The 
# selected winning configurations are then exported to a JSON file, which is 
# consumed downstream by run_introspection_modules.sh --all to re-run and 
# evaluate each module on the test split.
#
# USAGE:
#   1. Inspect the Achievable Curves First:
#         python -m src.introspection_module.select_introspection_configs --csv outputs/results/introspection_module/parameter_sweep.csv --show-curves
#
#   2. Then Select at a Shared Target FAR:
#         python -m src.introspection_module.select_introspection_configs --csv outputs/results/introspection_module/parameter_sweep.csv --target-far select_target --output outputs/results/introspection_module/selected_params.json
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
import json
import argparse

import pandas as pd
from logging_module import LogLevel, log


# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# Absolute FAR Distance Beyond which a Selection is Flagged. A Module Selected
# Further than this from the Shared Target is Not Really Operating at the Same
# FAR as the Others, so the FNR Comparison Against Them is Not Like-for-Like.
FAR_MATCH_TOLERANCE = 0.01

# Column Widths Used by the Operating-Curve Tables.
PARAMS_COL_WIDTH = 60
METRIC_COL_WIDTH = 10

# Column Widths Used by the Achievable-FAR-Range Table.
MODULE_COL_WIDTH = 32
RANGE_COL_WIDTH = 26
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               ARGUMENT PARSING
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Defines & parses command-line arguments for the selection script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Select, per Introspection Module, the Swept Configuration "
                     "Closest to a Shared Target FAR.")

    parser.add_argument('--csv', type=str,
                         default='outputs/results/introspection_module/parameter_sweep.csv',
                         help='Path to the Parameter-Sweep Results CSV Produced by '
                              'run_introspection_modules.sh --sweep.')

    parser.add_argument('--target-far', type=float, default=None,
                         help='Shared Target False Alarm Rate to Match Across '
                              'Modules. Required Unless --show-curves is Used Alone.')

    parser.add_argument('--output', type=str,
                         default='outputs/results/introspection_module/selected_params.json',
                         help='Where to Write the Selected Per-Module Configurations.')

    parser.add_argument('--show-curves', action='store_true',
                         help='Print Each Module\'s Full (FAR, FNR) Curve, Sorted by '
                              'FAR, Before Selecting -- Use this to Choose --target-far.')

    return parser.parse_args()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               SELECTION HELPER
# ------------------------------------------------------------------------------
def select_closest_to_target(module_df: pd.DataFrame, target_far: float) -> pd.Series:
    """
    Picks the row whose false_alarm_rate is closest to target_far, breaking
    ties by the lowest false_negative_rate.

    Args:
        module_df (pd.DataFrame): Sweep rows for a single module.
        target_far (float): Shared target false alarm rate.

    Returns:
        pd.Series: The selected row.
    """
    # Rank Every Row by Distance from the Target FAR, then by FNR to Break Ties.
    ranked = module_df.assign(
        far_distance=(module_df['false_alarm_rate'] - target_far).abs()
    ).sort_values(by=['far_distance', 'false_negative_rate'], ascending=[True, True])

    # Return the Best-Ranked Row.
    return ranked.iloc[0]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               DISPLAY HELPER
# ------------------------------------------------------------------------------
def print_formatted_table(module_name: str, module_df: pd.DataFrame) -> None:
    """
    Prints a cleanly formatted table of every configuration evaluated for a
    single module, sorted by ascending FAR so the rows read as that module's
    operating curve from rarely-escalating to frequently-escalating.

    Args:
        module_name (str): Identifier of the introspection module.
        module_df (pd.DataFrame): Sweep rows for that module.

    Returns:
        None
    """
    # Order the Rows so the Table Reads as an Operating Curve.
    sorted_df = module_df.sort_values('false_alarm_rate')

    # Build the Horizontal Rule Once so Every Row Lines Up.
    divider = ("+" + "-" * (PARAMS_COL_WIDTH + 2) +
               "+" + "-" * (METRIC_COL_WIDTH + 2) +
               "+" + "-" * (METRIC_COL_WIDTH + 2) + "+")

    # Title the Table with the Module & How Many Configurations it Covers.
    header_title = f"MODULE: {module_name} ({len(module_df)} configs)"

    # Print the Table Header.
    print("\n")
    print(f"{header_title}")
    print(divider)
    print(f"| {'Parameters (JSON)':<{PARAMS_COL_WIDTH}} | "
          f"{'FAR':<{METRIC_COL_WIDTH}} | {'FNR':<{METRIC_COL_WIDTH}} |")
    print(divider)

    # Print One Row per Configuration. Six Decimal Places, Since Neighbouring
    # Configurations Can Differ by Very Little at Low FAR.
    for _, row in sorted_df.iterrows():
        params = str(row['params_json'])
        far = f"{row['false_alarm_rate']:.6f}"
        fnr = f"{row['false_negative_rate']:.6f}"
        print(f"| {params:<{PARAMS_COL_WIDTH}} | "
              f"{far:<{METRIC_COL_WIDTH}} | {fnr:<{METRIC_COL_WIDTH}} |")

    # Close the Table.
    print(divider)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                   MAIN
# ------------------------------------------------------------------------------
def main() -> None:
    # Read Optional Parsed Arguments. 
    args = parse_args()

    # Load the Sweep Results.
    if not os.path.exists(args.csv):
        log(LogLevel.ERROR, "select_introspection_configs:load",
            "Parameter-Sweep CSV Not Found: %s.", args.csv)
        raise FileNotFoundError(f"Parameter-Sweep CSV Not Found: {args.csv}")

    # Read Every Configuration Evaluated During the Sweep.
    sweep_df = pd.read_csv(args.csv)

    # Confirm the Expected Columns are Present.
    required_cols = {'module', 'params_json', 'false_alarm_rate', 'false_negative_rate', 'run_dir'}
    missing_cols = required_cols - set(sweep_df.columns)
    if missing_cols:
        log(LogLevel.ERROR, "select_introspection_configs:load",
            "Sweep CSV is Missing Expected Columns: %s.", sorted(missing_cols))
        raise ValueError(f"Sweep CSV is Missing Expected Columns: {sorted(missing_cols)}")

    # Drop Rows from Runs that Failed to Produce a Metric. Copied so the Column
    # Clean-Up Below Writes to this Frame Rather than a View of sweep_df.
    valid_df = sweep_df.dropna(subset=['false_alarm_rate', 'false_negative_rate']).copy()
    dropped = len(sweep_df) - len(valid_df)
    if dropped:
        log(LogLevel.WARNING, "select_introspection_configs:load",
            "Ignoring %d Row(s) with Missing FAR/FNR.", dropped)

    # Nothing Usable Left to Select From.
    if valid_df.empty:
        log(LogLevel.ERROR, "select_introspection_configs:load",
            "No Rows with Valid FAR/FNR in %s. Nothing to Select.", args.csv)
        raise ValueError("Parameter-Sweep CSV Contains No Usable Rows.")

    # Strip Stray Whitespace so the Params Column Aligns in the Tables Below &
    # json.loads() Never Trips on a Padded Value.
    valid_df['params_json'] = valid_df['params_json'].astype(str).str.strip()

    # Optionally Print Each Module's Full Operating Curve for Manual Inspection.
    if args.show_curves:
        for module_name, module_df in valid_df.groupby('module'):
            print_formatted_table(module_name, module_df)

        # Print the Achievable FAR Range per Module so a Sensible Shared Target can be Chosen.
        print("\n")
        print("Achievable FAR Range per Module")
        print("+" + "-" * (MODULE_COL_WIDTH + 2) + "+" + "-" * (RANGE_COL_WIDTH + 2) + "+")
        for module_name, module_df in valid_df.groupby('module'):
            min_far = module_df['false_alarm_rate'].min()
            max_far = module_df['false_alarm_rate'].max()
            range_str = f"{min_far:.4f} -- {max_far:.4f}"
            print(f"| {module_name:<{MODULE_COL_WIDTH}} | {range_str:<{RANGE_COL_WIDTH}} |")
        print("+" + "-" * (MODULE_COL_WIDTH + 2) + "+" + "-" * (RANGE_COL_WIDTH + 2) + "+")

        # Report the Overlap and the Band Every Module can Actually Reach. (A  Target 
        # Outside this Band Cannot be Matched by at Least One Module).
        overlap_low = valid_df.groupby('module')['false_alarm_rate'].min().max()
        overlap_high = valid_df.groupby('module')['false_alarm_rate'].max().min()
        if overlap_low <= overlap_high:
            print(f"\n  Common Overlap Band : {overlap_low:.4f} -- {overlap_high:.4f}")
            print("  [INFO] Choose --target-far inside this band for best alignment.")
        else:
            log(LogLevel.WARNING, "select_introspection_configs:curves",
                "No FAR Value is Reachable by Every Module (Ranges Do Not Overlap).")

    # Selection Requires a Target FAR.
    if args.target_far is None:
        if args.show_curves:
            print("\nNo --target-far Supplied -- Stopping after --show-curves.\n")
            return
        log(LogLevel.ERROR, "select_introspection_configs:args",
            "--target-far is Required (Unless Only --show-curves was Requested).")
        raise ValueError("--target-far is Required.")

    # Select the Closest Configuration per Module.
    selected = {}

    # Banner the Selection Stage so it Stands Apart from the Curve Tables Above.
    header_str = f" SELECTING CONFIGURATIONS CLOSEST TO TARGET FAR: {args.target_far:.4f} "
    print("\n" + header_str.center(88, "="))
    print()

    for module_name, module_df in valid_df.groupby('module'):
        # Find the Winning Introspection Module Parameters. 
        winner = select_closest_to_target(module_df, args.target_far)

        # Record the Winning Configuration Alongside the Validation Metrics it was 
        # Chosen On at Selection Time.
        selected[module_name] = {
            'params': json.loads(winner['params_json']),
            'false_alarm_rate': float(winner['false_alarm_rate']),
            'false_negative_rate': float(winner['false_negative_rate']),
            'validation_run_dir': winner['run_dir'],
        }

        # Show the Winning Configuration for this Module.
        print(f"  * {module_name:<28} | FAR={winner['false_alarm_rate']:.4f} | FNR={winner['false_negative_rate']:.4f}")
        print(f"    Params: {winner['params_json']}\n")

        # Flag Any Module that Could Not Get Close to the Shared Target.
        far_distance = abs(float(winner['false_alarm_rate']) - args.target_far)
        if far_distance > FAR_MATCH_TOLERANCE:
            log(LogLevel.WARNING, "select_introspection_configs:selection",
                "Module '%s' Best FAR is %.4f (%.4f away from target %.4f).",
                module_name, float(winner['false_alarm_rate']), far_distance, args.target_far)

    # Warn if the Sweep Did Not Cover Every Tunable Module.
    expected_modules = {'confidence_threshold', 'rolling_confidence_trend',
                        'coast_count_threshold', 'kalman_filter_nis'}
    absent_modules = expected_modules - set(selected)
    if absent_modules:
        log(LogLevel.WARNING, "select_introspection_configs:selection",
            "No Sweep Rows Found for: %s. These Modules Will be Skipped by "
            "run_introspection_modules.sh --all.", sorted(absent_modules))

    # Save the Selected Configurations.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'target_far': args.target_far, 'modules': selected}, f, indent=2)

    # Point at the Output File.
    print(f"Selected Configurations Written to: {args.output}\n")
# ------------------------------------------------------------------------------


# Call the Main Function.
if __name__ == "__main__":
    main()
