# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: visualise_tracking_results.py %
# Date Created:            %date_created: Mon May 08 12:26:32 2026 %
# CM Version               %version: 2 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# This script visualises the baseline backbone model evaluation results stored
# in model_results_tracking_metrics.csv by generating a 3x3 grid of scatter plots
# comparing detection performance metrics (AFDR, CEDR, FTPS) against model
# complexity and inference speed.
#
# The plots are intended to support comparative analysis of backbone
# architectures by highlighting performance trade-offs, acceptable operating
# regions, and overall model efficiency. The final figure is exported as a
# high-resolution PNG suitable for reporting and technical documentation.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from Project Root:
#       - python3 -m src.data.original_dataset.visualise_tracking_results
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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ------------------------------------------------------------------------------
#                                   CONFIGURATION
# ------------------------------------------------------------------------------
# Load Data.
df = pd.read_csv('results/core_improvement/model_results_tracking_metrics.csv')

# Variables Used for the X-axis & Y-axis Comparisons.
x_vars   = ['Params', 'GFLOPS', 'FPS']
y_vars   = ['AFDR',   'CEDR',   'FTPS']

# Human-Readable Axis Labels for Each Metric.
x_labels = ['Parameters (Millions)', 'Computational Complexity (GFLOPS)', 'Inference Speed (FPS)']
y_labels = ['AFDR (%)', 'CEDR (%)', 'FTPS (%)']

# Consistent Colour Mapping for Each Backbone Architecture.
MODEL_COLOURS = {
    'MobileNetV4-S':   '#1f77b4',
    'EfficientViT-M2': '#ff7f0e',
    'EfficientViT-B2': '#2ca02c',
    'ResNet34':        '#d62728',
    'HRNet-W32':       '#9467bd',
    'HRNet-W32 T2D':      '#17becf',
    '*Winning Sol.':   '#8c564b',
}

# Acceptable FPS Operating Region Used for Visual Highlighting.
FPS_GREEN_LOW   = 8.5
FPS_GREEN_HIGH  = 11.5

# Padding Ratio Applied to Axis Limits So Points Do Not Sit Directly Against
# Subplot Boundaries.
PADDING = 0.05

# Configure the Overall Plotting Theme.
sns.set_theme(style="whitegrid")

# Create a 3x3 Subplot Layout for All Metric Comparisons.
fig, axes = plt.subplots(3, 3, figsize=(16, 9), dpi=300)

# Add a Shared Figure Title.
fig.suptitle('Backbone Performance Trade-offs',
             fontsize=20, fontweight='bold', y=0.90)

# Define Reusable Padding Function.
def padded_limits(series, padding=PADDING):
    """
    Calculate padded axis limits for improved plot readability.

    Args:
        series (pd.Series): Data series used to determine axis limits.
        padding (float):    Fractional padding applied to the data range.

    Returns:
        tuple: Lower and upper axis limits with padding applied.
    """
    lo, hi = series.min(), series.max()
    span = hi - lo if hi != lo else abs(hi) or 1.0
    return lo - span * padding, hi + span * padding

# Compute x-axis Limits So all Plots Remain Visually Aligned.
x_limits = [padded_limits(df[col]) for col in x_vars]

# Compute y-axis Limits (linear scale for all three metrics: AFDR, CEDR, FTPS).
y_limits = [padded_limits(df[col]) for col in y_vars]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                DRAW EACH SUBPLOT
# ------------------------------------------------------------------------------
# Generate Each Subplot Independently.
for i, y_col in enumerate(y_vars):
    for j, x_col in enumerate(x_vars):
        # Make Axis.
        ax = axes[i, j]

        # Configure the Subplot Background Styling.
        ax.set_facecolor('white')
        ax.set_axisbelow(True)

        # Draw Major Grid Lines.
        ax.grid(True, which='major', color='#cccccc', linewidth=0.6)

        # Add a Subtle Cross-Hatched Background for Visual Structure.
        hatch_patch = mpatches.FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes, linewidth=0, facecolor='none', edgecolor='#cccccc', hatch='///', zorder=0)
        ax.add_patch(hatch_patch)

        # Apply Shared x-axis Limits Across Each Subplot Column.
        ax.set_xlim(x_limits[j])

        # Apply Shared y-axis Limits Across Each Subplot Row (all linear now).
        ax.set_ylim(y_limits[i])

        # Highlight the Acceptable FPS Operating Range.
        if x_col == 'FPS':
            ax.axvspan(FPS_GREEN_LOW, FPS_GREEN_HIGH, color='#2ca02c', alpha=0.15, zorder=1)
            ax.axvline(FPS_GREEN_LOW,  color='#2ca02c', lw=1.0, linestyle='--', zorder=2)
            ax.axvline(FPS_GREEN_HIGH, color='#2ca02c', lw=1.0, linestyle='--', zorder=2)

        # Plot All Backbone Model Data Points.
        for _, row in df.iterrows():
            colour = MODEL_COLOURS.get(row['Model'], '#333333')
            ax.scatter(row[x_col], row[y_col], s=100, color=colour, zorder=3)

        # Only Display x-axis Labels on the Bottom Row.
        if i == 2:
            ax.set_xlabel(x_labels[j], fontsize=10, fontweight='bold', labelpad=10)
        else:
            ax.set_xlabel('')

        # Only Display y-axis Labels on the Left Column.
        if j == 0:
            ax.set_ylabel(y_labels[i], fontsize=10, fontweight='bold', labelpad=10)
        else:
            ax.set_ylabel('')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                SHARED LEGEND
# ------------------------------------------------------------------------------
# Create Legend Entries for Each Model Colour.
model_handles = [
    mpatches.Patch(color=colour, label=model)
    for model, colour in MODEL_COLOURS.items()
]

# Create Legend Entries for Highlighted Operating Regions.
region_handles = [
    mpatches.Patch(color='#2ca02c', alpha=0.4, label='Optimal FPS (9-11 FPS)'),
]

# Add a Shared Legend Beneath the Subplot Grid.
fig.legend(
    handles=model_handles + region_handles,
    loc='lower center',
    ncol=len(model_handles) + len(region_handles),
    fontsize=8.5,
    frameon=True,
    bbox_to_anchor=(0.5, -0.03),
)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                LAYOUT & SAVE
# ------------------------------------------------------------------------------
# Optimise Subplot Spacing & Export the Final Figure.
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.09)

# Save the Figure as a High-Resolution PNG Image.
plt.savefig('results/core_improvement/SAA_Backbone_Tradeoffs_Tracking.png', bbox_inches='tight')
print("Saved: results/core_improvement/SAA_Backbone_Tradeoffs_Tracking.png")
# ------------------------------------------------------------------------------
