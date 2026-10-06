# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: visualise_corruption_results.py %
# Date Created:            %date_created: Fri Sep 11 15:50:21 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 3 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# This script visualises the corruption ramp sweep results stored in
# corruption_ramp_sweep_results.csv by generating a 2x3 grid of small-multiple
# line plots, each showing how a single metric behaves as a function of ramp
# length (frames), with one line per ramp shape (linear / sigmoid).
#
# This sweep examines how increasing corruption ramp length affects the measured 
# corruption characteristics and PSNR performance for linear and sigmoid ramp 
# shapes. The six panels show complementary metrics covering corruption severity, 
# ramp behaviour, ramp coverage, and PSNR, with reduced sigmoid PSNR sample 
# coverage explicitly indicated at longer ramp lengths. A consistent visual style
# is maintained across all panels with shared colours, padded axes, a hatched 
# background, a common legend, and high-resolution PNG export.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from Project Root:
#       - python3 -m src.data.corruption_dataset.visualise_corruption_results
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
import matplotlib.lines as mlines
import seaborn as sns

# ------------------------------------------------------------------------------
#                                   CONFIGURATION
# ------------------------------------------------------------------------------
# Load Data.
df = pd.read_csv('results/corruption_dataset/corruption_ramp_sweep_results.csv')

# Split Off the Baseline Row (no ramp_shape / ramp_length) From the Ramp Runs.
baseline_row = df[df['ramp_shape'].isna()].iloc[0]
ramp_df = df[df['ramp_shape'].notna()].copy()
ramp_df['ramp_length_frames'] = ramp_df['ramp_length_frames'].astype(float)

# Quantify PSNR Reliability (The Fraction of Ramp-Covered Frames that Contributed 
# a Peak-PSNR Sample).
ramp_df['psnr_reliability_pct'] = (
    ramp_df['psnr_at_peak_count'] / ramp_df['scr_ramp_coverage_count'] * 100.0
)

# Consistent Colour Mapping for Each Ramp Shape.
SHAPE_COLOURS = {
    'linear':  '#1f77b4',
    'sigmoid': '#ff7f0e',
}

# Ramp Length (Frames) Beyond which Sigmoid PSNR-at-Peak Values are  Pulled from 
# too Small a Sample.
SIGMOID_PSNR_TRUST_CUTOFF = 10

# Padding Ratio Applied to Axis Limits So Points Do Not Sit Directly Against
# Subplot Boundaries.
PADDING = 0.05

# Configure the Overall Plotting Theme.
sns.set_theme(style="whitegrid")

# Create a 2x3 Subplot Layout, One Panel Per Metric.
fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)

# Add a Shared Figure Title.
fig.suptitle('AOT Dataset Corruption Analysis',
             fontsize=20, fontweight='bold', y=0.97)


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
    # Get the Minimum & Maximum Values from the Series.
    lo, hi = series.min(), series.max()

    # Calculate the Range of the Data, or Use a Fallback if all Values are the Same.
    span = hi - lo if hi != lo else abs(hi) or 1.0

    # Return the Minimum & Mximum Values with Padding Added.
    return lo - span * padding, hi + span * padding


# Shared x-axis Limits (Ramp Length) so all Panels Remain Visually Aligned.
X_LIMITS = padded_limits(ramp_df['ramp_length_frames'])
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                            SHARED PANEL DECORATION
# ------------------------------------------------------------------------------
def style_axis(ax, x_label, y_label, show_xlabel, show_ylabel):
    """
    Apply the shared visual styling and axis labels to a subplot.

    Args:
        ax (matplotlib.axes.Axes): Axes object to style.
        x_label (str): Label displayed on the x-axis.
        y_label (str): Label displayed on the y-axis.
        show_xlabel (bool): Whether to display the x-axis label.
        show_ylabel (bool): Whether to display the y-axis label.

    Returns:
        None: The supplied axes object is modified in place.
    """
    # Set a Clean White Subplot Background.
    ax.set_facecolor('white')

    # Draw Grid Lines Behind Plotted Data.
    ax.set_axisbelow(True)

    # Add Major Grid Lines.
    ax.grid(True, which='major', color='#cccccc', linewidth=0.6)

    # Create the Diagonal Hatch Overlay Pattern for the Background.
    hatch_patch = mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax.transAxes, linewidth=0,
        facecolor='none', edgecolor='#cccccc', hatch='///', zorder=0
    )

    # Add the Hatch Overlay to the Subplot.
    ax.add_patch(hatch_patch)

    # Apply the Shared Ramp-Length Limits.
    ax.set_xlim(X_LIMITS)

    # Check Whether the X-axis Label Should Be Displayed.
    if show_xlabel:
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold', labelpad=10)

    # Check Whether the Y-axis Label Should Be Displayed.
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold', labelpad=10)


def plot_shape_lines(ax, y_col, std_col=None, band=False):
    """
    Plot one metric line for each corruption ramp shape.

    Each ramp shape is plotted against ramp length using the predefined
    colour mapping. An optional standard-deviation band can be added when
    a corresponding standard-deviation column is supplied.

    Args:
        ax (matplotlib.axes.Axes): Axes object on which to plot the data.
        y_col (str): DataFrame column containing the metric mean values.
        std_col (str | None): DataFrame column containing standard-deviation
                              values used for the optional uncertainty band.
        band (bool): Whether to display the standard-deviation band.

    Returns:
        None: The supplied axes object is modified in place.
    """
    # Plot Each Configured Ramp Shape.
    for shape, colour in SHAPE_COLOURS.items():
        # Select Rows Belonging to the Current Ramp Shape.
        sub = ramp_df[ramp_df['ramp_shape'] == shape].sort_values('ramp_length_frames')

        # Extract Ramp Lengths for the X-axis.
        x = sub['ramp_length_frames'].values

        # Extract Metric Means for the Y-axis.
        y = sub[y_col].values

        # Add Uncertainty Only when Available.
        if band and std_col is not None:
            # Extract Standard-Deviation Values.
            std = sub[std_col].values

            # Shade the Mean +/- Standard-Deviation Region.
            ax.fill_between(x, y - std, y + std, color=colour, alpha=0.15, zorder=2)

        # Plot the Metric Mean for the Current Ramp Shape.
        ax.plot(x, y, color=colour, lw=2.0, marker='o', markersize=6, zorder=4)


def plot_psnr_peak(ax):
    """
    Plot PSNR-at-peak results for linear and sigmoid corruption ramps.

    Linear results are plotted normally. Sigmoid results are split at the
    configured trust cutoff: results at or below the cutoff use a solid line,
    while results above the cutoff use a dashed line with hollow markers to
    indicate reduced PSNR sample coverage.

    Args:
        ax (matplotlib.axes.Axes): Axes object on which to plot the PSNR data.

    Returns:
        None: The supplied axes object is modified in place.
    """
    # Select Linear Ramp Results.
    lin = ramp_df[ramp_df['ramp_shape'] == 'linear'].sort_values('ramp_length_frames')

    # Plot the Complete Linear PSNR Series.
    ax.plot(lin['ramp_length_frames'], lin['psnr_at_peak_mean'],
             color=SHAPE_COLOURS['linear'], lw=2.0, marker='o', markersize=6, zorder=4)

    # Select Sigmoid Ramp Results.
    sig = ramp_df[ramp_df['ramp_shape'] == 'sigmoid'].sort_values('ramp_length_frames') 

    # Select Sigmoid Results within the Trusted Range.
    trusted = sig[sig['ramp_length_frames'] <= SIGMOID_PSNR_TRUST_CUTOFF]

    # Select Sigmoid Results Beyond the Trust Cutoff.
    untrusted = sig[sig['ramp_length_frames'] >= SIGMOID_PSNR_TRUST_CUTOFF]

    # Plot the Trusted Sigmoid PSNR Series.
    ax.plot(trusted['ramp_length_frames'], trusted['psnr_at_peak_mean'],
             color=SHAPE_COLOURS['sigmoid'], lw=2.0, marker='o', markersize=6, zorder=4)
    
    # Plot the Reduced-Coverage Sigmoid PSNR Series.
    ax.plot(untrusted['ramp_length_frames'], untrusted['psnr_at_peak_mean'],
             color=SHAPE_COLOURS['sigmoid'], lw=1.6, linestyle='--',
             marker='o', markersize=6, markerfacecolor='white',
             markeredgecolor=SHAPE_COLOURS['sigmoid'], markeredgewidth=1.5, zorder=4)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                PANEL 1: TERMINAL SCR
# ------------------------------------------------------------------------------
# Select the Top-Left Subplot.
ax = axes[0, 0]

# Plot SCR Means with Standard-Deviation Bands.
plot_shape_lines(ax, 'terminal_scr_mean', std_col='terminal_scr_std', band=True)

# Apply Shared Styling & Axis Labels.
style_axis(ax, 'Ramp Length (Frames)', 'Terminal SCR (mean \u00b1 std)',
           show_xlabel=False, show_ylabel=True)

# Set the Subplot Title.
ax.set_title('Terminal SCR', fontsize=11, fontweight='bold')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                          PANEL 2: BACKGROUND COMPLEXITY
# ------------------------------------------------------------------------------
# Select the Top-Middle Subplot.
ax = axes[0, 1]

# Plot Background Complexity for Each Ramp Shape.
plot_shape_lines(ax, 'background_complexity_mean')

# Retrieve the Clean Baseline Value.
baseline_val = baseline_row['clean_background_complexity_mean']

# Add a Horizontal Reference Line for the Clean Baseline.
ax.axhline(baseline_val, color='#333333', lw=1.2, linestyle=':', zorder=2)

# Label the Clean Baseline Reference Line.
ax.text(X_LIMITS[1], baseline_val, f'  Clean baseline ({baseline_val:.1f})',
        fontsize=7.5, color='#333333', va='bottom', ha='right')

# Apply Shared Styling & Axis Labels.
style_axis(ax, 'Ramp Length (Frames)', 'Background Complexity (mean)',
           show_xlabel=False, show_ylabel=True)

# Set the Subplot Title.
ax.set_title('Background Complexity', fontsize=11, fontweight='bold')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           PANEL 3: RAMP MONOTONICITY
# ------------------------------------------------------------------------------
# Select the Top-Right Subplot.
ax = axes[0, 2]

# Plot Monotonicity Means with Standard-Deviation Bands.
plot_shape_lines(ax, 'scr_monotonicity_mean', std_col='scr_monotonicity_std', band=True)

# Apply Shared Styling & Axis Labels.
style_axis(ax, 'Ramp Length (Frames)', 'SCR Monotonicity (mean \u00b1 std)',
           show_xlabel=False, show_ylabel=True)

 # Set the Subplot Title.
ax.set_title('Ramp Monotonicity', fontsize=11, fontweight='bold')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                             PANEL 4: RAMP COVERAGE
# ------------------------------------------------------------------------------
# Select the Bottom-Left Subplot.
ax = axes[1, 0]

# Plot Ramp Coverage for Each Ramp Shape.
plot_shape_lines(ax, 'scr_ramp_coverage_mean')

# Apply Shared Styling & Axis Labels.
style_axis(ax, 'Ramp Length (Frames)', 'SCR Ramp Coverage (mean)',
           show_xlabel=True, show_ylabel=True)

# Set the Subplot Title.
ax.set_title('Ramp Coverage', fontsize=11, fontweight='bold')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                             PANEL 5: PSNR AT PEAK
# ------------------------------------------------------------------------------
# Select the Bottom-Middle Subplot.
ax = axes[1, 1]

# Plot PSNR at Peak with the Sigmoid Coverage Distinction.
plot_psnr_peak(ax)

# Apply Shared Styling & Axis Labels.
style_axis(ax, 'Ramp Length (Frames)', 'PSNR at Peak (mean, dB)',
           show_xlabel=True, show_ylabel=True)

# Set the Subplot Title.
ax.set_title('PSNR at Peak', fontsize=11, fontweight='bold')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           PANEL 6: PSNR RELIABILITY
# ------------------------------------------------------------------------------
# Select the Bottom-Right Subplot.
ax = axes[1, 2]

# Plot the Percentage of Ramp Frames Contributing to PSNR.
plot_shape_lines(ax, 'psnr_reliability_pct')

# Apply Shared Styling & Axis Labels.
style_axis(ax, 'Ramp Length (Frames)', 'PSNR Sample Reliability (%)',
           show_xlabel=True, show_ylabel=True)

# Set the Subplot Title.
ax.set_title('PSNR Reliability', fontsize=11, fontweight='bold')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                SHARED LEGEND
# ------------------------------------------------------------------------------
# Create Legend Handles for the Two Ramp Shapes.
shape_handles = [
    mlines.Line2D([], [], color=colour, marker='o', markersize=6, lw=2.0, label=shape.capitalize())
    for shape, colour in SHAPE_COLOURS.items()
]

# Create the Legend Handle for Reduced Sigmoid PSNR Coverage.
untrusted_handle = mlines.Line2D(
    [], [], color=SHAPE_COLOURS['sigmoid'], lw=1.6, linestyle='--',
    marker='o', markersize=6, markerfacecolor='white',
    markeredgecolor=SHAPE_COLOURS['sigmoid'], markeredgewidth=1.5,
    label=f'Sigmoid (Reduced PSNR Sample Coverage)'
)

# Create the Legend Handle for the Clean Baseline.
baseline_handle = mlines.Line2D(
    [], [], color='#333333', lw=1.2, linestyle=':', label='Clean Baseline Reference'
)

# Add the Shared Legend Below All Subplots.
fig.legend(
    handles=shape_handles + [untrusted_handle, baseline_handle],
    loc='lower center',
    ncol=4,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                                LAYOUT & SAVE
# ------------------------------------------------------------------------------
# Optimise Subplot Spacing & Export the Final Figure.
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.10, hspace=0.30)

# Save the Figure as a High-Resolution PNG Image.
plt.savefig('results/corruption_dataset/Corruption_Dataset_Ramp_Sweep.png', bbox_inches='tight')
print("Saved: results/corruption_dataset/Corruption_Dataset_Ramp_Sweep.png")
# ------------------------------------------------------------------------------
