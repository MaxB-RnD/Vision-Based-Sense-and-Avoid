


# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: visualise_results.py %
# Date Created:            %date_created: Mon May 08 12:26:32 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Visualises the Basline Backbone Model Testing completed in the model_results.csv,
# this is presented as a 9x9 grid of scatter plots.
#
# USAGE INSTRUCTIONS:
#   1. Run the Script from Project Root:
#       - python3 -m src.data.visualise_results.py
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
import seaborn as sns
from adjustText import adjust_text

# Load the Data.
df = pd.read_csv('results/core_improvement/model_results.csv')

# Define the Metrics.
x_vars = ['Params', 'GFLOPS', 'FPS']
y_vars = ['AFDR', 'FPPI', 'EDR']

x_labels = ['Parameters (Millions)', 'Computational Complexity (GFLOPS)', 'Inference Speed (FPS)']
y_labels = ['AFDR (%)', 'FPPI', 'EDR Total (%)']

# Define Distinct Colours for the Three Columns (Blue, Orange, Green).
col_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 3. Set up the PowerPoint-Friendly Figure.
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(16, 9), dpi=300)
fig.suptitle('Backbone Performance Trade-offs', fontsize=16, fontweight='bold', y=0.98)

# 4. Generate the 3x3 Grid.
for i, y_col in enumerate(y_vars):
    for j, x_col in enumerate(x_vars):
        ax = axes[i, j]
        
        # Plot the Points Using the Colour Assigned to this Specific Column.
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, s=100, color=col_colors[j])
        
        # Create a List to Store the Text Objects.
        texts = []
        for idx, row in df.iterrows():
            texts.append(ax.text(row[x_col], row[y_col], f"{row['Model']}", fontsize=8))
        
        # Magically Repel the Labels Away from Each Other & Draw Connecting Lines.
        adjust_text(texts, ax=ax, force_points=0.2, force_text=0.2,
                    expand_points=(1, 1), expand_text=(1, 1),
                    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, shrinkA=5, shrinkB=5))
        
        # Set Axis Titles, Matching the X-axis Text Colour to the Dot Colours.
        if i == 2: 
            ax.set_xlabel(x_labels[j], fontsize=10, fontweight='bold', color=col_colors[j])
        else:
            ax.set_xlabel('')
            
        if j == 0: 
            ax.set_ylabel(y_labels[i], fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('')

# Clean Up Layout & Save.
plt.tight_layout()
plt.subplots_adjust(top=0.93) 
plt.savefig('results/core_improvement/SAA_Backbone_Tradeoffs_Slide.png', bbox_inches='tight')
plt.show()
