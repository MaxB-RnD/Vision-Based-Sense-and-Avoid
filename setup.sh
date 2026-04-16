#!/bin/bash
# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: setup.sh %
# Date Created:            %date_created: Mon Mar 23 20:15:12 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# One-time setup script for the Vision-Based Sense and Avoid project. Installs 
# all required Python dependencies and the airborne metrics evaluation package.
#
# USAGE INSTRUCTIONS:
#   1. Create and Activate a Virtual Environment.
#   2. Run the Script from Project Root:
#       - bash setup.sh
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

# Detect CUDA Driver and Install Correct PyTorch Build.
echo "[setup] Detecting CUDA Driver Version..."

# Extract the Driver Version Major Number from Nvidia-smi.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)

# Install the Correct Dependencies for the Graphics Driver. 
if [ -z "$GPU_NAME" ]; then
    # No CUDA Driver Found. 
    echo "[setup] WARNING: nvidia-smi Not Found or No GPU Detected. Skipping PyTorch GPU Install."
    echo "[setup] Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --break-system-packages

elif echo "$GPU_NAME" | grep -qiE "RTX 50|50[0-9]{2}"; then
    # Driver 570+ Supports CUDA 12.8+ (Blackwell / RTX 50-series)
    echo "[setup] Detected Blackwell GPU ($GPU_NAME). Installing PyTorch cu128..."
    pip install -r configs/dependencies/requirements_torch_blackwell.txt --break-system-packages

else
    # All Other GPUs Use cu124.
    echo "[setup] Detected GPU ($GPU_NAME). Installing PyTorch cu124..."
    pip install -r configs/dependencies/requirements_torch_workstation.txt --break-system-packages
fi

# Install Remaining Project Dependencies.
echo "[setup] Installing Project Dependencies..."
pip install -r configs/dependencies/requirements.txt --break-system-packages

# Install the Airborne Metrics Evaluation Package.
echo "[setup] Installing Airborne Metrics Package..."
pip install core/metrics --break-system-packages

# Print Message when Complete.
echo "[setup] Setup Complete."
