# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: corruptions.py %
# Date Created:            %date_created: Wed Aug 12 17:54:22 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Registry of continuously parameterised image corruption functions used to
# generate gradually degrading images for the introspection test set. The 
# available corruptions simulate common image degradation effects including 
# sensor noise, defocus blur, and colour quantisation. Each corruption can be
# applied at a variable severity to produce progressively degraded imagery. The
# registry provides a central collection of the available corruption types for 
# use by the image degradation pipeline.
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

import cv2
import numpy as np

# ------------------------------------------------------------------------------
#                           CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------
# ISO Noise Parameters.
ISO_NOISE_PEAK_CLEAN = 260.0
ISO_NOISE_PEAK_WORST = 6.0
ISO_NOISE_GAUSSIAN_STD_MAX = 22.0

# Maximum Defocus Blur Kernel Size.
DEFOCUS_MAX_KERNEL = 25

# Colour Quantisation Bit Depth Range.
QUANT_BITS_CLEAN = 8
QUANT_BITS_WORST = 2
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def _clampSeverity(severity: float) -> float:
    """
    Clamps a severity value to the valid [0, 1] range.

    Args:
        severity (float): Requested severity value.

    Returns:
        float: Severity clamped to [0, 1].
    """
    # Guard Against Out-of-Range Severities from Upstream Scheduling Bugs.
    return max(0.0, min(1.0, float(severity)))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       CORRUPTION 1 -- ISO / SENSOR NOISE
# ------------------------------------------------------------------------------
def applyIsoNoise(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Applies Poisson-Gaussian sensor noise, scaling continuously with severity.

    Simulates the two dominant sensor noise sources: Poisson (shot) noise from
    photon counting statistics, and additive Gaussian (read) noise from sensor
    electronics. Both components grow monotonically with severity so this is
    safe to drive with a continuous ramp.

    Args:
        image (np.ndarray): Input image, any integer dtype, HxW or HxWxC.
        severity (float):   Corruption severity in [0, 1]. 0 returns the image
                            unchanged, 1 is worst-case noise.

    Returns:
        np.ndarray: Noisy image, same shape and dtype as the input.
    """
    # Clamp Severity to the Valid Range.
    severity = _clampSeverity(severity)

    # Return the Original Image Unchanged when Severity is Zero.
    if severity == 0.0:
        return image

    # Preserve Original dtype for the Final Cast.
    original_dtype = image.dtype
    image_f = image.astype(np.float32)

    # Interpolate the Simulated Photon Count. Lower Peak --> More Visible Shot Noise.
    peak = ISO_NOISE_PEAK_CLEAN + severity * (ISO_NOISE_PEAK_WORST - ISO_NOISE_PEAK_CLEAN)

    # Apply Poisson (Shot) Noise by Treating Normalised Intensity as a Photon Count.
    normalised = np.clip(image_f / 255.0, 0.0, 1.0)
    poisson_noisy = np.random.poisson(normalised * peak).astype(np.float32) / peak * 255.0

    # Apply Additive Gaussian (Read) Noise on Top.
    gaussian_std = severity * ISO_NOISE_GAUSSIAN_STD_MAX
    gaussian_noisy = poisson_noisy + np.random.normal(0.0, gaussian_std, image_f.shape).astype(np.float32)

    # Clip Back into Valid Range & Restore Original dtype.
    return np.clip(gaussian_noisy, 0, 255).astype(original_dtype)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                       CORRUPTION 2 -- DEFOCUS BLUR
# ------------------------------------------------------------------------------
def applyDefocusBlur(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Applies Gaussian defocus blur, with kernel radius scaling continuously
    with severity.

    Args:
        image (np.ndarray): Input image, any integer dtype, HxW or HxWxC.
        severity (float):   Corruption severity in [0, 1]. 0 returns the image
                            unchanged, 1 is worst-case (widest) blur.

    Returns:
        np.ndarray: Blurred image, same shape and dtype as the input.
    """
    # Clamp Severity to the Valid Range.
    severity = _clampSeverity(severity)

    # Return the Original Image Unchanged when Severity is Zero.
    if severity == 0.0:
        return image

    # Interpolate Kernel Size from Severity. Must be a Positive Odd Integer for CV2.
    kernel_size = int(round(1 + severity * (DEFOCUS_MAX_KERNEL - 1)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # A Kernel Size of 1 is a No-Op, Guard Against Wasted Work.
    if kernel_size <= 1:
        return image

    # Apply Gaussian Blur.
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                   CORRUPTION 3 -- COLOUR QUANTISATION
# ------------------------------------------------------------------------------
def applyColorQuantization(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Applies bit-depth reduction (colour/intensity quantisation), with the
    number of retained levels decreasing continuously as severity increases.

    Args:
        image (np.ndarray): Input image, any integer dtype, HxW or HxWxC.
        severity (float):   Corruption severity in [0, 1]. 0 returns the image
                            unchanged (full bit depth), 1 is worst-case
                            (heaviest quantisation).

    Returns:
        np.ndarray: Quantised image, same shape and dtype as the input.
    """
    # Clamp Severity to the Valid Range.
    severity = _clampSeverity(severity)

    # Return the Original Image Unchanged when Severity is Zero.
    if severity == 0.0:
        return image

    # Interpolate Effective Bit Depth from Severity.
    bits = QUANT_BITS_CLEAN + severity * (QUANT_BITS_WORST - QUANT_BITS_CLEAN)
    levels = max(2.0, 2.0 ** bits)

    # Preserve Original dtype for the Final Cast.
    original_dtype = image.dtype
    image_f = image.astype(np.float32)

    # Quantise Each Channel to the Reduced Number of Levels.
    step = 256.0 / levels
    quantized = np.floor(image_f / step) * step + step / 2.0

    # Clip Back into Valid Range & Restore Original dtype.
    return np.clip(quantized, 0, 255).astype(original_dtype)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           CORRUPTION REGISTRY
# ------------------------------------------------------------------------------
# Maps a Filesystem-Safe Corruption Name to its Corresponding Function.
CORRUPTION_REGISTRY = {
    "iso_noise":         applyIsoNoise,
    "defocus_blur":      applyDefocusBlur,
    "color_quantization": applyColorQuantization,
}
# ------------------------------------------------------------------------------
