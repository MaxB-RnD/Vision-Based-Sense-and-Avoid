# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: prediction_structure.py %
# Date Created:            %date_created: Wed Jul 16 22:32:47 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Data structures used throughout the prediction and evaluation pipeline. 
# Defines lightweight dataclasses representing bounding boxes, ground-truth
# annotations, detection results, and per-frame collections of prediction and
# reference data. These structures provide a consistent interface for object
# localisation, matching, tracking, and frame-level transformation metadata.
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
from typing import List


# ------------------------------------------------------------------------------
#                                  BOX
# ------------------------------------------------------------------------------
@dataclass
class Box:
    """
    Axis-aligned bounding box represented using centre coordinates. The bounding
    box is stored using centre position and dimensions while exposing convenience
    properties for computing the corresponding edge coordinates when required.
    """
    # Centre X Coordinate.
    cx: float

    # Centre Y Coordinate.
    cy: float

    # Bounding Box Width.
    w: float

    # Bounding Box Height.
    h: float


    @property
    def left(self):
        """
        Left edge of the bounding box.
        """
        return self.cx - self.w / 2


    @property
    def right(self):
        """
        Right edge of the bounding box.
        """
        return self.cx + self.w / 2


    @property
    def top(self):
        """
        Top edge of the bounding box.
        """
        return self.cy - self.h / 2


    @property
    def bottom(self):
        """
        Bottom edge of the bounding box.
        """
        return self.cy + self.h / 2
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               GT ITEM
# ------------------------------------------------------------------------------
@dataclass
class GtItem(Box):
    """
    Ground-truth object annotation. Extends the base bounding box with 
    identification information, object distance, and the confidence score of the
    matched prediction.
    """
    # Unique Ground-Truth Object Identifier.
    item_id: str

    # Estimated Object Distance.
    distance: float

    # Confidence Score of the Matched Detection.
    matched_conf: float = 0.0
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                               DETECTED ITEM
# ------------------------------------------------------------------------------
@dataclass
class DetectedItem(Box):
    """
    Predicted object detection. Stores the predicted bounding box together with
    confidence, estimated distance, frame-to-frame motion, tracking information,
    matching state, and submission metadata used throughout the evaluation 
    pipeline.
    """
    # Unique Detection Identifier.
    item_id: str

    # Estimated Object Distance.
    distance: float

    # Detection Confidence Score.
    confidence: float

    # Estimated Horizontal Motion.
    dx: float

    # Estimated Vertical Motion.
    dy: float

    # Indicates Whether the Detection Matches a Planned Ground-Truth Object.
    is_matched_planned: bool = False

    # Indicates Whether the Detection Matches an Unplanned Ground-Truth Object.
    is_matched_unplanned: bool = False

    # Identifier of the Matched Ground-Truth Object.
    matched_item_id: str = ''

    # Assigned Multi-Frame Tracking Identifier.
    track_id: int = -1

    # Indicates Whether the Detection Should Be Included in Submission Output.
    add_to_submit: bool = False

    # Source Producing the Detection.
    source: str = 'DETECTED'

    # Number of Consecutive Frames the Detection Has Been Coasted.
    coast_count: int = 0
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              FRAME ITEMS
# ------------------------------------------------------------------------------
@dataclass
class FrameItems:
    """
    Container holding all prediction and reference data for a single frame.
    Groups together predicted detections, planned and unplanned ground-truth
    objects, associated image filenames, and the estimated frame-to-frame
    affine transformation parameters.
    """
    # Predicted Object Detections.
    predicted: List[DetectedItem]

    # Planned Ground-Truth Objects.
    gt_planned: List[GtItem]

    # Unplanned Ground-Truth Objects.
    gt_unplanned: List[GtItem]

    # Current Frame Image Filename.
    frame_img_fn: str = ''

    # Previous Frame Image Filename.
    frame_img_prev_fn: str = ''

    # Estimated Horizontal Translation.
    transform_dx: float = 0

    # Estimated Vertical Translation.
    transform_dy: float = 0

    # Estimated Rotation Angle.
    transform_angle: float = 0

    # Estimated Transformation Error Metric.
    transform_error: float = 0
# ------------------------------------------------------------------------------
