# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: tracking.py %
# Date Created:            %date_created: Wed Jul 16 23:14:21 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Object tracking utilities used to associate detections across consecutive
# video frames. Provides a simple motion-compensated multi-frame tracker that
# links object detections into persistent tracks using estimated frame-to-frame
# camera transforms, predicted object motion, and heuristic association metrics.
# Also includes utilities for preparing tracking results for submission and
# evaluation.
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

import numpy as np
import pickle
import os
from tqdm import tqdm
import copy
import json
import math

from . import config
from . import common_utils
from .prediction_structure import *


# ------------------------------------------------------------------------------
#                     FRAME TRANSFORMATION UTILITIES
# ------------------------------------------------------------------------------
def build_transform(frame_items):
    """
    Construct the Affine Transform Between Consecutive Frames.

    Args:
        frame_items: FrameItems instance containing the estimated motion
            parameters.

    Returns:
        np.ndarray: Affine transformation matrix describing the previous-to-
        current frame mapping.
    """
    # Frame Dimensions.
    w = 2448
    h = 2048

    # Build the Affine Camera Motion Transform.
    prev_tr = common_utils.build_geom_transform(
        dst_w=w,
        dst_h=h,
        src_center_x=w / 2 + frame_items.transform_dx,
        src_center_y=h / 2 + frame_items.transform_dy,
        scale_x=1.0,
        scale_y=1.0,
        angle=frame_items.transform_angle,
        return_params=True
    )

    # Return the Estimated Transform.
    return prev_tr


def track_items_trivial(flight_items):
    """
    Apply a Simple Per-Frame Tracking Strategy.

    Args:
        flight_items: Dictionary containing per-frame detections.

    Returns:
        Dictionary containing updated frame detections.
    """
    for frame_num, frame_items in flight_items.items():
        # Reset Frame Detection Counter.
        nb_frame_detections = 0

        for detected_object in frame_items.predicted:
            # Disable Submission Until the Detection Passes Quality Checks.
            detected_object.add_to_submit = False

            # Ignore Distant Objects.
            if detected_object.distance > 800:
                continue

            # Ignore Low-Confidence Detections.
            if detected_object.confidence < 0.8:
                continue

            # Assign a Temporary Per-Frame Track Identifier.
            detected_object.add_to_submit = True
            detected_object.track_id = nb_frame_detections + 1
            nb_frame_detections += 1

    # Return the Flight Items.
    return flight_items
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                          SIMPLE OFFSET TRACKER
# ------------------------------------------------------------------------------
class SimpleOffsetTracker:
    """
    Motion-compensated multi-frame object tracker.

    Associates detections across consecutive frames by projecting previous
    detections into the current frame using the estimated camera motion and
    comparing object position, size, confidence, and predicted motion.

    Confirmed tracks are assigned persistent track identifiers suitable for
    evaluation and submission.
    """

    def __init__(self, min_track_size=8, threshold_to_find=0.8, threshold_to_continue=0.7,
                threshold_distance=32, min_distance=850,
                max_coast_frames=3, coast_decay=0.9):
        """
        Initialise the Tracking Parameters.

        Args:
            min_track_size: Minimum track length before submission.
            threshold_to_find: Confidence required to initialise or extend tracks.
            threshold_to_continue: Minimum confidence required for association.
            threshold_distance: Maximum association distance.
            min_distance: Ignore objects beyond this range.
            max_coast_frames: Maximum number of consecutive coasted frames.
            coast_decay: Confidence multiplier applied during coasting.
        """
        # Store Detection History for Previous Frames.
        self.prev_frames_items: List[List[DetectedItem]] = [[]]

        # Configure Tracking Parameters.
        self.min_track_size = min_track_size
        self.threshold_to_find = threshold_to_find
        self.threshold_to_continue = threshold_to_continue
        self.threshold_distance = threshold_distance
        self.track_ids_in_use = set()
        self.min_distance = min_distance
        self.max_coast_frames = max_coast_frames
        self.coast_decay = coast_decay


    # --------------------------------------------------------------------------
    #                         Track ID Management
    # --------------------------------------------------------------------------
    def allocate_track_id(self):
        """
        Allocate the Lowest Available Track Identifier.

        Returns:
            int: Unique tracking identifier.
        """
        # Search the Available Track ID Pool.
        for i in range(1, 64):
            # Assign the First Unused Identifier.
            if i not in self.track_ids_in_use:
                self.track_ids_in_use.add(i)
                return i
        
        # Only Occur if All IDs are in Use Simultaneously.
        return 1
    

    def release_track_id(self, track_id):
        """
        Explicitly frees a track ID so it can be reallocated. Must only be
        called once a track is confirmed terminated.
        """
        # Return the Track Identifier to the Available Pool.
        self.track_ids_in_use.discard(track_id)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    #                         Track State Updates
    # --------------------------------------------------------------------------
    def inject_coasted_items(self, coasted_items):
        """
        Insert Coasted Detections into the Tracking History.

        Args:
            coasted_items: List of predicted detections generated through coasting.
        """
        # Ensure at Least One History Buffer Exists.
        if not self.prev_frames_items:
            self.prev_frames_items.append([])

        # Append Coasted Detections to the Most Recent Frame.
        self.prev_frames_items[-1].extend(coasted_items)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    #                         Track Association
    # --------------------------------------------------------------------------
    def distance(self, cur_item: DetectedItem, prev_item: DetectedItem, frames_offset):
        """
        Compute the Association Cost Between Two Detections.

        Args:
            cur_item: Current frame detection.
            prev_item: Candidate detection from a previous frame.
            frames_offset: Number of frames separating the detections.

        Returns:
            float: Association cost. Lower values indicate a better match.
        """
        # Estimate the Detection Scale.
        diag = (cur_item.w ** 2 + cur_item.h ** 2) ** 0.5
        predict_distance = ((cur_item.dx * frames_offset) ** 2 + (cur_item.dx * frames_offset) ** 2) ** 0.5

        # Increase the Search Radius for Longer Motion Predictions.
        diag32 = max(32.0, diag, predict_distance * 0.5) / 32.0

        # Compare Object Scale Between Frames.
        diff_w = abs(math.log2(max(16, cur_item.w) / max(16, prev_item.w)))
        diff_h = abs(math.log2(max(16, cur_item.h) / max(16, prev_item.h)))

        # Predict the Expected Relative Offset.
        offset_x = cur_item.cx + cur_item.dx * frames_offset - prev_item.cx
        offset_y = cur_item.cy + cur_item.dy * frames_offset - prev_item.cy

        # Compute Euclidean Position Error.
        offset = (offset_x**2 + offset_y**2) ** 0.5

        # Combine All Association Penalties into a Single Score.
        res = offset / diag32 + diff_w * 16 + diff_h * 16 + (1.0 - prev_item.confidence) * 8
        return res
    

    def process_frame_detections(self, detections: List[DetectedItem], transform: np.ndarray) -> List[DetectedItem]:
        """
        Associate Detections with Existing Tracks.

        Args:
            detections: Current frame detections.
            transform: Previous-to-current frame affine transform.

        Returns:
            List[DetectedItem]: Confirmed tracked detections.
        """
        # Project Historical Detections into the Current Frame.
        for prev_frame_items in self.prev_frames_items:
            prev_items_pos = np.array([[it.cx, it.cy] for it in prev_frame_items])

            # Skip Empty History Buffers.
            if len(prev_items_pos) == 0:
                continue
            
            prev_items_pos_projected = (transform[:2, :2] @ prev_items_pos.T).T + transform[:2, 2]

            # Apply the Estimated Camera Motion.
            for i, item in enumerate(prev_frame_items):
                item.cx, item.cy = prev_items_pos_projected[i, :]

        # Store Detections That Will Become the New Tracking History.
        items_to_keep = []

        # Work on a Copy to Preserve the Original Detections.
        detections = copy.deepcopy(detections)

        # Track IDs Claimed by a cur_item ALREADY Processed THIS Frame.
        claimed_track_ids_this_frame = set()

        for cur_item in detections:
            # Ignore Low-Confidence Detections.
            if cur_item.confidence < self.threshold_to_continue:
                continue
            
            # Ignore Very Distant Objects.
            if cur_item.distance > self.min_distance:
                continue
            
            # Initialise Runtime Tracking State.
            cur_item.prev_item_idx = None
            cur_item.next_item_idx = None
            cur_item.track_id = -1
            cur_item.items_in_track = 1

            min_distance = 1e10
            prev_item_distances = []

            # Search Up To Three Frames Back for the Best Association.
            for frames_offset in [1, 2, 3]:
                # Stop Searching Once a Suitable Match Is Found.
                if len(self.prev_frames_items) < frames_offset:
                    break

                # Associate with the Best Historical Detection. Excludes Historical Detections Already Claimed THIS Frame.
                prev_item_distances = [
                    self.distance(cur_item, prev_item, frames_offset)
                    if (prev_item.next_item_idx is None
                        and prev_item.track_id not in claimed_track_ids_this_frame)
                    else 1000
                    for prev_item in self.prev_frames_items[-frames_offset]
                ]

                # Only Extend Track Length Using Real Detections.
                min_distance = min(prev_item_distances) if prev_item_distances else 1e10
                if min_distance < self.threshold_distance * frames_offset:
                    break
            
            # Associate with the Best Historical Detection.
            if min_distance < self.threshold_distance * frames_offset:
                # Retrieve the Best Matching Previous Detection.
                prev_item_idx = int(np.nanargmin(prev_item_distances))
                prev_item = self.prev_frames_items[-frames_offset][prev_item_idx]

                # Link the Previous and Current Detections.
                prev_item.next_item_idx = (frames_offset, len(items_to_keep))
                cur_item.prev_item_idx = (-frames_offset, prev_item_idx)

                # Only Extend Track Length Using Real Detections.
                prev_was_real_evidence = getattr(prev_item, 'source', 'DETECTED') == 'DETECTED'

                if cur_item.confidence > self.threshold_to_find and prev_was_real_evidence:
                    cur_item.items_in_track = prev_item.items_in_track + 1
                else:
                    cur_item.items_in_track = prev_item.items_in_track

                # Reuse the Existing Track Identifier.
                cur_item.track_id = prev_item.track_id

                # Promote Mature Tracks to Submission.
                if cur_item.items_in_track > self.min_track_size:
                    cur_item.add_to_submit = True

                    # Allocate a New Track Identifier When Required.
                    if cur_item.track_id < 0:
                        cur_item.track_id = self.allocate_track_id()

                # Claim this track_id for the Remainder of the Current Frame.
                if cur_item.track_id >= 0:
                    claimed_track_ids_this_frame.add(cur_item.track_id)

            # Add the Detection to the Tracking History.
            items_to_keep.append(cur_item)

        # Store the Updated Frame History.
        self.prev_frames_items.append(items_to_keep)

        # Retain Only the Most Recent Three Frames.
        self.prev_frames_items = self.prev_frames_items[-3:]

        # Return Confirmed Detections Only.
        return [cur_item for cur_item in items_to_keep if cur_item.add_to_submit]
    # --------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    FLIGHT-LEVEL TRACKING PIPELINE
# ------------------------------------------------------------------------------
def track_items_offset(flight_items):
    """
    Track Objects Across an Entire Flight Sequence.

    Args:
        flight_items: Dictionary containing detections for every frame.

    Returns:
        Dictionary containing tracked detections.
    """
    # Create the Motion-Based Tracker.
    tracker = SimpleOffsetTracker()

    # Track Each Frame Sequentially.
    res = {}

    for frame_num, frame_items in flight_items.items():
        # Build the Frame-to-Frame Transform.
        transform = build_transform(frame_items)
        items_to_submit = tracker.process_frame_detections(frame_items.predicted, transform)
        
        # Replace Raw Detections with Confirmed Tracks.
        frame_items.predicted = copy.deepcopy(items_to_submit)
        res[frame_num] = frame_items

    # Return the Updated Flight Sequence.
    return res
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    SUBMISSION DATA PREPARATION
# ------------------------------------------------------------------------------
def prepare_submission_json(all_items):
    """
    Convert Tracked Detections into Submission Format.

    Args:
        all_items: Dictionary containing tracked detections.

    Returns:
        List[Dict]: Submission-ready prediction structure.
    """
    # Store Submission Entries.
    res = []

    # Process Each Flight Sequence Independently.
    for (part, flight_id), flight_items in all_items.items():
        # Process Each Frame Within the Flight Sequence.
        for frame_num, frame_items in flight_items.items():
            # Store Valid Detections for the Current Frame.
            frame_detections = []

            # Convert Internal Detection Objects into Submission Format.
            for detected_object in frame_items.predicted:
                # Skip Detections Not Selected for Submission.
                if detected_object.add_to_submit:
                    frame_detections.append(dict(
                        x=float(detected_object.cx),
                        y=float(detected_object.cy),
                        w=float(detected_object.w),
                        h=float(detected_object.h),
                        track_id=detected_object.track_id,
                        s=float(detected_object.confidence),
                        n='airborne'
                    ))
            
            # Append the Frame Result to the Submission List.
            res.append({
                'img_name': frame_items.frame_img_fn.split('/')[-1][:-3] + 'png',
                'detections': frame_detections
            })

    # Return the Complete Submission Structure.
    return res
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    TRACKING PIPELINE EXECUTION
# ------------------------------------------------------------------------------
def run_tracking():
    """
    Run the Complete Offline Tracking Pipeline. Loads matched detections, performs
    multi-frame tracking for each flight, generates the submission structure, and
    saves the final prediction file.
    """
    # Load Detection Results.
    with common_utils.timeit_context('load data'):
        matched_items = pickle.load(open(f'{config.DATA_DIR}/pred_matched_oof_100_mix6_2.pkl', 'rb'))

    # Store the Processed Item Matches.
    matched_items_processed = {}

    # Track Every Flight Sequence.
    with common_utils.timeit_context('process tracking'):
        for (part, flight_id), flight_items in tqdm(matched_items.items()):
            matched_items_processed[(part, flight_id)] = track_items_offset(flight_items)

    # Generate Submission Predictions.
    with common_utils.timeit_context('generate results'):
        results_pred = prepare_submission_json(matched_items_processed)

    # Save the Tracking Results.
    with common_utils.timeit_context('save results'):
        os.makedirs('../output/pred_tracking/track_offset/', exist_ok=True)
        with open("../output/pred_tracking/track_offset/result_mix_6.2_offset_0.8-0.7-d32-n8-3steps_v2.json", 'w') as fp:
            json.dump(results_pred, fp)
# ------------------------------------------------------------------------------


# Call the Main Function
if __name__ == '__main__':
    run_tracking()
