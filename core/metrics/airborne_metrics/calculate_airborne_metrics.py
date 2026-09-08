# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""This module computes airborne level metrics.
The module assumes the following inputs available :
1) Ground truth enriched with encounters data frame (artifact of calculate_encounters.py)
2) Ground truth vs. Detection matches data frame (artifact of match_groundtruth_results.py)
3) Below/Mixed/Above horizon indicator for encounter (Below == -1, Above == 1, Mixed  = (-1,1))
The module saves an artifact that provides detection information for each encounter:
1) The range at which the encounter was detected 
2) The latency (in frames) it took to detect the encounter
""" 

import argparse
import json
import logging
import os
import re
from functools import partial
from collections import  Counter, OrderedDict

import numpy as np
import pandas as pd

from airborne_metrics.calculate_encounters import exclude_encounters_from_evaluation, DEFAULT_MIN_DETECTION_RANGE_M, ENC_END_RANGE_SCALE 
from airborne_metrics.pandas_utils import is_in_df_columns 
from airborne_metrics.script_utils import setup_logging

RESULTS_NAME_PREFIX_ = 'airborne_metrics'
DEFAULT_MIN_DETECTION_SCORE_ = 0.0 # minimum possible score of detection algorithm
DEFAULT_MAX_DETECTION_SCORE_ = 1.0 # maximum possible score of detection algorithm
DEFAULT_TEMPORAL_WINDOW_FOR_FL_DR_ = 30 # frames to consider for frame level detection rate calculation
DEFAULT_MIN_FL_DR_ = 0.5 # that means that within 30 frames at least 15 should be detected
DEFAULT_METRICS_VALUE_PRECISION_ = 5 # required for binary search for target value of target metrics
SCORE_PRECISION_ = 0.00001 # required for binary search for target value of target metrics
MINS_PER_FLIGHT = 2 # each flight has two minutes 
MINS_TO_HOURS = 1 / 60 
TARGET_FAR_PER_HOUR = 1 / 2 # 1 False Alarm per 2 hours
RANGES_TO_DETECT = [300] # the metrics is reported for detections before 300 m
FAST_DETECTION_WINDOW_ = 30 # frames - an encounter is detected fast if it is detected within 3 secs.

# New Metric Constants.
DEFAULT_CEDR_MIN_ENCOUNTER_FRAMES_ = 8  # Matches the tracker's own min_track_size confidence gate.
DEFAULT_FTPS_MIN_TRACK_FRAMES_    = 8   # Matches min_track_size, same reasoning as CEDR.
DEFAULT_FTPS_THRESHOLDS_          = [0.50, 0.80, 0.95]   # Headline reportable thresholds.
DEFAULT_FTPS_DEFAULT_THRESHOLD_   = 0.80                 # Day-to-day working threshold.
DEFAULT_FTPS_AVERAGE_RANGE_       = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]  # mAP-style 50:5:95.
DEFAULT_INTROSPECTION_FAILURE_THRESHOLD_ = 0.80  # Confidence threshold below which a matched detection is 
                                                 # counted as an introspection failure.
DEFAULT_MIN_MATCHED_FRAMES_FOR_DRIFT_ = 5   # Minimum number of matched frames required before 
                                            # calculating drift metrics, preventing unreliable 
                                            # measurements from very short track overlaps.
DEFAULT_EARLY_TERMINATION_MARGIN_ = 3       # A track ending within this many frames of its real object's last 
                                            # appearance is explained by normal coast-then-drop behaviour, not 
                                            # a genuine failure.
DEFAULT_REACQUISITION_GRACE_PERIOD_ = 3     # A track that terminates but whose real object is picked back up 
                                            # by ANY track (same or new track_id) within this many frames is a 
                                            # successful reacquisition, not a premature-termination failure.
DEFAULT_FRAME_RATE_HZ_ = 10.0               # Matches FAST_DETECTION_WINDOW_ (30 frames == 3 secs) and the
                                            # visualiser's VIDEO_FPS.

# Frame level detection rate calculation constants
DEFAULT_MIN_OBJECT_AREA = 200 # minimum object area for frame level 
NOT_BIRD_QUERY = 'id.str.contains("Flock") == False and id.str.contains("Bird") == False'
GT_ABOVE_AREA_QUERY = 'gt_area > {min_area}'
GT_BELOW_AREA_QUERY = 'gt_area <= {min_area}'
DEFAULT_OBJECT_OF_INTEREST_QUERY = GT_ABOVE_AREA_QUERY + ' and ' + NOT_BIRD_QUERY

# QUERIES 
HORIZON_QUERIES = {'Below Horizon': 'is_above_horizon == -1', 
                   'Mixed': '-1 < is_above_horizon < 1',
                   'Above Horizon': 'is_above_horizon == 1',
                   'All': 'is_above_horizon == is_above_horizon'}
FALSE_DETECTION_QUERY = 'gt_det_no_match == 1'
POSITIVE_DETECTION_QUERY = 'gt_det_match == 1'
PLANNED_INTRUDERS = 'range_distance_m == range_distance_m' # that is range is a number (not NaN)
NON_PLANNED_INTRUDERS = 'range_distance_m != range_distance_m' # that is range is NaN
NON_PLANNED_AIRCRAFT = NON_PLANNED_INTRUDERS + ' and ' + NOT_BIRD_QUERY


log = logging.getLogger(__name__)

def add_flags(parser):
    """Utility function adding command line arguments to the parser"""
    # input files 
    parser.add_argument('--groundtruth-results-matches-filename', '-m', required=True, 
                        help='Path to the ground truth and detection matches data frame in .csv format')
    parser.add_argument('--encounters-with-groundtruth-filename', '-e', required=True, 
                        help='Path to the ground truth enriched with encounters data frame in .csv format')
    # output  
    parser.add_argument('--output-dir-path', '-o',
                        help='Desired folder to save the output data frame with '
                        'match/no match between groundtruth and detections')
    parser.add_argument('--results-name-prefix', type=str, default=RESULTS_NAME_PREFIX_,
                        help='Prefix for results filename')
    parser.add_argument('--save-intermediate-results', default=False, action='store_true',
                        help='Specify this if saving intermediate data frame with encounters and '
                        'corresponding moving frame level detection rate is needed')
    # working point parameters 
    parser.add_argument('--min-object-area', type=float, default=DEFAULT_MIN_OBJECT_AREA,
                        help='The minimum object area for average frame level detection rate calculation')
    parser.add_argument('--min-enc-range-upper-bound', type=float,  
                        default=ENC_END_RANGE_SCALE * DEFAULT_MIN_DETECTION_RANGE_M,
                        help='The minimum range of the encounter should be not less than this value, '
                        'default is {}'.format(ENC_END_RANGE_SCALE * DEFAULT_MIN_DETECTION_RANGE_M))
    parser.add_argument('--max-enc-range-lower-bound', type=float, 
                        help='The maximum range of the encounter should be not less than this value')
    parser.add_argument('--target-metrics', '-t', type=str, choices=['far', 'fppi', 'fl_dr'], 
                       help='Provide metrics, FAR or FPPI or FL_DR (frame-level detection rate), '
                       'to determine a working point. This is useful when comparing to other algorithms. '
                       'If None is provided, detection score threshold (default = 0) will be used')
    parser.add_argument('--target-value', '-f', type=float, 
                       help='Provide the value for the expected target metrics (if chosen). '
                       'The default target values is calculated if target metrics is FAR. '
                       'If target metrics is FPPI or FL_DR and target values is None - error will be thrown')
    parser.add_argument('--target-value-precision', type=int, default=DEFAULT_METRICS_VALUE_PRECISION_,
                       help='Precision with which to calculate targeted value. Provide this value '
                       'if you want the metrics to calculate the score based on specific target metrics')
    parser.add_argument('--min-det-score', type=float, default=DEFAULT_MIN_DETECTION_SCORE_,
                        help='Minimum possible detection score. Provide this value if you want '
                       'the metrics to calculate the score for working point based on target metrics')
    parser.add_argument('--max-det-score', type=float, default=DEFAULT_MAX_DETECTION_SCORE_,
                        help='Maximum possible detection score. Provide this value if you want '
                       'the metrics to calculate the score for working point based on target metrics')
    parser.add_argument('--detection-score-threshold', '-s', type=float, 
                        default=DEFAULT_MIN_DETECTION_SCORE_,
                        help='Detection score threshold for working point')
    # parameters 
    parser.add_argument('--use-track-fl-dr', action='store_true',
                        help='Setting up this flag will require the same track_id in detections that ' 
                        'contribute to the encounter level detection rate calculation')
    parser.add_argument('--fl-dr-temporal-win', type=int, default=DEFAULT_TEMPORAL_WINDOW_FOR_FL_DR_,  
                        help='Temporal window for moving frame level detection rate')
    parser.add_argument('--raw-results-dir', default=None,
                        help='Path to the run directory containing one result.json per flight '
                        '(NOT --groundtruth-results-matches-filename\'s stitched/metrics-format '
                        'file). When provided, each failed track\'s failure_onset_frame and '
                        'warning_lead_time_seconds are written back into that flight\'s raw '
                        'result.json for visualisation. Skipped entirely if omitted.')
    parser.add_argument('--min-fl-dr', type=float, default=DEFAULT_MIN_FL_DR_,
                         help='Minimum frame level detection rate within the temporal window')

def _assert_non_negative(value, value_name):
    """assertion helper"""
    assert value >= 0.0, ('{} is expected to be non-negative'.format(value_name))

def _assert_strictly_positive(value, value_name):
    """assertion helper"""
    assert value > 0.0, ('{} is expected to be strictly positive, but received {}'.format(
                                                                        value_name, value))

def check_flags(flags):
    """Utility function to check the input"""
    _assert_non_negative(flags.min_det_score, 'Minimum detection score')
    _assert_non_negative(flags.max_det_score, 'Maximum detection score')
    _assert_non_negative(flags.detection_score_threshold, 'Detection score threshold')
    assert (flags.target_value is not None or flags.target_metrics is None 
        or flags.target_metrics == 'far'), (
                'If target-metrics is specified as fppi or fl_dr, target-value should be provided ')
    if flags.target_value is not None:
        _assert_non_negative(flags.target_value, 'Target value')
        _assert_strictly_positive(flags.target_value_precision, 'Target value precision')
    _assert_strictly_positive(flags.fl_dr_temporal_win, 
                                'Temporal window for moving frame level detection rate')
    assert 0 < flags.min_fl_dr <= 1.0, (
                        'Minimum frame level detection rate should be in (0,1] range')
    assert flags.output_dir_path is None or not os.path.isfile(flags.output_dir_path), (
                                                 'Directory name is expected as output path')
    assert (flags.groundtruth_results_matches_filename.endswith('.csv') and 
            flags.encounters_with_groundtruth_filename.endswith('.csv')), (
            'Unsupported file format, please provide .csv produced by calculate_encounters.py and '
            'match_groundtruth_results.py')

##########################################################
# Frame-level FPPI and PD and strict FAR  calculation code
##########################################################

def _calc_num_no_match_detections(df_matches):
    """helper to calculate number of not matched detection within the provided data frame of matches"""
    log.info('Calculating the number of detections that did not match ground truth') 
    if not is_in_df_columns(df_matches, ['detection_id']):
        # this is an edge case when there are 0 detections
        return 0.0, 0, 0
    assert is_in_df_columns(df_matches, ['detection_id', 'gt_det_no_match']), (
        'One or more of detection_id, gt_det_no_match columns is not found, cannot calculate')
    # Grouping by 'detection_id' yields all the match results between the detection with 
    # this specific id and a group of ground truth intruders that were evaluated for a match 
    # (within the same frame).
    # gt_det_no_match will be 1 if the detection does not match the specific intruder
    # if minimum of 'gt_det_no_match' over the group of intruders equals 1, it means that the 
    # detection did not match any of those intruders, hence it is a false detection    
    df_detections = df_matches[['detection_id', 'gt_det_no_match']].groupby(
                                            'detection_id')['gt_det_no_match'].min().reset_index(0)
    number_of_detections = df_matches['detection_id'].nunique()
    assert number_of_detections == len(df_detections), ('something went wrong with grouping detections, '
                        'expected {}, but got {}'.format(number_of_detections, len(df_detections)))

    num_no_match_detections = len(df_detections.query(FALSE_DETECTION_QUERY)) 
    log.info('No match calculation: Number of detections without a match = %d out of %d '
            'unique detections', num_no_match_detections, number_of_detections)
    return num_no_match_detections

def _calc_num_unique_track_ids_with_no_match_detection(df_matches):
    """helper to calculate number of unique tracks that correspond to at least one not matched detection"""
    log.info('Calculating the number of unique tracks ids that that correspond to at least one '
            'not matched detection') 
    if not is_in_df_columns(df_matches, ['detection_id']):
        # this is an edge case when there are 0 detections
        return 0.0, 0, 0
    required_cols = ['flight_id', 'detection_id', 'track_id', 'gt_det_no_match']
    assert is_in_df_columns(df_matches, required_cols), (
        'One or more of detection_id, track_id, gt_det_no_match columns is not found, cannot calculate')
    
    # Grouping by 'detection_id' yields all the match results between the detection with 
    # this specific id and a group of ground truth intruders that were evaluated for a match 
    # (within the same frame).
    # gt_det_no_match will be 1 if the detection does not match the specific intruder
    # if minimum of 'gt_det_no_match' over the group of intruders equals 1, it means that the 
    # detection did not match any of those intruders, hence it is a false detection 
    # Only unique track_ids that correspond to false detection are counted    
    df_detections = df_matches[required_cols].groupby('detection_id')[
                ['flight_id', 'track_id', 'gt_det_no_match']].agg({
                'flight_id':'first', 'track_id': 'first', 'gt_det_no_match': 'min'}).reset_index(0)
    num_false_tracks = df_detections.query(FALSE_DETECTION_QUERY).groupby(['flight_id', 'track_id']).ngroups
    log.info('Number of unique track_ids that correspond to at least one false detection %d', 
                                                                                    num_false_tracks)
    return num_false_tracks

def _filter_matches_based_on_detection_score(df_matches, min_score):
    """helper to filter data frame of matches based on detection score"""
    min_score = min_score if min_score is not None else 0    
    if min_score < 0:
        raise ValueError('min_score should be positive or zero or None')
    elif min_score > 0:
        log.info('Filtering score threshold = %.3f', min_score) 
        assert is_in_df_columns(df_matches, ['s']), 's (score) column is not found, cannot filter'
        df_matches_filtered = df_matches.query('s >= {}'.format(min_score))
        return df_matches_filtered
    return df_matches

def compute_false_positives_per_image(df_matches, total_frames_processed=None, min_score=None):
    """Compute FPPI based on a data frame of matches - useful for frame-level metrics 
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        total_frames_processed: int, number of processed flights if different from the number of unique
                frames ('img_name') in the provided df_matches
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, false positives per image
    """
    # determine how many images were processed 
    if total_frames_processed is not None and total_frames_processed <= 0:
        raise ValueError('total_frames_processed should be strictly positive')
    if total_frames_processed is None:      
        log.info('FPPI calculation: Using unique image names in the provided data frame to calculate '
                                                                'total number of processed frames')
        assert is_in_df_columns(df_matches, ['img_name']), 'img_name column is not found, cannot calculate'
        total_frames_processed = df_matches['img_name'].nunique()
    log.info('FPPI calculation: Total number of processed frames is %d', total_frames_processed)
    df_matches = _filter_matches_based_on_detection_score(df_matches, min_score)
    fppi = _calc_num_no_match_detections(df_matches) / total_frames_processed
    log.info('FPPI = %.5f', fppi)
    return fppi

def compute_false_alarms_per_hour(df_matches, total_flights_processed=None, min_score=None):
    """Compute strict FAR based on a data frame of matches, based on the following definition
    Overall False Alarm Rate (strict FA) - a number of unique reported track ids, 
    which correspond to at least one false positive cluster, divided by total number of hours 
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        total_flights_processed: int, number of processed flights if different from the number of unique
                flights ('flight_id') in the provided df_matches
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, false alarms per hour
    """
    # determine how many images were processed 
    if total_flights_processed is not None and total_flights_processed <= 0:
        raise ValueError('total_flights_processed should be strictly positive')
    if total_flights_processed is None:      
        log.info('FAR calculation: Using unique flight ids in the provided data frame to calculate '
                                                                'total number of processed flights')
        assert is_in_df_columns(df_matches, ['flight_id']), 'flight_id column not found, cannot calculate'
        total_flights_processed = df_matches['flight_id'].nunique()
    total_hours_processed = total_flights_processed * MINS_PER_FLIGHT * MINS_TO_HOURS
    _assert_strictly_positive(total_flights_processed, 'Total processed hours')
    log.info('FAR calculation: Total number of processed flights is %d', total_flights_processed)    
    log.info('FAR calculation: Total number of processed hours is %.3f', total_hours_processed)
   
    df_matches = _filter_matches_based_on_detection_score(df_matches, min_score)

    num_false_tracks =_calc_num_unique_track_ids_with_no_match_detection(df_matches)
    far = num_false_tracks / total_hours_processed
    log.info('FAR = %.5f', far)
    return far

def _calc_num_detected_intruders(df_matches):
    """helper to calculate number of detected intruders"""
    log.info('Calculating the number of intruders that were matched by detections') 
    if not is_in_df_columns(df_matches, ['detection_id']):
        # this is an edge case when there are 0 detections
        return 0.0, 0, 0
    assert is_in_df_columns(df_matches, ['detection_id', 'id', 'gt_det_match']), (
        'One or more of detection_id, id, gt_det_match columns is not found, cannot calculate')
    
    # When grouping by 'img_name', 'id' we get all the match results between the specific 
    # intruder/object (within specific frame) and a group of detections that were evaluated for a match 
    # "gt_det_match" will be 1 if a detection does match the specific intruder
    # if maximum of 'gt_det_match' over the group of detections equals 1, it means that the at least
    # one detection matched the intruder and hence the intruder is detected 
    df_intrudes = df_matches[['img_name', 'id', 'gt_det_match']].groupby(
                                    ['img_name', 'id'])['gt_det_match'].max().reset_index(0)
    num_detected_intruders = len(df_intrudes.query(POSITIVE_DETECTION_QUERY)) 
    log.info('Detected intruders calculation: Number of detected intruders = %d ', num_detected_intruders)
    return num_detected_intruders

def compute_probability_of_detection(df_matches, min_score=None):
    """Compute frame-level PD of valid intruders - useful for frame-level metrics.
    This function does NOT assume planned intruders 
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, frame-level probability of detection
    """
    tot_objects_to_detect = df_matches.groupby(['img_name', 'id']).ngroups
    if tot_objects_to_detect == 0:
        return 0.0, 0, 0
    log.info('PD calculation: Number of intruders to detect = %d', tot_objects_to_detect)

    df_matches = _filter_matches_based_on_detection_score(df_matches, min_score)

    # each intruder/object is identified by img_name and id 
    # gt_det_match will be 1 if there is match between some detection and this object
    # grouping by 'img_name', 'id' will create a group with all the matches to all the detections
    # in this frame (img_name) for this specific object 
    # if there is at least one detection that matches this objects the maximum of gt_det_match equals 1
    num_matched_objects = _calc_num_detected_intruders(df_matches) 
    pd = num_matched_objects / tot_objects_to_detect
    log.info('PD = %.3f = %d / %d', pd, num_matched_objects, tot_objects_to_detect)
    return pd, num_matched_objects, tot_objects_to_detect

def _get_planned_intruders_in_range(df_matches, min_range, max_range=None):
    """helper to get only planned intruders in range"""
    if max_range is not None:
        assert is_in_df_columns(df_matches, ['range_distance_m']), (
                            'range_distance_m column is not found - cannot filter based on range')
        df_matches_in_range = df_matches.query(
                                        '{} <= range_distance_m <= {}'.format(min_range, max_range))
    else:
        # evaluate against planed intruders only
        df_matches_in_range = df_matches.query('range_distance_m != range_distance_m')
    return df_matches_in_range

def compute_probability_of_detection_of_planned_intruders(df_matches, max_range=None, min_range=0,
                                                         min_score=None):
    """Compute frame-level PD of PLANNED intruders - useful for frame-level metrics.
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        max_range: float, maximum range of intruder to consider for evaluation, if not provided all
                planned intruders with valid range are used 
        min_range: float, minimum range of intruder to consider for evaluation, if not provided all
                planned intruders with valid range are used 
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, probability of detection
    """
    log.info('PD calculation: Intruders Range =  [%.1f, %.1f]', min_range, max_range)
    df_matches_in_range = _get_planned_intruders_in_range(df_matches, min_range, max_range)  
    return compute_probability_of_detection(df_matches_in_range, min_score)

def compute_probability_of_detection_small_objects(df_matches, min_area=DEFAULT_MIN_OBJECT_AREA, 
                                    min_score=None, obj_query = DEFAULT_OBJECT_OF_INTEREST_QUERY):
    """Compute frame-level PD of PLANNED intruders - useful for frame-level metrics.
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        min_area: float, minimum area of intruder to consider for evaluation, default is 300
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, probability of detection

    """
    aircraft_area_query = obj_query.format(min_area=min_area)
    log.info('PD calculation: %s', aircraft_area_query)
    df_matches_area = df_matches.query(aircraft_area_query)
    return compute_probability_of_detection(df_matches_area, min_score)

##########################################################
# Airborne-level PD calculation code
##########################################################

def get_valid_encounters(df_encounters):
    return df_encounters.query('is_valid_encounter == True')

def combine_encounter_with_gt_det_matches(df_matches, df_encounters):
    """Combines two data frames based on encounter identifier
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results
        df_encounters: pd.DataFrame, data frame with ground truth and and encounter info
    Returns:
        pd.DataFrame, combined data frame
    """
    cols_to_combine_on = ['flight_id', 'img_name', 'frame', 'id']
    required_columns_str = ', '.join(cols_to_combine_on) 
    assert is_in_df_columns(df_matches, cols_to_combine_on), (
        'One or more out of {} columns is not found in data frame of matches, '
                                'cannot combine'.format(required_columns_str))
    assert is_in_df_columns(df_encounters, cols_to_combine_on), (
        'One or more out of {} columns is not found in data frame of encounters, '
                                'cannot combine'.format(required_columns_str))
    df = df_encounters.merge(df_matches, on=cols_to_combine_on, 
            how='left', suffixes=['_orig', '']).sort_values(by =['encounter_id','img_name', 'frame'])
    return df

def augment_with_moving_frame_level_detection_rate_per_encounter(df_matches, temporal_window):
    """adds moving frame level detection rate per encounter, based on the provided temporal_window"""

    required_columns_str = ', '.join(['encounter_id', 'gt_det_match']) 
    assert is_in_df_columns(df_matches, ['encounter_id', 'gt_det_match']), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    df_matches = df_matches.fillna(value={'gt_det_match': 0})
    df_with_moving_fl_dr = df_matches.groupby(['encounter_id'])['gt_det_match'].rolling(
                                                temporal_window).apply(np.mean).reset_index()
    df_with_moving_fl_dr = df_with_moving_fl_dr.rename(columns={'gt_det_match': 'fl_dr'})
    return df_with_moving_fl_dr

def augment_with_moving_most_common_track_id_count_per_encounter(df_matches, temporal_window):
    """adds moving frame level detection rate per encounter, based on the provided temporal_window"""

    required_columns_str = ', '.join(['encounter_id', 'matched_track_id']) 
    assert is_in_df_columns(df_matches, ['encounter_id', 'matched_track_id']), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    
    def get_most_common_freq(all_track_ids): 
        all_track_ids_list = []
        for element in all_track_ids:
            all_track_ids_list.extend(element)
        track_id_counter = Counter(all_track_ids_list)
        most_common = track_id_counter.most_common(2) # taking 2 most common track ids 
                                        # 2 because one can be -1, and is not counted for detection
        if  most_common[0][0] != -1:
            most_common_freq = (most_common[0][1] / temporal_window) 
        else:
            if len(most_common) > 1:
                most_common_freq = (most_common[1][1] / temporal_window) 
            else:
                most_common_freq = 0
        return most_common_freq

    def my_rolling_apply_char(frame, window, func):
        index = frame.index
        values = [0 if i + 1 - window < 0 else func(frame.iloc[i + 1 - window : i + 1]) for 
                                                                        i in range(0, len(frame))]
        return pd.DataFrame(data={'track_fl_dr': values}, index=index).reindex(frame.index)

    df_same_track_id_count = df_matches.groupby('encounter_id')

    df_matched_track_freq = pd.DataFrame(columns=['encounter_id', 'track_fl_dr'])
    for encounter_name, group in df_same_track_id_count: 
        df_res = my_rolling_apply_char(group['matched_track_id'], temporal_window, get_most_common_freq)
        df_res = df_res.assign(encounter_id = encounter_name)
        df_matched_track_freq = pd.concat([df_matched_track_freq, df_res])
    df_matched_track_freq.index.name = 'frame'
    return df_matched_track_freq.reset_index()


# ------------------------------------------------------------------------------
#                CONTINUOUS ENCOUNTER-LEVEL DETECTION RATE (CEDR)
# ------------------------------------------------------------------------------
def augment_with_dominant_track_detection_rate_per_encounter(df, exclude_coasted=True):
    """
    Flags Each Frame of an Encounter as a CEDR Match Based on Track Continuity.

    Unlike augment_with_moving_most_common_track_id_count_per_encounter, this
    is NOT a rolling window. For each encounter, the single track_id that
    matches the object most often across the WHOLE encounter is identified as
    the "dominant" track. A frame is only flagged as a CEDR match if the
    dominant track_id itself produced the match in that frame -- a model that
    detects the object correctly but under several disconnected track_ids
    gets no credit for those fragmented frames, since none of them belongs to
    the single dominant track.

    Args:
        df (pd.DataFrame):
            Combined encounter/ground-truth-match dataframe. Must contain
            'encounter_id' and 'matched_track_id' columns.
        exclude_coasted (bool):
            If True and a 'source' column is present, matches produced by
            COASTED (motion-predicted) frames are excluded from counting
            toward the dominant track's tally. CEDR is intended to measure
            genuine detection coverage, not the tracker's own motion
            assumptions. Models that don't emit a 'source' column (i.e. any
            detect-to-track pipeline with no coasting) are unaffected.

    Returns:
        pd.DataFrame: Same as input, with two additional columns:
            dominant_track_id: The track_id that matched most often within
                                each encounter (-1 if the encounter has no
                                matches at all).
            cedr_match: 1 if this row's frame was matched by the dominant
                        track (and, unless excluded, not coasted), else 0.
    """
    # Confirm the Columns Required for This Augmentation are Present.
    required_cols = ['encounter_id', 'matched_track_id']
    assert is_in_df_columns(df, required_cols), (
        'One Or More Out of {} Columns is Not Found, Cannot Augment'.format(', '.join(required_cols)))
    
    # Ignore Unmatched Rows (matched_track_id == -1) When Finding the
    # Most Common Real Track ID for This Encounter.
    def _dominant_track_id(group):
        ids = [tid for tid in group if tid != -1]
        if not ids:
            return -1
        return Counter(ids).most_common(1)[0][0]

    # Identify the Single Dominant Track ID for Every Encounter.
    dominant = df.groupby('encounter_id')['matched_track_id'].apply(_dominant_track_id)
    dominant = dominant.rename('dominant_track_id').reset_index()
    df = df.merge(dominant, on='encounter_id', how='left')

    # A Frame Counts as a CEDR Match Only if the Dominant Track Produced It.
    is_dominant_match = (df['matched_track_id'] == df['dominant_track_id']) & (df['dominant_track_id'] != -1)

    # Exclude Tracking-to-Detect Coasted Frames from the Numerator, if Requested.
    if exclude_coasted and 'source' in df.columns:
        is_dominant_match = is_dominant_match & (df['source'] != 'COASTED')
    
    # Store the Final Per-Frame CEDR Match Flag.
    df = df.assign(cedr_match=is_dominant_match.astype(int))
    return df


def compute_cedr(df_matches, df_val_encounters, min_score,
                  min_encounter_frames=DEFAULT_CEDR_MIN_ENCOUNTER_FRAMES_):
    """
    Computes Continuous Encounter-Level Detection Rate (CEDR) per Encounter.

    CEDR = (frames matched by the encounter's single dominant track_id,
            excluding coasted frames) / (total frames the object is in view)

    Replaces the windowed, binary detected/not-detected EDR check with a
    continuous per-encounter percentage that also enforces single-track
    continuity -- a model that detects an object correctly but under several
    disconnected track_ids scores lower than one that tracks it continuously
    under one ID.

    Args:
        df_matches (pd.DataFrame):
            Ground-truth/detection match dataframe, must include 's'
            (confidence), 'track_id', and 'gt_det_match' columns.
        df_val_encounters (pd.DataFrame):
            Valid encounters dataframe used to scope frames to their
            encounter_id.
        min_score (float):
            Minimum confidence score required for a detection to count as a
            match. Detections below this threshold are treated as unmatched
            before CEDR is computed.
        min_encounter_frames (int):
            Minimum total encounter length, in frames, required for an
            encounter to be scored by CEDR at all. Defaults to
            DEFAULT_CEDR_MIN_ENCOUNTER_FRAMES_, matching the tracker's own
            min_track_size confidence gate rather than an arbitrary value.
            Encounters shorter than this are excluded entirely -- not scored
            as a 0, simply not applicable.

    Returns:
        pd.DataFrame: One row per qualifying encounter, with columns:
            encounter_id, flight_id, tot_num_frames,
            num_dominant_track_matches, cedr.
    """
    # Work on a Copy to Avoid Mutating the Caller's DataFrame.
    df_matches = df_matches.copy()

    # Treat Any Detection Below min_score as Unmatched.
    below_thresh_det = df_matches['s'] < min_score
    df_matches.loc[below_thresh_det, 'gt_det_match'] = 0

    # Record the Track ID Responsible for Each Match, -1 Where There is None.
    df_matches = df_matches.assign(matched_track_id=[
        track_id if is_match else -1
        for track_id, is_match in zip(df_matches['track_id'], df_matches['gt_det_match'])
    ])

    # Scope Matches to Their Encounters & Flag the Dominant-Track CEDR Matches.
    df = combine_encounter_with_gt_det_matches(df_matches, df_val_encounters)
    df = augment_with_dominant_track_detection_rate_per_encounter(df)

    # Aggregate to One Row per Encounter.
    df_cedr = df.groupby('encounter_id').agg(
        flight_id=('flight_id', 'first'),
        tot_num_frames=('frame', 'count'),
        num_dominant_track_matches=('cedr_match', 'sum'),
    ).reset_index()

    # Drop Encounters Too Short to be Meaningfully Scored.
    df_cedr = df_cedr.query('tot_num_frames >= @min_encounter_frames').copy()

    # Compute the Final Per-Encounter CEDR Percentage.
    df_cedr['cedr'] = df_cedr['num_dominant_track_matches'] / df_cedr['tot_num_frames']

    # Return the Per-Encounter CEDR Results.
    return df_cedr
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                     FALSE TRACKS PER SEQUENCE (FTPS)
# ------------------------------------------------------------------------------
def augment_with_track_unmatched_ratio(df_matches, min_score):
    """
    Computes the Unmatched-Frame Ratio for Every Track.

    Inverts the same pattern track_fl_dr uses for matches -- but computed
    once over each track's ENTIRE lifespan, not a rolling window, since
    FTPS judges whether a whole trajectory was wrong, not a moment of it.

    Multiple ground-truth objects can appear in the same frame, which would
    duplicate a single detection's row once per GT pairing after the outer
    join in match_groundtruth_results.py. This collapses back to one row per
    (flight_id, track_id, frame) before counting, so a track's frame count
    isn't inflated by multi-object frames.

    Args:
        df_matches (pd.DataFrame): Ground-truth/detection match dataframe.
        min_score (float): Detections below this confidence are treated as
                            unmatched, same convention as compute_cedr.

    Returns:
        pd.DataFrame: One row per (flight_id, track_id) with tot_num_frames,
        num_matched_frames, num_unmatched_frames, unmatched_ratio.
    """
    # Work on a Copy & Drop Rows With No Associated Track (Missed GT, No Detection).
    df = df_matches.copy()
    df = df[df['track_id'].notna()]

    # Treat Any Detection Below min_score as Unmatched.
    below_thresh_det = df['s'] < min_score
    df.loc[below_thresh_det, 'gt_det_match'] = 0

    # Collapse Multi-GT-Per-Frame Duplication to One Row per Detection-Frame.
    df_frame_level = df.groupby(['flight_id', 'track_id', 'frame'])['gt_det_match'].max().reset_index()

    # Aggregate to One Row per Track.
    df_tracks = df_frame_level.groupby(['flight_id', 'track_id']).agg(
        tot_num_frames=('frame', 'count'),
        num_matched_frames=('gt_det_match', 'sum'),
    ).reset_index()

    # Compute the Unmatched Ratio for Each Track.
    df_tracks['num_unmatched_frames'] = df_tracks['tot_num_frames'] - df_tracks['num_matched_frames']
    df_tracks['unmatched_ratio'] = df_tracks['num_unmatched_frames'] / df_tracks['tot_num_frames']
    return df_tracks


def compute_ftps(df_matches, min_score, num_sequences,
                  min_track_frames=DEFAULT_FTPS_MIN_TRACK_FRAMES_,
                  thresholds=DEFAULT_FTPS_THRESHOLDS_):
    """
    Computes False Tracks Per Sequence (FTPS) at Multiple Thresholds.

    Replaces raw per-frame FPPI with a track-level count of fully wrong
    trajectories: a track only counts as a false track once its unmatched-
    frame ratio crosses a threshold, so a single stray frame on an otherwise
    correct track is ignored, unlike HFAR's "at least one unmatched frame"
    rule. Reported at multiple thresholds, mAP-IoU-style, rather than a
    single contestable cutoff.

    Tracks shorter than min_track_frames are excluded entirely (not scored
    as a false track by default), matching the tracker's own min_track_size
    confidence gate -- same reasoning as CEDR's minimum encounter length.

    Args:
        df_matches (pd.DataFrame): Ground-truth/detection match dataframe.
        min_score (float): Confidence threshold, forwarded to
                            augment_with_track_unmatched_ratio.
        num_sequences (int): Number of flight sequences evaluated, used as
                              the "Per Sequence" denominator.
        min_track_frames (int): Minimum track length to be judged.
        thresholds (List[float]): Unmatched-ratio thresholds to report
                                   individually (e.g. FTPS@50/80/95).

    Returns:
        Tuple[pd.DataFrame, dict, float]:
            df_eligible: Per-track unmatched ratios for all judged tracks.
            ftps_by_threshold: {threshold: FTPS value} for each of `thresholds`.
            ftps_averaged: Mean FTPS across DEFAULT_FTPS_AVERAGE_RANGE_
                           (50:5:95), mirroring mAP@[.5:.95].
    """
    # Compute Per-Track Unmatched Ratios & Apply the Minimum Track Length Gate.
    df_tracks = augment_with_track_unmatched_ratio(df_matches, min_score)
    df_eligible = df_tracks.query('tot_num_frames >= @min_track_frames').copy()
    num_eligible_tracks = len(df_eligible)

    # Count False Tracks at Each Headline Threshold.
    ftps_rate_by_threshold = {}
    ftps_count_by_threshold = {}
    for thresh in thresholds:
        n_false_tracks = int((df_eligible['unmatched_ratio'] >= thresh).sum())
        ftps_count_by_threshold[thresh] = n_false_tracks
        ftps_rate_by_threshold[thresh] = n_false_tracks / num_sequences if num_sequences else 0.0

    # Compute the Averaged FTPS Across the Full mAP-Style Threshold Range.
    avg_scores = []
    for thresh in DEFAULT_FTPS_AVERAGE_RANGE_:
        n_false_tracks = int((df_eligible['unmatched_ratio'] >= thresh).sum())
        avg_scores.append(n_false_tracks / num_sequences if num_sequences else 0.0)
    ftps_averaged = float(np.mean(avg_scores)) if avg_scores else 0.0

    return df_eligible, ftps_rate_by_threshold, ftps_count_by_threshold, ftps_averaged, num_eligible_tracks
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           INTROSPECTION METRICS
# ------------------------------------------------------------------------------
def augment_with_failure_onset_frame(df_matches, min_score,
                                      failure_threshold=DEFAULT_INTROSPECTION_FAILURE_THRESHOLD_,
                                      early_termination_margin=DEFAULT_EARLY_TERMINATION_MARGIN_,
                                      reacquisition_grace_period=DEFAULT_REACQUISITION_GRACE_PERIOD_):
    """
    Computes the Failure Onset Frame per Track, using two failure types.

    For sustained failures, the Failure Onset Frame is the earliest frame t0
    such that the trailing unmatched-ratio, computed from t0 to the track's
    final frame, never again drops below failure_threshold. For early
    termination, it is the track's final submitted frame when the
    ground-truth object remains present beyond early_termination_margin AND
    is not successfully reacquired (by any track) within
    reacquisition_grace_period frames of that final submission.

    Args:
        df_matches (pd.DataFrame): Ground-truth/detection match dataframe, must 
            include 's' (confidence), 'gt_det_match', 'track_id', 'flight_id', 
            and 'frame' columns.

        min_score (float): Minimum confidence score required for a detection to 
            count as a match. Detections below this threshold are treated as 
            unmatched before onset is computed.

        failure_threshold (float): Unmatched-ratio threshold above which a track 
            is considered a genuine failure.

        early_termination_margin (int): Minimum frame gap between a track's last
            submission and the ground-truth object's last appearance to count as
            a CANDIDATE early termination (before the reacquisition check below).

        reacquisition_grace_period (int): Number of frames after a track's last
            submission during which the same ground-truth object may be picked
            back up by any track (same or new track_id) and still count as a
            successful reacquisition rather than a failure.

    Returns:
        pd.DataFrame: One row per failed track (either type), with columns
            flight_id, track_id, failure_onset_frame, tot_num_frames,
            failure_type ('sustained' or 'early_termination').
    """
    # Work on a Copy & Treat Any Detection Below min_score as Unmatched.
    df = df_matches.copy()
    below_thresh_det = df['s'] < min_score
    df.loc[below_thresh_det, 'gt_det_match'] = 0

    # Collapse Multi-GT-Per-Frame Duplication to One Row per Detection-Frame.
    df_frame_level = df.groupby(['flight_id', 'track_id', 'frame'])['gt_det_match'].max().reset_index()
    df_frame_level = df_frame_level.sort_values(['flight_id', 'track_id', 'frame'])

    # Initialise Storage for Onset Rows.
    sustained_rows = []
    sustained_track_ids = set()

    # Compute Sustained Failure Onset per Track.
    for (flight_id, track_id), group in df_frame_level.groupby(['flight_id', 'track_id']):
        # Get Frames & GT Matches.
        frames = group['frame'].tolist()
        matches = group['gt_det_match'].tolist()
        n = len(frames)

        # Skip Tracks Below the Failure Threshold.
        overall_unmatched_ratio = 1 - (sum(matches) / n)
        if overall_unmatched_ratio < failure_threshold:
            # Not a Genuine Failure.
            continue

        # Scan Backward to Find the Earliest Failure Frame.
        onset_idx = n - 1
        for i in range(n - 1, -1, -1):
            # Calculate Suffix Unmatched Ratio.
            suffix = matches[i:]
            suffix_unmatched_ratio = 1 - (sum(suffix) / len(suffix))

            # Stop When the Suffix Drops Below the Threshold.
            if suffix_unmatched_ratio >= failure_threshold:
                onset_idx = i
            else:
                break

        # Record the Sustained Failure Onset.
        sustained_rows.append({
            'flight_id': flight_id,
            'track_id': track_id,
            'failure_onset_frame': frames[onset_idx],
            'tot_num_frames': n,
            'failure_type': 'sustained',
        })
        sustained_track_ids.add((flight_id, track_id))

    # Last Frame Each Track Was Submitted.
    df_submitted = df[df['track_id'].notna()]
    track_last_frame = df_submitted.groupby(['flight_id', 'track_id'])['frame'].max()

    # First Frame Each Track Was Submitted.
    track_first_frame = df_submitted.groupby(['flight_id', 'track_id'])['frame'].min()

    # Find the Dominant GT Object Matched by Each Track.
    df_real_matches = df_submitted[df_submitted['gt_det_match'] == 1]
    dominant_gt = (
        df_real_matches.groupby(['flight_id', 'track_id', 'id'])
        .size()
        .reset_index(name='match_count')
        .sort_values('match_count', ascending=False)
        .drop_duplicates(subset=['flight_id', 'track_id'])
        .set_index(['flight_id', 'track_id'])['id']
    )

    # Last Frame Each GT Object Is Present.
    true_object_last_frame = df.dropna(subset=['id']).groupby(['flight_id', 'id'])['frame'].max()

    # Frames on Which Each GT Object was Genuinely Matched by ANY Track. 
    matched_frames_by_gt = df_real_matches.groupby(['flight_id', 'id'])['frame'].apply(
        lambda frames: sorted(set(frames))
    )

    # Store Early Termination Failure Rows.
    early_term_rows = []

    # Check Each Track for Early Termination.
    for (flight_id, track_id), last_frame in track_last_frame.items():
        # Skip Tracks Already Classified as Sustained Failures.
        if (flight_id, track_id) in sustained_track_ids:
            continue

        # Skip Tracks Without a Dominant GT Object.
        if (flight_id, track_id) not in dominant_gt.index:
            continue

        # Shouldn't Happen, but Guard Against a Missing Lookup Regardless.
        gt_id = dominant_gt.loc[(flight_id, track_id)]
        if (flight_id, gt_id) not in true_object_last_frame.index:
            continue  

        # Compare Track and GT Final Frames.
        true_end = true_object_last_frame.loc[(flight_id, gt_id)]
        gap = true_end - last_frame

        # The GT Object Ended at (or Close Enough to) the Same Time as the Track.
        if gap <= early_termination_margin:
            continue

        # CANDIDATE Early Termination. Check for GT Reacquisition within Grace Period.
        reacquisition_window_end = last_frame + reacquisition_grace_period
        gt_matched_frames = matched_frames_by_gt.get((flight_id, gt_id), [])
        was_reacquired = any(
            last_frame < f <= reacquisition_window_end for f in gt_matched_frames
        )

        # Successfully Reacquired Within the Grace Period -- Not a Failure.
        if was_reacquired:
            continue

        # Genuine Early Termination: No Track Picked the Object Back Up in Time.
        first_frame = track_first_frame.loc[(flight_id, track_id)]
        early_term_rows.append({
            'flight_id': flight_id,
            'track_id': track_id,
            'failure_onset_frame': last_frame,
            'tot_num_frames': int(true_end - first_frame + 1),
            'failure_type': 'early_termination',
        })

    # Combine Sustained and Early Termination Failures.
    return pd.DataFrame(sustained_rows + early_term_rows)


def _awlt_and_fnr_for_tracks(df_onset, first_warning, frame_rate_hz=DEFAULT_FRAME_RATE_HZ_):
    """
    Shared AWLT (Mean & Median) / False Negative Rate Calculation Over a Set
    of Failed Tracks.

    Factored out of compute_introspection_metrics() so the Exact Same Logic
    can be Applied Once to the Full (Pooled) Set of Failed Tracks, and Again
    to Each failure_type Subset in Turn, Without Duplicating the Loop.

    Args:
        df_onset (pd.DataFrame): Rows for the Failed Tracks to Evaluate.
            Must Include 'flight_id', 'track_id', 'failure_onset_frame'.
        first_warning (Dict[Tuple, int]): Maps (flight_id, track_id) to the
            First Frame on Which that Track was Ever Warned About.
        frame_rate_hz (float): Frame rate used to convert the raw frame-count
            lead time into seconds, matching the units written per-flight
            into result.json by write_introspection_outcomes_to_results().

    Returns:
        Tuple[Optional[float], Optional[float], float, int, int]: (awlt_mean, awlt_median, fnr, num_warned_in_advance, num_missed).
        awlt_mean / awlt_median are in SECONDS.
    """
    # AWLT & False Negative Rate (Over the Supplied Failed Tracks)
    lead_times = []
    num_missed = 0

    # Identify the Failed Track & its Failure Onset Frame.
    for _, row in df_onset.iterrows():
        # Store the Warnings & Onset Frame.
        key = (row['flight_id'], row['track_id'])
        t0 = row['failure_onset_frame']
        t_warn = first_warning.get(key)

        # Never Warned About at All. Counts as a Missed Failure.
        if t_warn is None:
            num_missed += 1

        # Only Warnings Issued Strictly Before Failure Count Toward AWLT.
        elif t_warn < t0:
            # Warned About STRICTLY BEFORE Onset. Convert the raw frame
            # delta to seconds -- this used to be left as a raw frame
            # count, which silently disagreed (by a factor of
            # frame_rate_hz) with the per-flight warning_lead_time_seconds
            # values written into each flight's result.json.
            lead_times.append((t0 - t_warn) / frame_rate_hz)

    # Average Warning Lead Time (Seconds) Over Tracks Warned Before Failure Onset.
    awlt_mean = (sum(lead_times) / len(lead_times)) if lead_times else None
    awlt_median = float(np.median(lead_times)) if lead_times else None

    # False Negative Rate = Missed Failures / Total Failed Tracks.
    fnr = (num_missed / len(df_onset)) if len(df_onset) else 0.0

    # Return the Results. 
    return awlt_mean, awlt_median, fnr, len(lead_times), num_missed


def compute_introspection_metrics(df_matches, min_score, num_sequences,
                                   failure_threshold=DEFAULT_INTROSPECTION_FAILURE_THRESHOLD_,
                                   early_termination_margin=DEFAULT_EARLY_TERMINATION_MARGIN_,
                                   reacquisition_grace_period=DEFAULT_REACQUISITION_GRACE_PERIOD_,
                                   frame_rate_hz=DEFAULT_FRAME_RATE_HZ_):
    """
    Computes Average Warning Lead Time (AWLT), False Negative Rate, and False 
    Alarm Rate (FAR) Together.

    All three share the same Failure Onset Frame / per-track warning-history
    groundwork, so they're computed in a single pass rather than three separate 
    functions re-deriving the same intermediate data.

    AWLT: Mean number of frames between a track's first introspection warning 
        and its Failure Onset Frame, averaged only over failed tracks that were 
        warned about strictly BEFORE onset (a genuine advance warning).

    False Negative Rate: Proportion of genuine failures (per Failure Onset Frame 
        / FTPS) for which no warning was ever raised, before or during the failure.

    False Alarm Rate (FAR): Proportion of tracks FTPS classifies as correct (i.e. 
        never failed) on which a warning was raised anyway.

    Args:
        df_matches (pd.DataFrame): Ground-truth/detection match dataframe. Must 
            include the standard match columns plus a 'warning' column (True/False
            per detection-frame).
        min_score (float): Confidence threshold, forwarded to 
            augment_with_failure_onset_frame().
        num_sequences (int): Present for interface consistency with 
            compute_cedr()/compute_ftps() -- not currently used in the calculation
            itself, since AWLT/FNR/FAR are all ratios or means over tracks, not 
            per-sequence rates.
        failure_threshold (float): Forwarded to augment_with_failure_onset_frame() 
            -- see that function's docstring.
        early_termination_margin (int): Forwarded to augment_with_failure_onset_frame()
            -- see that function's docstring.
        reacquisition_grace_period (int): Forwarded to augment_with_failure_onset_frame()
            -- see that function's docstring.

    Returns:
        dict: {
            awlt_mean: mean lead time in SECONDS (None if no track was ever warned about strictly 
                    before its onset frame), pooled over ALL failure types,
            awlt_median: median lead time in SECONDS (None if no track was ever warned about strictly 
                    before its onset frame), pooled over ALL failure types,
            fnr: False Negative Rate, pooled over ALL failure types,
            far: False Alarm Rate,
            num_failed_tracks: total tracks FTPS/onset classifies as genuine failures,
            num_correct_tracks: total tracks never classified as a failure,
            num_warned_in_advance: number of failed tracks that contributed a lead time to AWLT,
            num_missed: number of failed tracks never warned about at all,
            num_false_alarms: number of correct tracks warned about anyway,
            by_failure_type: dict keyed by failure_type ('sustained', i.e. divergence-type, and 
                             'early_termination', i.e. premature track-loss).
        }
    
    Raises:
        AssertionError: if 'warning' is not present in df_matches -- this means 
        the introspective predictor's warning field isn't flowing all the way 
        through register_object_and_location() and normalize_json_result_to_dataframe() 
        into the results this function reads.
    """
    # Confirm the 'warning' Column Exists Before Doing Any Real Work.
    assert is_in_df_columns(df_matches, ['warning']), (
        "'warning' column not found -- confirm the introspective predictor's "
        "warning field is flowing through register_object_and_location() and "
        "normalize_json_result_to_dataframe().")

    # Compute Failure Onset Frames & Build the Set of Genuinely Failed Tracks.
    df_onset = augment_with_failure_onset_frame(
        df_matches, min_score, failure_threshold,
        early_termination_margin=early_termination_margin,
        reacquisition_grace_period=reacquisition_grace_period)
    failed_track_ids = set(zip(df_onset['flight_id'], df_onset['track_id']))

    # Find the First Frame on Which Each Track Was Ever Warned About.
    df_warn = df_matches[df_matches['warning'] == True].groupby(['flight_id', 'track_id'])['frame'].min().reset_index()
    df_warn = df_warn.rename(columns={'frame': 'first_warning_frame'})
    first_warning = {(r.flight_id, r.track_id): r.first_warning_frame for r in df_warn.itertuples()}

    # "Correct" Tracks are Every Track NOT Classified as a Genuine Failure.
    all_track_ids = set(zip(df_matches['flight_id'], df_matches['track_id']))
    correct_track_ids = all_track_ids - failed_track_ids

    # False Alarm Rate (Over Correct Tracks). Not Failure-Type-Specific.
    num_false_alarms = sum(1 for key in correct_track_ids if key in first_warning)
    far = (num_false_alarms / len(correct_track_ids)) if correct_track_ids else 0.0

    # AWLT (Mean & Median) & False Negative Rate, Pooled Over ALL Failure Types.
    awlt_mean, awlt_median, fnr, num_warned_in_advance, num_missed = _awlt_and_fnr_for_tracks(
        df_onset, first_warning, frame_rate_hz=frame_rate_hz)

    # Same Calculation Again, but Restricted to Each Failure Type in Turn.
    by_failure_type = {}
    for failure_type in sorted(df_onset['failure_type'].unique()) if len(df_onset) else []:
        df_onset_type = df_onset[df_onset['failure_type'] == failure_type]
        type_awlt_mean, type_awlt_median, type_fnr, type_num_warned, type_num_missed = _awlt_and_fnr_for_tracks(
            df_onset_type, first_warning, frame_rate_hz=frame_rate_hz)
        by_failure_type[failure_type] = dict(
            awlt_mean=type_awlt_mean, awlt_median=type_awlt_median, fnr=type_fnr,
            num_failed_tracks=len(df_onset_type),
            num_warned_in_advance=type_num_warned,
            num_missed=type_num_missed,
        )

    # Return All Three Pooled Metrics Plus the Per-Failure-Type Breakdown.
    return dict(
        awlt_mean=awlt_mean, awlt_median=awlt_median, fnr=fnr, far=far,
        num_failed_tracks=len(df_onset),
        num_correct_tracks=len(correct_track_ids),
        num_warned_in_advance=num_warned_in_advance,
        num_missed=num_missed,
        num_false_alarms=num_false_alarms,
        by_failure_type=by_failure_type,
        df_onset=df_onset,
        first_warning=first_warning,
    )


def _parse_img_timestamp_ns(img_name):
    """
    Extracts the leading nanosecond epoch timestamp encoded at the start of
    an AOT-C frame filename (e.g. '1566544363793015157acfeb72c...df7f.png'
    -> 1566544363793015157). Returns None if no leading digit run is found.

    This is the ONLY frame-position signal that is unambiguous across every
    context these filenames get used in (dense raw folder listing, sparse
    result.json entries, etc.) since it is a property of the frame itself,
    not of its position in whichever list happens to be indexing it.
    """
    match = re.match(r'^(\d+)', os.path.basename(str(img_name)))
    return int(match.group(1)) if match else None


def write_introspection_outcomes_to_results(df_onset, first_warning, raw_results_dir,
                                             frame_rate_hz=DEFAULT_FRAME_RATE_HZ_,
                                             df_matches=None):
    """
    Stamps Each Failed Track's Failure Onset Frame & Warning Lead Time Back
    into its Flight's Raw Per-Flight result.json File on Disk.
    """
    if not raw_results_dir or not os.path.exists(raw_results_dir):
        log.warning('No raw_results_dir Supplied -- Skipping Introspection Outcome Write-Back.')
        return 0

    def _norm_track_id(tid):
        if tid is None:
            return None
        try:
            return int(float(tid))
        except (ValueError, TypeError):
            return str(tid)

    # 1. Map (flight_id, global_frame) -> img_name from df_matches.
    frame_to_img = {}
    if df_matches is not None and 'img_name' in df_matches.columns and 'frame' in df_matches.columns:
        for row in df_matches[['flight_id', 'frame', 'img_name']].drop_duplicates().itertuples():
            frame_to_img[(str(row.flight_id), int(row.frame))] = os.path.basename(str(row.img_name))

    # 2. Normalize first_warning keys to (flight_id, int_track_id).
    norm_first_warning = {}
    for (fid, tid), warn_frame in first_warning.items():
        norm_first_warning[(str(fid), _norm_track_id(tid))] = int(warn_frame)

    # 3. Build onset lookup with raw global frames and target image names.
    onset_lookup = {}
    for row in df_onset.itertuples():
        fid = str(row.flight_id)
        norm_tid = _norm_track_id(row.track_id)
        t0_global = int(row.failure_onset_frame)
        t_warn_global = norm_first_warning.get((fid, norm_tid))

        onset_lookup[(fid, norm_tid)] = {
            't0_global': t0_global,
            't_warn_global': t_warn_global,
            'onset_img_name': frame_to_img.get((fid, t0_global)),
            'warn_img_name': frame_to_img.get((fid, t_warn_global)) if t_warn_global is not None else None,
        }

    flights_with_failures = {fid for fid, _ in onset_lookup.keys()}

    # Scan directory tree once
    disk_folders = [
        d for d in os.listdir(raw_results_dir)
        if os.path.isdir(os.path.join(raw_results_dir, d))
    ]

    # Map bare hashes to candidate folder names
    hash_to_folders = {}
    for folder in disk_folders:
        cleaned_hash = re.sub(r'^part\d{1,2}[_-]?', '', folder, flags=re.IGNORECASE)
        hash_to_folders.setdefault(cleaned_hash, set()).add(folder)
        hash_to_folders.setdefault(folder, set()).add(folder)

    updated_count = 0
    for fid in flights_with_failures:
        # Find all unique candidate disk folders matching this fid
        candidates = sorted(hash_to_folders.get(fid, set()))

        # Suffix fallback if prefix strip didn't catch a non-standard pattern
        if not candidates:
            candidates = sorted({d for d in disk_folders if d.endswith(fid)})

        # Hard fail if ambiguous or missing
        if len(candidates) > 1:
            raise RuntimeError(
                f"FATAL: Ambiguous flight mapping for flight_id '{fid}'. "
                f"Multiple folders on disk match: {candidates}. "
                f"Aborting to avoid writing into the wrong flight folder."
            )
        elif len(candidates) == 0:
            log.warning('No result folder on disk matches flight_id %s -- skipping write-back.', fid)
            continue

        target_folder = candidates[0]
        result_path = os.path.join(raw_results_dir, target_folder, 'result.json')
        if not os.path.exists(result_path):
            log.warning('result.json not found at %s', result_path)
            continue

        with open(result_path, 'r') as f:
            entries = json.load(f)

        img_to_local_idx = {}
        for local_idx, entry in enumerate(entries, start=1):
            img_name = os.path.basename(entry.get('img_name', ''))
            if img_name:
                img_to_local_idx[img_name] = local_idx

        total_frames = len(entries)

        for local_idx, entry in enumerate(entries, start=1):
            for det in entry.get('detections', []):
                norm_tid = _norm_track_id(det.get('track_id'))
                info = onset_lookup.get((fid, norm_tid))

                local_onset_frame = None
                local_warn_frame = None
                onset_img_name = None
                warn_img_name = None
                lead_time_seconds = None

                if info is not None:
                    onset_img_name = info['onset_img_name']
                    if onset_img_name and onset_img_name in img_to_local_idx:
                        local_onset_frame = img_to_local_idx[onset_img_name]
                    elif info['t0_global'] <= total_frames:
                        local_onset_frame = info['t0_global']

                    warn_img_name = info['warn_img_name']
                    if warn_img_name and warn_img_name in img_to_local_idx:
                        local_warn_frame = img_to_local_idx[warn_img_name]
                    elif info['t_warn_global'] is not None and info['t_warn_global'] <= total_frames:
                        local_warn_frame = info['t_warn_global']

                    t0_ns = _parse_img_timestamp_ns(onset_img_name) if onset_img_name else None
                    twarn_ns = _parse_img_timestamp_ns(warn_img_name) if warn_img_name else None

                    if t0_ns is not None and twarn_ns is not None:
                        lead_time_seconds = round((t0_ns - twarn_ns) / 1e11, 3)
                    elif local_onset_frame is not None and local_warn_frame is not None:
                        lead_time_seconds = round((local_onset_frame - local_warn_frame) / frame_rate_hz, 3)
                    elif info['t_warn_global'] is not None:
                        lead_time_seconds = round((info['t0_global'] - info['t_warn_global']) / frame_rate_hz, 3)

                det['failure_onset_frame'] = local_onset_frame
                det['failure_onset_img_name'] = onset_img_name
                det['warning_img_name'] = warn_img_name
                det['warning_lead_time_seconds'] = lead_time_seconds

        with open(result_path, 'w') as f:
            json.dump(entries, f)

        updated_count += 1

    log.info('Wrote failure_onset_frame / warning_lead_time_seconds Back into '
             '%d / %d Flights\' result.json Files.', updated_count, len(flights_with_failures))

    return updated_count


def compute_positional_divergence(df_matches, min_score, min_matched_frames=DEFAULT_MIN_MATCHED_FRAMES_FOR_DRIFT_):
    """
    Computes Positional Divergence (Track Drift) per Track.

    For every track, fits a simple linear trend (least-squares slope) of IoU against 
    frame index, restricted to frames where the track is still considered matched 
    (gt_det_match == 1).

    Args:
        df_matches (pd.DataFrame): Ground-truth/detection match dataframe. Must
            include the standard match columns plus 'iou'.
        min_score (float): Minimum confidence score required for a detection to 
            count as a match.
        min_matched_frames (int): Minimum number of matched frames a track must 
            have before its IoU trend is considered meaningful.

    Returns:
        dict: {
            mean_iou_slope: mean regression slope across all eligible tracks,
            pct_tracks_drifting: proportion of eligible tracks with a negative slope,
            num_eligible_tracks: number of tracks that met min_matched_frames and 
                were included in the above,
        }

    Raises:
        AssertionError: if 'iou' is not present in df_matches -- this column 
            should already exist from the standard IoU-matching step and its 
            absence indicates a broken or reordered pipeline upstream, not 
            something this function should silently work around.
    """
    # Confirm the 'iou' Column Exists.
    assert is_in_df_columns(df_matches, ['iou']), (
        "'iou' column not found -- this should already exist from "
        "augment_with_iou_match() in match_groundtruth_results.py.")
    
    # Work on a Copy & Treat Any Detection Below min_score as Unmatched.
    df = df_matches.copy()
    below_thresh_det = df['s'] < min_score
    df.loc[below_thresh_det, 'gt_det_match'] = 0

    # Restrict to Frames Where the Track is Still Considered Matched.
    df_matched = df[df['gt_det_match'] == 1]

    # Fit a Linear Trend of IoU vs. Frame Index for Each Eligible Track.
    slopes = []
    for (flight_id, track_id), group in df_matched.groupby(['flight_id', 'track_id']):
        group = group.sort_values('frame')

        # Skip Tracks with Too Few Matched Frames for a Meaningful Trend.
        if len(group) < min_matched_frames:
            continue

        # Manual Least-Squares Slope.
        x = group['frame'].values.astype(float)
        y = group['iou'].values.astype(float)
        x_mean, y_mean = x.mean(), y.mean()
        denom = ((x - x_mean) ** 2).sum()

        # All Frame Indices Identical -- Cannot Fit a Trend, Skip.
        if denom == 0:
            continue
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        slopes.append(slope)

    # Aggregate Across All Eligible Tracks.
    mean_slope = (sum(slopes) / len(slopes)) if slopes else None
    pct_drifting = (sum(1 for s in slopes if s < 0) / len(slopes)) if slopes else 0.0

    # Return Slope & Drift.
    return dict(
        mean_iou_slope=mean_slope,
        pct_tracks_drifting=pct_drifting,
        num_eligible_tracks=len(slopes),
    )
# ------------------------------------------------------------------------------


def augment_with_diff_to_first_frame(df_encounters):
    """adds difference in frame between each frame in encounter and the first frame of the encounter"""
    required_columns_str = ', '.join(['frame', 'framemin']) 
    assert is_in_df_columns(df_encounters, ['frame', 'framemin']), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))

    diff_to_first_frame = df_encounters['frame'] - df_encounters['framemin']
    df_encounters = df_encounters.assign(delta_to_min_frame = diff_to_first_frame)
    return df_encounters

def augment_with_detection_info(df_encounters_info, fl_dr_thresh, use_track_fl_dr=False):
    """adds maximum moving frame level detection rate to each encounter and if it its above
    or equal to the provided threshold the detection range and latency are added""" 
   
    fl_dr_col = 'fl_dr'
    if use_track_fl_dr:
        fl_dr_col = 'track_' + fl_dr_col
    def calc_detection_info(df):
        detection_range = np.nan
        det_latency = np.nan
        max_fl_dr = df[fl_dr_col].max() 
        if max_fl_dr >= fl_dr_thresh:
            first_index_above_thresh = df[df[fl_dr_col].ge(fl_dr_thresh, fill_value=0)].index[0]
            detection_range = df['range_distance_m'][first_index_above_thresh]
            det_latency = df['delta_to_min_frame'][first_index_above_thresh]
        return pd.Series(data=[max_fl_dr, detection_range, det_latency], 
                        index=['max_fl_dr', 'det_range_m', 'det_latency_frames'])
    required_cols = ['encounter_id', 'range_distance_m', 'delta_to_min_frame', fl_dr_col]
    required_columns_str = ', '.join(required_cols) 
    assert is_in_df_columns(df_encounters_info, required_cols), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    df_enc_det_info = df_encounters_info.groupby(['encounter_id'])[
                ['range_distance_m', fl_dr_col, 'delta_to_min_frame']].apply(calc_detection_info)    
    return df_enc_det_info

def compute_moving_frame_level_detection_rate_per_encounter(df_matches, df_val_encounters, 
                                                min_score, fl_dr_temporal_win, use_track_fl_dr=False):
    """Computes moving frame level detection rate per encounter. The detection matches 
    are counted within the provided fl_dr_temporal_win, which slides across the frames that belong
    to the encounter. Detections with score less then min_score are filtered. 
    The detection rate is calculated only for valid encounters.
    """
    required_cols = ['s', 'flight_id', 'img_name', 'frame', 'id', 'gt_det_match']
    if use_track_fl_dr:
        required_cols += ['track_id']
    required_columns_str = ', '.join(required_cols) 
    assert is_in_df_columns(df_matches, required_cols), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    assert is_in_df_columns(df_val_encounters, ['encounter_id']), (
        'encounter_id column is not found, cannot augment')
    
    def flatten(list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
        return list_of_lists[:1] + flatten(list_of_lists[1:]) 
    
    log.info('Thresholding score')
    below_thresh_det = df_matches['s'] < min_score
    df_matches.loc[below_thresh_det, 'gt_det_match'] =  0 
    df_intruders_matches = df_matches.groupby(
            ['flight_id', 'img_name', 'frame', 'id'], as_index=False)['gt_det_match'].max()
    
    num_encounters =  df_val_encounters['encounter_id'].nunique()
    log.info('Number of encounters to detect %d', num_encounters)
    df_val_encounters = augment_with_diff_to_first_frame(df_val_encounters)
    log.info('Combining encounters with results')

    df = combine_encounter_with_gt_det_matches(df_intruders_matches, df_val_encounters)
    
    log.info('Grouping data frame with matches to getdetection matches per encounter')
    df_encounters_with_frame_matches = df.groupby(
                                    ['encounter_id','frame'])['gt_det_match'].max().reset_index(0)
    log.info('Augmenting with moving frame level detection rate, this might take some time')
    df_with_moving_fl_dr = augment_with_moving_frame_level_detection_rate_per_encounter(
                                        df_encounters_with_frame_matches, fl_dr_temporal_win)
    
    log.info('Merge frame_level detection rate ')
    df = df.merge(df_with_moving_fl_dr, on=['encounter_id','frame'], how='left')    

    if use_track_fl_dr:
        df_matches = df_matches.assign(matched_track_id = [track_id if is_match else  -1 
                for track_id, is_match in zip(df_matches['track_id'], df_matches['gt_det_match'])])
        log.info('Grouping data frame with matches to get matched track_ids per frame and object')
        df_matched_track_ids = df_matches.groupby(['flight_id', 'img_name', 'frame', 'id'], 
                                as_index=False)['matched_track_id'].agg(lambda x: flatten(list(x)))

        df2 = combine_encounter_with_gt_det_matches(df_matched_track_ids, df_val_encounters)
        log.info('Grouping data frame with matches to get matched track_ids per encounter and frame')
        df_encounters_with_matched_track_ids = df2.groupby(
            ['encounter_id','frame'])['matched_track_id'].agg(lambda x: flatten(list(x))).reset_index(0)
        df_with_track_id_count = augment_with_moving_most_common_track_id_count_per_encounter(
                                            df_encounters_with_matched_track_ids, fl_dr_temporal_win)
        df2 = df2.merge(df_with_track_id_count, on=['encounter_id','frame'], how='left')
        df = df.merge(df2[['encounter_id','frame', 'matched_track_id', 'track_fl_dr']], 
                                            on=['encounter_id','frame'], how='left')
    # asserting correctness of detection rate calculation based on track_id compared to regular fl_dr
    assert is_in_df_columns(df, ['fl_dr', 'track_fl_dr']), 'fl_dr or track_fl_dr not found'
    df_wrong = df.query('fl_dr < track_fl_dr') 
    assert len(df_wrong) == 0, 'track frame level detection rate is wrong'  
    return df

def get_encounter_frame_level_info(df):
    """Provides basic information about encounters:
    encounter_id - encounter id 
    flight_id - which flight it belongs to 
    framemin - first frame of the encounter 
    framemax - last frame of the encounter 
    tot_num_frames - total number of frames in encounters (without gaps)
    num_matched_frames  - number of frames within encounter with matched ground truth
    is_above_horizon - 1 - above horizon, 0 - below horizon, any value in (-1, 1) is mixed 
    """
    required_cols = ['encounter_id', 'flight_id', 'framemin', 'framemax','framecount', 'gt_det_match', 
                    'frame', 'is_above_horizon']
    required_columns_str = ', '.join(required_cols) 
    assert is_in_df_columns(df, required_cols), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    
    df_partial= df[required_cols]
    df_encounter_frame_level_info = df_partial.groupby('encounter_id').agg(
                {'flight_id': 'first', 'framemin': 'first', 'framemax': 'first', 'framecount': 'first', 
                'frame': 'count', 'gt_det_match': 'sum', 'is_above_horizon': 'mean'}).reset_index(0)    

    df_encounter_frame_level_info = df_encounter_frame_level_info.rename(columns={
                                                            'frame': 'tot_num_frames', 
                                                            'gt_det_match': 'num_matched_frames'})
    assert len(df_encounter_frame_level_info.query('tot_num_frames != framecount')) == 0, (
            'something went wrong frame counts do not agree')
    
    return df_encounter_frame_level_info

def _augment_with_is_encounter_detected_before_range(df_encounter_detections, 
                                                    temporal_win, ranges_to_detect):
    """ helper to figure out if encounter is detected based on detection range and latency"""
    is_detected_fast = df_encounter_detections['det_latency_frames'] < FAST_DETECTION_WINDOW_ #frames
    for range_to_detect in ranges_to_detect:
        is_detected_before_this_range = df_encounter_detections['det_range_m'] >= range_to_detect
        df_encounter_detections = df_encounter_detections.assign(detected_before = 
                                1 * np.logical_or(is_detected_before_this_range, is_detected_fast))
        df_encounter_detections = df_encounter_detections.rename(columns={'detected_before':
                                                    'detected_before_{}'.format(range_to_detect)})
    return df_encounter_detections


def compute_encounter_detections(df_enc_with_fl_dr, min_frame_level_dr, temporal_win, 
                                    ranges_to_detect=RANGES_TO_DETECT, use_track_fl_dr=False):
    """Computes if encounter was detected based on provided function is_encounter_detected_func
    Params:
        df: pd.DataFrame, data frame with encounters and matches of detection to intruders info
        min_frame_level_dr: float, minimum frame level rate required for encounter
        ranges_to_detect: List[float], list of ranges at which to evaluate detection
    Returns:
        pd.DataFrame of encounters with information on detection per encounter   
    """ 
    assert is_in_df_columns(df_enc_with_fl_dr, ['encounter_id']), ('encounter_id not in data frame, '
                                                    'cannot calculate encounter detections')
    log.info('Checking if encounters were detected')
    df_encouter_detection_info = augment_with_detection_info(df_enc_with_fl_dr, min_frame_level_dr, 
                                                                                use_track_fl_dr)
    df_encounter_frame_level_info = get_encounter_frame_level_info(df_enc_with_fl_dr)      
    df_encounter_info = df_encounter_frame_level_info.merge(df_encouter_detection_info, 
                                                            on='encounter_id', how='left')
    df_encounter_detections = _augment_with_is_encounter_detected_before_range(df_encounter_info, 
                                                                    temporal_win, ranges_to_detect)
    return df_encounter_detections

def search_score_for_target_func(min_score, max_score, target_func, target_func_val, 
                                target_func_val_precision):
    """This function performs a search for a score that receives a certain value of the provided 
    target function. The search is done using binary search. There is an assumption that the function
    is monotonical with respect to the scores. The direction is determined based on minimum and middle
    scores
    Parameters:
        min_score: float, minimal score possible
        max_score: float, maximal score possible
        target_func: function handler, there is an assumption that the function is monotonical vs. score
        target_func_val: float, expected value of the function, such that
             target_func_val @ output_score = target_func_val
        target_func_val_precision: int, since the search is done over float outputs of target_func_val
            there is a need to define a precision at which we compare the results
    Returns:
        float, output_score such that target_func_val @ output_score = target_func_val 
        Note that if target_func_val is not reachable the output_score will match the closes function value
    """
    min_s = min_score
    max_s = max_score
    min_s_func_val = target_func(min_score=min_s)
    mid_s = min_s + (max_s - min_s) / 2
    mid_s_func_val = round(target_func(min_score=mid_s), target_func_val_precision)
    if mid_s_func_val <= min_s_func_val:
        move_min_to_mid = lambda mid, target: mid > target
    else:
        move_min_to_mid = lambda mid, target: mid < target
    while max_s - min_s > SCORE_PRECISION_:
        if mid_s_func_val == target_func_val:
            return mid_s
        elif move_min_to_mid(mid_s_func_val, target_func_val):
            min_s = mid_s
        else:
            max_s = mid_s
        mid_s = min_s + (max_s - min_s) / 2
        mid_s_func_val = round(target_func(min_score=mid_s), target_func_val_precision)
    return max_s
    
def get_working_point_based_on_metrics(df_matches, target_metrics, target_value, target_value_precision,
                                       min_det_score, max_det_score, max_range):
    """Determines the score to threshold detections.
    Parameters:
        df_matches: pd.DataFrame, data frame with matches between intruders and detections
        target_metrics: str, what metrics to use to determine the score
        target_value: float, expected value of the metrics
        target_value_precision: int, with which precision to calculate the value 
        min_det_score: float, minimum detection score
        max_det_score: float, maximum detection score
    Returns:
        float, the score to threshold the detections
    """
    thresh_score = None 
    search_score_func = None 
    if 'far' in target_metrics:
        search_score_func = partial(compute_false_alarms_per_hour, df_matches)
    elif 'fppi' in target_metrics:
        search_score_func = partial(compute_false_positives_per_image, df_matches)
    elif 'fl_dr' in target_metrics:
        search_score_func = partial(compute_probability_of_detection_of_planned_intruders, 
                                    df_matches, max_range=max_range, min_range=0)

    if search_score_func is not None:
        log.info('%s = %.5f will be used as metrics for score threshold search', target_metrics, 
                                                                                    target_value)
        thresh_score = search_score_for_target_func(min_det_score, max_det_score, 
                                        search_score_func, target_value, target_value_precision)
    return thresh_score

def get_max_range_based_on_encounters_info(df_encounters):
    """looks at all the frames with valid encounters and returns their maximum range"""
    assert is_in_df_columns(df_encounters, ['range_distance_m']), ('range_distance_m not in data frame, '
                                                                    'cannot calculate maximum range')
    return get_valid_encounters(df_encounters)['range_distance_m'].max()

def _is_min_score_as_expected(df_matches, expected_min_score):
    """assert that minimum score in the results is as expected"""
    assert is_in_df_columns(df_matches, ['s']), ('s not in data frame, cannot check minimum score')
    min_score_results = df_matches['s'].min()
    assert min_score_results >= expected_min_score, ('Expected min score = {} is greater than '
                'minimal score = {} in the results'.format(expected_min_score, min_score_results))
    return min_score_results

def _summarize_encounter_detection_rate(summary, eval_criteria, num_det_encs, num_total_encs):
    summary[eval_criteria] = OrderedDict()
    summary[eval_criteria]['Encounters'] = OrderedDict()
    for max_range in RANGES_TO_DETECT:
        summary[eval_criteria]['Encounters'][max_range] = OrderedDict()
        for num_key, num_value in num_det_encs[max_range].items():
            if num_total_encs[num_key]:
                dr_enc = float(num_value / num_total_encs[num_key])
            else:
                dr_enc  = 0.0
            summary[eval_criteria]['Encounters'][max_range][num_key] = {
                                                        'detected': int(num_value), 
                                                        'total': int(num_total_encs[num_key]),
                                                        'dr': dr_enc}
            log.info('Max. range %d: %s: %d / %d  = %.3f', max_range, num_key, num_value, 
                                                            num_total_encs[num_key], dr_enc)
    return summary

####################################################################################################
# MAIN
####################################################################################################

def run(flags):
    log.info('Reading ground truth detection matches from %s', 
                                    flags.groundtruth_results_matches_filename)
    df_gt_det_matches = pd.read_csv(flags.groundtruth_results_matches_filename, low_memory=False)
    
    min_score_results = _is_min_score_as_expected(df_gt_det_matches, flags.min_det_score)
    flags.min_det_score = max(min_score_results, flags.min_det_score)

    log.warning('Reading ground truth with encounters from %s', 
                                    flags.encounters_with_groundtruth_filename)
    df_encounters = pd.read_csv(flags.encounters_with_groundtruth_filename, low_memory=False)
    df_encounters = exclude_encounters_from_evaluation(df_encounters, flags.min_enc_range_upper_bound,
                                                      flags.max_enc_range_lower_bound)
    max_encounter_range = get_max_range_based_on_encounters_info(df_encounters)
    log.info('Maximum range of encounter is %.2f', round(max_encounter_range, 2))
    
    if flags.target_metrics is not None:    
        log.info('Determining threshold for detection score')
        if flags.target_value is None:
            if flags.target_metrics != 'far':
                raise ValueError('Please provide target value for {}'.format(flags.target_metrics))
            target_value = TARGET_FAR_PER_HOUR
        else:
            target_value = flags.target_value 
        log.info('Will use {} target value for {} calculation'.format(target_value, flags.target_metrics))
        thresh_score = get_working_point_based_on_metrics(df_gt_det_matches, flags.target_metrics, 
            target_value, flags.target_value_precision, flags.min_det_score, flags.max_det_score, 
                                                                            max_encounter_range)
    else:
        log.info('The provided minimum detection score %.5f will be used', flags.detection_score_threshold)           
        thresh_score = max(flags.min_det_score, flags.detection_score_threshold)
        
    log.info('Frame level metrics calculation for score threshold = {}'.format(thresh_score))
    df_no_dupl_objs = df_gt_det_matches.drop_duplicates(['img_name', 'id'])
    num_planned = len(df_no_dupl_objs.query(PLANNED_INTRUDERS))
    num_non_planned = len(df_no_dupl_objs.query(NON_PLANNED_INTRUDERS))
    num_non_planned_aircraft = len(df_no_dupl_objs.query(NON_PLANNED_AIRCRAFT))
    far = compute_false_alarms_per_hour(df_gt_det_matches, min_score=thresh_score) 
    fppi = compute_false_positives_per_image(df_gt_det_matches, min_score=thresh_score)
    fl_dr_range, num_det_range, num_tot_range = compute_probability_of_detection_of_planned_intruders(
                df_gt_det_matches, min_score=thresh_score, max_range=max_encounter_range, min_range=0)
    fl_dr_above_area, num_det_above_area, num_tot_above_area = compute_probability_of_detection_small_objects(
                        df_gt_det_matches, min_area=flags.min_object_area, min_score=thresh_score, 
                        obj_query= GT_ABOVE_AREA_QUERY + ' and ' + NOT_BIRD_QUERY)
    fl_dr_below_area, num_det_below_area, num_tot_below_area = compute_probability_of_detection_small_objects(
                        df_gt_det_matches, min_area=flags.min_object_area, min_score=thresh_score,
                        obj_query= GT_BELOW_AREA_QUERY + ' and ' + NOT_BIRD_QUERY)

    df_val_encounters = get_valid_encounters(df_encounters)
    df_val_encounters_with_fl_dr  = compute_moving_frame_level_detection_rate_per_encounter(df_gt_det_matches, 
        df_val_encounters, thresh_score, flags.fl_dr_temporal_win, use_track_fl_dr=flags.use_track_fl_dr)
    df_final_results = compute_encounter_detections(df_val_encounters_with_fl_dr, 
                    flags.min_fl_dr, flags.fl_dr_temporal_win, use_track_fl_dr=False)
    if flags.use_track_fl_dr:
        df_final_results_track_fl_dr = compute_encounter_detections(df_val_encounters_with_fl_dr, 
                    flags.min_fl_dr, flags.fl_dr_temporal_win, use_track_fl_dr=True)
    
    # Making New Metrics.
    df_cedr = compute_cedr(df_gt_det_matches, df_val_encounters, thresh_score)
    num_sequences = df_gt_det_matches['flight_id'].nunique()
    df_ftps_tracks, ftps_rate_by_threshold, ftps_count_by_threshold, ftps_averaged, ftps_num_eligible = compute_ftps(
        df_gt_det_matches, thresh_score, num_sequences
    )

    # Making Introspection Metrics.
    introspection_metrics = compute_introspection_metrics(df_gt_det_matches, thresh_score, num_sequences)
    divergence_metrics = compute_positional_divergence(df_gt_det_matches, thresh_score)

    # Write failure_onset_frame / warning_lead_time_seconds Back into Each
    # Flight's Raw result.json, so the Visualiser can Read them Directly
    # Rather than Having the Notebook Source them Separately. Only Runs if
    # --raw-results-dir was Supplied. 'df_onset' / 'first_warning' are Popped
    # off Here so they Never Reach the JSON-Serialised Summary Below.
    df_onset = introspection_metrics.pop('df_onset')
    first_warning = introspection_metrics.pop('first_warning')
    write_introspection_outcomes_to_results(
        df_onset, first_warning, flags.raw_results_dir, df_matches=df_gt_det_matches
    )

    # saving intermidiate results
    log.info('Saving results')
    if flags.output_dir_path is None:
        output_dir = os.path.dirname(flags.groundtruth_results_matches_filename)
    else:
        output_dir = flags.output_dir_path
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
   
    working_point_far_str = str(round(far, DEFAULT_METRICS_VALUE_PRECISION_)).replace('.', '_')
    if flags.save_intermediate_results:
        log.info('Saving intermediate results')
        results_filename = os.path.join(output_dir, 
            flags.results_name_prefix + '_moving_{}_fl_dr_far_{}.csv'.format(
                                                flags.fl_dr_temporal_win, working_point_far_str))
        log.info('Data frame with moving frame level detection rate is saved to %s', results_filename)
        df_val_encounters_with_fl_dr.to_csv(results_filename)

    results_filename = os.path.join(output_dir, 
        flags.results_name_prefix + '_moving_{}_fl_dr_{}_encounter_detections_far_{}'.format(
            flags.fl_dr_temporal_win, str(flags.min_fl_dr).replace('.','p'), working_point_far_str))
    df_final_results.to_csv(results_filename + '.csv')
    df_final_results_track_fl_dr.to_csv(results_filename + '_tracking.csv')
    log.info('Data frame with information on encounter detection is saved to %s.csv and %s_tracking.csv',
                                                                results_filename, results_filename)
    df_final_results.to_json(results_filename + '.json', orient='records', lines=True, indent=4)
    df_final_results_track_fl_dr.to_csv(results_filename + '_tracking.csv')
    
    # Saving New Metrics. 
    df_cedr.to_csv(results_filename + '_cedr.csv')
    df_ftps_tracks.to_csv(results_filename + '_ftps_tracks.csv')

    log.info('Data frame with information on encounter detection is saved to %s.json', results_filename)
    
    # calculation of metrics
    log.info('Calculating final summary')
    num_total_encounters = {}
    for query_key, query_value in HORIZON_QUERIES.items():
        num_total_encounters[query_key] = len(df_final_results.query(query_value))
    num_det_encounters = {}
    if flags.use_track_fl_dr:
        num_det_encounters_tracking = {}
    for det_range in RANGES_TO_DETECT: 
        num_det_encounters[det_range] = {}
        num_det_encounters_tracking[det_range] = {}
        for query_key, query_value in HORIZON_QUERIES.items():
            num_det_encounters[det_range][query_key] = df_final_results.query(query_value)[
                                                        'detected_before_{}'.format(det_range)].sum()
            if flags.use_track_fl_dr:
                num_det_encounters_tracking[det_range][query_key] = df_final_results_track_fl_dr.query(
                                            query_value)['detected_before_{}'.format(det_range)].sum()
    
    summary = {} 
    log.info('Summary')
    summary['gt_encounters'] = flags.encounters_with_groundtruth_filename
    summary['gt_det_matches'] = flags.groundtruth_results_matches_filename
    summary['target_metrics'] = flags.target_metrics
    summary['target_value'] = flags.target_value
    summary['min_det_score'] = float(thresh_score)
    log.info('The minimum detection score is %.3f', thresh_score)
    summary['ftps_rate_by_threshold']   = {f"{int(round(t*100))}": v for t, v in ftps_rate_by_threshold.items()}
    summary['ftps_count_by_threshold']  = {f"{int(round(t*100))}": v for t, v in ftps_count_by_threshold.items()}
    summary['ftps_averaged']            = ftps_averaged
    summary['ftps_num_eligible_tracks'] = int(ftps_num_eligible)
    summary['ftps_num_sequences']       = int(num_sequences)
    log.info('FTPS: %s | avg(50:95): %.4f | %d eligible tracks across %d sequences',
             {k: round(v, 4) for k, v in summary['ftps_rate_by_threshold'].items()},
             ftps_averaged, ftps_num_eligible, num_sequences)
    summary['fppi'] = float(fppi)
    log.info('FPPI: %.5f', fppi)
    summary['far'] = float(far)
    log.info('HFAR: %.5f', far)
    summary['num_planned_intruders'] = int(num_planned)
    summary['num_non_planned_intruders'] = int(num_non_planned)
    summary['num_non_planned_aircraft'] = int(num_non_planned_aircraft)
    summary['cedr_mean'] = float(df_cedr['cedr'].mean()) if len(df_cedr) else 0.0
    summary['cedr_num_qualifying_encounters'] = int(len(df_cedr))
    log.info('CEDR (mean over %d qualifying encounters, min %d frames): %.5f',
             len(df_cedr), DEFAULT_CEDR_MIN_ENCOUNTER_FRAMES_, summary['cedr_mean'])
    summary['awlt_mean'] = introspection_metrics['awlt_mean']
    summary['awlt_median'] = introspection_metrics['awlt_median']
    summary['false_negative_rate'] = introspection_metrics['fnr']
    summary['false_alarm_rate'] = introspection_metrics['far']
    summary['num_failed_tracks'] = introspection_metrics['num_failed_tracks']
    summary['num_correct_tracks'] = introspection_metrics['num_correct_tracks']
    summary['awlt_mean_by_failure_type'] = {
        ft: metrics['awlt_mean'] for ft, metrics in introspection_metrics['by_failure_type'].items()
    }
    summary['awlt_median_by_failure_type'] = {
        ft: metrics['awlt_median'] for ft, metrics in introspection_metrics['by_failure_type'].items()
    }
    summary['false_negative_rate_by_failure_type'] = {
        ft: metrics['fnr'] for ft, metrics in introspection_metrics['by_failure_type'].items()
    }
    summary['num_failed_tracks_by_failure_type'] = {
        ft: metrics['num_failed_tracks'] for ft, metrics in introspection_metrics['by_failure_type'].items()
    }
    summary['mean_iou_slope'] = divergence_metrics['mean_iou_slope']
    summary['pct_tracks_drifting'] = divergence_metrics['pct_tracks_drifting']
    summary['num_eligible_drift_tracks'] = divergence_metrics['num_eligible_tracks']
    log.info('AWLT Mean: %s | AWLT Median: %s | FNR: %.4f | FAR: %.4f | Mean IoU Slope: %s',
             introspection_metrics['awlt_mean'], introspection_metrics['awlt_median'],
             introspection_metrics['fnr'], introspection_metrics['far'],
             divergence_metrics['mean_iou_slope'])
    log.info('By Failure Type -- AWLT Median: %s | FNR: %s',
             summary['awlt_median_by_failure_type'], summary['false_negative_rate_by_failure_type'])
    log.info('Planned Aircraft: %d', num_planned)
    log.info('Non-Planned Airborne: %d', num_non_planned)
    log.info('Non-Planned Aircraft: %d', num_non_planned_aircraft)
    tot_aircraft = num_non_planned_aircraft + num_planned
    log.info('All Aircraft: %d', tot_aircraft)
    summary['max_range'] = float(max_encounter_range)
    summary['tot_aircraft_in_range'] = int(num_tot_range)
    summary['det_aircraft_in_range'] = int(num_det_range)
    summary['fl_dr_in_range'] = float(fl_dr_range)
    log.info('AFDR, aircraft with range <= %.2f: %.5f = %d / %d', 
                                     max_encounter_range,fl_dr_range, num_det_range, num_tot_range)
    tot_aircraft_included_in_fl_dr_area = num_tot_above_area + num_tot_below_area
    assert tot_aircraft == tot_aircraft_included_in_fl_dr_area, (
     'Expected number of aircraft is {}, but got {} '.format(tot_aircraft, 
                                                            tot_aircraft_included_in_fl_dr_area))
    summary['thresh_area'] = float(flags.min_object_area)
    summary['tot_aircraft_above_area'] = int(num_det_above_area)
    summary['det_aircraft_above_area'] = int(num_tot_above_area)
    summary['fl_dr_above_area'] = float(fl_dr_above_area)
    log.info('AFDR, aircraft with area > %d: %.5f = %d / %d', 
                    flags.min_object_area, fl_dr_above_area, num_det_above_area, num_tot_above_area)
    summary['tot_aircraft_below_area'] = int(num_det_below_area)
    summary['det_aircraft_below_area'] = int(num_tot_below_area)
    summary['fl_dr_below_area'] = float(fl_dr_below_area)
    log.info('AFDR, aircraft with area <= %d: %.5f = %d / %d', 
                    flags.min_object_area, fl_dr_below_area, num_det_below_area, num_tot_below_area)
    log.info('Detected Encounters based on Detections: ')
    summary = _summarize_encounter_detection_rate(summary, 'Detection', num_det_encounters, 
                                                                        num_total_encounters)
    if flags.use_track_fl_dr:
        log.info('Detected Encounters based on Tracking: ')
        summary = _summarize_encounter_detection_rate(summary, 'Tracking', num_det_encounters_tracking, 
                                                                            num_total_encounters)
    summary_json = os.path.join(output_dir, 'summary_far_{}_min_intruder_fl_dr_{}_in_win_{}.json'.format(
            working_point_far_str, str(flags.min_fl_dr).replace('.','p'), flags.fl_dr_temporal_win))
    log.info('Saving summary to %s', summary_json)
    with open(summary_json, 'w') as fj:
        json.dump(summary, fj, indent=4)

    return far, summary_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates airborne metrics given encounters')
    add_flags(parser)
    parser.add_argument('--log-level', default=logging.getLevelName(logging.INFO), 
                                                    help='Logging verbosity level')
    args = parser.parse_args()
    setup_logging(args.log_level)    
    check_flags(args)
    run(args)