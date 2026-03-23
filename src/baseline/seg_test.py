# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: seg_test.py %
# Date Created:            %date_created: Mon Mar 02 09:00:00 2021 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: Dmytro Poplavskiy %
# Integrated By:           %integrated_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# Winning solution entrypoint for the AICrowd Airborne Object Tracking
# Challenge (2021). Implements the SegPredictor class which uses a segmentation-
# based detection and tracking pipeline (SegTracker) to identify airborne
# objects in sequential flight frames.
#
# Original solution by Dmytro Poplavskiy (Dmytro.Poplavskiy@gmail.com).
# Full write-up: https://www.aicrowd.com/challenges/airborne-object-tracking-challenge
#
# USAGE INSTRUCTIONS:
#    Run the Script from Project Root:    
#       - python -m src.baseline.seg_test
#
# ------------------------------------------------------------------------------
# -                          L I C E N S I N G                                 -
# ------------------------------------------------------------------------------
# Original work Copyright (c) 2021 Dmytro Poplavskiy.
# Integrated into this project under the terms of the original competition
# open-source requirements. Modifications and integration by Lockheed Martin
# Australia are protected under applicable copyright and intellectual property
# laws.
# ------------------------------------------------------------------------------

import sys
import os
import cv2
from tqdm import tqdm
import re

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.patches
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
from evaluator.airborne_detection import AirbornePredictor
from seg_tracker.seg_tracker import SegDetector, SegTrackerFromOffset

MIN_TRACK_LEN = int(os.getenv("MIN_TRACK_LEN", 30))
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.985))


class SegPredictor(AirbornePredictor):
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    def inference_setup(self) -> None:
        self.detector = SegDetector()
        self.tracker = SegTrackerFromOffset(detector=self.detector)
        self.visualize_pred = False

    def get_all_frame_images(self, flight_id: str) -> List[str]:
        frames = []
        flight_folder = join(str(self.test_data_path), flight_id)
        for frame in sorted(listdir(flight_folder)):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)
        return frames
    
    def flight_started(self) -> None:
        self.track_id_results: Dict = {}
        self.visited_frame: Dict = {}
        self.track_len_so_far: Dict = {}
        self.tracker = SegTrackerFromOffset(detector=self.detector)

    def proxy_register_object_and_location(
        self,
        class_name: str,
        track_id: int,
        bbox: List[float],
        confidence: float,
        img_name: str
    ) -> None:
        # MIN_TRACK_LEN check
        if track_id not in self.track_len_so_far:
            self.track_len_so_far[track_id] = 0
        self.track_len_so_far[track_id] += 1
        if self.track_len_so_far[track_id] <= MIN_TRACK_LEN:
            return

        # MIN_SCORE check
        if confidence < MIN_SCORE:
            return

        if img_name not in self.visited_frame:
            self.visited_frame[img_name] = []
        if track_id in self.visited_frame[img_name]:
            raise Exception('two entities within the same frame {} have the same track id'.format(img_name))
        self.visited_frame[img_name].append(track_id)

        self.register_object_and_location(class_name, track_id, bbox, confidence, img_name)

    def inference(self, flight_id: str) -> None:
        self.flight_started()

        prev_frame = None

        for frame_image in tqdm(self.get_all_frame_images(flight_id)):
            frame_image_path = self.get_frame_image_location(flight_id, frame_image)
            frame = cv2.imread(str(frame_image_path), cv2.IMREAD_GRAYSCALE)  # type: ignore[attr-defined]

            if prev_frame is None:
                prev_frame = frame
                continue

            results = self.tracker.predict(image=frame, prev_image=prev_frame)
            prev_frame = frame

            class_name = 'airborne'

            ax = None
            if self.visualize_pred and len(results):
                plt.imshow(frame)
                ax = plt.gca()

            for res in results:
                track_id = int(res['track_id'])
                confidence = float(res['conf'])
                cx = float(res['cx']) + float(res['offset'][0])
                cy = float(res['cy']) + float(res['offset'][1])
                w = float(res['w'])
                h = float(res['h'])

                bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                self.register_object_and_location(class_name, track_id, bbox, confidence, frame_image)

                if self.visualize_pred and ax is not None:
                    rect = matplotlib.patches.Rectangle(bbox[:2], w, h, linewidth=1, edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    plt.text(bbox[0], bbox[1], f'{int(confidence * 100)}', c='yellow')

            if self.visualize_pred and len(results):
                plt.show()


def convert_and_copy_generated_results_to_metrics_folder() -> None:
    """Transfer generated results to metrics codebase bbox format."""
    import json
    flight_results = json.loads(open("data/results/run0/result.json").read())
    for i in range(len(flight_results)):
        for j in range(len(flight_results[i]['detections'])):
            x = flight_results[i]['detections'][j]['x']
            y = flight_results[i]['detections'][j]['y']
            w = flight_results[i]['detections'][j]['w'] - x
            h = flight_results[i]['detections'][j]['h'] - y

            flight_results[i]['detections'][j]['x'] = x + w/2
            flight_results[i]['detections'][j]['y'] = y + h/2
            flight_results[i]['detections'][j]['w'] = w
            flight_results[i]['detections'][j]['h'] = h

    with open("data/evaluation/result/result.json", 'w') as fp:
        json.dump(flight_results, fp)

def get_next_run_dir(base_dir: str) -> str:
    i = 0
    while os.path.exists(os.path.join(base_dir, f"run{i}")):
        i += 1
    return f"run{i}"

def get_last_run(results_dir):
    runs = []

    for name in os.listdir(results_dir):
        match = re.match(r"run(\d+)", name)
        if match:
            runs.append((int(match.group(1)), name))
    
    if not runs:
        return None
    
    # Return the run with the highest number
    return max(runs, key=lambda x: x[0])[1]


if __name__ == "__main__":
    submission = SegPredictor()
    submission.run()

    # convert_and_copy_generated_results_to_metrics_folder()