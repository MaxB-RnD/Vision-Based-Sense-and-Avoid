import os
import json
import argparse
from PIL import Image
import supervision
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("image_dir", help="top level image directory")
args = parser.parse_args()
image_dir = args.image_dir


"""
# for aot
id2label = {0: "Helicopter", 1:'Bird', 2:'Airborne', 3:'Flock', 4:'Airplane', 5:'Drone'}
"""
# for aod
id2label = {1: "Aeroplane", 2: "Bird", 3: "Drone", 4: "Helicopter"}
for subset in os.listdir(image_dir):
    if subset in {"train"}:
        data = []
        with open(image_dir + subset + "/metadata.jsonl", "r") as file:
            for line in file:
                data.append(json.loads(line))

        for instance in tqdm(data):
            image_name = instance["file_name"]
            bbox = instance["objects"]["bbox"]
            class_id = instance["objects"]["category"]

            image = Image.open(os.path.join(image_dir, subset, image_name))
            annotated_image = image.copy()
            annotated_image = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
            annotated_image = supervision.LabelAnnotator().annotate(
                annotated_image,
                supervision.Detections(
                    xyxy=(
                        np.array(
                            [
                                [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
                                for bb in bbox
                            ]
                        )
                    ),
                    class_id=np.array(class_id),
                ),
                labels=[id2label[id] for id in class_id],
            )
            annotated_image = supervision.BoxAnnotator().annotate(
                annotated_image,
                supervision.Detections(
                    xyxy=(
                        np.array(
                            [
                                [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
                                for bb in bbox
                            ]
                        )
                    ),
                    class_id=np.array(class_id),
                ),
            )
            write_path = image_dir + "/gt_annotations/" + subset + "/gt_" + image_name
            cv2.imwrite(write_path, np.array(annotated_image))
