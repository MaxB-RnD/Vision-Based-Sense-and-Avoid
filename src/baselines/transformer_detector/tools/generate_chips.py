from shapely import Polygon
import shapely
from PIL import Image
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

x0, x1, x2, x3 = 0, 599, 1208, 1848
(
    y0,
    y1,
    y2,
    y3,
) = 0, 469, 1237, 1408
y_idx = [y0, y1, y2, y3]
x_idx = [x0, x1, x2, x3]

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Name of top level of the dataset")
parser.add_argument("output_dir", help="Name of directory to write chips to")
args = parser.parse_args()
data_dir = args.data_dir
op_dir = args.output_dir


def write_chips(image, ann, name, image_id):
    chips = {}
    annotations = []
    # get the image
    arr = np.array(image)
    idx = 0  # index which chip we're on
    for x in x_idx:
        for y in y_idx:
            chip_id = f"{idx:02d}"
            # create the chip in this virtual polygon space
            chip_dims = Polygon(
                ((x, y), (x, y + 639), (x + 639, y + 639), (x + 639, y), (x, y))
            )
            # instantiate a dict to hold annotations for this chip
            info = {"objects": {"bbox": [], "category": [], "area": []}}

            # for rgb
            if arr.shape[0] == 3:
                chip = arr[:, y : y + 640, x : x + 640]
            # for single channel
            else:
                chip = arr[y : y + 640, x : x + 640]
            # this indexing might need to change if the images index down, right
            # as opposed to up, right as in our polygon space

            info["image"] = chip
            bbox_count = 0
            for bbox in ann["bbox"]:
                bbox = Polygon(
                    (
                        (bbox[0], bbox[1]),  # bottom left
                        (bbox[0], bbox[1] + bbox[3]),  # top left
                        (bbox[0] + bbox[2], bbox[1] + bbox[3]),  # top right
                        (bbox[0] + bbox[2], bbox[1]),  # bottom right
                        (bbox[0], bbox[1]),  # bottom left
                    )
                )

                if shapely.intersects(bbox, chip_dims):
                    new_bbox = shapely.intersection(chip_dims, bbox)
                    abs_bounds = new_bbox.bounds
                    trans_bounds = [
                        abs_bounds[0] - x,
                        abs_bounds[1] - y,
                        abs_bounds[2] - abs_bounds[0],
                        abs_bounds[3] - abs_bounds[1],
                    ]
                    info["objects"]["bbox"].append(trans_bounds)
                    info["objects"]["category"].append(ann["category"][bbox_count])
                    info["objects"]["area"].append(shapely.area(new_bbox))
                bbox_count += 1

            if info["objects"]["bbox"]:
                if all(area > 0 for area in info["objects"]["area"]):
                    chips[str(idx) + name] = chip
                    annotations.append(
                        {
                            "file_name": str(idx) + name,
                            "image_id": int(str(image_id) + chip_id),
                            "objects": info["objects"],
                        }
                    )
            idx += 1

    return chips, annotations

    # return a list of valid chip images and their corresponding annotation dictionaries


for subset in os.listdir(data_dir):
    if subset in {"train", "test", "validation"}:
        with open(data_dir + subset + "/metadata.jsonl", "r") as metadata:
            data = [json.loads(line) for line in metadata]

        output_metadata = []
        for line in tqdm(data):
            image = Image.open(os.path.join(data_dir, subset, line["file_name"]))
            ann = line["objects"]

            chips, annotations = write_chips(
                image, ann, line["file_name"], line["image_id"]
            )
            output_metadata.extend(annotations)

            for chip in chips.keys():
                image = Image.fromarray(chips[chip])
                image.save(os.path.join(op_dir, subset, chip))

        with open(
            os.path.join(op_dir, subset, "metadata.jsonl"), "wt", encoding="utf-8"
        ) as f:
            for record in output_metadata:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
