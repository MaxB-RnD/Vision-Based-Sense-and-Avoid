import json
import os
from tqdm import tqdm

"""
This script rewrites the annotations json into a 
jsonl where each line of the file is a dict with the structure:

{'file_name': *.jpg,
'objects': {'bbox': list[list]
            'class_labels': list[int]
            'id': list[int]
            'area': list[int]
            }
        }

Where each list will have a length equal to the number
of detections in the image

"""

json_path = ""
with open(json_path) as file:
    data = json.load(file)

height, width = ""


image_names = []
image_names.extend(os.listdir(""))

img_id_lookup = {image["id"]: image["file_name"] for image in data["images"]}
id_img_lookup = {value: key for key, value in img_id_lookup.items()}

object_lookup = {}
id_set = set()
for ann in tqdm(data["annotations"]):
    if ann["image_id"] not in id_set:
        id_set.add(ann["image_id"])
        object_lookup[img_id_lookup[ann["image_id"]]] = {
            "ann_idx": [ann["id"]],
            "bbox": [
                [
                    int(round(ann["bbox"][0])),
                    int(round(ann["bbox"][1])),
                    int(round(ann["bbox"][2])),
                    int(round(ann["bbox"][3])),
                ]
            ],
            "id": [ann["image_id"]],
            "category": [ann["category_id"]],
            "area": [ann["area"]],
            "width": width,
            "height": height,
        }  # maps file name to object for new files
    else:
        object_lookup[img_id_lookup[ann["image_id"]]]["bbox"].append(
            [
                int(round(ann["bbox"][0])),
                int(round(ann["bbox"][1])),
                int(round(ann["bbox"][2])),
                int(round(ann["bbox"][3])),
            ]
        )
        object_lookup[img_id_lookup[ann["image_id"]]]["category"].append(
            ann["category_id"]
        )
        object_lookup[img_id_lookup[ann["image_id"]]]["ann_idx"].append(ann["id"])
        object_lookup[img_id_lookup[ann["image_id"]]]["area"].append(ann["area"])
        # appends data to the object list for seen files

lines = [
    {
        "file_name": name,
        "image_id": id_img_lookup[name],
        "objects": object_lookup[name],
    }
    for name in image_names
    if name in object_lookup.keys()
]


with open("", "wt", encoding="utf-8") as f:
    for record in lines:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
