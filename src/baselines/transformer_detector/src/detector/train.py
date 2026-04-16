#!/usr/bin/env python
from dataclasses import dataclass
from datasets import load_dataset
from huggingface_hub import configure_http_backend
from pprint import pprint
from transformers import AutoImageProcessor
from torch.utils.data import Dataset
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer

import albumentations as A
import argparse
import torch
import requests
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="Model checkpoint to use for inference",
    default="PekingU/rtdetr_v2_r50vd",
)
parser.add_argument("--data_dir", default="./data/aod4")
parser.add_argument("--image_size", default=1024)
parser.add_argument("--certfile_path", default="$HOME/.certs/Combined_pem.pem")
parser.add_argument("--output_directory", default="./models/finetune/new_model")
parser.add_argument(
    "--category_map",
    default=(
        {0: "drone", 1: "Aeroplane", 2: "Bird", 3: "Drone", 4: "Helicopter"},
        {"drone": 0, "Aeroplane": 1, "Bird": 2, "Drone": 3, "Helicopter": 4},
    ),
)

args = parser.parse_args()
CHECKPOINT = args.model
DATA = args.data_dir
CERTS = args.certfile_path
CATEGORY_MAP = args.category_map
IMAGE_SIZE = args.image_size
OP_DIR = args.output_directory


# configure ssl certs
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = CERTS
    return session


configure_http_backend(backend_factory=backend_factory)

dataset = load_dataset(DATA)

if "validation" not in dataset:
    split = dataset["train"].train_test_split(0.15, seed=1337)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]


# Get mapping from category id to category name
id2label = CATEGORY_MAP[0]
label2id = CATEGORY_MAP[1]

image_processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=True,
    size={"width": IMAGE_SIZE, "height": IMAGE_SIZE},
    use_fast=True,
)

train_augmentation_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category"],
        clip=True,
        min_area=25,
        min_width=1,
        min_height=1,
        filter_invalid_bboxes=True,
    ),
)

# to make sure boxes are clipped to image size and there is no boxes with area < 1 pixel
validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category"],
        clip=True,
        min_area=1,
        min_width=1,
        min_height=1,
        filter_invalid_bboxes=True,
    ),
)


class TransformerDataset(Dataset):
    def __init__(self, dataset, image_processor, transform=None):
        self.dataset = dataset
        self.image_processor = image_processor
        self.transform = transform

    @staticmethod
    def format_image_annotations_as_coco(image_id, categories, boxes):
        """Format one set of image annotations to the COCO format

        Args:
            image_id (str): image id. e.g. "0001"
            categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
            boxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
                ([center_x, center_y, width, height] in absolute coordinates)

        Returns:
            dict: {
                "image_id": image id,
                "annotations": list of formatted annotations
            }
        """
        annotations = []
        for category, bbox in zip(categories, boxes):
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": list(bbox),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image_id = sample["image_id"]
        image = sample["image"]
        boxes = sample["objects"]["bbox"]
        categories = sample["objects"]["category"]

        # Convert image to RGB numpy array
        image = np.array(image.convert("RGB"))

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category=categories)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        # Format annotations in COCO format for image_processor
        formatted_annotations = self.format_image_annotations_as_coco(
            image_id, categories, boxes
        )

        # Apply the image processor transformations: resizing, rescaling, normalization
        result = self.image_processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )

        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result


# use the standard labelling format

train_dataset = TransformerDataset(
    dataset["train"], image_processor, transform=train_augmentation_and_transform
)
validation_dataset = TransformerDataset(
    dataset["validation"], image_processor, transform=validation_transform
)
test_dataset = TransformerDataset(
    dataset["test"], image_processor, transform=validation_transform
)


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:
    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            # -------------------------------
            # quick fix here - sort out later
            batch_image_sizes = torch.tensor([[1024, 1024] for _ in range(len(batch))])
            # ------------------------------

            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, image_size_batch):
                # here we have "yolo" format (x_center, y_center, width, height) in relative coordinates 0..1
                # and we need to convert it to "pascal" format (x_min, y_min, x_max, y_max) in absolute coordinates
                if size.numel() == 1:
                    height = width = size
                else:
                    height, width = size

                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([[width, height, width, height]])

                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
            )
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):

        predictions, targets = (
            evaluation_results.predictions,
            evaluation_results.label_ids,
        )

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class
        ):
            class_name = (
                id2label[class_id.item()] if id2label is not None else class_id.item()
            )
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


eval_compute_metrics_fn = MAPEvaluator(
    image_processor=image_processor, threshold=0.01, id2label=id2label
)


model = AutoModelForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir=OP_DIR,
    num_train_epochs=25,
    max_grad_norm=1,
    learning_rate=5e-5,
    warmup_steps=300,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    dataloader_num_workers=4,
    metric_for_best_model="eval_map_small",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=10,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    report_to="tensorboard",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()

metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval")
pprint(metrics)
