#!/usr/bin/env python3

import argparse
import time
from tqdm import tqdm

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input video (or image) file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output.mp4",
        help="File name for the resulting video with boxes.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        metavar="N",
        help="Maximum frames to read (default: all). Note: >0.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        metavar="K",
        help="Number of frames to skip between processed ones (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="B",
        help="GPU batch size (default: 4).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="Detection confidence threshold."
    )
    parser.add_argument("--model", type=str, default="PekingU/rtdetr_r18vd_coco_o365")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    return parser.parse_args()


def load_frames(
    cap, max_frames: int, skip: int, frame_limit: int, skip_flag: bool = True
):
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % skip == 0:
            yield frame
            saved += 1
            if 0 < max_frames <= saved:
                break
        frame_idx += 1
        # Stop when we have processed the maximum frames *in theory*.
        if 0 < max_frames <= frame_idx // skip:
            break
    print(f"\nLoaded {saved} frames treating every {skip}th frame.")


def process_batch(
    batch, image_processor, model, device, threshold: float, target_labels=set()
):
    """
    Perform inference on a *batch* of frames and overlay
    bounding boxes for the target labels.
    """
    # Convert list of cv2 frames (BGR) to PIL images (RGB)
    pil_images = [
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch
    ]

    inputs = image_processor(images=pil_images, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([im.size[::-1] for im in pil_images]).to(device)

    # Obtain detection results for each image
    results = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )

    # Draw
    for sel_img, det in zip(batch, results):
        for score, label, box in zip(det["scores"], det["labels"], det["boxes"]):
            # Only plot wanted classes
            text = model.config.id2label[label.item()]
            if text not in target_labels:
                continue

            x, y, x2, y2 = [int(v.item()) for v in box]  # convert to int
            # colour = (0, 0, 255)  # red in BGR
            cv2.rectangle(sel_img, (x, y), (x2, y2), (0, 0, 255), 1)

            txt = f"{text} [{score.item():.2f}]"
            cv2.putText(
                sel_img,
                txt,
                (x, y - 5 if y - 5 > 10 else y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return batch  # modified in place


def write_video(frames, output_path: str, fps: float, codec="mp4v"):
    if not frames:
        raise RuntimeError("No frames to write")

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for idx, frame in enumerate(frames):
        writer.write(frame)
    writer.release()


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model_repo = args.model
    print("\nLoading model…")
    image_processor = AutoImageProcessor.from_pretrained(model_repo)
    model = AutoModelForObjectDetection.from_pretrained(model_repo)
    model = model.to(device)
    model.eval()

    # Target labels we care about
    TARGET_LABELS = {"airplane", "bird", "kite"}

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open {args.input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    if args.max_frames > 0 and args.max_frames > total_frames:
        max_frames = total_frames
        print(
            f"Requested {args.max_frames} frames, but the file only has {total_frames}."
        )

    processed_frames = []
    batch = []
    bar = tqdm(total=max_frames, desc="Frames processed")
    start = time.perf_counter()

    for frame in load_frames(cap, max_frames, args.skip, args.max_frames):
        batch.append(frame)
        if len(batch) == args.batch_size:
            processed_frames.extend(
                process_batch(
                    batch,
                    image_processor,
                    model,
                    device,
                    args.threshold,
                    TARGET_LABELS,
                )
            )
            torch.cuda.empty_cache()  # help with memory churn
            batch = []
            bar.update(args.batch_size)
    # process the tail
    if batch:
        processed_frames.extend(
            process_batch(
                batch,
                image_processor,
                model,
                device,
                args.threshold,
                TARGET_LABELS,
            )
        )
        bar.update(len(batch))

    cap.release()
    bar.close()
    elapsed = time.perf_counter() - start
    print(
        f"\nInference finished in {elapsed:.2f}s. Total frames written: {len(processed_frames)}"
    )

    print("\nWriting output…")
    write_video(processed_frames, args.output_path, args.fps)

    print(f"video written to {args.output_path}")


if __name__ == "__main__":
    main()
