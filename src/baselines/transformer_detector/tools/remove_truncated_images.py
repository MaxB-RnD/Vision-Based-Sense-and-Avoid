#!/usr/bin/env python3
"""
clean_truncated_images.py  -*- coding: utf-8 -*-

Recursively (or non recursively) scan a directory for image files that Pillow
cannot load, and delete them.

Requirements:
  - Python 3.8+
  - Pillow (install via pip install pillow)

Example usage:

    # Show what would be removed, without deleting anything
    python cleanuncated_images.py --dry-run /path/to/images

    # Actually delete every broken image
    python clean_truncated_images.py /path/to/images

    # Only target JPEG and PNG extensions
    python clean_truncated_images.py --ext .jpg,.jpeg,.png /path/to/images

    # Include subdirectories
    python clean_truncated_images.py --recursive /path/to/images
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("ERROR: Pillow is not installed. Run: pip install pillow", file=sys.stderr)
    sys.exit(1)


def is_image_file(path: Path, allowed_exts=None) -> bool:
    """Return True if path is a file that looks like an image by extension."""
    if not path.is_file():
        return False
    if allowed_exts:
        return path.suffix.lower() in allowed_exts
    # Default set of common imager extensions
    return path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".ico",
        ".heif",
        ".heic",
        ".avif",
    }


def is_broken_image(path: Path) -> bool:
    """Return True if Pillow cannot open & load the whole image."""
    try:
        with Image.open(path) as img:
            # .load() forces Pillow to read the full image data
            img.load()
        return False
    except (OSError, SyntaxError, UnidentifiedImageError):
        # Any of these indicates that the file is not a valid image
        return True


def process_directory(
    root: Path,
    recursive: bool,
    dry_run: bool,
    allowed_exts=None,
):
    """Walk a directory and delete broken images."""
    metrics = {"total": 0, "deleted": 0, "skipped": 0, "broken": 0}
    walker = root.rglob("*") if recursive else root.glob("*")

    for entry in tqdm(walker):
        if not entry.is_file():
            continue

        if not is_image_file(entry, allowed_exts):
            continue

        metrics["total"] += 1

        if is_broken_image(entry):
            metrics["broken"] += 1
            if dry_run:
                print(f"[DRY RUN] Would delete: {entry}")
                metrics["skipped"] += 1
            else:
                try:
                    entry.unlink()
                    print(f"Deleted: {entry}")
                    metrics["deleted"] += 1
                except Exception as exc:
                    print(f"[ERROR] Could not delete {entry}: {exc}")
        else:
            metrics["skipped"] += 1

    # Summary
    print("\n=== CLEANUP SUMMARY ===")
    print(f"Total image files found: {metrics['total']}")
    print(f"Broken images detected: {metrics['broken']}")
    print(f"Images deleted: {metrics['deleted']}")
    print(f"Images skipped: {metrics['skipped']}")
    if dry_run:
        print("DRY RUN – nothing was actually removed.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        type=Path,
        help="Root path to scan for images",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan recursively into sub‑directories",
    )
    p.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Show what would be deleted; do not actually delete",
    )
    p.add_argument(
        "-e",
        "--ext",
        default="",
        help="Comma‑separated list of extensions to process (e.g. .jpg,.png). "
        "If omitted, a broad set of common image types is used.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.path.is_dir():
        print(f"ERROR: {args.path!s} is not a directory.", file=sys.stderr)
        sys.exit(1)

    allowed_exts = None
    if args.ext:
        # Normalize: strip whitespace, convert to lower case, ensure leading dot
        allowed_exts = {
            ext.strip().lower()
            if ext.strip().startswith(".")
            else f".{ext.strip().lower()}"
            for ext in args.ext.split(",")
            if ext.strip()
        }

    process_directory(
        root=args.path,
        recursive=args.recursive,
        dry_run=args.dry_run,
        allowed_exts=allowed_exts,
    )


if __name__ == "__main__":
    main()
