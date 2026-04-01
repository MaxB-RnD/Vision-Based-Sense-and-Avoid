# Project Name

## 📑 Contents

- [🛸 Overview](#-overview)  
- [📋 Project Details](#-project-details)  
- [🗂️ Project Structure](#️-project-structure)  
- [⚙️ Requirements](#️-requirements)  
- [🛠️ Setup](#️-setup)  
- [🗃️ Dataset](#️-dataset)  
  - [Data Tools](#data-tools)  
  - [Visualisation](#visualisation)  
- [📊 Baseline](#-baseline)  
  - [Winning Solution](#winning-solution)  
  - [Competition Results (Full AOT Dataset)](#competition-results-full-aot-dataset)  
  - [Hard Dataset Results](#hard-dataset-results)

---

## 🛸 Overview

A brief description of what this project does, why it exists, and what problem it solves.  
Keep this high-level and focused on intent rather than implementation details.

---

## 🗂️ Project Structure

```text
Vision-Based-Sense-and-Avoid/
├── configs/ 
│   ├── datasets/     # Flight ID sets for different custom libraries.
│   ├── models/       # Trained model weights (.pth files).
├── core/             # Competition helper library (AOT dataset access).
├── docs/             # Research documentation and additional resources.
├── notebooks/        # End-to-end model evaluation notebooks.
├── src/
│   ├── data/         # Dataset selection, cleaning, and visualization scripts.
│   └── baseline/     # Baseline solution scripts and models for evaluation.
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.11.
- Virtual Environment (recommended).
- Git LFS (for the model weight files).
- AWS credentials are **NOT** required - the AOT dataset is hosted in a public S3 bucket.
- See `requirements.txt` for full package dependencies.

---

## 🛠️ Setup

**1. Install Git LFS (one-time setup):**

   ```bash
   git lfs install
   ```

**2. Clone this Repository:**

   ```bash
   git clone https://github.com/MaxB-RnD/Vision-Based-Sense-and-Avoid.git
   cd Vision-Based-Sense-and-Avoid
   ```

**3. Download Git LFS Files:**

   ```bash
   git lfs pull
   ```

   > ⚠️ *If you skip this step, the model weights will appear as pointer files instead of actual content.*

**4. Create & Activate a Virtual Environment:**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate        # Linux/macOS/WSL
   .venv\Scripts\activate           # Windows
   ```

**5. Install Dependencies:**

   ```bash
      bash setup.sh
   ```

   > *This installs all dependencies from `requirements.txt` and the airborne metrics evaluation package from `core/metrics`.*

---

## 🗃️ Dataset

The [Airborne Object Tracking (AOT) Dataset](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge#dataset) is a collection of high-resolution flight sequences captured onboard aerial vehicles. Airborne objects average just 0.01% of the image size (down to a few pixels) making it a challenging benchmark for real-world sense-and-avoid detection.

|                       |                                                                  |
|-----------------------|------------------------------------------------------------------|
| **Full Dataset**      | ~11TB via `s3://airborne-obj-detection-challenge-training/`.     |
| **Partial Dataset**   | ~500GB containing valid planned encounters only (`partial=True`).|
| **Flights**           | ~2,332 sequences across 3 parts.                                 |
| **Frames per Flight** | ~120s at 10 fps (~4–5GB per flight).                             |
| **Resolution**        | 2448 × 2048 px.                                                  |

Brief instructions on how to build or set up the project.

Examples:
- How to configure and build.
- How to install dependencies.
- Any required environment variables.

---

2. **`data_builder.py`:** Filters ground truth metadata using configurable criteria to identify challenging flights. Saves the selected flight IDs to a .txt file for use with download_data.py.

   ```bash
   python -m src.data.data_builder
   ```

3. **`download_data.py`:** Reads the selected flight IDs saved during browsing and downloads the full image sequences for each flight from S3. Prompts for confirmation before starting and displays an estimated total download size.

Basic examples of how to use or run the project.

4. **`view_data.py`:** Visualises downloaded flights either as an annotated frame-by-frame OpenCV viewer or as an exported H.264 `.mp4` video. Bounding boxes, object IDs, and distances are drawn on every detection.

---

## Development Notes

Optional section for:
- Design decisions.
- Known limitations.
- Future improvements.
- Gotchas or assumptions.

---

## License

---

## 📊 Baseline

### Winning Solution

The baseline used in this project is the [Winning Solution](https://github.com/pdima/airborne_tracking_winning_solution.git) 
from the AIcrowd Airborne Object Tracking Challenge, achieving the highest scores
on both the public and private leaderboards. The solution uses an ensemble of
three detection backbones, HRNet32, gernet_m (with BiFPN from EfficientDet D2),
and DLA60, combined with a predicted offset-based tracking algorithm.

**Model weights are stored in `data/models/baseline/` via Git LFS:**

| Weight File                                  | Description                        |
|----------------------------------------------|------------------------------------|
| `030_tr_tsn_rn34_w3_crop_borders_255_0.pth`  | ResNet34 Crop Border Baseline.     |
| `120_hrnet32_all_2220.pth`                   | HRNet32 Backbone.                  |
| `120_gernet_m_b2_all_2220.pth`               | GerNet-M with BiFPN (EfficientDet).|
| `120_dla60_256_sgd_all_rerun_2220.pth`       | DLA60 Backbone (Winning Ensemble). |
| `130_hrnet48_all_2220.pth`                   | HRNet48 Backbone.                  |

### Competition Results (Full AOT Dataset)

Performance of the winning solution (Submission 22) on the AIcrowd competition leaderboards:

| Leaderboard | Score Threshold | FPPI    | AFDR     | HFAR | EDR      |
|-------------|-----------------|---------|----------|------|----------|
| Public      | 0.60            | 5.0e-05 | **0.91** | 0.38 | **0.99** |
| Private     | 0.60            | 7.8e-05 | 0.89     | 0.38 |  0.96    |

**Metric Definitions:**

- **AFDR** *(Average Frame-level Detection Rate)*: Ratio of detected airborne objects to all objects that should be detected, considering all planned aircraft within 700m.
- **FPPI** *(False Positives Per Image)*: Ratio of false positive airborne reports to the total number of images in the dataset.
- **EDR** *(Encounter-level Detection Rate)*: Ratio of successfully detected encounters to total encounters, where an encounter is successfully detected if the object is tracked for at least 3 seconds before reaching 300m or within the first 3 seconds of the encounter.
- **HFAR** *(False Alarm Rate per Hour)*: Number of unique reported track IDs containing at least one false positive, divided by total dataset hours.

### Hard Dataset Results

The table below reports baseline performance broken down by scenario category across the custom
hard dataset (1,618 flights). Flights were selected from the full AOT dataset using configurable
filter criteria to isolate the most challenging sequences, providing a more targeted benchmark
than the full competition dataset.

| Scenario                     | Flights | FPPI | AFDR | HFAR | EDR (Total) | EDR (Above) | EDR (Mixed) | EDR (Below)
|------------------------------|---------|------|------|------|-------------|
| **Overall (All Criteria)**   |**1,618**| —    | —    | —    | —           |
| Small Object (bbox ≤ 36 px²) | 662     | —    | —    | —    | —           |
| Below Horizon                | 1,053   | —    | —    | —    | —           |
| Multi-Object                 | 627     | —    | —    | —    | —           |
| No Visible Object            | 354     | —    | —    | —    | —           |

*Results evaluated at a minimum detection confidence of 0.60 and a minimum track length of 30 frames (≈ 3 seconds at 10 fps).*

---
