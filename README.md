# Vision-Based Sense and Avoid: Monocular Airborne Object Tracking for Safe Drone Flight

## 1. Overview

This project tackles the challenge of detecting and tracking tiny airborne objects using only a
single monocular camera onboard a drone. Working with the Airborne Object Tracking (AOT) dataset,
a real-world collection of 5.9M+ images across diverse flight conditions, the goal is to develop
novel deep learning and multi-object tracking algorithms that achieve early detection with an
extremely low false alarm rate, enabling safe collision avoidance without unnecessary manoeuvres.

Objects in this dataset can appear as just a few pixels per frame, making this one of the most
challenging small-object detection problems in computer vision, and one of the most critical for
future autonomous flight systems.

For background on the dataset and benchmarks, see the
[Airborne Object Tracking Challenge on AIcrowd](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge).

---

<br/>

## 2. Project Details

<div align="center">

<table>
  <tr><th>Field</th><th>Details</th></tr>
  <tr><td><b>Sponsor</b></td><td>Lockheed Martin Australia – Advanced Systems & Technology Group</td></tr>
  <tr><td><b>Supervisor</b></td><td>Dr Feras Dayoub (<a href="mailto:feras.dayoub@adelaide.edu.au">email</a>)</td></tr>
  <tr><td><b>Co-Supervisor</b></td><td>Ms Kendall Louise Jenner (<a href="mailto:kendall.l.jenner@global.lmco.com">email</a>)</td></tr>
  <tr><td><b>Second Marker</b></td><td>Dr Lei Chen (<a href="mailto:lei.chen@adelaide.edu.au">email</a>)</td></tr>
</table>

</div>

---

<br/>

## 3. Project Structure

```text
Vision-Based-Sense-and-Avoid/
├── configs/
│   ├── datasets/          # Flight selections and evaluation subsets.
│   ├── dependencies/      # Python and PyTorch requirements.
│   └── models/            # Pretrained model weights and checkpoints.
│
├── core/                  # AICrowd helper library and evaluation metrics.
├── data/                  # Local AOT dataset (external symlink).
├── docs/
│   └── Images/            # Documentation figures and visual assets.
│
├── notebooks/
│   ├── baselines.ipynb    
│   ├── core_improvement.ipynb
│   ├── demo.ipynb
│   └── helpers/           # Notebook utilities and visualisation tools.
│
├── outputs/
│   ├── logs/              # Evaluation and training logs.
│   └── results/           # Generated inference & training outputs.
│
├── results/               # Processed benchmarking results and summaries.
│
├── src/
│   ├── baselines/         # SegTracker, RT-DETR v2, and YOLOv11 pipelines.
│   ├── core_improvement/  # Backbone replacement experiments.
│   └── data/              # Dataset preparation and visualisation utilities.
|
├── README.md
└── setup.sh
```

---

<br/>

## 4. Requirements

- Python 3.11.
- Virtual Environment (recommended).
- Git LFS (for the model weight files).
- AWS credentials are **NOT** required - the AOT dataset is hosted in a public S3 bucket.
- See `requirements.txt` for full package dependencies.

---

<br/>

## 5. Setup

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

<br/>

## 6. Dataset

The [Airborne Object Tracking (AOT) Dataset](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge#dataset) is a large-scale benchmark for airborne object detection and tracking. It contains high-resolution flight sequences captured from onboard aircraft cameras under diverse environmental and operational conditions. A key challenge of the dataset is that target aircraft often occupy only a few pixels in each frame, with airborne objects averaging approximately **0.01% of the image area**. This makes early detection particularly difficult and provides a realistic benchmark for vision-based Sense-and-Avoid (SAA) systems.

<div align="center">

| Dataset Property     | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| **Full Dataset**     | ~11 TB stored in the public AOT S3 bucket                    |
| **Partial Dataset**  | ~500 GB containing valid planned encounters (`partial=True`) |
| **Flights**          | ~2,332 flight sequences                                      |
| **Frame Rate**       | 10 FPS                                                       |
| **Typical Duration** | ~120 seconds per flight                                      |
| **Resolution**       | 2448 × 2048 pixels                                           |

</div>

### 6.1 Evaluation Metrics

Performance is evaluated using the official Airborne Object Tracking Challenge metrics, which balance early detection capability against false alarm generation.

<div align="center">

| Metric | Description |
|---------|-------------|
| **AFDR** (Average Frame-level Detection Rate) | Percentage of airborne objects successfully detected at the frame level, considering planned aircraft within 700 m. |
| **EDR** (Encounter Detection Rate) | Percentage of encounters successfully detected. An encounter is considered detected when the target is tracked for at least 3 seconds before reaching 300 m, or within the first 3 seconds of the encounter. |
| **FPPI** (False Positives Per Image) | Average number of false positive detections generated per image. Lower values indicate fewer false alarms. |
| **HFAR** (False Alarm Rate per Hour) | Number of unique false-positive tracks reported per hour of flight data. Lower values indicate greater operational reliability. |

</div>

### 6.2 Example Flight Visualisation

The example below shows a complete annotated flight sequence with object detections, identifiers, and range information overlaid on each frame.

<div align="center">
    <img src="docs/Images/Data_Viewer_Video.gif" width="900">
</div>

### 6.3 Data Tools

Several utility scripts are provided to support dataset exploration, preparation, visualisation, and results analysis. All scripts are executed from the project root.

1. **`browse_data.py`:** Browses dataset ground truth metadata entry by entry without downloading any images. Use this to explore object distances, bounding box sizes, and horizon classifications to decide which flights are worth downloading.

   ```bash
   python -m src.data.browse_data
   ```

2. **`data_builder.py`:** Filters ground truth metadata using configurable criteria to identify challenging flights. Saves the selected flight IDs to a .txt file for use with download_data.py.

   ```bash
   python -m src.data.data_builder
   ```

3. **`download_data.py`:** Reads the selected flight IDs saved during browsing and downloads the full image sequences for each flight from S3. Prompts for confirmation before starting and displays an estimated total download size.

   ```bash
   python -m src.data.download_data
   ```

4. **`view_data.py`:** Visualises downloaded flights either as an annotated frame-by-frame OpenCV viewer or as an exported H.264 `.mp4` video. Bounding boxes, object IDs, and distances are drawn on every detection.

   ```bash
   python -m src.data.view_data
   ```

5. **`split_dataset.py`:** Generate reproducible train, validation, and test splits from a flight list. The script ensures there is no overlap between splits and records metadata required for reproducibility.

   ```bash
   python -m src.data.split_dataset
   ```

6. **`visualise_results.py`:** Generate comparative performance visualisations from evaluation results. The script creates a grid of scatter plots showing the trade-offs between detection performance, model complexity, and inference speed, and exports the resulting figure as a high-resolution PNG suitable for reports and technical documentation.

   ```bash
   python -m src.data.visualise_results
   ```

### 6.4 Running Inference & Evaluation

To reproduce the results shown in this README, inference and evaluation are provided through two main workflows:

1. **Notebook-Based Workflow:** Full inference and analysis can be run using the notebooks in the `notebooks/` directory. These provide an interactive workflow for both baseline evaluation and core improvement experiments.

2. **Script-Based Workflow (Recommended):** Automated evaluation scripts are provided for reproducible benchmarking:

   ```bash
   src/baselines/baslines_evaluator.py
   ```
   ```bash
   src/core_improvement/core_improvement_evaluator.py
   ```

Each script includes its own internal documentation and can be executed directly from the terminal. The script-based workflow is the recommended approach for reproducible evaluation, while notebooks are primarily intended for experimentation, debugging, and visual analysis.

---

<br/>

## 7. Baseline Evaluation

### 7.1 Baseline Model

The baseline for this project is the [Winning Solution](https://github.com/pdima/airborne_tracking_winning_solution.git) from the AIcrowd Airborne Object Tracking Challenge, which achieved top performance on both the public and private leaderboards. The method is based on an ensemble of four detection backbones:

<div align="center">

<table>
  <tr><th>Baseline Backbones</th></tr>
  <tr><td>HRNet-W32</td></tr>
  <tr><td>HRNet-W48</td></tr>
  <tr><td>GerNet-M (with BiFPN from EfficientDet-D2)</td></tr>
  <tr><td>DLA-60</td></tr>
</table>

</div>

These backbones are combined with a motion-offset-based tracking pipeline for consistent multi-frame association of airborne objects.


### 7.2 Evaluation Dataset

All baseline evaluations are performed on the **AOT Hard Dataset (1,618 flights)**. This dataset is a curated subset of the full AOT benchmark designed to focus on challenging operational scenarios, including:

<div align="center">

<table>
  <tr><th>Evaluation Scenarios</th></tr>
  <tr><td>Small object encounters</td></tr>
  <tr><td>Multi-object scenes</td></tr>
  <tr><td>Below-horizon targets</td></tr>
  <tr><td>Flights with no visible objects</td></tr>
</table>

</div>

This subset provides a more challenging evaluation environment compared to the original full competition dataset.

### 7.3 Baseline Results

Performance of the baseline model on the **AOT Hard Dataset (1,618 flights)**, broken down by scenario type.

<div align="center">

<table>
<tr>
<th>Scenario</th>
<th>Flights</th>
<th>FPPI</th>
<th>AFDR</th>
<th>HFAR</th>
<th>EDR (Total)</th>
<th>EDR (Above)</th>
<th>EDR (Mixed)</th>
<th>EDR (Below)</th>
</tr>

<tr>
<td><b>Overall (All Criteria)</b></td>
<td>1,618</td>
<td>0.000016</td>
<td>0.7374</td>
<td>0.0887</td>
<td>0.6948</td>
<td>0.6366</td>
<td>0.5724</td>
<td>0.7605</td>
</tr>

<tr>
<td>Small Object (bbox ≤ 36 px²)</td>
<td>662</td>
<td>0.00</td>
<td>0.7212</td>
<td>0.00</td>
<td>0.6318</td>
<td>0.6364</td>
<td>0.4118</td>
<td>0.6761</td>
</tr>

<tr>
<td>Below Horizon</td>
<td>1,053</td>
<td>0.00</td>
<td>0.7572</td>
<td>0.00</td>
<td>0.7028</td>
<td>0.3158</td>
<td>0.5333</td>
<td>0.7746</td>
</tr>

<tr>
<td>Multi-Object</td>
<td>627</td>
<td>0.00</td>
<td>0.7082</td>
<td>0.00</td>
<td>0.6990</td>
<td>0.6970</td>
<td>0.4444</td>
<td>0.7385</td>
</tr>

<tr>
<td>No Visible Object</td>
<td>354</td>
<td>0.000042</td>
<td>0.6026</td>
<td>0.2400</td>
<td>0.2297</td>
<td>0.2727</td>
<td>0.1667</td>
<td>0.1667</td>
</tr>

</table>

</div>

<p align="center">
  <em>Results are evaluated at a minimum detection confidence of 0.60 and a minimum track length of 30 frames (≈ 3 seconds at 10 fps).</em>
</p>

---
