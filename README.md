# Vision-Based Sense and Avoid: Monocular Airborne Object Tracking for Safe Drone Flight

## 🛸 Overview

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

## 📋 Project Details

| Field            | Details                                                                                                |
|------------------|--------------------------------------------------------------------------------------------------------|
| **Sponsor**      | Lockheed Martin Australia – Advanced Systems & Technology Group                                        |
| **Supervisor**   | Dr Feras Dayoub ([feras.dayoub@adelaide.edu.au](mailto:feras.dayoub@adelaide.edu.au))                  |
| **Co-Supervisor**| Ms Kendall Louise Jenner ([kendall.l.jenner@global.lmco.com](mailto:kendall.l.jenner@global.lmco.com)) |
| **Second Marker**| Dr Lei Chen ([lei.chen@adelaide.edu.au](mailto:lei.chen@adelaide.edu.au))                              |

---

## 🤖 Features

> _To be completed as the project develops._

---

## 🗂️ Project Structure

```text
Vision-Based-Sense-and-Avoid/
├── core/           # Competition Repo Helper Library (AOT Dataset Access).
├── data/           # Stores Flight ID Sets for Different Custom Libraries.
├── docs/           # Further Information on Research.
├── src/
│   └── data/       # Dataset Selection, Cleaning, and Visualisation Scripts.
│       ├── browse_data.py
│       ├── download_data.py
│       └── view_data.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8.
- Virtual Environment (recommended).
- AWS credentials are **NOT** required — the AOT dataset is hosted in a public S3 bucket.
- See `requirements.txt` for full package dependencies.

---

## 🛠️ Setup

**1. Clone this Repository:**

   ```bash
   git clone https://github.com/MaxB-RnD/Vision-Based-Sense-and-Avoid.git
   cd Vision-Based-Sense-and-Avoid
   ```

**2. Create & Activate a Virtual Environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Linux/macOS/WSL
   .venv\Scripts\activate           # Windows
   ```

**3. Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🗃️ Dataset

The [Airborne Object Tracking (AOT) Dataset](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge#dataset) is a collection of high-resolution flight sequences captured onboard aerial vehicles. Airborne objects average just 0.01% of the image size (down to a few pixels) making it a challenging benchmark for real-world sense-and-avoid detection. 

|                       |                                                              |
|-----------------------|--------------------------------------------------------------|
| **Full Dataset**      | ~11TB via `s3://airborne-obj-detection-challenge-training/`. |
| **Partial Dataset**   | ~500GB containing valid planned encounters only (`partial=True`).     |
| **Flights**           | ~2,332 sequences across 3 parts.                             |
| **Frames per Flight** | ~120s at 10 fps (~4–5GB per flight).                         |
| **Resolution**        | 2448 × 2048 px.                                              |

### Data Tools

Three scripts are provided to **browse, select, download,** and visualise flights from the dataset. All scripts are run from the project root.

1. **`browse_data.py`:** Browses dataset ground truth metadata entry by entry without downloading any images. Use this to explore object distances, bounding box sizes, and horizon classifications to decide which flights are worth downloading.

   ```bash
   python -m src.data.browse_data
   ```

2. **`download_data.py`:** Reads the selected flight IDs saved during browsing and downloads the full image sequences for each flight from S3. Prompts for confirmation before starting and displays an estimated total download size.

   ```bash
   python -m src.data.download_data
   ```

3. **`view_data.py`:** Visualises downloaded flights either as an annotated frame-by-frame OpenCV viewer or as an exported H.264 `.mp4` video. Bounding boxes, object IDs, and distances are drawn on every detection.

   ```bash
   python -m src.data.view_data
   ```

### Visualisation

| **Frame 1**                                                          | **Frame 2**                                                     | **Frame 3**                                                      | **Annotated Video**                                          |
|:--------------------------------------------------------------------:|:---------------------------------------------------------------:|:----------------------------------------------------------------:|:------------------------------------------------------------:|
| ![Frame 1: Object at Distance](docs/images/Data_Viewer_Picture_1.png)| ![Frame 2: Object Closer](docs/images/Data_Viewer_Picture_2.png)| ![Frame 3: Object Closest](docs/images/Data_Viewer_Picture_3.png)| ![Annotated Flight Video](docs/images/Data_Viewer_Video.gif) |
| Early Detection                                                      | Closing Distance                                                | Near Range                                                       | Full Sequence                                                |

---




Step-by-step
1. Create a new branch from your current commits

Right now your current branch likely contains the extra commits.

git branch data-scripts

or

git checkout -b data-scripts

Now all your work is saved in data-scripts.

2. Find the commit you want main to stop at

Run:

git log --oneline

Example:

a34b21c add data cleaning script
8fd1231 collect dataset
7c91d2f fix README
3ab9210 initial project setup

If you want main to stop at fix README (7c91d2f), note that hash.

3. Reset main back to that commit

Switch to main:

git checkout main

Reset:

git reset --hard 7c91d2f

Now main no longer contains the data script commits.

4. If you've already pushed to remote

You may need to force push:

git push --force

Be careful if others are using the repo.

✅ Final result:

main
 └── clean project history

data-scripts
 └── dataset collection + cleaning commits

💡 Tip: Another safe method (especially if pushed already) is git revert instead of reset.

If you want, paste your git log --oneline and I can show you exactly which command to run so nothing breaks.