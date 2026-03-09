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

> _To be completed as algorithms and tooling are developed._

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