# Vision-Based Sense and Avoid: Monocular Airborne Object Tracking for Safe Drone Flight

## Overview

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

## Project Details

| Field            | Details                                                                 |
|------------------|-------------------------------------------------------------------------|
| **Sponsor**      | Lockheed Martin Australia – Advanced Systems & Technology Group         |
| **Supervisor**   | Dr Feras Dayoub (feras.dayoub@adelaide.edu.au)                          |
| **Co-Supervisor**| Ms Kendall Louise Jenner (kendall.l.jenner@global.lmco.com)             |
| **Second Marker**| Dr Lei Chen (lei.chen@adelaide.edu.au)                                  |

---

## Features

> _To be completed as the project develops._

---

## Project Structure

```
Vision-Based-Sense-and-Avoid/
├── core/           # Competition repo helper library (AOT dataset access).
├── data/           # Scripts for dataset selection, cleaning, and visualisation.
│   └── browse_data.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8.
- Virtual Environment (recommended)
- AWS credentials are **NOT** required — the AOT dataset is hosted in a public S3 bucket.
- See `requirements.txt` for full package dependencies.

---

## Setup

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd Vision-Based-Sense-and-Avoid
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Linux/macOS/WSL
   .venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset

> _To be completed as algorithms and tooling are developed._

---

## Usage

> _To be completed as algorithms and tooling are developed._

---

## Development Notes

> _To be completed as the project develops._

---
