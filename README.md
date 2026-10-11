# Vision-Based Sense and Avoid: Monocular Airborne Object Tracking for Safe Drone Flight

<div align="center">

**Max Busato**  
*The University of Adelaide & Lockheed Martin Australia*

| [Research Paper (PDF)](docs/Research_Paper.pdf) | [Honours Thesis (PDF)](docs/Honours_Thesis.pdf) | [AICrowd Challenge](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge) | [Data Viewer Demo](docs/GIFs/Data_Viewer_Video.gif) |

</div>

<br/>

<div align="center">
  <img src="docs/GIFs/Data_Viewer_Video.gif" width="900" alt="Data Viewer Encounter"/>
</div>

<p align="center">
  <em>Long-range airborne encounter from the AOT benchmark. A monocular vision pipeline must detect and track incoming aircraft when they occupy only a few pixels against cluttered terrain and sky backgrounds.</em>
</p>

---

<br/>

## 1. Overview

This repository contains the official codebase and experimental evaluation framework associated with the research paper and Honours thesis:

> **"Vision-Based Sense and Avoid: Monocular Airborne Object Tracking for Safe Drone Flight"**  
> *Max Busato, in partnership with Lockheed Martin Australia – Advanced Systems & Technology Group and The University of Adelaide.*

Autonomous Sense-and-Avoid (SAA) systems require early, dependable detection of unexpected airborne traffic to plan safe separation manoeuvres. Using the **Airborne Object Tracking (AOT)** dataset—spanning over 5.9M high-resolution frames across diverse aerial encounters—this project investigates:

1. **Backbone Optimisation:** Replacing heavy multi-network ensembles with streamlined, real-time single-backbone detectors (HRNet-W32, EfficientViT, MobileNetV4, ResNet34) accelerated via PyTorch CUDA graph compilation (`torch.compile`).
2. **Tracking-to-Detect (T2D):** A motion-predicted coasting architecture designed to bridge momentary detector dropouts without compounding false positives.
3. **Introspective Self-Monitoring:** Learning-free runtime introspection modules (rolling EWMA confidence trends, consecutive coast counters, Kalman Normalised Innovation Squared) that warn of tracking degradation before ground-truth loss occurs.
4. **Refined Tracking Metrics:** Introducing Continuous Encounter Detection Rate (**CEDR**) and False Tracks Per Sequence (**FTPS**) to rigorously measure sustained track continuity and false trajectory frequency over raw per-frame metrics.

Full experimental data, architectural analyses, and statistical results are detailed in the [Research Paper](docs/Research_Paper.pdf) and [Honours Thesis](docs/Honours_Thesis.pdf).

---

<br/>

## 2. Project Details

<div align="center">

| Role | Name & Affiliation | Contact |
| :--- | :--- | :--- |
| **Industry Sponsor** | Lockheed Martin Australia – ASTG | [Lockheed Martin Australia](https://www.lockheedmartin.com/en-au/index.html) |
| **Academic Supervisor** | Dr. Feras Dayoub (*The University of Adelaide*) | [`feras.dayoub@adelaide.edu.au`](mailto:feras.dayoub@adelaide.edu.au) |
| **Industry Co-Supervisor** | Ms. Kendall Louise Jenner (*Lockheed Martin Australia*) | [`kendall.l.jenner@global.lmco.com`](mailto:kendall.l.jenner@global.lmco.com) |
| **Academic Second Marker** | Dr. Lei Chen (*The University of Adelaide*) | [`lei.chen@adelaide.edu.au`](mailto:lei.chen@adelaide.edu.au) |

</div>

---

<br/>

## 3. Citation

If you find this work, evaluation protocol, or codebase helpful in your research, please cite:

```bibtex
@article{busato2026senseandavoid,
  title   = {Vision-Based Sense and Avoid: Monocular Airborne Object Tracking for Safe Drone Flight},
  author  = {Busato, Max and Dayoub, Feras and Jenner, Kendall Louise},
  journal = {The University of Adelaide and Lockheed Martin Australia Technical Report},
  year    = {2026}
}
```

```bibtex
@thesis{busato2026thesis,
  title   = {Monocular Long-Range Airborne Object Tracking and Introspective Self-Monitoring for Autonomous Sense-and-Avoid},
  author  = {Busato, Max},
  school  = {School of Computer and Mathematical Sciences, The University of Adelaide},
  type    = {Honours Thesis},
  year    = {2026}
}
```

---

<br/>

## 4. Repository Structure

```text
Vision-Based-Sense-and-Avoid/
├── configs/
│   ├── datasets/               # Flight splits and evaluation manifests
│   ├── dependencies/           # Environment dependency definitions
│   └── models/                 # Pretrained weights and checkpoint configs
│
├── core/                       # AICrowd airborne evaluation metric engine
├── data/                       # Local AOT dataset storage & symlinks
├── docs/                       # Research papers, thesis PDFs, figures, and GIFs
│   ├── GIFs/                   # Annotated flight clips and comparison GIFs
│   ├── Images/                 # Documentation diagrams and architecture figures
│   ├── Honours_Thesis.pdf      # Complete Honours Thesis document
│   └── Research_Paper.pdf      # Peer-reviewed research manuscript
│
├── notebooks/
│   ├── baselines.ipynb         # Original AICrowd winning solution benchmarking
│   ├── core_improvement.ipynb  # Backbone replacement and T2D evaluation
│   ├── introspection_module.ipynb  # Introspection module evaluation on corrupted data
│   ├── demo.ipynb              # Quickstart inference walkthrough
│   └── helpers/                # Plotting, summary builders, and video helpers
│
├── outputs/                    # Local run outputs, logs, and evaluation caches
├── results/                    # Exported trade-off summaries and benchmark CSVs
├── src/
│   ├── baselines/              # Ensemble baseline predictors
│   ├── core_improvement/      # Custom backbones (HRNet, MobileNet, EfficientViT)
│   ├── introspection_module/  # Real-time introspection monitors & evaluators
│   └── data/                   # Dataset downloading, filtering, and corruption tools
│
├── README.md
└── setup.sh
```

---

<br/>

## 5. Getting Started

### 5.1 Prerequisites
- Linux / WSL2 (Ubuntu 20.04+ recommended)
- Python 3.11
- NVIDIA GPU with CUDA 12.0+ support (for MAX-AUTOTUNE graph compilation)
- Git LFS (required to pull model weights)

### 5.2 Environment Setup

**1. Clone the repository with Git LFS:**
```bash
git lfs install
git clone https://github.com/MaxB-RnD/Vision-Based-Sense-and-Avoid.git
cd Vision-Based-Sense-and-Avoid
git lfs pull
```
> ⚠️ *Ensure `git lfs pull` runs completely. Model checkpoints in `configs/models/` will remain invalid pointer stubs without it.*

**2. Initialize Python Environment:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies & Build Packages:**
```bash
bash setup.sh
```
*This installs all core PyTorch requirements and sets up the airborne challenge metrics evaluation library (`core/metrics`).*

---

<br/>

## 6. Dataset Preparation & Utilities

The [Airborne Object Tracking (AOT)](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge#dataset) dataset comprises over 11 TB of monocular flight sequences hosted openly on AWS S3 (AWS credentials are not required).

Several CLI utilities are provided under `src/data/` to browse, download, and preprocess subsets:

* **Browse Flight Metadata:** Inspect encounter properties (object pixel size, distance, horizon profile) without downloading full sequences:
  ```bash
  python -m src.data.browse_data
  ```
* **Build Targeted Subsets:** Query flights matching challenging criteria (e.g. sub-36 px² targets, multi-aircraft encounters):
  ```bash
  python -m src.data.data_builder
  ```
* **Download Flight Sequences:** Download image frames for flights specified in a split manifest:
  ```bash
  python -m src.data.download_data
  ```
* **Annotated Video Rendering:** Render frame-by-frame tracker output bounding boxes with ground truth overlays:
  ```bash
  python -m src.data.view_data
  ```

---

<br/>

## 7. Running Experiments & Reproduction

Experiments can be executed either via automated CLI scripts or step-by-step through the provided interactive notebooks.

### 7.1 Interactive Notebooks (`notebooks/`)

* **`notebooks/baselines.ipynb`:** Reproduces the 2021 competition-winning SegTracker four-model ensemble baseline across the curated AOT Hard evaluation subset.
* **`notebooks/core_improvement.ipynb`:** Evaluates single-backbone replacements (MobileNetV4, EfficientViT-M2/B2, ResNet34, HRNet-W32), CUDA graph acceleration, and the Tracking-to-Detect (T2D) motion architecture under AFDR, CEDR, and FTPS.
* **`notebooks/introspection_module.ipynb`:** Evaluates runtime introspection strategies (Static Confidence, EWMA Trend, Coast-Count, Kalman NIS) under temporally ramped image corruptions (sensor noise, optical defocus, low light).

### 7.2 Automated Terminal Evaluation

For batch testing and headless benchmarks:

```bash
# Run baseline ensemble evaluation
python -m src.baselines.baselines_evaluator

# Run single backbone & T2D evaluation
python -m src.core_improvement.core_improvement_evaluator

# Run introspection failure detection evaluation
python -m src.introspection_module.introspection_evaluator
```

Full numeric benchmarking tables, diagnostic curves, and trade-off comparisons across all models and configurations are available in the [Honours Thesis](docs/Honours_Thesis.pdf) and [Research Paper](docs/Research_Paper.pdf).
