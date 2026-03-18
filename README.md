# Project Name

## Summary

A brief description of what this project does, why it exists, and what problem it solves.  
Keep this high-level and focused on intent rather than implementation details.

---

## Features

- Key feature or capability
- Another important feature
- Optional: platform, language, or performance highlights

---

## 🗂️ Project Structure

```text
Vision-Based-Sense-and-Avoid/
├── core/           # Competition Repo Helper Library (AOT Dataset Access).
├── data/           # Stores Flight ID Sets for Different Custom Libraries.
├── docs/           # Further Information on Research.
├── notebooks/      # End-to-End Model Evaluation Notebooks.
│   └── demo.ipynb
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
- AWS credentials are **NOT** required - the AOT dataset is hosted in a public S3 bucket.
- See `requirements.txt` for full package dependencies.

---

## Requirements

- Language version (e.g., C++20, Python 3.11).
- Build tools or dependencies.
- Operating system support (if relevant).

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

## Usage

Basic examples of how to use or run the project.

- Command-line usage.
- Example input/output.
- How this project is intended to be consumed (library, tool, service, etc.).

---

## Development Notes

Optional section for:
- Design decisions.
- Known limitations.
- Future improvements.
- Gotchas or assumptions.

---

## License

Specify the license for this project (or note if it is private/internal).
