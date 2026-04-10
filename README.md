![SJSU](./docs/assets/SJSU_Primary_mark_Web.png)

# DFU Analysis Pipeline

Classical computer vision pipeline for diabetic foot ulcer segmentation, feature extraction, and rule-based severity classification on the FUSeg dataset.

---

## Dataset

**FUSeg — Foot Ulcer Segmentation Challenge**
University of Wisconsin-Milwaukee, MICCAI 2021

- 1,210 clinically photographed diabetic foot ulcer images
- Pixel-wise binary segmentation masks annotated by wound care experts
- 889 patients, collected over 2 years, fully de-identified (HIPAA)
- Split: 810 train / 200 validation / 200 test

---

## Pipeline Architecture

```
Raw image → Preprocessing → Segmentation → Feature Extraction → Severity Classification → Output
```

---

## Project Structure

```
dfu-analysis-pipeline/
├── data/                        # FUSeg dataset (not tracked in git)
├── src/
│   ├── segmentation.py          # WoundSegmenter class
│   ├── analysis.py              # WoundAreaAnalyzer class
│   ├── features.py              # ColorFeatureExtractor class
│   ├── morphology.py            # MorphologyAnalyzer class
│   ├── dataset.py               # WoundDataset — loading and validation
│   └── visualization.py         # All plots
├── outputs/
│   ├── annotated_images/        # Original, predicted, and GT overlays
│   ├── plots/                   # All generated plots
│   └── metrics.csv              # Per-image results DataFrame
├── main.py                      # Orchestration script
├── calibrate.py                 # HSV threshold calibration
├── mock_data.py                 # Mock inputs for development/testing
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### 1. Clone the repository

Open VS Code terminal or any terminal, then navigate to your desired folder:

```
cd ~/<folder-name>
```

- Replace `~/<folder-name>` with wherever you want to store the project.
- Common locations: `~/repos` or `~/projects`

```bash
git clone https://github.com/rbhogal/dfu-analysis-pipeline
cd dfu-analysis-pipeline
```

### 2. Create and activate the conda environment

```bash
conda create -n dfu-analysis python=3.13.9
conda activate dfu-analysis
```

You only need to create the environment once. Every time you return to work on this project just run:

```bash
conda activate dfu-analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import cv2; import numpy; import pandas; import matplotlib; print('all good:', cv2.__version__)"
```

If a version number prints without errors you are fully set up.

### 5. Select Interpreter

Open the command palette — Cmd+Shift+P
Type Python: Select Interpreter
Look for the one that says dfu-analysis

### Download Dataset

1. Clone or download the dataset from the [FUSeg GitHub repository](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge)
2. Place files in the following structure:

```
data/
└── fuseg/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── validation/
    │   ├── images/
    │   └── labels/
    └── test/
        └── images/
```

---

## Git Workflow

See 👉 [./docs/workflow.md](./docs/workflow.md)
