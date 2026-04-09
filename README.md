![SJSU](./SJSU_Primary_mark_Web.png)

# DFU Analysis Pipeline

Classical computer vision pipeline for diabetic foot ulcer segmentation, feature extraction, and severity analysis on the FUSeg dataset.

---

## Dataset

**FUSeg — Foot Ulcer Segmentation Challenge**
University of Wisconsin-Milwaukee, MICCAI 2021

- 1,210 clinically photographed diabetic foot ulcer images
- Pixel-wise binary segmentation masks annotated by wound care experts
- 889 patients, collected over 2 years, fully de-identified (HIPAA)
- Split: 810 train / 200 validation / 200 test

---

### Download

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

## Pipeline Architecture

```
Raw image → Preprocessing → Segmentation → Feature Extraction → Analysis → Output
```
