# DFU Analysis Pipeline

## Project Structure

```
dfu-analysis-pipeline/
├── data/                        # FUSeg dataset (not tracked in git)
├── src/
│   ├── segmentation.py          # WoundSegmenter class — Aim 1
│   ├── analysis.py              # WoundAreaAnalyzer class — Aim 2
│   ├── features.py              # ColorFeatureExtractor class — Aim 3
│   ├── morphology.py            # MorphologyAnalyzer class — Aim 4
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
