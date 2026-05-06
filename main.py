"""
main.py — DFU Analysis Pipeline Orchestration Script

Runs the full pipeline on the FUSeg validation set:
    1. Load and validate all image/mask pairs
    2. Segment wound region using HSV thresholding (WoundSegmenter)
    3. Measure wound area and analyze IoU error (WoundAreaAnalyzer)
    4. Compute boundary morphology metrics (MorphologyAnalyzer)
    5. Extract tissue composition and classify severity (SeverityClassifier)
    6. Aggregate all results into a metrics DataFrame
    7. Generate all 9 plots
    8. Save outputs to organized directory structure

Usage:
    conda activate dfu-analysis
    python main.py

Outputs:
    outputs/annotated_images/   — original, predicted, gt overlays + thumbs
    outputs/plots/              — all 9 plots
    outputs/metrics.csv         — per-image results DataFrame
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from src.dataset import WoundDataset
from src.segmentation import WoundSegmenter
from src.analysis import WoundAreaAnalyzer
from src.morphology import MorphologyAnalyzer
from src.severity import SeverityClassifier
from src.visualization import (
    plot_segmentation_overlays,
    plot_iou_histogram,
    plot_wound_area_histogram,
    plot_failure_barchart,
    plot_circularity_vs_iou_scatter,
    plot_boundary_complexity_distribution,
    plot_severity_distribution,
    plot_tissue_composition_stacked_bar,
    plot_area_vs_necrotic_scatter,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_DIR = "data/validation/images"
MASK_DIR = "data/validation/labels"
OUTPUT_DIR = "outputs/"
MIN_AREA = 500  # minimum contour area in pixels
BRIGHTNESS_THRESH = 100  # mean brightness threshold for failure analysis
SPLIT_THRESH = 0.20  # contours within 20% = split wound
LARGE_AREA_THRESH = None  # set dynamically from 75th percentile after loading


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 60)
    print("DFU ANALYSIS PIPELINE")
    print("=" * 60)

    # ── 1. SETUP ─────────────────────────────────────────────────

    dataset = WoundDataset()
    dataset.setup_output_dir(OUTPUT_DIR)

    pairs = dataset.load_all_pairs(IMAGE_DIR, MASK_DIR)

    if len(pairs) == 0:
        print("[ERROR] No valid image/mask pairs found. Check data directory.")
        sys.exit(1)

    # initialize modules
    segmenter = WoundSegmenter(
        min_area=MIN_AREA,
        split_thresh=SPLIT_THRESH,
    )
    analyzer = WoundAreaAnalyzer(brightness_threshold=BRIGHTNESS_THRESH)
    morphology = MorphologyAnalyzer()
    severity = SeverityClassifier()

    results = []
    overlays = []  # tracks (image_id, iou_tier) for overlay plot

    print(f"\n[INFO] Running pipeline on {len(pairs)} images...\n")

    # ── 2. MAIN LOOP ──────────────────────────────────────────────

    for i, (image_id, image, gt_mask) in enumerate(pairs):

        # progress indicator every 20 images
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing image {i+1}/{len(pairs)}...")

        # ── SEGMENTATION ──────────────────────────────────────────

        hsv, hue_label = segmenter.preprocess(image)
        pred_mask = segmenter.build_masks(hsv, hue_label)
        contour, pred_mask, status = segmenter.get_wound_contour(pred_mask)

        # handle segmentation failure — log and skip downstream analysis
        if status == "failed":
            results.append(
                {
                    "image_id": image_id,
                    "status": "failed",
                    "hue_label": hue_label,
                    "iou": None,
                    "iou_tier": None,
                    "failure_reason": "no_valid_contour",
                    "area_px": None,
                    "mean_brightness": None,
                    "circularity": None,
                    "convexity": None,
                    "perimeter_area_ratio": None,
                    "boundary_complexity": None,
                    "granulation_pct": None,
                    "slough_pct": None,
                    "necrotic_pct": None,
                    "severity_score": None,
                    "severity_label": None,
                }
            )
            continue

        # compute IoU against ground truth
        iou = segmenter.compute_iou(pred_mask, gt_mask)

        # save overlay images
        segmenter.save_overlays(image_id, image, pred_mask, gt_mask, OUTPUT_DIR)

        # ── AREA + ERROR ANALYSIS ─────────────────────────────────

        area = analyzer.calc_area(pred_mask)
        perimeter = analyzer.calc_perimeter(contour)
        mean_brightness = analyzer.calc_mean_brightness(image, pred_mask)
        iou_tier = analyzer.classify_iou_performance(iou)

        # failure reason computed after dataset stats (needs median_area)
        # stored temporarily — updated after loop completes
        failure_reason = None

        # ── MORPHOLOGY ────────────────────────────────────────────

        shape = morphology.calculate_metrics(contour)
        complexity = morphology.classify_boundary_complexity(
            shape["circularity"], shape["convexity"]
        )

        # ── SEVERITY ──────────────────────────────────────────────

        tissue = severity.extract_tissue_composition(image, pred_mask, hsv)
        label, score = severity.classify_severity(
            area,
            tissue["granulation_pct"],
            tissue["slough_pct"],
            tissue["necrotic_pct"],
            large_area_threshold=LARGE_AREA_THRESH,
        )

        # ── AGGREGATE ROW ─────────────────────────────────────────

        results.append(
            {
                "image_id": image_id,
                "status": status,
                "hue_label": hue_label,
                "iou": iou,
                "iou_tier": iou_tier,
                "failure_reason": failure_reason,
                "area_px": area,
                "mean_brightness": mean_brightness,
                "circularity": shape["circularity"],
                "convexity": shape["convexity"],
                "perimeter_area_ratio": shape["perimeter_area_ratio"],
                "boundary_complexity": complexity,
                "granulation_pct": tissue["granulation_pct"],
                "slough_pct": tissue["slough_pct"],
                "necrotic_pct": tissue["necrotic_pct"],
                "severity_score": score,
                "severity_label": label,
            }
        )

        overlays.append((image_id, iou_tier))

    # ── 3. BUILD DATAFRAME ────────────────────────────────────────

    df = pd.DataFrame(results)
    df_valid = df[df["status"] != "failed"].copy()

    # set LARGE_AREA_THRESH dynamically from 75th percentile
    # then compute failure reasons using real median area
    if len(df_valid) > 0:
        median_area = df_valid["area_px"].median()
        large_area_thresh = df_valid["area_px"].quantile(0.75)
        analyzer.median_area = median_area

        # recompute failure reasons now that we have dataset-level stats
        df_valid["failure_reason"] = df_valid.apply(
            lambda row: analyzer.identify_failure(
                row["iou"], row["area_px"], row["mean_brightness"]
            ),
            axis=1,
        )

        # update failed rows
        df.update(df_valid[["image_id", "failure_reason"]].set_index("image_id"))

    # compute dataset statistics
    if len(df_valid) > 0:
        stats = analyzer.calc_dataset_statistics(df_valid)
    else:
        stats = {}
        print("[WARNING] No valid segmentations — cannot compute statistics.")

    # save results
    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    df.to_csv(results_path, index=False)
    print(f"\n[INFO] Metrics saved to {results_path}")

    # ── 4. PRINT SUMMARY ──────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total images processed:     {len(df)}")
    print(f"Successful segmentations:   {len(df_valid)}")
    print(f"Failed segmentations:       {len(df[df['status'] == 'failed'])}")
    print(f"Split wound cases:          {len(df[df['status'] == 'split_wound'])}")

    if len(df_valid) > 0:
        print(f"\nSegmentation Performance (Aim 1):")
        print(f"  Mean IoU:               {df_valid['iou'].mean():.3f}")
        print(f"  Std IoU:                {df_valid['iou'].std():.3f}")
        print(f"  Min IoU:                {df_valid['iou'].min():.3f}")
        print(f"  Max IoU:                {df_valid['iou'].max():.3f}")
        print(
            f"  Good    (>=0.65):       {len(df_valid[df_valid['iou'] >= 0.65])}  "
            f"({len(df_valid[df_valid['iou'] >= 0.65])/len(df_valid)*100:.1f}%)"
        )
        print(
            f"  Partial (0.40-0.65):    {len(df_valid[(df_valid['iou'] >= 0.40) & (df_valid['iou'] < 0.65)])}  "
            f"({len(df_valid[(df_valid['iou'] >= 0.40) & (df_valid['iou'] < 0.65)])/len(df_valid)*100:.1f}%)"
        )
        print(
            f"  Poor    (<0.40):        {len(df_valid[df_valid['iou'] < 0.40])}  "
            f"({len(df_valid[df_valid['iou'] < 0.40])/len(df_valid)*100:.1f}%)"
        )

        print(f"\nWound Area Statistics (Aim 1):")
        print(f"  Mean area:              {df_valid['area_px'].mean():.0f} px")
        print(f"  Median area:            {df_valid['area_px'].median():.0f} px")
        print(f"  Std area:               {df_valid['area_px'].std():.0f} px")
        print(f"  Min area:               {df_valid['area_px'].min():.0f} px")
        print(f"  Max area:               {df_valid['area_px'].max():.0f} px")

        print(f"\nBoundary Morphology (Aim 2):")
        for label in ["regular", "moderate", "irregular"]:
            count = len(df_valid[df_valid["boundary_complexity"] == label])
            mean_iou = df_valid[df_valid["boundary_complexity"] == label]["iou"].mean()
            print(
                f"  {label:<12}            {count:>4} images   mean IoU={mean_iou:.3f}"
            )

        print(f"\nSeverity Distribution (Aim 3):")
        for label in ["mild", "moderate", "severe"]:
            count = len(df_valid[df_valid["severity_label"] == label])
            pct = count / len(df_valid) * 100
            print(f"  {label:<12}            {count:>4} images   ({pct:.1f}%)")

        print(f"\nHue Label Breakdown:")
        for label in ["red-pink", "yellow-tan", "mixed"]:
            count = len(df[df["hue_label"] == label])
            valid = df_valid[df_valid["hue_label"] == label]
            mean_iou = valid["iou"].mean() if len(valid) > 0 else float("nan")
            iou_str = (
                f"mean IoU={mean_iou:.3f}"
                if not np.isnan(mean_iou)
                else "no valid results"
            )
            print(f"  {label:<15}         {count:>4} images   {iou_str}")

        # best HSV range note
        if df_valid["iou"].mean() > 0.55:
            best_hue = df_valid.groupby("hue_label")["iou"].mean().idxmax()
            print(f"\n[INFO] Best performing hue range: {best_hue}")

    print("=" * 60)

    # ── 5. GENERATE PLOTS ─────────────────────────────────────────

    if len(df_valid) == 0:
        print("\n[WARNING] No valid segmentations — skipping plots.")
        dataset.print_summary(mean_iou=None)
        return

    print("\n[INFO] Generating plots...")

    plots_dir = os.path.join(OUTPUT_DIR, "plots")

    # Plot 1 — segmentation overlay grid
    plot_segmentation_overlays(
        df_valid, os.path.join(OUTPUT_DIR, "annotated_images"), plots_dir
    )
    print("  [1/9] Segmentation overlay grid")

    # Plot 2 — IoU distribution histogram
    plot_iou_histogram(df_valid, plots_dir)
    print("  [2/9] IoU distribution histogram")

    # Plot 3 — wound area histogram
    plot_wound_area_histogram(df_valid, plots_dir)
    print("  [3/9] Wound area histogram")

    # Plot 4 — failure analysis bar chart
    plot_failure_barchart(df, plots_dir)
    print("  [4/9] Failure analysis bar chart")

    # Plot 5 — circularity vs IoU scatter
    plot_circularity_vs_iou_scatter(df_valid, plots_dir)
    print("  [5/9] Circularity vs IoU scatter")

    # Plot 6 — boundary complexity distribution
    plot_boundary_complexity_distribution(df_valid, plots_dir)
    print("  [6/9] Boundary complexity distribution")

    # Plot 7 — severity distribution
    plot_severity_distribution(df_valid, plots_dir)
    print("  [7/9] Severity distribution bar chart")

    # Plot 8 — stacked tissue composition
    plot_tissue_composition_stacked_bar(df_valid, plots_dir)
    print("  [8/9] Stacked tissue composition")

    # Plot 9 — area vs necrotic pct scatter
    plot_area_vs_necrotic_scatter(df_valid, plots_dir)
    print("  [9/9] Area vs necrotic pct scatter")

    print(f"\n[INFO] All plots saved to {plots_dir}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────

    dataset.print_summary(mean_iou=df_valid["iou"].mean())

    print("\nPipeline complete.")
    print(f"  Results:  {results_path}")
    print(f"  Overlays: {os.path.join(OUTPUT_DIR, 'annotated_images/')}")
    print(f"  Plots:    {plots_dir}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
