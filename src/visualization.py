import os
import cv2
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# COLORS — used across all plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "good": "#4ade80",  # green
    "partial": "#f59e0b",  # yellow
    "poor": "#f87171",  # red
    "regular": "#4ade80",  # green
    "moderate": "#f59e0b",  # yellow
    "irregular": "#f87171",  # red
    "mild": "#4ade80",  # green
    "severe": "#f87171",  # red
    "blue": "#60a5fa",  # blue — used for histograms
    "gran": "#e53e3e",  # dark red — granulation tissue
    "slough": "#d69e2e",  # gold — slough tissue
    "necrotic": "#4a5568",  # dark gray — necrotic tissue
}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Segmentation Overlay Grid
# Shows best, median, and worst segmentation results side by side
# ─────────────────────────────────────────────────────────────────────────────


def plot_segmentation_overlays(df_valid, annotated_dir, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    if len(df_valid) < 3:
        print("Not enough images to make overlay grid --skipped.")
        return

    # sort by IoU score
    df_sorted = df_valid.sort_values("iou", ascending=False).reset_index(drop=True)

    # pick best 3, middle 3, worst 3
    n = len(df_sorted)
    mid = n // 2

    best_ids = df_sorted.iloc[0:3]["image_id"].tolist()
    median_ids = df_sorted.iloc[mid - 1 : mid + 2]["image_id"].tolist()
    worst_ids = df_sorted.iloc[-3:]["image_id"].tolist()

    rows = [
        ("Best", best_ids),
        ("Median", median_ids),
        ("Worst", worst_ids),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Segmentation Results — Best, Median, Worst IoU", fontsize=14)

    for row_index, (row_label, image_ids) in enumerate(rows):
        for col_index, image_id in enumerate(image_ids):

            ax = axes[row_index][col_index]
            iou = df_valid[df_valid["image_id"] == image_id]["iou"].values[0]

            # load predicted overlay image
            image_path = os.path.join(annotated_dir, f"{image_id}_predicted.jpg")

            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Image not found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.set_title(f"{row_label}: {image_id}\nIoU = {iou:.3f}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot1_segmentation_overlays.png"), dpi=150)
    plt.close()
    print("Saved: plot1_segmentation_overlays.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — IoU Distribution Histogram
# Shows how accurate segmentation was across all images
# ─────────────────────────────────────────────────────────────────────────────


def plot_iou_histogram(df_valid, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    iou_values = df_valid["iou"].dropna().tolist()
    mean_iou = sum(iou_values) / len(iou_values)

    plt.figure(figsize=(10, 6))

    plt.hist(iou_values, bins=20, range=(0, 1), color=COLORS["blue"], edgecolor="white")

    plt.axvline(
        mean_iou,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean IoU = {mean_iou:.3f}",
    )

    plt.xlabel("IoU Score")
    plt.ylabel("Number of Images")
    plt.title("IoU Score Distribution — FUSeg Validation Set (Aim 1)")
    plt.legend()
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot2_iou_histogram.png"), dpi=150)
    plt.close()
    print("Saved: plot2_iou_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Wound Area Histogram
# Shows distribution of wound sizes across the dataset
# ─────────────────────────────────────────────────────────────────────────────


def plot_wound_area_histogram(df_valid, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    area_values = df_valid["area_px"].dropna().tolist()
    mean_area = sum(area_values) / len(area_values)
    median_area = sorted(area_values)[len(area_values) // 2]

    plt.figure(figsize=(10, 6))

    plt.hist(area_values, bins=25, color=COLORS["blue"], edgecolor="white")

    plt.axvline(
        mean_area,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_area:.0f} px",
    )
    plt.axvline(
        median_area,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_area:.0f} px",
    )

    plt.xlabel("Wound Area (pixels)")
    plt.ylabel("Number of Images")
    plt.title("Wound Area Distribution — FUSeg Validation Set (Aim 1)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot3_wound_area_histogram.png"), dpi=150)
    plt.close()
    print("Saved: plot3_wound_area_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Failure Analysis Bar Chart
# Shows how many images failed and why
# ─────────────────────────────────────────────────────────────────────────────


def plot_failure_barchart(df, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    # replace None with "none (success)"
    failure_reasons = df["failure_reason"].fillna("none (success)").tolist()

    # count each reason
    reason_counts = {}
    for reason in failure_reasons:
        if reason in reason_counts:
            reason_counts[reason] += 1
        else:
            reason_counts[reason] = 1

    labels = list(reason_counts.keys())
    counts = list(reason_counts.values())

    plt.figure(figsize=(10, 6))

    bars = plt.bar(labels, counts, color=COLORS["blue"], edgecolor="white")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Failure Reason")
    plt.ylabel("Number of Images")
    plt.title("Segmentation Failure Analysis — FUSeg Validation Set (Aim 1)")
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot4_failure_barchart.png"), dpi=150)
    plt.close()
    print("Saved: plot4_failure_barchart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Circularity vs IoU Scatter
# Tests if more irregular wounds have lower IoU scores
# ─────────────────────────────────────────────────────────────────────────────


def plot_circularity_vs_iou_scatter(df_valid, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    df_plot = df_valid.dropna(subset=["circularity", "iou", "boundary_complexity"])
    regular = df_plot[df_plot["boundary_complexity"] == "regular"]
    moderate = df_plot[df_plot["boundary_complexity"] == "moderate"]
    irregular = df_plot[df_plot["boundary_complexity"] == "irregular"]

    plt.figure(figsize=(10, 7))

    plt.scatter(
        regular["circularity"],
        regular["iou"],
        color=COLORS["regular"],
        label=f"Regular (n={len(regular)})",
        alpha=0.7,
        edgecolors="white",
        s=40,
    )

    plt.scatter(
        moderate["circularity"],
        moderate["iou"],
        color=COLORS["moderate"],
        label=f"Moderate (n={len(moderate)})",
        alpha=0.7,
        edgecolors="white",
        s=40,
    )

    plt.scatter(
        irregular["circularity"],
        irregular["iou"],
        color=COLORS["irregular"],
        label=f"Irregular (n={len(irregular)})",
        alpha=0.7,
        edgecolors="white",
        s=40,
    )

    plt.xlabel("Circularity (0 = irregular, 1 = perfect circle)")
    plt.ylabel("IoU Score")
    plt.title("Boundary Complexity vs Segmentation Accuracy (Aim 2)")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot5_circularity_vs_iou.png"), dpi=150)
    plt.close()
    print("Saved: plot5_circularity_vs_iou.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Boundary Complexity Distribution
# Shows how many wounds are regular, moderate, or irregular
# ─────────────────────────────────────────────────────────────────────────────


def plot_boundary_complexity_distribution(df_valid, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    df_plot = df_valid.dropna(subset=["boundary_complexity"])
    regular_count = len(df_plot[df_plot["boundary_complexity"] == "regular"])
    moderate_count = len(df_plot[df_plot["boundary_complexity"] == "moderate"])
    irregular_count = len(df_plot[df_plot["boundary_complexity"] == "irregular"])

    labels = ["Regular", "Moderate", "Irregular"]
    counts = [regular_count, moderate_count, irregular_count]
    colors = [COLORS["regular"], COLORS["moderate"], COLORS["irregular"]]

    plt.figure(figsize=(8, 6))

    bars = plt.bar(labels, counts, color=colors, edgecolor="white")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.xlabel("Boundary Complexity")
    plt.ylabel("Number of Images")
    plt.title("Wound Boundary Complexity Distribution — FUSeg Validation Set (Aim 2)")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot6_boundary_complexity.png"), dpi=150)
    plt.close()
    print("Saved: plot6_boundary_complexity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Severity Distribution Bar Chart
# Shows how many wounds are mild, moderate, or severe
# ─────────────────────────────────────────────────────────────────────────────


def plot_severity_distribution(df_valid, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    df_plot = df_valid.dropna(subset=["severity_label"])
    mild_count = len(df_plot[df_plot["severity_label"] == "mild"])
    moderate_count = len(df_plot[df_plot["severity_label"] == "moderate"])
    severe_count = len(df_plot[df_plot["severity_label"] == "severe"])
    total = len(df_plot)

    labels = ["Mild", "Moderate", "Severe"]
    counts = [mild_count, moderate_count, severe_count]
    colors = [COLORS["mild"], COLORS["moderate"], COLORS["severe"]]

    plt.figure(figsize=(8, 6))

    bars = plt.bar(labels, counts, color=colors, edgecolor="white")

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100 if total > 0 else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Severity")
    plt.ylabel("Number of Images")
    plt.title("Wound Severity Distribution — FUSeg Validation Set (Aim 3)")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot7_severity_distribution.png"), dpi=150)
    plt.close()
    print("Saved: plot7_severity_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — Stacked Tissue Composition Bar Chart
# Shows what tissue types make up each severity level
# ─────────────────────────────────────────────────────────────────────────────


def plot_tissue_composition_stacked_bar(df_valid, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    df_plot = df_valid.dropna(
        subset=["severity_label", "granulation_pct", "slough_pct", "necrotic_pct"]
    )

    severity_levels = ["mild", "moderate", "severe"]
    gran_means = []
    slough_means = []
    necrotic_means = []

    for level in severity_levels:
        subset = df_plot[df_plot["severity_label"] == level]

        if len(subset) == 0:
            gran_means.append(0)
            slough_means.append(0)
            necrotic_means.append(0)
        else:
            gran_means.append(subset["granulation_pct"].mean())
            slough_means.append(subset["slough_pct"].mean())
            necrotic_means.append(subset["necrotic_pct"].mean())

    x = [0, 1, 2]
    width = 0.5
    labels = ["Mild", "Moderate", "Severe"]

    plt.figure(figsize=(9, 6))

    plt.bar(
        x,
        gran_means,
        width,
        label="Granulation",
        color=COLORS["gran"],
        edgecolor="white",
    )
    plt.bar(
        x,
        slough_means,
        width,
        label="Slough",
        color=COLORS["slough"],
        edgecolor="white",
        bottom=gran_means,
    )

    necrotic_bottom = [g + s for g, s in zip(gran_means, slough_means)]
    plt.bar(
        x,
        necrotic_means,
        width,
        label="Necrotic",
        color=COLORS["necrotic"],
        edgecolor="white",
        bottom=necrotic_bottom,
    )

    plt.xticks(x, labels)
    plt.xlabel("Severity Level")
    plt.ylabel("Mean Tissue Composition")
    plt.title("Tissue Composition by Severity — FUSeg Validation Set (Aim 3)")
    plt.legend()
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot8_tissue_composition.png"), dpi=150)
    plt.close()
    print("Saved: plot8_tissue_composition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9 — Wound Area vs Necrotic Percentage Scatter
# Tests if larger wounds have more necrotic tissue
# ─────────────────────────────────────────────────────────────────────────────


def plot_area_vs_necrotic_scatter(df_valid, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)

    df_plot = df_valid.dropna(subset=["area_px", "necrotic_pct", "severity_label"])
    mild = df_plot[df_plot["severity_label"] == "mild"]
    moderate = df_plot[df_plot["severity_label"] == "moderate"]
    severe = df_plot[df_plot["severity_label"] == "severe"]

    plt.figure(figsize=(10, 7))

    plt.scatter(
        mild["area_px"],
        mild["necrotic_pct"],
        color=COLORS["mild"],
        label=f"Mild (n={len(mild)})",
        alpha=0.7,
        edgecolors="white",
        s=40,
    )

    plt.scatter(
        moderate["area_px"],
        moderate["necrotic_pct"],
        color=COLORS["moderate"],
        label=f"Moderate (n={len(moderate)})",
        alpha=0.7,
        edgecolors="white",
        s=40,
    )

    plt.scatter(
        severe["area_px"],
        severe["necrotic_pct"],
        color=COLORS["severe"],
        label=f"Severe (n={len(severe)})",
        alpha=0.7,
        edgecolors="white",
        s=40,
    )

    plt.xlabel("Wound Area (pixels)")
    plt.ylabel("Necrotic Tissue Percentage")
    plt.title("Wound Area vs Necrotic Tissue — FUSeg Validation Set (Aim 3)")
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot9_area_vs_necrotic.png"), dpi=150)
    plt.close()
    print("Saved: plot9_area_vs_necrotic.png")
