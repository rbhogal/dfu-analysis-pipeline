import os
import cv2
import matplotlib.pyplot as plt


def save_visualizations(image, mask, metrics, output_dir="outputs", image_name="sample"):
    """
    Saves:
    1. Original image with wound contour
    2. Original image and mask side-by-side
    3. Bar chart of morphology metrics
    """

    annotated_dir = os.path.join(output_dir, "annotated_images")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Make sure mask is uint8
    mask = mask.astype("uint8")

    # Convert 0/1 mask to 0/255 mask if needed
    if mask.max() == 1:
        mask = mask * 255

    # Threshold mask just in case it is grayscale
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find wound contour
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Copy image so original is not changed
    annotated = image.copy()

    # Draw contour if found
    if len(contours) > 0:
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

    # Save contour image
    annotated_path = os.path.join(annotated_dir, f"{image_name}_contour.png")
    cv2.imwrite(annotated_path, annotated)

    # Save image and mask side-by-side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Wound Mask")
    plt.axis("off")

    plt.tight_layout()

    comparison_path = os.path.join(plots_dir, f"{image_name}_image_mask.png")
    plt.savefig(comparison_path)
    plt.close()

    # Save metrics bar chart
    if metrics is not None and len(metrics) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Value")
        plt.title("Wound Morphology Metrics")
        plt.tight_layout()

        metrics_path = os.path.join(plots_dir, f"{image_name}_metrics.png")
        plt.savefig(metrics_path)
        plt.close()

        print(f"Saved metrics plot: {metrics_path}")
    else:
        print("No metrics found. Metrics plot was not created.")

    print(f"Saved contour image: {annotated_path}")
    print(f"Saved image/mask plot: {comparison_path}")


if __name__ == "__main__":
    print("visualization.py is working")


