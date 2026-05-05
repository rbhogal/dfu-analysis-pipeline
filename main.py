import os
import cv2

from src.visualization import save_visualizations
from src.morphology import MorphologyAnalyzer

# Folders
images_dir = "data/train/images"
labels_dir = "data/train/labels"

# Create analyzer
analyzer = MorphologyAnalyzer()

# Loop through all image files
for filename in os.listdir(images_dir):

    # Skip hidden/system files
    if filename.startswith("."):
        continue

    image_path = os.path.join(images_dir, filename)
    mask_path = os.path.join(labels_dir, filename)

    # Check if matching mask exists
    if not os.path.exists(mask_path):
        print(f"Mask not found for {filename}")
        continue

    # Load image
    image = cv2.imread(image_path)

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Check loading
    if image is None:
        print(f"Could not load image: {filename}")
        continue

    if mask is None:
        print(f"Could not load mask: {filename}")
        continue

    # Calculate metrics
    metrics = analyzer.calculate_metrics(mask)

    if metrics is None:
        print(f"No contour found in {filename}")
        continue

    # Remove file extension for cleaner output names
    image_name = os.path.splitext(filename)[0]

    # Save visualizations
    save_visualizations(
        image=image,
        mask=mask,
        metrics=metrics,
        output_dir="outputs",
        image_name=image_name
    )

    print(f"Processed: {filename}")

print("All images processed successfully.")

  