import os
import cv2
from utils.morphology_analyzer import MorphologyAnalyzer


# Folder that contains your mask/label images
label_folder = "data/Foot Ulcer Segmentation Challenge/train/labels"

analyzer = MorphologyAnalyzer()

for filename in os.listdir(label_folder):
    if filename.endswith(".png"):
        mask_path = os.path.join(label_folder, filename)

        mask = cv2.imread(mask_path, 0)

        if mask is None:
            print(f"{filename}: Mask not found")
            continue

        metrics = analyzer.calculate_metrics(mask)

        if metrics is None:
            print(f"{filename}: No contour found")
            continue

        label = analyzer.classify_boundary_complexity(
            metrics["circularity"],
            metrics["convexity"]
        )

        print("\nFile:", filename)
        print("Area:", metrics["area"])
        print("Perimeter:", metrics["perimeter"])
        print("Hull Area:", metrics["hull_area"])
        print("Circularity:", metrics["circularity"])
        print("Convexity:", metrics["convexity"])
        print("Boundary Classification:", label)