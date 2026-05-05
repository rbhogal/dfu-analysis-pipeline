import cv2
import numpy as np


class MorphologyAnalyzer:
    def calculate_metrics(self, mask):
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
        convexity = area / hull_area if hull_area != 0 else 0

        return {
            "area": area,
            "perimeter": perimeter,
            "hull_area": hull_area,
            "circularity": circularity,
            "convexity": convexity
        }

    def classify_boundary_complexity(self, circularity, convexity):
        if circularity > 0.70 and convexity > 0.85:
            return "regular"
        elif circularity > 0.50:
            return "moderate"
        else:
            return "irregular"