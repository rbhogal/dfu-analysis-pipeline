import cv2
import numpy as np


class MorphologyAnalyzer:
    def calculate_metrics(self, contour):

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
        convexity = area / hull_area if hull_area != 0 else 0
        perimeter_area_ratio = perimeter / area if area != 0 else 0

        return {
            "area": area,
            "perimeter": perimeter,
            "hull_area": hull_area,
            "circularity": circularity,
            "convexity": convexity,
            "perimeter_area_ratio": perimeter_area_ratio,
        }

    def classify_boundary_complexity(self, circularity, convexity):
        if circularity > 0.70 and convexity > 0.85:
            return "regular"
        elif circularity > 0.50:
            return "moderate"
        else:
            return "irregular"
