import cv2
import numpy as np

class WoundAreaAnalyzer:

    def calc_area(self, mask):
        return np.count_nonzero(mask)

    def calc_perimeter(self, contour):
        if contour is None:
            return 0.0
        return cv2.arcLength(contour, closed=True)

    def calc_mean_brightness(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wound_pixels = gray[mask > 0]
        if wound_pixels.size == 0:
            return 0.0
        return np.mean(wound_pixels)

    def classify_iou_performance(self, iou_score):
        if iou_score >= 0.65:
            return "good"
        elif iou_score >= 0.55:
            return "partial"
        else:
            return "poor"

    def identify_failure(self, iou, area, mean_brightness, median_area, brightness_threshold):
        if iou < 0.50:
            if area < median_area:
                return "small_wound_failure"
            elif mean_brightness < brightness_threshold:
                return "low_light_failure"
            else:
                return "ambiguous_color"
        return None

    def calc_dataset_statistics(self, areas):
        return {
            "mean": np.mean(areas),
            "median": np.median(areas),
            "std": np.std(areas),
            "min": np.min(areas),
            "max": np.max(areas)
        }