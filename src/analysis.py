import cv2
import numpy as np


class WoundAreaAnalyzer:

    def __init__(self, brightness_threshold=100):
        self.brightness_threshold = brightness_threshold
        self.median_area = None

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
        elif iou_score >= 0.40:
            return "partial"
        else:
            return "poor"

    def identify_failure(self, iou, area, mean_brightness):
        if iou < 0.50:
            if self.median_area is not None and area < self.median_area:
                return "small_wound_failure"
            elif mean_brightness < self.brightness_threshold:
                return "low_light_failure"
            else:
                return "ambiguous_color"
        return None

    def calc_dataset_statistics(self, df):
        return {
            "mean": df["area_px"].mean(),
            "median": df["area_px"].median(),
            "std": df["area_px"].std(),
            "min": df["area_px"].min(),
            "max": df["area_px"].max(),
        }
