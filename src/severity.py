import cv2
import numpy as np

GRANULATION_HUE = (0, 20)
SLOUGH_HUE = (20, 40)
NECROTIC_SAT_MAX = 50
NECROTIC_VAL_MAX = 80
LARGE_AREA_THRESHOLD = 5000


class SeverityClassifier:

    def extract_tissue_composition(self, image, pred_mask, hsv_image):
        h, s, v = cv2.split(hsv_image)
        wound = pred_mask > 0
        total = np.count_nonzero(wound)

        if total == 0:
            return {"granulation_pct": 0, "slough_pct": 0, "necrotic_pct": 0}

        necrotic = wound & (s <= NECROTIC_SAT_MAX) & (v <= NECROTIC_VAL_MAX)
        slough = wound & (h >= SLOUGH_HUE[0]) & (h <= SLOUGH_HUE[1])
        gran = wound & ((h <= GRANULATION_HUE[1]) | (h >= 160))  # wraps around 0/179

        return {
            "granulation_pct": np.count_nonzero(gran) / total,
            "slough_pct": np.count_nonzero(slough) / total,
            "necrotic_pct": np.count_nonzero(necrotic) / total,
        }

    def classify_severity(
        self, area, granulation_pct, slough_pct, necrotic_pct, large_area_threshold=None
    ):
        threshold = (
            large_area_threshold
            if large_area_threshold is not None
            else LARGE_AREA_THRESHOLD
        )

        score = 0
        if area > threshold:
            score += 1
        if necrotic_pct > 0.20:
            score += 1
        if slough_pct > 0.40:
            score += 1
        if granulation_pct < 0.30:
            score += 1

        if score >= 3:
            return "severe", score
        elif score >= 2:
            return "moderate", score
        else:
            return "mild", score
