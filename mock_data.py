import numpy as np
import cv2

# ─────────────────────────────────────────
# MOCK IMAGE DATA
# ─────────────────────────────────────────


def get_mock_image():
    """Fake BGR image — 256x256 with skin-like tones"""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = (80, 120, 180)  # skin-ish BGR background
    cv2.ellipse(
        image, (128, 128), (60, 45), 0, 0, 360, (30, 60, 140), -1
    )  # wound region
    return image


def get_mock_hsv():
    """Fake HSV image matching get_mock_image()"""
    return cv2.cvtColor(get_mock_image(), cv2.COLOR_BGR2HSV)


def get_mock_gt_mask():
    """Fake ground truth mask — pixel value 1 = wound (matches FUSeg format)"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (60, 45), 0, 0, 360, 1, -1)
    return mask


def get_mock_pred_mask():
    """Fake predicted mask — slightly offset from GT to simulate imperfect segmentation"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (133, 132), (55, 42), 0, 0, 360, 255, -1)  # slightly off
    return mask


def get_mock_contour():
    """Fake contour in OpenCV format — matches pred_mask ellipse roughly"""
    points = []
    for angle in range(0, 360, 10):
        rad = np.radians(angle)
        x = int(133 + 55 * np.cos(rad))
        y = int(132 + 42 * np.sin(rad))
        points.append([[x, y]])
    return np.array(points, dtype=np.int32)


# ─────────────────────────────────────────
# MOCK COMPUTED VALUES
# ─────────────────────────────────────────


def get_mock_iou():
    return 0.54


def get_mock_area():
    return 7285  # pixels


def get_mock_perimeter():
    return 312.4  # pixels


def get_mock_mean_brightness():
    return 108.3


def get_mock_image_id():
    return "mock_001"


# ─────────────────────────────────────────
# MOCK DATAFRAME ROW
# matches exact column names from compiled DataFrame
# ─────────────────────────────────────────


def get_mock_row():
    return {
        "image_id": "mock_001",
        "status": "success",
        "iou": 0.54,
        "iou_tier": "partial",
        "failure_reason": None,
        "area_px": 7285,
        "mean_brightness": 108.3,
        "circularity": 0.68,
        "convexity": 0.81,
        "perimeter_area_ratio": 0.043,
        "boundary_complexity": "moderate",
        "granulation_pct": 0.45,
        "slough_pct": 0.32,
        "necrotic_pct": 0.23,
        "severity_score": 2,
        "severity_label": "moderate",
    }


# ─────────────────────────────────────────
# MOCK DATAFRAME — multiple rows for plot testing
# ─────────────────────────────────────────


def get_mock_dataframe():
    import pandas as pd

    rows = [
        {
            "image_id": "mock_001",
            "status": "success",
            "iou": 0.54,
            "iou_tier": "partial",
            "failure_reason": None,
            "area_px": 7285,
            "mean_brightness": 108,
            "circularity": 0.68,
            "convexity": 0.81,
            "perimeter_area_ratio": 0.043,
            "boundary_complexity": "moderate",
            "granulation_pct": 0.45,
            "slough_pct": 0.32,
            "necrotic_pct": 0.23,
            "severity_score": 2,
            "severity_label": "moderate",
        },
        {
            "image_id": "mock_002",
            "status": "success",
            "iou": 0.81,
            "iou_tier": "good",
            "failure_reason": None,
            "area_px": 12400,
            "mean_brightness": 130,
            "circularity": 0.88,
            "convexity": 0.94,
            "perimeter_area_ratio": 0.021,
            "boundary_complexity": "regular",
            "granulation_pct": 0.72,
            "slough_pct": 0.18,
            "necrotic_pct": 0.10,
            "severity_score": 0,
            "severity_label": "mild",
        },
        {
            "image_id": "mock_003",
            "status": "success",
            "iou": 0.21,
            "iou_tier": "poor",
            "failure_reason": "small_wound",
            "area_px": 980,
            "mean_brightness": 88,
            "circularity": 0.31,
            "convexity": 0.54,
            "perimeter_area_ratio": 0.089,
            "boundary_complexity": "irregular",
            "granulation_pct": 0.12,
            "slough_pct": 0.41,
            "necrotic_pct": 0.47,
            "severity_score": 3,
            "severity_label": "severe",
        },
        {
            "image_id": "mock_004",
            "status": "success",
            "iou": 0.67,
            "iou_tier": "good",
            "failure_reason": None,
            "area_px": 9100,
            "mean_brightness": 115,
            "circularity": 0.74,
            "convexity": 0.87,
            "perimeter_area_ratio": 0.031,
            "boundary_complexity": "regular",
            "granulation_pct": 0.38,
            "slough_pct": 0.44,
            "necrotic_pct": 0.18,
            "severity_score": 1,
            "severity_label": "mild",
        },
        {
            "image_id": "mock_005",
            "status": "failed",
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
        },
    ]
    return pd.DataFrame(rows)
