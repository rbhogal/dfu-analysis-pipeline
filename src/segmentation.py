import os
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HSV RANGE CONSTANTS
# These define the color ranges for wound tissue types in HSV color space.
# H (Hue):        0-180 in OpenCV (not 0-360)
# S (Saturation): 0-255
# V (Value):      0-255
#
# Tune these using calibrate.py before running the full pipeline.
# ─────────────────────────────────────────────────────────────────────────────

# Granulation tissue — red/pink hues (healthy healing tissue)
GRANULATION_LOWER_1 = np.array([0, 140, 51])  # lower red hue range
GRANULATION_UPPER_1 = np.array([16, 255, 255])

GRANULATION_LOWER_2 = np.array([166, 140, 51])  # upper red hue range (wraps around 180)
GRANULATION_UPPER_2 = np.array([180, 255, 255])

# Slough tissue — yellow/tan hues (stalled healing, necrotic debris)
SLOUGH_LOWER = np.array([15, 36, 199])
SLOUGH_UPPER = np.array([25, 255, 255])

# Hue thresholds for dominant hue classification
HUE_RED_PINK_MAX = 20  # mean hue <= this → red-pink dominant
HUE_YELLOW_MIN = 20  # mean hue > this and <= HUE_YELLOW_MAX → yellow dominant
HUE_YELLOW_MAX = 40

# Morphological kernel size
MORPH_KERNEL_SIZE = (5, 5)

# Minimum contour area in pixels — contours below this are noise
MIN_AREA = 500

# Split wound threshold — two contours within this ratio are treated as split wound
SPLIT_THRESH = 0.20

# Overlay opacity for visualization
OVERLAY_ALPHA = 0.4


class WoundSegmenter:
    """
    Core segmentation class for diabetic foot ulcer images.

    Implements a multi-strategy HSV masking pipeline:
    1. Preprocess image (resize, convert to HSV, blur)
    2. Detect dominant wound hue to select masking strategy
    3. Build appropriate color mask (granulation, slough, or both)
    4. Apply morphological cleanup to remove noise and fill holes
    5. Extract largest valid contour
    6. Handle failure cases and split wounds
    7. Compute IoU against ground truth mask
    8. Save overlay images for visualization

    Logical complexity:
    - Multi-branch hue detection selects masking strategy per image
    - Contour selection handles failure, split wound, and success cases
    - IoU handles zero-union edge case safely
    """

    def __init__(
        self,
        min_area=MIN_AREA,
        split_thresh=SPLIT_THRESH,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        overlay_alpha=OVERLAY_ALPHA,
    ):
        self.min_area = min_area
        self.split_thresh = split_thresh
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, morph_kernel_size
        )
        self.overlay_alpha = overlay_alpha

    # ─────────────────────────────────────────────────────────────
    # STEP 1 — PREPROCESSING
    # ─────────────────────────────────────────────────────────────

    def preprocess(self, image):
        """
        Prepare image for segmentation.

        Steps:
        - Resize to standard size (512x512) for consistent processing
        - Apply Gaussian blur to reduce noise
        - Convert BGR to HSV color space
        - Compute mean hue across non-background pixels
        - Classify dominant hue as red-pink, yellow-tan, or mixed

        Args:
            image (np.ndarray): Raw BGR image loaded by WoundDataset

        Returns:
            tuple: (hsv_image, hue_label)
                hsv_image  (np.ndarray): HSV version of the image
                hue_label  (str): "red-pink" | "yellow-tan" | "mixed"
        """
        # resize for consistent processing
        resized = cv2.resize(image, (512, 512))

        # gaussian blur to reduce noise before color analysis
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)

        # convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # compute mean hue across non-background pixels
        # background pixels tend to have very low saturation — filter them out
        saturation_channel = hsv[:, :, 1]
        non_background = saturation_channel > 30  # pixels with meaningful saturation

        if np.sum(non_background) == 0:
            # fallback if no meaningful pixels found
            hue_label = "mixed"
        else:
            mean_hue = np.mean(hsv[:, :, 0][non_background])

            # classify dominant hue
            # ── logical complexity: multi-branch hue classification ──
            if mean_hue <= HUE_RED_PINK_MAX:
                hue_label = "red-pink"
            elif HUE_YELLOW_MIN < mean_hue <= HUE_YELLOW_MAX:
                hue_label = "yellow-tan"
            else:
                hue_label = "mixed"

        return hsv, hue_label

    # ─────────────────────────────────────────────────────────────
    # STEP 2 — MASK BUILDING
    # ─────────────────────────────────────────────────────────────

    def build_masks(self, hsv, hue_label):
        """
        Build a binary wound mask using HSV color range thresholding.

        Selects masking strategy based on dominant hue:
        - red-pink  → granulation mask only
        - yellow-tan → slough mask only
        - mixed     → both masks combined with bitwise OR

        Applies morphological operations after masking:
        - MORPH_CLOSE: fills holes inside wound region
        - MORPH_OPEN:  removes noise outside wound region

        Args:
            hsv       (np.ndarray): HSV image from preprocess()
            hue_label (str): "red-pink" | "yellow-tan" | "mixed"

        Returns:
            np.ndarray: Binary mask (0 or 255) of wound region
        """
        # ── logical complexity: strategy selection based on hue_label ──
        if hue_label == "red-pink":
            mask = self._build_granulation_mask(hsv)

        elif hue_label == "yellow-tan":
            mask = self._build_slough_mask(hsv)

        else:
            # mixed — combine both masks
            granulation_mask = self._build_granulation_mask(hsv)
            slough_mask = self._build_slough_mask(hsv)
            mask = cv2.bitwise_or(granulation_mask, slough_mask)

        # morphological cleanup
        # close first — fill holes before removing noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)

        return mask

    def _build_granulation_mask(self, hsv):
        """
        Build mask for granulation tissue (red/pink hues).

        Red hues wrap around 180 in OpenCV HSV so we need two ranges:
        - Range 1: hue 0-15   (lower red)
        - Range 2: hue 165-180 (upper red, wraps around)

        Args:
            hsv (np.ndarray): HSV image

        Returns:
            np.ndarray: Binary mask for granulation tissue
        """
        mask1 = cv2.inRange(hsv, GRANULATION_LOWER_1, GRANULATION_UPPER_1)
        mask2 = cv2.inRange(hsv, GRANULATION_LOWER_2, GRANULATION_UPPER_2)
        return cv2.bitwise_or(mask1, mask2)

    def _build_slough_mask(self, hsv):
        """
        Build mask for slough tissue (yellow/tan hues).

        Args:
            hsv (np.ndarray): HSV image

        Returns:
            np.ndarray: Binary mask for slough tissue
        """
        return cv2.inRange(hsv, SLOUGH_LOWER, SLOUGH_UPPER)

    # ─────────────────────────────────────────────────────────────
    # STEP 3 — CONTOUR EXTRACTION
    # ─────────────────────────────────────────────────────────────

    def get_wound_contour(self, mask):
        """
        Extract the wound contour from the binary mask.

        Handles three cases:
        1. No valid contour found        → status = "failed"
        2. Split wound detected          → status = "split_wound"
        3. Single valid contour found    → status = "success"

        Split wound detection: if the second largest contour is within
        SPLIT_THRESH (20%) of the largest contour area, both are
        treated as a split wound and merged into a single filled mask.

        Args:
            mask (np.ndarray): Binary mask from build_masks()

        Returns:
            tuple: (contour, pred_mask, status)
                contour   (np.ndarray | None): Largest valid contour points
                pred_mask (np.ndarray | None): Filled binary mask of wound region
                status    (str): "success" | "failed" | "split_wound"
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filter contours below minimum area threshold
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        # ── logical complexity: three-branch contour handling ──

        # case 1: no valid contours — segmentation failed
        if len(valid_contours) == 0:
            return None, None, "failed"

        # sort by area descending
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        largest_area = cv2.contourArea(valid_contours[0])

        # case 2: check for split wound
        if len(valid_contours) >= 2:
            second_area = cv2.contourArea(valid_contours[1])
            area_ratio = second_area / largest_area if largest_area > 0 else 0

            if area_ratio >= (1 - self.split_thresh):
                # split wound — merge top two contours into one filled mask
                pred_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(pred_mask, [valid_contours[0]], -1, 255, -1)
                cv2.drawContours(pred_mask, [valid_contours[1]], -1, 255, -1)
                return valid_contours[0], pred_mask, "split_wound"

        # case 3: single largest valid contour — success
        largest_contour = valid_contours[0]
        pred_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(pred_mask, [largest_contour], -1, 255, -1)

        return largest_contour, pred_mask, "success"

    # ─────────────────────────────────────────────────────────────
    # STEP 4 — IoU COMPUTATION
    # ─────────────────────────────────────────────────────────────

    def compute_iou(self, pred_mask, gt_mask):
        """
        Compute Intersection over Union between predicted and ground truth masks.

        Handles the edge case where union is zero to avoid division by zero.

        Args:
            pred_mask (np.ndarray): Predicted binary mask (0 or 255)
            gt_mask   (np.ndarray): Ground truth binary mask (0 or 255)

        Returns:
            float: IoU score in range [0.0, 1.0]
                   Returns 0.0 if union is zero
        """
        # resize gt_mask to match pred_mask if sizes differ
        if pred_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(
                gt_mask,
                (pred_mask.shape[1], pred_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # convert to boolean for logical operations
        pred_bool = pred_mask > 0
        gt_bool = gt_mask > 0

        intersection = np.logical_and(pred_bool, gt_bool).sum()
        union = np.logical_or(pred_bool, gt_bool).sum()

        # ── logical complexity: zero-union guard ──
        if union == 0:
            return 0.0

        return float(intersection / union)

    # ─────────────────────────────────────────────────────────────
    # STEP 5 — OVERLAY GENERATION
    # ─────────────────────────────────────────────────────────────

    def save_overlays(self, image_id, image, pred_mask, gt_mask, output_dir):
        """
        Save three overlay images and a thumbnail for visualization.

        Outputs:
        - {image_id}_original.jpg     — original wound image
        - {image_id}_predicted.jpg    — predicted mask overlay in green
        - {image_id}_groundtruth.jpg  — ground truth overlay in blue
        - thumbs/{image_id}.jpg       — 128x128 thumbnail of original

        Overlay color convention:
        - Green (0, 255, 0) = predicted mask
        - Blue  (255, 0, 0) = ground truth mask (BGR format)

        Args:
            image_id   (str):        Filename identifier for this image
            image      (np.ndarray): Original BGR image (512x512 after preprocess)
            pred_mask  (np.ndarray): Predicted binary mask (0 or 255)
            gt_mask    (np.ndarray): Ground truth binary mask (0 or 255)
            output_dir (str):        Base output directory (e.g. "outputs/")
        """
        annotated_dir = os.path.join(output_dir, "annotated_images")
        thumbs_dir = os.path.join(annotated_dir, "thumbs")

        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(thumbs_dir, exist_ok=True)

        # resize image and masks to consistent size
        image = cv2.resize(image, (512, 512))
        pred_mask = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # save original
        original_path = os.path.join(annotated_dir, f"{image_id}_original.jpg")
        if not os.path.exists(original_path):
            cv2.imwrite(original_path, image)

        # save predicted overlay (green)
        pred_overlay = self._create_overlay(image, pred_mask, color=(0, 255, 0))
        pred_path = os.path.join(annotated_dir, f"{image_id}_predicted.jpg")
        if not os.path.exists(pred_path):
            cv2.imwrite(pred_path, pred_overlay)

        # save ground truth overlay (blue)
        gt_overlay = self._create_overlay(image, gt_mask, color=(255, 0, 0))
        gt_path = os.path.join(annotated_dir, f"{image_id}_groundtruth.jpg")
        if not os.path.exists(gt_path):
            cv2.imwrite(gt_path, gt_overlay)

        # save thumbnail
        thumb = cv2.resize(image, (128, 128))
        thumb_path = os.path.join(thumbs_dir, f"{image_id}.jpg")
        if not os.path.exists(thumb_path):
            cv2.imwrite(thumb_path, thumb)

    def _create_overlay(self, image, mask, color):
        """
        Blend a colored mask onto the original image.

        Args:
            image (np.ndarray): Original BGR image
            mask  (np.ndarray): Binary mask (0 or 255)
            color (tuple):      BGR color for the overlay (e.g. (0, 255, 0) for green)

        Returns:
            np.ndarray: Image with colored mask blended at self.overlay_alpha opacity
        """
        colored_region = np.zeros_like(image)
        colored_region[mask > 0] = color
        blended = cv2.addWeighted(
            image, 1 - self.overlay_alpha, colored_region, self.overlay_alpha, 0
        )
        # draw contour border on top for clarity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, color, 2)
        return blended
