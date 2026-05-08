import os
import cv2
import numpy as np


class WoundDataset:
    """
    Handles loading, validation, and output directory setup for the FUSeg dataset.
    
    Responsibilities:
    - Load image/mask pairs from disk
    - Validate that files exist and masks are binary
    - Auto-correct non-binary masks
    - Set up output directory structure
    - Skip and warn on missing files rather than crashing
    """

    def __init__(self):
        self.skipped = []   # tracks skipped image_ids with reasons

    # ─────────────────────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────────────────────

    def load_pair(self, image_path, mask_path):
        """
        Load a single image/mask pair from disk.

        Validates:
        - Image file exists
        - Mask file exists
        - Mask is binary (pixel values in [0, 1, 255])
        - If not binary: applies correction threshold (>127 -> 255, else 0)

        Args:
            image_path (str): Path to the wound image file
            mask_path  (str): Path to the ground truth mask file

        Returns:
            tuple: (image, mask) as numpy arrays, or (None, None) if loading fails

        Notes:
            FUSeg mask pixel values are 1 (wound) and 0 (background).
            This differs from the OpenCV convention of 255/0.
            The pipeline normalizes masks to 0/255 for consistency.
        """
        # check image exists
        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found, skipping: {image_path}")
            return None, None

        # check mask exists
        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask not found, skipping: {mask_path}")
            return None, None

        # load image in BGR
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not read image, skipping: {image_path}")
            return None, None

        # load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARNING] Could not read mask, skipping: {mask_path}")
            return None, None

        # FUSeg masks use pixel value 1 for wound — normalize to 255
        if mask.max() == 1:
            mask = mask * 255

        # validate mask is binary — only 0 and 255 allowed
        unique_values = set(np.unique(mask))
        allowed_values = {0, 1, 255} # alt syntax for set in python

        if not unique_values.issubset(allowed_values):
            print(f"[WARNING] Non-binary mask detected, applying correction: {mask_path}")
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)

        return image, mask

    def load_all_pairs(self, image_dir, mask_dir):
        """
        Load all valid image/mask pairs from the given directories.

        Matches images to masks by filename. Skips pairs where either
        file is missing, logging a warning for each skip.

        Args:
            image_dir (str): Path to directory containing wound images
            mask_dir  (str): Path to directory containing ground truth masks

        Returns:
            list: List of tuples (image_id, image, mask) for all valid pairs
                  image_id is the filename without extension
        """
        valid_pairs = []

        # get all image files sorted for reproducibility
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(image_files) == 0:
            print(f"[ERROR] No image files found in: {image_dir}")
            return valid_pairs

        print(f"[INFO] Found {len(image_files)} images in {image_dir}")

        for filename in image_files:
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, filename)

            # try matching mask by same filename with .png extension
            mask_filename = image_id + ".png"
            mask_path = os.path.join(mask_dir, mask_filename)

            # fallback: try same extension as image
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, filename)

            image, mask = self.load_pair(image_path, mask_path)

            if image is None or mask is None:
                self.skipped.append(image_id)
                continue

            valid_pairs.append((image_id, image, mask))

        print(f"[INFO] Loaded {len(valid_pairs)} valid pairs")
        if self.skipped:
            print(f"[INFO] Skipped {len(self.skipped)} pairs due to missing files")

        return valid_pairs

    # ─────────────────────────────────────────────────────────────
    # OUTPUT DIRECTORY SETUP
    # ─────────────────────────────────────────────────────────────

    def setup_output_dir(self, base_dir):
        """
        Create the output directory structure if it does not already exist.

        Creates:
            outputs/
            outputs/annotated_images/
            outputs/plots/

        Does not overwrite existing files.

        Args:
            base_dir (str): Path to the base output directory (e.g. "outputs/")
        """
        subdirs = [
            base_dir,
            os.path.join(base_dir, "annotated_images"),
            os.path.join(base_dir, "plots"),
        ]

        for directory in subdirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"[INFO] Created directory: {directory}")
            else:
                print(f"[INFO] Directory already exists: {directory}")

    # ─────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────

    def print_summary(self, mean_iou=None):
        """
        Print a summary after all processing is complete.

        Args:
            mean_iou (float, optional): Mean IoU across the dataset
        """
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Skipped pairs:     {len(self.skipped)}")
        if self.skipped:
            for s in self.skipped:
                print(f"  - {s}")
        if mean_iou is not None:
            print(f"Mean IoU:          {mean_iou:.3f}")
        print("=" * 50 + "\n")