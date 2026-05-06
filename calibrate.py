"""
calibrate.py — Interactive HSV Threshold Calibration Tool

Use this script to find the correct HSV range constants for segmentation.py
before running the full pipeline on the FUSeg dataset.

Usage:
    conda activate dfu-analysis
    python calibrate.py

Controls:
    n         → next image
    p         → previous image
    g         → switch to granulation mask mode
    s         → switch to slough mask mode
    c         → print current constants to terminal
    q         → quit and print final constants

Output:
    Prints tuned HSV constants ready to copy into segmentation.py
"""

import os
import glob
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GRANULATION_IMAGES_DIR = "data/calibration/images/granulation"
SLOUGH_IMAGES_DIR = "data/calibration/images/slough"
IMAGE_DIR = SLOUGH_IMAGES_DIR # * Using a selected sample of 20 images for each in calibration folder. 
WINDOW_SIZE = (512, 512)


# ─────────────────────────────────────────────────────────────────────────────
# TRACKBAR CALLBACK — required by OpenCV but not used directly
# ─────────────────────────────────────────────────────────────────────────────


def nothing(x):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def load_images(image_dir):
    """Load all image paths from the validation directory."""
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not paths:
        print(f"[ERROR] No images found in {image_dir}")
        print("        Make sure you have downloaded the FUSeg dataset.")
        print("        See README.md for download instructions.")
        exit(1)
    print(f"[INFO] Found {len(paths)} images in {image_dir}")
    return paths


def preprocess_image(image_path):
    """Load and preprocess a single image for calibration."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Could not load image: {image_path}")
        return None, None

    # resize to consistent display size
    image = cv2.resize(image, WINDOW_SIZE)

    # apply gaussian blur — same as segmentation.py pipeline
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return image, hsv


def build_mask_from_trackbars(hsv, window_name, mode):
    """
    Build a mask using current trackbar values.

    For granulation mode: applies two ranges and combines with bitwise_or
    to handle the red hue wrapping around 180.

    For slough mode: applies a single yellow-tan range.

    Args:
        hsv         (np.ndarray): HSV image
        window_name (str):        OpenCV window name to read trackbars from
        mode        (str):        "granulation" | "slough"

    Returns:
        np.ndarray: Binary mask (0 or 255)
    """
    h_min = cv2.getTrackbarPos("H min", window_name)
    h_max = cv2.getTrackbarPos("H max", window_name)
    h_min2 = cv2.getTrackbarPos("H min2", window_name)
    h_max2 = cv2.getTrackbarPos("H max2", window_name)
    s_min = cv2.getTrackbarPos("S min", window_name)
    v_min = cv2.getTrackbarPos("V min", window_name)

    lower1 = np.array([h_min, s_min, v_min])
    upper1 = np.array([h_max, 255, 255])
    lower2 = np.array([h_min2, s_min, v_min])
    upper2 = np.array([h_max2, 255, 255])

    if mode == "granulation":
        # red wraps around — need two ranges
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # slough — single yellow range
        mask = cv2.inRange(hsv, lower1, upper1)

    # apply same morphological cleanup as segmentation.py
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def create_display(image, hsv, mask, image_path, mode, image_idx, total):
    """
    Create the display frame shown in the calibration window.

    Shows three panels side by side:
    - Original image
    - Mask overlay (wound region highlighted)
    - Binary mask

    Args:
        image      (np.ndarray): Original BGR image
        hsv        (np.ndarray): HSV image
        mask       (np.ndarray): Current binary mask
        image_path (str):        Path to current image for display
        mode       (str):        Current calibration mode
        image_idx  (int):        Current image index
        total      (int):        Total number of images

    Returns:
        np.ndarray: Combined display frame
    """
    # colored overlay on original image
    overlay = image.copy()
    colored_region = np.zeros_like(image)

    if mode == "granulation":
        colored_region[mask > 0] = (0, 255, 0)  # green for granulation
    else:
        colored_region[mask > 0] = (0, 215, 255)  # yellow for slough

    blended = cv2.addWeighted(image, 0.6, colored_region, 0.4, 0)

    # draw contours on overlay
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = (0, 255, 0) if mode == "granulation" else (0, 215, 255)
    cv2.drawContours(blended, contours, -1, color, 2)

    # binary mask as BGR for display
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # combine three panels side by side
    display = np.hstack([image, blended, mask_bgr])

    # add info text
    filename = os.path.basename(image_path)
    mode_str = "GRANULATION (g)" if mode == "granulation" else "SLOUGH (s)"
    cv2.putText(
        display,
        f"{filename}  [{image_idx+1}/{total}]",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        display,
        f"Mode: {mode_str}  |  n=next  p=prev  c=print  q=quit",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # wound pixel count
    wound_pixels = int(np.sum(mask > 0))
    cv2.putText(
        display,
        f"Wound pixels: {wound_pixels}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return display


def print_constants(window_name, mode):
    """Print current trackbar values as copy-pasteable constants for segmentation.py."""
    h_min = cv2.getTrackbarPos("H min", window_name)
    h_max = cv2.getTrackbarPos("H max", window_name)
    h_min2 = cv2.getTrackbarPos("H min2", window_name)
    h_max2 = cv2.getTrackbarPos("H max2", window_name)
    s_min = cv2.getTrackbarPos("S min", window_name)
    v_min = cv2.getTrackbarPos("V min", window_name)

    print("\n" + "=" * 50)
    print(f"CURRENT CONSTANTS — {mode.upper()} MODE")
    print("=" * 50)

    if mode == "granulation":
        print(f"GRANULATION_LOWER_1 = np.array([{h_min},  {s_min}, {v_min}])")
        print(f"GRANULATION_UPPER_1 = np.array([{h_max},  255, 255])")
        print(f"GRANULATION_LOWER_2 = np.array([{h_min2}, {s_min}, {v_min}])")
        print(f"GRANULATION_UPPER_2 = np.array([{h_max2}, 255, 255])")
    else:
        print(f"SLOUGH_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"SLOUGH_UPPER = np.array([{h_max}, 255, 255])")

    print("=" * 50)
    print("Copy these into the constants section of segmentation.py\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 50)
    print("HSV CALIBRATION TOOL")
    print("=" * 50)
    print("Controls:")
    print("  n → next image")
    print("  p → previous image")
    print("  g → granulation mode (red/pink)")
    print("  s → slough mode (yellow/tan)")
    print("  c → print current constants")
    print("  q → quit and print final constants")
    print("=" * 50 + "\n")

    # load images
    image_paths = load_images(IMAGE_DIR)
    total = len(image_paths)
    image_idx = 0
    mode = "granulation"  # start in granulation mode

    # create window
    window_name = "HSV Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_SIZE[0] * 3, WINDOW_SIZE[1] + 80)

    # ── create trackbars ──
    # Range 1 — lower red (granulation) or yellow (slough)
    cv2.createTrackbar("H min", window_name, 0, 180, nothing)
    cv2.createTrackbar("H max", window_name, 15, 180, nothing)
    # Range 2 — upper red (granulation only, wraps around 180)
    cv2.createTrackbar("H min2", window_name, 165, 180, nothing)
    cv2.createTrackbar("H max2", window_name, 180, 180, nothing)
    # Shared saturation and value minimums
    cv2.createTrackbar("S min", window_name, 40, 255, nothing)
    cv2.createTrackbar("V min", window_name, 50, 255, nothing)

    # load first image
    image, hsv = preprocess_image(image_paths[image_idx])

    while True:
        if image is None:
            image_idx = (image_idx + 1) % total
            image, hsv = preprocess_image(image_paths[image_idx])
            continue

        # build mask from current trackbar values
        mask = build_mask_from_trackbars(hsv, window_name, mode)

        # create and show display
        display = create_display(
            image, hsv, mask, image_paths[image_idx], mode, image_idx, total
        )
        cv2.imshow(window_name, display)

        # handle key presses
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            # quit — print final constants
            print_constants(window_name, mode)
            break

        elif key == ord("n"):
            # next image
            image_idx = (image_idx + 1) % total
            image, hsv = preprocess_image(image_paths[image_idx])
            print(
                f"[INFO] Image {image_idx + 1}/{total}: {os.path.basename(image_paths[image_idx])}"
            )

        elif key == ord("p"):
            # previous image
            image_idx = (image_idx - 1) % total
            image, hsv = preprocess_image(image_paths[image_idx])
            print(
                f"[INFO] Image {image_idx + 1}/{total}: {os.path.basename(image_paths[image_idx])}"
            )

        elif key == ord("g"):
            # switch to granulation mode
            mode = "granulation"
            cv2.setTrackbarPos("H min", window_name, 0)
            cv2.setTrackbarPos("H max", window_name, 15)
            cv2.setTrackbarPos("H min2", window_name, 165)
            cv2.setTrackbarPos("H max2", window_name, 180)
            cv2.setTrackbarPos("S min", window_name, 40)
            cv2.setTrackbarPos("V min", window_name, 50)
            print("[INFO] Switched to GRANULATION mode")

        elif key == ord("s"):
            # switch to slough mode
            mode = "slough"
            cv2.setTrackbarPos("H min", window_name, 15)
            cv2.setTrackbarPos("H max", window_name, 35)
            cv2.setTrackbarPos("H min2", window_name, 15)
            cv2.setTrackbarPos("H max2", window_name, 35)
            cv2.setTrackbarPos("S min", window_name, 30)
            cv2.setTrackbarPos("V min", window_name, 50)
            print("[INFO] Switched to SLOUGH mode")

        elif key == ord("c"):
            # print current constants
            print_constants(window_name, mode)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
