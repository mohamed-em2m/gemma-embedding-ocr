"""Handwriting and document image preprocessing utilities.

Includes:
- Slant estimation and deslanting (shear transformation).
- Adaptive binarization (Otsu & Sauvola thresholding).
- Contrast enhancement and border padding.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_slant_angle(
    gray: np.ndarray,
    min_angle: float = -45.0,
    max_angle: float = 45.0,
    num_steps: int = 19,
) -> float:
    """Estimate the dominant handwriting slant angle using vertical projection variance.

    The optimal vertical alignment angle maximizes the variance of vertical projection
    profiles (strokes aligned vertically create sharp peaks and deep troughs).

    Args:
        gray: Grayscale uint8 image of the word (dark text on light background).
        min_angle: Minimum angle in degrees to test.
        max_angle: Maximum angle in degrees to test.
        num_steps: Number of candidate angles to test.

    Returns:
        Estimated slant angle in degrees.
    """
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 0.0

    # Invert so text ink is foreground (positive)
    # Binarize with Otsu
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -1.0

    angles = np.linspace(min_angle, max_angle, num_steps)
    for angle in angles:
        # Shear matrix: x' = x - tan(angle) * y
        shear = float(np.tan(np.radians(angle)))
        # Center of shearing is the vertical midpoint
        m = np.array(
            [[1.0, -shear, shear * (h / 2.0)], [0.0, 1.0, 0.0]], dtype=np.float32
        )

        # Warped binary image
        sheared = cv2.warpAffine(
            bin_img,
            m,
            (int(w), int(h)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )

        # Compute vertical projection
        v_proj = np.sum(sheared > 0, axis=0, dtype=np.float32)
        # Vertical alignment metric: variance of vertical projection
        score = float(np.var(v_proj))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle


def deslant_image(
    image: np.ndarray,
    angle: float | None = None,
    min_angle: float = -40.0,
    max_angle: float = 40.0,
    bg_color: int = 255,
) -> np.ndarray:
    """Deslant a word image by shearing along the horizontal axis.

    Args:
        image: RGB or grayscale image array.
        angle: Fixed slant angle in degrees. If None, angle is estimated.
        min_angle: Minimum slant angle for automatic estimation.
        max_angle: Maximum slant angle for automatic estimation.
        bg_color: Background color value (default 255 white).

    Returns:
        Deslanted image array matching the input type and number of channels.
    """
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return image.copy()

    is_rgb = image.ndim == 3 and image.shape[2] == 3
    if is_rgb:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image if image.ndim == 2 else image[:, :, 0]

    if angle is None:
        angle = estimate_slant_angle(gray, min_angle=min_angle, max_angle=max_angle)

    # If angle is minimal, no noticeable deslanting required
    if abs(angle) < 3.0:
        return image.copy()

    shear = float(np.tan(np.radians(angle)))
    # Calculate extra width needed so character edges aren't clipped
    extra_w = int(abs(shear) * h)
    new_w = w + extra_w

    # Transformation matrix with horizontal centering shift
    shift_x = extra_w / 2.0
    m = np.array(
        [[1.0, -shear, shift_x + shear * (h / 2.0)], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    border_tuple: tuple[float, ...] = (
        (float(bg_color), float(bg_color), float(bg_color))
        if is_rgb
        else (float(bg_color),)
    )
    deslanted: np.ndarray = cv2.warpAffine(
        image,
        m,
        (int(new_w), int(h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_tuple,
    )
    return deslanted


def adaptive_binarize(
    image: np.ndarray,
    method: str = "otsu",
    denoise: bool = True,
) -> np.ndarray:
    """Enhance and binarize a handwriting image into clean black ink on white background.

    Args:
        image: Input RGB or grayscale image.
        method: Binarization method ('otsu' or 'adaptive_gaussian').
        denoise: Whether to apply Gaussian blur denoising first.

    Returns:
        RGB uint8 image with clean black text on white background (H, W, 3).
    """
    is_rgb = image.ndim == 3 and image.shape[2] == 3
    if is_rgb:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy() if image.ndim == 2 else image[:, :, 0].copy()

    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if method == "adaptive_gaussian":
        bin_img = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25,
            C=11,
        )
    else:
        # Otsu thresholding
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Return as 3-channel RGB uint8 for model encoders
    return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)


def estimate_char_count_and_width(
    image: np.ndarray,
    min_char_width: float = 10.0,
) -> tuple[int, float]:
    """Estimate character count and average character width from word crop geometry.

    In Latin handwriting and print, character width is strongly bounded by line height
    and aspect ratio. Average character width in Latin script is roughly 0.45x - 0.70x of
    the core x-height / total line height.

    Args:
        image: Word crop (H, W, C) or (H, W).
        min_char_width: Minimum physical pixel width for a valid single glyph.

    Returns:
        (estimated_chars, avg_char_width_px)
    """
    h, w = image.shape[:2]
    if h < 6 or w < 6:
        return 1, float(w)

    # Convert to grayscale binary to count connected ink components or transitions
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

    # Otsu binary (text positive)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Height-based aspect ratio estimate
    # In Latin words, typical character aspect ratio (width / height) is ~0.45 to ~0.65
    nominal_char_width = max(min_char_width, h * 0.52)
    aspect_est = max(1, int(round(w / nominal_char_width)))

    # 2. Ink transition count estimate (horizontal scan along midline)
    mid_y = h // 2
    y_band = bin_img[max(0, mid_y - 2) : min(h, mid_y + 3), :]
    profile = np.max(y_band, axis=0) > 0
    # Count transitions from 0 to 1 (stroke starts)
    diffs = np.diff(profile.astype(np.int32))
    stroke_starts = int(np.sum(diffs > 0))
    # In Latin cursive, average letters have ~1.5 to 2 vertical strokes
    transition_est = max(1, int(round(stroke_starts / 1.6))) if stroke_starts > 0 else 1

    # Fuse aspect estimate and transition estimate
    fused_count = int(round(0.6 * aspect_est + 0.4 * transition_est))
    fused_count = max(1, fused_count)
    avg_char_w = float(w) / float(fused_count)

    return fused_count, avg_char_w


def preprocess_handwriting_crop(
    crop: np.ndarray,
    deslant: bool = True,
    binarize: bool = True,
    padding: int = 4,
) -> np.ndarray:
    """Full preprocessing pipeline for handwriting word crops.

    Steps:
    1. Deslant image (upright vertical strokes).
    2. Adaptive Otsu binarization (clean black text on pure white paper).
    3. Padding with white border to preserve outer stroke boundaries.

    Args:
        crop: Input RGB uint8 crop.
        deslant: Whether to apply automatic slant correction.
        binarize: Whether to apply adaptive binarization.
        padding: Border padding size in pixels.

    Returns:
        Preprocessed RGB uint8 image.
    """
    out = crop.copy()

    # Ensure RGB uint8
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    elif out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)

    # 1. Deslant
    if deslant:
        out = deslant_image(out)

    # 2. Binarize & contrast enhance
    if binarize:
        out = adaptive_binarize(out, method="otsu", denoise=True)

    # 3. Add light white padding so strokes don't touch image borders
    if padding > 0:
        out = cv2.copyMakeBorder(
            out,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return out
