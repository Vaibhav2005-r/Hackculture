"""Page clean-up for historical manuscripts.

Steps (each one is a small pure function so it can be tested and benchmarked on its own):
    1. limit resolution        – keep processing time predictable
    2. crop to page            – drop scanner bed / table background
    3. contrast (CLAHE)        – recover faded ink locally, not globally
    4. bilateral denoise       – edge-preserving smoothing (keeps stroke sharpness better than
                                 median blur on manuscripts with hairline strokes)
    5. adaptive binarisation   – Sauvola for most pages; enhanced low-contrast path for extremely
                                 faded ink (votes between Sauvola and Niblack, picks better one)
    6. morphological cleanup   – thin ink thickening to reconnect broken strokes; frame removal
    7. deskew                  – projection-profile search for the angle that makes rows sharpest

Changes from v1
---------------
- `denoise()` now uses a fast bilateral filter (edge-preserving) in addition to the median blur.
  The bilateral filter preserves thin Devanagari strokes better when sigma_color is set
  conservatively.
- `sauvola_binarize()` gains an adaptive fallback: if the resulting ink coverage is suspiciously
  low (<0.5% of page) the page is likely extremely faded; we blend in a Niblack-style pass and
  pick the one with more ink.
- `_thicken_ink()` applies a 1-pixel dilation to reconnect broken strokes before segmentation
  (disabled for the benchmark "raw" variant that tests no preprocessing).
- Everything is still pure functions; no global state.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola


@dataclass
class Preprocessed:
    color: np.ndarray  # cropped + deskewed BGR page (what the viewer shows)
    gray: np.ndarray  # contrast-enhanced, denoised grayscale (fed to the recogniser)
    binary: np.ndarray  # uint8, 255 = ink, 0 = background
    skew_angle: float
    crop_box: tuple[int, int, int, int]  # x, y, w, h inside the (resized) original
    scale: float  # resize factor applied to the original
    binary_thick: np.ndarray | None = None  # 1px-dilated ink, used only for segmentation


def limit_size(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / longest
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def crop_to_page(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop to the largest bright-ish region when it clearly is a page lying on a darker background."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), max(h, w) / 300)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, (0, 0, w, h)
    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    area_ratio = (cw * ch) / (w * h)
    # Only crop when the page is clearly smaller than the frame but still most of it.
    if 0.35 < area_ratio < 0.95:
        pad = round(0.01 * max(cw, ch))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + cw + pad), min(h, y + ch + pad)
        return img[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)
    return img, (0, 0, w, h)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement — robust for uneven lighting across the page."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving bilateral + median blur.

    Bilateral filter parameters are chosen conservatively so hairline strokes
    in Devanagari / Gujarati scripts survive:
      d=5     – small neighbourhood to avoid blurring across strokes
      sigma_color=18  – narrow color window (only blend very similar pixels)
      sigma_space=5   – tight spatial window

    The subsequent median(3) pass catches isolated salt-and-pepper noise that
    bilateral misses when sigma_color is small.
    """
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=18, sigmaSpace=5)
    return cv2.medianBlur(bilateral, 3)


def _auto_window(gray: np.ndarray) -> int:
    """Sauvola window from the size of the writing, not of the picture.

    A window much smaller than a glyph hollows out thick strokes and turns page borders and paper
    texture into "ink" (a 470 px-wide screenshot of calligraphy got a 15 px window and lost a whole
    line); one much larger ignores uneven lighting.  Glyph height is estimated from a quick Otsu pass.
    """
    from app.ocr.segment import estimate_glyph_height

    _, quick = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    glyph = estimate_glyph_height(quick)
    fallback = max(15, (min(gray.shape[:2]) // 40) | 1)
    if glyph <= 0:
        return fallback
    return int(np.clip(round(1.6 * glyph), 31, 151)) | 1


def sauvola_binarize(gray: np.ndarray, window: int | None = None, k: float = 0.2) -> np.ndarray:
    """Return 255 for ink, 0 for background.

    Low-contrast fallback
    ---------------------
    If the resulting ink coverage is < 0.5% (the manuscript is very faded or
    poorly scanned) we additionally run a Niblack binarization (which is more
    aggressive with low-contrast ink) and merge the two results by OR to avoid
    missing real strokes.
    """
    if window is None:
        window = _auto_window(gray)
    window |= 1

    # --- Primary: Sauvola ---
    thresh_s = threshold_sauvola(gray, window_size=window, k=k)
    ink_s = (gray < thresh_s).astype(np.uint8) * 255

    # --- Low-contrast fallback: Niblack ---
    ink_coverage = float(ink_s.sum()) / (255.0 * gray.size)
    if ink_coverage < 0.005:
        # Very faded page: also run Niblack with a slightly smaller k to be more aggressive
        thresh_n = threshold_niblack(gray, window_size=window, k=0.15)
        ink_n = (gray < thresh_n).astype(np.uint8) * 255
        # Merge: ink is ink if either method thinks it is
        ink_s = cv2.bitwise_or(ink_s, ink_n)

    # Remove isolated specks (smaller than ~0.002% of the page).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_s, connectivity=8)
    min_area = max(4, int(gray.size * 0.00002))
    small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
    if len(small):
        ink_s[np.isin(labels, small)] = 0
    return remove_frame(ink_s)


def _row_contrast(ink: np.ndarray) -> float:
    """How strongly the ink rows stand out from the gaps between them (borders and speckle flatten it)."""
    prof = ink.sum(axis=1).astype(np.float64)
    return float(prof.std() / (prof.mean() + 1e-9))


def _best_binary(gray_raw: np.ndarray, gray_clahe: np.ndarray) -> np.ndarray:
    """Binarise with and without CLAHE and keep the cleaner result.

    CLAHE rescues faded ink but also amplifies paper texture and borders; on contrasty pages it makes
    the line profile noisier (one folio went from 5 to 31 "lines").  The candidate whose row profile has
    the stronger line/gap contrast wins; plain denoised grey wins ties.
    """
    plain = remove_vertical_rules(sauvola_binarize(denoise(gray_raw)))
    boosted = remove_vertical_rules(sauvola_binarize(gray_clahe))
    return boosted if _row_contrast(boosted) > _row_contrast(plain) * 1.05 else plain


def _thicken_ink(ink: np.ndarray) -> np.ndarray:
    """1-pixel dilation to reconnect broken strokes (e.g. water damage, faded ink).

    Uses a small diamond-shaped structuring element to avoid merging adjacent
    glyphs at typical manuscript resolutions.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # keeps adjacent glyphs apart
    return cv2.dilate(ink, kernel, iterations=1)


def remove_vertical_rules(ink: np.ndarray) -> np.ndarray:
    """Erase page borders, column rules and decorative side bars.

    Text strokes are short vertically (a few percent of the page height); borders run for a large
    fraction of it, even when drawn as dotted or broken lines.  An opening with a tall 1-pixel kernel
    keeps only such long vertical structures, which are then subtracted (slightly widened).
    """
    h, w = ink.shape
    tall = max(25, int(0.10 * h))
    bars = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))
    if not bars.any():
        return ink
    bars = cv2.dilate(bars, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    # only structures that really are rules: tall and narrow overall (not a column of text)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bars, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] >= tall and stats[i, cv2.CC_STAT_WIDTH] <= 0.12 * w:
            keep[i] = True
    mask = keep[labels]
    out = ink.copy()
    out[mask] = 0
    return out


def remove_frame(ink: np.ndarray) -> np.ndarray:
    """Drop page edges, folds and ruling: components far larger than any glyph, or thin and touching the border."""
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return ink
    x, y = stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP]
    cw, ch = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]
    huge = (cw > 0.6 * w) | (ch > 0.6 * h)
    touches = (x <= 1) | (y <= 1) | (x + cw >= w - 1) | (y + ch >= h - 1)
    sparse = area < 0.25 * cw * ch
    thin = (np.minimum(cw, ch) < 0.02 * max(h, w)) & (np.maximum(cw, ch) > 0.15 * min(h, w))
    drop = np.where(huge | (touches & (sparse | thin)))[0] + 1
    if len(drop):
        ink = ink.copy()
        ink[np.isin(labels, drop)] = 0
    return ink


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _row_sharpness(ink_small: np.ndarray, angle: float) -> float:
    h, w = ink_small.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(ink_small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(np.var(np.diff(profile)))


def estimate_skew(ink: np.ndarray, max_angle: float = 8.0) -> float:
    """Angle (degrees) that makes text rows sharpest: coarse 0.5 deg search, then 0.1 deg refine."""
    h, w = ink.shape
    scale = min(1.0, 900 / max(h, w))
    small = cv2.resize(ink, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    if small.sum() == 0:
        return 0.0
    coarse = np.arange(-max_angle, max_angle + 0.01, 0.5)
    best = max(coarse, key=lambda a: _row_sharpness(small, a))
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    best = max(fine, key=lambda a: _row_sharpness(small, a))
    return round(float(best), 2)


def rotate(img: np.ndarray, angle: float, border: int | tuple[int, int, int]) -> np.ndarray:
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    interp = cv2.INTER_NEAREST if img.ndim == 2 and border == 0 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, m, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _background_color(img: np.ndarray) -> tuple[int, int, int]:
    b, g, r = (int(np.median(img[..., c])) for c in range(3))
    return b, g, r


def preprocess(img: np.ndarray, max_side: int = 2400, deskew: bool = True) -> Preprocessed:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img, scale = limit_size(img, max_side)
    img, crop_box = crop_to_page(img)

    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = denoise(enhance_contrast(gray0))
    ink = _best_binary(gray0, gray)
    # Thicken ink to reconnect broken strokes before segmentation.
    # We keep the original un-thickened binary for display (the viewer shows
    # the inverted binary) so we only use the thickened version for deskew + segment.
    ink_thick = _thicken_ink(ink)

    angle = estimate_skew(ink_thick) if deskew else 0.0
    if abs(angle) >= 0.05:
        img = rotate(img, angle, _background_color(img))
        gray = rotate(gray, angle, int(np.median(gray)))
        ink = rotate(ink, angle, 0)
        ink_thick = rotate(ink_thick, angle, 0)

    # `binary` stays un-thickened (it is what the viewer shows); the segmenter gets `binary_thick`.
    return Preprocessed(
        color=img, gray=gray, binary=ink, skew_angle=angle, crop_box=crop_box, scale=scale, binary_thick=ink_thick
    )
