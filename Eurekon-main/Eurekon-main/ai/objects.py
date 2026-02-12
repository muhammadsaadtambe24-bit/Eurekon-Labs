"""
Lightweight object/content detection using visual heuristics.
No ML models - uses color analysis, edge detection, and patterns.
Tuned for accuracy and precision to reduce false positives.
"""

from PIL import Image
import numpy as np
from typing import List, Set, Tuple


# Object/content categories we can detect
DETECTABLE_OBJECTS = [
    "text-heavy",    # Documents, presentations with lots of text
    "people",        # Likely contains people (skin tones)
    "nature",        # Plants, flowers, outdoor scenes
    "animals",       # Animal fur/patterns
]

# Minimum confidence (0.0-1.0) to include a category in results
CONFIDENCE_THRESHOLD = 0.52


def detect_objects(image_path: str, ocr_text: str = "") -> List[str]:
    """
    Detect likely objects/content in an image using heuristics.
    Returns only categories that meet a confidence threshold to reduce false positives.

    Args:
        image_path: Path to the image file
        ocr_text: OCR-extracted text (if available) for text density analysis

    Returns:
        List of detected object categories (e.g., ["text-heavy", "people"])
    """
    try:
        img = Image.open(image_path)

        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize for consistent analysis (slightly larger for better texture)
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        img_array = np.array(img)

        # Compute shared features once
        h, w = img_array.shape[:2]
        total_pixels = h * w
        gray = np.mean(img_array, axis=2).astype(np.uint8) if len(img_array.shape) == 3 else img_array

        # Get confidence scores for each category
        text_score = _score_text(img_array, gray, ocr_text)
        people_score = _score_people(img_array, h, w, total_pixels)
        nature_score = _score_nature(img_array, total_pixels)
        animal_score = _score_animals(img_array, gray, total_pixels)

        # If image is strongly text/document, reduce confidence in others (avoid doc with a plant = nature)
        if text_score >= 0.7:
            nature_score *= 0.5
            people_score *= 0.6
            animal_score *= 0.5

        detected = []
        if text_score >= CONFIDENCE_THRESHOLD:
            detected.append("text-heavy")
        if people_score >= CONFIDENCE_THRESHOLD:
            detected.append("people")
        if nature_score >= CONFIDENCE_THRESHOLD:
            detected.append("nature")
        if animal_score >= CONFIDENCE_THRESHOLD:
            detected.append("animals")

        return sorted(detected)

    except Exception as e:
        print(f"Error detecting objects in {image_path}: {e}")
        return []


def _score_text(img_array: np.ndarray, gray: np.ndarray, ocr_text: str) -> float:
    """
    Score how much the image looks like text/document content.
    Uses bimodal intensity, edge density, and OCR length.
    """
    h, w = gray.shape
    total = h * w

    # Bimodal: text has dark (ink) and light (paper) peaks, few mid-tones
    dark_pixels = np.sum(gray < 70)
    light_pixels = np.sum(gray > 185)
    mid_pixels = np.sum((gray >= 70) & (gray <= 185))

    dark_ratio = dark_pixels / total
    light_ratio = light_pixels / total
    mid_ratio = mid_pixels / total

    # Strong bimodal: high dark + high light, low mid
    bimodal_strength = 0.0
    if mid_ratio < 0.5 and (dark_ratio + light_ratio) > 0.5:
        bimodal_strength = (1.0 - mid_ratio) * (dark_ratio + light_ratio)
    bimodal_strength = min(1.0, bimodal_strength * 1.2)

    # Row variance: text has repeated line structure
    row_vars = np.var(gray, axis=1)
    variance_changes = np.sum(np.abs(np.diff(row_vars)) > 150)
    line_structure = min(1.0, variance_changes / max(1, h * 0.25)) if h > 1 else 0.0

    # OCR is strong evidence
    ocr_score = 0.0
    if ocr_text and len(ocr_text.strip()) > 80:
        ocr_score = min(1.0, len(ocr_text) / 500)  # cap at 500 chars
    elif ocr_text and len(ocr_text.strip()) > 20:
        ocr_score = 0.4

    # Combine: require either strong OCR or (bimodal + some line structure)
    if ocr_score >= 0.5:
        return 0.5 + ocr_score * 0.5
    visual = bimodal_strength * 0.7 + line_structure * 0.3
    if visual > 0.6 and mid_ratio < 0.45:
        return visual
    return ocr_score * 0.6 + visual * 0.4


def _score_people(img_array: np.ndarray, h: int, w: int, total_pixels: int) -> float:
    """
    Score likelihood of people using skin tone detection in YCbCr and spatial clustering.
    """
    # YCbCr is more reliable for skin than RGB (avoids brown objects)
    # Skin in YCbCr: Cr in [133, 173], Cb in [77, 127], Y for brightness
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    cb = (128 - 0.169 * r - 0.331 * g + 0.5 * b).astype(np.float32)
    cr = (128 + 0.5 * r - 0.419 * g - 0.081 * b).astype(np.float32)

    # Skin mask: standard YCbCr ranges for skin
    skin_mask = (
        (cr >= 133) & (cr <= 173) &
        (cb >= 77) & (cb <= 127) &
        (y >= 60) & (y <= 250)
    )

    skin_ratio = np.sum(skin_mask) / total_pixels

    # Require spatial clustering: at least one block (e.g. 1/4 of image) has notable skin
    block_h, block_w = max(1, h // 4), max(1, w // 4)
    max_block_ratio = 0.0
    for i in range(0, h - block_h + 1, block_h):
        for j in range(0, w - block_w + 1, block_w):
            block = skin_mask[i : i + block_h, j : j + block_w]
            max_block_ratio = max(max_block_ratio, np.sum(block) / block.size)

    # Score: need both global presence and at least one cluster (face/arm region)
    if skin_ratio < 0.03:
        return 0.0
    if max_block_ratio < 0.08:
        return skin_ratio * 2.0  # weak: scattered skin-colored pixels only
    # Strong: clustered skin (likely face/hand)
    return min(1.0, 0.3 + skin_ratio * 4.0 + max_block_ratio * 1.5)


def _score_nature(img_array: np.ndarray, total_pixels: int) -> float:
    """
    Score natural/outdoor content: green (plants), sky blue, earth tones.
    """
    if total_pixels > 8000:
        flat = img_array.reshape(-1, 3)
        idx = np.random.choice(len(flat), 8000, replace=False)
        pixels = flat[idx]
    else:
        pixels = img_array.reshape(-1, 3)

    n = len(pixels)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # Green-dominant (plants, grass) – require clear green
    green_strong = (g > r) & (g > b) & (g > 90) & (g - r > 15)
    green_ratio = np.sum(green_strong) / n

    # Sky blue – but not UI blue (sky often has some green)
    blue_sky = (b > r) & (b > g) & (b > 100) & (b - g < 80)
    blue_ratio = np.sum(blue_sky) / n

    # Earth/brown
    brown = (r > 60) & (r < 160) & (g > 40) & (g < 130) & (b > 20) & (b < 90) & (r > g) & (g > b)
    brown_ratio = np.sum(brown) / n

    # Avoid false positive: mostly blue (e.g. screenshot/UI) with no green
    if blue_ratio > 0.35 and green_ratio < 0.08:
        return 0.0

    # Nature: dominant green, or outdoor (green + blue + brown)
    if green_ratio > 0.28:
        return min(1.0, 0.5 + green_ratio)
    if green_ratio > 0.18 and (blue_ratio > 0.12 or brown_ratio > 0.12):
        return 0.4 + green_ratio + (blue_ratio + brown_ratio) * 0.5
    if green_ratio > 0.12 and blue_ratio > 0.15 and brown_ratio > 0.08:
        return 0.35 + green_ratio + blue_ratio
    return green_ratio * 2.0 if green_ratio > 0.10 else 0.0


def _score_animals(img_array: np.ndarray, gray: np.ndarray, total_pixels: int) -> float:
    """
    Score animal presence: fur-like texture + animal-like colors (brown, black, white, orange).
    """
    # Texture on a downscaled patch
    small = np.array(
        Image.fromarray(gray).resize((120, 120), Image.Resampling.LANCZOS)
    )
    h, w = small.shape
    window_size = 5
    texture_scores = []
    for i in range(window_size, h - window_size, 3):
        for j in range(window_size, w - window_size, 3):
            window = small[i - window_size : i + window_size, j - window_size : j + window_size]
            texture_scores.append(np.std(window))

    if not texture_scores:
        return 0.0

    avg_texture = np.mean(texture_scores)
    # Fur: moderate texture (narrower band to avoid grass/fabric)
    fur_texture = 0.0
    if 18 < avg_texture < 48:
        fur_texture = 1.0 - abs(avg_texture - 33) / 20.0
        fur_texture = max(0.0, fur_texture)

    # Animal-like colors
    if total_pixels > 8000:
        flat = img_array.reshape(-1, 3)
        idx = np.random.choice(len(flat), 8000, replace=False)
        pixels = flat[idx]
    else:
        pixels = img_array.reshape(-1, 3)

    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    brown_tan = (r > 85) & (r < 175) & (g > 55) & (g < 135) & (b > 25) & (b < 95) & (r > g) & (g > b)
    black_fur = (np.maximum(np.maximum(r, g), b) < 55)
    white_fur = (np.minimum(np.minimum(r, g), b) > 198)
    orange_ginger = (r > 175) & (g > 75) & (g < 145) & (b < 85)

    animal_pixels = np.sum(brown_tan | black_fur | white_fur | orange_ginger)
    color_ratio = animal_pixels / len(pixels)

    # Need both texture and color; avoid flagging documents (high black/white from text)
    dark_light_doc = np.sum((r < 60) | (r > 240)) / len(pixels)
    if dark_light_doc > 0.6:
        color_ratio *= 0.4  # likely document

    combined = fur_texture * 0.55 + min(1.0, color_ratio * 2.0) * 0.45
    return combined


def get_object_hints(image_metadata: dict) -> List[str]:
    """
    Helper to get object hints from existing metadata.
    """
    return image_metadata.get("objects", [])