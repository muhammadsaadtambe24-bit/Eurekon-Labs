"""
Image classification using rule-based heuristics with confidence scoring.
Returns multiple tags per image based on visual characteristics.
"""

from PIL import Image
import numpy as np
from typing import List, Dict, Tuple


# def is_text_heavy(image_type: str, keywords: List[str]) -> bool:
#     """
#     Heuristic: return True ONLY if image is likely to contain significant text.
#     Used to skip OCR for normal photos (improves processing speed).

#     Returns True for:
#     - image_type in {"document", "screenshot"}
#     - OR keywords indicating text-heavy content (receipt, invoice, bill, poster,
#       id card, flyer, text-heavy)

#     Args:
#         image_type: Primary image type from classification (e.g. "photo", "document")
#         keywords: List of tags/keywords (from vision tags + object detection)

#     Returns:
#         True if OCR should run; False to skip OCR and save processing time
#     """
#     type_lower = (image_type or "").lower()
#     keywords_lower = [str(k).lower() for k in (keywords or [])]

#     # Document/screenshot types almost always have text worth extracting
#     if type_lower in {"document", "screenshot"}:
#         return True

#     # Keywords that indicate text-heavy content (documents, receipts, posters, etc.)
#     text_heavy_keywords = {
#         "receipt", "invoice", "bill", "poster", "id_card", "id card",
#         "flyer", "text-heavy", "document", "screenshot"
#     }
#     if any(kw in text_heavy_keywords for kw in keywords_lower):
#         return True

#     return False


def classify_image(image_path: str) -> List[str]:
    """
    Classify an image into multiple categories using confidence-based scoring.
    
    Returns top 2-3 most confident tags to avoid noise while capturing
    the image's multiple characteristics.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of classification tags (e.g., ["photo", "screenshot"])
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate confidence scores for each category
        scores = _calculate_classification_scores(img)
        
        # Sort by confidence and take top 2-3 tags
        # Minimum confidence threshold: 0.3 (30%)
        MIN_CONFIDENCE = 0.3
        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top tags that meet minimum confidence
        top_tags = [tag for tag, score in sorted_tags if score >= MIN_CONFIDENCE][:3]
        
        # Always return at least one tag (fallback to highest score)
        if not top_tags:
            top_tags = [sorted_tags[0][0]]
        
        return top_tags
        
    except Exception as e:
        print(f"Error classifying image {image_path}: {e}")
        return ["photo"]  # Safe fallback


def _calculate_classification_scores(img: Image.Image) -> Dict[str, float]:
    """
    Calculate confidence scores for each image category.
    
    Scores range from 0.0 to 1.0, where higher means more confident.
    Multiple categories can have high scores (non-exclusive).
    
    Returns:
        Dictionary mapping category names to confidence scores
    """
    width, height = img.size
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Convert to numpy for analysis
    img_array = np.array(img)
    
    # Initialize scores
    scores = {
        "photo": 0.0,
        "screenshot": 0.0,
        "graphic": 0.0,
        "diagram": 0.0,
        "document": 0.0
    }
    
    # --- SCREENSHOT DETECTION ---
    # Screenshots tend to have specific aspect ratios and sharp edges
    screenshot_score = 0.0
    
    # Common screenshot aspect ratios (16:9, 16:10, 4:3, etc.)
    common_ratios = [16/9, 16/10, 4/3, 3/2, 21/9]
    ratio_match = any(abs(aspect_ratio - r) < 0.05 or abs(aspect_ratio - 1/r) < 0.05 
                      for r in common_ratios)
    if ratio_match:
        screenshot_score += 0.3
    
    # Screenshots often have large uniform regions (UI elements)
    uniformity = _calculate_uniformity(img_array)
    if uniformity > 0.3:
        screenshot_score += 0.4
    
    # Sharp edges indicate UI elements
    edge_sharpness = _calculate_edge_sharpness(img_array)
    if edge_sharpness > 0.5:
        screenshot_score += 0.3
    
    scores["screenshot"] = min(screenshot_score, 1.0)
    
    # --- GRAPHIC/DESIGN DETECTION ---
    # Graphics have limited color palettes and geometric shapes
    graphic_score = 0.0
    
    unique_colors = _count_unique_colors(img_array)
    if unique_colors < 50:
        graphic_score += 0.5  # Very limited palette = graphic
    elif unique_colors < 200:
        graphic_score += 0.3
    
    # High saturation suggests designed graphics
    saturation = _calculate_saturation(img_array)
    if saturation > 0.6:
        graphic_score += 0.3
    
    # Sharp edges also indicate graphics
    if edge_sharpness > 0.6:
        graphic_score += 0.2
    
    scores["graphic"] = min(graphic_score, 1.0)
    
    # --- DIAGRAM DETECTION ---
    # Diagrams are usually simple, with lines and shapes
    diagram_score = 0.0
    
    # Low color count + high contrast = likely diagram
    if unique_colors < 30:
        diagram_score += 0.4
    
    # High white/light background percentage
    light_pixel_ratio = _calculate_light_pixel_ratio(img_array)
    if light_pixel_ratio > 0.7:
        diagram_score += 0.3
    
    # Sharp edges
    if edge_sharpness > 0.7:
        diagram_score += 0.3
    
    scores["diagram"] = min(diagram_score, 1.0)
    
    # --- DOCUMENT DETECTION ---
    # Documents are typically portrait, high-contrast, white background
    document_score = 0.0
    
    # Portrait orientation
    if aspect_ratio < 0.9:  # Taller than wide
        document_score += 0.3
    
    # Very high white background
    if light_pixel_ratio > 0.8:
        document_score += 0.4
    
    # Low saturation (mostly black text on white)
    if saturation < 0.2:
        document_score += 0.3
    
    scores["document"] = min(document_score, 1.0)
    
    # --- PHOTO DETECTION ---
    # Photos have natural color variation, organic shapes
    photo_score = 0.0
    
    # Wide range of colors
    if unique_colors > 500:
        photo_score += 0.4
    
    # Moderate saturation (not too flat, not too vibrant)
    if 0.3 < saturation < 0.7:
        photo_score += 0.3
    
    # Color variance (natural scenes have gradients)
    color_variance = _calculate_color_variance(img_array)
    if color_variance > 0.4:
        photo_score += 0.3
    
    # Photos typically don't have huge uniform regions
    if uniformity < 0.4:
        photo_score += 0.2
    
    scores["photo"] = min(photo_score, 1.0)
    
    return scores


def _calculate_uniformity(img_array: np.ndarray) -> float:
    """
    Calculate what percentage of the image is uniform color regions.
    Returns value between 0.0 and 1.0.
    """
    # Downsample for performance
    small = Image.fromarray(img_array).resize((50, 50), Image.Resampling.LANCZOS)
    small_array = np.array(small)
    
    # Calculate standard deviation of each pixel's neighborhood
    # Low std = uniform region
    h, w = small_array.shape[:2]
    uniform_pixels = 0
    total_pixels = 0
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            neighborhood = small_array[i-1:i+2, j-1:j+2]
            std = np.std(neighborhood)
            if std < 10:  # Very similar colors
                uniform_pixels += 1
            total_pixels += 1
    
    return uniform_pixels / total_pixels if total_pixels > 0 else 0.0


def _calculate_edge_sharpness(img_array: np.ndarray) -> float:
    """
    Calculate edge sharpness (0.0 = soft/blurry, 1.0 = sharp edges).
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array
    
    # Simple gradient-based edge detection
    small = Image.fromarray(gray).resize((100, 100), Image.Resampling.LANCZOS)
    small_array = np.array(small)
    
    # Compute gradients
    dx = np.abs(np.diff(small_array, axis=1))
    dy = np.abs(np.diff(small_array, axis=0))
    
    # High gradient values = sharp edges
    edge_strength = (np.mean(dx) + np.mean(dy)) / 2
    
    # Normalize to 0-1 range (255 is max possible gradient)
    return min(edge_strength / 50, 1.0)


def _count_unique_colors(img_array: np.ndarray, sample_size: int = 10000) -> int:
    """
    Count approximate unique colors (sample-based for performance).
    """
    h, w = img_array.shape[:2]
    total_pixels = h * w
    
    # Sample pixels if image is large
    if total_pixels > sample_size:
        indices = np.random.choice(total_pixels, sample_size, replace=False)
        flat_img = img_array.reshape(-1, 3)
        sampled = flat_img[indices]
    else:
        sampled = img_array.reshape(-1, 3)
    
    # Count unique RGB combinations
    unique = np.unique(sampled, axis=0)
    
    # Scale up if we sampled
    if total_pixels > sample_size:
        scaling_factor = total_pixels / sample_size
        return int(len(unique) * scaling_factor)
    
    return len(unique)


def _calculate_saturation(img_array: np.ndarray) -> float:
    """
    Calculate average saturation (0.0 = grayscale, 1.0 = vivid colors).
    """
    # Convert to HSV
    img_pil = Image.fromarray(img_array)
    hsv = img_pil.convert('HSV')
    hsv_array = np.array(hsv)
    
    # Saturation is the second channel in HSV
    saturation_channel = hsv_array[:, :, 1]
    
    # Normalize to 0-1 range
    return np.mean(saturation_channel) / 255.0


def _calculate_light_pixel_ratio(img_array: np.ndarray) -> float:
    """
    Calculate percentage of light/white pixels (brightness > 200).
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    light_pixels = np.sum(gray > 200)
    total_pixels = gray.size
    
    return light_pixels / total_pixels if total_pixels > 0 else 0.0


def _calculate_color_variance(img_array: np.ndarray) -> float:
    """
    Calculate color variance across the image (higher = more varied colors).
    """
    # Downsample for performance
    small = Image.fromarray(img_array).resize((50, 50), Image.Resampling.LANCZOS)
    small_array = np.array(small)
    
    # Calculate variance for each color channel
    r_var = np.var(small_array[:, :, 0])
    g_var = np.var(small_array[:, :, 1])
    b_var = np.var(small_array[:, :, 2])
    
    # Average variance across channels
    avg_var = (r_var + g_var + b_var) / 3
    
    # Normalize to 0-1 range (max variance is around 6000)
    return min(avg_var / 6000, 1.0)
