"""
Extract dominant colors from images and map to base color palette.
Maps similar shades to base colors for better search UX.
"""

from PIL import Image
import numpy as np
from typing import List, Tuple
from collections import Counter


# Base color palette - only these colors are returned
BASE_COLORS = [
    "red", "blue", "green", "yellow", 
    "black", "white", "gray", "brown", "purple", "pink"  # Added pink as distinct color
]


# RGB ranges for base colors (center point and tolerance)
COLOR_DEFINITIONS = {
    "red": {
        "ranges": [
            ((200, 0, 0), (255, 100, 100)),    # Pure red
            ((150, 0, 0), (255, 50, 50)),      # Dark red/maroon
        ]
    },
    "pink": {  # Pink as its own base color, not mapped to red
        "ranges": [
            ((255, 150, 180), (255, 220, 240)), # Light pink
            ((255, 100, 150), (255, 200, 220)), # Medium pink
            ((200, 100, 130), (255, 180, 200)), # Dusty pink
        ]
    },
    "blue": {
        "ranges": [
            ((0, 0, 150), (100, 100, 255)),    # Pure blue
            ((0, 100, 150), (100, 200, 255)),  # Cyan/teal
            ((0, 50, 100), (80, 150, 200)),    # Dark blue
        ]
    },
    "green": {
        "ranges": [
            ((0, 150, 0), (100, 255, 100)),
            ((0, 100, 0), (80, 180, 80)),
            ((100, 200, 100), (200, 255, 200)),
            ((70, 110, 40), (140, 200, 120)),  # NEW
        ]
    },
    "yellow": {
        "ranges": [
            ((200, 200, 0), (255, 255, 100)),  # Pure yellow
            ((180, 180, 0), (255, 255, 150)),  # Gold/mustard
        ]
    },
    "purple": {
        "ranges": [
            ((128, 0, 128), (200, 100, 200)),  # Purple/violet
            ((100, 0, 150), (180, 80, 255)),   # Deep purple
        ]
    },
    "brown": {
        "ranges": [
            ((100, 50, 0), (180, 120, 80)),    # Brown
            ((80, 40, 0), (150, 100, 50)),     # Dark brown
        ]
    },
    "black": {
        "ranges": [
            ((0, 0, 0), (50, 50, 50)),         # True black
        ]
    },
    "white": {
        "ranges": [
            ((200, 200, 200), (255, 255, 255)), # White/very light
        ]
    },
    "gray": {  # Narrowed range - only true neutral grays
        "ranges": [
            ((60, 60, 60), (190, 190, 190)),   # Gray range (tightened to avoid pastels)
        ]
    },
}


def extract_colors(image_path: str, max_colors: int = 3) -> List[str]:
    """
    Extract dominant base colors from an image.
    
    Returns only base colors from the simplified palette.
    Maps similar shades (cyan→blue, etc.) automatically.
    Pink is now a distinct color, not mapped to red.
    
    Args:
        image_path: Path to the image file
        max_colors: Maximum number of colors to return (default 3)
        
    Returns:
        List of base color names, ordered by dominance
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for performance (analyze smaller version)
        # This also helps focus on dominant colors vs. noise
        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        # Get pixel data
        pixels = np.array(img).reshape(-1, 3)
        
        # Map each pixel to its closest base color
        color_counts = _map_pixels_to_base_colors(pixels)

        # Post-process gray: only keep gray when no vibrant color is dominant.
        # Gray should act as a fallback, not co-exist with clearly vibrant colors.
        total_pixels = sum(color_counts.values())
        if total_pixels > 0 and "gray" in color_counts:
            vibrant_colors = {
                "red",
                "blue",
                "green",
                "yellow",
                "orange",  # Included for completeness, even if not a base color
                "purple",
                "pink",
                "brown",
            }
            has_dominant_vibrant = False
            for c in vibrant_colors:
                count = color_counts.get(c, 0)
                # Treat a color as "meaningfully dominant" if it covers at least 20% of pixels
                if count / total_pixels >= 0.2:
                    has_dominant_vibrant = True
                    break

            # If any vibrant color is dominant, drop gray from the palette
            if has_dominant_vibrant:
                color_counts.pop("gray", None)
        
        # Sort by frequency and return top N
        sorted_colors = [color for color, count in color_counts.most_common(max_colors)]
        
        # Always return at least one color
        if not sorted_colors:
            sorted_colors = ["gray"]  # Neutral fallback
        
        return sorted_colors
        
    except Exception as e:
        print(f"Error extracting colors from {image_path}: {e}")
        return ["gray"]  # Safe fallback


def _map_pixels_to_base_colors(pixels: np.ndarray) -> Counter:
    """
    Map each pixel to its closest base color.
    
    Uses RGB distance calculation to determine which base color
    each pixel belongs to. This automatically handles shade mapping
    (maroon→red, cyan→blue, etc.) Pink is now distinct from red.
    
    Returns:
        Counter object with base color frequencies
    """
    color_counts = Counter()
    
    # Sample pixels for large images (performance optimization)
    if len(pixels) > 5000:
        sample_indices = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[sample_indices]
    
    for pixel in pixels:
        r, g, b = pixel
        
        # Skip very transparent or invalid pixels
        if np.isnan(r) or np.isnan(g) or np.isnan(b):
            continue
        
        # Find closest base color
        closest_color = _classify_pixel_color(r, g, b)
        color_counts[closest_color] += 1
    
    return color_counts


def _classify_pixel_color(r: int, g: int, b: int) -> str:
    # Convert to Python ints (prevents overflow everywhere)
    r, g, b = int(r), int(g), int(b)

    # 1️⃣ Exact range matches FIRST (authoritative)
    for color_name, definition in COLOR_DEFINITIONS.items():
        for min_rgb, max_rgb in definition["ranges"]:
            if (_in_range(r, min_rgb[0], max_rgb[0]) and
                _in_range(g, min_rgb[1], max_rgb[1]) and
                _in_range(b, min_rgb[2], max_rgb[2])):
                return color_name

    # 2️⃣ Strict grayscale detection (neutral only)
    color_variance = max(r, g, b) - min(r, g, b)
    if color_variance < 15:
        avg_brightness = (r + g + b) // 3
        if avg_brightness < 60:
            return "black"
        elif avg_brightness > 200:
            return "white"
        else:
            return "gray"

    # 3️⃣ PINK (must come before red)
    if r > 180 and r > g and r > b:
        if g > 110 and b > 110 and (r - g) < 90 and (r - b) < 90:
            return "pink"

    # 4️⃣ RED (true red only)
    if r > 140 and r > g + 40 and r > b + 40:
        return "red"

    # 5️⃣ PURPLE
    if r > 90 and b > 90 and g < min(r, b) - 30:
        return "purple"

    # 6️⃣ YELLOW
    if r > 170 and g > 170 and b < 120:
        return "yellow"

    # 7️⃣ GREEN (expanded but controlled)
    if g > 100 and g > r + 15 and g > b + 15:
        return "green"

    # 8️⃣ BLUE
    if b > 100 and b > r + 15 and b > g + 15:
        return "blue"

    # 9️⃣ BROWN
    if 80 < r < 180 and 50 < g < 130 and b < g:
        return "brown"

    return "gray"



def _in_range(value: int, min_val: int, max_val: int) -> bool:
    """Check if value is within range (inclusive)."""
    return min_val <= value <= max_val


def get_color_variations(base_color: str) -> List[str]:
    """
    Get all color variations that map to a base color.
    
    This is used for search: when user searches "red",
    we want to match images tagged with red (including maroon, crimson, etc.)
    Pink is now a separate color and does not map to red.
    
    Args:
        base_color: Base color name (e.g., "red")
        
    Returns:
        List containing the base color (for backward compatibility)
    """
    # Since we now only store base colors, just return the input
    # This function exists for backward compatibility with existing code
    if base_color in BASE_COLORS:
        return [base_color]
    
    # Handle legacy color names that might still exist in old data
    legacy_mapping = {
        "maroon": "red",
        "crimson": "red",
        "cyan": "blue",
        "teal": "blue",
        "navy": "blue",
        "lime": "green",
        "olive": "green",
        "gold": "yellow",
        "orange": "yellow",  # Orange is close to yellow
        "tan": "brown",
        "beige": "brown",
        "violet": "purple",
        "magenta": "purple",
        # Pink is now a base color, not legacy
    }
    
    return [legacy_mapping.get(base_color.lower(), "gray")]