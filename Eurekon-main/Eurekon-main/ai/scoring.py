"""
Search scoring system using point-based ranking.
Assigns points for different types of matches and ranks results.
"""

from typing import List, Dict, Any, Tuple


# SCORING WEIGHTS - Higher = more important
# These control how much each type of match contributes to final score
WEIGHTS = {
    # Exact matches are worth more than partial
    "color_exact": 10,           # Image has the exact color searched
    "type_exact": 15,            # Image type exactly matches search
    "ocr_exact_word": 20,        # OCR contains exact word match
    "ocr_partial_word": 5,       # OCR contains partial word match
    "object_match": 12,          # Object detection matches query
    
    # Filename matches (less reliable but still useful)
    "filename_exact": 8,         # Filename contains exact query
    "filename_partial": 3,       # Filename partially matches
    
    # Bonus points
    "multi_color_match": 5,      # Multiple colors match
    "fresh_image_bonus": 2,      # Recently added/modified
}


def calculate_search_score(
    image_metadata: Dict[str, Any],
    query_terms: List[str],
    color_filter: str = None,
    type_filter: str = None
) -> Tuple[int, Dict[str, int]]:
    """
    Calculate search relevance score for an image.
    
    Returns both the total score and a breakdown of points by category
    for transparency/debugging.
    
    Args:
        image_metadata: Image metadata dict with colors, tags, ocr_text, etc.
        query_terms: List of search terms (keywords)
        color_filter: Optional color to filter by
        type_filter: Optional image type to filter by
        
    Returns:
        Tuple of (total_score, score_breakdown)
        - total_score: Integer sum of all points
        - score_breakdown: Dict showing points from each match type
    """
    score = 0
    breakdown = {}  # For debugging/transparency
    
    # --- COLOR MATCHING ---
    if color_filter:
        image_colors = image_metadata.get('colors', [])
        
        if color_filter in image_colors:
            score += WEIGHTS["color_exact"]
            breakdown['color_exact'] = WEIGHTS["color_exact"]
            
            # Bonus if multiple colors match (unlikely but possible)
            color_matches = sum(1 for c in image_colors if c == color_filter)
            if color_matches > 1:
                bonus = WEIGHTS["multi_color_match"]
                score += bonus
                breakdown['multi_color_bonus'] = bonus
        else:
            # No match = score 0 for this image (filter requirement)
            return 0, {"filtered_out": "color_mismatch"}
    
    # --- TYPE MATCHING ---
    if type_filter:
        image_tags = image_metadata.get('tags', [])
        
        if type_filter in image_tags:
            score += WEIGHTS["type_exact"]
            breakdown['type_exact'] = WEIGHTS["type_exact"]
        else:
            # No match = score 0 for this image (filter requirement)
            return 0, {"filtered_out": "type_mismatch"}
    
    # --- KEYWORD MATCHING (OCR text) ---
    ocr_text = image_metadata.get('ocr_text', '').lower()
    
    if query_terms and ocr_text:
        ocr_words = set(ocr_text.split())
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Exact word match in OCR
            if term_lower in ocr_words:
                points = WEIGHTS["ocr_exact_word"]
                score += points
                breakdown[f'ocr_exact:{term}'] = points
            
            # Partial match (term is substring of OCR text)
            elif term_lower in ocr_text:
                points = WEIGHTS["ocr_partial_word"]
                score += points
                breakdown[f'ocr_partial:{term}'] = points
    
    # --- OBJECT MATCHING ---
    # Match query terms against detected objects
    objects = image_metadata.get('objects', [])
    
    if query_terms and objects:
        for term in query_terms:
            term_lower = term.lower()
            
            # Check if term matches any object
            for obj in objects:
                if term_lower in obj.lower() or obj.lower() in term_lower:
                    points = WEIGHTS["object_match"]
                    score += points
                    breakdown[f'object:{obj}'] = points
                    break  # Only count once per term
    
    # --- FILENAME MATCHING ---
    filename = image_metadata.get('filename', '').lower()
    
    if query_terms and filename:
        # Remove extension for matching
        filename_no_ext = filename.rsplit('.', 1)[0]
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Exact match (term is entire filename or word in filename)
            filename_words = filename_no_ext.replace('_', ' ').replace('-', ' ').split()
            if term_lower in filename_words:
                points = WEIGHTS["filename_exact"]
                score += points
                breakdown[f'filename_exact:{term}'] = points
            
            # Partial match (term is substring)
            elif term_lower in filename_no_ext:
                points = WEIGHTS["filename_partial"]
                score += points
                breakdown[f'filename_partial:{term}'] = points
    
    # --- FRESHNESS BONUS ---
    # Slightly prefer recently added images (optional)
    # This helps when many images have similar scores
    timestamp = image_metadata.get('timestamp', 0)
    if timestamp:
        # Images from last 7 days get small bonus
        import time
        current_time = time.time()
        age_days = (current_time - timestamp) / (24 * 3600)
        
        if age_days < 7:
            bonus = WEIGHTS["fresh_image_bonus"]
            score += bonus
            breakdown['fresh_bonus'] = bonus
    
    return score, breakdown


def rank_search_results(
    all_images: List[Dict[str, Any]],
    query_terms: List[str],
    color_filter: str = None,
    type_filter: str = None,
    max_results: int = 50
) -> List[Dict[str, Any]]:
    """
    Rank all images by search relevance and return top results.
    
    Args:
        all_images: List of all image metadata dicts
        query_terms: Search query split into terms
        color_filter: Optional color filter
        type_filter: Optional type filter
        max_results: Maximum number of results to return
        
    Returns:
        List of image metadata dicts, sorted by relevance (best first)
    """
    scored_images = []
    
    for image in all_images:
        score, breakdown = calculate_search_score(
            image, query_terms, color_filter, type_filter
        )
        
        # Skip images with score 0 (filtered out or no matches)
        if score > 0:
            # Attach score for sorting
            image_with_score = image.copy()
            image_with_score['_search_score'] = score
            image_with_score['_score_breakdown'] = breakdown
            scored_images.append(image_with_score)
    
    # Sort by score (highest first)
    scored_images.sort(key=lambda x: x['_search_score'], reverse=True)
    
    # Return top N results
    return scored_images[:max_results]


def explain_score(image_metadata: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation of why an image scored well.
    
    Useful for debugging and showing users why results were ranked.
    
    Args:
        image_metadata: Image metadata with '_score_breakdown' attached
        
    Returns:
        Human-readable explanation string
    """
    if '_score_breakdown' not in image_metadata:
        return "No scoring information available"
    
    breakdown = image_metadata['_score_breakdown']
    total_score = image_metadata.get('_search_score', 0)
    
    if not breakdown:
        return f"Score: {total_score} (no matches)"
    
    # Build explanation
    parts = [f"Total Score: {total_score}"]
    
    # Group by match type
    color_points = sum(v for k, v in breakdown.items() if 'color' in k)
    type_points = sum(v for k, v in breakdown.items() if 'type' in k)
    ocr_points = sum(v for k, v in breakdown.items() if 'ocr' in k)
    object_points = sum(v for k, v in breakdown.items() if 'object' in k)
    filename_points = sum(v for k, v in breakdown.items() if 'filename' in k)
    bonus_points = sum(v for k, v in breakdown.items() if 'bonus' in k)
    
    if color_points:
        parts.append(f"  - Color match: +{color_points}")
    if type_points:
        parts.append(f"  - Type match: +{type_points}")
    if ocr_points:
        parts.append(f"  - Text match: +{ocr_points}")
    if object_points:
        parts.append(f"  - Object match: +{object_points}")
    if filename_points:
        parts.append(f"  - Filename match: +{filename_points}")
    if bonus_points:
        parts.append(f"  - Bonuses: +{bonus_points}")
    
    return "\n".join(parts)


def adjust_weights(custom_weights: Dict[str, int]) -> None:
    """
    Allow customizing scoring weights at runtime.
    
    This lets you tune the search ranking without code changes.
    
    Args:
        custom_weights: Dict of weight names to new values
    """
    global WEIGHTS
    
    for key, value in custom_weights.items():
        if key in WEIGHTS:
            WEIGHTS[key] = value
        else:
            print(f"Warning: Unknown weight key '{key}' ignored")


def get_current_weights() -> Dict[str, int]:
    """
    Get current scoring weights for inspection/debugging.
    
    Returns:
        Copy of current WEIGHTS dict
    """
    return WEIGHTS.copy()
