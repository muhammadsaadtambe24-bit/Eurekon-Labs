"""
OCR Module - Extract text from images using EasyOCR
Handles screenshots, documents, bills, and any text-containing images
Optimized for web app usage with caching and batch processing.

Singleton: EasyOCR reader is created exactly once per process and reused by all
OCR calls (including background threads). A lock ensures no double initialization.
"""

import easyocr
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level singleton: one EasyOCR reader per process
_reader: Optional[easyocr.Reader] = None
# Lock so only one thread can perform initialization (prevents multiple Reader() calls)
_reader_lock = threading.Lock()


def get_reader() -> easyocr.Reader:
    """
    Return the single EasyOCR reader instance, creating it once on first use.
    Thread-safe: easyocr.Reader() is called only once per process even under
    concurrent uploads or background OCR threads.
    """
    global _reader
    if _reader is not None:
        return _reader
    with _reader_lock:
        # Double-check: another thread may have initialized while we waited
        if _reader is None:
            logger.info("Initializing EasyOCR reader...")
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized successfully")
    return _reader


def extract_text(image_path: str, preserve_case: bool = False) -> str:
    """
    Extract all readable text from an image using EasyOCR.

    Skips OCR when the image has zero text likelihood (not document and not
    text-heavy by vision heuristic), avoiding unnecessary reader load and CPU.

    Args:
        image_path: Path to the image file
        preserve_case: If True, preserve original case; if False, convert to lowercase

    Returns:
        Extracted text as a string (normalized whitespace)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image file is invalid or corrupted
    """
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        reader = get_reader()
        results = reader.readtext(image_path)

        # Extract text from results
        text = " ".join([res[1] for res in results])

        # Normalize whitespace
        text = " ".join(text.split())

        # Convert to lowercase if requested
        if not preserve_case:
            text = text.lower()

        logger.info(f"Successfully extracted {len(text)} characters from {image_path}")
        return text

    except Exception as e:
        logger.error(f"OCR error for {image_path}: {e}")
        raise ValueError(f"Failed to process image: {e}")


def extract_text_with_positions(image_path: str) -> List[Dict]:
    """
    Extract text along with bounding box positions and confidence scores.
    Useful for advanced search features and text highlighting.

    Args:
        image_path: Path to the image file

    Returns:
        List of dictionaries containing:
            - text: Extracted text
            - bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - confidence: Confidence score (0-1)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image file is invalid or corrupted
    """
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        reader = get_reader()
        results = reader.readtext(image_path)

        parsed_results = []
        for bbox, text, confidence in results:
            parsed_results.append({
                'text': text,
                'bbox': bbox,
                'confidence': confidence
            })

        logger.info(f"Extracted {len(parsed_results)} text regions from {image_path}")
        return parsed_results

    except Exception as e:
        logger.error(f"OCR position extraction error for {image_path}: {e}")
        raise ValueError(f"Failed to extract positioned text: {e}")


def get_text_confidence(image_path: str) -> float:
    """
    Get average confidence score for OCR results.

    Args:
        image_path: Path to the image file

    Returns:
        Average confidence score (0-100)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image file is invalid or corrupted
    """
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        reader = get_reader()
        results = reader.readtext(image_path)

        if not results:
            logger.warning(f"No text detected in {image_path}")
            return 0.0

        confidences = [res[2] * 100 for res in results]  # EasyOCR gives 0-1
        avg_confidence = sum(confidences) / len(confidences)

        logger.info(f"Average OCR confidence for {image_path}: {avg_confidence:.2f}%")
        return round(avg_confidence, 2)

    except Exception as e:
        logger.error(f"Confidence check error for {image_path}: {e}")
        raise ValueError(f"Failed to calculate confidence: {e}")


def batch_extract_text(image_paths: List[str], preserve_case: bool = False) -> Dict[str, str]:
    """
    Extract text from multiple images in batch for better performance.

    Args:
        image_paths: List of image file paths
        preserve_case: If True, preserve original case; if False, convert to lowercase

    Returns:
        Dictionary mapping image paths to extracted text
    """
    results = {}
    reader = get_reader()  # Initialize once for all images

    for image_path in image_paths:
        try:
            if not Path(image_path).exists():
                logger.warning(f"Skipping non-existent file: {image_path}")
                results[image_path] = ""
                continue

            ocr_results = reader.readtext(image_path)
            text = " ".join([res[1] for res in ocr_results])
            text = " ".join(text.split())

            if not preserve_case:
                text = text.lower()

            results[image_path] = text

        except Exception as e:
            logger.error(f"Batch OCR error for {image_path}: {e}")
            results[image_path] = ""

    logger.info(f"Batch processed {len(image_paths)} images")
    return results


def contains_text(image_path: str, min_confidence: float = 0.5) -> bool:
    """
    Quick check if an image contains readable text above a confidence threshold.
    Useful for filtering images before full OCR processing.

    Args:
        image_path: Path to the image file
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        True if text is detected with sufficient confidence
    """
    if not Path(image_path).exists():
        return False

    try:
        reader = get_reader()
        results = reader.readtext(image_path)

        # Check if any text meets the confidence threshold
        for _, _, confidence in results:
            if confidence >= min_confidence:
                return True

        return False

    except Exception as e:
        logger.error(f"Text detection error for {image_path}: {e}")
        return False


def search_text_in_image(image_path: str, search_term: str, case_sensitive: bool = False) -> bool:
    """
    Search for specific text within an image.

    Args:
        image_path: Path to the image file
        search_term: Text to search for
        case_sensitive: Whether the search should be case-sensitive

    Returns:
        True if search term is found in the image
    """
    try:
        extracted_text = extract_text(image_path, preserve_case=case_sensitive)

        if not case_sensitive:
            search_term = search_term.lower()

        return search_term in extracted_text

    except Exception as e:
        logger.error(f"Text search error for {image_path}: {e}")
        return False
