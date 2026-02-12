"""
Image processing orchestrator - coordinates all AI modules.
Optimizes metadata storage and retrieval for performance.
"""

import os
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

# Import AI modules
from . import vision
from . import color
from . import objects
from . import scoring
from . import ocr


# -------------------------------------------------------------------
# OCR keyword gate
# OCR will run ONLY if one of these keywords is present
# -------------------------------------------------------------------

OCR_TRIGGER_KEYWORDS = {
    "text-heavy",
    "document",
    "invoice",
    "receipt",
    "bill",
    "id",
    "id_card",
    "card",
    "paper",
    "form",
    "statement",
    "letter"
}


def should_run_ocr(keywords: list[str]) -> bool:
    """
    Decide whether OCR should run based on semantic keywords only.
    No heuristics, no image analysis, no guessing.
    """
    if not keywords:
        return False

    keyword_set = {k.lower() for k in keywords}
    return not OCR_TRIGGER_KEYWORDS.isdisjoint(keyword_set)


class ImageProcessor:
    """
    Central processor for image analysis and metadata management.
    
    Handles:
    - Processing images with all AI modules
    - Loading/saving metadata efficiently
    - Caching to avoid redundant processing
    """
    
    def __init__(self, metadata_path: str = "metadata.json"):
        """
        Initialize processor.
        
        Args:
            metadata_path: Path to JSON file storing image metadata
        """
        self.metadata_path = metadata_path
        self._metadata_cache = None  # Lazy load
        self._cache_dirty = False    # Track if we need to save
    
    def process_image(
        self, 
        image_path: str, 
        ocr_text: str = "",
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through all AI modules.
        
        Extracts:
        - Colors (base colors only)
        - Image type tags (photo, screenshot, etc.)
        - Object hints (text-heavy, people, nature, animals)
        
        Args:
            image_path: Path to image file
            ocr_text: Pre-extracted OCR text (optional)
            force_reprocess: If True, ignore existing metadata
            
        Returns:
            Metadata dict with all extracted features
        """
        filename = os.path.basename(image_path)
        
        # Check if already processed (unless forced)
        if not force_reprocess:
            existing = self._get_cached_metadata(filename)
            if existing:
                return existing
        
        # Process image through all modules
        metadata = {
            'filename': filename,
            'path': image_path,
            'timestamp': time.time(),
        }
        
        # Extract colors (base colors only, max 3)
        try:
            metadata['colors'] = color.extract_colors(image_path, max_colors=3)
        except Exception as e:
            print(f"Color extraction failed for {filename}: {e}")
            metadata['colors'] = ["gray"]
        
        # Classify image type (multiple tags possible)
        try:
            metadata['tags'] = vision.classify_image(image_path)
        except Exception as e:
            print(f"Image classification failed for {filename}: {e}")
            metadata['tags'] = ["photo"]
        
        # Detect objects/content
        try:
            metadata['objects'] = objects.detect_objects(image_path, ocr_text)
        except Exception as e:
            print(f"Object detection failed for {filename}: {e}")
            metadata['objects'] = []
        
        # Store OCR text if provided
        metadata['ocr_text'] = ocr_text
        
        # Update cache
        self._update_cached_metadata(filename, metadata)
        
        return metadata
    
    def process_batch(
        self, 
        image_paths: List[str],
        ocr_texts: Optional[Dict[str, str]] = None,
        force_reprocess: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images efficiently.
        
        Loads metadata once, processes all images, saves once.
        Much faster than processing images individually.
        
        Args:
            image_paths: List of image file paths
            ocr_texts: Optional dict mapping filename -> OCR text
            force_reprocess: If True, ignore existing metadata
            
        Returns:
            List of metadata dicts
        """
        if ocr_texts is None:
            ocr_texts = {}
        
        results = []
        
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            ocr_text = ocr_texts.get(filename, "")
            
            metadata = self.process_image(
                image_path, 
                ocr_text, 
                force_reprocess
            )
            results.append(metadata)
        
        # Save all changes at once (optimization)
        self._save_metadata()
        
        return results
    
    def search(
        self,
        query: str = "",
        color_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search images using the scoring system.
        
        Args:
            query: Search keywords (space-separated)
            color_filter: Filter by specific color
            type_filter: Filter by image type
            max_results: Maximum results to return
            
        Returns:
            List of matching image metadata, ranked by relevance
        """
        # Load all metadata
        all_images = self._load_all_metadata()
        
        # Parse query into terms
        query_terms = query.strip().split() if query else []
        
        # Use scoring module to rank results
        results = scoring.rank_search_results(
            all_images,
            query_terms,
            color_filter,
            type_filter,
            max_results
        )
        
        return results
    
    def get_image_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific image.
        
        Args:
            filename: Image filename
            
        Returns:
            Metadata dict or None if not found
        """
        return self._get_cached_metadata(filename)
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all processed images.
        
        Returns:
            List of all image metadata dicts
        """
        return self._load_all_metadata()
    
    def update_metadata(self, filename: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields in an image's metadata.
        
        Args:
            filename: Image filename
            updates: Dict of fields to update
            
        Returns:
            True if successful, False if image not found
        """
        metadata = self._get_cached_metadata(filename)
        if not metadata:
            return False
        
        metadata.update(updates)
        self._update_cached_metadata(filename, metadata)
        self._save_metadata()
        
        return True
    
    def delete_metadata(self, filename: str) -> bool:
        """
        Remove metadata for an image (e.g., after deletion).
        
        Args:
            filename: Image filename
            
        Returns:
            True if removed, False if not found
        """
        self._ensure_cache_loaded()
        
        if filename in self._metadata_cache:
            del self._metadata_cache[filename]
            self._cache_dirty = True
            self._save_metadata()
            return True
        
        return False
    
    # --- Internal cache management (optimization) ---
    
    def _ensure_cache_loaded(self):
        """Lazy load metadata cache from disk."""
        if self._metadata_cache is None:
            self._metadata_cache = {}
            
            if os.path.exists(self.metadata_path):
                try:
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Handle both dict and list formats
                    if isinstance(data, dict):
                        self._metadata_cache = data
                    elif isinstance(data, list):
                        # Convert list to dict for faster lookups
                        self._metadata_cache = {
                            item['filename']: item 
                            for item in data
                        }
                except Exception as e:
                    print(f"Error loading metadata: {e}")
                    self._metadata_cache = {}
    
    def _get_cached_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get metadata from cache (loads cache if needed)."""
        self._ensure_cache_loaded()
        return self._metadata_cache.get(filename)
    
    def _update_cached_metadata(self, filename: str, metadata: Dict[str, Any]):
        """Update metadata in cache."""
        self._ensure_cache_loaded()
        self._metadata_cache[filename] = metadata
        self._cache_dirty = True
    
    def _load_all_metadata(self) -> List[Dict[str, Any]]:
        """Load all metadata as a list."""
        self._ensure_cache_loaded()
        return list(self._metadata_cache.values())
    
    def _save_metadata(self):
        """Save metadata cache to disk (only if changed)."""
        if not self._cache_dirty:
            return  # No changes, skip save
        
        self._ensure_cache_loaded()
        
        try:
            # Create backup of existing file
            if os.path.exists(self.metadata_path):
                backup_path = f"{self.metadata_path}.backup"
                import shutil
                shutil.copy2(self.metadata_path, backup_path)
            
            # Save as dict for faster future loads
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self._metadata_cache, f, indent=2, ensure_ascii=False)
            
            self._cache_dirty = False
            
        except Exception as e:
            print(f"Error saving metadata: {e}")
            # Restore backup if save failed
            if os.path.exists(f"{self.metadata_path}.backup"):
                import shutil
                shutil.copy2(f"{self.metadata_path}.backup", self.metadata_path)
    
    def optimize_metadata_storage(self):
        """
        Optimize metadata storage by cleaning up and reorganizing.
        
        Call this periodically to:
        - Remove orphaned metadata (files no longer exist)
        - Compact the JSON file
        - Rebuild indexes
        """
        self._ensure_cache_loaded()
        
        # Remove metadata for files that no longer exist
        to_remove = []
        for filename, metadata in self._metadata_cache.items():
            image_path = metadata.get('path', '')
            if image_path and not os.path.exists(image_path):
                to_remove.append(filename)
        
        for filename in to_remove:
            del self._metadata_cache[filename]
            print(f"Removed orphaned metadata for: {filename}")
        
        if to_remove:
            self._cache_dirty = True
            self._save_metadata()
            print(f"Cleaned up {len(to_remove)} orphaned entries")


# Global processor instance (singleton pattern)
_processor_instance = None


def get_processor(metadata_path: str = "metadata.json") -> ImageProcessor:
    """
    Get the global ImageProcessor instance.
    
    Uses singleton pattern to ensure only one instance exists,
    which improves performance by maintaining a single metadata cache.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        ImageProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = ImageProcessor(metadata_path)
    
    return _processor_instance


# ============================================================================
# Flask app.py Compatibility Layer
# ============================================================================
# These functions provide a simple interface compatible with the existing
# Flask app.py, which expects standalone functions rather than the class-based
# ImageProcessor approach.
#
# Execution order for background OCR:
# 1. Foreground (sync): process_image_foreground_only() runs vision, color,
#    object detection, metadata. Saves with ocr_text="", ocr_status="pending".
# 2. Response returned immediately; images render with non-OCR metadata.
# 3. Background (thread): run_ocr_background() runs OCR + OCR-derived keywords,
#    then updates metadata with ocr_text, ocr_keywords, ocr_status="done".


def process_image_foreground_only(filepath: str, filename: str = None) -> Dict[str, Any]:
    """
    Run only non-OCR analysis so upload can return immediately.
    Used for background-OCR pipeline: vision, color, object/visual keywords.
    OCR and OCR-derived keywords run later in run_ocr_background().
    """
    filepath = os.path.abspath(os.path.normpath(filepath))
    colors_list = ["gray"]
    image_type = "photo"
    keywords = []
    img_metadata = {}

    try:
        try:
            colors_list = color.extract_colors(filepath, max_colors=3)
        except Exception as e:
            print(f"Color extraction failed for {filepath}: {e}")

        tags = []
        try:
            tags = vision.classify_image(filepath)
        except Exception as e:
            print(f"Vision classification failed for {filepath}: {e}")
        image_type = tags[0] if tags else "photo"

        objects_list = []
        try:
            objects_list = objects.detect_objects(filepath, "")
        except Exception as e:
            print(f"Object detection failed for {filepath}: {e}")

        try:
            img_metadata = get_image_metadata(filepath)
        except Exception as e:
            print(f"Metadata failed for {filepath}: {e}")

        # Visual keywords only (no OCR, no detect_content_keywords)
        keywords = list(set((tags[1:] if len(tags) > 1 else []) + objects_list))

        return {
            "ocr_text": "",
            "ocr_status": "pending",
            "ocr_keywords": [],
            "colors": colors_list,
            "image_type": image_type,
            "keywords": keywords,
            "metadata": img_metadata,
        }
    except Exception as e:
        print(f"Error in foreground processing for {filepath}: {e}")
        return {
            "ocr_text": "",
            "ocr_status": "pending",
            "ocr_keywords": [],
            "colors": colors_list,
            "image_type": image_type,
            "keywords": keywords,
            "metadata": img_metadata or {},
        }


# -------- Single-worker OCR queue (no concurrent OCR) --------
# One thread processes jobs sequentially so OCR never runs in parallel.
# metadata_lock: shared with app to prevent race when upload saves while OCR worker saves
metadata_lock = threading.Lock()
_ocr_queue: queue.Queue = queue.Queue()
_ocr_worker_started = False
_ocr_worker_lock = threading.Lock()


def _ocr_worker() -> None:
    """Single background worker: process one OCR job at a time to avoid overload."""
    while True:
        try:
            job = _ocr_queue.get()
            if job is None:
                break
            filepath, image_id, metadata_file_path = job
            _process_one_ocr_job(filepath, image_id, metadata_file_path)
        except Exception as e:
            print(f"OCR worker error: {e}")
        finally:
            try:
                _ocr_queue.task_done()
            except Exception:
                pass


def _process_one_ocr_job(filepath: str, image_id: str, metadata_file_path: str) -> None:
    """
    Process a single OCR job: skip if non-text-heavy, else run OCR and set
    ocr_status to done/failed only after real execution; never mark done if skipped.
    Uses metadata_lock to avoid overwriting concurrent uploads; reloads before save.
    """
    filepath = os.path.abspath(os.path.normpath(filepath))
    metadata_file_path = os.path.abspath(metadata_file_path)

    try:
        with metadata_lock:
            with open(metadata_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as e:
        print(f"OCR job: could not load metadata for {image_id}: {e}")
        return

    images = data.get("images", [])
    record = None
    for img in images:
        if img.get("id") == image_id:
            record = img
            break
    if not record:
        print(f"OCR job: image {image_id} not found in metadata")
        return

    # Same keyword format as process_image: vision tags[1:] + objects (for is_text_heavy)
    image_type = (record.get("image_type") or "photo").lower()
    keywords = list(record.get("keywords") or [])
    if not should_run_ocr(keywords):
        print(f"[OCR] Skipped for {image_id} | keywords={keywords}")
        record["ocr_status"] = "skipped"
        record["ocr_text"] = ""
        record["ocr_keywords"] = []
        try:
            with open(metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"OCR job: could not save metadata (skipped) for {image_id}: {e}")
        return

    record["ocr_status"] = "running"
    try:
        print(f"[OCR] Running for {image_id} | keywords={keywords}")
        with metadata_lock:
            with open(metadata_file_path, "r", encoding="utf-8") as f:
                fresh = json.load(f)
            for img in fresh.get("images", []):
                if img.get("id") == image_id:
                    img["ocr_status"] = "running"
                    break
            with open(metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(fresh, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"OCR job: could not save metadata (running) for {image_id}: {e}")

    try:
        ocr_text = extract_text(filepath)
        ocr_keywords = detect_content_keywords(filepath, ocr_text)
        # Re-run object detection with OCR text (aligns with full process_image)
        objects_list = objects.detect_objects(filepath, ocr_text)
        existing_kw = set(record.get("keywords") or [])
        merged_keywords = list(existing_kw | set(objects_list))
        record["ocr_status"] = "done"
        record["ocr_text"] = ocr_text
        record["ocr_keywords"] = ocr_keywords
        record["keywords"] = merged_keywords
        print(f"[OCR] Done for {image_id} | chars={len(ocr_text)}")

    except Exception as e:
        print(f"Background OCR failed for {filepath}: {e}")
        record["ocr_status"] = "failed"
        record["ocr_text"] = ""
        record["ocr_keywords"] = []

    # Reload before save so we don't overwrite images added by concurrent uploads
    try:
        with metadata_lock:
            with open(metadata_file_path, "r", encoding="utf-8") as f:
                fresh = json.load(f)
            for img in fresh.get("images", []):
                if img.get("id") == image_id:
                    img["ocr_status"] = record["ocr_status"]
                    img["ocr_text"] = record["ocr_text"]
                    img["ocr_keywords"] = record["ocr_keywords"]
                    if "keywords" in record:
                        img["keywords"] = record["keywords"]
                    break
            with open(metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(fresh, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"OCR job: could not save metadata (final) for {image_id}: {e}")


def _ensure_ocr_worker_started() -> None:
    global _ocr_worker_started
    with _ocr_worker_lock:
        if not _ocr_worker_started:
            t = threading.Thread(target=_ocr_worker, daemon=True)
            t.start()
            _ocr_worker_started = True


def run_ocr_background(filepath: str, image_id: str, metadata_file_path: str) -> None:
    """
    Enqueue one OCR job. A single worker processes jobs one at a time.
    Upload returns immediately; status is updated only after real execution
    (done/failed/skipped) so ocr_status never reflects a fake completion.
    """
    _ensure_ocr_worker_started()
    _ocr_queue.put((filepath, image_id, os.path.abspath(metadata_file_path)))


def process_image(filepath: str, filename: str = None) -> Dict[str, Any]:
    """
    Flask-compatible wrapper for image processing.
    
    This function matches the signature expected by app.py and returns
    a dictionary with the exact keys the Flask app expects:
    - ocr_text: str
    - colors: list[str]
    - image_type: str (single primary type)
    - keywords: list[str]
    - metadata: dict
    
    Args:
        filepath: Path to the image file (relative or absolute; normalized to absolute)
        filename: Original filename (optional, for display purposes)
        
    Returns:
        Dictionary with OCR text, colors, type, keywords, and metadata
    """
    # Normalize to absolute path so OCR and all modules resolve the same file (avoids cwd/path mismatch)
    filepath = os.path.abspath(os.path.normpath(filepath))

    ocr_text = ""
    colors_list = ["gray"]
    image_type = "photo"
    keywords = []
    img_metadata = {}

    try:
        # --- Colors ---
        try:
            colors_list = color.extract_colors(filepath, max_colors=3)
        except Exception as e:
            print(f"Color extraction failed for {filepath}: {e}")

        # --- Vision / image type (run before OCR to decide if OCR is needed) ---
        tags = []
        try:
            tags = vision.classify_image(filepath)
        except Exception as e:
            print(f"Vision classification failed for {filepath}: {e}")
        image_type = tags[0] if tags else "photo"

        # --- Object detection (with empty OCR first, for is_text_heavy heuristic) ---
        objects_list = []
        try:
            objects_list = objects.detect_objects(filepath, "")
        except Exception as e:
            print(f"Object detection failed for {filepath}: {e}")

        # --- OCR: only run for text-heavy images (saves time on normal photos) ---
        # is_text_heavy uses image_type + tags + objects to decide if OCR is worthwhile.
        # Skipping OCR for photos/selfies improves processing speed significantly.
        pre_keywords = list(set((tags[1:] if len(tags) > 1 else []) + objects_list))
        if vision.is_text_heavy(image_type, pre_keywords):
            try:
                ocr_text = extract_text(filepath)
                # Re-run object detection with OCR text for better "text-heavy" hint
                objects_list = objects.detect_objects(filepath, ocr_text)
            except Exception as e:
                print(f"OCR failed for {filepath}: {e}")
        else:
            ocr_text = ""

        # --- Image metadata ---
        try:
            img_metadata = get_image_metadata(filepath)
        except Exception as e:
            print(f"Metadata failed for {filepath}: {e}")

        # --- Keywords: tags + objects + OCR-derived ---
        keywords = list(set((tags[1:] if len(tags) > 1 else []) + objects_list))
        try:
            content_keywords = detect_content_keywords(filepath, ocr_text)
            keywords.extend(content_keywords)
        except Exception as e:
            print(f"Content keywords failed for {filepath}: {e}")
        keywords = list(set(keywords))

        return {
            "ocr_text": ocr_text,
            "colors": colors_list,
            "image_type": image_type,
            "keywords": keywords,
            "metadata": img_metadata,
        }
    except Exception as e:
        print(f"Error processing image {filepath}: {e}")
        return {
            "ocr_text": ocr_text,
            "colors": colors_list,
            "image_type": image_type,
            "keywords": keywords,
            "metadata": img_metadata or {},
        }


def extract_text(filepath: str) -> str:
    """
    Extract text from image using EasyOCR (ai/ocr.py).

    Args:
        filepath: Path to image file (relative or absolute; normalized inside)

    Returns:
        Extracted text string, or empty string on failure
    """
    try:
        path = os.path.abspath(os.path.normpath(filepath))
        if not os.path.isfile(path):
            print(f"OCR skipped: not a file or missing: {path}")
            return ""
        return ocr.extract_text(path, preserve_case=False)
    except Exception as e:
        print(f"OCR error for {filepath!r}: {e}")
        return ""


def get_image_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract basic image metadata (dimensions, format, size).
    
    Args:
        filepath: Path to image file
        
    Returns:
        Dictionary with image metadata
    """
    try:
        img = Image.open(filepath)
        file_size = os.path.getsize(filepath)
        
        return {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode,
            'size_bytes': file_size,
            'size_kb': round(file_size / 1024, 2)
        }
    except Exception as e:
        print(f"Error getting metadata for {filepath}: {e}")
        return {}


def detect_content_keywords(filepath: str, ocr_text: str) -> List[str]:
    """
    Detect content-based keywords from OCR text and image analysis.
    
    Extracts meaningful keywords like:
    - Document types (invoice, receipt, bill)
    - Payment related (UPI, payment, transaction)
    - ID cards (student ID, identity card)
    - Common entities
    
    Args:
        filepath: Path to image file
        ocr_text: Extracted OCR text
        
    Returns:
        List of detected keywords
    """
    keywords = []
    
    if not ocr_text:
        return keywords
    
    ocr_lower = ocr_text.lower()
    
    # Payment-related keywords
    payment_terms = [
        ('upi', ['upi', 'unified payment', 'bhim']),
        ('payment', ['paid', 'payment', 'transaction']),
        ('gpay', ['google pay', 'gpay', 'g pay']),
        ('phonepe', ['phonepe', 'phone pe']),
        ('paytm', ['paytm', 'pay tm']),
    ]
    
    for keyword, patterns in payment_terms:
        if any(pattern in ocr_lower for pattern in patterns):
            keywords.append(keyword)
    
    # Document types
    doc_terms = [
        ('invoice', ['invoice', 'bill to']),
        ('receipt', ['receipt', 'received with thanks']),
        ('bill', ['bill', 'amount due', 'total amount']),
    ]
    
    for keyword, patterns in doc_terms:
        if any(pattern in ocr_lower for pattern in patterns):
            keywords.append(keyword)
    
    # ID cards
    id_terms = [
        ('id_card', ['id card', 'identity card', 'student id', 'employee id']),
        ('student', ['student', 'enrollment', 'roll no']),
    ]
    
    for keyword, patterns in id_terms:
        if any(pattern in ocr_lower for pattern in patterns):
            keywords.append(keyword)
    
    # Financial
    if any(term in ocr_lower for term in ['₹', 'rs.', 'inr', 'amount']):
        keywords.append('financial')
    
    # Remove duplicates
    keywords = list(set(keywords))
    
    return keywords

