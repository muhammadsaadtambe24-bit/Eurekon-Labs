"""
AI module for image search application.

Provides:
- Image classification (photo, screenshot, graphic, etc.)
- Color extraction (simplified base color palette)
- Object/content detection (heuristic-based)
- Search scoring and ranking
- Metadata processing and caching

Usage:
    from ai import get_processor
    
    processor = get_processor()
    metadata = processor.process_image('path/to/image.jpg', ocr_text='...')
    results = processor.search(query='meeting notes', color_filter='blue')
"""

# Main interface - use this in Flask app
from .processor import ImageProcessor, get_processor, process_image

# Individual modules (for advanced use or testing)
from . import vision
from . import color
from . import objects
from . import scoring

__all__ = [
    # Main interface (recommended)
    'get_processor',
    'ImageProcessor',
    'process_image',
    
    # Individual modules (advanced)
    'vision',
    'color',
    'objects',
    'scoring',
]

__version__ = '2.0.0'