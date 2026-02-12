"""
Natural Language Image Search Web App
Flask backend with AI-powered image processing and search
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
# ========== 🔵 CHANGE #1: Import hashlib for duplicate detection 🔵 ==========
import hashlib
# ========== 🔵 END CHANGE #1 🔵 ==========

# Import AI modules (foreground-only for fast response; OCR runs in single-worker queue)
from ai.processor import process_image_foreground_only, run_ocr_background, metadata_lock

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
METADATA_FILE = 'metadata/data.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('metadata', exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ========== 🔵 CHANGE #2: Add function to calculate file hash for duplicate detection 🔵 ==========
def calculate_file_hash(file_content):
    """
    Calculate SHA256 hash of file content for duplicate detection.
    Args:
        file_content: Binary file content
    Returns:
        str: Hexadecimal hash string
    """
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(file_content)
    return hash_sha256.hexdigest()


def find_duplicate_image(file_hash, metadata):
    """
    Check if an image with the same hash already exists.
    Args:
        file_hash: SHA256 hash of the file
        metadata: Current metadata dictionary
    Returns:
        dict or None: Duplicate image record if found, None otherwise
    """
    for img in metadata.get('images', []):
        if img.get('file_hash') == file_hash:
            return img
    return None
# ========== 🔵 END CHANGE #2 🔵 ==========


def load_metadata():
    """Load metadata from JSON file"""
    if not os.path.exists(METADATA_FILE):
        return {'images': []}
    try:
        with metadata_lock:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {'images': []}


def save_metadata(data):
    """Save metadata to JSON file (thread-safe with OCR worker)"""
    with metadata_lock:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_relevance(image_data, query_terms):
    """
    Calculate relevance score for an image based on query terms.
    Higher score = more relevant.
    """
    score = 0
    query_terms = [term.lower() for term in query_terms]

    # Check OCR text (highest weight)
    ocr_text = image_data.get('ocr_text', '').lower()
    for term in query_terms:
        if term in ocr_text:
            score += 10  # Strong match
            # Bonus for exact word match
            if f' {term} ' in f' {ocr_text} ':
                score += 5

    # Check colors (medium weight)
    colors = [c.lower() for c in image_data.get('colors', [])]
    for term in query_terms:
        if term in colors:
            score += 7

    # Check image type
    image_type = image_data.get('image_type', '').lower()
    for term in query_terms:
        if term in image_type:
            score += 5

    # Check keywords (visual + OCR-derived)
    keywords = [k.lower() for k in image_data.get('keywords', [])]
    ocr_kw = [k.lower() for k in image_data.get('ocr_keywords', [])]
    all_keywords = keywords + ocr_kw
    for term in query_terms:
        if term in all_keywords:
            score += 8
        for keyword in all_keywords:
            if term in keyword or keyword in term:
                score += 3

    # Check filename
    filename = image_data.get('original_filename', '').lower()
    for term in query_terms:
        if term in filename:
            score += 4

    return score


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/upload', methods=['POST'])
def upload_images():
    """
    Handle multiple image uploads.
    Processes each image through the AI pipeline.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    # Enforce a hard limit of 50 images per upload request
    if len(files) > 50:
        return jsonify({'error': 'You can upload a maximum of 50 images per upload.'}), 400

    metadata = load_metadata()
    uploaded = []
    errors = []
    # ========== 🔵 CHANGE #3: Add duplicates tracking 🔵 ==========
    duplicates = []
    # ========== 🔵 END CHANGE #3 🔵 ==========
    ocr_jobs = []  # Collect (filepath, image_id) to start AFTER metadata is saved

    for file in files:
        if file and allowed_file(file.filename):
            # ========== 🔵 CHANGE #4: Read file content and calculate hash 🔵 ==========
            original_filename = secure_filename(file.filename)
            
            # Read file content for hash calculation
            file_content = file.read()
            file_hash = calculate_file_hash(file_content)
            
            # Check for duplicate
            duplicate_img = find_duplicate_image(file_hash, metadata)
            if duplicate_img:
                duplicates.append({
                    'filename': original_filename,
                    'duplicate_of': duplicate_img['original_filename'],
                    'reason': 'Identical file already exists'
                })
                continue  # Skip this file
            
            # Reset file pointer after reading
            file.seek(0)
            # ========== 🔵 END CHANGE #4 🔵 ==========

            # Generate unique filename
            ext = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"

            # Use absolute path so OCR and AI pipeline resolve the file regardless of cwd
            filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))

            try:
                # Save file
                file.save(filepath)

                # Foreground only: vision, color, object detection (no OCR). Return immediately.
                analysis = process_image_foreground_only(filepath, original_filename)

                # ========== 🔵 CHANGE #5: Add file_hash to image record 🔵 ==========
                # Create image record with ocr_status="pending"; OCR runs in background
                image_record = {
                    'id': uuid.uuid4().hex,
                    'filename': unique_filename,
                    'original_filename': original_filename,
                    'file_hash': file_hash,  # NEW: Store hash for duplicate detection
                    'uploaded_at': datetime.now().isoformat(),
                    'ocr_text': analysis.get('ocr_text', ''),
                    'ocr_status': analysis.get('ocr_status', 'pending'),
                    'ocr_keywords': analysis.get('ocr_keywords', []),
                    'colors': analysis['colors'],
                    'image_type': analysis['image_type'],
                    'keywords': analysis['keywords'],
                    'metadata': analysis['metadata']
                }
                # ========== 🔵 END CHANGE #5 🔵 ==========

                metadata['images'].append(image_record)
                uploaded.append({
                    'id': image_record['id'],
                    'filename': unique_filename,
                    'original_filename': original_filename
                })
                ocr_jobs.append((filepath, image_record['id']))

            except Exception as e:
                errors.append({'filename': original_filename, 'error': str(e)})
        else:
            errors.append({'filename': file.filename, 'error': 'File type not allowed'})

    # CRITICAL: Save metadata BEFORE starting OCR workers (worker loads from disk)
    save_metadata(metadata)

    # Start background OCR only after metadata is persisted
    metadata_abs = os.path.abspath(METADATA_FILE)
    for filepath, image_id in ocr_jobs:
        run_ocr_background(filepath, image_id, metadata_abs)

    # ========== 🔵 CHANGE #6: Return duplicates info in response 🔵 ==========
    return jsonify({
        'uploaded': uploaded,
        'errors': errors,
        'duplicates': duplicates,  # NEW: List of skipped duplicate files
        'total_images': len(metadata['images'])
    })
    # ========== 🔵 END CHANGE #6 🔵 ==========


@app.route('/api/search', methods=['GET'])
def search_images():
    """
    Search images using natural language query.
    Returns images sorted by relevance.
    """
    query = request.args.get('q', '').strip()

    if not query:
        # Return all images if no query
        metadata = load_metadata()
        images = metadata.get('images', [])
        return jsonify({
            'query': '',
            'results': [
                {
                    'id': img['id'],
                    'filename': img['filename'],
                    'original_filename': img['original_filename'],
                    'image_type': img.get('image_type', 'other'),
                    'colors': img.get('colors', []),
                    'keywords': img.get('keywords', []),
                    'ocr_status': img.get('ocr_status', 'done'),
                    'ocr_keywords': img.get('ocr_keywords', []),
                    'relevance': 0
                }
                for img in images
            ],
            'total': len(images)
        })

    # Split query into search terms
    query_terms = query.lower().split()

    # Also add common synonyms/variations
    expanded_terms = set(query_terms)

    # Expand common terms
    expansions = {
        'upi': ['payment', 'transaction', 'gpay', 'phonepe', 'paytm'],
        'payment': ['upi', 'transaction', 'paid', 'money'],
        'id': ['identity', 'card', 'student', 'college'],
        'card': ['id', 'identity'],
        'bill': ['invoice', 'receipt', 'payment'],
        'receipt': ['bill', 'invoice'],
        'screenshot': ['screen', 'capture'],
        'photo': ['picture', 'image', 'pic'],
        'doc': ['document', 'paper'],
        'document': ['doc', 'paper', 'file'],
    }

    for term in query_terms:
        if term in expansions:
            expanded_terms.update(expansions[term])

    metadata = load_metadata()
    images = metadata.get('images', [])

    # Calculate relevance for each image
    results = []
    for img in images:
        score = calculate_relevance(img, list(expanded_terms))
        if score > 0:  # Only include images with some relevance
            results.append({
                'id': img['id'],
                'filename': img['filename'],
                'original_filename': img['original_filename'],
                'image_type': img.get('image_type', 'other'),
                'colors': img.get('colors', []),
                'keywords': img.get('keywords', []),
                'ocr_status': img.get('ocr_status', 'done'),
                'ocr_keywords': img.get('ocr_keywords', []),
                'relevance': score
            })

    # Sort by relevance (highest first)
    results.sort(key=lambda x: x['relevance'], reverse=True)

    return jsonify({
        'query': query,
        'results': results,
        'total': len(results)
    })


@app.route('/api/images', methods=['GET'])
def get_all_images():
    """Get all uploaded images"""
    metadata = load_metadata()
    images = metadata.get('images', [])

    return jsonify({
        'images': [
            {
                'id': img['id'],
                'filename': img['filename'],
                'original_filename': img['original_filename'],
                'image_type': img.get('image_type', 'other'),
                'colors': img.get('colors', []),
                'keywords': img.get('keywords', []),
                'ocr_status': img.get('ocr_status', 'done'),
                'ocr_keywords': img.get('ocr_keywords', []),
                'uploaded_at': img.get('uploaded_at', '')
            }
            for img in images
        ],
        'total': len(images)
    })


@app.route('/api/ocr-progress', methods=['GET'])
def ocr_progress():
    """
    Get global OCR progress across all images.
    Returns a simple aggregate so the frontend can show a single progress indicator.
    """
    metadata = load_metadata()
    images = metadata.get('images', [])

    # Auto-timeout: mark very old pending/running jobs as failed so progress
    # does not get stuck forever if OCR hangs or the process was interrupted.
    # This is a backend concern only; frontend and APIs stay the same.
    STUCK_MINUTES = 10
    now = datetime.now()
    touched = False

    for img in images:
        status = (img.get('ocr_status') or '').lower()
        if status not in ('pending', 'running'):
            continue

        uploaded_at_str = img.get('uploaded_at')
        if not uploaded_at_str:
            continue

        try:
            uploaded_at = datetime.fromisoformat(uploaded_at_str)
        except Exception:
            continue

        age_minutes = (now - uploaded_at).total_seconds() / 60.0
        if age_minutes >= STUCK_MINUTES:
            img['ocr_status'] = 'failed'
            # Do not trust partial text/keywords if job is considered failed
            img['ocr_text'] = ''
            img['ocr_keywords'] = []
            touched = True

    if touched:
        save_metadata(metadata)

    total_jobs = 0
    completed_jobs = 0
    running_jobs = 0
    pending_jobs = 0

    for img in images:
        status = (img.get('ocr_status') or '').lower()
        if not status:
            continue

        total_jobs += 1
        if status in ('done', 'failed', 'skipped'):
            completed_jobs += 1
        elif status == 'running':
            running_jobs += 1
        elif status == 'pending':
            pending_jobs += 1

    return jsonify({
        'total': total_jobs,
        'completed': completed_jobs,
        'running': running_jobs,
        'pending': pending_jobs,
    })


@app.route('/api/images/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Delete an image by ID"""
    metadata = load_metadata()
    images = metadata.get('images', [])

    # Find and remove the image
    image_to_delete = None
    for i, img in enumerate(images):
        if img['id'] == image_id:
            image_to_delete = images.pop(i)
            break

    if not image_to_delete:
        return jsonify({'error': 'Image not found'}), 404

    # Delete file from disk
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_to_delete['filename'])
    if os.path.exists(filepath):
        os.remove(filepath)

    # Save updated metadata
    save_metadata(metadata)

    return '', 204


@app.route('/api/clear', methods=['POST'])
def clear_all_images():
    """Clear all images and metadata."""
    # Clear metadata
    save_metadata({'images': []})

    # Delete uploaded files
    upload_dir = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return jsonify({'success': True})



if __name__ == '__main__':
    # Initialize empty metadata file if it doesn't exist
    if not os.path.exists(METADATA_FILE):
        save_metadata({'images': []})

    app.run(host='0.0.0.0', port=5000, debug=False)