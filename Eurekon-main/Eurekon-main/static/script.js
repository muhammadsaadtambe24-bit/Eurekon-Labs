/**
 * Natural Language Image Search - Frontend JavaScript
 * Handles file uploads, search, and UI interactions
 */

// ===== DOM Elements =====
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const getStartedBtn = document.getElementById('getStartedBtn');
const uploadSection = document.querySelector('.upload-section');
const resultsSection = document.querySelector('.results-section');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressCount = document.getElementById('progressCount');
const progressDetails = document.getElementById('progressDetails');
const compressionStatus = document.getElementById('compressionStatus');
const uploadStatus = document.getElementById('uploadStatus');
const ocrStatus = document.getElementById('ocrStatus');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const resultsGrid = document.getElementById('resultsGrid');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const emptyState = document.getElementById('emptyState');
const ocrProgressContainer = document.getElementById('ocrProgressContainer');
const ocrProgressText = document.getElementById('ocrProgressText');
const ocrProgressFill = document.getElementById('ocrProgressFill');
const imageModal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const modalInfo = document.getElementById('modalInfo');
const modalClose = document.getElementById('modalClose');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// ===== State =====
let currentImages = [];
let ocrProgressTimer = null;
let lastOcrCompleted = 0;
let processingState = {
    total: 0,
    compressed: 0,
    uploaded: 0
};

// ===== OCR Progress Functions =====

async function fetchOcrProgress() {
    if (!ocrProgressContainer) return;

    try {
        const response = await fetch('/api/ocr-progress');
        if (!response.ok) {
            return;
        }

        const data = await response.json();
        const total = data.total || 0;
        const completed = data.completed || 0;

        // New OCR work detected: clear reload guard so we can reload once per cycle
        if (total && completed < total) {
            try {
                sessionStorage.removeItem('ocrReloadedOnce');
            } catch (e) {
                // Ignore storage issues
            }
        }

        if (!total || completed >= total) {
            ocrProgressContainer.style.display = 'none';
            if (ocrProgressTimer) {
                clearInterval(ocrProgressTimer);
                ocrProgressTimer = null;
            }
            if (ocrStatus) {
                // Global OCR queue is idle or complete
                updateStageStatus(ocrStatus, 'completed', total ? `✓ ${completed} processed` : 'Idle');
            }
            lastOcrCompleted = completed || 0;

            // When OCR is fully complete, trigger a one-time full page reload
            if (total && completed >= total) {
                try {
                    const already = sessionStorage.getItem('ocrReloadedOnce');
                    if (!already) {
                        sessionStorage.setItem('ocrReloadedOnce', 'true');
                        window.location.reload();
                    }
                } catch (e) {
                    // Ignore storage issues; safest fallback is no reload loop
                }
            }
            return;
        }

        const percent = Math.round((completed / total) * 100);
        ocrProgressContainer.style.display = 'block';
        if (ocrProgressText) {
            ocrProgressText.textContent = `OCR processing: ${completed} / ${total} completed`;
        }
        if (ocrProgressFill) {
            ocrProgressFill.style.width = `${percent}%`;
        }
        if (ocrStatus) {
            updateStageStatus(ocrStatus, 'processing', `${completed}/${total} processing`);
        }

        // If OCR has progressed since last poll, refresh images so grayscale state updates
        if (completed !== lastOcrCompleted) {
            lastOcrCompleted = completed;
            // Refresh current gallery view without full page reload
            // (uses existing search/loadImages logic)
            loadImages();
        }
    } catch (error) {
        // Fail silently; OCR UI is non-critical
        console.error('OCR progress error:', error);
    }
}

function startOcrProgressPolling() {
    if (ocrProgressTimer) return;
    fetchOcrProgress();
    ocrProgressTimer = setInterval(fetchOcrProgress, 3000);
}

// ===== Utility Functions =====

/** 
 * Show toast notification
 */
function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    toast.className = 'toast show ' + type;

    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Update progress stage status text
 */
function updateStageStatus(element, status, text) {
    if (!element) return;
    element.className = `stage-status ${status}`;
    element.textContent = text;
}

/**
 * Update overall upload/compression/OCR progress bar
 */
function updateProgress(completed, total, stage) {
    if (!uploadProgress || !progressFill || !progressText) return;
    if (!total || total <= 0) {
        progressFill.style.width = '0%';
        if (progressCount) {
            progressCount.textContent = `0/0`;
        }
        return;
    }

    let percentage = Math.round((completed / total) * 100);

    // Treat compression as first half, upload/OCR as second half
    if (stage === 'compression') {
        percentage = Math.round((completed / total) * 50);
    } else if (stage === 'upload' || stage === 'processing') {
        percentage = 50 + Math.round((completed / total) * 50);
    } else if (stage === 'complete') {
        percentage = 100;
    }

    if (percentage < 0) percentage = 0;
    if (percentage > 100) percentage = 100;

    progressFill.style.width = `${percentage}%`;

    if (progressCount) {
        progressCount.textContent = `${completed}/${total}`;
    }

    if (stage === 'compression') {
        progressText.textContent = 'Compressing images...';
    } else if (stage === 'upload') {
        progressText.textContent = 'Uploading images...';
    } else if (stage === 'processing') {
        progressText.textContent = 'AI processing...';
    } else if (stage === 'complete') {
        progressText.textContent = 'Complete!';
    }
}

// Client-side image compression (borrowed from friend's version)
async function compressImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');

                    const MAX = 1024; // max dimension for faster uploads
                    const scale = Math.min(1, MAX / Math.max(img.width, img.height));

                    canvas.width = img.width * scale;
                    canvas.height = img.height * scale;

                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                    canvas.toBlob(
                        (blob) => {
                            if (!blob) {
                                resolve(file); // fallback to original if compression fails
                                return;
                            }
                            const compressedFile = new File([blob], file.name, {
                                type: 'image/jpeg',
                                lastModified: Date.now()
                            });
                            resolve(compressedFile);
                        },
                        'image/jpeg',
                        0.75
                    );
                };
                img.onerror = () => reject(new Error('Failed to load image for compression'));
                img.src = e.target.result;
            } catch (err) {
                reject(err);
            }
        };

        reader.onerror = () => reject(new Error('Failed to read file for compression'));
        reader.readAsDataURL(file);
    });
}


// ===== Upload Functions =====

/**
 * Handle file upload
 */
async function uploadFiles(files) {
    if (!files || files.length === 0) return;

    // Filter valid image files
    let validFiles = Array.from(files).filter(file =>
        file.type.startsWith('image/')
    );

    if (validFiles.length === 0) {
        showToast('Please select valid image files', 'error');
        return;
    }

    // Enforce client-side limit of 50 images per upload action
    const MAX_FILES_PER_UPLOAD = 50;
    if (validFiles.length > MAX_FILES_PER_UPLOAD) {
        showToast(`You can upload a maximum of ${MAX_FILES_PER_UPLOAD} images at once. Only the first ${MAX_FILES_PER_UPLOAD} will be processed.`, 'error');
        validFiles = validFiles.slice(0, MAX_FILES_PER_UPLOAD);
    }

    // Reset processing state
    processingState = {
        total: validFiles.length,
        compressed: 0,
        uploaded: 0
    };

    // Show progress UI
    if (uploadProgress) {
        uploadProgress.style.display = 'block';
    }
    if (progressFill) {
        progressFill.style.width = '0%';
    }
    if (progressText) {
        progressText.textContent = 'Starting...';
    }
    if (progressCount) {
        progressCount.textContent = `0/${validFiles.length}`;
    }

    // Reset stage statuses
    updateStageStatus(compressionStatus, 'waiting', 'Waiting...');
    updateStageStatus(uploadStatus, 'waiting', 'Waiting...');
    updateStageStatus(ocrStatus, 'waiting', 'Waiting...');

    try {
        updateStageStatus(compressionStatus, 'processing', 'Compressing...');
        updateStageStatus(uploadStatus, 'waiting', 'Waiting...');
        updateStageStatus(ocrStatus, 'waiting', 'Waiting...');

        let uploadedCount = 0;

        // Compress → upload each image sequentially so results appear one-by-one
        for (let i = 0; i < validFiles.length; i++) {
            const file = validFiles[i];

            // ----- Compress this image -----
            const compressed = await compressImage(file);
            processingState.compressed = i + 1;
            updateStageStatus(
                compressionStatus,
                'processing',
                `${processingState.compressed}/${processingState.total}`
            );
            updateProgress(processingState.compressed, processingState.total, 'compression');

            // ----- Upload this image (single-image batch) -----
            const formData = new FormData();
            formData.append('files', compressed);

            updateStageStatus(uploadStatus, 'processing', `Uploading ${i + 1}/${processingState.total}...`);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const result = await response.json();

            if (result.uploaded && result.uploaded.length > 0) {
                uploadedCount += result.uploaded.length;
                processingState.uploaded = uploadedCount;

                updateStageStatus(
                    uploadStatus,
                    'processing',
                    `${processingState.uploaded}/${processingState.total} uploaded`
                );
                updateProgress(processingState.uploaded, processingState.total, 'upload');

                // Render newly uploaded images immediately
                await loadImages();

                // Ensure OCR progress polling is active while background OCR runs
                startOcrProgressPolling();
            }

            if (result.errors && result.errors.length > 0) {
                console.error('Upload errors:', result.errors);
                showToast(`${result.errors.length} file(s) failed to upload`, 'error');
            }
        }

        updateStageStatus(compressionStatus, 'completed', `✓ ${processingState.total} compressed`);
        updateStageStatus(uploadStatus, 'completed', `✓ ${processingState.uploaded} uploaded`);

        // OCR stage text is driven by global /api/ocr-progress polling;
        // by the time uploads complete, images are queued for OCR.
        updateStageStatus(ocrStatus, 'processing', 'Running OCR...');

        // Mark overall progress complete visually
        updateProgress(processingState.total, processingState.total, 'complete');

        if (processingState.uploaded > 0) {
            showToast(`Successfully processed ${processingState.uploaded} image(s)`, 'success');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload failed. Please try again.', 'error');
        updateStageStatus(uploadStatus, 'error', '✗ Failed');
        updateStageStatus(ocrStatus, 'error', '✗ Failed');
    } finally {
        // Hide progress after a delay
        setTimeout(() => {
            if (uploadProgress) {
                uploadProgress.style.display = 'none';
            }
            if (progressFill) {
                progressFill.style.width = '0%';
            }
        }, 3000);
    }
}

// ===== Search Functions =====

/**
 * Search images with query
 */
async function searchImages(query = '') {
    try {
        const url = query
            ? `/api/search?q=${encodeURIComponent(query)}`
            : '/api/images';

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error('Search failed');
        }

        const result = await response.json();

        // ========== 🔵 CHANGE #1: LIMIT TO TOP 10 SEARCH RESULTS 🔵 ==========
        // Update UI
        if (query) {
            resultsTitle.textContent = `Results for "${query}"`;
            // Limit search results to top 5 most relevant
            const allResults = result.results || [];
            currentImages = allResults.slice(0, 5);
            
            // Show notification if more results were found
            if (allResults.length > 5) {
                showToast(`Showing top 5 of ${allResults.length} results`, 'info');
            }
        } else {
            resultsTitle.textContent = 'Your Images';
            const allImages = result.images || [];
            // Newest images should appear at the top in "Your Images"
            currentImages = allImages.slice().reverse();
        }
        // ========== 🔵 END CHANGE #1 🔵 ==========

        resultsCount.textContent = `${currentImages.length} image(s)`;
        renderImages(currentImages, !!query);

        // Auto-scroll to results if searching
        if (query && currentImages.length > 0) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

    } catch (error) {
        console.error('Search error:', error);
        showToast('Search failed. Please try again.', 'error');
    }
}

/**
 * Load all images
 */
async function loadImages() {
    await searchImages('');
}

/**
 * Search with a hint tag
 */
function searchWithHint(hint) {
    searchInput.value = hint;
    searchImages(hint);
}

// Make it global for onclick handlers
window.searchWithHint = searchWithHint;

// ===== Render Functions =====

/**
 * Render image cards
 */
function renderImages(images, showRelevance = false) {
    if (!images || images.length === 0) {
        resultsGrid.innerHTML = `
            <div class="empty-state" id="emptyState">
                <div class="empty-icon">📷</div>
                <h3>No images found</h3>
                <p>${showRelevance ? 'Try a different search query' : 'Upload some images to get started'}</p>
            </div>
        `;
        return;
    }

    // Merge visual keywords + OCR-derived (when available)
    const allKeywords = (img) => [...(img.keywords || []), ...(img.ocr_keywords || [])];

    resultsGrid.innerHTML = images.map(img => {
        const isOcrPending = img.ocr_status === 'pending' || img.ocr_status === 'running';

        return `
        <div class="image-card ${isOcrPending ? 'ocr-pending' : ''}" data-testid="card-image-${img.id}" onclick="openImage('${img.id}')">
            ${showRelevance && img.relevance ? `<span class="relevance-badge">${img.relevance} pts</span>` : ''}
            <button class="delete-btn" onclick="event.stopPropagation(); deleteImage('${img.id}')" data-testid="button-delete-${img.id}">&times;</button>
            <img src="/uploads/${escapeHtml(img.filename)}" alt="${escapeHtml(img.original_filename)}" loading="lazy">
            <div class="image-card-info">
                <div class="image-card-name">${escapeHtml(img.original_filename)}</div>
                <div class="image-card-tags">
                ${allKeywords(img).slice(0, 2).map(keyword =>
                    `<span class="image-tag keyword">${escapeHtml(keyword)}</span>`
                ).join('')}
                ${img.image_type
                ? `<span class="image-tag type">${escapeHtml(img.image_type)}</span>`
                : ''
                }

                ${(img.colors || []).slice(0, 2).map(color =>
                `<span class="image-tag color">${escapeHtml(color)}</span>`
                ).join('')}

                </div>
            </div>
        </div>`;
    }).join('');
}

// ===== Image Actions =====

/**
 * Open image in modal
 */
function openImage(imageId) {
    const image = currentImages.find(img => img.id === imageId);
    if (!image) return;

    modalImage.src = `/uploads/${image.filename}`;
    modalImage.alt = image.original_filename;

    const modalKeywords = [...(image.keywords || []), ...(image.ocr_keywords || [])];
    modalInfo.innerHTML = `
        <div style="color: var(--text-primary); margin-bottom: 8px;">
            <strong>${escapeHtml(image.original_filename)}</strong>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        ${modalKeywords.slice(0, 3).map(k =>
            `<span class="image-tag keyword">${escapeHtml(k)}</span>`
        ).join('')}
        ${image.image_type ? `<span class="image-tag type">${escapeHtml(image.image_type)}</span>` : ''}

        ${(image.colors || []).slice(0,3).map(c =>
            `<span class="image-tag color">${escapeHtml(c)}</span>`
        ).join('')}
      
        </div>
    `;

    imageModal.classList.add('active');
}

// Make it global for onclick handlers
window.openImage = openImage;

/**
 * Delete an image
 */
async function deleteImage(imageId) {
    if (!confirm('Are you sure you want to delete this image?')) {
        return;
    }

    try {
        const response = await fetch(`/api/images/${imageId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast('Image deleted', 'success');
            loadImages();
        } else {
            throw new Error('Delete failed');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast('Failed to delete image', 'error');
    }
}

// Make it global for onclick handlers
window.deleteImage = deleteImage;

// ===== Event Listeners =====

// File input change
fileInput.addEventListener('change', (e) => {
    uploadFiles(e.target.files);
    fileInput.value = ''; // Reset for same file re-upload
});

// Drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    uploadFiles(e.dataTransfer.files);
});

// Search
searchBtn.addEventListener('click', () => {
    searchImages(searchInput.value.trim());
});

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        searchImages(searchInput.value.trim());
    }
});

// Clear search on empty input
searchInput.addEventListener('input', (e) => {
    if (e.target.value === '') {
        loadImages();
    }
});

// Modal close
modalClose.addEventListener('click', () => {
    imageModal.classList.remove('active');
});

// Get Started Scroll
getStartedBtn.addEventListener('click', () => {
    uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) {
        imageModal.classList.remove('active');
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        imageModal.classList.remove('active');
    }
});

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    loadImages();
});

document.getElementById('clearAllBtn').addEventListener('click', async () => {
    if (!confirm('Delete all uploaded images?')) return;

    const res = await fetch('/api/clear', { method: 'POST' });
    const data = await res.json();

    if (data.success) {
        loadImages(); // reload gallery
        showToast('All images deleted');
    }
});

const scrollTopBtn = document.getElementById('scrollTopBtn');

window.addEventListener('scroll', () => {
    if (window.scrollY > 300) {
        scrollTopBtn.style.display = 'block';
    } else {
        scrollTopBtn.style.display = 'none';
    }
});

scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});