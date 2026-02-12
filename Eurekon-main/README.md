📷 EUREKON

A Flask-based AI-powered image search web app that allows users to upload images and search them using natural language, colors, visual cues, and OCR.

✨ Features

⚡ Fast multi-image upload

🎨 Dominant color extraction (base colors only)

🖼️ Vision-based image classification (photo, screenshot, document, etc.)

🔍 Object & visual keyword detection

🧠 Background OCR using EasyOCR

Runs asynchronously

Uploads return instantly

OCR is skipped for non-text images

📊 OCR progress tracking (pending, running, done, skipped, failed)

🔎 Natural language search

OCR text

Colors

Image type

Visual + OCR-derived keywords


🧠 Background OCR Design (Important)

OCR does NOT block uploads

A single background worker processes OCR jobs sequentially

OCR is triggered only when image keywords indicate text, such as:

document

screenshot

text-heavy

Large images are downscaled before OCR to:

Reduce RAM usage

Improve OCR speed

Prevent memory crashes


🚀 How It Works

Upload images

Saved immediately

Vision, color, and object detection run in foreground

OCR runs in background

Only if keywords indicate text

Status updated in metadata

Search anytime

OCR results appear automatically once finished


📌 OCR Status Values

Each image tracks its OCR state:

pending – queued for OCR

running – OCR in progress

done – OCR completed successfully

skipped – OCR skipped (not text-heavy)

failed – OCR error


🛠️ Tech Stack

Backend: Flask (Python)

OCR: EasyOCR

Image Processing: Pillow, NumPy

Frontend: HTML, CSS, Vanilla JS

Storage: JSON-based metadata (lightweight & fast)


🧪 Performance Notes

Uploading dozens of images remains fast

OCR is memory-safe and non-blocking

Designed to scale cleanly without race conditions


💤 Status

Background OCR pipeline implemented and stable.
Further optimizations and UI enhancements coming next.
