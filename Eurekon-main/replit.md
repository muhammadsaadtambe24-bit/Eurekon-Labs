# Natural Language Image Search Web App

## Overview

A full-stack web application that allows users to upload images (photos, screenshots, documents) and search them using natural language queries. The system uses AI-powered processing to extract text via OCR, detect dominant colors, and classify image types, enabling semantic search across uploaded images.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Hybrid Backend Design
The application uses a unique hybrid architecture where a Node.js/TypeScript server acts as a process manager that spawns and manages a Python Flask backend. This design allows leveraging Python's rich AI/ML ecosystem while maintaining a modern TypeScript build system.

- **Entry Point**: `server/index.ts` spawns `python3 app.py`
- **API Server**: Flask handles all HTTP requests for image upload, processing, and search
- **Static Files**: Flask serves the frontend from `static/` and `templates/` directories

### AI Processing Pipeline
Located in the `ai/` package, three modules handle image analysis:

1. **OCR Module** (`ai/ocr.py`): Uses Tesseract (pytesseract) to extract text from images, enabling search of screenshots, documents, and bills
2. **Color Module** (`ai/color.py`): Extracts dominant colors using PIL and maps RGB values to human-readable color names for natural language queries like "blue images"
3. **Vision Module** (`ai/vision.py`): Classifies images by type (photo, screenshot, document) using heuristics based on aspect ratio, color variance, and format

### Data Storage
- **Image Files**: Stored locally in `uploads/` directory
- **Metadata**: JSON file at `metadata/data.json` stores extracted text, detected colors, image type, and keywords for each uploaded image
- **No Database Required**: The MVP intentionally avoids database complexity by using file-based storage

### Frontend Architecture
Two frontend implementations exist:
1. **Flask Templates** (`templates/index.html` + `static/`): Vanilla HTML/CSS/JS for the image search interface
2. **React App** (`client/`): A modern React/TypeScript setup with shadcn/ui components, currently showing a 404 page (routes not configured)

The React frontend uses:
- Vite for bundling with path aliases (`@/`, `@shared/`, `@assets/`)
- TailwindCSS with shadcn/ui component library
- React Query for data fetching
- Wouter for client-side routing

### Database Schema (Prepared but Unused)
Drizzle ORM is configured with PostgreSQL for future use. The current schema (`shared/schema.ts`) only defines a users table, but the image search functionality uses JSON file storage instead.

## External Dependencies

### Python Dependencies
- **pytesseract**: OCR engine wrapper for Tesseract
- **Pillow (PIL)**: Image processing and manipulation
- **Flask**: Web framework for API endpoints

### Node.js Dependencies
- **Express**: HTTP server (for static file serving in production)
- **Drizzle ORM + drizzle-kit**: Database toolkit configured for PostgreSQL
- **@tanstack/react-query**: Server state management
- **shadcn/ui components**: Full Radix UI primitive library with Tailwind styling

### Build Tools
- **Vite**: Frontend bundler with React plugin
- **esbuild**: Server-side TypeScript bundling
- **tsx**: TypeScript execution for development

### External Services
- **PostgreSQL**: Database configured via `DATABASE_URL` environment variable (schema exists but not actively used for core functionality)