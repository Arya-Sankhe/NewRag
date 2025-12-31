# ColPali Vision RAG - Implementation Plan

> **For**: AI Coding Agent (Claude Opus 4.5 or similar)  
> **Companion Document**: `colpali_vision_rag_implementation.md` (contains all source code)

---

## Overview

Build a standalone visual document RAG system using ColPali (colSmol-500M) with Byaldi wrapper. The system allows users to upload PDF documents, ask questions, and receive answers with relevant page images displayed in the chat.

### Tech Stack
- **Backend**: FastAPI + Python 3.11 + Byaldi + OpenAI
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **Deployment**: Docker Compose

---

## Prerequisites

Before starting, the agent needs:
1. A new empty directory for the project
2. Docker and Docker Compose installed
3. OpenAI API key (for GPT-4o-mini vision)

---

## Phase 1: Project Setup

### Step 1.1: Create Directory Structure

```bash
mkdir -p colpali-vision-rag/{backend/app/{api/routes,core,models},frontend/src/{app,components,lib}}
cd colpali-vision-rag
```

### Step 1.2: Create Root Files

| File | Content |
|------|---------|
| `.env.example` | See specification doc |
| `docker-compose.yml` | See specification doc |
| `README.md` | Basic project description |

---

## Phase 2: Backend Implementation

Execute in this exact order:

### Step 2.1: Create Backend Base Files

| Order | File | Purpose |
|-------|------|---------|
| 1 | `backend/requirements.txt` | Python dependencies |
| 2 | `backend/config.py` | Configuration settings |
| 3 | `backend/Dockerfile` | Container definition |

### Step 2.2: Create Data Models

| Order | File | Purpose |
|-------|------|---------|
| 4 | `backend/app/__init__.py` | Empty init file |
| 5 | `backend/app/models/__init__.py` | Empty init file |
| 6 | `backend/app/models/schemas.py` | Pydantic request/response models |

### Step 2.3: Create Core Logic

| Order | File | Purpose |
|-------|------|---------|
| 7 | `backend/app/core/__init__.py` | Empty init file |
| 8 | `backend/app/core/retriever.py` | Byaldi/ColPali wrapper (main logic) |
| 9 | `backend/app/core/llm.py` | OpenAI GPT-4o vision client |

### Step 2.4: Create API Routes

| Order | File | Purpose |
|-------|------|---------|
| 10 | `backend/app/api/__init__.py` | Empty init file |
| 11 | `backend/app/api/routes/__init__.py` | Route exports |
| 12 | `backend/app/api/routes/health.py` | Health check endpoint |
| 13 | `backend/app/api/routes/documents.py` | Upload, list, delete documents |
| 14 | `backend/app/api/routes/chat.py` | Query and answer endpoint |

### Step 2.5: Create Main Entry

| Order | File | Purpose |
|-------|------|---------|
| 15 | `backend/main.py` | FastAPI app initialization |

### Checkpoint: Backend Verification

```bash
cd backend
pip install -r requirements.txt
python main.py
# Should start on http://localhost:8000
curl http://localhost:8000/health
# Expected: {"status":"healthy","model_loaded":false}
```

---

## Phase 3: Frontend Implementation

### Step 3.1: Create Frontend Base Files

| Order | File | Purpose |
|-------|------|---------|
| 16 | `frontend/package.json` | Node dependencies |
| 17 | `frontend/tsconfig.json` | TypeScript config (Next.js default) |
| 18 | `frontend/next.config.js` | Next.js config |
| 19 | `frontend/tailwind.config.js` | Tailwind config |
| 20 | `frontend/postcss.config.js` | PostCSS config |
| 21 | `frontend/Dockerfile` | Container definition |

### Step 3.2: Create App Structure

| Order | File | Purpose |
|-------|------|---------|
| 22 | `frontend/src/app/globals.css` | Global styles + Tailwind |
| 23 | `frontend/src/app/layout.tsx` | Root layout |
| 24 | `frontend/src/app/page.tsx` | Main page (chat + sidebar) |

### Step 3.3: Create Utilities

| Order | File | Purpose |
|-------|------|---------|
| 25 | `frontend/src/lib/api.ts` | API client functions |
| 26 | `frontend/src/lib/utils.ts` | Helper utilities (cn function) |

### Step 3.4: Create Components

| Order | File | Purpose |
|-------|------|---------|
| 27 | `frontend/src/components/ChatInterface.tsx` | Main chat UI |
| 28 | `frontend/src/components/MessageBubble.tsx` | Message with image display |
| 29 | `frontend/src/components/DocumentManager.tsx` | Upload and list documents |

### Checkpoint: Frontend Verification

```bash
cd frontend
npm install
npm run dev
# Should start on http://localhost:3000
```

---

## Phase 4: Docker Integration

### Step 4.1: Test Docker Build

```bash
# From project root
docker-compose build
```

### Step 4.2: Run with Docker

```bash
# Create .env file with OPENAI_API_KEY
cp .env.example .env
# Edit .env with actual API key

docker-compose up
```

---

## Phase 5: End-to-End Verification

### Test 1: Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","model_loaded":false}
```

### Test 2: Document Upload
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@sample.pdf"
# Expected: {"success":true,"message":"Indexed 1 document(s)","documents_added":1}
```

### Test 3: Document List
```bash
curl http://localhost:8000/api/v1/documents
# Expected: {"documents":[{"id":"0","name":"sample","page_count":N}],"total":1}
```

### Test 4: Chat Query
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is this document about?","include_images":true}'
# Expected: {"answer":"...","sources":[{"doc_id":0,"page_num":1,"score":0.XX,"image_base64":"..."}]}
```

### Test 5: Frontend Manual Test
1. Open http://localhost:3000
2. Upload a PDF in the sidebar
3. Wait for indexing to complete
4. Type a question in the chat
5. Verify answer appears with page thumbnails
6. Click a thumbnail to view full page

---

## Critical Implementation Notes

### 1. Byaldi Index Storage
- Index stored in `backend/data/index/`
- Registry file tracks document metadata
- Docker volume preserves data across restarts

### 2. Image Handling
- PDFs converted to PNG per page (150 DPI)
- Images stored in `backend/data/pages/{doc_name}/`
- Base64 encoded for API responses
- Frontend displays as clickable thumbnails

### 3. ColPali Model Loading
- Model loads lazily on first query (~30-60 seconds)
- ~2.5-3GB RAM usage on CPU
- Subsequent queries are fast

### 4. Error Handling
- Backend should handle missing API key gracefully
- Frontend should show loading states during upload
- Chat should display errors as system messages

---

## File Reference

All complete source code is in the companion document:
**`colpali_vision_rag_implementation.md`**

Copy code verbatim from that document for each file listed above.

---

## Success Criteria

The implementation is complete when:
- [ ] `docker-compose up` starts both services
- [ ] PDF upload works via UI
- [ ] Chat returns answers with page images
- [ ] Images are clickable in chat messages
- [ ] Document list updates after upload
- [ ] Clear all documents works
