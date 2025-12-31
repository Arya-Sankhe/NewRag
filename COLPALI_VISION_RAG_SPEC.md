# ColPali Vision RAG - Complete Implementation Specification

> **Purpose**: This document provides complete specifications for building a standalone vision-first RAG system using ColPali (colSmol-500M) with Byaldi wrapper. Designed for one-shot implementation by an AI coding agent.

---

## Project Overview

**Name**: ColPali Vision RAG  
**Description**: A document Q&A system that uses visual embeddings instead of OCR, capable of understanding charts, tables, and layouts natively.

### Core Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Vision Retrieval | ColPali (colSmol-500M) via Byaldi | Latest |
| LLM | OpenAI GPT-4o-mini | API |
| Backend | FastAPI + Python | 3.11+ |
| Frontend | Next.js 14 + TypeScript | 14.x |
| UI Components | shadcn/ui + Tailwind CSS | Latest |
| Containerization | Docker + Docker Compose | Latest |

---

## Project Structure

```
colpali-vision-rag/
├── docker-compose.yml
├── README.md
├── .env.example
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                    # FastAPI app entry
│   ├── config.py                  # Configuration
│   └── app/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── documents.py   # Upload, list, delete
│       │   │   ├── chat.py        # Query endpoint
│       │   │   └── health.py      # Health check
│       │   └── deps.py            # Dependencies
│       ├── core/
│       │   ├── __init__.py
│       │   ├── retriever.py       # Byaldi wrapper
│       │   └── llm.py             # OpenAI client
│       └── models/
│           ├── __init__.py
│           └── schemas.py         # Pydantic models
│
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── next.config.js
    ├── tailwind.config.js
    ├── tsconfig.json
    └── src/
        ├── app/
        │   ├── layout.tsx
        │   ├── page.tsx           # Main chat page
        │   └── globals.css
        ├── components/
        │   ├── ui/                # shadcn components
        │   ├── ChatInterface.tsx
        │   ├── DocumentManager.tsx
        │   └── MessageBubble.tsx
        └── lib/
            ├── api.ts             # API client
            └── utils.ts
```

---

## Backend Implementation

### File: `backend/requirements.txt`

```
# ColPali + Byaldi
colpali-engine>=0.3.5
byaldi>=0.1.0

# PyTorch (CPU for compatibility)
torch>=2.2.0,<2.6.0
torchvision>=0.17.0

# Transformers
transformers>=4.46.2

# Image processing
Pillow>=10.0.0
pdf2image>=1.16.0

# Web framework
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9

# OpenAI
openai>=1.12.0

# Utilities
pydantic>=2.5.0
python-dotenv>=1.0.0
aiofiles>=23.2.0
```

---

### File: `backend/config.py`

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
UPLOADS_DIR = DATA_DIR / "uploads"
PAGES_DIR = DATA_DIR / "pages"

# Create directories
for d in [DATA_DIR, INDEX_DIR, UPLOADS_DIR, PAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ColPali Configuration
COLPALI_MODEL = os.getenv("COLPALI_MODEL", "vidore/colSmol-500M")
COLPALI_DEVICE = os.getenv("COLPALI_DEVICE", "cpu")  # or "cuda"
INDEX_NAME = "documents"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Retrieval Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_IMAGES_IN_RESPONSE = int(os.getenv("MAX_IMAGES_IN_RESPONSE", "3"))
```

---

### File: `backend/app/models/schemas.py`

```python
from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class DocumentInfo(BaseModel):
    id: str
    name: str
    page_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int

class UploadResponse(BaseModel):
    success: bool
    message: str
    documents_added: int

class RetrievedPage(BaseModel):
    doc_id: int
    page_num: int
    score: float
    image_base64: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    include_images: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[RetrievedPage]

class ClearResponse(BaseModel):
    success: bool
    message: str
```

---

### File: `backend/app/core/retriever.py`

```python
"""
ColPali Retriever using Byaldi wrapper.
Handles document indexing and visual retrieval.
"""

import base64
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import io

from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path

import config

class ColPaliRetriever:
    """Singleton wrapper for Byaldi/ColPali model."""
    
    _instance = None
    _model = None
    _document_registry: Dict[str, Dict] = {}  # Track indexed documents
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _ensure_model_loaded(self):
        """Lazy load the ColPali model."""
        if self._model is not None:
            return
        
        print(f"🔄 Loading ColPali model: {config.COLPALI_MODEL}")
        
        # Check if index exists
        index_path = config.INDEX_DIR / config.INDEX_NAME
        if index_path.exists():
            print(f"📁 Loading existing index from {index_path}")
            self._model = RAGMultiModalModel.from_index(
                str(index_path),
                index_root=str(config.INDEX_DIR)
            )
            self._load_registry()
        else:
            print(f"🆕 Creating new model instance")
            self._model = RAGMultiModalModel.from_pretrained(
                config.COLPALI_MODEL,
                device=config.COLPALI_DEVICE
            )
        
        print(f"✅ ColPali model ready")
    
    def _load_registry(self):
        """Load document registry from disk."""
        registry_path = config.INDEX_DIR / "registry.json"
        if registry_path.exists():
            self._document_registry = json.loads(registry_path.read_text())
    
    def _save_registry(self):
        """Save document registry to disk."""
        registry_path = config.INDEX_DIR / "registry.json"
        registry_path.write_text(json.dumps(self._document_registry, indent=2))
    
    def index_pdf(self, pdf_path: Path) -> Tuple[int, int]:
        """
        Index a PDF document.
        
        Returns:
            Tuple of (doc_id, page_count)
        """
        self._ensure_model_loaded()
        
        doc_name = pdf_path.stem
        
        # Convert PDF to images
        print(f"📄 Converting PDF to images: {doc_name}")
        images = convert_from_path(str(pdf_path), dpi=150)
        page_count = len(images)
        
        # Save page images for later retrieval
        doc_pages_dir = config.PAGES_DIR / doc_name
        doc_pages_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            img_path = doc_pages_dir / f"page_{i+1}.png"
            img.save(str(img_path), "PNG")
        
        # Index with Byaldi
        print(f"🔍 Indexing {page_count} pages with ColPali...")
        
        # Check if this is first document or adding to existing
        if not self._document_registry:
            # First document - create new index
            self._model.index(
                input_path=str(doc_pages_dir),
                index_name=config.INDEX_NAME,
                store_collection_with_index=True,
                overwrite=True
            )
            doc_id = 0
        else:
            # Add to existing index
            doc_id = len(self._document_registry)
            self._model.add_to_index(
                input_path=str(doc_pages_dir),
                store_collection_with_index=True
            )
        
        # Update registry
        self._document_registry[doc_name] = {
            "id": doc_id,
            "name": doc_name,
            "page_count": page_count,
            "path": str(doc_pages_dir)
        }
        self._save_registry()
        
        print(f"✅ Indexed {doc_name}: {page_count} pages")
        return doc_id, page_count
    
    def search(
        self, 
        query: str, 
        k: int = None,
        include_images: bool = True
    ) -> List[Dict]:
        """
        Search for relevant pages.
        
        Returns:
            List of result dicts with doc_id, page_num, score, and optionally image_base64
        """
        self._ensure_model_loaded()
        k = k or config.TOP_K_RESULTS
        
        if not self._document_registry:
            return []
        
        print(f"🔍 Searching: '{query[:50]}...'")
        
        results = self._model.search(query, k=k)
        
        processed = []
        for result in results:
            item = {
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": float(result.score)
            }
            
            if include_images:
                # Load and encode the page image
                image_b64 = self._get_page_image_base64(
                    result.doc_id, 
                    result.page_num
                )
                if image_b64:
                    item["image_base64"] = image_b64
            
            processed.append(item)
        
        return processed
    
    def _get_page_image_base64(self, doc_id: int, page_num: int) -> Optional[str]:
        """Get base64 encoded page image."""
        # Find document by ID
        for doc_name, doc_info in self._document_registry.items():
            if doc_info["id"] == doc_id:
                img_path = Path(doc_info["path"]) / f"page_{page_num}.png"
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
        return None
    
    def get_documents(self) -> List[Dict]:
        """Get list of indexed documents."""
        return [
            {
                "id": info["id"],
                "name": info["name"],
                "page_count": info["page_count"]
            }
            for info in self._document_registry.values()
        ]
    
    def clear_index(self):
        """Clear all indexed documents."""
        import shutil
        
        # Clear directories
        for d in [config.INDEX_DIR, config.PAGES_DIR]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        
        # Reset state
        self._document_registry = {}
        self._model = None
        
        print("🗑️ Index cleared")
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Singleton instance
retriever = ColPaliRetriever()
```

---

### File: `backend/app/core/llm.py`

```python
"""
LLM client for generating answers from retrieved context.
"""

from typing import List, Dict
from openai import OpenAI
import config

class LLMClient:
    """OpenAI client for answer generation."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
    
    def generate_answer(
        self,
        query: str,
        retrieved_pages: List[Dict],
        include_images: bool = True
    ) -> str:
        """
        Generate an answer using retrieved page images as context.
        
        Uses GPT-4o vision capabilities to understand page images.
        """
        if not retrieved_pages:
            return "I couldn't find any relevant information in the documents."
        
        # Build messages with images
        messages = [
            {
                "role": "system",
                "content": """You are a helpful document assistant. Answer questions based on the provided document pages.
                
Instructions:
- Answer based ONLY on the visible content in the provided page images
- Be specific and cite page numbers when relevant
- If the answer is not found in the pages, say so clearly
- Format your response with markdown for readability"""
            }
        ]
        
        # Add user message with images
        content = [{"type": "text", "text": f"Question: {query}\n\nHere are the relevant document pages:"}]
        
        # Add page images (limit to avoid token limits)
        pages_to_include = retrieved_pages[:config.MAX_IMAGES_IN_RESPONSE]
        
        for page in pages_to_include:
            if include_images and page.get("image_base64"):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page['image_base64']}",
                        "detail": "high"
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"[Page {page['page_num']} from Document {page['doc_id']}, relevance: {page['score']:.2f}]"
                })
        
        messages.append({"role": "user", "content": content})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3
        )
        
        return response.choices[0].message.content


# Singleton instance
llm_client = LLMClient()
```

---

### File: `backend/app/api/routes/documents.py`

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
from pathlib import Path

import config
from app.models.schemas import (
    DocumentListResponse, 
    DocumentInfo, 
    UploadResponse,
    ClearResponse
)
from app.core.retriever import retriever

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    docs = retriever.get_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs)
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and index PDF documents."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    indexed_count = 0
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        
        # Save uploaded file
        upload_path = config.UPLOADS_DIR / file.filename
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        try:
            # Index with ColPali
            doc_id, page_count = retriever.index_pdf(upload_path)
            indexed_count += 1
        except Exception as e:
            print(f"Error indexing {file.filename}: {e}")
            continue
    
    return UploadResponse(
        success=indexed_count > 0,
        message=f"Indexed {indexed_count} document(s)",
        documents_added=indexed_count
    )


@router.delete("/clear", response_model=ClearResponse)
async def clear_documents():
    """Clear all indexed documents."""
    try:
        retriever.clear_index()
        return ClearResponse(success=True, message="All documents cleared")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### File: `backend/app/api/routes/chat.py`

```python
from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse, RetrievedPage
from app.core.retriever import retriever
from app.core.llm import llm_client

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Query the document collection and generate an answer.
    """
    # Retrieve relevant pages
    results = retriever.search(
        query=request.query,
        include_images=request.include_images
    )
    
    # Generate answer using LLM with vision
    answer = llm_client.generate_answer(
        query=request.query,
        retrieved_pages=results,
        include_images=request.include_images
    )
    
    # Build response
    sources = [
        RetrievedPage(
            doc_id=r["doc_id"],
            page_num=r["page_num"],
            score=r["score"],
            image_base64=r.get("image_base64")
        )
        for r in results
    ]
    
    return ChatResponse(answer=answer, sources=sources)
```

---

### File: `backend/app/api/routes/health.py`

```python
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.retriever import retriever

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=retriever.is_loaded
    )
```

---

### File: `backend/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from app.api.routes import documents, chat, health

app = FastAPI(
    title="ColPali Vision RAG API",
    description="Visual document retrieval using ColPali",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    print("🚀 Starting ColPali Vision RAG API")
    print(f"   Model: {config.COLPALI_MODEL}")
    print(f"   Device: {config.COLPALI_DEVICE}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
```

---

### File: `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/index data/uploads data/pages

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Frontend Implementation

### File: `frontend/package.json`

```json
{
  "name": "colpali-vision-rag-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "react-dropzone": "^14.2.3",
    "react-markdown": "^9.0.1",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.1",
    "class-variance-authority": "^0.7.0",
    "lucide-react": "^0.323.0",
    "@radix-ui/react-scroll-area": "^1.0.5",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-slot": "^1.0.2"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.4.17",
    "postcss": "^8.4.35",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  }
}
```

---

### File: `frontend/src/lib/api.ts`

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface DocumentInfo {
  id: string;
  name: string;
  page_count: number;
}

export interface RetrievedPage {
  doc_id: number;
  page_num: number;
  score: number;
  image_base64?: string;
}

export interface ChatResponse {
  answer: string;
  sources: RetrievedPage[];
}

export async function uploadDocuments(files: File[]): Promise<{ success: boolean; message: string }> {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  return response.json();
}

export async function getDocuments(): Promise<{ documents: DocumentInfo[]; total: number }> {
  const response = await fetch(`${API_BASE}/api/v1/documents`);
  return response.json();
}

export async function clearDocuments(): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE}/api/v1/documents/clear`, {
    method: 'DELETE',
  });
  return response.json();
}

export async function sendMessage(query: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, include_images: true }),
  });
  return response.json();
}
```

---

### File: `frontend/src/components/ChatInterface.tsx`

```tsx
'use client';

import { useState } from 'react';
import { sendMessage, RetrievedPage } from '@/lib/api';
import { MessageBubble } from './MessageBubble';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: RetrievedPage[];
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await sendMessage(userMessage);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: '❌ Error: Failed to get response' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">👋 Welcome to ColPali Vision RAG</p>
            <p className="text-sm mt-2">Upload documents and ask questions about them</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-700 rounded-lg px-4 py-2 animate-pulse">
              Thinking...
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents..."
            className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-2 rounded-lg font-medium transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
```

---

### File: `frontend/src/components/MessageBubble.tsx`

```tsx
'use client';

import ReactMarkdown from 'react-markdown';
import { RetrievedPage } from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: RetrievedPage[];
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-gray-700 text-gray-100'
        }`}
      >
        {/* Message content */}
        <div className="prose prose-invert prose-sm max-w-none">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Source images for assistant messages */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-3 border-t border-gray-600">
            <p className="text-xs text-gray-400 mb-2">📄 Retrieved Pages:</p>
            <div className="flex gap-2 overflow-x-auto pb-2">
              {message.sources.slice(0, 3).map((source, i) => (
                <div key={i} className="flex-shrink-0">
                  {source.image_base64 && (
                    <img
                      src={`data:image/png;base64,${source.image_base64}`}
                      alt={`Page ${source.page_num}`}
                      className="h-32 rounded border border-gray-500 cursor-pointer hover:opacity-80"
                      onClick={() => {
                        // Open full image in new tab
                        const w = window.open();
                        if (w) {
                          w.document.write(`<img src="data:image/png;base64,${source.image_base64}" />`);
                        }
                      }}
                    />
                  )}
                  <p className="text-xs text-gray-400 mt-1 text-center">
                    Page {source.page_num} ({(source.score * 100).toFixed(0)}%)
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

---

### File: `frontend/src/components/DocumentManager.tsx`

```tsx
'use client';

import { useCallback, useEffect, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadDocuments, getDocuments, clearDocuments, DocumentInfo } from '@/lib/api';

export function DocumentManager() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await getDocuments();
      setDocuments(response.documents);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setSelectedFiles(prev => [...prev, ...acceptedFiles]);
    setMessage(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: true,
  });

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    setIsUploading(true);
    setMessage(null);

    try {
      const result = await uploadDocuments(selectedFiles);
      setMessage(result.message);
      setSelectedFiles([]);
      await fetchDocuments();
    } catch (error) {
      setMessage('❌ Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleClear = async () => {
    if (!confirm('Clear all documents?')) return;
    setIsLoading(true);
    try {
      await clearDocuments();
      setDocuments([]);
      setMessage('🗑️ All documents cleared');
    } catch (error) {
      setMessage('❌ Clear failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-lg font-semibold">📚 Documents</h2>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-gray-400">Drop PDFs here or click to select</p>
      </div>

      {/* Selected files */}
      {selectedFiles.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Selected: {selectedFiles.length} file(s)</p>
          <div className="flex flex-wrap gap-2">
            {selectedFiles.map((f, i) => (
              <span
                key={i}
                className="bg-gray-700 px-2 py-1 rounded text-sm cursor-pointer hover:bg-red-600"
                onClick={() => setSelectedFiles(prev => prev.filter((_, idx) => idx !== i))}
              >
                {f.name} ✕
              </span>
            ))}
          </div>
          <button
            onClick={handleUpload}
            disabled={isUploading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 py-2 rounded-lg font-medium"
          >
            {isUploading ? 'Uploading...' : 'Upload Documents'}
          </button>
        </div>
      )}

      {message && <p className="text-sm text-gray-300">{message}</p>}

      {/* Document list */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-400">{documents.length} document(s) indexed</span>
          <button
            onClick={handleClear}
            disabled={documents.length === 0 || isLoading}
            className="text-xs text-red-400 hover:text-red-300 disabled:text-gray-600"
          >
            Clear All
          </button>
        </div>
        {documents.map((doc) => (
          <div key={doc.id} className="bg-gray-800 rounded px-3 py-2 text-sm flex justify-between">
            <span>{doc.name}</span>
            <span className="text-gray-500">{doc.page_count} pages</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

### File: `frontend/src/app/page.tsx`

```tsx
import { ChatInterface } from '@/components/ChatInterface';
import { DocumentManager } from '@/components/DocumentManager';

export default function Home() {
  return (
    <main className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <aside className="w-80 border-r border-gray-700 overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-xl font-bold">🔍 ColPali Vision RAG</h1>
          <p className="text-xs text-gray-400 mt-1">Visual document retrieval</p>
        </div>
        <DocumentManager />
      </aside>

      {/* Main chat area */}
      <section className="flex-1 flex flex-col">
        <ChatInterface />
      </section>
    </main>
  );
}
```

---

### File: `frontend/src/app/layout.tsx`

```tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ColPali Vision RAG',
  description: 'Visual document retrieval with ColPali',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
```

---

### File: `frontend/src/app/globals.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-gray-900 text-white;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: #1f2937;
}

::-webkit-scrollbar-thumb {
  background: #4b5563;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #6b7280;
}
```

---

### File: `frontend/tailwind.config.js`

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {},
  },
  plugins: [],
};
```

---

### File: `frontend/next.config.js`

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
};

module.exports = nextConfig;
```

---

### File: `frontend/Dockerfile`

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
```

---

## Docker Compose

### File: `docker-compose.yml`

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - backend_data:/app/data
    environment:
      - COLPALI_MODEL=vidore/colSmol-500M
      - COLPALI_DEVICE=cpu
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=gpt-4o-mini
      - FRONTEND_URL=http://localhost:3000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

volumes:
  backend_data:
```

---

### File: `.env.example`

```
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
COLPALI_MODEL=vidore/colSmol-500M
COLPALI_DEVICE=cpu
```

---

## Verification Plan

### Automated Tests

1. **Backend health check**:
   ```bash
   curl http://localhost:8000/health
   # Expected: {"status":"healthy","model_loaded":false}
   ```

2. **Document upload test**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/documents/upload \
     -F "files=@test.pdf"
   # Expected: {"success":true,...}
   ```

3. **Chat test**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query":"What is in this document?"}'
   # Expected: {"answer":"...","sources":[...]}
   ```

### Manual Verification

1. Run `docker-compose up --build`
2. Open http://localhost:3000
3. Upload a PDF document
4. Ask a question about the document
5. Verify images appear in the response

---

## Key Implementation Notes

1. **Model Loading**: ColPali model loads on first use (~2-3GB RAM)
2. **PDF to Images**: Uses `pdf2image` (requires `poppler-utils` in Docker)
3. **Image Display**: Base64 encoded images in chat for simplicity
4. **Byaldi Index**: Stored in `/app/data/index/` volume
5. **GPT-4o Vision**: Used for answer generation from page images
