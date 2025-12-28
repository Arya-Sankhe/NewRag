# FastAPI Backend for RAG System

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Fix Python path for both local and Docker environments
if os.path.exists('/app/project'):
    sys.path.insert(0, '/app/project')
    sys.path.insert(0, '/app/backend')
else:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_current_dir, '..', 'project'))
    sys.path.insert(0, _current_dir)

from api.routes import chat, documents

app = FastAPI(
    title="RAG System API",
    description="API for document ingestion and retrieval-augmented generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://frontend:3000",
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])


@app.on_event("startup")
async def startup_event():
    """Initialize shared resources on startup."""
    print("🚀 Starting RAG System API...")
    # Lazy initialization - models loaded on first request
    # This keeps startup fast while sharing the singleton


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    
    print(f"\n🚀 Starting FastAPI server on http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs\n")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
