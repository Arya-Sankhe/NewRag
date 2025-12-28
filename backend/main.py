# FastAPI Backend for RAG System

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Fix Python path for both local and Docker environments
# PYTHONPATH is set in Dockerfile, but also set here for local development
_current_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.exists('/app/project'):
    # Docker environment - PYTHONPATH already set, but ensure it's in sys.path
    if '/app/project' not in sys.path:
        sys.path.insert(0, '/app/project')
    if '/app/backend' not in sys.path:
        sys.path.insert(0, '/app/backend')
else:
    # Local development
    project_path = os.path.abspath(os.path.join(_current_dir, '..', 'project'))
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)

# Now import routes - they will use the paths we just set up
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
    allow_origins=["*"],  # Allow all origins for development
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
    print("🚀 RAG System API started successfully!")
    print("📚 API Documentation: http://localhost:8000/docs")


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
