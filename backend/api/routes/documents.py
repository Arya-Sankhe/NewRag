"""
Document management API routes.

Provides endpoints for:
- Listing indexed documents
- Uploading and indexing new documents (PDF, Markdown)
- Clearing all documents
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'project'))

from api.models.responses import (
    DocumentListResponse, 
    DocumentInfo, 
    UploadResultResponse,
    ClearResponse,
    ErrorResponse
)
from core.rag_system import RAGSystem
from core.document_manager import DocumentManager

router = APIRouter()

# Global RAG system instance (initialized once)
_rag_system = None
_doc_manager = None


def get_rag_system() -> RAGSystem:
    """Get or initialize the RAG system singleton."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
        _rag_system.initialize()
    return _rag_system


def get_document_manager() -> DocumentManager:
    """Get or initialize the document manager singleton."""
    global _doc_manager
    if _doc_manager is None:
        _doc_manager = DocumentManager(get_rag_system())
    return _doc_manager


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """
    List all indexed documents in the knowledge base.
    
    Returns a list of document names and the total count.
    """
    try:
        doc_manager = get_document_manager()
        files = doc_manager.get_markdown_files()
        
        documents = [DocumentInfo(name=f) for f in files]
        
        return DocumentListResponse(
            documents=documents,
            count=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResultResponse)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF or Markdown files to upload"),
    enable_vlm: bool = Form(False, description="Enable VLM for enhanced image captions")
):
    """
    Upload and index documents into the knowledge base.
    
    Accepts PDF and Markdown files. PDFs are processed with OCR and image extraction.
    Set enable_vlm=True for AI-generated image captions (slower, uses OpenAI API).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Filter valid file types
    valid_extensions = {'.pdf', '.md'}
    valid_files = []
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext in valid_extensions:
            valid_files.append(file)
    
    if not valid_files:
        raise HTTPException(
            status_code=400, 
            detail="No valid files provided. Accepted formats: PDF, Markdown"
        )
    
    try:
        doc_manager = get_document_manager()
        
        # Save uploaded files to temp directory
        temp_paths = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            for file in valid_files:
                temp_path = os.path.join(temp_dir, file.filename)
                with open(temp_path, 'wb') as f:
                    content = await file.read()
                    f.write(content)
                temp_paths.append(temp_path)
            
            # Process documents
            added, skipped = doc_manager.add_documents(
                temp_paths, 
                enable_vlm=enable_vlm
            )
            
            vlm_status = "with VLM captions" if enable_vlm else "without VLM"
            
            return UploadResultResponse(
                added=added,
                skipped=skipped,
                vlm_enabled=enable_vlm,
                message=f"Successfully processed {added + skipped} documents ({vlm_status}). Added: {added}, Skipped: {skipped}"
            )
            
        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", response_model=ClearResponse)
async def clear_all_documents():
    """
    Clear all documents from the knowledge base.
    
    This removes all indexed documents, vector embeddings, and parent chunks.
    Use with caution - this action cannot be undone.
    """
    try:
        doc_manager = get_document_manager()
        doc_manager.clear_all()
        
        return ClearResponse(
            success=True,
            message="All documents cleared from knowledge base"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh", response_model=DocumentListResponse)
async def refresh_document_list():
    """
    Refresh and return the current document list.
    
    Same as GET /documents but as POST for explicit refresh semantics.
    """
    return await list_documents()
