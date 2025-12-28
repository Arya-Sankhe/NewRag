"""
Pydantic models for API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatMessageResponse(BaseModel):
    """Response model for a chat message."""
    response: str = Field(..., description="The assistant's response")
    thread_id: str = Field(..., description="Session/thread ID")
    has_images: bool = Field(False, description="Whether the response contains images")


class ChatStreamToken(BaseModel):
    """Model for streaming chat tokens via WebSocket."""
    type: str = Field(..., description="Token type: 'token', 'done', 'error', 'images'")
    content: Optional[str] = Field(None, description="Token content")
    images: Optional[List[dict]] = Field(None, description="Image data if type is 'images'")


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    name: str = Field(..., description="Document filename")
    indexed_at: Optional[datetime] = Field(None, description="When the document was indexed")


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentInfo] = Field(default_factory=list, description="List of indexed documents")
    count: int = Field(0, description="Total number of documents")


class UploadProgressUpdate(BaseModel):
    """Progress update during document upload."""
    progress: float = Field(..., ge=0, le=1, description="Progress percentage (0-1)")
    current_file: str = Field(..., description="Currently processing file")
    status: str = Field(..., description="Current status message")


class UploadResultResponse(BaseModel):
    """Response model for document upload result."""
    added: int = Field(0, description="Number of documents successfully added")
    skipped: int = Field(0, description="Number of documents skipped (already indexed)")
    vlm_enabled: bool = Field(False, description="Whether VLM captions were enabled")
    message: str = Field(..., description="Result message")


class ClearResponse(BaseModel):
    """Response model for clear operations."""
    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Result message")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
