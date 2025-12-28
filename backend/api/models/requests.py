"""
Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message."""
    message: str = Field(..., min_length=1, description="The user's message")
    thread_id: Optional[str] = Field(None, description="Session/thread ID for conversation continuity")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload configuration."""
    enable_vlm: bool = Field(False, description="Enable VLM for enhanced image captions")


class ClearSessionRequest(BaseModel):
    """Request model for clearing a chat session."""
    thread_id: str = Field(..., description="Session/thread ID to clear")
