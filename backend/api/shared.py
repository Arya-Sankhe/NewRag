"""
Shared application state and singletons.

This module provides a single shared RAG system instance
to avoid loading heavy models (CLIP, embeddings) multiple times.
"""

import os
import sys

# Fix Python path for both local and Docker environments
if os.path.exists('/app/project'):
    sys.path.insert(0, '/app/project')
    sys.path.insert(0, '/app/backend')

from core.rag_system import RAGSystem
from core.document_manager import DocumentManager
from core.chat_interface import ChatInterface

# Single shared RAG system instance
_rag_system: RAGSystem = None
_doc_manager: DocumentManager = None


def get_rag_system() -> RAGSystem:
    """
    Get the shared RAG system singleton.
    
    This ensures heavy models (CLIP, embeddings) are only loaded once.
    """
    global _rag_system
    if _rag_system is None:
        print("🔧 Initializing shared RAG system...")
        _rag_system = RAGSystem()
        _rag_system.initialize()
        print("✓ RAG system ready")
    return _rag_system


def get_document_manager() -> DocumentManager:
    """Get the shared document manager singleton."""
    global _doc_manager
    if _doc_manager is None:
        _doc_manager = DocumentManager(get_rag_system())
    return _doc_manager


def create_chat_interface() -> ChatInterface:
    """
    Create a new chat interface using the shared RAG system.
    
    Each chat session gets its own ChatInterface but shares
    the underlying RAG system (and its loaded models).
    """
    return ChatInterface(get_rag_system())
