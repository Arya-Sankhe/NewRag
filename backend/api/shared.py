"""
Shared application state and singletons.

This module provides a single shared RAG system instance
to avoid loading heavy models (CLIP, embeddings) multiple times.
"""

import os
import sys

# Fix Python path for both local and Docker environments
_current_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.exists('/app/project'):
    # Docker environment
    if '/app/project' not in sys.path:
        sys.path.insert(0, '/app/project')
    if '/app/backend' not in sys.path:
        sys.path.insert(0, '/app/backend')
else:
    # Local development
    _project_path = os.path.join(_current_dir, '..', '..', '..', 'project')
    _backend_path = os.path.join(_current_dir, '..', '..')
    if os.path.abspath(_project_path) not in sys.path:
        sys.path.insert(0, os.path.abspath(_project_path))
    if os.path.abspath(_backend_path) not in sys.path:
        sys.path.insert(0, os.path.abspath(_backend_path))

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
