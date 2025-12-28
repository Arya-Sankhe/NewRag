"""
Chat API routes with WebSocket streaming support.

Provides endpoints for:
- WebSocket streaming chat (primary)
- HTTP synchronous chat (fallback)
- Session management
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
import os
import sys
import json
import asyncio
import uuid

# Add project directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'project'))

from api.models.requests import ChatMessageRequest, ClearSessionRequest
from api.models.responses import ChatMessageResponse, ClearResponse, ChatStreamToken
from core.rag_system import RAGSystem
from core.chat_interface import ChatInterface

router = APIRouter()

# Store active RAG systems per session
_sessions: dict[str, tuple[RAGSystem, ChatInterface]] = {}


def get_or_create_session(thread_id: Optional[str] = None) -> tuple[str, RAGSystem, ChatInterface]:
    """
    Get or create a chat session.
    
    Args:
        thread_id: Optional existing thread ID
        
    Returns:
        Tuple of (thread_id, rag_system, chat_interface)
    """
    global _sessions
    
    if thread_id and thread_id in _sessions:
        rag_system, chat_interface = _sessions[thread_id]
        return thread_id, rag_system, chat_interface
    
    # Create new session
    new_thread_id = thread_id or str(uuid.uuid4())
    rag_system = RAGSystem()
    rag_system.initialize()
    rag_system.thread_id = new_thread_id
    chat_interface = ChatInterface(rag_system)
    
    _sessions[new_thread_id] = (rag_system, chat_interface)
    
    return new_thread_id, rag_system, chat_interface


def clear_session(thread_id: str) -> bool:
    """Clear a specific session."""
    global _sessions
    
    if thread_id in _sessions:
        rag_system, chat_interface = _sessions[thread_id]
        chat_interface.clear_session()
        del _sessions[thread_id]
        return True
    return False


@router.websocket("/stream")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.
    
    Protocol:
    1. Connect with optional query param: ?thread_id=xxx
    2. Send JSON messages: {"message": "user query"}
    3. Receive JSON tokens: {"type": "token|done|error|images", "content": "..."}
    
    Token types:
    - "token": Partial response token
    - "done": Response complete
    - "error": Error occurred
    - "images": Image data (sent after "done")
    """
    await websocket.accept()
    
    # Get thread_id from query params or create new
    thread_id = websocket.query_params.get("thread_id")
    thread_id, rag_system, chat_interface = get_or_create_session(thread_id)
    
    # Send session info
    await websocket.send_json({
        "type": "session",
        "thread_id": thread_id
    })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "").strip()
            
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "content": "Empty message"
                })
                continue
            
            # Check for special commands
            if message == "__clear__":
                clear_session(thread_id)
                thread_id, rag_system, chat_interface = get_or_create_session()
                await websocket.send_json({
                    "type": "session",
                    "thread_id": thread_id
                })
                await websocket.send_json({
                    "type": "cleared",
                    "content": "Session cleared"
                })
                continue
            
            try:
                # Process chat message
                # Note: Current implementation is synchronous
                # For true streaming, we'd need to modify ChatInterface
                # to yield tokens. For now, we send the full response.
                response = chat_interface.chat(message, [])
                
                # Send response (as single message for now)
                # TODO: Implement true token streaming with LangGraph callbacks
                await websocket.send_json({
                    "type": "token",
                    "content": response
                })
                
                await websocket.send_json({
                    "type": "done",
                    "content": ""
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {thread_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(request: ChatMessageRequest):
    """
    Send a chat message and receive a response (synchronous).
    
    This is a fallback endpoint for testing or clients that don't support WebSocket.
    For real-time streaming, use the WebSocket endpoint at /api/v1/chat/stream.
    """
    try:
        thread_id, rag_system, chat_interface = get_or_create_session(request.thread_id)
        
        # Run synchronous chat in thread pool to not block
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chat_interface.chat(request.message, [])
        )
        
        # Check if response contains images
        has_images = "📸 Related Images:" in response
        
        return ChatMessageResponse(
            response=response,
            thread_id=thread_id,
            has_images=has_images
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=ClearResponse)
async def clear_chat_session(request: ClearSessionRequest):
    """
    Clear a chat session.
    
    This resets the conversation history and starts a fresh session.
    """
    success = clear_session(request.thread_id)
    
    if success:
        return ClearResponse(
            success=True,
            message=f"Session {request.thread_id} cleared"
        )
    else:
        return ClearResponse(
            success=False,
            message=f"Session {request.thread_id} not found"
        )


@router.get("/session")
async def create_new_session():
    """
    Create a new chat session and return the thread ID.
    
    Use this to get a fresh session before connecting via WebSocket.
    """
    thread_id, _, _ = get_or_create_session()
    return {"thread_id": thread_id}
