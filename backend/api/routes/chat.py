"""
Chat API routes with WebSocket streaming support.

Provides endpoints for:
- WebSocket streaming chat (primary)
- HTTP synchronous chat (fallback)
- Session management
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
import asyncio
import uuid

from api.shared import get_rag_system, create_chat_interface
from api.models.requests import ChatMessageRequest, ClearSessionRequest
from api.models.responses import ChatMessageResponse, ClearResponse

router = APIRouter()

# Store chat interfaces per session (lightweight - shares RAG system)
_chat_sessions: dict[str, object] = {}


def get_or_create_chat_session(thread_id: Optional[str] = None):
    """
    Get or create a chat session.
    
    Each session gets its own ChatInterface but shares the RAG system.
    """
    global _chat_sessions
    
    if thread_id and thread_id in _chat_sessions:
        return thread_id, _chat_sessions[thread_id]
    
    # Create new session with shared RAG system
    new_thread_id = thread_id or str(uuid.uuid4())
    
    # Get shared RAG system and create new chat interface
    rag_system = get_rag_system()
    chat_interface = create_chat_interface()
    
    # Set thread ID on the shared RAG system for this session
    # Note: This is a simplification - for true multi-session, 
    # we'd need separate thread management
    
    _chat_sessions[new_thread_id] = chat_interface
    
    return new_thread_id, chat_interface


def clear_chat_session(thread_id: str) -> bool:
    """Clear a specific chat session."""
    global _chat_sessions
    
    if thread_id in _chat_sessions:
        chat_interface = _chat_sessions[thread_id]
        chat_interface.clear_session()
        del _chat_sessions[thread_id]
        return True
    return False


@router.websocket("/stream")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.
    
    Protocol:
    1. Connect with optional query param: ?thread_id=xxx
    2. Send JSON messages: {"message": "user query"}
    3. Receive JSON tokens: {"type": "token|done|error", "content": "..."}
    """
    await websocket.accept()
    
    # Get thread_id from query params or create new
    thread_id = websocket.query_params.get("thread_id")
    thread_id, chat_interface = get_or_create_chat_session(thread_id)
    
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
                clear_chat_session(thread_id)
                thread_id, chat_interface = get_or_create_chat_session()
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
                # Process chat message in thread pool (blocking operation)
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: chat_interface.chat(message, [])
                )
                
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
    
    Fallback for clients that don't support WebSocket.
    """
    try:
        thread_id, chat_interface = get_or_create_chat_session(request.thread_id)
        
        # Run synchronous chat in thread pool
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chat_interface.chat(request.message, [])
        )
        
        has_images = "📸 Related Images:" in response
        
        return ChatMessageResponse(
            response=response,
            thread_id=thread_id,
            has_images=has_images
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=ClearResponse)
async def clear_session_endpoint(request: ClearSessionRequest):
    """Clear a chat session."""
    success = clear_chat_session(request.thread_id)
    
    return ClearResponse(
        success=success,
        message=f"Session {request.thread_id} {'cleared' if success else 'not found'}"
    )


@router.get("/session")
async def create_new_session():
    """Create a new chat session and return the thread ID."""
    thread_id, _ = get_or_create_chat_session()
    return {"thread_id": thread_id}
