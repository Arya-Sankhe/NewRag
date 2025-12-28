/**
 * API client for backend communication.
 * 
 * Uses window.location to determine the backend URL dynamically,
 * so it works on localhost, VPS, or any other deployment.
 */

// Dynamically determine API URL based on current page location
function getApiUrl(): string {
    if (typeof window === 'undefined') {
        // Server-side: use environment variable or default
        return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    }

    // Client-side: use same host, different port
    const host = window.location.hostname;
    const protocol = window.location.protocol;

    // If there's an explicit API URL set, use it
    if (process.env.NEXT_PUBLIC_API_URL && process.env.NEXT_PUBLIC_API_URL !== 'http://localhost:8000') {
        return process.env.NEXT_PUBLIC_API_URL;
    }

    // Otherwise, use the same host with port 8000
    return `${protocol}//${host}:8000`;
}

function getWsUrl(): string {
    if (typeof window === 'undefined') {
        return process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
    }

    const host = window.location.hostname;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

    if (process.env.NEXT_PUBLIC_WS_URL && process.env.NEXT_PUBLIC_WS_URL !== 'ws://localhost:8000') {
        return process.env.NEXT_PUBLIC_WS_URL;
    }

    return `${protocol}//${host}:8000`;
}

/**
 * Fetch documents list from the backend.
 */
export async function getDocuments(): Promise<DocumentListResponse> {
    const API_URL = getApiUrl();
    const response = await fetch(`${API_URL}/api/v1/documents`);
    if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.statusText}`);
    }
    return response.json();
}

/**
 * Upload documents to the backend.
 */
export async function uploadDocuments(
    files: File[],
    enableVlm: boolean = false,
    onProgress?: (progress: number) => void
): Promise<UploadResultResponse> {
    const API_URL = getApiUrl();
    const formData = new FormData();

    files.forEach(file => {
        formData.append('files', file);
    });
    formData.append('enable_vlm', enableVlm.toString());

    const response = await fetch(`${API_URL}/api/v1/documents/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
}

/**
 * Clear all documents from the knowledge base.
 */
export async function clearDocuments(): Promise<ClearResponse> {
    const API_URL = getApiUrl();
    const response = await fetch(`${API_URL}/api/v1/documents/clear`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        throw new Error(`Failed to clear documents: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Send a chat message via HTTP (fallback for non-WebSocket clients).
 */
export async function sendChatMessage(
    message: string,
    threadId?: string
): Promise<ChatMessageResponse> {
    const API_URL = getApiUrl();
    const response = await fetch(`${API_URL}/api/v1/chat/message`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message,
            thread_id: threadId,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Chat request failed');
    }

    return response.json();
}

/**
 * Create a WebSocket connection for streaming chat.
 */
export function createChatWebSocket(
    threadId?: string,
    onMessage?: (data: ChatStreamMessage) => void,
    onOpen?: () => void,
    onClose?: () => void,
    onError?: (error: Event) => void
): WebSocket {
    const WS_URL = getWsUrl();
    const url = threadId
        ? `${WS_URL}/api/v1/chat/stream?thread_id=${threadId}`
        : `${WS_URL}/api/v1/chat/stream`;

    console.log('Connecting to WebSocket:', url);
    const ws = new WebSocket(url);

    ws.onopen = () => {
        console.log('WebSocket connected');
        onOpen?.();
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data) as ChatStreamMessage;
            onMessage?.(data);
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        onClose?.();
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
    };

    return ws;
}

/**
 * Get a new session ID.
 */
export async function createSession(): Promise<{ thread_id: string }> {
    const API_URL = getApiUrl();
    const response = await fetch(`${API_URL}/api/v1/chat/session`);
    if (!response.ok) {
        throw new Error(`Failed to create session: ${response.statusText}`);
    }
    return response.json();
}

// Types
export interface DocumentInfo {
    name: string;
    indexed_at?: string;
}

export interface DocumentListResponse {
    documents: DocumentInfo[];
    count: number;
}

export interface UploadResultResponse {
    added: number;
    skipped: number;
    vlm_enabled: boolean;
    message: string;
}

export interface ClearResponse {
    success: boolean;
    message: string;
}

export interface ChatMessageResponse {
    response: string;
    thread_id: string;
    has_images: boolean;
}

export interface ChatStreamMessage {
    type: 'session' | 'token' | 'done' | 'error' | 'images' | 'cleared';
    content?: string;
    thread_id?: string;
    images?: Array<{
        data_url: string;
        caption?: string;
        page_number?: number;
    }>;
}
