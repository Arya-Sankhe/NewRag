/**
 * Shared TypeScript types for the frontend.
 */

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

export interface Document {
    name: string;
    indexedAt?: Date;
}

export interface UploadProgress {
    progress: number;
    currentFile: string;
    status: string;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
