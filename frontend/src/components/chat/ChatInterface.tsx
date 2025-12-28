'use client';

import { useCallback, useEffect, useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { createChatWebSocket, sendChatMessage, ChatStreamMessage } from '@/lib/api';
import { Message, ConnectionStatus } from '@/lib/types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
    const [useHttpFallback, setUseHttpFallback] = useState(false);
    const [threadId, setThreadId] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const responseRef = useRef<string>('');
    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = 2;

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    // Connect WebSocket with fallback
    const connect = useCallback(() => {
        if (useHttpFallback) {
            setConnectionStatus('connected');
            return;
        }

        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        setConnectionStatus('connecting');

        wsRef.current = createChatWebSocket(
            threadId || undefined,
            (data: ChatStreamMessage) => {
                switch (data.type) {
                    case 'session':
                        setThreadId(data.thread_id || null);
                        reconnectAttempts.current = 0;
                        break;
                    case 'token':
                        responseRef.current += data.content || '';
                        setMessages(prev => {
                            const lastMessage = prev[prev.length - 1];
                            if (lastMessage?.role === 'assistant') {
                                return [
                                    ...prev.slice(0, -1),
                                    { ...lastMessage, content: responseRef.current }
                                ];
                            } else {
                                return [
                                    ...prev,
                                    {
                                        id: crypto.randomUUID(),
                                        role: 'assistant',
                                        content: responseRef.current,
                                        timestamp: new Date()
                                    }
                                ];
                            }
                        });
                        break;
                    case 'done':
                        setIsLoading(false);
                        responseRef.current = '';
                        break;
                    case 'error':
                        setIsLoading(false);
                        setMessages(prev => [
                            ...prev,
                            {
                                id: crypto.randomUUID(),
                                role: 'assistant',
                                content: `❌ Error: ${data.content}`,
                                timestamp: new Date()
                            }
                        ]);
                        responseRef.current = '';
                        break;
                    case 'cleared':
                        setMessages([]);
                        responseRef.current = '';
                        break;
                }
            },
            () => {
                setConnectionStatus('connected');
                reconnectAttempts.current = 0;
            },
            () => {
                setConnectionStatus('disconnected');
            },
            () => {
                // WebSocket error - try fallback after max attempts
                reconnectAttempts.current += 1;
                if (reconnectAttempts.current >= maxReconnectAttempts) {
                    console.log('WebSocket failed, switching to HTTP fallback mode');
                    setUseHttpFallback(true);
                    setConnectionStatus('connected');
                } else {
                    setConnectionStatus('error');
                    // Try to reconnect after a short delay
                    setTimeout(() => connect(), 1000);
                }
            }
        );
    }, [threadId, useHttpFallback]);

    // Connect on mount
    useEffect(() => {
        connect();
        return () => {
            wsRef.current?.close();
        };
    }, [connect]);

    // Send message via HTTP fallback
    const sendMessageHttp = useCallback(async (messageText: string) => {
        setIsLoading(true);
        try {
            const response = await sendChatMessage(messageText, threadId || undefined);
            setThreadId(response.thread_id);
            setMessages(prev => [
                ...prev,
                {
                    id: crypto.randomUUID(),
                    role: 'assistant',
                    content: response.response,
                    timestamp: new Date()
                }
            ]);
        } catch (error) {
            setMessages(prev => [
                ...prev,
                {
                    id: crypto.randomUUID(),
                    role: 'assistant',
                    content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                    timestamp: new Date()
                }
            ]);
        } finally {
            setIsLoading(false);
        }
    }, [threadId]);

    // Send message
    const sendMessage = useCallback(() => {
        if (!input.trim() || isLoading) return;
        if (!useHttpFallback && connectionStatus !== 'connected') return;

        const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');

        if (useHttpFallback) {
            sendMessageHttp(userMessage.content);
        } else {
            setIsLoading(true);
            responseRef.current = '';
            wsRef.current?.send(JSON.stringify({ message: userMessage.content }));
        }
    }, [input, isLoading, connectionStatus, useHttpFallback, sendMessageHttp]);

    // Clear session
    const clearSession = useCallback(() => {
        setMessages([]);
        setThreadId(null);
        if (!useHttpFallback && wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ message: '__clear__' }));
        }
    }, [useHttpFallback]);

    // Handle Enter key
    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const statusColor = {
        connecting: 'bg-yellow-500',
        connected: 'bg-green-500',
        disconnected: 'bg-gray-500',
        error: 'bg-red-500'
    };

    const statusText = useHttpFallback
        ? 'HTTP Mode'
        : connectionStatus;

    const canSend = useHttpFallback
        ? !isLoading
        : (connectionStatus === 'connected' && !isLoading);

    return (
        <div className="flex flex-col h-[calc(100vh-200px)] min-h-[500px]">
            {/* Header */}
            <div className="flex items-center justify-between pb-4">
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${useHttpFallback ? 'bg-blue-500' : statusColor[connectionStatus]}`} />
                    <span className="text-sm text-muted-foreground capitalize">{statusText}</span>
                    {useHttpFallback && (
                        <span className="text-xs text-muted-foreground">(WebSocket unavailable)</span>
                    )}
                </div>
                <Button variant="outline" size="sm" onClick={clearSession}>
                    Clear Chat
                </Button>
            </div>

            <Separator className="mb-4" />

            {/* Messages */}
            <ScrollArea className="flex-1 pr-4" ref={scrollRef}>
                {messages.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-muted-foreground">
                        <p>Ask me anything about your documents!</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {messages.map((message) => (
                            <div
                                key={message.id}
                                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-[80%] rounded-lg px-4 py-2 ${message.role === 'user'
                                        ? 'bg-primary text-primary-foreground'
                                        : 'bg-muted'
                                        }`}
                                >
                                    {message.role === 'assistant' ? (
                                        <ReactMarkdown
                                            remarkPlugins={[remarkGfm]}
                                            components={{
                                                // eslint-disable-next-line @next/next/no-img-element
                                                img: ({ src, alt }) => (
                                                    <img
                                                        src={src}
                                                        alt={alt || 'Image'}
                                                        className="max-w-full h-auto rounded-md my-2"
                                                        style={{ maxHeight: '400px' }}
                                                    />
                                                ),
                                                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                                                code: ({ children }) => (
                                                    <code className="bg-background/50 px-1 py-0.5 rounded text-sm">
                                                        {children}
                                                    </code>
                                                ),
                                            }}
                                        >
                                            {message.content}
                                        </ReactMarkdown>
                                    ) : (
                                        <p>{message.content}</p>
                                    )}
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="bg-muted rounded-lg px-4 py-2">
                                    <div className="flex gap-1">
                                        <span className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                        <span className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                        <span className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </ScrollArea>

            <Separator className="my-4" />

            {/* Input */}
            <div className="flex gap-2">
                <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your message..."
                    className="min-h-[60px] resize-none"
                    disabled={!canSend && !useHttpFallback}
                />
                <Button
                    onClick={sendMessage}
                    disabled={!input.trim() || !canSend}
                    className="h-auto"
                >
                    Send
                </Button>
            </div>
        </div>
    );
}
