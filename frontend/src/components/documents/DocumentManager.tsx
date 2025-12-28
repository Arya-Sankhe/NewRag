'use client';

import { useCallback, useEffect, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
    getDocuments,
    uploadDocuments,
    clearDocuments,
    DocumentInfo
} from '@/lib/api';

export function DocumentManager() {
    const [documents, setDocuments] = useState<DocumentInfo[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [enableVlm, setEnableVlm] = useState(false);
    const [uploadMessage, setUploadMessage] = useState<string | null>(null);
    const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

    // Fetch documents on mount
    const fetchDocuments = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await getDocuments();
            setDocuments(response.documents);
        } catch (error) {
            console.error('Failed to fetch documents:', error);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchDocuments();
    }, [fetchDocuments]);

    // Dropzone configuration
    const onDrop = useCallback((acceptedFiles: File[]) => {
        setSelectedFiles(prev => [...prev, ...acceptedFiles]);
        setUploadMessage(null);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'text/markdown': ['.md'],
        },
        multiple: true,
    });

    // Upload files
    const handleUpload = async () => {
        if (selectedFiles.length === 0) return;

        setIsUploading(true);
        setUploadMessage(null);

        try {
            const result = await uploadDocuments(selectedFiles, enableVlm);
            setUploadMessage(result.message);
            setSelectedFiles([]);
            await fetchDocuments();
        } catch (error) {
            setUploadMessage(`❌ Error: ${error instanceof Error ? error.message : 'Upload failed'}`);
        } finally {
            setIsUploading(false);
        }
    };

    // Clear all documents
    const handleClear = async () => {
        if (!confirm('Are you sure you want to clear all documents? This cannot be undone.')) {
            return;
        }

        setIsLoading(true);
        try {
            await clearDocuments();
            setDocuments([]);
            setUploadMessage('🗑️ All documents cleared');
        } catch (error) {
            setUploadMessage(`❌ Error: ${error instanceof Error ? error.message : 'Clear failed'}`);
        } finally {
            setIsLoading(false);
        }
    };

    // Remove file from selection
    const removeFile = (index: number) => {
        setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    };

    return (
        <div className="space-y-6">
            {/* Upload Section */}
            <Card>
                <CardHeader>
                    <CardTitle>Add New Documents</CardTitle>
                    <CardDescription>
                        Upload PDF or Markdown files. PDFs are processed with OCR and image extraction.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {/* Dropzone */}
                    <div
                        {...getRootProps()}
                        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive
                                ? 'border-primary bg-primary/5'
                                : 'border-muted-foreground/25 hover:border-primary/50'
                            }`}
                    >
                        <input {...getInputProps()} />
                        {isDragActive ? (
                            <p className="text-primary">Drop the files here...</p>
                        ) : (
                            <div className="space-y-2">
                                <p className="text-muted-foreground">
                                    Drag & drop files here, or click to select
                                </p>
                                <p className="text-sm text-muted-foreground/75">
                                    Accepts: PDF, Markdown
                                </p>
                            </div>
                        )}
                    </div>

                    {/* Selected Files */}
                    {selectedFiles.length > 0 && (
                        <div className="space-y-2">
                            <p className="text-sm font-medium">Selected files:</p>
                            <div className="flex flex-wrap gap-2">
                                {selectedFiles.map((file, index) => (
                                    <Badge
                                        key={index}
                                        variant="secondary"
                                        className="cursor-pointer hover:bg-destructive hover:text-destructive-foreground"
                                        onClick={() => removeFile(index)}
                                    >
                                        {file.name} ✕
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* VLM Toggle */}
                    <div className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            id="vlm-toggle"
                            checked={enableVlm}
                            onChange={(e) => setEnableVlm(e.target.checked)}
                            className="rounded"
                        />
                        <label htmlFor="vlm-toggle" className="text-sm">
                            🧠 Enable VLM Captions (uses AI vision for detailed image descriptions)
                        </label>
                    </div>

                    {/* Upload Button */}
                    <Button
                        onClick={handleUpload}
                        disabled={selectedFiles.length === 0 || isUploading}
                        className="w-full"
                    >
                        {isUploading ? 'Processing...' : 'Add Documents'}
                    </Button>

                    {/* Upload Message */}
                    {uploadMessage && (
                        <p className={`text-sm ${uploadMessage.startsWith('❌') ? 'text-destructive' : 'text-muted-foreground'}`}>
                            {uploadMessage}
                        </p>
                    )}
                </CardContent>
            </Card>

            {/* Document List Section */}
            <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                        <CardTitle>Knowledge Base</CardTitle>
                        <CardDescription>
                            {documents.length} document{documents.length !== 1 ? 's' : ''} indexed
                        </CardDescription>
                    </div>
                    <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={fetchDocuments} disabled={isLoading}>
                            {isLoading ? 'Loading...' : 'Refresh'}
                        </Button>
                        <Button
                            variant="destructive"
                            size="sm"
                            onClick={handleClear}
                            disabled={documents.length === 0 || isLoading}
                        >
                            Clear All
                        </Button>
                    </div>
                </CardHeader>
                <CardContent>
                    {documents.length === 0 ? (
                        <p className="text-muted-foreground text-center py-8">
                            📭 No documents in the knowledge base
                        </p>
                    ) : (
                        <ScrollArea className="h-[200px]">
                            <div className="space-y-2">
                                {documents.map((doc, index) => (
                                    <div
                                        key={index}
                                        className="flex items-center justify-between p-2 rounded bg-muted/50"
                                    >
                                        <span className="text-sm">{doc.name}</span>
                                        <Badge variant="outline" className="text-xs">PDF</Badge>
                                    </div>
                                ))}
                            </div>
                        </ScrollArea>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
