import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { DocumentManager } from "@/components/documents/DocumentManager";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            RAG Assistant
          </h1>
          <p className="text-muted-foreground mt-2">
            Upload documents and chat with your knowledge base
          </p>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="chat" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="chat" className="text-lg">
              💬 Chat
            </TabsTrigger>
            <TabsTrigger value="documents" className="text-lg">
              📁 Documents
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chat" className="mt-0">
            <ChatInterface />
          </TabsContent>

          <TabsContent value="documents" className="mt-0">
            <DocumentManager />
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <footer className="mt-12 text-center text-sm text-muted-foreground">
          <p>Powered by LangGraph + OpenAI + Qdrant</p>
        </footer>
      </main>
    </div>
  );
}
