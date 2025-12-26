import uuid
import os
import config
from langchain_openai import ChatOpenAI
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph


def get_llm():
    """Get OpenAI LLM instance."""
    api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in config.py "
            "or as an environment variable."
        )
    
    print(f"🤖 Using OpenAI LLM: {config.OPENAI_LLM_MODEL}")
    return ChatOpenAI(
        model=config.OPENAI_LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=api_key
    )


class RAGSystem:
    
    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        
    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)
        
        llm = get_llm()
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)
        
    def get_config(self):
        return {"configurable": {"thread_id": self.thread_id}}
    
    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())