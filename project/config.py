import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Directory Configuration ---
MARKDOWN_DIR = "markdown_docs"
PARENT_STORE_PATH = "parent_store"
QDRANT_DB_PATH = "qdrant_db"

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- OpenAI Configuration ---
# API key loaded from .env file (OPENAI_API_KEY=sk-...)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM Model for queries
OPENAI_LLM_MODEL = "gpt-4o-mini"  # Options: gpt-4o-mini, gpt-4o, gpt-4-turbo

# Embedding model for document indexing
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # Options: text-embedding-3-small, text-embedding-3-large

# Temperature for responses (0 = deterministic, 1 = creative)
LLM_TEMPERATURE = 0

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 10000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- PDF Parser Configuration ---
# Docling for advanced OCR and image extraction
ENABLE_DOCLING = True
DOCLING_OCR_ENABLED = True
DOCLING_IMAGE_SCALE = 2.0
DOCLING_GENERATE_CAPTIONS = False
