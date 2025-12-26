# --- Directory Configuration ---
MARKDOWN_DIR = "markdown_docs"
PARENT_STORE_PATH = "parent_store"
QDRANT_DB_PATH = "qdrant_db"

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"

# --- LLM Provider Configuration ---
# Set USE_OPENAI = True to use OpenAI, False for Ollama (local)
USE_OPENAI = True

# OpenAI Configuration
OPENAI_API_KEY = ""  # Set your API key here or use environment variable
OPENAI_MODEL = "gpt-4o-mini"  # Options: gpt-4o-mini, gpt-4o, gpt-4-turbo

# Ollama Configuration (local, heavier but free)
OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# Shared LLM settings  
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

# --- Docling Parser Configuration ---
ENABLE_DOCLING = True
DOCLING_OCR_ENABLED = True
DOCLING_IMAGE_SCALE = 2.0
DOCLING_GENERATE_CAPTIONS = False  # Requires VLM model, set True if available
