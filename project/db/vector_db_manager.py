import os
import config
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


def get_openai_embeddings():
    """Get OpenAI embeddings with API key from config or environment."""
    api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in config.py "
            "or as an environment variable."
        )
    return OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        api_key=api_key
    )


class VectorDbManager:
    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__embeddings = get_openai_embeddings()
        
        # Get embedding dimension by testing
        test_embedding = self.__embeddings.embed_query("test")
        self.__embedding_dim = len(test_embedding)
        print(f"🔗 Using OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL} (dim={self.__embedding_dim})")

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.__embedding_dim, 
                    distance=qmodels.Distance.COSINE
                ),
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__embeddings,
                retrieval_mode=RetrievalMode.DENSE,  # OpenAI embeddings are dense only
            )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")
            raise