from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

import os

def get_secret(secret_name):
    """Retrieve the secret from environment variables."""
    secret = os.getenv(secret_name)
    if secret is None:
        raise ValueError(f"Secret {secret_name} not found in environment variables.")
    return secret


# Initialize Qdrant client and retriever
def get_retriever():
    # Get Qdrant credentials from Hugging Face environment variables
    qdrant_url = get_secret("QDRANT_URL")
    qdrant_api_key = get_secret("QDRANT_API_KEY")

    # Initialize embedding model (HuggingFace MiniLM)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Set up Qdrant client
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name="Legal Guide PK-app",  # Your existing collection name
        embeddings=embeddings                  # Use the same embedding model
    )
    
    # Return the retriever for document search
    return qdrant.as_retriever(search_kwargs={"k": 5})
