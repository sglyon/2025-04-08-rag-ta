import getpass
import os

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

# Get OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Default embeddings
default_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def initialize_vector_store(
    embedding_function: Embeddings = default_embeddings,
    collection_name: str = "rag_ta_collection",
    persist_directory: str = "./chroma_db",
) -> Chroma:
    """
    Initializes a ChromaDB vector store.

    Args:
        embedding_function: The embedding function to use for the vector store.
        collection_name: The name of the collection within the vector store.
        persist_directory: The directory where the vector store data will be persisted.

    Returns:
        An initialized Chroma vector store instance.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
    return vector_store
