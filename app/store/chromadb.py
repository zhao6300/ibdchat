from .store_base import StoreBase
import os
from typing import List
from dotenv import load_dotenv, find_dotenv
from app.models import *
from langchain_chroma import Chroma
from langchain.schema import Document


load_dotenv(find_dotenv())

class ChromaDB(StoreBase):
    def __init__(self, collection_name: str = "test_db"):
        self.collection_name = collection_name

        model_type = os.getenv("EMBEDDING_MODEL_TYPE")
        model_name = os.getenv("EMBEDDING_MODEL_NAME")
        model_key = os.getenv("EMBEDDING_MODEL_API_KEY")
        modal_base_url = os.getenv("EMBEDDING_MODEL_BASE_URL")
        self.embedding = EmbeddingModel.get(model_type)(
            model_key, model_name, modal_base_url)
        self.db = Chroma(collection_name=collection_name,
                         embedding_function=self.embedding)

    def add_documents(self, documents: List[Document]):
        """Add documents to the ChromaDB collection."""
        return self.db.add_documents(documents)

    def retrieve(self, query: str, k: int = 2) -> List[Document]:
        """Retrieve documents based on a query."""
        return self.db.similarity_search(query, k=k)