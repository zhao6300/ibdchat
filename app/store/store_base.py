from typing import List
from langchain.schema import Document

class StoreBase:
    def add_documents(self, documents: List[Document]):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    
    
    def retrieve(self, query: str, k: int = 2) -> List[Document]:
        raise NotImplementedError("This method should be overridden by subclasses.")