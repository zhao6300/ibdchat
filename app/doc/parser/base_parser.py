from langchain_core.documents import BaseDocumentTransformer, Document

from typing import List
class BaseParser:
    def split_documents(self)-> List[Document]:
        """
        Parses the document and returns a list of parsed items.
        
        Returns:
            list: A list of parsed items.
        """
        raise NotImplementedError("Subclasses must implement this method.")
