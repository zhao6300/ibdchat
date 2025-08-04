from .base_parser import BaseParser

from langchain_core.documents import BaseDocumentTransformer, Document
from typing import List


class PptxParser(BaseParser):
    def __init__(self):
        pass

    def split_documents(self, file_path) -> List[Document]:
        """
        Parses the PPTX file and returns a list of Document items.

        Returns:
            List[Document]: A list of parsed documents from the PPTX file.
        """
        # Implementation for parsing the PPTX file goes here
        # For now, returning an empty list as a placeholder
        return []
