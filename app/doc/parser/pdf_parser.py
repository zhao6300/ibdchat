from .chunk import ChunkBase
from .base_parser import BaseParser
from typing import List, Literal, Optional, Dict
from langchain_core.documents import BaseDocumentTransformer, Document


class PDFChunk(ChunkBase):
    def __init__(self, text: str):
        self.text = text


class PdfParser(BaseParser):
    def __init__(self, file_path: str):
        pass

    def split_documents(self) -> List[PDFChunk]:
        """
        Parses the PDF file and returns a list of PDFChunk items.

        Returns:
            List[PDFChunk]: A list of parsed chunks from the PDF file.
        """
        # Implementation for parsing the PDF file goes here
        # For now, returning an empty list as a placeholder
        return []
