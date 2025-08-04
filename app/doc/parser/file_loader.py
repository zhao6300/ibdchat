from langchain_core.documents import BaseDocumentTransformer, Document
from typing import Optional, List


class FileLoader:
    def __init__(self):
        pass

    def load_file(self) -> str:
        """
        Load the file content.
        """
        raise NotImplementedError("Subclasses should implement this method.")
