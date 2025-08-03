from langchain_core.documents import BaseDocumentTransformer, Document

class FileLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_file(self) -> str:
        """
        Load the file content.
        """
        raise NotImplementedError("Subclasses should implement this method.")
