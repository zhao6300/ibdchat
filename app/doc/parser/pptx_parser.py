
from .base_parser import BaseParser

class PptxParser(BaseParser):
    def __init__(self, file_path: str):
        self.file_path = file_path