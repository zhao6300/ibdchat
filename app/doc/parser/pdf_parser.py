from .chunk import ChunkBase
from .base_parser import BaseParser
from typing import List, Literal, Optional, Dict


class PDFChunk(ChunkBase):
    def __init__(self, text: str):
        self.text = text


class PdfParser(BaseParser):
    def __init__(self, file_path: str):
        pass

    def parse(self) -> List[PDFChunk]:
        pass
