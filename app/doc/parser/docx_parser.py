from .chunk import ChunkBase
from .base_parser import BaseParser
from typing import List, Literal, Optional, Dict


class DocxChunk(ChunkBase):
    def __init__(self, text: str):
        self.text = text

class DocxParser(BaseParser):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse(self) -> List[DocxChunk]:
        pass
