from .chunk import ChunkBase
from .base_parser import BaseParser
from typing import List, Literal, Optional, Dict

from .utils import find_codec, get_text
import re


class TxtChunk(ChunkBase):
    def __init__(self, text: str):
        self.text = text


class TxtParser(BaseParser):
    def __init__(self, file_path: str):
        pass

    def parse(self) -> List[TxtChunk]:
        pass
