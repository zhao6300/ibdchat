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

    def split_documents(self) -> List[TxtChunk]:
        """
        Parses the TXT file and returns a list of TxtChunk items.

        Returns:
            List[TxtChunk]: A list of parsed chunks from the TXT file.
        """
        # Implementation for parsing the TXT file goes here
        # For now, returning an empty list as a placeholder
        return []
