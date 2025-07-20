
from .base_parser import BaseParser


class PicParser(BaseParser):
    def __init__(self, file_path: str):
        self.file_path = file_path
