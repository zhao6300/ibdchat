from .base_parser import BaseParser

import pandas as pd


class ExcelParser(BaseParser):
    def __init__(self, file_path: str):
        self.file_path = file_path
