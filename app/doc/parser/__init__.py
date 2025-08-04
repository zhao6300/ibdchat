from langchain_core.documents import BaseDocumentTransformer, Document
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .pptx_parser import PptxParser
from .txt_parser import TxtParser
from .excel_parser import ExcelParser
from .pic_parser import PicParser
from .markdown_parser import MarkdownParser
from .html_parser import HtmlParser

from typing import Optional

__all__ = [
    "DocxParser",
    "PdfParser",
    "PptxParser",
    "TxtParser",
    "ExcelParser",
    "PicParser",
    "MarkdownParser",
    "HtmlParser",
]

suffixs = [
    "pdf",
    "docx",
    "pptx",
    "txt",
    "excel",
    "pic",
    "markdown",
    "html"
]

parsers = {
    "pdf": PdfParser(),
    "docx": DocxParser(),
    "pptx": PptxParser(),
    "txt": TxtParser(),
    "excel": ExcelParser(),
    "pic": PicParser(),
    "markdown": MarkdownParser(),
    "html": HtmlParser(),
}


def parse_document(file_path: str, file_type: Optional[str] = None) -> Document:
    """
    Parse a document based on its file type.

    Args:
        file_path (str): The path to the document file.
        file_type (str): The type of the document (e.g., 'pdf', 'docx', 'txt').

    Returns:
        Document: The parsed document.
    """

    suffix = file_path.split('.')[-1]
    if not file_type:
        file_type = suffix
    file_type = file_type.lower()
    parser = parsers.get(file_type.lower())
    if not parser:
        raise ValueError(f"Unsupported file type: {file_type}")

    return parser.split_documents(file_path)
