"""
Модуль загрузки данных: структура «папка = класс», чтение документов (.txt, .docx, .pdf).
"""

from .data_loader import (
    extract_text_from_docx,
    extract_text_from_pdf,
    load_corpus_from_class_folders,
    read_txt_document,
)
from .document_text import read_text_from_document, supported_document_extensions
from .hf_loader import load_corpus_from_huggingface

__all__ = [
    "load_corpus_from_class_folders",
    "load_corpus_from_huggingface",
    "read_txt_document",
    "read_text_from_document",
    "supported_document_extensions",
    "extract_text_from_docx",
    "extract_text_from_pdf",
]
