"""
Извлечение текста из файлов для классификации и (при необходимости) обучения.

Поддерживаются: .txt, .docx (Word 2007+), .pdf.
Старый формат .doc (бинарный Word 97–2003) не поддерживается — сохраните как .docx.
"""

from __future__ import annotations

import os

# Кодировка для .txt
TXT_ENCODING = "utf-8"


def extract_text_from_docx(file_path: str) -> str:
    """Текст из документа Word (.docx): абзацы и ячейки таблиц."""
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError("Для .docx установите: pip install python-docx") from e

    document = Document(file_path)
    parts: list[str] = []
    for para in document.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                t = cell.text.strip()
                if t:
                    parts.append(t)
    return "\n".join(parts)


def extract_text_from_pdf(file_path: str) -> str:
    """Текст из PDF (постранично). Подходит для типовых «текстовых» PDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("Для .pdf установите: pip install pymupdf") from e

    parts: list[str] = []
    with fitz.open(file_path) as doc:
        for page in doc:
            block = page.get_text()
            if block and block.strip():
                parts.append(block)
    return "\n\n".join(parts)


def read_text_from_document(file_path: str, txt_encoding: str = TXT_ENCODING) -> str:
    """
    Читает текст из файла по расширению.

    Parameters
    ----------
    file_path : str
        Путь к .txt, .docx или .pdf.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, encoding=txt_encoding, errors="replace") as f:
            return f.read()
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    raise ValueError(
        f"Неподдерживаемый формат «{ext}». "
        "Допустимы файлы: .txt, .docx, .pdf"
    )
