"""
Загрузка документов для обучения классификатора.

Основной сценарий: каталог с подпапками — имя подпапки = метка класса,
внутри — файлы документов (см. document_text.supported_document_extensions), один файл = один документ.

Извлечение текста из Word/PDF — модуль document_text.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd

from .document_text import (
    extract_text_from_docx,
    extract_text_from_pdf,
    read_text_from_document,
    supported_document_extensions,
)

# --- Параметры чтения текстовых файлов (вынесены в переменные) ---
# Кодировка при чтении .txt
TXT_FILE_ENCODING = "utf-8"

# Расширения файлов при обходе папок классов (синхронно с document_text)
SUPPORTED_DOCUMENT_EXTENSIONS = supported_document_extensions()


def read_txt_document(file_path: str, encoding: str = TXT_FILE_ENCODING) -> str:
    """
    Читает содержимое одного текстового документа (.txt).

    Parameters
    ----------
    file_path : str
        Полный путь к файлу.
    encoding : str
        Кодировка (по умолчанию UTF-8).
    """
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def _list_class_subdirs(root_dir: str) -> List[str]:
    """Возвращает отсортированные имена непосредственных подпапок (имена классов)."""
    names: List[str] = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and not name.startswith("."):
            names.append(name)
    return names


def _collect_document_paths(class_dir: str) -> List[str]:
    """Файлы .txt / .docx / .pdf в папке класса (без рекурсии в подпапки)."""
    paths: List[str] = []
    for name in sorted(os.listdir(class_dir)):
        if name.startswith("."):
            continue
        lower = name.lower()
        if not any(lower.endswith(ext) for ext in SUPPORTED_DOCUMENT_EXTENSIONS):
            continue
        full = os.path.join(class_dir, name)
        if os.path.isfile(full):
            paths.append(full)
    return paths


def load_corpus_from_class_folders(
    root_dir: str,
    encoding: str = TXT_FILE_ENCODING,
) -> Tuple[pd.Series, pd.Series]:
    """
    Строит датасет из структуры каталогов: root/ИмяКласса/*.txt|docx|pdf

    Каждая подпапка первого уровня — отдельный класс. Внутри — документы поддерживаемых форматов.

    Returns
    -------
    texts : pd.Series
        Тексты документов.
    labels : pd.Series
        Метки классов (строки — имена папок).
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Каталог не найден: {root_dir}")

    class_names = _list_class_subdirs(root_dir)
    if not class_names:
        raise ValueError(
            f"В '{root_dir}' нет подпапок с классами. "
            "Ожидается: корень/НазваниеКласса/файлы (.txt, .docx, .pdf)"
        )

    all_texts: List[str] = []
    all_labels: List[str] = []

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        doc_paths = _collect_document_paths(class_dir)
        for path in doc_paths:
            try:
                raw = read_text_from_document(path, txt_encoding=encoding)
            except (OSError, ValueError) as e:
                raise RuntimeError(f"Не удалось прочитать файл: {path}") from e
            stripped = raw.strip()
            if not stripped:
                continue
            all_texts.append(stripped)
            all_labels.append(class_name)

    if not all_texts:
        raise ValueError(
            f"Не найдено ни одного непустого документа в подпапках '{root_dir}'. "
            "Положите файлы .txt, .docx или .pdf внутрь папок классов."
        )

    texts = pd.Series(all_texts, dtype=object)
    labels = pd.Series(all_labels, dtype=object)
    return texts, labels
