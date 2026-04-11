"""
Загрузка обучающих данных из CSV и применение предобработки текста.

Ожидается CSV с двумя столбцами (по умолчанию text и label), кодировка UTF-8.

Корпус из папок «класс = подпапка» и файлов .txt загружается в модуле data.data_loader
(load_corpus_from_class_folders) и передаётся в обучение через train_from_document_folders.
"""

import os
from typing import Optional, Tuple

import pandas as pd

from preprocessing import TextPreprocessor

from . import config


def load_csv_dataset(
    csv_path: str,
    text_column: str = config.CSV_TEXT_COLUMN,
    label_column: str = config.CSV_LABEL_COLUMN,
    encoding: str = "utf-8",
) -> Tuple[pd.Series, pd.Series]:
    """
    Читает CSV с колонками текста и метки класса.

    Возвращает два объекта Series: тексты и метки (в исходном виде — строка или число).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)

    if text_column not in df.columns:
        raise ValueError(f"В CSV нет столбца '{text_column}'. Доступные: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"В CSV нет столбца '{label_column}'. Доступные: {list(df.columns)}")

    texts = df[text_column].fillna("").astype(str)
    labels = df[label_column]
    # Удаляем строки с пустым текстом
    mask = texts.str.strip() != ""
    texts = texts[mask].reset_index(drop=True)
    labels = labels[mask].reset_index(drop=True)

    return texts, labels


def preprocess_series(
    texts: pd.Series,
    preprocessor: Optional[TextPreprocessor] = None,
) -> list:
    """Предобрабатывает каждый текст (лемматизация, стоп-слова) — список строк для TF-IDF."""
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    return preprocessor.preprocess_batch(list(texts), as_string=True)
