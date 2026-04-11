"""
Загрузка обучающего корпуса из Hugging Face Datasets (однократная загрузка, дальше кэш локально).

Подходит для наборов вроде обращений граждан к органам власти: текст на русском + метка темы/категории.
Пример: Adilbai/kz-gov-complaints-data-kz-ru — поля text_ru и category (см. карточку набора на HF).

Требуется пакет: pip install datasets
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def load_corpus_from_huggingface(
    dataset_id: str,
    split: str = "train",
    text_column: str = "text_ru",
    label_column: str = "category",
) -> Tuple[pd.Series, pd.Series]:
    """
    Скачивает (при первом запуске) и читает split набора, возвращает тексты и метки классов.

    Parameters
    ----------
    dataset_id : str
        Идентификатор на Hugging Face, например "Adilbai/kz-gov-complaints-data-kz-ru".
    split : str
        Обычно "train".
    text_column, label_column : str
        Имена столбцов с текстом документа и с меткой класса.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Для загрузки с Hugging Face установите пакет: pip install datasets"
        ) from e

    ds = load_dataset(dataset_id, split=split)
    df = ds.to_pandas()

    if text_column not in df.columns:
        raise ValueError(
            f"В наборе нет столбца '{text_column}'. Доступные: {list(df.columns)}"
        )
    if label_column not in df.columns:
        raise ValueError(
            f"В наборе нет столбца '{label_column}'. Доступные: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").astype(str)
    labels = df[label_column]
    # Пустые тексты отбрасываем
    mask = texts.str.strip() != ""
    texts = texts[mask].reset_index(drop=True)
    labels = labels[mask].reset_index(drop=True)

    if len(texts) == 0:
        raise ValueError("После фильтрации не осталось ни одного непустого текста.")

    return texts, labels
