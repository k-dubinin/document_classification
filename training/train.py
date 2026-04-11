"""
Обучение классификаторов на предобработанных текстах.

Сначала тексты приводятся к виду «строка лемм» (как в дипломе: предобработка отдельно),
затем векторизатор TF-IDF и классификатор обучаются на train-части.

Источник данных: CSV, каталог «подпапка = класс» с .txt, либо набор Hugging Face
(см. data.data_loader и data.hf_loader).
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data.data_loader import load_corpus_from_class_folders
from data.hf_loader import load_corpus_from_huggingface
from preprocessing import TextPreprocessor

from . import config
from .dataset import load_csv_dataset, preprocess_series
from .pipelines import (
    build_pipeline_linear_svc,
    build_pipeline_logistic_regression,
    build_pipeline_naive_bayes,
)


def train_test_split_data(
    texts_processed: List[str],
    labels: pd.Series,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Разделяет данные на обучающую и тестовую выборки (стратификация по классам).
    """
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        texts_processed,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return list(X_train), list(X_test), y_train, y_test


def fit_pipeline(
    pipeline: Pipeline,
    X_train: List[str],
    y_train: np.ndarray,
) -> Pipeline:
    """Обучает переданный пайплайн."""
    pipeline.fit(X_train, y_train)
    return pipeline


def _build_pipeline_for_kind(model_kind: str) -> Pipeline:
    """Создаёт пайплайн TF-IDF + классификатор по коду вида модели."""
    if model_kind == "logreg":
        return build_pipeline_logistic_regression()
    if model_kind == "nb":
        return build_pipeline_naive_bayes()
    if model_kind == "svm":
        return build_pipeline_linear_svc()
    raise ValueError("model_kind должен быть 'logreg', 'nb' или 'svm'")


def train_from_csv(
    csv_path: str,
    model_kind: str,
    text_column: str = config.CSV_TEXT_COLUMN,
    label_column: str = config.CSV_LABEL_COLUMN,
) -> Tuple[Pipeline, List[str], np.ndarray, List[str], np.ndarray, TextPreprocessor]:
    """
    Полный цикл: загрузка CSV, предобработка, split, обучение одной модели.

    model_kind: 'logreg', 'nb' или 'svm'
    """
    texts, labels = load_csv_dataset(csv_path, text_column, label_column)
    preprocessor = TextPreprocessor()
    texts_processed = preprocess_series(texts, preprocessor)

    X_train, X_test, y_train, y_test = train_test_split_data(texts_processed, labels)

    pipeline = _build_pipeline_for_kind(model_kind)
    fit_pipeline(pipeline, X_train, y_train)
    return pipeline, X_train, y_train, X_test, y_test, preprocessor


def train_from_document_folders(
    data_root: str,
    model_kind: str,
) -> Tuple[Pipeline, List[str], np.ndarray, List[str], np.ndarray, TextPreprocessor]:
    """
    Обучение по структуре каталогов: data_root/ИмяКласса/*.txt

    model_kind: 'logreg', 'nb' или 'svm'
    """
    texts, labels = load_corpus_from_class_folders(data_root)
    preprocessor = TextPreprocessor()
    texts_processed = preprocess_series(texts, preprocessor)

    X_train, X_test, y_train, y_test = train_test_split_data(texts_processed, labels)

    pipeline = _build_pipeline_for_kind(model_kind)
    fit_pipeline(pipeline, X_train, y_train)
    return pipeline, X_train, y_train, X_test, y_test, preprocessor


def train_both_models_from_csv(
    csv_path: str,
    text_column: str = config.CSV_TEXT_COLUMN,
    label_column: str = config.CSV_LABEL_COLUMN,
) -> Dict[str, Any]:
    """
    Обучает Logistic Regression и Naive Bayes на одних и тех же данных.
    Возвращает словарь с пайплайнами, тестовой выборкой и предобработчиком.
    """
    texts, labels = load_csv_dataset(csv_path, text_column, label_column)
    preprocessor = TextPreprocessor()
    texts_processed = preprocess_series(texts, preprocessor)

    X_train, X_test, y_train, y_test = train_test_split_data(texts_processed, labels)

    pipe_lr = build_pipeline_logistic_regression()
    pipe_nb = build_pipeline_naive_bayes()

    fit_pipeline(pipe_lr, X_train, y_train)
    fit_pipeline(pipe_nb, X_train, y_train)

    return {
        "pipeline_logreg": pipe_lr,
        "pipeline_nb": pipe_nb,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "class_names": list(np.unique(labels)),
    }


def train_both_models_from_document_folders(data_root: str) -> Dict[str, Any]:
    """
    Обучает Logistic Regression и Naive Bayes на одном корпусе из папок с .txt.
    """
    texts, labels = load_corpus_from_class_folders(data_root)
    preprocessor = TextPreprocessor()
    texts_processed = preprocess_series(texts, preprocessor)

    X_train, X_test, y_train, y_test = train_test_split_data(texts_processed, labels)

    pipe_lr = build_pipeline_logistic_regression()
    pipe_nb = build_pipeline_naive_bayes()

    fit_pipeline(pipe_lr, X_train, y_train)
    fit_pipeline(pipe_nb, X_train, y_train)

    return {
        "pipeline_logreg": pipe_lr,
        "pipeline_nb": pipe_nb,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "class_names": list(np.unique(labels)),
    }


def train_from_huggingface(
    dataset_id: str,
    model_kind: str,
    split: str = config.HF_DEFAULT_SPLIT,
    text_column: str = config.HF_DEFAULT_TEXT_COLUMN,
    label_column: str = config.HF_DEFAULT_LABEL_COLUMN,
) -> Tuple[Pipeline, List[str], np.ndarray, List[str], np.ndarray, TextPreprocessor]:
    """
    Обучение по табличному набору с Hugging Face (текст + метка класса).

    Подходит для корпусов обращений граждан и аналогичных документов с рубрикой/темой.
    """
    texts, labels = load_corpus_from_huggingface(
        dataset_id, split=split, text_column=text_column, label_column=label_column
    )
    preprocessor = TextPreprocessor()
    texts_processed = preprocess_series(texts, preprocessor)

    X_train, X_test, y_train, y_test = train_test_split_data(texts_processed, labels)

    pipeline = _build_pipeline_for_kind(model_kind)
    fit_pipeline(pipeline, X_train, y_train)
    return pipeline, X_train, y_train, X_test, y_test, preprocessor


def train_both_models_from_huggingface(
    dataset_id: str,
    split: str = config.HF_DEFAULT_SPLIT,
    text_column: str = config.HF_DEFAULT_TEXT_COLUMN,
    label_column: str = config.HF_DEFAULT_LABEL_COLUMN,
) -> Dict[str, Any]:
    """Logistic Regression и Naive Bayes на одном наборе с Hugging Face."""
    texts, labels = load_corpus_from_huggingface(
        dataset_id, split=split, text_column=text_column, label_column=label_column
    )
    preprocessor = TextPreprocessor()
    texts_processed = preprocess_series(texts, preprocessor)

    X_train, X_test, y_train, y_test = train_test_split_data(texts_processed, labels)

    pipe_lr = build_pipeline_logistic_regression()
    pipe_nb = build_pipeline_naive_bayes()

    fit_pipeline(pipe_lr, X_train, y_train)
    fit_pipeline(pipe_nb, X_train, y_train)

    return {
        "pipeline_logreg": pipe_lr,
        "pipeline_nb": pipe_nb,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "class_names": list(np.unique(labels)),
    }
