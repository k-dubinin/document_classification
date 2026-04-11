"""
Обучение моделей: загрузка CSV, пайплайны TF-IDF + классификатор, сохранение в joblib.

Параметры векторизаторов и классификаторов задаются в training.config.
"""

from .persistence import load_model_bundle, save_model_bundle
from .train import (
    train_both_models_from_csv,
    train_both_models_from_document_folders,
    train_both_models_from_huggingface,
    train_from_csv,
    train_from_document_folders,
    train_from_huggingface,
)

__all__ = [
    "train_from_csv",
    "train_from_document_folders",
    "train_from_huggingface",
    "train_both_models_from_csv",
    "train_both_models_from_document_folders",
    "train_both_models_from_huggingface",
    "save_model_bundle",
    "load_model_bundle",
]
