"""
Предсказание класса для нового текста по сохранённой модели.
"""

from .predictor import (
    predict_class,
    predict_document_path,
    predict_from_file,
    predict_with_details,
)

__all__ = [
    "predict_class",
    "predict_from_file",
    "predict_document_path",
    "predict_with_details",
]
