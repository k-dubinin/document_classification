"""
Предсказание категории текстового документа по сохранённой модели (.joblib).

Вход: строка текста или файл .txt / .docx / .pdf (извлечение текста — data.document_text).
Предобработка совпадает с этапом обучения (лемматизация, стоп-слова).
"""

from typing import Tuple, Union

from sklearn.pipeline import Pipeline

from preprocessing import TextPreprocessor

from training.persistence import load_model_bundle


def predict_class(
    raw_text: str,
    pipeline: Pipeline,
    preprocessor: TextPreprocessor,
) -> Union[str, int]:
    """
    Возвращает предсказанный класс для одного текста.
    Сначала предобработка (как при обучении), затем predict пайплайна.
    """
    processed = preprocessor.preprocess(raw_text, as_string=True)
    label = pipeline.predict([processed])[0]
    return label


def predict_from_file(
    raw_text: str,
    bundle_path: str,
) -> Tuple[Union[str, int], Pipeline, TextPreprocessor]:
    """
    Загружает модель с диска и предсказывает класс по уже готовой строке текста.

    Возвращает (метка, pipeline, preprocessor) — последние два полезны для повторных вызовов.
    """
    pipeline, preprocessor = load_model_bundle(bundle_path)
    label = predict_class(raw_text, pipeline, preprocessor)
    return label, pipeline, preprocessor


def predict_document_path(
    document_path: str,
    bundle_path: str,
) -> Tuple[Union[str, int], Pipeline, TextPreprocessor]:
    """
    Классификация файла: .txt, .docx или .pdf — текст извлекается автоматически.
    """
    from data.document_text import read_text_from_document

    raw_text = read_text_from_document(document_path)
    return predict_from_file(raw_text, bundle_path)
