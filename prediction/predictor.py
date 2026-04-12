"""
Предсказание категории текстового документа по сохранённой модели (.joblib).

Вход: строка текста или файл (извлечение текста — data.document_text).
Предобработка совпадает с этапом обучения (лемматизация, стоп-слова).

Для моделей с predict_proba (Logistic Regression, Naive Bayes) возвращаются вероятности по классам.
Для LinearSVC — значения decision_function (не вероятности; для порогов по «уверенности» используйте отступ от нуля).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.pipeline import Pipeline

from preprocessing import TextPreprocessor

from training.persistence import load_model_bundle

logger = logging.getLogger(__name__)


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


def predict_with_details(
    raw_text: str,
    bundle_path: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Предсказание с вероятностями (если есть predict_proba) или с decision_function (LinearSVM).

    Returns
    -------
    dict с ключами:
      label — предсказанный класс;
      probabilities — dict класс -> вероятность (только для LR, NB);
      probability_top — список (класс, вероятность), отсортирован по убыванию, top_k;
      decision_scores — dict класс -> оценка (для SVM и как дополнение);
      score_top — top_k по оценкам (для интерпретации SVM).
    """
    pipeline, preprocessor = load_model_bundle(bundle_path)
    processed = preprocessor.preprocess(raw_text, as_string=True)
    x_list = [processed]

    label = pipeline.predict(x_list)[0]
    classes = list(pipeline.classes_)
    cls_str = [str(c) for c in classes]

    out: Dict[str, Any] = {
        "label": label,
        "probabilities": None,
        "probability_top": None,
        "decision_scores": None,
        "score_top": None,
    }

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(x_list)[0]
        out["probabilities"] = {str(c): float(p) for c, p in zip(classes, proba)}
        ranked: List[Tuple[str, float]] = sorted(
            zip(cls_str, proba.astype(float)),
            key=lambda t: -t[1],
        )[: max(1, top_k)]
        out["probability_top"] = [[c, float(p)] for c, p in ranked]
        logger.info(
            "Классификация: модель с predict_proba, класс=%s, топ_вероятность=%.4f",
            label,
            ranked[0][1] if ranked else 0.0,
        )
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x_list)
        row = np.asarray(scores).reshape(-1)
        if row.size == 1 and len(classes) == 2:
            margin = float(row[0])
            out["decision_scores"] = {
                cls_str[0]: float(-margin),
                cls_str[1]: float(margin),
            }
            ranked = sorted(out["decision_scores"].items(), key=lambda t: -t[1])[: max(1, top_k)]
            out["score_top"] = [[c, float(s)] for c, s in ranked]
        else:
            out["decision_scores"] = {c: float(v) for c, v in zip(cls_str, row)}
            ranked = sorted(zip(cls_str, row.astype(float)), key=lambda t: -t[1])[: max(1, top_k)]
            out["score_top"] = [[c, float(v)] for c, v in ranked]
        logger.info(
            "Классификация: модель с decision_function (без predict_proba), класс=%s",
            label,
        )
    else:
        logger.info("Классификация: класс=%s (расширенные оценки недоступны)", label)

    out["label"] = str(out["label"])
    return out
