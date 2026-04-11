"""
Сохранение и загрузка обученной модели (пайплайн + предобработчик текста).

Предобработчик сохраняется вместе с моделью, чтобы при predict использовать
те же правила лемматизации и стоп-слов, что и при обучении.
"""

import os
from typing import Any, Dict, Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline

from preprocessing import TextPreprocessor


def save_model_bundle(
    pipeline: Pipeline,
    preprocessor: TextPreprocessor,
    file_path: str,
) -> None:
    """
    Сохраняет в один файл joblib:
      - pipeline: TfidfVectorizer + классификатор;
      - preprocessor: объект для лемматизации и стоп-слов (тот же, что при обучении).
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    bundle: Dict[str, Any] = {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
    }
    joblib.dump(bundle, file_path)


def load_model_bundle(file_path: str) -> Tuple[Pipeline, TextPreprocessor]:
    """Загружает пайплайн и предобработчик. Ожидается формат save_model_bundle."""
    data = joblib.load(file_path)
    if not isinstance(data, dict) or "pipeline" not in data:
        raise ValueError(
            "Неизвестный формат файла. Ожидается результат save_model_bundle "
            "(словарь с ключами 'pipeline' и 'preprocessor')."
        )
    pipeline = data["pipeline"]
    preprocessor = data.get("preprocessor")
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    return pipeline, preprocessor
