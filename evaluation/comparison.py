"""
Сравнение нескольких обученных моделей по одной тестовой выборке.

Используется после команды compare: обе модели обучаются на одном train
и оцениваются на одном test — таблица показывает, какая архитектура лучше по метрикам.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from evaluation.metrics import compute_main_metrics


def rows_for_comparison(
    models: Dict[str, Pipeline],
    X_test: list,
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    """
    Считает метрики для каждой модели и возвращает список строк (удобно для таблицы).

    models: словарь «краткое имя» -> обученный Pipeline
    """
    rows: List[Dict[str, object]] = []
    for short_name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        metrics = compute_main_metrics(y_test, y_pred)
        row: Dict[str, object] = {"model": short_name}
        row.update(metrics)
        rows.append(row)
    return rows


def print_metrics_comparison_table(rows: List[Dict[str, object]]) -> None:
    """Печатает таблицу сравнения метрик в консоль."""
    df = pd.DataFrame(rows)
    # Читаемые заголовки для диплома / отчёта
    rename = {
        "model": "Модель",
        "accuracy": "Accuracy",
        "precision_weighted": "Precision (weighted)",
        "recall_weighted": "Recall (weighted)",
        "f1_weighted": "F1 (weighted)",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
    }
    df = df.rename(columns=rename)
    numeric_cols = [c for c in df.columns if c != "Модель"]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].round(4)
    print("\n=== Сравнение моделей по метрикам (тестовая выборка) ===\n")
    print(df.to_string(index=False))
