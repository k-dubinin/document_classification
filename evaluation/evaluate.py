"""
Оценка качества обученной модели: метрики, отчёты, сохранение результатов.

Печатает матрицу ошибок и classification_report, при необходимости сохраняет JSON и PNG.
"""

import os
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.pipeline import Pipeline

from evaluation.metrics import compute_main_metrics
from evaluation.reporting import (
    build_metrics_payload,
    print_classification_report_ru,
    print_confusion_matrix,
    save_confusion_matrix_png,
    save_metrics_json,
)

from training import config


def evaluate_and_report(
    pipeline: Pipeline,
    X_test: list,
    y_test: np.ndarray,
    model_name: str,
    output_dir: Optional[str] = None,
    labels_order: Optional[Union[list, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Предсказание на тесте, вывод матрицы ошибок и отчёта, сохранение JSON и PNG.

    Если задан output_dir — пишутся файлы:
      {краткое_имя}_metrics.json
      {краткое_имя}_confusion_matrix.png
    """
    y_pred = pipeline.predict(X_test)

    print_confusion_matrix(y_test, y_pred, labels=labels_order)
    print_classification_report_ru(y_test, y_pred)

    main_metrics = compute_main_metrics(y_test, y_pred)
    print("\n--- Сводные метрики ---")
    for key, value in main_metrics.items():
        print(f"  {key}: {value:.4f}")

    payload = build_metrics_payload(
        y_test,
        y_pred,
        main_metrics,
        model_name,
        labels_order=labels_order,
    )

    if output_dir:
        slug = model_name.lower().replace(" ", "_")
        json_path = os.path.join(output_dir, f"{slug}{config.FILENAME_METRICS_SUFFIX}")
        png_path = os.path.join(output_dir, f"{slug}{config.FILENAME_CONFUSION_MATRIX_SUFFIX}")

        save_metrics_json(payload, json_path)
        save_confusion_matrix_png(
            y_test,
            y_pred,
            png_path,
            title=f"Матрица ошибок — {model_name}",
            labels=labels_order,
        )
        print(f"\nМетрики сохранены: {json_path}")
        print(f"Матрица ошибок (PNG): {png_path}")

    return payload
