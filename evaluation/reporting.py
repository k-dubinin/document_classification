"""
Вывод матрицы ошибок, отчёта по классам, сохранение метрик и графиков.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> None:
    """Печатает матрицу ошибок в консоль."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n--- Матрица ошибок (confusion matrix) ---")
    if labels is not None:
        print("Метки классов (порядок строк/столбцов):", labels)
    print(cm)


def print_classification_report_ru(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Печатает classification_report sklearn (precision/recall/f1 по классам)."""
    print("\n--- Отчёт по классам (classification report) ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def build_metrics_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    main_metrics: Dict[str, float],
    model_name: str,
    labels_order: Optional[Union[List, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Собирает словарь для сохранения в JSON: общие метрики + отчёт по классам в виде dict.

    Матрица ошибок строится в порядке labels_order (как строки/столбцы графика и консоли).
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=labels_order,
    )
    if labels_order is None:
        labels_order = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    return {
        "model_name": model_name,
        "main_metrics": main_metrics,
        "classification_report": report_dict,
        "class_names": [str(x) for x in labels_order],
        "confusion_matrix": cm.tolist(),
    }


def _json_safe(obj: Any) -> Any:
    """Приводит значения (в т.ч. numpy) к типам, удобным для json.dump."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics_json(payload: Dict[str, Any], file_path: str) -> None:
    """Сохраняет метрики и вспомогательную информацию в JSON."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    safe_payload = _json_safe(payload)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, ensure_ascii=False, indent=2)


def save_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    file_path: str,
    title: str = "Матрица ошибок",
    labels: Optional[Union[List, np.ndarray]] = None,
) -> None:
    """
    Сохраняет изображение матрицы ошибок в PNG (для вставки в дипломную работу).
    Подписи классов на русском: используем системный шрифт Windows при необходимости.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = os.path.dirname(file_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    # Поддержка кириллицы в подписях на типичной установке Windows
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 6))
    display_labels = labels
    if display_labels is None:
        display_labels = np.unique(np.concatenate([y_true, y_pred]))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=display_labels,
        ax=ax,
        colorbar=True,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close(fig)
