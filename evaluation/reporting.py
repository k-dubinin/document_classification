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
    Сохраняет confusion matrix в PNG с нормальным отображением
    русских подписей и длинных названий классов.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # ==========================================
    # ПАПКА ДЛЯ СОХРАНЕНИЯ
    # ==========================================

    directory = os.path.dirname(file_path)

    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    # ==========================================
    # СОКРАЩЕННЫЕ НАЗВАНИЯ КЛАССОВ
    # ==========================================

    SHORT_LABELS = {
        "Управление контроля рекламы и недобросовестной конкуренции":
            "Реклама",

        "Управление регулирования электроэнергетики":
            "Электроэнергетика",

        "Управление регулирования связи и информационных технологий":
            "Связь и ИТ",

        "Управление контроля строительства и природных ресурсов":
            "Строительство",

        "Управление контроля финансовых рынков":
            "Финансы",

        "Управление регулирования топливно-энергетического комплекса и химической промышленности":
            "ТЭК и химпром",

        "Управления регулирования транспорта":
            "Транспорт",

        "Управление регионального тарифного регулирования":
            "Тарифы",
    }

    # ==========================================
    # LABELS
    # ==========================================

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    # Короткие названия только для отображения
    short_labels = [
        SHORT_LABELS.get(label, label)
        for label in labels
    ]

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    # ==========================================
    # НАСТРОЙКА ШРИФТОВ
    # ==========================================

    plt.rcParams["font.sans-serif"] = [
        "Segoe UI",
        "DejaVu Sans",
        "Arial",
    ]

    plt.rcParams["axes.unicode_minus"] = False

    # ==========================================
    # ОПРЕДЕЛЕНИЕ РАЗМЕРА ФИГУРЫ В ЗАВИСИМОСТИ ОТ КОЛИЧЕСТВА КЛАССОВ
    # ==========================================

    n_classes = len(labels)
    # Увеличиваем размер фигуры в зависимости от количества классов
    figsize_multiplier = max(1, n_classes / 6)
    fig_width = max(10, 14 * figsize_multiplier)
    fig_height = max(8, 14 * figsize_multiplier)

    # ==========================================
    # FIGURE
    # ==========================================

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Heatmap
    im = ax.imshow(cm, cmap="Blues", interpolation='nearest')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)  # Увеличиваем размер шрифта для цветовой шкалы

    # ==========================================
    # ОСИ
    # ==========================================

    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))

    ax.set_xticklabels(short_labels)
    ax.set_yticklabels(short_labels)

    # Поворот подписей X
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=12  # Увеличиваем размер шрифта
    )

    # Поворот подписей Y
    plt.setp(
        ax.get_yticklabels(),
        fontsize=12  # Увеличиваем размер шрифта
    )

    # ==========================================
    # ЗНАЧЕНИЯ ВНУТРИ КЛЕТОК
    # ==========================================

    # Определяем порог для изменения цвета текста
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10 if n_classes > 10 else 12  # Уменьшаем размер шрифта для больших матриц
            )

    # ==========================================
    # ЗАГОЛОВКИ
    # ==========================================

    ax.set_title(title, fontsize=16, pad=20)

    ax.set_xlabel(
        "Предсказанный класс",
        fontsize=14
    )

    ax.set_ylabel(
        "Истинный класс",
        fontsize=14
    )

    # ==========================================
    # ОТСТУПЫ
    # ==========================================

    plt.tight_layout()

    # ==========================================
    # СОХРАНЕНИЕ
    # ==========================================

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches='tight'  # Улучшает отступы
    )

    plt.close(fig)