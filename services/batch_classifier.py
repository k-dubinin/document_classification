"""
Пакетная классификация документов из директории.

Цель: аккуратно вынести логику пакетной обработки из UI.

Сценарий:
  - входная папка содержит документы (можно с подпапками)
  - каждый файл читается через data.document_text.read_text_from_document
  - файл классифицируется моделью (.joblib)
  - файл КОПИРУЕТСЯ в output_dir/<класс>/имя_файла
"""

from __future__ import annotations

import os
import shutil
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.document_text import read_text_from_document, supported_document_extensions
from prediction.predictor import predict_with_details


@dataclass
class BatchItemResult:
    input_path: str
    ok: bool
    label: Optional[str] = None
    probability: Optional[float] = None  # для LR/NB
    score: Optional[float] = None  # для SVM (decision_function; не probability)
    manual_review_required: str = "no"  # yes / no
    review_reason: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None


def iter_document_paths(input_dir: str, recursive: bool = True) -> List[str]:
    """
    Список файлов поддерживаемых форматов в директории.
    По умолчанию рекурсивно.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")

    exts = set(supported_document_extensions())
    pattern = "**/*" if recursive else "*"
    files: List[str] = []
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            files.append(str(p))
    files.sort(key=lambda x: x.lower())
    return files


def _safe_class_dir_name(label: str) -> str:
    """
    Имя папки класса. Для простоты оставляем как есть, но убираем самые проблемные символы.
    """
    s = str(label).strip()
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        s = s.replace(ch, "_")
    return s if s else "Unknown"


def _unique_destination(dst: Path) -> Path:
    """
    Если файл с таким именем уже есть в папке класса — добавляем суффикс _1, _2, ...
    """
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def classify_directory(
    model_path: str,
    input_dir: str,
    output_dir: str,
    *,
    recursive: bool = True,
    top_k: int = 5,
    manual_review_probability_threshold: float = 0.20,
    manual_review_folder_name: str = "Требует_проверки",
) -> Iterator[BatchItemResult]:
    """
    Классифицирует все документы из input_dir и копирует их в output_dir/<класс>/.

    Возвращает итератор результатов по каждому файлу (для отображения прогресса в UI).
    """
    files = iter_document_paths(input_dir, recursive=recursive)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        try:
            text = read_text_from_document(file_path)
            if not text or not str(text).strip():
                raise ValueError("пустой текст (файл пустой или извлечение не удалось)")

            details: Dict = predict_with_details(text, model_path, top_k=top_k)
            label = str(details.get("label"))

            prob = None
            if details.get("probabilities"):
                prob = details["probabilities"].get(label)
                if prob is None:
                    prob = details["probabilities"].get(str(label))

            score = None
            if prob is None and details.get("decision_scores"):
                score = details["decision_scores"].get(label)
                if score is None:
                    score = details["decision_scores"].get(str(label))

            manual_review_required = "no"
            review_reason = None
            if prob is not None and float(prob) < float(manual_review_probability_threshold):
                manual_review_required = "yes"
                review_reason = "low_confidence_probability"
                class_dir = out_root / _safe_class_dir_name(manual_review_folder_name)
            else:
                class_dir = out_root / _safe_class_dir_name(label)
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = _unique_destination(class_dir / Path(file_path).name)
            shutil.copy2(file_path, dst)

            yield BatchItemResult(
                input_path=file_path,
                ok=True,
                label=label,
                probability=float(prob) if prob is not None else None,
                score=float(score) if score is not None else None,
                manual_review_required=manual_review_required,
                review_reason=review_reason,
                output_path=str(dst),
            )
        except Exception as e:
            yield BatchItemResult(
                input_path=file_path,
                ok=False,
                manual_review_required="no",
                error=str(e),
            )


def write_batch_report_csv(results: List[BatchItemResult], csv_path: str) -> str:
    """
    Сохраняет CSV-отчёт пакетной классификации.
    """
    dst = Path(csv_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_path",
                "ok",
                "predicted_class",
                "probability",
                "score",
                "manual_review_required",
                "review_reason",
                "output_path",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "input_path": r.input_path,
                    "ok": "yes" if r.ok else "no",
                    "predicted_class": r.label or "",
                    "probability": "" if r.probability is None else f"{float(r.probability):.6f}",
                    "score": "" if r.score is None else f"{float(r.score):.6f}",
                    "manual_review_required": r.manual_review_required,
                    "review_reason": r.review_reason or "",
                    "output_path": r.output_path or "",
                    "error": r.error or "",
                }
            )
    return str(dst)
