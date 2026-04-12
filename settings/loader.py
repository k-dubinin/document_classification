"""
Загрузка настроек из YAML или JSON без правки кода Python.

Порядок выбора файла:
  1) путь из переменной окружения APP_CONFIG;
  2) пара аргументов --config путь в sys.argv (в любом месте после имени скрипта);
  3) settings/default.yaml относительно корня проекта.

Секции:
  - корень файла: имена переменных как в training/config.py (TEST_SIZE, TFIDF_*, …);
  - ocr: пороги и параметры извлечения PDF (см. data.document_text);
  - logging: level, file.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

_FLAT: Dict[str, Any] = {}
_LOAD_PATH: Optional[str] = None


def peek_config_path_from_argv() -> Optional[str]:
    """Ищет --config путь в sys.argv (до полного разбора argparse)."""
    argv = sys.argv[1:]
    for i, x in enumerate(argv):
        if x == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--config="):
            return x.split("=", 1)[1].strip() or None
    return os.environ.get("APP_CONFIG")


def _load_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
        else:
            try:
                import yaml
            except ImportError as e:
                raise ImportError("Для YAML установите: pip install pyyaml") from e
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Корень конфигурации должен быть объектом (словарём).")
    return data


def _coerce_training_value(name: str, value: Any) -> Any:
    """Приводит значения из файла к типам, ожидаемым в training/config."""
    if name == "TFIDF_NGRAM_RANGE" and isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    if name == "LR_CLASS_WEIGHT" and value is not None and isinstance(value, str):
        if value.lower() in ("none", "null", ""):
            return None
    return value


def _apply_to_training_module(raw_top: Dict[str, Any]) -> None:
    import training.config as tc

    for key, value in raw_top.items():
        if key in ("ocr", "logging"):
            continue
        if not hasattr(tc, key):
            continue
        setattr(tc, key, _coerce_training_value(key, value))


def _flatten_ocr(ocr: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in ocr.items():
        out[f"ocr.{k}"] = v


def _flatten_logging(log: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in log.items():
        out[f"logging.{k}"] = v


def init_app(project_root: str, config_path: Optional[str] = None) -> None:
    """
    Загружает конфиг, применяет к training.config, настраивает логирование.
    Безопасно вызывать повторно с тем же путём (пропуск).
    """
    global _FLAT, _LOAD_PATH

    path = config_path or peek_config_path_from_argv()
    if not path:
        path = os.path.join(project_root, "settings", "default.yaml")

    if not os.path.isfile(path):
        _FLAT = {}
        _LOAD_PATH = None
        _setup_logging(project_root, {})
        return

    if path == _LOAD_PATH:
        return

    data = _load_file(path)
    raw = dict(data)
    ocr_block = raw.pop("ocr", {}) or {}
    log_block = raw.pop("logging", {}) or {}

    flat: Dict[str, Any] = {}
    if isinstance(ocr_block, dict):
        _flatten_ocr(ocr_block, flat)
    if isinstance(log_block, dict):
        _flatten_logging(log_block, flat)

    _apply_to_training_module(raw)
    for k, v in raw.items():
        if k not in ("ocr", "logging"):
            flat[k] = v

    _FLAT = flat
    _LOAD_PATH = path
    _setup_logging(project_root, flat)
    logging.getLogger(__name__).info("Загружен конфиг: %s", os.path.abspath(path))


def get_setting(key: str, default: Any = None) -> Any:
    """Доступ по ключу вида ocr.PDF_PAGE_TEXT_MIN_BEFORE_OCR или logging.level."""
    return _FLAT.get(key, default)


def _setup_logging(project_root: str, flat: Dict[str, Any]) -> None:
    raw_level = flat.get("logging.level")
    level_name = str(raw_level if raw_level is not None else "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    file_rel = flat.get("logging.file")
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if file_rel:
        log_path = os.path.join(project_root, str(file_rel))
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for h in handlers:
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(h)
