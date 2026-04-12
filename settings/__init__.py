"""Загрузка YAML/JSON настроек и логирование (см. settings.loader)."""

from .loader import get_setting, init_app, peek_config_path_from_argv

__all__ = ["init_app", "get_setting", "peek_config_path_from_argv"]
