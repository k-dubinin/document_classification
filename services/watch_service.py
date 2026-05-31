"""
Сервис мониторинга для автоматической классификации документов
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional
from threading import Thread, Event

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from prediction.predictor import predict_with_details
from data.document_text import read_text_from_document


class DocumentWatchHandler(FileSystemEventHandler):
    """
    Обработчик событий файловой системы для мониторинга новых документов
    """
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        review_dir: str,
        confidence_threshold: float = 20.0,
        logger: logging.Logger = None
    ):
        super().__init__()
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.review_dir = Path(review_dir)
        self.confidence_threshold = confidence_threshold / 100.0  # Преобразуем проценты в долю
        self.logger = logger or logging.getLogger(__name__)

        # Создаем директории, если они не существуют
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)

        # Хранение обработанных файлов для предотвращения дубликатов
        self.processed_files = set()

    def on_created(self, event):
        """
        Обработка события создания нового файла
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Проверяем расширение файла
        allowed_extensions = {".txt", ".docx", ".pdf", ".rtf", ".html", ".htm", ".odt", ".md"}
        if file_path.suffix.lower() not in allowed_extensions:
            return

        # Проверяем, не является ли файл временным
        if file_path.name.startswith('~') or file_path.name.startswith('.'):
            return

        # Ждем немного, чтобы убедиться, что файл полностью записан
        time.sleep(0.5)

        # Проверяем, не обрабатывается ли файл уже
        if file_path in self.processed_files:
            return

        self.processed_files.add(file_path)

        # Обрабатываем файл в отдельном потоке, чтобы не блокировать мониторинг
        thread = Thread(target=self._process_file, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _process_file(self, file_path: Path):
        """
        Обработка одного файла
        """
        try:
            self.logger.info(f"Обнаружен новый файл для обработки: {file_path}")

            # Извлекаем текст из документа
            try:
                text = read_text_from_document(str(file_path))
            except Exception as e:
                self.logger.error(f"Ошибка при извлечении текста из файла {file_path}: {e}")
                return

            if not text or not text.strip():
                self.logger.warning(f"Файл {file_path} не содержит текста или не может быть прочитан")
                return

            # Выполняем предсказание
            result = predict_with_details(
                text,
                bundle_path=self.model_path,
                top_k=1
            )

            predicted_class = result['label']
            confidence = result.get('probability_top', result.get('score_top', [[0]]))[0][1] if result.get('probability_top') or result.get('score_top') else 0

            self.logger.info(f"Файл {file_path} классифицирован как '{predicted_class}' с уверенностью {confidence:.2%}")

            # Определяем, нужно ли отправить на ручную проверку
            if confidence < self.confidence_threshold:
                # Перемещаем в директорию для ручной проверки
                target_path = self.review_dir / file_path.name
                self.logger.info(f"Уверенность модели ({confidence:.2%}) ниже порога ({self.confidence_threshold:.2%}), файл перемещается в папку ручной проверки: {target_path}")
            else:
                # Перемещаем в директорию соответствующего класса
                class_dir = self.output_dir / predicted_class
                class_dir.mkdir(exist_ok=True)
                target_path = class_dir / file_path.name
                self.logger.info(f"Файл перемещается в папку класса '{predicted_class}': {target_path}")

            # Перемещаем файл
            try:
                file_path.rename(target_path)
                self.logger.info(f"Файл успешно перемещен: {target_path}")
            except Exception as e:
                self.logger.error(f"Ошибка при перемещении файла {file_path} в {target_path}: {e}")

        except Exception as e:
            self.logger.error(f"Ошибка при обработке файла {file_path}: {e}")


class WatchService:
    """
    Сервис мониторинга директории для автоматической классификации документов
    """
    def __init__(
        self,
        model_path: str,
        input_dir: str,
        output_dir: str,
        review_dir: str,
        confidence_threshold: float = 20.0,
        recursive: bool = True,
        polling_interval: float = 1.0,
        logger: logging.Logger = None
    ):
        self.model_path = model_path
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.review_dir = Path(review_dir)
        self.confidence_threshold = confidence_threshold
        self.recursive = recursive
        self.polling_interval = polling_interval
        self.logger = logger or logging.getLogger(__name__)

        self.observer = Observer()
        self.handler = DocumentWatchHandler(
            model_path=model_path,
            output_dir=output_dir,
            review_dir=review_dir,
            confidence_threshold=confidence_threshold,
            logger=logger
        )
        self.running = False
        self.thread = None

    def start(self):
        """
        Запуск сервиса мониторинга
        """
        if self.running:
            return

        # Создаем директории, если они не существуют
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)

        # Настраиваем наблюдение
        self.observer.schedule(
            self.handler,
            str(self.input_dir),
            recursive=self.recursive
        )

        self.observer.start()
        self.running = True

        self.logger.info(f"Сервис мониторинга запущен. Отслеживается директория: {self.input_dir}")
        self.logger.info(f"Модель: {self.model_path}")
        self.logger.info(f"Выходная директория: {self.output_dir}")
        self.logger.info(f"Директория для ручной проверки: {self.review_dir}")
        self.logger.info(f"Порог уверенности: {self.confidence_threshold}%")

    def stop(self):
        """
        Остановка сервиса мониторинга
        """
        if not self.running:
            return

        self.observer.stop()
        self.observer.join()
        self.running = False

        self.logger.info("Сервис мониторинга остановлен")

    def is_running(self) -> bool:
        """
        Проверка, запущен ли сервис
        """
        return self.running