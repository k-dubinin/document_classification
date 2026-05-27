"""
Тестовый скрипт для проверки функциональности watch-сервиса
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь Python для импорта модулей
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.watch_service import WatchService
import logging


def test_watch_service_creation():
    """
    Тест создания экземпляра WatchService
    """
    print("Тест: Создание экземпляра WatchService")
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Создаем сервис с фиктивными параметрами для теста
        service = WatchService(
            model_path="models/fake_model.joblib",  # будет проверен при запуске
            input_dir="test_input",
            output_dir="test_output",
            review_dir="test_review",
            confidence_threshold=20.0,
            recursive=True,
            polling_interval=1.0,
            logger=logger
        )
        
        print(f"Сервис создан успешно")
        print(f"Входная директория: {service.input_dir}")
        print(f"Выходная директория: {service.output_dir}")
        print(f"Директория для проверки: {service.review_dir}")
        print(f"Порог уверенности: {service.confidence_threshold}%")
        
        # Проверяем, что директории были созданы
        assert service.input_dir.exists(), "Входная директория должна быть создана"
        assert service.output_dir.exists(), "Выходная директория должна быть создана"
        assert service.review_dir.exists(), "Директория для проверки должна быть создана"
        
        print("Все директории успешно созданы")
        
        # Проверяем состояние сервиса
        assert not service.is_running(), "Сервис не должен быть запущен сразу после создания"
        
        print("Тест пройден успешно!")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        raise


if __name__ == "__main__":
    test_watch_service_creation()