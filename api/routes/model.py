"""
Маршрут для получения информации о загруженной модели и управления watch-сервисом
"""
from fastapi import APIRouter, Request, Query
from api.schemas.responses import ModelInfoResponse
import re
import os
from pathlib import Path
import joblib
import threading
from typing import Dict

from services.watch_service import WatchService


router = APIRouter()

# Глобальный экземпляр watch-сервиса
watch_service_instance = None
watch_service_lock = threading.Lock()


def extract_real_model_type_from_pipeline(pipeline):
    """
    Извлекает реальный тип модели из внутренней структуры пайплайна
    """
    # Если это пайплайн sklearn, ищем классификатор внутри
    if hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps:
        clf = pipeline.named_steps['clf']
        return clf.__class__.__name__
    elif hasattr(pipeline, 'steps'):
        # Проходим по шагам пайплайна и ищем классификатор
        for step_name, step_obj in pipeline.steps:
            if 'clf' in step_name.lower() or 'classifier' in step_name.lower():
                return step_obj.__class__.__name__
    
    # Если классификатор не найден, возвращаем общий тип пайплайна
    return pipeline.__class__.__name__


def extract_model_type_from_filename(filename):
    """
    Извлекает имя модели из имени файла без расширения
    """
    # Извлекаем базовое имя файла без расширения
    base_name = os.path.splitext(filename)[0].lower()
    
    # Если имя начинается с 'pipeline_', убираем этот префикс
    if base_name.startswith('pipeline_'):
        return base_name[9:]  # Убираем 'pipeline_'
    else:
        return base_name


def extract_pipeline_from_loaded_object(loaded_obj):
    """
    Извлекает объект пайплайна из загруженного объекта, вне зависимости от его структуры
    """
    if isinstance(loaded_obj, tuple):
        # Если это кортеж, предполагаем, что первый элемент - это пайплайн
        return loaded_obj[0] if len(loaded_obj) > 0 else loaded_obj
    elif isinstance(loaded_obj, dict):
        # Если это словарь, ищем ключи, связанные с пайплайном
        if 'pipeline' in loaded_obj:
            return loaded_obj['pipeline']
        elif 'model' in loaded_obj:
            return loaded_obj['model']
        else:
            # Если нет очевидного ключа, возвращаем первый элемент, если он есть
            return next(iter(loaded_obj.values()), loaded_obj)
    else:
        # В противном случае возвращаем сам объект
        return loaded_obj


def get_model_info_by_path(model_path: str):
    """
    Получает информацию о модели по пути к файлу
    """
    if not os.path.exists(model_path):
        return None
        
    try:
        loaded_obj = joblib.load(model_path)
        
        # Извлекаем пайплайн из загруженного объекта
        pipeline = extract_pipeline_from_loaded_object(loaded_obj)
        
        # Получаем имя класса модели
        actual_model_type = extract_real_model_type_from_pipeline(pipeline)
        
        # Извлекаем тип модели из имени файла
        model_filename = os.path.basename(model_path)
        filename_model_type = extract_model_type_from_filename(model_filename)
        
        # Определяем тип модели: сначала используем имя из файла, потом реальный тип
        if filename_model_type:
            # Если извлекли из имени файла, используем его как основное имя
            model_type = f"{filename_model_type} ({actual_model_type})"
        else:
            # Если не удалось извлечь из имени файла, используем только реальный тип
            model_type = actual_model_type
        
        # Получаем классы, если они доступны
        classes = []
        if hasattr(pipeline, 'classes_'):
            try:
                classes = pipeline.classes_.tolist() if hasattr(pipeline.classes_, 'tolist') else list(pipeline.classes_)
            except:
                classes = []
        
        return {"model": model_type, "classes": classes}
    except Exception as e:
        return None


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request):
    """
    Возвращает информацию о загруженной модели
    """
    # Получаем доступ к модели из состояния приложения
    app_state = request.app.state
    if not hasattr(app_state, 'pipeline'):
        return {"model": "none", "classes": []}
    
    # Извлекаем пайплайн из загруженного объекта
    pipeline = extract_pipeline_from_loaded_object(app_state.pipeline)
    
    # Получаем имя класса модели
    actual_model_type = extract_real_model_type_from_pipeline(pipeline)
    
    # Извлекаем тип модели из имени файла
    model_filename = ""
    if hasattr(app_state, 'model_path'):
        model_filename = os.path.basename(app_state.model_path)
    
    filename_model_type = extract_model_type_from_filename(model_filename)
    
    # Определяем тип модели: сначала используем имя из файла, потом реальный тип
    if filename_model_type:
        # Если извлекли из имени файла, используем его как основное имя
        model_type = f"{filename_model_type} ({actual_model_type})"
    else:
        # Если не удалось извлечь из имени файла, используем только реальный тип
        model_type = actual_model_type
    
    # Получаем классы, если они доступны
    classes = []
    if hasattr(pipeline, 'classes_'):
        try:
            classes = pipeline.classes_.tolist() if hasattr(pipeline.classes_, 'tolist') else list(pipeline.classes_)
        except:
            classes = []
    
    return {"model": model_type, "classes": classes}


@router.get("/model/info-by-name", response_model=ModelInfoResponse)
async def model_info_by_name(model_name: str = Query(..., description="Имя файла модели из директории models")):
    """
    Возвращает информацию о модели по имени файла
    """
    # Проверяем, что имя файла содержит .joblib
    if not model_name.endswith('.joblib'):
        model_name += '.joblib'
    
    # Формируем путь к модели
    model_path = os.path.join("models", model_name)
    
    # Получаем информацию о модели
    model_info = get_model_info_by_path(model_path)
    
    if model_info is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Модель не найдена: {model_path}")
    
    return model_info


@router.post("/watch/start")
async def start_watch_service(
    model_path: str = Query(..., description="Путь к .joblib файлу модели"),
    input_dir: str = Query(..., description="Входная директория для мониторинга"),
    output_dir: str = Query(..., description="Выходная директория для классифицированных файлов"),
    review_dir: str = Query(..., description="Директория для файлов на ручной проверке"),
    confidence_threshold: float = Query(20.0, description="Порог уверенности для ручной проверки (в процентах)"),
    recursive: bool = Query(True, description="Мониторить подпапки рекурсивно"),
    polling_interval: float = Query(1.0, description="Интервал проверки файловой системы (в секундах)")
):
    """
    Запускает сервис мониторинга директории для автоматической классификации документов
    """
    global watch_service_instance
    
    with watch_service_lock:
        if watch_service_instance and watch_service_instance.is_running():
            return {"status": "error", "message": "Сервис мониторинга уже запущен"}
        
        try:
            # Настройка логирования
            import logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            
            # Создаем и запускаем сервис мониторинга
            watch_service_instance = WatchService(
                model_path=model_path,
                input_dir=input_dir,
                output_dir=output_dir,
                review_dir=review_dir,
                confidence_threshold=confidence_threshold,
                recursive=recursive,
                polling_interval=polling_interval,
                logger=logger
            )
            
            watch_service_instance.start()
            
            return {
                "status": "success", 
                "message": "Сервис мониторинга успешно запущен",
                "input_dir": input_dir,
                "output_dir": output_dir,
                "review_dir": review_dir
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при запуске сервиса мониторинга: {str(e)}"}


@router.post("/watch/stop")
async def stop_watch_service():
    """
    Останавливает сервис мониторинга директории
    """
    global watch_service_instance
    
    with watch_service_lock:
        if not watch_service_instance or not watch_service_instance.is_running():
            return {"status": "error", "message": "Сервис мониторинга не запущен"}
        
        try:
            watch_service_instance.stop()
            return {"status": "success", "message": "Сервис мониторинга успешно остановлен"}
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при остановке сервиса мониторинга: {str(e)}"}


@router.get("/watch/status")
async def watch_service_status():
    """
    Возвращает статус сервиса мониторинга директории
    """
    global watch_service_instance
    
    if watch_service_instance and watch_service_instance.is_running():
        return {
            "status": "running",
            "message": "Сервис мониторинга активен",
            "input_dir": str(watch_service_instance.input_dir),
            "output_dir": str(watch_service_instance.output_dir),
            "review_dir": str(watch_service_instance.review_dir)
        }
    else:
        return {"status": "stopped", "message": "Сервис мониторинга не запущен"}