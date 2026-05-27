"""
REST API для системы автоматической классификации русскоязычных документов.
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь Python для импорта модулей
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
from typing import Optional

from settings.loader import init_app, get_setting
from training import config as training_config


# Инициализируем приложение с загрузкой конфигурации
init_app(str(PROJECT_ROOT))

# Получаем настройки API (если не заданы, используем значения по умолчанию)
API_HOST = get_setting('API.HOST', '0.0.0.0')
API_PORT = get_setting('API.PORT', 8000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер жизненного цикла приложения.
    Загружает модель при старте и освобождает ресурсы при завершении.
    """
    # Загрузка модели при старте приложения
    # Попробуем использовать путь к модели по умолчанию
    model_path = 'models/pipeline_logreg.joblib'
    if not os.path.exists(model_path):
        print(f"Модель не найдена по пути: {model_path}")
        # Попробуем найти любую модель в директории models
        models_dir = PROJECT_ROOT / "models"
        model_files = list(models_dir.glob("*.joblib"))
        if model_files:
            model_path = str(model_files[0])
            print(f"Найдена модель: {model_path}")
        else:
            print("Не найдено ни одной модели в директории models/")
            yield
            return

    try:
        loaded_obj = joblib.load(model_path)
        # Проверяем структуру загруженного объекта
        if isinstance(loaded_obj, dict):

            app.state.pipeline = loaded_obj.get('pipeline')
            app.state.preprocessor = loaded_obj.get('preprocessor')
        elif isinstance(loaded_obj, tuple) and len(loaded_obj) >= 2:

            app.state.pipeline, app.state.preprocessor = loaded_obj
        else:
            raise ValueError(f"Неизвестная структура модели: {type(loaded_obj)}")

        app.state.model_path = model_path  # Сохраняем путь к модели для использования в predict_with_details
        pass
    except Exception as e:
        # Обработка ошибки загрузки модели
        yield
        return

    yield  # Здесь выполняется работа приложения

    # Освобождение ресурсов при завершении
    if hasattr(app.state, 'pipeline'):
        delattr(app.state, 'pipeline')
    if hasattr(app.state, 'preprocessor'):
        delattr(app.state, 'preprocessor')
    if hasattr(app.state, 'model_path'):
        delattr(app.state, 'model_path')


app = FastAPI(
    title="API системы классификации документов",
    description="REST API для автоматической классификации русскоязычных документов",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Импортируем маршруты после определения app
from api.routes import classify, batch, health, model

# Подключаем маршруты
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(model.router, prefix="/api/v1", tags=["model"])
app.include_router(classify.router, prefix="/api/v1", tags=["classification"])
app.include_router(batch.router, prefix="/api/v1", tags=["batch"])


@app.get("/")
async def root():
    return {"message": "API системы классификации документов", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )