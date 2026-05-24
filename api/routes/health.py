"""
Маршрут проверки работоспособности API
"""
from fastapi import APIRouter, Request
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model: str
    classes: list


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Возвращает статус сервера
    """
    return {"status": "ok"}


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request):
    """
    Возвращает информацию о загруженной модели
    """
    # Получаем доступ к модели из состояния приложения
    app_state = request.app.state
    if not hasattr(app_state, 'pipeline'):
        return {"model": "none", "classes": []}
    
    # Получаем имя модели и классы
    pipeline = app_state.pipeline
    
    # Определяем тип модели
    model_type = type(pipeline).__name__
    
    # Получаем классы, если они доступны
    classes = []
    if hasattr(pipeline, 'classes_'):
        try:
            classes = pipeline.classes_.tolist() if hasattr(pipeline.classes_, 'tolist') else list(pipeline.classes_)
        except:
            classes = []
    
    return {"model": model_type, "classes": classes}