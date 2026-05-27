"""
Схемы ответов API
"""
from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    predicted_class: str = Field(..., example="Пример класса")
    confidence: float = Field(..., example=0.95)


class BatchResponse(BaseModel):
    """
    Схема ответа для результатов пакетной обработки.
    
    Attributes:
        processed (int): Общее количество обработанных документов.
        low_confidence (int): Количество документов с низкой уверенностью в результате.
        errors (int): Количество документов, обработанных с ошибками.
        output_dir (str): Директория, в которой сохранены результаты обработки.
    """
    processed: int
    low_confidence: int
    errors: int
    output_dir: str


class ErrorResponse(BaseModel):
    error: str = Field(..., example="Пример сообщения об ошибке")


class HealthResponse(BaseModel):
    status: str = Field(..., example="OK")


class ModelInfoResponse(BaseModel):
    """
    Схема ответа для информации о модели классификации.
    
    Attributes:
        model (str): Имя/идентификатор используемой модели.
        classes (list): Список известных классов, которые может определять модель.
    """
    model: str
    classes: list