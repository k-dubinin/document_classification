"""
Маршруты для классификации документов
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field
import tempfile
import os
from pathlib import Path
import uuid

from prediction.predictor import predict_with_details
from data.document_text import read_text_from_document
from api.schemas.responses import ClassificationResponse, ErrorResponse


class TextRequest(BaseModel):
    text: str = Field(..., example="Пример текста документа для классификации")


router = APIRouter()


@router.post("/classify/text", response_model=ClassificationResponse)
async def classify_text(request: Request, text_request: TextRequest):
    """
    Классификация текста документа
    """
    if not text_request.text or not text_request.text.strip():
        raise HTTPException(status_code=400, detail="Текст документа не может быть пустым")

    try:
        # Получаем доступ к модели из состояния приложения
        app_state = request.app.state
        if not hasattr(app_state, 'model_path'):
            raise HTTPException(status_code=500, detail="Модель не загружена")

        # Выполняем предсказание
        result = predict_with_details(
            text_request.text,
            bundle_path=app_state.model_path,  # Путь к модели из app.state
            top_k=1
        )

        return ClassificationResponse(
            predicted_class=result['label'],
            confidence=result.get('probability_top', result.get('score_top', [[0]]))[0][1] if result.get('probability_top') or result.get('score_top') else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при классификации текста: {str(e)}")


@router.post("/classify/file", response_model=ClassificationResponse)
async def classify_file(request: Request, file: UploadFile = File(...)):
    """
    Классификация файла документа
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не имеет имени")

    # Проверяем расширение файла
    allowed_extensions = {".txt", ".docx", ".pdf", ".rtf", ".html", ".htm", ".odt", ".md"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла: {file_extension}. "
                   f"Поддерживаемые форматы: {', '.join(allowed_extensions)}"
        )

    # Создаем временный файл
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = Path(tempfile.gettempdir()) / temp_filename

    try:
        # Сохраняем загруженный файл во временный файл
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Извлекаем текст из документа
        try:
            text = read_text_from_document(str(temp_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при извлечении текста из файла: {str(e)}")

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Файл не содержит текста или не может быть прочитан")

        # Получаем доступ к модели из состояния приложения
        app_state = request.app.state
        if not hasattr(app_state, 'model_path'):
            raise HTTPException(status_code=500, detail="Модель не загружена")

        # Выполняем предсказание
        result = predict_with_details(
            text,
            bundle_path=app_state.model_path,  # Путь к модели из app.state
            top_k=1
        )

        return ClassificationResponse(
            predicted_class=result['label'],
            confidence=result.get('probability_top', result.get('score_top', [[0]]))[0][1] if result.get('probability_top') or result.get('score_top') else 0
        )

    except HTTPException:
        # Если это HTTPException, просто перебросим его
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")
    finally:
        # Удаляем временный файл
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass  # Игнорируем ошибки при удалении временного файла