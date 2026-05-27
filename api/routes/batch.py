"""
Маршрут для пакетной классификации документов
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from pydantic import BaseModel
import zipfile
import tempfile
import os
import shutil
from pathlib import Path
import uuid

from services.batch_classifier import classify_directory
from api.schemas.responses import BatchResponse, ErrorResponse


class BatchRequest(BaseModel):
    threshold: float = 20.0  # Порог вероятности для ручной проверки


router = APIRouter()


def collect_batch_results(model_path: str, input_dir: str, output_dir: str, threshold: float):
    """
    Обертка для classify_directory, которая собирает результаты и возвращает статистику
    """
    # Переменные для отслеживания статистики
    processed = 0
    ok_count = 0
    review_count = 0
    err_count = 0
    all_results = []

    # Обработка директории
    for res in classify_directory(
        model_path,
        input_dir,
        output_dir,
        recursive=True,
        top_k=1,
        manual_review_probability_threshold=threshold / 100.0,
    ):
        all_results.append(res)
        processed += 1

        if res.ok:
            if res.manual_review_required == "yes":
                review_count += 1
            else:
                ok_count += 1
        else:
            err_count += 1

    return {
        "processed": processed,
        "ok_count": ok_count,
        "review_count": review_count,
        "err_count": err_count,
        "all_results": all_results
    }


@router.post("/batch", response_model=BatchResponse)
async def batch_classify(
    request: Request,
    zip_file: UploadFile = File(None),
    threshold: float = Query(20.0, description="Порог вероятности для ручной проверки"),
    input_dir: str = Query(None, description="Входная директория с документами (вместо ZIP-архива)")
):
    """
    Пакетная классификация документов из ZIP-архива или из директории
    """
    # Создаем временные директории
    temp_dir = Path(tempfile.gettempdir()) / f"temp_batch_{uuid.uuid4()}"
    input_dir_path = temp_dir / "input"
    output_dir = temp_dir / "output"

    input_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if zip_file is not None:
            # Обработка случая с ZIP-архивом
            if not zip_file.filename or not zip_file.filename.endswith('.zip'):
                raise HTTPException(status_code=400, detail="Ожидается ZIP-архив")

            # Сохраняем загруженный ZIP-файл
            zip_path = temp_dir / zip_file.filename
            with open(zip_path, "wb") as f:
                content = await zip_file.read()
                f.write(content)

            # Распаковываем ZIP-архив
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(input_dir_path)
        elif input_dir is not None:
            # Обработка случая с директорией
            # Копируем содержимое указанной директории во временную
            source_path = Path(input_dir)
            if not source_path.exists() or not source_path.is_dir():
                raise HTTPException(status_code=400, detail=f"Директория не существует или не является директорией: {input_dir}")

            # Копируем содержимое директории
            for item in source_path.iterdir():
                dest_item = input_dir_path / item.name
                if item.is_file():
                    shutil.copy2(item, dest_item)
                elif item.is_dir():
                    shutil.copytree(item, dest_item)
        else:
            raise HTTPException(status_code=400, detail="Необходимо указать либо ZIP-архив, либо входную директорию")

        # Получаем доступ к пути модели из состояния приложения
        app_state = request.app.state
        if not hasattr(app_state, 'model_path'):
            raise HTTPException(status_code=500, detail="Модель не загружена")

        # Запускаем пакетную классификацию
        result = collect_batch_results(
            model_path=app_state.model_path,  # Путь к модели
            input_dir=str(input_dir_path),
            output_dir=str(output_dir),
            threshold=threshold
        )

        # Формируем ответ
        response_data = {
            "processed": result.get("processed", 0),
            "low_confidence": result.get("review_count", 0),
            "errors": result.get("err_count", 0),
            "output_dir": str(output_dir)
        }

        return BatchResponse(**response_data)

    except HTTPException:
        # Если это HTTPException, просто перебросим его
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при пакетной обработке: {str(e)}")
    finally:
        # Удаляем временные файлы
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass