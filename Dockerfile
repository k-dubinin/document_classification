# базовый образ Python 3.11 slim
FROM python:3.11-slim

# рабочая директория
WORKDIR /app

# ENV переменные
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MPLBACKEND=Agg

# Устанавливаем системные зависимости, включая Tesseract OCR и Poppler (для ообработки PDF)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    poppler-utils \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Python зависимости
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Открываем порт 8501 для Streamlit
EXPOSE 8501

# Команда запуска приложения
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]