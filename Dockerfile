# Используем базовый образ Python 3.11 slim
FROM python:3.11-slim-bookworm

# Установка Tesseract OCR и Poppler для работы с PDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование остальных файлов
COPY . .

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV MPLBACKEND=Agg

# Открытие порта 8501 для Streamlit
EXPOSE 8501

# Запуск приложения
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0"]