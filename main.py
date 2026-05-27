"""
Точка входа в приложение.

Поддерживаемые режимы:
- Интерактивный запуск Streamlit UI
- CLI-режим с подкомандами
"""
import os
import sys
import time
import click
from pathlib import Path

# Добавляем корневую директорию в путь Python для импорта модулей
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from settings.loader import init_app, get_setting
from training.train import train_and_save_model, compare_models
from prediction.predictor import predict_from_file, predict_from_text
from services.batch_classifier import batch_classify_directory
from evaluation.evaluate import evaluate_and_report
from services.watch_service import WatchService
import logging


# Инициализируем приложение с загрузкой конфигурации
init_app(str(PROJECT_ROOT))


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Путь к YAML/JSON конфигурационному файлу')
def cli(config):
    """Консольный интерфейс для системы классификации документов."""
    if config:
        init_app(str(PROJECT_ROOT), config_path=config)


@cli.command()
@click.option('--data-dir', type=click.Path(exists=True), default=None,
              help='Путь к директории с подпапками-классами (если не задан — из конфига)')
@click.option('--model', type=click.Choice(['logreg', 'nb', 'svm']), default='logreg',
              help='Тип модели: logreg (Logistic Regression), nb (Naive Bayes), svm (SVM)')
@click.option('--out', type=click.Path(), default=None,
              help='Директория для сохранения модели и отчётов (по умолчанию из конфига)')
def run(data_dir, model, out):
    """
    Быстрый запуск: обучение модели на папке из конфига и классификация тестового документа.
    """
    # Получаем настройки
    if data_dir is None:
        data_dir = get_setting('QUICK_START_DATA_DIR', 'data/corpus_txt')
    if out is None:
        out = get_setting('OUTPUT_DIR', 'models')

    print(f"Быстрый запуск: обучение {model} на {data_dir}, сохранение в {out}")
    
    # Обучаем модель
    start_time = time.time()
    model_path = train_and_save_model(
        data_dir=data_dir,
        model_type=model,
        output_dir=out
    )
    training_time = time.time() - start_time
    
    print(f"Модель обучена и сохранена: {model_path}")
    print(f"Время обучения: {training_time:.2f} секунд")
    
    # Выполняем тестовую классификацию
    sample_text = "Пример текста для классификации."
    result = predict_from_text(sample_text, model_path)
    print(f"Тестовая классификация: {result}")


@cli.command()
@click.option('--csv', type=click.Path(exists=True), help='Путь к CSV файлу')
@click.option('--data-dir', type=click.Path(exists=True), help='Путь к директории с подпапками-классами')
@click.option('--hf', type=click.STRING, help='Идентификатор датасета на Hugging Face')
@click.option('--model', type=click.Choice(['logreg', 'nb', 'svm'], case_sensitive=False), required=True,
              help='Тип модели: logreg (Logistic Regression), nb (Naive Bayes), svm (SVM)')
@click.option('--out', type=click.Path(), default=None,
              help='Директория для сохранения модели и отчётов (по умолчанию из конфига)')
@click.option('--text-column', default='text', help='Название столбца с текстом (для CSV)')
@click.option('--label-column', default='label', help='Название столбца с метками (для CSV)')
@click.option('--hf-split', default='train', help='Название сплита в датасете Hugging Face')
@click.option('--hf-text-column', default='text', help='Название столбца с текстом в датасете Hugging Face')
@click.option('--hf-label-column', default='label', help='Название столбца с метками в датасете Hugging Face')
def train(csv, data_dir, hf, model, out, text_column, label_column, hf_split, hf_text_column, hf_label_column):
    """
    Обучение одной модели на одном источнике данных.
    """
    if sum(bool(x) for x in [csv, data_dir, hf]) != 1:
        raise click.UsageError("Необходимо указать ровно один источник данных: --csv, --data-dir или --hf")

    if out is None:
        out = get_setting('OUTPUT_DIR', 'models')

    print(f"Обучение модели {model}...")
    
    start_time = time.time()
    model_path = train_and_save_model(
        csv_path=csv,
        data_dir=data_dir,
        hf_dataset=hf,
        model_type=model,
        output_dir=out,
        text_column=text_column,
        label_column=label_column,
        hf_split=hf_split,
        hf_text_column=hf_text_column,
        hf_label_column=hf_label_column
    )
    training_time = time.time() - start_time
    
    print(f"Модель обучена и сохранена: {model_path}")
    print(f"Время обучения: {training_time:.2f} секунд")
    
    # Оценка модели
    print("Оценка модели...")
    evaluate_and_report(model_path, training_time)


@cli.command()
@click.option('--csv', type=click.Path(exists=True), help='Путь к CSV файлу')
@click.option('--data-dir', type=click.Path(exists=True), help='Путь к директории с подпапками-классами')
@click.option('--hf', type=click.STRING, help='Идентификатор датасета на Hugging Face')
@click.option('--out', type=click.Path(), default=None,
              help='Директория для сохранения моделей и отчётов (по умолчанию из конфига)')
@click.option('--text-column', default='text', help='Название столбца с текстом (для CSV)')
@click.option('--label-column', default='label', help='Название столбца с метками (для CSV)')
@click.option('--hf-split', default='train', help='Название сплита в датасете Hugging Face')
@click.option('--hf-text-column', default='text', help='Название столбца с текстом в датасете Hugging Face')
@click.option('--hf-label-column', default='label', help='Название столбца с метками в датасете Hugging Face')
def compare(csv, data_dir, hf, out, text_column, label_column, hf_split, hf_text_column, hf_label_column):
    """
    Обучение и сравнение двух моделей: Logistic Regression и Naive Bayes.
    """
    if sum(bool(x) for x in [csv, data_dir, hf]) != 1:
        raise click.UsageError("Необходимо указать ровно один источник данных: --csv, --data-dir или --hf")

    if out is None:
        out = get_setting('OUTPUT_DIR', 'models')

    print("Обучение и сравнение моделей Logistic Regression и Naive Bayes...")
    
    start_time = time.time()
    results = compare_models(
        csv_path=csv,
        data_dir=data_dir,
        hf_dataset=hf,
        output_dir=out,
        text_column=text_column,
        label_column=label_column,
        hf_split=hf_split,
        hf_text_column=hf_text_column,
        hf_label_column=hf_label_column
    )
    training_time = time.time() - start_time
    
    print(f"Модели обучены и сохранены: {results}")
    print(f"Время обучения: {training_time:.2f} секунд")


@cli.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Путь к .joblib файлу модели')
@click.option('--text', type=click.STRING, help='Текст документа для классификации')
@click.option('--file', type=click.Path(exists=True), help='Путь к файлу документа для классификации')
@click.option('--probs/--no-probs', default=False, help='Показать вероятности/оценки для топ-K классов')
@click.option('--top-k', type=int, default=5, help='Количество классов для отображения с вероятностями')
@click.option('--json/--no-json', default=False, help='Вывести результат в формате JSON')
def predict(model, text, file, probs, top_k, json):
    """
    Классификация одного документа.
    """
    if not text and not file:
        raise click.UsageError("Необходимо указать либо --text, либо --file")
    if text and file:
        raise click.UsageError("Необходимо указать только один из --text или --file")

    if file:
        result = predict_from_file(file, model, top_k=top_k if probs else 1)
    else:
        result = predict_from_text(text, model, top_k=top_k if probs else 1)

    if json:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if 'label' in result:
            print(f"Предсказанный класс: {result['label']}")
        if 'probability' in result:
            print(f"Вероятность: {result['probability']:.2%}")
        elif 'score' in result:
            print(f"Оценка: {result['score']:.3f}")
        
        if probs:
            if 'probability_top' in result:
                print("\nТоп-{} классов с вероятностями:".format(top_k))
                for label, prob in result['probability_top']:
                    print(f"  {label}: {prob:.2%}")
            elif 'score_top' in result:
                print("\nТоп-{} классов с оценками:".format(top_k))
                for label, score in result['score_top']:
                    print(f"  {label}: {score:.3f}")


@cli.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Путь к .joblib файлу модели')
@click.option('--input-dir', type=click.Path(exists=True), required=True, help='Входная директория с документами')
@click.option('--output-dir', type=click.Path(), required=True, help='Выходная директория для классифицированных файлов')
@click.option('--threshold', type=float, default=20.0, 
              help='Порог вероятности для ручной проверки (в процентах, по умолчанию 20)')
@click.option('--recursive/--no-recursive', default=True, help='Обрабатывать файлы рекурсивно в подпапках')
def batch(model, input_dir, output_dir, threshold, recursive):
    """
    Пакетная классификация документов из директории.
    """
    print(f"Пакетная классификация:")
    print(f"  Модель: {model}")
    print(f"  Входная директория: {input_dir}")
    print(f"  Выходная директория: {output_dir}")
    print(f"  Порог вероятности: {threshold}%")
    print(f"  Рекурсивная обработка: {recursive}")
    
    # Выполняем пакетную классификацию
    for result in batch_classify_directory(
        model_path=model,
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        top_k=1,
        manual_review_probability_threshold=threshold / 100.0
    ):
        if result.ok:
            if result.manual_review_required == "yes":
                print(f"  [!] {result.file_path} -> Требует проверки (вероятность: {result.probability:.2%})")
            else:
                print(f"  [+] {result.file_path} -> {result.predicted_class} (вероятность: {result.probability:.2%})")
        else:
            print(f"  [-] {result.file_path} -> ОШИБКА: {result.error_message}")


@cli.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Путь к .joblib файлу модели')
@click.option('--input-dir', type=click.Path(exists=True), required=True, help='Входная директория для мониторинга')
@click.option('--output-dir', type=click.Path(), default=None, help='Выходная директория (по умолчанию: output/classified)')
@click.option('--review-dir', type=click.Path(), default=None, help='Директория для файлов на ручной проверке (по умолчанию: output/manual_review)')
@click.option('--threshold', type=float, default=20.0, 
              help='Порог уверенности для ручной проверки (в процентах, по умолчанию 20)')
@click.option('--recursive/--no-recursive', default=True, help='Мониторить подпапки рекурсивно')
@click.option('--poll-interval', type=float, default=1.0, help='Интервал проверки файловой системы (в секундах)')
def watch(model, input_dir, output_dir, review_dir, threshold, recursive, poll_interval):
    """
    Запуск режима мониторинга директории для автоматической классификации документов.
    """
    # Устанавливаем значения по умолчанию
    if output_dir is None:
        output_dir = "output/classified"
    if review_dir is None:
        review_dir = "output/manual_review"
    
    print(f"Запуск режима мониторинга:")
    print(f"  Модель: {model}")
    print(f"  Входная директория: {input_dir}")
    print(f"  Выходная директория: {output_dir}")
    print(f"  Директория для ручной проверки: {review_dir}")
    print(f"  Порог уверенности: {threshold}%")
    print(f"  Рекурсивный мониторинг: {recursive}")
    print(f"  Интервал проверки: {poll_interval} сек.")
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Создаем и запускаем сервис мониторинга
    service = WatchService(
        model_path=model,
        input_dir=input_dir,
        output_dir=output_dir,
        review_dir=review_dir,
        confidence_threshold=threshold,
        recursive=recursive,
        polling_interval=poll_interval,
        logger=logger
    )
    
    try:
        service.start()
        print("Сервис мониторинга запущен. Нажмите Ctrl+C для остановки.")
        
        # Бесконечный цикл до прерывания
        while service.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nПолучен сигнал остановки...")
        service.stop()
        print("Сервис мониторинга остановлен.")


if __name__ == "__main__":
    # Если запускается без подкоманды, используем команду run по умолчанию
    if len(sys.argv) == 1:
        sys.argv.insert(1, 'run')
    
    cli()