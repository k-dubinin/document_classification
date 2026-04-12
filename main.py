"""
Система автоматической классификации текстовых документов (локально, NLP + sklearn).

Тема диплома: разработка системы классификации документов — вход: файлы текстовых форматов
(.txt, .md, .docx, .pdf, .odt, .rtf, .html) или готовая строка (см. data.document_text).

Структура проекта:
  data/          — корпус из папок, поддерживаемые расширения — как у read_text_from_document;
  preprocessing/ — токенизация, стоп-слова, лемматизация (pymorphy2);
  training/      — TF-IDF + Logistic Regression / Naive Bayes / Linear SVM, joblib;
  evaluation/    — accuracy, precision, recall, F1, confusion matrix, report, JSON;
  prediction/    — класс нового документа;
  models/        — сохранённые модели и отчёты.

Простой запуск (из корня проекта, после pip install -r requirements.txt):
  python main.py
  python main.py run
  Обе команды обучают модель на папке data/corpus_txt (подпапки = классы документов).

Другие примеры:
  python main.py train --data-dir data/corpus_txt --model logreg
  python main.py train --hf Adilbai/kz-gov-complaints-data-kz-ru
  python main.py compare --hf Adilbai/kz-gov-complaints-data-kz-ru
  python main.py predict --model models/pipeline_logreg.joblib --file документ.docx
  python main.py predict --model models/pipeline_logreg.joblib --file документ.pdf
  python main.py predict --model models/pipeline_logreg.joblib --file документ.pdf --probs --top-k 5
  python main.py predict --model models/pipeline_logreg.joblib --text "..." --json
  python main.py --config settings/my.yaml train --data-dir data/corpus_txt
  Настройки по умолчанию: settings/default.yaml (или APP_CONFIG=путь).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

# Запуск простым способом: «python main.py» из каталога проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Конфиг YAML/JSON и логирование — до импорта training (подхватываются пороги TF-IDF, OCR, пути)
from settings.loader import init_app, peek_config_path_from_argv

init_app(PROJECT_ROOT, peek_config_path_from_argv())

from evaluation.comparison import print_metrics_comparison_table, rows_for_comparison
from evaluation.evaluate import evaluate_and_report
from data.document_text import read_text_from_document
from preprocessing.text_preprocessor import ensure_nltk_stopwords_downloaded
from prediction.predictor import predict_from_file, predict_with_details
from training import config
from training.persistence import save_model_bundle
from training.train import (
    train_both_models_from_csv,
    train_both_models_from_document_folders,
    train_both_models_from_huggingface,
    train_from_csv,
    train_from_document_folders,
    train_from_huggingface,
)


def _model_output_path_and_title(model_kind: str) -> Tuple[str, str]:
    """Имя файла .joblib и человекочитаемое название модели для отчёта."""
    if model_kind == "logreg":
        return config.FILENAME_VECTORIZER_MODEL_LR, "Logistic Regression"
    if model_kind == "nb":
        return config.FILENAME_VECTORIZER_MODEL_NB, "Naive Bayes"
    if model_kind == "svm":
        return config.FILENAME_VECTORIZER_MODEL_SVM, "Linear SVM"
    raise ValueError(f"Неизвестный тип модели: {model_kind}")


def cmd_train(args: argparse.Namespace) -> None:
    """Обучение одной модели (логистическая регрессия, наивный байес или линейный SVM)."""
    hf_id = getattr(args, "hf_dataset", None)
    if hf_id:
        pipeline, _X_train, _y_train, X_test, y_test, preprocessor = train_from_huggingface(
            dataset_id=hf_id,
            model_kind=args.model,
            split=args.hf_split,
            text_column=args.hf_text_column,
            label_column=args.hf_label_column,
        )
    elif args.data_dir:
        pipeline, _X_train, _y_train, X_test, y_test, preprocessor = train_from_document_folders(
            data_root=args.data_dir,
            model_kind=args.model,
        )
    else:
        pipeline, _X_train, _y_train, X_test, y_test, preprocessor = train_from_csv(
            csv_path=args.csv,
            model_kind=args.model,
            text_column=args.text_column,
            label_column=args.label_column,
        )

    out_name, model_title = _model_output_path_and_title(args.model)
    out_file = os.path.join(args.out, out_name)

    evaluate_and_report(
        pipeline,
        X_test,
        y_test,
        model_name=model_title,
        output_dir=args.out,
        labels_order=pipeline.classes_,
    )
    save_model_bundle(pipeline, preprocessor, out_file)
    print(f"\nМодель сохранена: {out_file}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Обучение обеих моделей и сравнение метрик на одной тестовой выборке."""
    hf_id = getattr(args, "hf_dataset", None)
    if hf_id:
        result = train_both_models_from_huggingface(
            dataset_id=hf_id,
            split=args.hf_split,
            text_column=args.hf_text_column,
            label_column=args.hf_label_column,
        )
    elif args.data_dir:
        result = train_both_models_from_document_folders(data_root=args.data_dir)
    else:
        result = train_both_models_from_csv(
            csv_path=args.csv,
            text_column=args.text_column,
            label_column=args.label_column,
        )

    pipe_lr = result["pipeline_logreg"]
    pipe_nb = result["pipeline_nb"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    preprocessor = result["preprocessor"]

    # Оценка и файлы для логистической регрессии
    evaluate_and_report(
        pipe_lr,
        X_test,
        y_test,
        model_name="Logistic Regression",
        output_dir=args.out,
        labels_order=pipe_lr.classes_,
    )
    save_model_bundle(
        pipe_lr,
        preprocessor,
        os.path.join(args.out, config.FILENAME_VECTORIZER_MODEL_LR),
    )

    # Оценка и файлы для наивного байеса
    evaluate_and_report(
        pipe_nb,
        X_test,
        y_test,
        model_name="Naive Bayes",
        output_dir=args.out,
        labels_order=pipe_nb.classes_,
    )
    save_model_bundle(
        pipe_nb,
        preprocessor,
        os.path.join(args.out, config.FILENAME_VECTORIZER_MODEL_NB),
    )

    rows = rows_for_comparison(
        {
            "LogisticRegression": pipe_lr,
            "MultinomialNB": pipe_nb,
        },
        X_test,
        y_test,
    )
    print_metrics_comparison_table(rows)
    print(f"\nСохранены модели: {config.FILENAME_VECTORIZER_MODEL_LR}, {config.FILENAME_VECTORIZER_MODEL_NB}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Предсказание класса: текст вручную или файл поддерживаемого формата."""
    if args.file:
        try:
            text = read_text_from_document(args.file)
        except (FileNotFoundError, ValueError, ImportError) as e:
            print("Ошибка чтения документа:", e)
            raise SystemExit(1) from e
    else:
        text = args.text

    if not text or not str(text).strip():
        print("Ошибка: пустой текст (не удалось извлечь содержимое или файл пустой).")
        raise SystemExit(1)

    if args.json or args.probs:
        details = predict_with_details(text, args.model, top_k=args.top_k)
        if args.json:
            print(json.dumps(details, ensure_ascii=False, indent=2))
        if args.probs and not args.json:
            print("Предсказанный класс:", details["label"])
            if details.get("probability_top"):
                print("Вероятности по классам (топ-%d):" % args.top_k)
                for item in details["probability_top"]:
                    c, p = item[0], item[1]
                    print(f"  {c}: {p:.4f}")
            elif details.get("score_top"):
                print(
                    "Оценки decision_function (топ-%d; для LinearSVM это не вероятности):"
                    % args.top_k
                )
                for item in details["score_top"]:
                    c, s = item[0], item[1]
                    print(f"  {c}: {s:.4f}")
            else:
                print("(Расширенные оценки для этой модели недоступны.)")
        return

    label, _pipe, _prep = predict_from_file(text, args.model)
    print("Предсказанный класс:", label)


def cmd_run(args: argparse.Namespace) -> None:
    """
    тестове обучение на собственных текстах: обучение на папке из config (по умолчанию data/corpus_txt).

    Сценарий диплома: классы — типы документов или темы обращений; каждый документ — отдельный файл.
    """
    data_root = args.data_dir or os.path.join(PROJECT_ROOT, config.QUICK_START_DATA_DIR)
    if not os.path.isdir(data_root):
        print("Каталог с обучающими документами не найден:", data_root)
        print("Создайте структуру: подпапки с именами классов и файлы .txt / .docx / .pdf внутри,")
        print("или укажите путь: python main.py run --data-dir ВАШ_КАТАЛОГ")
        raise SystemExit(1)

    print("Обучение модели. Корпус:", os.path.abspath(data_root))
    run_ns = argparse.Namespace(
        hf_dataset=None,
        data_dir=data_root,
        csv=None,
        model=args.model,
        out=args.out,
        text_column=config.CSV_TEXT_COLUMN,
        label_column=config.CSV_LABEL_COLUMN,
        hf_split=config.HF_DEFAULT_SPLIT,
        hf_text_column=config.HF_DEFAULT_TEXT_COLUMN,
        hf_label_column=config.HF_DEFAULT_LABEL_COLUMN,
    )
    cmd_train(run_ns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Классификация русскоязычных текстов (TF-IDF + классические модели).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML или JSON с настройками (иначе APP_CONFIG или settings/default.yaml). "
        "Укажите до имени команды, например: python main.py --config x.yaml train ...",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="Быстрое обучение на папке по умолчанию (см. config.QUICK_START_DATA_DIR)",
    )
    p_run.add_argument(
        "--data-dir",
        default=None,
        help="Каталог с классами-подпапками и документами (если не задан — из config)",
    )
    p_run.add_argument(
        "--model",
        choices=["logreg", "nb", "svm"],
        default=config.QUICK_START_MODEL,
        help="Тип модели (по умолчанию из config.QUICK_START_MODEL)",
    )
    p_run.add_argument("--out", default=config.DEFAULT_MODELS_DIR, help="Куда сохранить модель и метрики")
    p_run.set_defaults(func=cmd_run)

    p_train = sub.add_parser(
        "train",
        help="Обучить одну модель: CSV, папки с документами или набор Hugging Face",
    )
    src_train = p_train.add_mutually_exclusive_group(required=True)
    src_train.add_argument("--csv", help="Путь к CSV с колонками text и label")
    src_train.add_argument(
        "--data-dir",
        help="Каталог: подпапка = класс, внутри — .txt, .docx, .pdf и др.",
    )
    src_train.add_argument(
        "--hf",
        dest="hf_dataset",
        metavar="DATASET",
        help="Набор Hugging Face, напр. Adilbai/kz-gov-complaints-data-kz-ru (нужен pip install datasets)",
    )
    p_train.add_argument(
        "--model",
        choices=["logreg", "nb", "svm"],
        default="logreg",
        help="logreg — логистическая регрессия, nb — наивный байес, svm — линейный SVM",
    )
    p_train.add_argument(
        "--out",
        default=config.DEFAULT_MODELS_DIR,
        help="Каталог для сохранения модели, метрик и PNG матрицы ошибок",
    )
    p_train.add_argument("--text-column", default=config.CSV_TEXT_COLUMN, help="Имя столбца с текстом (CSV)")
    p_train.add_argument("--label-column", default=config.CSV_LABEL_COLUMN, help="Имя столбца с меткой (CSV)")
    p_train.add_argument("--hf-split", default=config.HF_DEFAULT_SPLIT, help="Сплит Hugging Face, обычно train")
    p_train.add_argument(
        "--hf-text-column",
        default=config.HF_DEFAULT_TEXT_COLUMN,
        help="Столбец с текстом в HF (для обращений часто text_ru)",
    )
    p_train.add_argument(
        "--hf-label-column",
        default=config.HF_DEFAULT_LABEL_COLUMN,
        help="Столбец с классом в HF (для примера — category)",
    )
    p_train.set_defaults(func=cmd_train)

    p_cmp = sub.add_parser(
        "compare",
        help="Обучить LogisticRegression и Naive Bayes на одних данных и сравнить метрики",
    )
    src_cmp = p_cmp.add_mutually_exclusive_group(required=True)
    src_cmp.add_argument("--csv", help="Путь к CSV")
    src_cmp.add_argument("--data-dir", help="Каталог: подпапка = класс, внутри — документы")
    src_cmp.add_argument(
        "--hf",
        dest="hf_dataset",
        metavar="DATASET",
        help="Набор Hugging Face для сравнения двух моделей",
    )
    p_cmp.add_argument("--out", default=config.DEFAULT_MODELS_DIR, help="Каталог для артефактов")
    p_cmp.add_argument("--text-column", default=config.CSV_TEXT_COLUMN)
    p_cmp.add_argument("--label-column", default=config.CSV_LABEL_COLUMN)
    p_cmp.add_argument("--hf-split", default=config.HF_DEFAULT_SPLIT)
    p_cmp.add_argument("--hf-text-column", default=config.HF_DEFAULT_TEXT_COLUMN)
    p_cmp.add_argument("--hf-label-column", default=config.HF_DEFAULT_LABEL_COLUMN)
    p_cmp.set_defaults(func=cmd_compare)

    p_pred = sub.add_parser("predict", help="Предсказать класс для текста документа")
    p_pred.add_argument("--model", required=True, help="Путь к .joblib (результат обучения)")
    src_pred = p_pred.add_mutually_exclusive_group(required=True)
    src_pred.add_argument("--text", help="Текст документа")
    src_pred.add_argument(
        "--file",
        help="Путь к файлу: .txt, .md, .docx, .pdf (в т.ч. сканы при OCR), .odt, .rtf, .html",
    )
    p_pred.add_argument(
        "--probs",
        action="store_true",
        help="Вывести вероятности по классам (predict_proba) или оценки SVM (decision_function)",
    )
    p_pred.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Сколько лучших классов показать с --probs",
    )
    p_pred.add_argument(
        "--json",
        action="store_true",
        help="Печать результата одним JSON (с вероятностями или оценками, если модель поддерживает)",
    )
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    ensure_nltk_stopwords_downloaded()
    if len(sys.argv) == 1:
        sys.argv.append("run")
    parser = build_parser()
    args = parser.parse_args()
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        init_app(PROJECT_ROOT, cfg_path)
    args.func(args)


if __name__ == "__main__":
    main()
