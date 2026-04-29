"""
UI для системы классификации документов.

Запуск:
  streamlit run ui/app.py

UI использует уже реализованные модули проекта:
  - data.document_text: извлечение текста из файлов
  - training.train: обучение по CSV / папкам / Hugging Face
  - evaluation.evaluate: метрики + сохранение JSON/PNG
  - training.persistence: сохранение модели joblib
  - prediction.predictor: предсказание + вероятности/оценки
  - services.batch_classifier: пакетная классификация директории + CSV-отчёт
  - training.config: константы и настройки
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import plotly.express as px
import streamlit as st

from data.document_text import read_text_from_document
from evaluation.evaluate import evaluate_and_report
from prediction.predictor import predict_with_details
from services.batch_classifier import (
    BatchItemResult,
    classify_directory,
    iter_document_paths,
    write_batch_report_csv,
)
from training import config
from training.persistence import save_model_bundle
from training.train import (
    train_from_csv,
    train_from_document_folders,
    train_from_huggingface,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_LABELS_RU: Dict[str, str] = {
    "logreg": "Логистическая регрессия (logreg)",
    "nb": "Наивный байес (nb)",
    "svm": "Линейный SVM (svm)",
}

MODEL_FILES_HINTS: Dict[str, str] = {
    config.FILENAME_VECTORIZER_MODEL_LR: "Логистическая регрессия (logreg)",
    config.FILENAME_VECTORIZER_MODEL_NB: "Наивный байес (nb)",
    config.FILENAME_VECTORIZER_MODEL_SVM: "Линейный SVM (svm)",
}


def _models_dir() -> str:
    return str(PROJECT_ROOT / config.DEFAULT_MODELS_DIR)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_model_files(models_dir: str) -> list[Path]:
    d = Path(models_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.joblib"), key=lambda p: p.name.lower())


def _display_name_for_model_file(path: Path) -> str:
    hint = MODEL_FILES_HINTS.get(path.name)
    if hint:
        return f"{path.name} — {hint}"
    return path.name


def _read_text_upload(tmp_dir: Path, uploaded) -> str:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / uploaded.name
    dst.write_bytes(uploaded.getbuffer())
    return read_text_from_document(str(dst))


def _render_metrics_files(out_dir: str) -> None:
    out = Path(out_dir)
    json_files = sorted(out.glob("*_metrics.json"))
    png_files = sorted(out.glob("*_confusion_matrix.png"))

    if json_files:
        st.subheader("Метрики (JSON)")
        for p in json_files:
            with st.expander(p.name, expanded=False):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    st.text(p.read_text(encoding="utf-8", errors="replace"))
                else:
                    st.json(data)

    if png_files:
        st.subheader("Матрица ошибок (PNG)")
        for p in png_files:
            st.image(str(p), caption=p.name, use_container_width=True)


def _predict_block(model_path: str, text: str, top_k: int) -> None:
    details = predict_with_details(text, model_path, top_k=top_k)
    st.success(f"Предсказанный класс: {details['label']}")

    if details.get("probability_top"):
        label = details["label"]
        prob = None
        if details.get("probabilities"):
            prob = details["probabilities"].get(label)
        if prob is not None:
            st.write(f"Вероятность предсказанного класса: **{float(prob):.4f}**")

        st.subheader(f"Топ-{top_k} по вероятности")
        st.table(
            [{"class": c, "probability": float(p)} for c, p in details["probability_top"]]
        )
    elif details.get("score_top"):
        st.info("**Важно:** Модель svm не поддерживает вычисление вероятностей принадлежности к классам.\n Вместо этого отображаются значения decision_function — оценки, показывающие, насколько документ близок к каждому классу. Чем выше значение по сравнению с другими категориями, тем более вероятным считается отнесение документа к соответствующему классу. Положительные значения обычно указывают на большую уверенность модели, отрицательные — на меньшую, а значения, близкие к нулю, означают, что документ находится близко к границе между классами.")
        st.subheader(f"Топ-{top_k} по оценке")
        st.table(
            [{"class": c, "score": float(s)} for c, s in details["score_top"]]
        )
    else:
        st.warning("Для этой модели недоступны вероятности/оценки.")

    with st.expander("Полный  результат в формате JSON", expanded=False):
        st.json(details)


st.set_page_config(
    page_title="Классификация документов",
    page_icon="📄",
    layout="wide",
)

st.title("Система автоматической классификации документов")
st.caption("Локально: извлечение текста → предобработка → TF‑IDF → классификатор (sklearn)")

tab_auto, tab_predict, tab_train, tab_about = st.tabs(
    ["Автоматическая классификация", "Классификация отдельного документа", "Обучение", "О системе"]
)
if "batch_result" not in st.session_state:
    st.session_state.batch_result = None


with tab_auto:
    st.subheader("Автоматическая классификация документов из директории")

    col_a, col_b = st.columns(2)
    with col_a:
        models_dir = _models_dir()
        model_files = _list_model_files(models_dir)
        display_map = {_display_name_for_model_file(p): str(p) for p in model_files}
        manual_key = "Указать свой путь…"
        keys = list(display_map.keys()) + [manual_key]

        default_key = None
        for k, v in display_map.items():
            if Path(v).name == config.FILENAME_VECTORIZER_MODEL_LR:
                default_key = k
                break
        idx = keys.index(default_key) if default_key in keys else 0
        chosen = st.selectbox("Модель (.joblib)", keys, index=idx, key="auto_model")

        auto_model_path: Optional[str]
        if chosen == manual_key:
            auto_model_path = st.text_input(
                "Путь к обученной модели (.joblib)",
                value=str(Path(models_dir) / config.FILENAME_VECTORIZER_MODEL_LR),
                key="auto_model_manual",
            )
        else:
            auto_model_path = display_map.get(chosen)

        input_dir = st.text_input(
            "Входная директория с документами",
            value=str(PROJECT_ROOT / "data" / "tmp"),
            help="Файлы НЕ перемещаются. Обработка устойчива к ошибкам отдельных файлов.",
        )
        recursive = st.checkbox("Искать файлы в подпапках (рекурсивно)", value=True)

    with col_b:
        default_out = str(PROJECT_ROOT / "output" / "classified_documents")
        output_dir = st.text_input(
            "Директория результата",
            value=default_out,
            help="Документы будут КОПИРОВАТЬСЯ в output/<класс>/имя_файла. Выходная папка не очищается.",
        )
        threshold_col, _ = st.columns([1, 3])
        with threshold_col:
            threshold_percent = st.number_input(
                "Порог ручной проверки по вероятности (%)",
                min_value=1,
                max_value=60,
                value=20,
                step=1,
                help="Применяется только для моделей с вероятностями (Logistic Regression / Naive Bayes). Для SVM не применяется.",
            )

        st.markdown(
            "**Поддерживаемые форматы**: `.txt`, `.md`, `.docx`, `.pdf`, `.odt`, `.rtf`, `.html`.\n\n"
            "Ошибки (битый файл, пустой текст, проблемы с OCR , неподдерживаемый формат) не останавливают процесс классификации."
        )

    # Предпросмотр количества файлов
    try:
        files_preview = iter_document_paths(input_dir, recursive=recursive) if input_dir else []
    except Exception:
        files_preview = []
    st.caption(f"Найдено файлов для обработки: {len(files_preview)}")

    clicked = st.button("Запустить автоматическую классификацию", type="primary")
    stats_box = st.empty()
    if clicked:
        if not auto_model_path or not str(auto_model_path).strip():
            st.error("Укажите путь к модели (.joblib).")
        elif not input_dir or not Path(input_dir).is_dir():
            st.error("Входная директория не найдена.")
        else:
            total = len(files_preview)
            progress = st.progress(0)

            processed = 0
            ok_count = 0
            review_count = 0
            err_count = 0
            lines: list[str] = []
            all_results: list[BatchItemResult] = []
            review_files: list[tuple[str, str, Optional[float]]] = []
            class_counts: Counter[str] = Counter()

            for res in classify_directory(
                auto_model_path,
                input_dir,
                output_dir,
                recursive=recursive,
                top_k=1,
                manual_review_probability_threshold=float(threshold_percent) / 100.0,
            ):
                all_results.append(res)
                processed += 1
                name = Path(res.input_path).name
                if res.ok:
                    if res.probability is not None:
                        if res.manual_review_required == "yes":
                            review_count += 1
                            review_files.append((name, str(res.label or ""), float(res.probability)))
                            lines.append(
                                f"Файл: {name} → класс: {res.label} → вероятность: {res.probability * 100:.1f}% "
                                f"→ Требуется ручная проверка"
                            )
                        else:
                            ok_count += 1
                            lines.append(f"Файл: {name} → класс: {res.label} → вероятность: {res.probability * 100:.1f}%")
                    elif res.score is not None:
                        ok_count += 1
                        lines.append(f"Файл: {name} → класс: {res.label} → score: {res.score:.4f} (SVM, не вероятность)")
                    else:
                        ok_count += 1
                        lines.append(f"Файл: {name} → класс: {res.label}")
                    class_counts[str(res.label or "Неизвестно")] += 1
                else:
                    err_count += 1
                    class_counts["Ошибки"] += 1
                    lines.append(f"Файл: {name} → ошибка: {res.error}")

                # UI обновления
                if total > 0:
                    progress.progress(min(1.0, processed / total))
                stats_box.info(
                    f"Обработано: {processed}/{total if total else processed} | "
                    f"Успешно: {ok_count} | Требуют проверки: {review_count} | Ошибок: {err_count}"
                )

            report_path = write_batch_report_csv(
                all_results,
                str(Path(output_dir) / "batch_classification_report.csv"),
            )

            stats_box.success(
                f"Обработано: {processed}/{total if total else processed} | "
                f"Успешно: {ok_count} | Требуют проверки: {review_count} | Ошибок: {err_count}"
            )

            st.session_state.batch_result = {
                "processed": processed,
                "ok_count": ok_count,
                "review_count": review_count,
                "err_count": err_count,
                "threshold_percent": threshold_percent,
                "output_dir": os.path.abspath(output_dir),
                "report_path": report_path,
                "review_files": review_files,
                "lines": lines,
                "class_counts": dict(class_counts),
            }

    if st.session_state.batch_result:
        result = st.session_state.batch_result

        stats_box.success(
            f"Обработано: {result['processed']} | "
            f"Успешно: {result['ok_count']} | "
            f"Требуют проверки: {result['review_count']} | "
            f"Ошибок: {result['err_count']}"
        )

        class_counts = result.get("class_counts", {})
        if class_counts:
            fig = px.pie(
                names=list(class_counts.keys()),
                values=list(class_counts.values()),
                hole=0.4,
                labels={"names": "Класс", "values": "Количество"},
            )
            fig.update_traces(
                textposition="inside",
                textinfo="label",
                hovertemplate="%{label}: %{value} (%{percent:.1%})",
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.subheader("Распределение по классам")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander(
            "Список обработанных файлов (Открыть полностью): ",
            expanded=False
        ):
            if result["lines"]:
                st.text("\n".join(result["lines"]))
            else:
                st.info("Список пуст.")

        with open(result["report_path"], "rb") as f:
            st.download_button(
                "Скачать CSV-отчёт",
                f,
                file_name="batch_classification_report.csv",
                mime="text/csv",
                key="download_csv_report"
            )

        if result["review_files"]:
            st.warning("Файлы для ручной проверки:")

            for name, predicted_class, prob in result["review_files"]:
                st.markdown(
                    f"📄 **{name}**\n\n"
                    f"→ Предсказанный класс: **{predicted_class}**\n\n"
                    f"→ Вероятность: **{prob * 100:.1f}%**\n\n"
                    #f"→ Требуется ручная проверка"
                )


with tab_predict:
    st.subheader("Классификация отдельного документа")

    col1, col2 = st.columns(2)
    with col1:
        models_dir = _models_dir()
        model_files = _list_model_files(models_dir)
        display_map = {_display_name_for_model_file(p): str(p) for p in model_files}
        manual_key = "Указать свой путь…"
        keys = list(display_map.keys()) + [manual_key]

        default_key = None
        for k, v in display_map.items():
            if Path(v).name == config.FILENAME_VECTORIZER_MODEL_LR:
                default_key = k
                break
        idx = keys.index(default_key) if default_key in keys else 0

        chosen = st.selectbox("Выберите модель (.joblib) или укажите путь", keys, index=idx)

        model_path: Optional[str]
        if chosen == manual_key:
            model_path = st.text_input(
                "Путь к обученной модели (.joblib)",
                value=str(Path(models_dir) / config.FILENAME_VECTORIZER_MODEL_LR),
            )
        else:
            model_path = display_map.get(chosen)
            if model_path is None:
                st.warning("Файл модели не выбран. Укажите путь к .joblib.")
                model_path = None

        src = st.radio("Источник текста", ["Файл документа", "Вставить текст"], horizontal=True)
        top_k = st.slider("Топ-K классов", min_value=1, max_value=10, value=5)
    with col2:
        st.markdown(
            "**Поддерживаемые форматы**: `.txt`, `.md`, `.docx`, `.pdf`, `.odt`, `.rtf`, `.html`.\n\n"
            "Для PDF со сканами используется OCR (если настроен и установлен Tesseract OCR)."
        )

    if src == "Файл документа":
        uploaded = st.file_uploader(
            "Загрузите документ",
            type=["txt", "md", "docx", "pdf", "odt", "rtf", "html", "htm"],
        )
        if uploaded and st.button("Классифицировать", type="primary"):
            try:
                text = _read_text_upload(PROJECT_ROOT / "data" / "tmp", uploaded)
            except Exception as e:
                st.error(f"Не удалось извлечь текст: {e}")
            else:
                if not text.strip():
                    st.warning("Текст пустой: файл не содержит текста или OCR не смог распознать.")
                else:
                    if model_path:
                        _predict_block(model_path, text, top_k=top_k)
    else:
        text = st.text_area("Текст документа", height=220)
        if st.button("Классифицировать", type="primary"):
            if not text.strip():
                st.warning("Вставьте непустой текст.")
            else:
                if model_path:
                    _predict_block(model_path, text, top_k=top_k)


with tab_train:
    st.subheader("Обучение модели")

    out_dir = st.text_input("Каталог для артефактов (models/…)", value=_models_dir())
    _ensure_dir(out_dir)

    model_choice = st.selectbox(
        "Модель",
        [MODEL_LABELS_RU["logreg"], MODEL_LABELS_RU["nb"], MODEL_LABELS_RU["svm"]],
        index=0,
    )
    model_kind = "logreg"
    for code, label in MODEL_LABELS_RU.items():
        if label == model_choice:
            model_kind = code
            break
    source = st.selectbox("Источник обучающих данных", ["Локальная директория (подпапка = класс)", "CSV (text,label)", "Hugging Face"], index=0)

    train_params: Dict[str, Any] = {}

    if source == "Локальная директория (подпапка = класс)":
        data_dir = st.text_input("Путь к директории с корпусными данными", value=str(PROJECT_ROOT / "data" / "corpus_txt"))
        train_params = {"kind": "dir", "data_dir": data_dir}
    elif source == "CSV (text,label)":
        csv_path = st.text_input("Путь к CSV", value=str(PROJECT_ROOT / "data" / "sample_train.csv"))
        text_col = st.text_input("Столбец текста", value=config.CSV_TEXT_COLUMN)
        label_col = st.text_input("Столбец класса", value=config.CSV_LABEL_COLUMN)
        train_params = {"kind": "csv", "csv_path": csv_path, "text_col": text_col, "label_col": label_col}
    else:
        dataset_id = st.text_input("Hugging Face dataset", value=config.HF_DEFAULT_DATASET)
        split = st.text_input("split", value=config.HF_DEFAULT_SPLIT)
        hf_text_col = st.text_input("Столбец текста", value=config.HF_DEFAULT_TEXT_COLUMN)
        hf_label_col = st.text_input("Столбец класса", value=config.HF_DEFAULT_LABEL_COLUMN)
        train_params = {
            "kind": "hf",
            "dataset_id": dataset_id,
            "split": split,
            "text_col": hf_text_col,
            "label_col": hf_label_col,
        }

    if st.button("Запустить обучение", type="primary"):
        with st.spinner("Обучение..."):
            try:
                if train_params["kind"] == "dir":
                    pipeline, _X_train, _y_train, X_test, y_test, preprocessor = train_from_document_folders(
                        data_root=train_params["data_dir"],
                        model_kind=model_kind,
                    )
                elif train_params["kind"] == "csv":
                    pipeline, _X_train, _y_train, X_test, y_test, preprocessor = train_from_csv(
                        csv_path=train_params["csv_path"],
                        model_kind=model_kind,
                        text_column=train_params["text_col"],
                        label_column=train_params["label_col"],
                    )
                else:
                    pipeline, _X_train, _y_train, X_test, y_test, preprocessor = train_from_huggingface(
                        dataset_id=train_params["dataset_id"],
                        model_kind=model_kind,
                        split=train_params["split"],
                        text_column=train_params["text_col"],
                        label_column=train_params["label_col"],
                    )

                # Оценка + артефакты
                title_map = {"logreg": "Logistic Regression", "nb": "Naive Bayes", "svm": "Linear SVM"}
                payload = evaluate_and_report(
                    pipeline,
                    X_test,
                    y_test,
                    model_name=title_map[model_kind],
                    output_dir=out_dir,
                    labels_order=pipeline.classes_,
                )

                # Сохранение модели
                name_map = {
                    "logreg": config.FILENAME_VECTORIZER_MODEL_LR,
                    "nb": config.FILENAME_VECTORIZER_MODEL_NB,
                    "svm": config.FILENAME_VECTORIZER_MODEL_SVM,
                }
                model_file = str(Path(out_dir) / name_map[model_kind])
                save_model_bundle(pipeline, preprocessor, model_file)

            except Exception as e:
                st.error(f"Ошибка обучения: {e}")
            else:
                st.success(f"Готово. Модель сохранена: {model_file}")
                with st.expander("Метрики (JSON payload)", expanded=False):
                    st.json(payload)
                _render_metrics_files(out_dir)


with tab_about:
    st.subheader("Как пользоваться")
    st.markdown(
        "- **Обучение**: Выберите модель из доступных (Логистическая регрессия, Наивный байес, Линейный SVM). Далее выберите источник данных (папка с подпапками-классами, CSV, Hugging Face). Затем подготовьте корпус обучающих данных (подпапки = классы) или возьмите готовый размеченный датасет из Hugging Face, обучите модель.\n"
        "- Все артефакты (модель, метрики, confusion matrix) сохраняются в `models/` (или выбранную директорию). \n"
        "- **Классификация**: выберите `.joblib` и загрузите документ — система выдаст класс и вероятности (если модель поддерживает).\n"
        "- **Автоматическая классификация**: пакетная обработка директории. Выберите обученную модель, укажите входную папку с документами, выходную папку для результатов. Документы будут скопированы в подпапки по предсказанным классам. Используется порог вероятности для выявления файлов, требующих ручной проверки (применяется только для моделей, поддерживающих вероятности). Поддерживаются форматы: `.txt`, `.md`, `.docx`, `.pdf`, `.odt`, `.rtf`, `.html`.  По итогу классификации формируется CSV-отчёт.\n"

    )
