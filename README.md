# Классификация текстовых документов

Локальная система на **Python**: предобработка русскоязычного текста (лемматизация, стоп-слова), **TF-IDF**, классификаторы **Logistic Regression**, **Multinomial Naive Bayes**, **Linear SVM**. Обучение по CSV, по папкам с файлами или по датасету **Hugging Face**. Предсказание по тексту или по файлу (в т.ч. Word, PDF с опциональным OCR).

---

## Установка

Из корня проекта:

```bash
pip install -r requirements.txt
```

При первом запуске может потребоваться загрузка стоп-слов NLTK (выполняется автоматически в `main.py`).

### Дополнительно (по желанию)

| Задача | Что установить |
|--------|----------------|
| Сканы в PDF (OCR) | [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (языки **rus**, **eng**). При необходимости задайте `TESSERACT_CMD` — полный путь к `tesseract.exe`. |
| Датасеты Hugging Face | Уже в `requirements.txt`: пакет `datasets` |

---

## Структура проекта

| Каталог / файл | Назначение |
|----------------|------------|
| `main.py` | Точка входа, CLI |
| `settings/` | `default.yaml` и загрузчик YAML/JSON |
| `data/` | Загрузка корпуса, извлечение текста из файлов (`document_text.py`, `data_loader.py`, `hf_loader.py`) |
| `preprocessing/` | Токенизация, стоп-слова, **pymorphy2** |
| `training/` | Пайплайны, обучение, `config.py`, сохранение **joblib** |
| `evaluation/` | Метрики, confusion matrix, отчёт, JSON, PNG |
| `prediction/` | Предсказание класса и вероятностей |
| `models/` | Сохранённые `.joblib`, метрики, графики (по умолчанию) |
| `data/corpus_txt/` | Пример корпуса: подпапки = классы |

---

## Конфигурация (без правки кода)

1. **Файл по умолчанию:** `settings/default.yaml` (скопируйте и меняйте).
2. **Переменная окружения:** `APP_CONFIG=путь/к/файлу.yaml` (или `.json`).
3. **Аргумент CLI:** `--config путь` — указывайте **до имени подкоманды**:

```bash
python main.py --config settings/my.yaml train --data-dir data/corpus_txt
```

В YAML задаются, в частности: параметры **TF-IDF**, обучения, путей, секция **`ocr`** (пороги PDF, масштаб OCR, `TESSERACT_LANG`), секция **`logging`** (`level`, `file` — путь к логу относительно корня проекта; `null` для файла — только консоль).

Подробнее см. комментарии в `settings/default.yaml` и `settings/loader.py`.

---

## Команды CLI

Общая форма:

```bash
python main.py [ГЛОБАЛЬНЫЕ ОПЦИИ] КОМАНДА [АРГУМЕНТЫ КОМАНДЫ]
```

### Глобальные опции

| Опция | Описание |
|-------|----------|
| `-h`, `--help` | Справка |
| `--config PATH` | YAML или JSON с настройками (см. выше) |

### Команда `run` — быстрый старт

Обучает одну модель на папке из конфига (`QUICK_START_DATA_DIR`, по умолчанию `data/corpus_txt`).

```bash
python main.py
python main.py run
python main.py run --data-dir путь/к/корпусу
python main.py run --model logreg
python main.py run --model nb
python main.py run --model svm
python main.py run --out models
```

| Аргумент | Описание |
|----------|----------|
| `--data-dir` | Корень с подпапками-классами (если не задан — из конфига) |
| `--model` | `logreg`, `nb` или `svm` |
| `--out` | Каталог для модели и отчётов (по умолчанию из конфига) |

Если запустить `python main.py` без аргументов, подставляется команда `run`.

---

### Команда `train` — обучение одной модели

Источник данных — **ровно один** из трёх вариантов: `--csv`, `--data-dir`, `--hf`.

```bash
# Папки: корень/ИмяКласса/файлы (.txt, .docx, .pdf, .odt, .rtf, .html, .md)
python main.py train --data-dir data/corpus_txt --model logreg
python main.py train --data-dir data/corpus_txt --model nb
python main.py train --data-dir data/corpus_txt --model svm

# CSV: столбцы text и label (имена можно сменить)
python main.py train --csv data/sample_train.csv --model logreg
python main.py train --csv data/sample_train.csv --text-column text --label-column label

# Hugging Face (нужен пакет datasets, первый раз — загрузка)
python main.py train --hf Adilbai/kz-gov-complaints-data-kz-ru --model logreg
python main.py train --hf ИМЯ/ДАТАСЕТА --hf-split train --hf-text-column text_ru --hf-label-column category
```

| Аргумент | Описание |
|----------|----------|
| `--csv` | Путь к CSV |
| `--data-dir` | Каталог с подпапками-классами |
| `--hf` | Идентификатор датасета на Hugging Face |
| `--model` | `logreg`, `nb`, `svm` |
| `--out` | Каталог для `.joblib`, JSON метрик, PNG матрицы ошибок |
| `--text-column`, `--label-column` | Для CSV |
| `--hf-split`, `--hf-text-column`, `--hf-label-column` | Для HF |

После обучения сохраняются, например:

- `pipeline_logreg.joblib` / `pipeline_multinomial_nb.joblib` / `pipeline_svm.joblib`
- `*_metrics.json`, `*_confusion_matrix.png`

---

### Команда `compare` — две модели и сравнение

Обучает **Logistic Regression** и **Naive Bayes** на одних данных, выводит сравнительную таблицу, сохраняет обе модели и отчёты.

```bash
python main.py compare --csv data/sample_train.csv
python main.py compare --data-dir data/corpus_txt
python main.py compare --hf Adilbai/kz-gov-complaints-data-kz-ru
```

Те же вспомогательные аргументы, что у `train` (кроме `--model`): `--out`, `--text-column`, `--label-column`, `--hf-*`.

---

### Команда `predict` — классификация документа

Нужен уже обученный файл **`.joblib`**. Текст задаётся через **`--text`** или **`--file`** (взаимоисключающие).

```bash
python main.py predict --model models/pipeline_logreg.joblib --text "Произвольный текст документа"
python main.py predict --model models/pipeline_logreg.joblib --file документ.txt
python main.py predict --model models/pipeline_logreg.joblib --file документ.docx
python main.py predict --model models/pipeline_logreg.joblib --file документ.pdf
```

Поддерживаемые расширения файлов: **`.txt`**, **`.md`**, **`.docx`**, **`.pdf`**, **`.odt`**, **`.rtf`**, **`.html`**, **`.htm`**.

#### Вероятности и JSON

| Аргумент | Описание |
|----------|----------|
| `--probs` | Вывести топ классов с вероятностями (**LR**, **NB**) или оценками **decision_function** (**SVM**; это не вероятности) |
| `--top-k K` | Сколько лучших классов показать (по умолчанию 5) |
| `--json` | Один JSON-объект с полями `label`, при наличии — `probabilities` / `probability_top` или `decision_scores` / `score_top` |

Примеры:

```bash
python main.py predict --model models/pipeline_logreg.joblib --file doc.txt --probs --top-k 5
python main.py predict --model models/pipeline_logreg.joblib --text "..." --json
```

---

## Формат данных для обучения из папок

```
корень/
  класс_1/
    документ1.txt
    документ2.pdf
  класс_2/
    ...
```

Имя подпапки первого уровня = метка класса. Файлы — поддерживаемые расширения (см. выше).

---

## Логирование

Настраивается в **`settings/default.yaml`** (секция `logging`): уровень (`DEBUG`, `INFO`, …) и опционально файл (например `logs/classification.log`). В лог пишутся чтение документов (путь, формат, длина текста), для PDF — сведения об OCR, при классификации — краткая информация о модели.

---

## Справка по всем флагам

```bash
python main.py -h
python main.py run -h
python main.py train -h
python main.py compare -h
python main.py predict -h
```

---

## Зависимости (кратко)

См. полный список в **`requirements.txt`**: **numpy**, **pandas**, **scikit-learn**, **joblib**, **matplotlib**, **pymorphy2**, **nltk**, **python-docx**, **pymupdf**, **striprtf**, **pytesseract**, **Pillow**, **PyYAML**, опционально **datasets**.

---

## Ограничения

- Старый формат **`.doc`** (Word 97–2003) не поддерживается — сохраните как **`.docx`**.
- Качество OCR сканов зависит от Tesseract, разрешения и качества скана.
- Для стратифицированного `train_test_split` нужно достаточно примеров на класс; при очень малом корпусе возможны ошибки разбиения.
