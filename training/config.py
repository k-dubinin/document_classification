"""
Параметры обучения и векторизации.
Все настройки собраны в одном месте — удобно менять для экспериментов и описать в дипломе.
"""

# --- Общие настройки ---
# Доля тестовой выборки при разбиении train/test
TEST_SIZE = 0.2

# Seed для воспроизводимости разбиения и моделей, где применимо
RANDOM_STATE = 42

# Имена столбцов в CSV по умолчанию
CSV_TEXT_COLUMN = "text"
CSV_LABEL_COLUMN = "label"

# --- Быстрый старт (команда: python main.py run или просто python main.py) ---
# Папка с документами: подпапки = классы (например: обращения по темам, типы внутренних документов)
QUICK_START_DATA_DIR = "data/corpus_txt"
QUICK_START_MODEL = "logreg"

# --- Датасет с Hugging Face (похожие данные: обращения граждан и т.п.) ---
# Пример: Adilbai/kz-gov-complaints-data-kz-ru — колонки text_ru (текст) и category (класс)
HF_DEFAULT_DATASET = "Adilbai/kz-gov-complaints-data-kz-ru"
HF_DEFAULT_SPLIT = "train"
HF_DEFAULT_TEXT_COLUMN = "text_ru"
HF_DEFAULT_LABEL_COLUMN = "category"

# --- TF-IDF (TfidfVectorizer) ---
# Ограничение числа признаков; None — без ограничения
TFIDF_MAX_FEATURES = 10_000

# Нижняя и верхняя граница документной частоты слова
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# Н-граммы: (1, 2) — униграммы и биграммы
TFIDF_NGRAM_RANGE = (1, 2)

# Нормализация L2 для вектора документа
TFIDF_NORM = "l2"

# Сублинейное масштабирование tf: 1 + log(tf)
TFIDF_SUBLINEAR_TF = True

# --- Logistic Regression ---
LR_C = 1.0
LR_MAX_ITER = 2000
LR_SOLVER = "lbfgs"
LR_CLASS_WEIGHT = None  # при дисбалансе классов можно поставить "balanced"

# --- Naive Bayes  ---
NB_ALPHA = 1.0

# --- Линейный SVM  ---
SVM_C = 1.0
SVM_MAX_ITER = 2000

# --- Пути для сохранения артефактов (относительно корня проекта, можно переопределить ) ---
DEFAULT_MODELS_DIR = "models"
DEFAULT_OUTPUT_DIR = "models"  # метрики и рисунки рядом с моделями

# Имена файлов при сохранении
FILENAME_VECTORIZER_MODEL_LR = "pipeline_logreg.joblib"
FILENAME_VECTORIZER_MODEL_NB = "pipeline_multinomial_nb.joblib"
FILENAME_VECTORIZER_MODEL_SVM = "pipeline_svm.joblib"
FILENAME_METRICS_SUFFIX = "_metrics.json"
FILENAME_CONFUSION_MATRIX_SUFFIX = "_confusion_matrix.png"
