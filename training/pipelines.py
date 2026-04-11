"""
Сборка пайплайнов: TF-IDF + классификатор.

Пайплайн sklearn объединяет векторизацию и классификатор в один объект:
удобно сохранять в joblib и вызывать predict одной строкой.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Создаёт векторизатор TF-IDF с параметрами из config."""
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        norm=config.TFIDF_NORM,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )


def build_pipeline_logistic_regression() -> Pipeline:
    """Пайплайн: TF-IDF + логистическая регрессия."""
    tfidf = build_tfidf_vectorizer()
    clf = LogisticRegression(
        C=config.LR_C,
        max_iter=config.LR_MAX_ITER,
        solver=config.LR_SOLVER,
        class_weight=config.LR_CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )


def build_pipeline_naive_bayes() -> Pipeline:
    """Пайплайн: TF-IDF + наивный байесовский классификатор (мультиномиальный)."""
    tfidf = build_tfidf_vectorizer()
    clf = MultinomialNB(alpha=config.NB_ALPHA)
    return Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )


def build_pipeline_linear_svc() -> Pipeline:
    """
    Пайплайн: TF-IDF + линейный SVM (LinearSVC).

    dual=False типично для текста, когда число объектов больше размерности признакового пространства.
    """
    tfidf = build_tfidf_vectorizer()
    clf = LinearSVC(
        C=config.SVM_C,
        max_iter=config.SVM_MAX_ITER,
        random_state=config.RANDOM_STATE,
        dual=False,
    )
    return Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )
