"""
Предобработка текста: нормализация, токенизация, удаление стоп-слов, лемматизация.

Используется pymorphy2 для морфологического анализа русского языка.
Стоп-слова загружаются из NLTK (русский набор); при недоступности — встроенный минимальный список.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Union

from pymorphy2 import MorphAnalyzer


# Минимальный набор русских стоп-слов, если NLTK недоступен
_FALLBACK_RU_STOPWORDS: Set[str] = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
    "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь",
    "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто",
    "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом",
    "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них",
    "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им",
    "более", "всегда", "конечно", "всю", "между",
}


def _load_nltk_russian_stopwords() -> Set[str]:
    """Загружает стоп-слова русского языка из NLTK."""
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("russian"))
    except (LookupError, ImportError):
        return set()


class TextPreprocessor:
    """
    Пайплайн предобработки: нижний регистр → токенизация → лемматизация → фильтрация стоп-слов.

    Параметры
    ----------
    use_stopwords : bool
        Удалять ли стоп-слова после лемматизации.
    min_token_length : int
        Минимальная длина токена (после лемматизации); более короткие отбрасываются.
    """

    _TOKEN_PATTERN = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

    def __init__(
        self,
        use_stopwords: bool = True,
        min_token_length: int = 2,
        extra_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.use_stopwords = use_stopwords
        self.min_token_length = min_token_length
        self._morph = MorphAnalyzer()

        nltk_sw = _load_nltk_russian_stopwords()
        self._stopwords: Set[str] = nltk_sw if nltk_sw else _FALLBACK_RU_STOPWORDS.copy()
        if extra_stopwords:
            self._stopwords.update(w.lower() for w in extra_stopwords)

    def tokenize(self, text: str) -> List[str]:
        """
        Разбивает текст на токены (буквы кириллицы, латиницы и цифры).
        Регистр не меняет; для единообразия обычно вызывают после lower().
        """
        if not text or not text.strip():
            return []
        return self._TOKEN_PATTERN.findall(text.lower())

    def lemmatize_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Возвращает список лемм для списка токенов (токены в нижнем регистре)."""
        lemmas: List[str] = []
        for token in tokens:
            t = token.lower().strip()
            if not t:
                continue
            parsed = self._morph.parse(t)[0]
            lemmas.append(parsed.normal_form)
        return lemmas

    def remove_stopwords(self, lemmas: Iterable[str]) -> List[str]:
        """Удаляет стоп-слова из последовательности лемм."""
        if not self.use_stopwords:
            return list(lemmas)
        return [w for w in lemmas if w not in self._stopwords]

    def filter_short_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Отбрасывает слишком короткие токены."""
        return [w for w in tokens if len(w) >= self.min_token_length]

    def preprocess(self, text: str, as_string: bool = True) -> Union[str, List[str]]:
        """
        Полный цикл предобработки одного текста.

        Порядок: нижний регистр → токенизация → лемматизация → стоп-слова → фильтр по длине.

        Parameters
        ----------
        text : str
            Исходный текст документа.
        as_string : bool
            Если True — возвращает строку из лемм через пробел (удобно для TF-IDF).
            Если False — список лемм.
        """
        text_lower = text.lower()
        raw_tokens = self._TOKEN_PATTERN.findall(text_lower)
        lemmas = self.lemmatize_tokens(raw_tokens)
        lemmas = self.remove_stopwords(lemmas)
        lemmas = self.filter_short_tokens(lemmas)
        if as_string:
            return " ".join(lemmas)
        return lemmas

    def preprocess_batch(
        self,
        texts: Iterable[str],
        as_string: bool = True,
    ) -> List[Union[str, List[str]]]:
        """Предобработка коллекции текстов."""
        return [self.preprocess(t, as_string=as_string) for t in texts]


def ensure_nltk_stopwords_downloaded() -> None:
    """
    Скачивает ресурс NLTK `stopwords`, если его ещё нет.
    Вызовите один раз при первом запуске на новой машине.
    """
    try:
        import nltk

        try:
            from nltk.corpus import stopwords

            stopwords.words("russian")
        except LookupError:
            nltk.download("stopwords", quiet=True)
    except ImportError:
        pass
