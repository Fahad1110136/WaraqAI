from langdetect import detect, LangDetectException
import pyarabic.araby as araby
import re

import config

# Arabic punctuation isn't covered by Python's standard string.punctuation,
# so it's listed explicitly here (Arabic comma، semicolon؛ question mark؟
# plus the common Latin punctuation marks that show up in mixed text).
_PUNCTUATION_PATTERN = re.compile(r"[.,!?;:،؛؟\"'“”‘’()\[\]{}\-–—/\\]")

def detect_language(text: str) -> str:
    text = text.strip()
    if not text:
        return "English"

    try:
        return detect(text)
    except LangDetectException:
        return "Arabic"

def get_search_mode(text: str) -> str:
    lang = detect_language(text)
    if lang in config.DENSE_ONLY_LANGUAGES:
        return "Dense", lang
    if lang in config.HYBRID_LANGUAGES:
        return "Hybrid", lang

    return config.DEFAULT_SEARCH_MODE, lang

def normalize_arabic(text: str) -> str:
    # Normalizes Arabic text for keyword (BM25) matching:
    #  - strips diacritics (tashkeel) — e.g. الْعَرَبِيَّة -> العربية
    #  - strips tatweel (elongation character, e.g. ـــ)
    #  - unifies alef forms (أ, إ, آ -> ا) so spelling variants of the same word match as the same token
    if not text:
        return text

    text = araby.strip_diacritics(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    return text

def tokenize_for_bm25(text: str) -> list:
    _, lang = get_search_mode(text)  # reuse detection; ignore the mode value
    if lang == "Arabic":
        text = normalize_arabic(text)

    text = _PUNCTUATION_PATTERN.sub(" ", text.lower())
    return text.split()