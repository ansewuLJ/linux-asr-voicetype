from __future__ import annotations

from dataclasses import dataclass

from .hotwords import Hotwords

LANGUAGE_MAP = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "yue": "Cantonese",
    "en": "English",
}


@dataclass(slots=True)
class AsrResult:
    text: str
    language: str
    success: bool
    error: str | None = None


def normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    lowered = language.strip().lower()
    return LANGUAGE_MAP.get(lowered, language)


def hotwords_to_context(hotwords: Hotwords) -> str:
    flat: list[str] = []
    for words in hotwords.values():
        flat.extend(words.keys())
    return ", ".join(flat)
