from __future__ import annotations

from pathlib import Path

Hotwords = dict[str, dict[str, float]]


def parse_hotwords_text(path: Path) -> Hotwords:
    result: Hotwords = {"default": {}}
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            # Weight is not used by runtime inference; keep backward compatibility
            # by accepting old "word weight" lines and ignoring the weight token.
            result["default"][parts[0]] = 1.0
    return result


def merge_hotwords(base: Hotwords, incoming: Hotwords) -> Hotwords:
    merged: Hotwords = {k: dict(v) for k, v in base.items()}
    for category, words in incoming.items():
        bucket = merged.setdefault(category, {})
        bucket.update(words)
    return merged


def count_hotwords(hotwords: Hotwords) -> int:
    return sum(len(words) for words in hotwords.values())
