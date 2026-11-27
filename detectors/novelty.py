"""Лёгкие локальные векторные представления кода для отсева похожих программ.

Это хэш-эмбеддинги по лексическим токенам, а не семантическая нейросеть.
В отличие от большого энкодера не требуют видеокарты и скачивания весов.
"""

import hashlib
import io
import math
import tokenize

DIMENSIONS = 192


def embed(text: str) -> list[float]:
    """Строит нормализованный вектор из имён, операторов и констант программы."""
    vector = [0.0] * DIMENSIONS
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type not in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
                continue
            name = f"{token.type}:{token.string}"
            digest = hashlib.blake2s(name.encode("utf-8"), digest_size=8).digest()
            pos = int.from_bytes(digest[:4], "big") % DIMENSIONS
            vector[pos] += 1.0
    except tokenize.TokenError:
        pass
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 9) for value in vector]


def similarity(left: list[float], right: list[float]) -> float:
    """Возвращает косинусное сходство двух нормализованных векторов."""
    return sum(a * b for a, b in zip(left, right))


def is_novel(source: str, previous: list[list[float]], threshold: float = 0.998) -> bool:
    """Отбраковывает слишком похожие на уже принятые варианты."""
    candidate = embed(source)
    return not any(similarity(candidate, vector) >= threshold for vector in previous)
