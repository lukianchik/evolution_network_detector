"""Метрики бинарного детектора сетевых аномалий без внешних библиотек."""


def measure(labels: list[int], predictions: list[int], fpr_penalty: float = 0.2) -> dict:
    """Возвращает F1, Precision, Recall, FPR и состав матрицы ошибок.

    Fitness сочетает F1 с небольшим штрафом за ложные срабатывания.
    """
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Размеры меток и ответов не совпадают")
    if any(type(item) is not int or item not in (0, 1) for item in predictions):
        raise ValueError("Детектор должен вернуть список целых 0 и 1")
    if any(type(item) is not int or item not in (0, 1) for item in labels):
        raise ValueError("Метки должны быть целыми 0 или 1")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = len(labels) - tp - fp - fn
    return from_counts(tp, fp, fn, tn, fpr_penalty)


def from_counts(tp: int, fp: int, fn: int, tn: int, fpr_penalty: float = 0.2) -> dict:
    """Рассчитывает объединённые метрики по матрице ошибок."""
    if not 0 <= fpr_penalty <= 2 or min(tp, fp, fn, tn) < 0 or not tp + fp + fn + tn:
        raise ValueError("Некорректная матрица ошибок или штраф FPR")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "fitness": round(f1 - fpr_penalty * fpr, 6),
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "fpr": round(fpr, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "count": tp + fp + fn + tn,
    }


def summarize(measurements: list[dict]) -> dict:
    """Усредняет вещественные показатели по независимым выборкам."""
    if not measurements:
        raise ValueError("Нет измерений")
    keys = ("fitness", "f1", "precision", "recall", "fpr")
    output = {key: round(sum(item[key] for item in measurements) / len(measurements), 6) for key in keys}
    for key in ("tp", "fp", "fn", "tn", "count"):
        output[key] = sum(item[key] for item in measurements)
    return output
