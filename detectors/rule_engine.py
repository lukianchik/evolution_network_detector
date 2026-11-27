"""Безопасные самоэволюционирующие правила и их локальная оценка.

Метка класса никогда не передаётся правилу; все вычисления — NumPy на CPU.
"""

from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from collections import OrderedDict

import numpy as np

from traffic_data.dataset import FEATURES as SYNTH_FEATURES, build_dataset
from traffic_data.dataset_real import EXPANDED_FEATURES, FEATURES as UNSW_FEATURES, load_unsw
from traffic_data.sampling import training_parts
from detectors.metrics import from_counts

MAX_RULES = 6


def features_for(dataset: str) -> tuple[str, ...]:
    """Возвращает перечень числовых признаков, разрешённых для задачи."""
    if dataset == "unsw":
        return EXPANDED_FEATURES
    if dataset == "synthetic":
        return SYNTH_FEATURES
    raise ValueError("Датасет должен быть unsw или synthetic")


def normalize(rule: dict, dataset: str) -> dict:
    """Проверяет структуру правил и возвращает каноническое представление.

    Это не Python-код: произвольные выражения, импорты и названия полей запрещены.
    """
    if not isinstance(rule, dict) or not isinstance(rule.get("rules"), list):
        raise ValueError("Нужен JSON-объект с массивом rules")
    items = rule["rules"]
    if not 1 <= len(items) <= MAX_RULES:
        raise ValueError(f"Количество правил: 1–{MAX_RULES}")
    allowed = set(features_for(dataset))
    remaining = 18

    def number(value):
        if type(value) not in (int, float) or abs(value) > 1e12 or not math.isfinite(value):
            raise ValueError("Некорректный порог")
        return round(float(value), 6)

    def node(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 3 or not isinstance(item, dict):
            raise ValueError("Недопустимая структура или сложность правила")
        op = item.get("operator")
        if op in ("and", "or", "not"):
            children = item.get("conditions")
            if not isinstance(children, list) or not (
                len(children) == 1 if op == "not" else 2 <= len(children) <= MAX_RULES
            ):
                raise ValueError("Недопустимое число условий в группе")
            return {"operator": op, "conditions": sorted((node(child, depth + 1) for child in children), key=identity)}
        feature = item.get("feature")
        if not isinstance(feature, str) or feature not in allowed or op not in (">=", "<=", "between"):
            raise ValueError("Неизвестный признак или оператор")
        result = {"feature": feature, "operator": op, "threshold": number(item.get("threshold"))}
        if op == "between":
            result["upper"] = number(item.get("upper"))
            if result["upper"] < result["threshold"]:
                raise ValueError("Границы интервала переставлены")
        return result

    normalized = []
    used = set()
    for item in items:
        condition = node(item, 1)
        key = condition_key(condition)
        if key in used:
            raise ValueError("Одинаковые условия в одном наборе не допускаются")
        used.add(key)
        weight = item.get("weight")
        if type(weight) is not int or weight not in (1, 2, 3):
            raise ValueError("Вес должен быть целым 1–3")
        normalized.append({**condition, "weight": weight})
    decision = rule.get("decision_threshold")
    if type(decision) is not int or not 1 <= decision <= sum(item["weight"] for item in normalized):
        raise ValueError("Некорректный порог итогового решения")
    normalized.sort(key=condition_key)
    return {"rules": normalized, "decision_threshold": decision}


def identity(rule: dict) -> str:
    """Создаёт стабильный ключ для фильтра повторных решений."""
    return json.dumps(rule, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def condition_key(item: dict) -> tuple[str, str]:
    """Возвращает ключ условия без веса."""
    name = item.get("feature")
    return (name, item["operator"]) if name else ("", identity(item["conditions"]) + item["operator"])


def complexity(rule: dict) -> int:
    """Считает узлы и второе сравнение каждого интервала."""
    def size(item):
        return 1 + (item["operator"] == "between") + sum(map(size, item.get("conditions", ())))
    return sum(map(size, rule["rules"]))


def measure_predictions(predicted: np.ndarray, labels: np.ndarray, penalty: float) -> dict:
    """Считает метрики для векторных бинарных предсказаний."""
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted)) - tp
    positive = int(np.count_nonzero(labels))
    return from_counts(tp, fp, positive - tp, len(labels) - positive - fp, penalty)


def expression(item: dict) -> str:
    """Создаёт эквивалентное выражение из проверенного дерева."""
    op = item["operator"]
    if op == "not":
        return f'(not {expression(item["conditions"][0])})'
    if op in ("and", "or"):
        return "(" + f" {op} ".join(map(expression, item["conditions"])) + ")"
    field = f'event["{item["feature"]}"]'
    if op == "between":
        return f'{item["threshold"]!r} <= {field} <= {item["upper"]!r}'
    return f'{field} {op} {item["threshold"]!r}'


def seed_rule(dataset: str) -> dict:
    """Создаёт слабый исходный детектор, сравнимый с исходным файлом."""
    if dataset == "unsw":
        item = {"feature": "ct_dst_src_ltm", "operator": ">=", "threshold": 20, "weight": 1}
    else:
        item = {"feature": "failed_logins", "operator": ">=", "threshold": 9, "weight": 1}
    return normalize({"rules": [item], "decision_threshold": 1}, dataset)


def to_python(rule: dict, dataset: str) -> str:
    """Генерирует читаемый Python-детектор из проверенных правил."""
    rule = normalize(rule, dataset)
    lines = ['"""Детектор, полученный локальной эволюцией правил."""', '',
             '# EVOLVE-BLOCK-START', 'def detect(event: dict) -> int:',
             '    """Суммирует веса сработавших условий сетевого потока."""',
             '    score = 0']
    for item in rule["rules"]:
        lines.append(f'    if {expression(item)}:')
        lines.append(f'        score += {item["weight"]}')
    lines.extend([f'    return int(score >= {rule["decision_threshold"]})',
                  '# EVOLVE-BLOCK-END', '', 'def run_experiment(events: list[dict]) -> list[int]:',
                  '    """Оценивает события без доступа к правильным ответам."""',
                  '    return [detect(event) for event in events]', ''])
    return "\n".join(lines)


@lru_cache(maxsize=2)
def load_parts(dataset: str) -> tuple[tuple, tuple]:
    """Готовит обучающую и проверочную части без чтения официального test CSV.

    Отдельная проверочная часть взята из train CSV; итоговый test остаётся закрытым.
    Размеры ограничены для 16 ГБ RAM и быстрого CPU-поиска.
    """
    names = UNSW_FEATURES if dataset == "unsw" else features_for(dataset)
    if dataset == "unsw":
        events, labels = load_unsw("train", 3600)
        pairs = list(zip(events, labels))
        if len(pairs) < 2:
            raise ValueError("Для train/validation необходимо не меньше двух записей")
        boundary = min(2400, max(1, 2 * len(pairs) // 3))
        train, validation = pairs[:boundary], pairs[boundary:]
    else:
        train = list(zip(*build_dataset(211, 800)))  # Перепаковывается ниже без мутации событий.
        validation = list(zip(*build_dataset(377, 400)))
    def pack(pairs: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        """Собирает матрицу числовых признаков и отдельные правильные метки."""
        x = np.asarray([[event[name] for name in names] for event, _ in pairs], dtype=np.float64)
        y = np.asarray([label for _, label in pairs], dtype=np.uint8)
        x.setflags(write=False)
        y.setflags(write=False)
        return x, y
    return pack(train), pack(validation)


class RuleEvaluator:
    """Предварительно загружает матрицы и быстро проверяет структурированные правила."""

    def __init__(self, dataset: str, fpr_penalty: float = 0.2, *, train_size: int | None = None,
                 validation_size: int = 4096, feature_set: str = "expanded", cache: bool = True):
        """Фиксирует состав выборок и сетки порогов из train."""
        if not 0 <= fpr_penalty <= 2:
            raise ValueError("Штраф FPR: 0–2")
        self.fpr_penalty = fpr_penalty
        self.dataset = dataset
        self.sample = {"id": "synthetic-211-377-v1" if dataset == "synthetic" else "unsw-legacy-small",
                       "feature_set": feature_set}
        self.validation_categories = None
        if dataset == "unsw" and train_size is not None:
            train, validation, self.validation_categories, self.sample = training_parts(
                train_size, validation_size, feature_set)
            self.train, self.validation = (train[0].T, train[1]), (validation[0].T, validation[1])
        else:
            self.train, self.validation = load_parts(dataset)
        self.features = (UNSW_FEATURES if dataset == "unsw" and (feature_set == "legacy" or train_size is None)
                         else features_for(dataset))
        self.indices = {name: index for index, name in enumerate(self.features)}
        self.sample.update(train_size=len(self.train[1]), validation_size=len(self.validation[1]))
        self._cached = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = 8 * 1024 * 1024 if cache else 0
        self.thresholds = {}
        for name, index in self.indices.items():
            column = self.train[0][:, index]
            unique = np.unique(column)
            grid = unique if len(unique) <= 16 else np.percentile(column, np.arange(0, 101, 5))
            if len(unique) > 16 and unique[0] == 0:
                grid = np.r_[grid, np.percentile(column[column > 0], [0, 10, 25, 50, 75, 90])]
            self.thresholds[name] = sorted({round(float(x), 6) for x in grid})

    def mask(self, item: dict, split: str) -> np.ndarray:
        """Вычисляет условие и переиспользует маски с ограничением памяти."""
        op = item["operator"]
        if op in ("and", "or", "not"):
            masks = [self.mask(child, split) for child in item["conditions"]]
            if op == "not":
                return ~masks[0]
            method = np.logical_and if op == "and" else np.logical_or
            return method.reduce(masks)
        feature = item["feature"]
        if feature not in self.indices:
            raise ValueError(f"Признак {feature} недоступен в выбранном наборе")
        threshold, upper = item["threshold"], item.get("upper")
        key = (split, feature, op, threshold, upper)
        if key in self._cached:
            self._cached.move_to_end(key)
            return self._cached[key]
        values = getattr(self, split)[0][:, self.indices[feature]]
        result = values <= threshold if op == "<=" else values >= threshold
        if op == "between":
            result &= values <= upper
        if result.nbytes <= self._cache_limit:
            while self._cache_bytes + result.nbytes > self._cache_limit:
                self._cache_bytes -= self._cached.popitem(last=False)[1].nbytes
            result.setflags(write=False)
            self._cached[key] = result
            self._cache_bytes += result.nbytes
        return result

    def scores(self, rule: dict, split: str) -> np.ndarray:
        """Суммирует веса проверенных условий."""
        scores = np.zeros(len(getattr(self, split)[1]), dtype=np.uint8)
        for condition in rule["rules"]:
            scores += self.mask(condition, split) * np.uint8(condition["weight"])
        return scores

    def _predictions(self, rule: dict, split: str) -> np.ndarray:
        """Вычисляет предсказания только для train и validation."""
        rule = normalize(rule, self.dataset)
        if split not in ("train", "validation"):
            raise ValueError("Разрешены только train и validation")
        return self.scores(rule, split) >= rule["decision_threshold"]

    def evaluate(self, rule: dict, split: str = "validation") -> dict:
        """Векторно считает матрицу ошибок и метрики без Python-циклов по событиям."""
        predicted = self._predictions(rule, split)
        labels = getattr(self, split)[1]
        return measure_predictions(predicted, labels, self.fpr_penalty)

    def mutate(self, parent: dict, rng: random.Random) -> dict:
        """Случайно меняет один компонент правила, используя квантили train."""
        result = json.loads(identity(parent))
        items = result["rules"]
        action = rng.choice(["add", "add", "threshold", "threshold", "weight", "remove", "operator", "decision"])
        if action == "add" and len(items) < MAX_RULES:
            options = [(name, op) for name in self.features for op in (">=", "<=")
                       if not any(condition_key(item) == (name, op) for item in items)]
            if options:
                name, op = rng.choice(options)
                items.append({"feature": name, "operator": op,
                              "threshold": rng.choice(self.thresholds[name]), "weight": rng.choice([1, 2])})
        elif action == "remove" and len(items) > 1:
            items.pop(rng.randrange(len(items)))
        elif action == "operator":
            index = rng.randrange(len(items))
            item = items[index]
            if item.get("feature") is None or item["operator"] == "between":
                return parent
            other = "<=" if item["operator"] == ">=" else ">="
            if not any(i != index and x.get("feature") == item["feature"] and x["operator"] == other
                       for i, x in enumerate(items)):
                item["operator"] = other
        elif action == "decision":
            result["decision_threshold"] += rng.choice([-1, 1])
        else:
            item = rng.choice(items)
            if action == "weight":
                item["weight"] = max(1, min(3, item["weight"] + rng.choice([-1, 1])))
            else:
                if item.get("feature") is None or item["operator"] == "between":
                    return parent
                thresholds = self.thresholds[item["feature"]]
                current = min(range(len(thresholds)), key=lambda index: abs(thresholds[index] - item["threshold"]))
                index = max(0, min(len(thresholds)-1, current + rng.choice([-2, -1, 1, 2])))
                item["threshold"] = thresholds[index]
        result["decision_threshold"] = max(1, min(result["decision_threshold"],
                                                  sum(item["weight"] for item in items)))
        return normalize(result, self.dataset)

    def crossover(self, a: dict, b: dict, rng: random.Random) -> dict:
        """Объединяет части двух родителей, затем делает небольшую мутацию."""
        conditions = {condition_key(item): item
                      for item in a["rules"] + b["rules"] }
        selections = rng.sample(list(conditions.values()), min(MAX_RULES, len(conditions)))
        child = {"rules": selections,
                 "decision_threshold": max(1, min(rng.choice([a["decision_threshold"], b["decision_threshold"]]),
                                                 sum(item["weight"] for item in selections)))}
        return self.mutate(normalize(child, self.dataset), rng)
