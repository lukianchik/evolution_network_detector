"""Внутренние стратегии и многокритериальный отбор."""

import math
from collections import Counter

import numpy as np

from detectors.rule_engine import MAX_RULES, condition_key, identity, measure_predictions, normalize

STRATEGIES = ("mutation", "crossover", "guided", "diversity", "error")
PROFILES = {
    "balanced": (0.10, 0.90, 0.45, 0.35),
    "low-fpr": (0.05, 0.75, 1.00, 0.20),
    "high-recall": (0.20, 0.95, 0.25, 0.80),
}


class Objective:
    """Применяет фиксированный профиль к метрикам и сложности."""

    def __init__(self, profile, max_fpr, min_recall, scalar):
        fpr, recall, self.fpr_weight, self.recall_weight = PROFILES[profile]
        self.max_fpr = fpr if max_fpr is None else max_fpr
        self.min_recall = recall if min_recall is None else min_recall
        self.scalar = scalar

    def utility(self, score, size=0):
        """Оценивает пригодность решения для заданного профиля."""
        if self.scalar:
            return score["fitness"]
        return (score["f1"] - self.fpr_weight * (score["fpr"] + 2 * max(0, score["fpr"] - self.max_fpr))
                - self.recall_weight * max(0, self.min_recall - score["recall"]) - 0.002 * size)

    def rank(self, row):
        """Учитывает устойчивость и разрыв train/validation."""
        risk = 0 if self.scalar else 0.15 * row["instability"] + 0.2 * row["generalization_gap"]
        return self.utility(row, row["complexity"]) - risk


def vector(row):
    """Приводит все цели к максимизации."""
    return (row["f1"], -row["fpr"], row["recall"], -row["complexity"],
            -row["instability"], -row["generalization_gap"])


def dominates(a, b):
    """Проверяет строгое Pareto-доминирование."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


class Archive:
    """Хранит ограниченный фронт и выбирает родителей из разных областей."""

    def __init__(self, objective, size=24):
        self.rows = []
        self.objective = objective
        self.size = size

    def add(self, row):
        """Обновляет фронт и сохраняет граничные точки."""
        if self.objective.scalar:
            self.rows = sorted(self.rows + [row], key=self.objective.rank, reverse=True)[:self.size]
            return
        score = vector(row)
        if any(vector(old) == score or dominates(vector(old), score) for old in self.rows):
            return
        self.rows = [old for old in self.rows if not dominates(score, vector(old))]
        self.rows.append(row)
        if len(self.rows) <= self.size:
            return
        distance = dict.fromkeys((old["id"] for old in self.rows), 0.0)
        for axis in range(len(score)):
            ordered = sorted(self.rows, key=lambda old: vector(old)[axis])
            width = vector(ordered[-1])[axis] - vector(ordered[0])[axis]
            if width:
                distance[ordered[0]["id"]] = distance[ordered[-1]["id"]] = math.inf
                for left, center, right in zip(ordered, ordered[1:], ordered[2:]):
                    distance[center["id"]] += (vector(right)[axis] - vector(left)[axis]) / width
        leader = self.best()
        self.rows.remove(min((old for old in self.rows if old is not leader),
                             key=lambda old: (distance[old["id"]], self.objective.rank(old))))

    def best(self):
        """Возвращает кандидата по фиксированному профилю."""
        return max(self.rows, key=self.objective.rank)

    def parent(self, rng, strategy):
        """Чередует зрелые решения с крайними точками фронта."""
        if strategy == "diversity":
            return rng.choice(self.rows)
        leaders = sorted(self.rows, key=self.objective.rank, reverse=True)
        return leaders[0] if rng.random() < 0.55 else rng.choice(leaders[:5])


class Coordinator:
    """Распределяет попытки по результативности и стоимости проверок."""

    def __init__(self, excluded):
        self.strategies = tuple(name for name in STRATEGIES if name not in excluded)
        self.attempts = Counter()
        self.gains = Counter()
        self.costs = Counter()
        self.accepted = Counter()
        self.last_used = {}

    def choose(self, attempts_left, stalled, archive_size, *, adaptive):
        """Исследует каждую стратегию и меняет приоритет при застое."""
        feasible = [name for name in self.strategies if name != "crossover" or archive_size > 1]
        total = sum(self.attempts.values())
        if not adaptive:
            return feasible[total % len(feasible)]
        untried = next((name for name in feasible if not self.attempts[name]), None)
        if untried:
            return untried
        oldest = min(feasible, key=lambda name: self.last_used.get(name, 0))
        if total - self.last_used.get(oldest, 0) >= 2 * len(feasible):
            return oldest

        def priority(name):
            count = self.attempts[name]
            cost = self.costs[name] / count
            reward = (self.gains[name] + 0.015 * self.accepted[name]) / count
            uncertainty = 0.06 * math.sqrt(math.log1p(total) / count)
            exploration = 0.035 if stalled and name in ("diversity", "error") else 0.0
            diversity = 0.025 if archive_size <= 2 and name in ("guided", "diversity") else 0.0
            budget = 1 + 0.12 * math.log1p(cost / max(3, attempts_left))
            return (reward + uncertainty + exploration + diversity) / budget

        return max(feasible, key=priority)


class TrainSearch:
    """Предлагает пороги и уточняет правило по агрегированным ошибкам train."""

    def __init__(self, evaluator, objective, rng):
        self.ev = evaluator
        self.objective = objective
        self.rng = rng
        self.y = evaluator.train[1]
        self.catalogue = [
            {"feature": name, "operator": op, "threshold": threshold, "weight": 1}
            for name, values in evaluator.thresholds.items()
            for op, thresholds in ((">=", values[1:]), ("<=", values[:-1]))
            for threshold in thresholds
        ]
        self.primitives = []
        self.initialized = False

    def initialize(self):
        """Ранжирует простые условия один раз, сохраняя оба края компромисса."""
        if self.initialized:
            return 0
        self.initialized = True
        scored = [(measure_predictions(self.ev.mask(item, "train"), self.y, self.ev.fpr_penalty), item)
                  for item in self.catalogue]
        by_profile = sorted(scored, key=lambda pair: self.objective.utility(pair[0]), reverse=True)
        by_low_fpr = sorted(scored, key=lambda pair: pair[0]["f1"] - pair[0]["fpr"], reverse=True)
        unique = {}
        for score, item in by_profile[:36] + by_low_fpr[:24]:
            if score["recall"]:
                unique.setdefault(identity(item), item)
        self.primitives = list(unique.values())
        return len(scored)

    def guided(self, parent, iteration):
        """Перебирает порог решения, веса, удаление и сильные условия."""
        items, cutoff = parent["rules"], parent["decision_threshold"]
        yield from ((items, target) for target in range(1, sum(item["weight"] for item in items) + 1))
        for index, item in enumerate(items):
            others = items[:index] + items[index + 1:]
            if others:
                yield others, cutoff
            for weight in (1, 2, 3):
                yield others + [{**item, "weight": weight}], cutoff
        if not self.primitives:
            return
        offset = (iteration * 8) % len(self.primitives)
        ring = self.primitives[offset:] + self.primitives[:offset]
        occupied = {condition_key(item) for item in items}
        for condition in ring[:12]:
            yield [condition], 1
            if condition_key(condition) not in occupied and len(items) < MAX_RULES:
                yield items + [condition], cutoff
                yield items + [condition], cutoff + 1

    def error_driven(self, parent, iteration):
        """Ранжирует отсечки FP и восстановление FN по агрегированным пересечениям."""
        items, cutoff = parent["rules"], parent["decision_threshold"]
        base = self.ev.scores(parent, "train") >= cutoff
        fp, tp = base & (self.y == 0), base & (self.y == 1)
        fn, tn = ~base & (self.y == 1), ~base & (self.y == 0)
        nfp, ntp, nfn, ntn = (max(1, np.count_nonzero(mask)) for mask in (fp, tp, fn, tn))
        if not self.catalogue:
            return [], 0
        window = self.rng.sample(self.catalogue, min(len(self.catalogue), 96))
        window.extend(self.primitives[iteration % 12:iteration % 12 + 16])
        occupied = {condition_key(item) for item in items}
        ranked = []
        seen = set()
        for item in window:
            key = identity(item)
            if key in seen:
                continue
            seen.add(key)
            mask = self.ev.mask(item, "train")
            guard = np.count_nonzero(fp & ~mask) / nfp
            loss = np.count_nonzero(tp & ~mask) / ntp
            recover = np.count_nonzero(fn & mask) / nfn
            false_alarm = np.count_nonzero(tn & mask) / ntn
            ranked.append((guard - 2 * loss, recover - 2 * false_alarm, item))
        proposals = []
        for axis in (0, 1):
            for entry in sorted(ranked, key=lambda row: row[axis], reverse=True)[:8]:
                item = entry[2]
                if len(items) < MAX_RULES and condition_key(item) not in occupied:
                    proposals.append((items + [item], cutoff + (axis == 0)))
                # Дерево сохраняет точную AND/OR-семантику простого родителя.
                if len(items) == 1 and "feature" in items[0]:
                    group = {"operator": ("and", "or")[axis],
                             "conditions": [items[0], item], "weight": 1}
                    proposals.append(([group], 1))
        proposals.extend((items[:i] + items[i + 1:], cutoff) for i in range(len(items)) if len(items) > 1)
        return proposals, len(seen)

    def candidates(self, strategy, parent, second, iteration):
        """Нормализует предложения локальной стратегии перед оценкой."""
        extra = 0
        if strategy == "crossover":
            return [self.ev.crossover(parent, second, self.rng)], extra
        if strategy == "mutation":
            return [self.ev.mutate(parent, self.rng) for _ in range(4)], extra
        if strategy == "diversity":
            # Простые правила из разных областей пространства дают новые линии поиска.
            choices = self.rng.sample(self.catalogue, min(12, len(self.catalogue)))
            return [{"rules": [item], "decision_threshold": 1} for item in choices], extra
        if strategy == "guided":
            proposals = self.guided(parent, iteration)
        else:
            proposals, extra = self.error_driven(parent, iteration)
        candidates = []
        for items, cutoff in proposals:
            threshold = min(cutoff, sum(item["weight"] for item in items))
            try:
                candidates.append(normalize({"rules": items, "decision_threshold": threshold}, self.ev.dataset))
            except ValueError:
                continue
        return candidates, extra


def choose_promotions(batch, limit):
    """Выбирает кандидатов по профилю и на границах компромисса train."""
    selected = {}
    orders = (lambda row: row["utility"],
              lambda row: row["train"]["f1"] - row["train"]["fpr"],
              lambda row: row["train"]["recall"] - 0.5 * row["train"]["fpr"])
    rankings = [iter(sorted(batch, key=key, reverse=True)) for key in orders]
    for index in range(limit):
        row = next((candidate for candidate in rankings[index % len(rankings)]
                    if candidate["key"] not in selected), None)
        if row is not None:
            selected[row["key"]] = row
    return list(selected.values())
