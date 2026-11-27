"""Независимый верификатор кандидатов ShinkaEvolve.

Записывает стандартные metrics.json и correct.json. Результат на отложенном
тестовом наборе здесь намеренно НЕ рассчитывается, чтобы не выбирать по нему код.
Запуск: python -m evaluation.evaluate --program_path seeds/initial.py --results_dir results/seed
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from traffic_data.dataset import build_dataset
from traffic_data.dataset_real import load_unsw, selected_dataset
from detectors.metrics import measure, summarize
from detectors.safe_program import predict
from artifacts.history import record_evaluation

VALIDATION_SEEDS = (21, 37, 49)
SAMPLE_SIZE = 400


def evaluate_program(path: Path) -> dict:
    """Проверяет формат ответов и рассчитывает среднее качество по трём выборкам."""
    if os.getenv("EVO_COMMON_VALIDATION") == "1":
        from detectors.rule_engine import RuleEvaluator
        dataset = selected_dataset()
        evaluator = RuleEvaluator(dataset, train_size=10000)
        x, y = evaluator.validation
        events = [dict(zip(evaluator.features, row.tolist())) for row in x]
        return measure(y.tolist(), predict(path, events), float(os.getenv("EVO_FPR_PENALTY", "0.2")))
    results = []
    if selected_dataset() == "unsw":
        # Берём только официальную обучающую часть; test не участвует в эволюции.
        events, labels = load_unsw("train", 1800, expanded=True)
        predictions = predict(path, list(events))
        results.append(measure(list(labels), predictions))
    else:
        for seed in VALIDATION_SEEDS:
            events, labels = build_dataset(seed, SAMPLE_SIZE)
            predictions = predict(path, events)
            results.append(measure(labels, predictions))
    return summarize(results)


def write_result(directory: Path, values: dict | None, error: str = "") -> None:
    """Сохраняет совместимые с ShinkaEvolve результаты оценки кандидата."""
    directory.mkdir(parents=True, exist_ok=True)
    correct = values is not None
    metrics = {
        "combined_score": values["fitness"] if correct else 0.0,
        "public": ({key: values[key] for key in ("f1", "precision", "recall", "fpr")} if correct else {}),
        "private": {},
        "extra_data": ({"confusion": {key: values[key] for key in ("tp", "fp", "fn", "tn")}} if correct else {}),
        "text_feedback": ("Улучшайте F1, но не ценой множества ложных тревог" if correct else error),
    }
    (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "correct.json").write_text(json.dumps({"correct": correct, "error": error}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """Принимает аргументы ShinkaEvolve, оценивает код и сохраняет метрики."""
    parser = argparse.ArgumentParser(description="Оценка детектора для ShinkaEvolve")
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    try:
        values = evaluate_program(Path(args.program_path))
        write_result(Path(args.results_dir), values)
        record_evaluation(Path(args.program_path), values)
        print(json.dumps(values, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        write_result(Path(args.results_dir), None, str(exc))
        print(f"Ошибка оценки: {exc}", file=sys.stderr)
        traceback.print_exc(limit=3)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
