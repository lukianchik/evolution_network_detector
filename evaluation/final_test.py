"""Независимая финальная проверка победившего детектора на скрытых тестовых событиях.

Тестовые seed не используются в процессе эволюции и выбора родителей.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from traffic_data.dataset import build_dataset
from traffic_data.dataset_real import iter_full_test, load_unsw, selected_dataset
from artifacts.history_store import candidate_path, has_history, read_history_payload, selected_row
from detectors.metrics import from_counts, measure, summarize
from detectors.safe_program import load_detector, predict
from ui.visualize import load_history

ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = (7001, 7002, 7003)


def test_program(path: Path, *, sample_size: int | None = None, fpr_penalty: float = 0.2,
                 reserve: bool = False, control_csv: Path | None = None) -> dict:
    """Считает полный UNSW test потоково или фиксированную синтетическую оценку."""
    measurements = []
    dataset = selected_dataset()
    if (reserve or control_csv is not None) and (dataset != "unsw" or sample_size is not None
                                                or (reserve and control_csv is not None)):
        raise ValueError("Укажи один полный контрольный набор для UNSW")
    if dataset == "unsw":
        if sample_size is None or reserve:
            detector = load_detector(path)
            counts = [0, 0, 0, 0]
            by_category = {}
            if reserve:
                from traffic_data.sampling import reserved_flows
                flows = reserved_flows()
            else:
                flows = iter_full_test(categories=True, control_csv=control_csv)
            for event, label, category in flows:
                prediction = detector(event)
                index = 0 if label and prediction else 1 if prediction else 2 if label else 3
                counts[index] += 1
                group = category or ("Unknown attack" if label else "Normal")
                by_category.setdefault(group, [0, 0, 0, 0])[index] += 1
            return {**from_counts(*counts, fpr_penalty),
                    "categories": {key: from_counts(*values, fpr_penalty) for key, values in by_category.items()}}
        events, labels = load_unsw("test", sample_size, expanded=True)
        measurements.append(measure(list(labels), predict(path, list(events)), fpr_penalty))
    else:
        for seed in TEST_SEEDS:
            events, labels = build_dataset(seed, 700)
            measurements.append(measure(labels, predict(path, events), fpr_penalty))
    return summarize(measurements)


def best_from_history(folder: Path) -> Path:
    """Возвращает путь записанного лидера или исторического победителя по fitness."""
    history = read_history_payload(folder) if has_history(folder) else None
    _mode, rows, _rejected = load_history(folder, history)
    best = selected_row(rows, history)
    return candidate_path(folder, best["path"])


def main() -> None:
    """Сравнивает исходный детектор с выбранным вариантом на отложенном тесте."""
    parser = argparse.ArgumentParser(description="Финальная оценка, без участия LLM")
    parser.add_argument("--run", default="results/demo", help="Каталог результатов эволюции")
    parser.add_argument("--program", help="Явный путь к Python-программе (опционально)")
    parser.add_argument("--dataset", choices=("synthetic", "unsw"), default=None)
    parser.add_argument("--sample-size", type=int, help="Ограниченная выборка UNSW вместо полного test")
    parser.add_argument("--reserve", action="store_true", help="Внутренний групповой резерв вместо test")
    parser.add_argument("--control-csv", type=Path, help="Отдельный размеченный CSV в схеме UNSW")
    parser.add_argument("--fpr-penalty", type=float, default=None)
    parser.add_argument("--output", type=Path, help="Сохранить JSON итоговой оценки")
    args = parser.parse_args()
    if args.dataset:
        os.environ["EVO_DATASET"] = args.dataset
    elif not args.program:
        name_file = Path(args.run).resolve() / "dataset.txt"
        if name_file.is_file():
            os.environ["EVO_DATASET"] = name_file.read_text(encoding="utf-8").strip()
    folder = Path(args.run).resolve()
    penalty = args.fpr_penalty
    winner = None
    baseline = ROOT / "seeds" / ("initial_unsw.py" if selected_dataset() == "unsw" else "initial.py")
    if args.program:
        path = Path(args.program).resolve()
    else:
        history = read_history_payload(folder) if has_history(folder) else None
        _mode, rows, _rejected = load_history(folder, history)
        winner = selected_row(rows, history)
        path = candidate_path(folder, winner["path"])
        baseline = candidate_path(folder, rows[0]["path"])
        if penalty is None and history is not None:
            penalty = history.get("statistics", {}).get("fpr_penalty", 0.2)
    penalty = 0.2 if penalty is None else penalty
    options = {"sample_size": args.sample_size, "fpr_penalty": penalty,
               "reserve": args.reserve, "control_csv": args.control_csv}
    base = test_program(baseline, **options)
    candidate = test_program(path, **options)
    split = "synthetic-test" if selected_dataset() == "synthetic" else "official-test"
    if args.reserve:
        split = "train-reserve"
    if args.control_csv is not None:
        split = "external-control"
    report = {"dataset": selected_dataset(), "sample_size": args.sample_size,
              "winner_id": winner["id"] if winner else None,
              "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "split": split, "fpr_penalty": penalty, "baseline": base, "candidate": candidate}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Разница F1: {candidate['f1'] - base['f1']:+.3f}; FPR: {candidate['fpr'] - base['fpr']:+.3f}")


if __name__ == "__main__":
    main()
