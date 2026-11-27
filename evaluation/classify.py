"""Применение выбранного детектора к структурированным сетевым событиям JSONL."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from detectors.safe_program import load_detector
from evaluation.final_test import best_from_history
from traffic_data.dataset_real import flow_features, selected_dataset


def classify(input_path: Path, detector_path: Path, output_path: Path) -> int:
    """Считывает события и записывает по одному ответу детектора на строку JSONL."""
    detector = load_detector(detector_path)
    count = 0
    if output_path.resolve() in {input_path.resolve(), detector_path.resolve()}:
        raise ValueError("Выходной файл совпадает с исходным")
    temporary = None
    try:
        with input_path.open(encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w", dir=output_path.parent, encoding="utf-8", delete=False
        ) as target:
            temporary = Path(target.name)
            index = 0
            while (line := source.readline(65537)):
                index += 1
                if len(line) > 65536:
                    raise ValueError("Строка JSONL слишком длинная")
                if not line.strip():
                    continue
                if count >= 5000:
                    raise ValueError("Максимум 5000 событий за запуск")
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"Строка {index}: ожидается JSON-объект")
                features = {**event, **flow_features(event)} if selected_dataset() == "unsw" else event
                prediction = detector(features)
                if type(prediction) not in (int, bool) or prediction not in (0, 1):
                    raise ValueError(f"Строка {index}: детектор вернул не 0/1")
                target.write(json.dumps({"event": event, "prediction": int(prediction)}, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main() -> None:
    """Позволяет указать файл событий и папку с результатами эволюции."""
    parser = argparse.ArgumentParser(description="Классификация структурированных сетевых событий")
    parser.add_argument("--input", default="examples/events.jsonl")
    parser.add_argument("--output", default="results/predictions.jsonl")
    parser.add_argument("--run", default="results/demo")
    parser.add_argument("--program")
    parser.add_argument("--dataset", choices=("synthetic", "unsw"))
    args = parser.parse_args()
    if not args.program:
        marker = Path(args.run) / "dataset.txt"
        if marker.is_file():
            os.environ["EVO_DATASET"] = marker.read_text(encoding="utf-8").strip()
    if args.dataset:
        os.environ["EVO_DATASET"] = args.dataset
    detector = Path(args.program).resolve() if args.program else best_from_history(Path(args.run).resolve())
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Обработано {classify(Path(args.input), detector, target)} событий, результат: {target}")


if __name__ == "__main__":
    main()
