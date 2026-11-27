"""Сохранение результатов попыток ShinkaEvolve для последующей визуализации."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def record_evaluation(program_path: Path, metrics: dict) -> None:
    """Сохраняет оценённую программу и метрики, если включена запись истории."""
    target = os.getenv("EVO_HISTORY_DIR", "").strip()
    if not target:
        return
    directory = Path(target).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    evaluated = directory / "candidates"
    evaluated.mkdir(exist_ok=True)
    # В штатном запуске Shinka одновременно выполняется не более одной оценки.
    counter = len(list(evaluated.glob("candidate_*.py")))
    destination = evaluated / f"candidate_{counter:04d}.py"
    shutil.copyfile(program_path, destination)
    item = {
        "generation": counter, "id": f"s{counter}", "parent": None,
        "kind": "shinka", "path": destination.relative_to(directory).as_posix(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }
    with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
