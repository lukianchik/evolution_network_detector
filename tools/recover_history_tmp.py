"""Возвращает результаты прерванной эволюции из старого history.tmp.

Пример: python -m tools.recover_history_tmp results/hybrid-groq-20261121-112536-04e1
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from artifacts.history_store import read_history_payload, save_history_payload


def recover(folder: Path) -> bool:
    """Восстанавливает только если временный файл содержит более свежий прогресс."""
    folder = Path(folder).resolve()
    candidate = folder / "history.tmp"
    if not candidate.is_file():
        print("Старого history.tmp нет; восстанавливать нечего.")
        return False
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("evaluations"), list):
        raise ValueError("history.tmp не содержит корректную историю")
    if not all(isinstance(row, dict) and "id" in row for row in data["evaluations"]):
        raise ValueError("history.tmp содержит некорректные записи")
    try:
        old = read_history_payload(folder)
    except FileNotFoundError:
        old = {"evaluations": [], "statistics": {}}
    before = len(old["evaluations"])
    after = len(data["evaluations"])
    old_attempts = old.get("statistics", {}).get("local_attempts", 0)
    new_attempts = data.get("statistics", {}).get("local_attempts", 0)
    if after < before or (after == before and new_attempts <= old_attempts):
        print(f"Актуальная история не старее: {before} кандидатов. Замена не нужна.")
        return False
    original = folder / "history.json"
    if original.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = folder / f"history.before-recovery.{stamp}.bak"
        shutil.copy2(original, backup)
        print(f"Резервная копия: {backup.name}")
    canonical = save_history_payload(folder, data)
    print(f"Восстановлено: {before} -> {after} кандидатов. "
          f"Сохранение: {'history.json' if canonical else 'резервный снимок'}")
    return True


def main() -> None:
    """Получает папку запуска через CLI."""
    parser = argparse.ArgumentParser(description="Восстановить историю из history.tmp после WinError 5")
    parser.add_argument("folder", type=Path, help="Папка конкретного эксперимента")
    args = parser.parse_args()
    recover(args.folder)


if __name__ == "__main__":
    main()
