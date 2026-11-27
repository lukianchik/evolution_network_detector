"""Устойчивое сохранение истории эксперимента на Windows и других ОС.

Основной history.json обновляется атомарно. При временной блокировке файла
несколько раз повторяем замену; если она не удалась, оставляем пригодный
для чтения уникальный резервный снимок и не прерываем эволюцию.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

RETRIES = (0.02, 0.04, 0.08, 0.16, 0.32)
RECOVERY_PATTERN = "history.recovery.*.json"


def selected_row(rows: list[dict], payload: dict | None = None) -> dict:
    """Читает зафиксированного лидера, сохраняя критерий старых архивов."""
    winner = (payload or {}).get("winner_id")
    if winner is None:
        return max(rows, key=lambda row: row["fitness"])
    for row in rows:
        if row["id"] == winner:
            return row
    raise ValueError("Зафиксированный лидер отсутствует в истории")


def candidate_path(folder: Path, relative: str) -> Path:
    """Resolve a saved Python candidate within its experiment directory."""
    # Older Windows runs stored backslashes in JSON even when read on Unix.
    root = folder.resolve()
    path = (root / relative.replace("\\", "/")).resolve()
    if path.suffix != ".py" or not path.is_relative_to(root):
        raise ValueError("Некорректный путь кандидата")
    return path


def _recovery_files(folder: Path) -> list[Path]:
    """Возвращает только резервные снимки в каталоге запуска."""
    return list(folder.glob(RECOVERY_PATTERN))


def has_history(folder: Path) -> bool:
    """Проверяет, есть ли полноценная или резервная история."""
    return (folder / "history.json").is_file() or any(_recovery_files(folder))


def read_history_payload(folder: Path) -> dict:
    """Читает наиболее свежую полную версию, включая аварийные снимки."""
    folder = Path(folder)
    files = ([folder / "history.json"] if (folder / "history.json").is_file() else [])
    files += _recovery_files(folder)
    # Во время очистки резервных снимков файл может исчезнуть между glob и stat.
    available = []
    for path in files:
        try:
            available.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    files = [path for _stamp, path in sorted(available, key=lambda item: (item[0], item[1].name), reverse=True)]
    if not files:
        raise FileNotFoundError(f"Не найдена история эксперимента в {folder}")
    last_error = None
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("evaluations"), list):
                raise ValueError(f"Неправильный формат истории: {path}")
            return payload
        except (OSError, ValueError, json.JSONDecodeError) as error:
            last_error = error
    raise ValueError(f"Не удалось прочитать историю: {last_error}")


def save_history_payload(folder: Path, payload: dict, *, retries=RETRIES) -> bool:
    """Атомарно сохраняет JSON; возвращает False при использовании резервного снимка.

    На Windows другой процесс может временно удерживать history.json,
    блокируя os.replace. Не затрагиваем рабочий JSON, пока новый не готов.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    fd, path = tempfile.mkstemp(prefix="history-write-", suffix=".tmp", dir=folder)
    temp = Path(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        target = folder / "history.json"
        for delay in (*retries, None):
            try:
                os.replace(temp, target)
                # Убираем резервные версии только после успешной замены.
                for old in _recovery_files(folder):
                    try:
                        old.unlink()
                    except OSError:
                        pass
                return True
            except PermissionError:
                if delay is None:
                    break
                time.sleep(delay)
        # Даже если history.json заблокирован, новый уникальный файл не занят.
        fallback = folder / f"history.recovery.{time.time_ns()}.{os.getpid()}.json"
        os.replace(temp, fallback)
        # Оставляем максимум три восстановительных снимка.
        previous = sorted(_recovery_files(folder), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for old in previous[3:]:
            try:
                old.unlink()
            except OSError:
                pass
        return False
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
