"""Локальная телеметрия фактических API-попыток без ключей и промптов."""
import json
import os
import threading
from pathlib import Path

LOCK = threading.Lock()


def initialize(folder: Path) -> None:
    """Создаёт новый журнал для одного CLI-запуска, не смешивая эксперименты."""
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "history.json").exists() or (folder / "history.jsonl").exists() or list(folder.glob("history.recovery.*.json")):
        raise ValueError("Каталог уже содержит эксперимент; укажи новый --output")
    path = folder / "api_usage.jsonl"
    path.touch(exist_ok=False)
    os.environ["EVO_USAGE_FILE"] = str(path)


def record(event: dict) -> None:
    """Пишет одно событие на запрос или ответ; отказ записи не мешает поиску."""
    target = os.getenv("EVO_USAGE_FILE")
    if not target:
        return
    try:
        with LOCK, Path(target).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        print("Не удалось записать телеметрию API", flush=True)


def read_usage(folder: Path) -> dict:
    """Разделяет HTTP-попытки, успешные ответы и известное потребление токенов."""
    path = folder / "api_usage.jsonl"
    out = {"http_attempts": 0, "http_errors": 0, "responses": 0, "known_usage_responses": 0,
           "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    out["usage_available"] = path.is_file()
    if not out["usage_available"]:
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("kind") == "attempt":
            out["http_attempts"] += 1
        elif item.get("kind") == "error":
            out["http_errors"] += 1
        elif item.get("kind") == "response":
            out["responses"] += 1
            usage = item.get("usage", {})
            if type(usage.get("total_tokens")) is int and usage["total_tokens"] >= 0:
                out["known_usage_responses"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    out[key] += value
    return out
