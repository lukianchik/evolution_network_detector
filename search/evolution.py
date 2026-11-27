"""Общие настройки и совместимая точка входа локального поиска."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROQ_SETTINGS = frozenset({"GROQ_API_KEY", "GROQ_MODEL", "GROQ_RPM", "GROQ_RPD",
                           "GROQ_TPM", "GROQ_TPD", "GROQ_MAX_WAIT"})


def load_env(path: Path) -> None:
    """Считывает простые настройки KEY=VALUE из .env, не перезаписывая переменные среды."""
    if not path.exists():
        return
    for row in path.read_text(encoding="utf-8-sig").splitlines():
        line = row.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in GROQ_SETTINGS:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


if __name__ == "__main__":
    from search.local_evolution import main

    raise SystemExit(main())
