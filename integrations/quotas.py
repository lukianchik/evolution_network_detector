"""Локальный учёт квот API через SQLite для нескольких процессов.

Это защитные ограничения проекта, не данные о фактическом балансе провайдера.
"""

import math
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class QuotaExceeded(RuntimeError):
    """Означает, что безопасный бюджет запросов или ожидания исчерпан."""


def limits_for(provider: str) -> dict:
    """Читает локальные осторожные ограничения из переменных окружения."""
    if provider != "groq":
        raise ValueError("Поддерживается Groq")
    prefix = "GROQ"
    defaults = (25, 900, 7000, 180000, 20)
    keys = ("RPM", "RPD", "TPM", "TPD", "MAX_WAIT")
    values = {name: int(os.getenv(prefix + "_" + name, str(default)))
              for name, default in zip(keys, defaults)}
    if any(value <= 0 for name, value in values.items() if name != "MAX_WAIT") or values["MAX_WAIT"] < 0:
        raise ValueError(f"Некорректные локальные лимиты {prefix}")
    return values


def quota_key(provider: str, model: str) -> str:
    """Возвращает лимитный пул Groq для модели."""
    if provider != "groq":
        raise ValueError("Поддерживается Groq")
    return "groq/" + model


def retry_after_seconds(headers, fallback: float = 8) -> float:
    """Обрабатывает Retry-After как число секунд или HTTP-дату."""
    value = next((value for key, value in headers.items() if key.lower() == "retry-after"), None) if headers else None
    if value is None:
        return float(fallback)
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = date.timestamp() - time.time()
        except (ValueError, TypeError, OverflowError, IndexError):
            return float(fallback)
    return max(1.0, seconds + 0.5) if math.isfinite(seconds) else float(fallback)


class QuotaLedger:
    """Резервирует один запрос и токены атомарно для всех процессов проекта."""

    def __init__(self, path: Path | None = None, *, clock=time.time, sleep=time.sleep):
        """Создаёт локальный журнал без API-ключей или текстов промптов."""
        self.path = Path(path or ROOT / ".state" / "quotas.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.sleep = sleep
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, pool TEXT, at REAL, tokens INTEGER)")
            db.execute("CREATE INDEX IF NOT EXISTS calls_pool_at ON calls(pool, at)")
            db.execute("CREATE TABLE IF NOT EXISTS cooldown (pool TEXT PRIMARY KEY, until REAL)")

    @contextmanager
    def _db(self):
        """Открывает отдельное соединение SQLite для каждого атомарного действия."""
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def reserve(self, pool: str, expected_tokens: int, limits: dict, max_wait: float | None = None) -> int:
        """Ожидает свободный слот и записывает запрос до отправки HTTP."""
        if expected_tokens < 1 or expected_tokens > min(limits["TPM"], limits["TPD"]):
            raise QuotaExceeded("Запрос превышает локальный токен-бюджет: уменьши контекст или MAX_TOKENS")
        budget = limits["MAX_WAIT"] if max_wait is None else min(limits["MAX_WAIT"], max(0, max_wait))
        started = self.clock()
        while True:
            now = self.clock()
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM calls WHERE at <= ?", (now - 86400,))
                cooldown = db.execute("SELECT until FROM cooldown WHERE pool=?", (pool,)).fetchone()
                wait = max(0.0, cooldown["until"] - now) if cooldown else 0.0
                for window, request_limit, token_limit in ((60, limits["RPM"], limits["TPM"]),
                                                           (86400, limits["RPD"], limits["TPD"])):
                    records = list(db.execute("SELECT at,tokens FROM calls WHERE pool=? AND at>? ORDER BY at,id", (pool, now-window)))
                    count, tokens = len(records), sum(record["tokens"] for record in records)
                    for record in records:
                        if count < request_limit and tokens + expected_tokens <= token_limit:
                            break
                        wait = max(wait, record["at"] + window - now + 0.05)
                        count -= 1
                        tokens -= record["tokens"]
                if wait <= 0:
                    cursor = db.execute("INSERT INTO calls(pool,at,tokens) VALUES(?,?,?)", (pool, now, expected_tokens))
                    db.commit()
                    return int(cursor.lastrowid)
                db.commit()
            if self.clock() - started + wait > budget:
                raise QuotaExceeded(f"Локальный лимит {pool}: следующий слот через ~{math.ceil(wait)} с. "
                                    "Достигнут установленный MAX_WAIT; промежуточные результаты сохранены")
            if wait >= 10 and (not hasattr(self, "_notice_until") or now >= self._notice_until):
                print(f"Локальный лимит {pool}: ожидаем ~{math.ceil(wait)} с, "
                      "для отмены нажми «Стоп».", flush=True)
                self._notice_until = now + 60
            self.sleep(min(wait, 1.0))

    def finish(self, reservation: int, used_tokens: int | None) -> None:
        """Заменяет оценку токенов фактическим значением, когда оно известно."""
        if used_tokens is not None and used_tokens >= 0:
            with self._db() as db:
                db.execute("UPDATE calls SET tokens=? WHERE id=?", (int(used_tokens), reservation))

    def pause(self, pool: str, seconds: float) -> None:
        """Устанавливает общий срок паузы после ответа API с кодом 429."""
        with self._db() as db:
            db.execute("INSERT INTO cooldown(pool,until) VALUES(?,?) "
                       "ON CONFLICT(pool) DO UPDATE SET until=max(until,excluded.until)",
                       (pool, self.clock() + max(1, seconds)))

    def snapshot(self, pool: str, limits: dict) -> dict:
        """Возвращает использование локальной квоты за последние минуту и сутки."""
        now = self.clock()
        with self._db() as db:
            windows = {}
            for name, window in (("minute", 60), ("day", 86400)):
                row = db.execute("SELECT COUNT(*) AS n, COALESCE(SUM(tokens),0) AS t "
                                 "FROM calls WHERE pool=? AND at>?", (pool, now-window)).fetchone()
                windows[name] = {"requests": row["n"], "tokens": row["t"]}
            row = db.execute("SELECT until FROM cooldown WHERE pool=?", (pool,)).fetchone()
        return {"pool": pool, "windows": windows,
                "cooldown_seconds": round(max(0, row["until"] - now), 1) if row else 0,
                "limits": limits}
