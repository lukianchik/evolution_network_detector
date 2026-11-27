"""Запросы Groq с локальным учётом квот и фактического использования."""

import io
import math
import os
import time
import urllib.error

from integrations.quotas import QuotaLedger, limits_for, quota_key, retry_after_seconds
from integrations.usage import record as record_usage

GROQ_BASE = "https://api.groq.com"


def estimated_tokens(messages: list[dict], max_tokens: int) -> int:
    """Оценивает расход токенов с запасом до отправки запроса."""
    length = sum(len(str(message.get("content", ""))) for message in messages)
    return math.ceil(length / 3) + max_tokens + 150


def parse_reset(error: urllib.error.HTTPError) -> float:
    """Читает подсказку сервера о сроке сброса лимита."""
    headers = error.headers
    retry = retry_after_seconds(headers, fallback=30)
    if headers:
        import re
        # Groq различает сброс дневных запросов и минутных токенов.
        remaining = headers.get("x-ratelimit-remaining-requests", "")
        which = ("x-ratelimit-reset-requests" if remaining == "0"
                 else "x-ratelimit-reset-tokens")
        raw = headers.get(which, "")
        pattern = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", raw.strip())
        if pattern and raw.strip():
            values = [float(value or 0) for value in pattern.groups()]
            retry = max(retry, values[0] * 3600 + values[1] * 60 + values[2])
    return retry


def groq_sdk_completion(key: str, model: str, messages: list[dict],
                        max_tokens: int, temperature: float) -> dict:
    """Вызывает Groq через официальный SDK и нормализует HTTP-ошибки.

    Внутренние повторы SDK отключены: временем ожидания и квотами управляет
    request_completion, чтобы не расходовать запросы незаметно для счётчика.
    """
    try:
        import groq
    except ImportError as error:
        raise RuntimeError("Для Groq установи зависимости: "
                           "python -m pip install -r requirements.txt") from error
    client = groq.Groq(api_key=key, base_url=GROQ_BASE, timeout=20, max_retries=0)
    try:
        reply = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, stream=False,
        )
        return reply.model_dump()
    except groq.APIStatusError as error:
        # Преобразуем ошибку в тот же формат, что использует обработчик квот.
        # Это сохраняет Retry-After и x-ratelimit-* для ожидания при HTTP 429.
        response = error.response
        detail = response.text[:600] if response is not None else str(error)[:600]
        headers = dict(response.headers) if response is not None else {}
        raise urllib.error.HTTPError(f"{GROQ_BASE}/openai/v1/chat/completions", error.status_code, detail,
                                     headers, io.BytesIO(detail.encode("utf-8"))) from error
    except groq.APIConnectionError as error:
        raise urllib.error.URLError(str(error)) from error
    finally:
        client.close()


def request_completion(provider: str, model: str, messages: list[dict],
                       max_tokens: int = 1800, temperature: float = 0.4,
                       *, ledger: QuotaLedger | None = None, max_attempts: int | None = None,
                       max_wait: float | None = None) -> dict:
    """Учитывает попытки, ожидание квоты и фактическое использование Groq."""
    if provider != "groq":
        raise ValueError("Неизвестный провайдер")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("Укажи GROQ_API_KEY в .env")
    limits = limits_for(provider)
    ledger = ledger or QuotaLedger()
    pool = quota_key(provider, model)
    budget = estimated_tokens(messages, max_tokens)
    attempts = 0
    http_attempts = 0
    temporary_failures = 0
    wait_limit = limits["MAX_WAIT"] if max_wait is None else min(limits["MAX_WAIT"], max_wait)
    started = time.monotonic()
    while True:
        if max_attempts is not None and http_attempts >= max_attempts:
            raise RuntimeError(f"{provider}: достигнут бюджет HTTP-попыток; локальный поиск продолжается")
        remaining = wait_limit - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError(f"{provider}: истекло предельное время ожидания MAX_WAIT; "
                               "промежуточные результаты сохранены")
        reservation = ledger.reserve(pool, budget, limits, max_wait=remaining)
        http_attempts += 1
        record_usage({"kind": "attempt", "provider": provider, "model": model})
        try:
            payload = groq_sdk_completion(key, model, messages, max_tokens, temperature)
            usage = payload.get("usage", {})
            record_usage({"kind": "response", "provider": provider, "model": model,
                          "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                                    if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}})
            used = usage.get("total_tokens") if isinstance(usage, dict) else None
            ledger.finish(reservation, used if type(used) is int and used >= 0 else None)
            return payload
        except urllib.error.HTTPError as error:
            record_usage({"kind": "error", "provider": provider, "model": model, "status": error.code})
            detail = error.read(600).decode("utf-8", "replace")
            if error.code == 429:
                attempts += 1
                pause = parse_reset(error)
                # Если суточный лимит известен из сообщения, безопаснее ждать суточного сброса.
                if any(word in detail.lower() for word in ("per day", "daily", "per_day", "rpd")):
                    pause = max(pause, 3600)
                ledger.pause(pool, pause)
                if max_attempts is not None and http_attempts >= max_attempts:
                    raise RuntimeError(f"{provider}: HTTP 429; исчерпан бюджет HTTP-попыток") from error
                print(f"{provider}: HTTP 429, ждём {math.ceil(pause)} секунд до повторного запроса "
                      f"(попытка {attempts}). Можно нажать «Стоп».", flush=True)
                continue
            if error.code in (500, 502, 503, 504):
                temporary_failures += 1
                if temporary_failures <= 2 and (max_attempts is None or http_attempts < max_attempts):
                    pause = min(12, 2 ** temporary_failures)
                    print(f"{provider}: HTTP {error.code}, временная ошибка сервиса. "
                          f"Повтор через {pause} с (из 2 повторов).", flush=True)
                    time.sleep(pause)
                    continue
                raise RuntimeError(
                    f"{provider}: HTTP {error.code} после {http_attempts} попыток. "
                    "Внешний сервис сейчас недоступен. "
                    "Проверь доступность сервиса."
                ) from error
            if provider == "groq" and error.code == 403 and "1010" in detail:
                raise RuntimeError(
                    "Groq HTTP 403 / Cloudflare 1010: соединение заблокировано "
                    "по сигнатуре HTTP-клиента. Ожидание квоты не поможет. "
                    "Проверь прямой запрос через официальный SDK и настройки "
                    "сети/прокси; при повторении обратись в поддержку Groq."
                ) from error
            raise RuntimeError(f"{provider} HTTP {error.code}") from error
        except urllib.error.URLError as error:
            record_usage({"kind": "error", "provider": provider, "model": model, "status": "connection"})
            raise RuntimeError(f"Нет связи с {provider}") from error
