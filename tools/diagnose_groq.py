"""Проверка доступа к Groq API без генерации текста и вывода ключа."""

import os
from pathlib import Path

from search.evolution import load_env


def main() -> int:
    """Проверяет, что официальный SDK может запросить список моделей."""
    load_env(Path(__file__).resolve().parents[1] / ".env")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("Не найден GROQ_API_KEY в .env")
        return 2
    try:
        import groq
    except ImportError:
        print("Установи зависимости: python -m pip install -r requirements.txt")
        return 2
    client = groq.Groq(api_key=key, timeout=20, max_retries=0)
    try:
        result = client.models.list()
        print(f"Groq API доступен через официальный SDK. Моделей в ответе: {len(result.data)}")
        print("Это проверка соединения; генерация кода и лимиты Chat Completions ещё не проверены.")
        return 0
    except groq.APIStatusError as error:
        print(f"Groq API ответил HTTP {error.status_code}.")
        content = (error.response.text if error.response is not None else str(error)).lower()
        if error.status_code == 403 and "1010" in content:
            print("Cloudflare 1010: запрос заблокирован по сигнатуре клиента, это не лимит квоты.")
        elif error.status_code == 401:
            print("Проверь GROQ_API_KEY; в терминал его не вставляй.")
        else:
            print("Посмотри настройки доступа в Groq Console или обратись в поддержку провайдера.")
        ray = error.response.headers.get("cf-ray") if error.response is not None else None
        if ray:
            print(f"Cloudflare Ray ID для обращения в поддержку: {ray}")
        return 1
    except groq.APIConnectionError as error:
        print(f"Проблема соединения с Groq: {type(error).__name__}. Проверь сеть и системный прокси.")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
