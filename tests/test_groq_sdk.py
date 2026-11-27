"""Проверки транспорта Groq и правил обработки HTTP-ошибок без интернета."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations import provider_api
from integrations.quotas import QuotaLedger


class FakeResponse:
    """Имитирует ответ Python SDK Groq."""

    def model_dump(self):
        """Возвращает структуру Chat Completions."""
        return {"choices": [{"message": {"content": "def detect(event): return 0"}}],
                "usage": {"total_tokens": 20}}


class FakeHttpError(Exception):
    """Имитирует ошибку HTTP официального Groq SDK."""

    def __init__(self, code: int, text: str, headers: dict | None = None):
        """Сохраняет код и заголовки ответа."""
        self.status_code = code
        self.response = types.SimpleNamespace(status_code=code, text=text, headers=headers or {})


class FakeConnectionError(Exception):
    """Имитирует сетевую ошибку Groq SDK."""


class FakeClient:
    """Имитирует Groq и считает обращения без раскрытия ключа."""

    calls = 0
    errors = []
    configs = []

    def __init__(self, **kwargs):
        """Сохраняет настройки SDK и имитирует его API."""
        type(self).configs.append(kwargs)
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        """Возвращает подставной ответ либо ошибку."""
        type(self).calls += 1
        if self.errors:
            problem = self.errors.pop(0)
            if problem:
                raise problem
        return FakeResponse()

    def close(self):
        """Закрывает тестовый клиент."""
        return None


class GroqSdkTests(unittest.TestCase):
    """Проверяет отдельный SDK, локальный ledger и отсутствие повтора при 403."""

    def setUp(self):
        """Сбрасывает имитацию перед каждым тестом."""
        FakeClient.calls = 0
        FakeClient.errors = []
        FakeClient.configs = []
        self.sdk = types.SimpleNamespace(Groq=FakeClient, APIStatusError=FakeHttpError,
                                         APIConnectionError=FakeConnectionError)

    def _run(self, ledger):
        """Отправляет один логический запрос через Groq."""
        return provider_api.request_completion(
            "groq", "openai/gpt-oss-120b", [{"role": "user", "content": "hello"}],
            max_tokens=30, ledger=ledger,
        )

    def test_sdk_is_used_without_implicit_retries(self):
        """Groq SDK вызывается напрямую с отключёнными скрытыми ретраями."""
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"groq": self.sdk}), \
             patch.dict(os.environ, {"GROQ_API_KEY": "fake"}):
            ledger = QuotaLedger(Path(directory) / "quota.db")
            reply = self._run(ledger)
        self.assertIn("choices", reply)
        self.assertEqual(FakeClient.calls, 1)
        self.assertEqual(FakeClient.configs[0]["max_retries"], 0)
        self.assertEqual(FakeClient.configs[0]["api_key"], "fake")

    def test_sdk_http_429_retries_after_short_pause(self):
        """Лимит Groq 429 ожидает Retry-After и повторяет вызов SDK."""
        class FakeLedger:
            """Обрабатывает паузы без реального ожидания."""

            def __init__(self):
                """Подготавливает журнал."""
                self.pauses = []
                self.finished = []

            def reserve(self, *_args, **_kwargs):
                """Возвращает идентификатор записи."""
                return 1

            def finish(self, *args):
                """Фиксирует учтённый запрос."""
                self.finished.append(args)

            def pause(self, *args):
                """Запоминает время ожидания."""
                self.pauses.append(args)

        FakeClient.errors = [FakeHttpError(429, "rate limited", {"Retry-After": "2"}), None]
        ledger = FakeLedger()
        with patch.dict(sys.modules, {"groq": self.sdk}), \
             patch.dict(os.environ, {"GROQ_API_KEY": "fake"}):
            reply = self._run(ledger)
        self.assertIn("choices", reply)
        self.assertEqual(FakeClient.calls, 2)
        self.assertGreaterEqual(ledger.pauses[0][1], 2)

    def test_1010_does_not_retry(self):
        """Cloudflare 1010 останавливает вызов вместо ожидания лимита."""
        FakeClient.errors = [FakeHttpError(403, "error code: 1010")]
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"groq": self.sdk}), \
             patch.dict(os.environ, {"GROQ_API_KEY": "fake"}):
            ledger = QuotaLedger(Path(directory) / "quota.db")
            with self.assertRaisesRegex(RuntimeError, "Cloudflare 1010"):
                self._run(ledger)
        self.assertEqual(FakeClient.calls, 1)


if __name__ == "__main__":
    unittest.main()
