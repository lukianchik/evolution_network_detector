"""Регрессия диагностики внешних API и остановки повторов Shinka."""
import io
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from integrations import provider_api
from search import run_shinka


class ShinkaReliabilityTests(unittest.TestCase):
    """Проверяет предварительные проверки без сетевых запросов."""

    def test_preflight_rejects_empty_response(self):
        """Пустой ответ LLM не считается доступностью генерации."""
        with patch.object(run_shinka, "request_completion", return_value={"choices": [{"message": {"content": ""}}]}):
            with self.assertRaisesRegex(RuntimeError, "пустой"):
                run_shinka.check_groq("test-model")

    def test_preflight_uses_groq(self):
        """Проверка направляется именно в реальный API Groq."""
        with patch.object(run_shinka, "request_completion", return_value={"choices": [{"message": {"content": "yes"}}]}) as request:
            run_shinka.check_groq("test-model")
            self.assertEqual(request.call_args.args[:2], ("groq", "test-model"))

    def test_preserves_proxy_for_external_apis(self):
        """Добавляет localhost, не теряя действующие исключения прокси в Windows и Linux."""
        # В Windows имена переменных окружения не зависят от регистра:
        # NO_PROXY и no_proxy — одна переменная. Нельзя задавать им разные
        # значения одновременно и ожидать, что сохранятся оба.
        for name, external_host in (("NO_PROXY", "example.com"),
                                    ("no_proxy", "example.org")):
            with self.subTest(variable=name):
                with patch.dict(os.environ, {name: external_host}):
                    run_shinka.ensure_local_bypass()
                    value = os.environ[name].split(",")
                    self.assertIn(external_host, value)
                    self.assertIn("127.0.0.1", value)
                    self.assertIn("localhost", value)
                    self.assertIn("::1", value)

    def test_groq_503_only_retries_twice(self):
        """После двух повторов 503 возвращается явная ошибка."""
        def unavailable(*args, **kwargs):
            raise urllib.error.HTTPError("https://example.invalid", 503,
                                         "unavailable", {}, io.BytesIO(b"temporarily unavailable"))
        from integrations.quotas import QuotaLedger
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"GROQ_API_KEY": "fake"}), \
                patch.object(provider_api, "groq_sdk_completion", side_effect=unavailable) as fetch, \
                patch.object(provider_api.time, "sleep"):
            ledger = QuotaLedger(path=Path(td)/"quota.db")
            with self.assertRaisesRegex(RuntimeError, "после 3 попыток"):
                provider_api.request_completion("groq", "test-model", [{"role": "user", "content": "Hi"}], 100, ledger=ledger)
            self.assertEqual(fetch.call_count, 3)


if __name__ == "__main__":
    unittest.main()
