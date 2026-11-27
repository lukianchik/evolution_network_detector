"""Проверки реального CSV и ожидания лимитов без сетевых запросов."""

import csv
import io
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from traffic_data import dataset_real
from integrations import provider_api
from evaluation.evaluate import evaluate_program
from evaluation.final_test import test_program as final_evaluate
from integrations.quotas import QuotaExceeded, QuotaLedger

ROOT = Path(__file__).resolve().parents[1]


class CsvTests(unittest.TestCase):
    """Проверяет раздельность train/test и отсутствие меток среди признаков."""

    def test_real_dataset(self):
        """Разметка никогда не попадает в словарь входного сетевого события."""
        with tempfile.TemporaryDirectory() as temp:
            train, test = Path(temp) / "train.csv", Path(temp) / "test.csv"
            names = list(dataset_real.FEATURES) + ["label", "attack_cat", "id"]
            for path, offset in ((train, 0), (test, 200)):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=names)
                    writer.writeheader()
                    for index in range(40):
                        label = int(index % 4 == 0)
                        row = {name: str((index + offset) % 20 + 1) for name in dataset_real.FEATURES}
                        row.update({"sttl": "180" if label else "10", "label": str(label),
                                    "attack_cat": "Fuzzers" if label else "Normal", "id": str(index)})
                        writer.writerow(row)
            dataset_real.load_unsw.cache_clear()
            with patch.object(dataset_real, "TRAIN", train), patch.object(dataset_real, "TEST", test):
                with patch.dict(os.environ, {"EVO_DATASET": "unsw"}):
                    events, labels = dataset_real.load_unsw("train", 20)
                    self.assertEqual(len(events), 20)
                    self.assertTrue(all("label" not in event and "attack_cat" not in event
                                        for event in events))
                    self.assertEqual(len(labels), 20)
                    values = evaluate_program(ROOT / "seeds" / "initial_unsw.py")
                    # Новый baseline намеренно простой: высокий F1 не требуется.
                    self.assertGreaterEqual(values["f1"], 0)
                    self.assertLessEqual(values["f1"], 1)
                    reference = final_evaluate(ROOT / "seeds" / "initial_unsw_strong.py")
                    self.assertGreater(reference["f1"], 0.5)
            dataset_real.load_unsw.cache_clear()


class QuotaTests(unittest.TestCase):
    """Проверяет ожидание освободившегося окна квоты и 429."""

    def test_ledger_wait_and_limit(self):
        """Лимит RPM заставляет ждать вместо немедленной ошибки."""
        class FakeClock:
            """Имитирует ход времени без настоящего сна."""
            now = 1000.0
            def time(self):
                """Возвращает управляемые часы."""
                return self.now
            def sleep(self, seconds):
                """Ускоренно сдвигает время."""
                self.now += seconds
        clock = FakeClock()
        limits = {"RPM": 1, "RPD": 2, "TPM": 5000, "TPD": 10000, "MAX_WAIT": 100}
        with tempfile.TemporaryDirectory() as temp:
            ledger = QuotaLedger(Path(temp) / "limits.db", clock=clock.time, sleep=clock.sleep)
            ledger.reserve("router", 100, limits)
            with self.assertRaises(QuotaExceeded):
                ledger.reserve("router", 100, limits, max_wait=2)
            ledger.reserve("router", 100, limits, max_wait=100)
            self.assertGreaterEqual(clock.now, 1060)
            self.assertEqual(ledger.snapshot("router", limits)["windows"]["minute"]["requests"], 1)

    def test_retry_after(self):
        """Парсит ожидание из заголовка Retry-After."""
        from integrations.quotas import retry_after_seconds
        self.assertGreaterEqual(retry_after_seconds({"Retry-After": "17"}), 17)

    def test_http_429_then_success(self):
        """При HTTP 429 передаёт паузу квотному счётчику и повторяет запрос."""
        class FakeLedger:
            """Фиксирует вызовы квот без локальной SQLite."""
            def __init__(self):
                """Создаёт счётчики вызовов."""
                self.reservations = 0
                self.pauses = []
            def reserve(self, *args, **kwargs):
                """Разрешает тестовый запрос."""
                self.reservations += 1
                return self.reservations
            def pause(self, pool, seconds):
                """Запоминает ожидание."""
                self.pauses.append((pool, seconds))
            def finish(self, *_args):
                """Ничего не делает при завершении."""
                return

        error = urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests",
                                       {"Retry-After": "3"}, io.BytesIO(b"rate limited"))
        success = {"choices": [{"message": {"content": "def detect(event): return 0"}}],
                   "usage": {"total_tokens": 15}}
        ledger = FakeLedger()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            with patch.object(provider_api, "groq_sdk_completion", side_effect=[error, success]) as http:
                result = provider_api.request_completion("groq", "test-model",
                                                         [{"role": "user", "content": "code"}],
                                                         max_tokens=20, ledger=ledger)
        self.assertEqual(http.call_count, 2)
        self.assertEqual(ledger.reservations, 2)
        self.assertTrue(ledger.pauses[0][1] >= 3)
        self.assertIn("choices", result)


if __name__ == "__main__":
    unittest.main()
