"""Регрессии релизного кандидата: полный test, безопасный AST, MCP и A/B."""
import csv
import io
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from app import ExperimentManager
from detectors.metrics import measure
from detectors.rule_engine import RuleEvaluator, identity, seed_rule
from detectors.safe_program import load_detector
from evaluation.final_test import test_program as final_evaluate
from integrations import mcp_metrics, provider_api
from integrations.usage import initialize, read_usage
from search.local_evolution import evolve_local
from tools.benchmark import run_suite
from traffic_data import dataset_real


class ReleaseTests(unittest.TestCase):
    """Проверяет новые контракты без внешних API."""

    def test_numeric_interpreter_does_not_exec(self):
        """Код вне detect игнорируется, корректная арифметика сохраняет смысл."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text('raise RuntimeError("outside")\ndef detect(event):\n    x = event.get("failed_logins", 0)\n    x += 2\n    return int(2 <= x < 5 and not False)\n', encoding="utf-8")
            with patch.dict(os.environ, {"EVO_DATASET": "synthetic"}):
                detector = load_detector(path)
                self.assertEqual(detector({"failed_logins": 2}), 1)
                self.assertEqual(detector({"failed_logins": 4}), 0)

    def test_intermediate_values_bounded(self):
        """Отклоняет разрастание чисел, строковые операции и нечисловые входы."""
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"EVO_DATASET": "synthetic"}):
            path = Path(tmp) / "candidate.py"
            cases = [('def detect(event):\n    x = 10000000000\n' + '    x *= x\n' * 8 + '    return 0', {}),
                     ('def detect(event): return int("a" * 1000000000 == "")', {}),
                     ('def detect(event): return int(event["failed_logins"] > 0)', {"failed_logins": "x"}),
                     ('def detect(event): return int(event["failed_logins"] > 0)', {"failed_logins": float("nan")})]
            for source, event in cases:
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_detector(path)(event)

    def test_full_unsw_test_more_than_sample_limit(self):
        """Считывает больше 5000 строк и не загружает train для финальной оценки."""
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"EVO_DATASET": "unsw"}):
            root = Path(tmp)
            path, detector = root / "test.csv", root / "candidate.py"
            detector.write_text('def detect(event): return int(event["sttl"] >= 100)', encoding="utf-8")
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[*dataset_real.FEATURES, "label"])
                writer.writeheader()
                for i in range(6001):
                    writer.writerow({**dict.fromkeys(dataset_real.FEATURES, 0), "sttl": 180 if i % 2 else 1, "label": i % 2})
            with patch.object(dataset_real, "TEST", path), patch.object(dataset_real, "TRAIN", root / "missing.csv"):
                result = final_evaluate(detector)
                self.assertEqual(result["count"], 6001)
                self.assertEqual(result["f1"], 1)
                self.assertEqual(result["fpr"], 0)
            with patch.object(dataset_real, "TEST", root / "unavailable.csv"):
                self.assertEqual(final_evaluate(detector, control_csv=path), result)

    def test_penalty_matches_mcp(self):
        """При другом штрафе локальный и MCP-оценщики согласованы."""
        rule = seed_rule("synthetic")
        local = RuleEvaluator("synthetic", 1.0).evaluate(rule)
        remote = mcp_metrics.evaluate_rule(identity(rule), "synthetic", fpr_penalty=1.0)["metrics"]
        self.assertEqual(local, remote)
        self.assertLess(measure([1, 0], [1, 1], 1)["fitness"], measure([1, 0], [1, 1])["fitness"])

    def test_telemetry_and_hybrid_http_budget(self):
        """При 429 одна попытка фиксируется, скрытого повтора нет."""
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"GROQ_API_KEY": "fake"}):
            folder = Path(tmp)
            initialize(folder)
            from integrations.quotas import QuotaLedger
            ledger = QuotaLedger(folder / "quota.db")
            error = urllib.error.HTTPError("https://example.invalid", 429, "limited", {"Retry-After": "2"}, io.BytesIO(b"rate limit"))
            with patch.object(provider_api, "groq_sdk_completion", side_effect=error) as api:
                with self.assertRaisesRegex(RuntimeError, "429"):
                    provider_api.request_completion("groq", "test-model", [{"role": "user", "content": "x"}],
                                                    ledger=ledger, max_attempts=1, max_wait=20)
                self.assertEqual(api.call_count, 1)
            self.assertEqual(read_usage(folder)["http_attempts"], 1)
            self.assertEqual(read_usage(folder)["known_usage_responses"], 0)

    def test_provider_usage_exact(self):
        """Сохраняет фактический usage только при известном значении."""
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"GROQ_API_KEY": "fake"}):
            folder = Path(tmp)
            initialize(folder)
            from integrations.quotas import QuotaLedger
            reply = {"usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}}
            with patch.object(provider_api, "groq_sdk_completion", return_value=reply):
                provider_api.request_completion("groq", "test-model", [{"role": "user", "content": "x"}],
                                                ledger=QuotaLedger(folder / "quota.db"))
            usage = read_usage(folder)
            self.assertEqual(usage["total_tokens"], 13)
            self.assertEqual(usage["known_usage_responses"], 1)
            self.assertEqual(usage["http_attempts"], 1)

    def test_history_restored_and_compared(self):
        """Архив после перезапуска доступен по ID и не объявляется завершённым без статуса."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "local-test"
            evolve_local("local", "synthetic", 1, folder, per_generation=3, mcp_check=False, fpr_penalty=0.5)
            with patch("app.RESULTS", Path(tmp)):
                manager = ExperimentManager()
                self.assertEqual(manager.snapshot("local-test")["status"], "completed")
                row = manager.comparisons()[0]
                self.assertEqual(row["penalty"], 0.5)
                self.assertEqual(row["dataset"], "synthetic")
                self.assertGreater(row["evaluated"], 0)

    def test_paired_benchmark_zero_budget(self):
        """Парные seed с нулевым бюджетом LLM дают идентичный результат и не используют API."""
        with tempfile.TemporaryDirectory() as tmp:
            result = run_suite(Path(tmp) / "bench", "synthetic", ["local", "hybrid"], [42, 43], 1, 3, 0,
                               0.5, final=True)
            self.assertEqual(len(result["runs"]), 4)
            self.assertTrue(all(row["status"] == "completed" for row in result["runs"]))
            for local, hybrid in zip(result["runs"][::2], result["runs"][1::2]):
                self.assertEqual(local["validation_fitness"], hybrid["validation_fitness"])
                self.assertEqual(local["test_f1"], hybrid["test_f1"])
                self.assertEqual(local["http_attempts"], 0)
                self.assertEqual(hybrid["http_attempts"], 0)
                self.assertTrue(hybrid["usage_available"])
