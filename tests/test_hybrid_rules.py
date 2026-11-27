"""Тесты самоэволюционирующих правил, MCP-метрик и бюджета внешних LLM."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations import mcp_metrics
from search import local_evolution
from detectors.rule_engine import RuleEvaluator, normalize, seed_rule, to_python
from detectors.safe_program import validate_detector, predict
from detectors.metrics import measure
from ui.visualize import load_history

ROOT = Path(__file__).resolve().parents[1]


class RuleEngineTests(unittest.TestCase):
    """Проверяет локальный поиск и отсутствие исполняемого пользовательского JSON."""

    def test_code_from_rule_is_valid(self):
        """Из структурированных условий получается допустимая Python-функция."""
        for dataset in ("synthetic", "unsw"):
            source = to_python(seed_rule(dataset), dataset)
            with patch.dict(os.environ, {"EVO_DATASET": dataset}):
                validate_detector(source)
            self.assertIn("def detect(event: dict)", source)

    def test_generated_python_matches_vectorized_metrics(self):
        """Визуализируемый Python возвращает ровно те же ответы, что и быстрый оценщик."""
        import random
        evaluator = RuleEvaluator("synthetic")
        rule = seed_rule("synthetic")
        rng = random.Random(17)
        for _ in range(9):
            rule = evaluator.mutate(rule, rng)
            source = to_python(rule, "synthetic")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "detector.py"
                path.write_text(source, encoding="utf-8")
                events = [{name: float(value) for name, value in zip(evaluator.features, row)}
                          for row in evaluator.validation[0]]
                with patch.dict(os.environ, {"EVO_DATASET": "synthetic"}):
                    actual = measure(evaluator.validation[1].tolist(), predict(path, events))
                expected = evaluator.evaluate(rule)
                self.assertEqual(actual["fitness"], expected["fitness"])
                self.assertEqual(actual["fpr"], expected["fpr"])

    def test_forbidden_fields_and_test_split(self):
        """Запрещены метки, опасные параметры и прямой вызов тестовой выборки."""
        base = seed_rule("synthetic")
        for field in ("label", "attack_cat", "__class__"):
            changed = json.loads(json.dumps(base))
            changed["rules"][0]["feature"] = field
            with self.assertRaises(ValueError):
                normalize(changed, "synthetic")
        for bad in (float("nan"), 1e100, "__import__('os')"):
            changed = json.loads(json.dumps(base))
            changed["rules"][0]["threshold"] = bad
            with self.assertRaises(ValueError):
                normalize(changed, "synthetic")
        with self.assertRaises(ValueError):
            RuleEvaluator("synthetic").evaluate(base, "test")

    def test_local_improves_without_external_calls(self):
        """Локальный цикл создаёт новые версии и не использует API."""
        with tempfile.TemporaryDirectory() as temp, patch.object(local_evolution, "request_completion") as api:
            folder = local_evolution.evolve_local("local", "synthetic", 3, Path(temp),
                                                   seed=42, per_generation=8, mcp_check=False)
            history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(history["statistics"]["llm_calls"], 0)
            self.assertEqual(history["statistics"]["local_attempts"], 24)
            self.assertGreater(len(history["evaluations"]), 2)
            self.assertGreater(max(row["fitness"] for row in history["evaluations"]),
                               history["evaluations"][0]["fitness"])
            self.assertTrue((folder / history["evaluations"][-1]["path"]).exists())
            api.assert_not_called()
            mode, rows, _ = load_history(folder)
            self.assertEqual(mode, "local")
            self.assertTrue(rows)

    def test_json_llm_response_is_validated(self):
        """Короткий JSON-ответ Groq проходит проверку whitelist полей без Python eval."""
        target = {"rules": [{"feature": "unique_ports", "operator": ">=", "threshold": 20, "weight": 1}],
                  "decision_threshold": 1}
        response = {"choices": [{"message": {"content": "```json\n" + json.dumps(target) + "\n```"}}]}
        with patch.dict(os.environ, {"GROQ_MODEL": "openai/gpt-oss-120b"}), \
                patch.object(local_evolution, "request_completion", return_value=response) as client:
            result = local_evolution.propose_rule("groq", RuleEvaluator("synthetic"), seed_rule("synthetic"),
                                                  {"f1": 0.1, "fpr": 0.2})
            self.assertEqual(result, normalize(target, "synthetic"))
            self.assertEqual(client.call_args.kwargs["max_tokens"], 480)

    def test_hybrid_calls_only_with_budget(self):
        """Гибрид просит новое правило только ограниченное число раз."""
        def improvement(*_args, **_kwargs):
            """Возвращает проверяемое новое правило вместо реальной LLM."""
            return normalize({"rules": [{"feature": "unique_ports", "operator": ">=",
                                         "threshold": 20, "weight": 1}],
                              "decision_threshold": 1}, "synthetic")
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), \
                patch.object(local_evolution, "propose_rule", side_effect=improvement) as api, \
                patch.object(local_evolution.Objective, "rank", return_value=0):
            folder = local_evolution.evolve_local("hybrid", "synthetic", 2, Path(temp),
                                                  seed=42, per_generation=6,
                                                  stagnation_limit=1, max_llm_calls=1, mcp_check=False)
            data = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(data["statistics"]["llm_calls"], 1)
            self.assertEqual(api.call_count, 1)
            self.assertEqual(data["statistics"]["local_attempts"], 12)

    def test_hybrid_zero_budget_needs_no_key(self):
        """При нулевом бюджете гибрид полностью автономен и ключ не нужен."""
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"GROQ_API_KEY": ""}), \
                patch.object(local_evolution, "propose_rule") as api:
            folder = local_evolution.evolve_local("hybrid", "synthetic", 1, Path(temp),
                                                  per_generation=4, stagnation_limit=1,
                                                  max_llm_calls=0, mcp_check=False)
            data = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(data["statistics"]["llm_calls"], 0)
            api.assert_not_called()

    def test_hybrid_failure_falls_back(self):
        """Ошибочная LLM не останавливает локальную эволюцию."""
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), \
                patch.object(local_evolution, "propose_rule", side_effect=RuntimeError("HTTP 403")) as api, \
                patch.object(local_evolution.Objective, "rank", return_value=0):
            folder = local_evolution.evolve_local("hybrid", "synthetic", 2, Path(temp),
                                                  seed=42, per_generation=5,
                                                  stagnation_limit=1, max_llm_calls=1, mcp_check=False)
            data = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(data["statistics"]["llm_calls"], 1)
            self.assertEqual(data["statistics"]["llm_errors"], 1)
            self.assertEqual(data["statistics"]["local_attempts"], 10)
            api.assert_called_once()


class McpHandlersTests(unittest.TestCase):
    """Проверяет содержимое инструментов MCP без сетевого подключения."""

    def test_evaluate_and_compare(self):
        """MCP использует тот же оценщик и не разрешает test."""
        raw = json.dumps(seed_rule("synthetic"))
        first = mcp_metrics.evaluate_rule(raw, dataset="synthetic")
        self.assertGreaterEqual(first["metrics"]["f1"], 0)
        second = mcp_metrics.compare_rules(raw, raw, dataset="synthetic")
        self.assertEqual(second["delta_f1"], 0)
        with self.assertRaises(ValueError):
            mcp_metrics.evaluate_rule(raw, dataset="synthetic", split="test")

    def test_saved_rule_and_summary(self):
        """MCP возвращает только правило и статистику разрешённого запуска."""
        name = "test-mcp-" + next(tempfile._get_candidate_names())
        directory = ROOT / "results" / name
        try:
            local_evolution.evolve_local("local", "synthetic", 1, directory, per_generation=3, mcp_check=False)
            best = mcp_metrics.get_best_rule(name)
            summary = mcp_metrics.get_search_summary(name)
            self.assertIn("rules", best["rule"])
            self.assertGreater(summary["evaluated"], 0)
            self.assertEqual(summary["statistics"]["llm_calls"], 0)
            with self.assertRaises(ValueError):
                mcp_metrics.get_search_summary("../../.env")
        finally:
            if directory.exists():
                shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
