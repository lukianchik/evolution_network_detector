"""Проверяет MCP-контроль поколений, обработку ответов и отказоустойчивость."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from search import local_evolution
from integrations import mcp_bridge
from integrations import mcp_metrics
from search.evolution import ROOT
from detectors.rule_engine import normalize


class McpWiringTests(unittest.TestCase):
    """Выполняет эволюцию с контрольными MCP-вызовами без внешних API."""

    def test_real_handlers_used_at_generation_boundary(self):
        """Координатор запрашивает compare_rules и проверяет совпадение метрик."""
        calls = []

        def protocol_mock(tools, timeout=15):
            """Имитирует транспорт, но вызывает реальные MCP-обработчики."""
            response = []
            for name, arguments in tools:
                calls.append(name)
                response.append(getattr(mcp_metrics, name)(**arguments))
            return response

        name = "test-mcp-coordinator-" + next(tempfile._get_candidate_names())
        folder = ROOT / "results" / name
        try:
            with patch.object(local_evolution, "call_tools", side_effect=protocol_mock):
                local_evolution.evolve_local("local", "synthetic", 2, folder,
                                             per_generation=3, mcp_check=True)
            report = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(calls, ["compare_rules", "compare_rules"])
            self.assertEqual(report["statistics"]["mcp_calls"], 2)
            self.assertEqual(report["statistics"]["mcp_verified"], 2)
            self.assertEqual(report["statistics"]["mcp_errors"], 0)
            self.assertEqual(len(report["mcp_events"]), 2)
            self.assertTrue(all(event["ok"] for event in report["mcp_events"]))
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_stagnation_feedback_is_aggregated(self):
        """При застое координатор передаёт статистику без сырых событий или меток."""
        calls, passed = [], []

        def protocol_mock(tools, timeout=15):
            """Подключает локальные функции за интерфейсом MCP."""
            result = []
            for name, arguments in tools:
                calls.append(name)
                result.append(getattr(mcp_metrics, name)(**arguments))
            return result

        def llm_mock(provider, evaluator, rule, score, *, mcp_summary=None):
            """Сохраняет фактические параметры, не вызывая модель."""
            passed.append(mcp_summary)
            return normalize({"rules": [{"feature": "unique_ports", "operator": ">=",
                                         "threshold": 20, "weight": 1}],
                              "decision_threshold": 1}, "synthetic")
        name = "test-mcp-stagnation-" + next(tempfile._get_candidate_names())
        folder = ROOT / "results" / name
        try:
            with patch.dict(local_evolution.os.environ, {"GROQ_API_KEY": "dummy"}), \
                    patch.object(local_evolution, "call_tools", side_effect=protocol_mock), \
                    patch.object(local_evolution, "propose_rule", side_effect=llm_mock), \
                    patch.object(local_evolution.Objective, "rank", return_value=0):
                local_evolution.evolve_local("hybrid", "synthetic", 1, folder,
                                             per_generation=4, stagnation_limit=1,
                                             max_llm_calls=1, mcp_check=True)
            stats = json.loads((folder / "history.json").read_text(encoding="utf-8"))["statistics"]
            self.assertEqual(stats["mcp_verified"], 1)
            self.assertEqual(stats["llm_calls"], 1)
            self.assertIn("compare_rules", calls)
            self.assertEqual(len(passed), 1)
            self.assertEqual(set(passed[0]), {"objective", "train_errors", "strategy_attempts", "strategy_gains"})
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_broken_mcp_does_not_fake_success(self):
        """Неработающий MCP отмечается ошибкой, локальная эволюция не прерывается."""
        with tempfile.TemporaryDirectory() as td, patch.object(
                local_evolution, "call_tools", side_effect=RuntimeError("no server")):
            local_evolution.evolve_local("local", "synthetic", 2, Path(td),
                                         per_generation=3, mcp_check=True)
            data = json.loads((Path(td) / "history.json").read_text(encoding="utf-8"))
            self.assertGreater(len(data["evaluations"]), 1)
            self.assertEqual(data["statistics"]["mcp_verified"], 0)
            self.assertEqual(data["statistics"]["mcp_errors"], 2)
            self.assertTrue(all(not event["ok"] for event in data["mcp_events"]))

    def test_inconsistent_mcp_fails_validation(self):
        """Медленно или неправильно работающий сервер не влияет на локальный fitness."""
        def misleading(_calls, timeout=15):
            return [{"first": {"fitness": 1, "f1": 1, "fpr": 0},
                     "second": {"fitness": 1, "f1": 1, "fpr": 0},
                     "delta_fitness": 0.0}]
        with tempfile.TemporaryDirectory() as td, patch.object(
                local_evolution, "call_tools", side_effect=misleading):
            local_evolution.evolve_local("local", "synthetic", 1, Path(td), per_generation=3)
            data = json.loads((Path(td) / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(data["statistics"]["mcp_verified"], 0)
            self.assertEqual(data["statistics"]["mcp_errors"], 1)
            self.assertLess(max(r["fitness"] for r in data["evaluations"]), 1.0)

    def test_parse_mcp_result(self):
        """Обычный текстовый и структурированный ответ MCP корректно читаются."""
        example = {"fitness": 0.42}
        self.assertEqual(mcp_bridge.parse_tool_result(SimpleNamespace(is_error=False,
                         structured_content=example)), example)
        self.assertEqual(mcp_bridge.parse_tool_result(SimpleNamespace(is_error=False,
                         structured_content=None, content=[SimpleNamespace(type="text", text=json.dumps(example))])), example)
        with self.assertRaisesRegex(RuntimeError, "ошибку"):
            mcp_bridge.parse_tool_result(SimpleNamespace(is_error=True,
                                  content=[SimpleNamespace(text="broken")]))

    def test_client_starts_stdio_server_with_python(self):
        """Проверяет сборку stdio-команды через интерфейс официального SDK."""
        import sys
        launched = []

        class FakeParams:
            """Хранит аргументы запуска MCP-процесса."""
            def __init__(self, **kwargs):
                launched.append(kwargs)

        class FakeClient:
            """Имитирует асинхронный транспортный контракт SDK."""
            def __init__(self, _parameters):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def call_tool(self, name, args):
                assert name == "evaluate_rule"
                assert args["dataset"] == "synthetic"
                return SimpleNamespace(is_error=False, structured_content={"metrics": {"f1": 0.4}})

        with patch.dict("os.environ", {"EVO_MCP_URL": ""}), patch.dict(sys.modules, {"mcp": SimpleNamespace(Client=FakeClient,
                        StdioServerParameters=FakeParams)}):
            result = mcp_bridge.call_tools([("evaluate_rule", {"dataset": "synthetic"})])
        self.assertEqual(result[0]["metrics"]["f1"], 0.4)
        self.assertEqual(launched[0]["command"], sys.executable)
        self.assertEqual(launched[0]["args"], ["-m", "integrations.mcp_metrics"])
        self.assertEqual(launched[0]["env"]["PYTHONUTF8"], "1")

    def test_bad_request_count(self):
        """Запрещает бесконечные пакеты вызовов MCP."""
        with self.assertRaises(ValueError):
            mcp_bridge.call_tools([])
        with self.assertRaises(ValueError):
            mcp_bridge.call_tools([("evaluate_rule", {})] * 5)


if __name__ == "__main__":
    unittest.main()
