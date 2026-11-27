"""Проверяет устойчивость истории к Windows-блокировке history.json."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artifacts import history_store
from search import local_evolution
import app
from integrations import mcp_metrics
from tools import recover_history_tmp
from ui.visualize import create_report, load_history


class HistoryWindowsTests(unittest.TestCase):
    """Проверяет временную и постоянную блокировку основного файла."""

    def test_transient_access_denied_retries_without_losing_data(self):
        """После отказов os.replace успешно обновляет history.json."""
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            actual = os.replace
            state = {"failures": 2}

            def simulate(src, dst):
                if Path(dst).name == "history.json" and state["failures"] > 0:
                    state["failures"] -= 1
                    raise PermissionError(13, "Access denied", str(dst))
                return actual(src, dst)

            with patch.object(history_store.os, "replace", side_effect=simulate), \
                    patch.object(history_store.time, "sleep") as sleep:
                ok = history_store.save_history_payload(folder, {"evaluations": [{"id": "r0"}]})
            self.assertTrue(ok)
            self.assertEqual(state["failures"], 0)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(len(history_store.read_history_payload(folder)["evaluations"]), 1)
            self.assertFalse(list(folder.glob("history-write-*.tmp")))

    def test_permanently_locked_main_file_does_not_abort_evolution(self):
        """При блокировке JSON весь цикл завершается и отчёт видит последние метрики."""
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            actual = os.replace
            failures = {"n": 0}

            def simulate(src, dst):
                if Path(dst).name == "history.json":
                    failures["n"] += 1
                    raise PermissionError(13, "Access denied", str(dst))
                return actual(src, dst)

            with patch.object(history_store.os, "replace", side_effect=simulate), \
                    patch.object(history_store.time, "sleep"):
                local_evolution.evolve_local("local", "synthetic", 2, folder,
                                             per_generation=3, mcp_check=False)
            self.assertGreater(failures["n"], 5)
            payload = history_store.read_history_payload(folder)
            self.assertEqual(payload["statistics"]["local_attempts"], 6)
            self.assertGreater(len(payload["evaluations"]), 1)
            self.assertFalse((folder / "history.json").exists())
            self.assertLessEqual(len(list(folder.glob("history.recovery.*.json"))), 3)
            self.assertEqual(len(load_history(folder)[1]), len(payload["evaluations"]))
            self.assertIn("F1", create_report(folder).read_text(encoding="utf-8"))
            self.assertFalse(list(folder.glob("history-write-*.tmp")))

    def test_canonical_write_cleans_previous_recovery(self):
        """После снятия блокировки резервный снимок больше не нужен."""
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            actual = os.replace

            def simulate(src, dst):
                if Path(dst).name == "history.json":
                    raise PermissionError(13, "Access denied")
                return actual(src, dst)

            with patch.object(history_store.os, "replace", side_effect=simulate), \
                    patch.object(history_store.time, "sleep"):
                self.assertFalse(history_store.save_history_payload(folder, {"evaluations": [{"id": "r0"}]}))
            self.assertTrue(list(folder.glob("history.recovery.*.json")))
            self.assertTrue(history_store.save_history_payload(folder, {"evaluations": [{"id": "r0"}, {"id": "r1"}]}))
            self.assertFalse(list(folder.glob("history.recovery.*.json")))
            self.assertEqual(len(history_store.read_history_payload(folder)["evaluations"]), 2)

    def test_panel_and_mcp_read_recovery_snapshot(self):
        """Панель и MCP отдают одинаковые актуальные результаты при заблокированном JSON."""
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            folder = base / "results" / "check-recovery"
            folder.mkdir(parents=True)
            seed = {"mode": "local", "evaluations": [{"id": "r0", "fitness": 0.1,
                    "f1": 0.2, "fpr": 0.0, "generation": 0, "path": "r0.py"}],
                    "statistics": {"local_attempts": 0}}
            update = {"mode": "local", "evaluations": [*seed["evaluations"],
                      {"id": "r1", "fitness": 0.3, "f1": 0.4, "fpr": 0.1,
                       "generation": 1, "path": "r1.py"}],
                      "statistics": {"local_attempts": 1}}
            (folder / "history.json").write_text(json.dumps(seed), encoding="utf-8")
            actual = os.replace

            def simulate(src, dst):
                if Path(dst).name == "history.json":
                    raise PermissionError(13, "Access denied")
                return actual(src, dst)

            with patch.object(history_store.os, "replace", side_effect=simulate), \
                    patch.object(history_store.time, "sleep"):
                self.assertFalse(history_store.save_history_payload(folder, update))
            manager = app.ExperimentManager()
            manager.runs["check-recovery"] = {"id": "check-recovery", "mode": "local",
                    "dataset": "synthetic", "status": "completed", "generations": 1,
                    "folder": folder, "logs": [], "return_code": 0}
            manager.active = "check-recovery"
            shown = manager.snapshot()
            self.assertEqual(len(shown["evaluations"]), 2)
            self.assertEqual(shown["statistics"]["local_attempts"], 1)
            with patch.object(mcp_metrics, "ROOT", base):
                summary = mcp_metrics.get_search_summary("check-recovery")
            self.assertEqual(summary["evaluated"], 2)
            self.assertEqual(summary["best_fitness"], 0.3)

    def test_legacy_tmp_recovery_preserves_original(self):
        """Восстанавливает потерянный кандидат из старого history.tmp и делает backup."""
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            old = {"evaluations": [{"id": "r0"}], "statistics": {"local_attempts": 1}}
            new = {"evaluations": [{"id": "r0"}, {"id": "r1"}],
                   "statistics": {"local_attempts": 2}}
            (folder / "history.json").write_text(json.dumps(old), encoding="utf-8")
            (folder / "history.tmp").write_text(json.dumps(new), encoding="utf-8")
            self.assertTrue(recover_history_tmp.recover(folder))
            self.assertFalse(recover_history_tmp.recover(folder))
            self.assertEqual(len(history_store.read_history_payload(folder)["evaluations"]), 2)
            self.assertEqual(len(list(folder.glob("history.before-recovery.*.bak"))), 1)
            self.assertTrue((folder / "history.tmp").exists())


if __name__ == "__main__":
    unittest.main()
