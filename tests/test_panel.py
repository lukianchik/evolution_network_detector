"""Тесты локальной панели, диффов кода и контролируемого запуска."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from app import ExperimentManager, PanelHandler
from ui.visualize import best_timeline, code_diff, create_report, load_history

ROOT = Path(__file__).resolve().parents[1]


class PanelTests(unittest.TestCase):
    """Проверяет чтение результатов и интерфейс локального HTTP-сервера."""

    @classmethod
    def setUpClass(cls):
        from search.local_evolution import evolve_local
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.results = Path(directory.name)
        cls.enterClassContext(patch("app.RESULTS", cls.results))
        evolve_local("local", "synthetic", 2, cls.results / "demo", per_generation=6, mcp_check=False)

    def test_history_and_diff(self):
        """Дифф показывает новые строки и не исполняет найденный Python-код."""
        manager = ExperimentManager()
        _mode, rows, _ = load_history(self.results / "demo")
        self.assertGreater(len(rows), 1)
        self.assertIn("example", manager.runs)
        chosen = next(row for row in rows if row.get("parent"))
        comparison = manager.code("example", chosen["id"], "parent")
        self.assertIn("def detect", comparison["after"])
        self.assertIn(chosen["parent"], comparison["base"])
        self.assertTrue(any(line.startswith(("+", "-")) for line in comparison["diff"]))
        self.assertIn("delta-plus", code_diff("one\n", "two\n", "a", "b"))
        self.assertIn("Эволюция лидера", best_timeline(rows))

    def test_report_contains_code_evolution(self):
        """Статический HTML показывает улучшения и версии без внешних зависимостей."""
        with tempfile.TemporaryDirectory() as directory:
            from search.local_evolution import evolve_local
            folder = evolve_local("local", "synthetic", 2, Path(directory), per_generation=6, mcp_check=False)
            content = create_report(folder).read_text(encoding="utf-8")
        self.assertIn("Изменения программы", content)
        self.assertIn("delta-plus", content)
        self.assertIn("Fitness по оценкам", content)
        self.assertIn("Эволюция лидера", content)

    def test_local_api_and_validation(self):
        """HTTP-панель отдаёт измерения и защищает команды от посторонних страниц."""
        manager = ExperimentManager()

        class Handler(PanelHandler):
            """Обработчик для временного тестового сервера."""
            pass

        Handler.manager = manager
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/api/config") as response:
                settings = json.load(response)
            self.assertEqual(set(settings["modes"]), {"local", "hybrid"})
            with urllib.request.urlopen(base + "/api/state?run=example") as response:
                status = json.load(response)
            self.assertGreater(len(status["evaluations"]), 2)
            candidate = status["evaluations"][1]["id"]
            with urllib.request.urlopen(base + f"/api/code?run=example&id={candidate}") as response:
                code = json.load(response)
            self.assertIn("diff", code)
            self.assertIn("def detect", code["after"])
            command = {"token": settings["token"], "mode": "retired-mode", "generations": 2}
            request = urllib.request.Request(base + "/api/start", data=json.dumps(command).encode(),
                                             headers={"Content-Type":"application/json"}, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(request)
            self.assertEqual(refused.exception.code, 403)
            request.add_header("Origin", base)
            with self.assertRaises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(request)
            self.assertEqual(refused.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
