"""Проверки короткой схемы эволюции в панели и в HTML-отчёте."""

import tempfile
import unittest
from pathlib import Path

from ui.visualize import create_report
import json

ROOT = Path(__file__).resolve().parents[1]


class RuleDiagramTests(unittest.TestCase):
    """Проверяет шаги, ветку MCP и корректность локального отчёта."""

    def test_panel_has_separate_runs_and_no_explanatory_blocks(self):
        """Текущий эксперимент отделён от истории; схема остаётся в отчёте."""
        text = (ROOT / "ui" / "panel.html").read_text(encoding="utf-8")
        self.assertIn('id="experimentView"', text)
        self.assertIn('id="runsView" hidden', text)
        self.assertNotIn('id="rule-flow-title"', text)
        self.assertIn('id="runComparison"', text)
        self.assertIn('id="timeline"', text)
        self.assertIn('id="profile"', text)
        self.assertNotIn('id="guided"', text)
        self.assertIn('id="bestFpr"', text)
        self.assertNotIn('id="runSelect"', text)

    def test_report_has_rule_specific_steps(self):
        """Отчёт локальной эволюции не называет LLM обязательной частью цикла."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "seed.py").write_text("def detect(event):\n    return 0\n", encoding="utf-8")
            history = {"mode": "local", "evaluations": [
                {"id": "p0", "generation": 0, "path": "seed.py", "fitness": 0.2,
                 "f1": 0.3, "fpr": 0.1}
            ], "novelty_rejected": 0}
            (folder / "history.json").write_text(json.dumps(history), encoding="utf-8")
            report = create_report(folder).read_text(encoding="utf-8")
            self.assertIn('Как эволюционирует правило', report)
            self.assertIn('Мутация', report)
            self.assertIn('Мини-схема архитектуры', report)
            self.assertIn('MCP: после поколения', report)
            self.assertNotIn('Проверка новизны', report)

    def test_report_avoids_claiming_code_mode_is_rule_mode(self):
        """Для прежнего ShinkaEvolve отображается цепочка LLM-кода, а не JSON-правил."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "seed.py").write_text("def detect(event):\n    return 0\n", encoding="utf-8")
            history = {"mode": "shinka", "evaluations": [
                {"id": "s0", "generation": 0, "path": "seed.py", "fitness": 0.2,
                 "f1": 0.3, "fpr": 0.1}
            ], "novelty_rejected": 0}
            (folder / "history.json").write_text(json.dumps(history), encoding="utf-8")
            report = create_report(folder).read_text(encoding="utf-8")
            self.assertIn('LLM-изменение', report)
            self.assertNotIn('MCP: после поколения', report)
