"""Проверки генерации данных, верификатора, мутаций без сети."""

import json
import tempfile
import unittest
from pathlib import Path

from traffic_data.dataset import build_dataset
from evaluation.evaluate import evaluate_program, write_result
from search.local_evolution import evolve_local
from detectors.rule_engine import RuleEvaluator, seed_rule, to_python
from evaluation.final_test import best_from_history, test_program as evaluate_test_program
from detectors.metrics import measure
from detectors.novelty import embed, is_novel, similarity
from detectors.safe_program import load_detector, predict
from ui.visualize import create_report

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    """Проверяет основные функции проекта на небольших данных."""

    def test_determinism_and_split(self):
        """Наборы событий воспроизводимы по seed и отличаются между выборками."""
        self.assertEqual(build_dataset(3, 30), build_dataset(3, 30))
        self.assertNotEqual(build_dataset(3, 30), build_dataset(4, 30))

    def test_metrics(self):
        """F1 и FPR рассчитываются по известной матрице ошибок."""
        values = measure([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(values["f1"], 0.5)
        self.assertAlmostEqual(values["fpr"], 0.5)
        self.assertEqual((values["tp"], values["fp"]), (1, 1))

    def test_seed_evaluates(self):
        """Исходная программа проходит проверку и даёт реальный результат."""
        result = evaluate_program(ROOT / "seeds" / "initial.py")
        self.assertGreater(result["f1"], 0.1)
        self.assertLess(result["f1"], 1)

    def test_reject_bad_code(self):
        """Генератор не может подсунуть импорты, файлы или чтение меток."""
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "bad.py"
            for code in (
                'def detect(event):\n    return __import__("os").system("echo x")',
                'def detect(event):\n    return int(event["label"])',
                'def detect(event):\n    while True: pass',
                'def detect(event):\n    return open("secret").read()',
                'def detect(event):\n    event["unique_ports"] = 100\n    return 1',
            ):
                file.write_text(code, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_detector(file)

    def test_mutation_and_crossover(self):
        """Мутации сохраняют допустимый Python и не портят родителя."""
        import random
        rng = random.Random(1)
        evaluator = RuleEvaluator("synthetic")
        parent = seed_rule("synthetic")
        child = evaluator.mutate(parent, rng)
        mix = evaluator.crossover(parent, child, rng)
        self.assertEqual(parent, seed_rule("synthetic"))
        with tempfile.TemporaryDirectory() as d:
            for item in (child, mix):
                path = Path(d) / "candidate.py"
                path.write_text(to_python(item, "synthetic"), encoding="utf-8")
                self.assertEqual(len(predict(path, build_dataset(13, 5)[0])), 5)

    def test_novelty(self):
        """Идентичный код отклоняется, новый не считается точной копией."""
        base = 'def detect(event): return 0'
        self.assertAlmostEqual(similarity(embed(base), embed(base)), 1, places=5)
        self.assertFalse(is_novel(base, [embed(base)]))
        self.assertTrue(is_novel('def detect(event): return int(event["outbound_mb"] > 10)', [embed(base)]))

    def test_offline_history_and_html(self):
        """Полный автономный цикл сохраняет версии и создаёт HTML-отчёт."""
        with tempfile.TemporaryDirectory() as d:
            folder = evolve_local("local", "synthetic", 3, Path(d), per_generation=3, mcp_check=False)
            report = create_report(folder)
            history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            self.assertTrue(report.is_file())
            self.assertGreater(len(history["evaluations"]), 2)
            self.assertIn("<svg", report.read_text(encoding="utf-8"))
            self.assertTrue(best_from_history(folder).exists())
            self.assertGreaterEqual(evaluate_test_program(best_from_history(folder))["f1"], 0)

    def test_shinka_result_contract(self):
        """Проверяет формат файлов, требуемых ShinkaEvolve."""
        with tempfile.TemporaryDirectory() as d:
            values = evaluate_program(ROOT / "seeds" / "initial.py")
            write_result(Path(d), values)
            m = json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))
            c = json.loads((Path(d) / "correct.json").read_text(encoding="utf-8"))
            self.assertIsInstance(m["combined_score"], float)
            self.assertIs(c["correct"], True)
