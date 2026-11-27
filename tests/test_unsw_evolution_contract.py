"""Регрессии разделения UNSW и единого движка."""

import ast
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from artifacts.history_store import read_history_payload, selected_row
from detectors.rule_engine import RuleEvaluator, seed_rule, to_python, normalize
from detectors.safe_program import validate_detector, interpret
from integrations.mcp_metrics import compare_rules
from search.local_evolution import evolve_local
from search.strategies import Archive, Objective, TrainSearch, dominates, vector
from traffic_data import dataset_real, sampling

ROOT = Path(__file__).resolve().parents[1]


class UnsWTrainContractTests(unittest.TestCase):
    """Проверяет группы, бюджет поиска и равенство разных представлений правила."""

    def test_grouped_partition_train_only(self):
        """Точные дубликаты остаются вместе, резерв и validation стабильны."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.csv"
            columns = (*dataset_real.FEATURES, "proto", "state", "service", "id", "label", "attack_cat")
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for i in range(400):
                    values = {feature: (i // 2) for feature in dataset_real.FEATURES}
                    label = (i // 2) % 2
                    writer.writerow({**values, "id": i, "proto": "tcp" if label else "udp",
                                     "service": "-", "state": "FIN", "label": label,
                                     "attack_cat": "Generic" if label else "Normal"})
            with patch.object(dataset_real, "TRAIN", path), patch.object(dataset_real, "TEST", path.with_name("absent")):
                evaluator = RuleEvaluator("unsw", train_size=100, validation_size=60)
                larger = RuleEvaluator("unsw", train_size=0, validation_size=60)
                self.assertFalse({"id", "attack_cat", "label"} & set(evaluator.features))
                self.assertTrue(np.array_equal(evaluator.validation[0], larger.validation[0]))
                self.assertNotEqual(evaluator.sample["id"], larger.sample["id"])
                self.assertEqual(evaluator.sample["reserved_size"], 48)
                x, y, _, _, parts = sampling.partition(path, path.stat().st_mtime_ns, 0, 60)
                memberships = [set(x[part, 0]) for part in parts]
                self.assertEqual(sum(map(len, memberships)), len(set.union(*memberships)))
                self.assertTrue(all(set(y[part]) == {0, 1} for part in parts))
                one = seed_rule("unsw")
                raw = json.dumps(one)
                remote = compare_rules(raw, raw, "unsw", train_size=100, validation_size=60)
                self.assertEqual(remote["sample_id"], evaluator.sample["id"])
                self.assertEqual(remote["second"], evaluator.evaluate(one))

    def test_nested_rule_exports_equivalent_predictions(self):
        """Группы, вес и интервал одинаково работают в NumPy и AST."""
        evaluator = RuleEvaluator("synthetic")
        rule = {"decision_threshold": 2, "rules": [
            {"operator": "or", "weight": 2, "conditions": [
                {"operator": "not", "conditions": [
                    {"feature": "failed_logins", "operator": "between", "threshold": 2, "upper": 10}]},
                {"operator": "and", "conditions": [
                    {"feature": "outbound_mb", "operator": ">=", "threshold": 9},
                    {"feature": "unique_ports", "operator": "<=", "threshold": 24}]}]},
        ]}
        func = validate_detector(to_python(rule, "synthetic"))
        features = frozenset(evaluator.features)
        predicted = [interpret(func, dict(zip(evaluator.features, map(float, row))), features)
                     for row in evaluator.validation[0]]
        self.assertTrue(np.array_equal(predicted, evaluator._predictions(rule, "validation")))
        rule["rules"][0]["conditions"][0]["conditions"][0]["feature"] = "label"
        with self.assertRaises(ValueError):
            normalize(rule, "synthetic")

    def test_pareto_retains_distinct_tradeoffs(self):
        """Сохраняет край с низким FPR и удаляет доминируемые правила."""
        archive = Archive(Objective("balanced", None, None, False), size=3)
        scores = [(0.91, 0.19, 0.94), (0.78, 0.03, 0.68), (0.64, 0.4, 0.57),
                  (0.88, 0.13, 0.89), (0.87, 0.11, 0.86)]
        for index, (f1, fpr, recall) in enumerate(scores):
            archive.add({"id": index, "f1": f1, "fpr": fpr, "recall": recall,
                         "complexity": 2, "instability": 0.01, "generalization_gap": 0.01})
        ids = {candidate["id"] for candidate in archive.rows}
        self.assertIn(1, ids)
        self.assertNotIn(2, ids)
        self.assertEqual(len(ids), 3)
        self.assertFalse(any(dominates(vector(a), vector(b)) for a in archive.rows for b in archive.rows))

    def test_train_strategies_run_without_validation(self):
        """Статистика ошибок и подсказки вычисляются только на train."""
        evaluator = RuleEvaluator("synthetic")
        evaluator.validation = None
        search = TrainSearch(evaluator, Objective("balanced", None, None, False), random.Random(43))
        self.assertGreater(search.initialize(), 10)
        self.assertEqual(search.initialize(), 0)
        seed = seed_rule("synthetic")
        guided, _ = search.candidates("guided", seed, None, 0)
        directed, checks = search.candidates("error", seed, None, 0)
        self.assertGreater(checks, 0)
        self.assertTrue(any("conditions" in rule["rules"][0] for rule in directed))
        self.assertGreater(max(evaluator.evaluate(rule, "train")["f1"] for rule in guided),
                           evaluator.evaluate(seed, "train")["f1"])

    def test_reproducible_search_and_capped_validation(self):
        """Одинаковый seed и нулевой LLM-бюджет сохраняют путь поиска."""
        with tempfile.TemporaryDirectory() as temp:
            history = []
            for mode in ("local", "hybrid"):
                folder = Path(temp) / mode
                evolve_local(mode, "synthetic", 6, folder, seed=43, per_generation=12, mcp_check=False)
                result = read_history_payload(folder)
                history.append(result["evaluations"])
                stats = result["statistics"]
                self.assertEqual(stats["llm_calls"], 0)
                self.assertGreater(stats["train_evaluations"], 100)
                self.assertLessEqual(stats["validation_evaluations"], 25)
                self.assertGreater(stats["strategy_attempts"]["error"], 0)
                self.assertEqual(selected_row(history[-1], result)["id"], result["winner_id"])
            self.assertEqual(*history)

    def test_shinka_asks_for_train(self):
        """ShinkaEvolve эволюционирует с официальным train."""
        tree = ast.parse((ROOT / "search" / "run_shinka.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "check_files"]
        self.assertEqual([node.args[0].value for node in calls], ["train"])
