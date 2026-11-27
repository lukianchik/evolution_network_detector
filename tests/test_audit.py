"""Regression coverage for artifact boundaries and metric inputs."""

import tempfile
import unittest
from pathlib import Path


from evaluation.classify import classify
from artifacts.history_store import candidate_path
from detectors.metrics import measure
from detectors.safe_program import validate_detector


class ArtifactTests(unittest.TestCase):
    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            measure([2], [1])

    def test_candidate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Windows может возвращать краткий путь TEMP (8.3).
            expected = (root / "candidates" / "seed.py").resolve()
            self.assertEqual(candidate_path(root, r"candidates\seed.py"), expected)
            self.assertEqual(candidate_path(root, "candidates/seed.py"), expected)
            for value in ("../x.py", ".env"):
                with self.assertRaises(ValueError):
                    candidate_path(root, value)

    def test_augmented_power_rejected(self):
        with self.assertRaises(ValueError):
            validate_detector("def detect(event):\n    x = 1\n    x **= 2\n    return x")

    def test_incomplete_classification_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, result, detector = root / "events", root / "output", root / "candidate.py"
            detector.write_text("def detect(event): return 0", encoding="utf-8")
            source.write_text("{}\n" * 5001, encoding="utf-8")
            result.write_text("previous", encoding="utf-8")
            with self.assertRaises(ValueError):
                classify(source, detector, result)
            self.assertEqual(result.read_text(), "previous")
            with self.assertRaises(ValueError):
                classify(source, detector, source)
