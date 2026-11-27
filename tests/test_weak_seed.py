"""Тесты слабого baseline и безопасных пояснений, созданных LLM."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from detectors.safe_program import load_detector, validate_detector

ROOT = Path(__file__).resolve().parents[1]


class WeakSeedTests(unittest.TestCase):
    """Проверяет исходный детектор и приём корректных мутаций."""

    def test_weak_rule_and_strong_reference(self):
        """Текущий Seed содержит одно правило, а прежний сохранён отдельно."""
        with patch.dict(os.environ, {"EVO_DATASET": "unsw"}):
            weak = load_detector(ROOT / "seeds" / "initial_unsw.py")
            strong = load_detector(ROOT / "seeds" / "initial_unsw_strong.py")
        event = {"ct_dst_src_ltm": 1, "sttl": 200}
        self.assertEqual(weak(event), 0)
        self.assertEqual(strong(event), 1)
        event = {"ct_dst_src_ltm": 22, "sttl": 1}
        self.assertEqual(weak(event), 1)
        self.assertEqual(strong(event), 1)

    def test_long_docstring_is_not_an_executable_constant(self):
        """Безвредный длинный докстринг не отклоняет полезный код модели."""
        source = ('def detect(event):\n'
                  '    """' + 'Описание обнаружения сетевых событий. ' * 12 + '"""\n'
                  '    return int(event["sttl"] >= 120)\n')
        with patch.dict(os.environ, {"EVO_DATASET": "unsw"}):
            self.assertIsNotNone(validate_detector(source))

    def test_long_executable_string_still_forbidden(self):
        """Обычные большие строковые константы и чтение меток запрещены."""
        with self.assertRaisesRegex(ValueError, "Слишком длинная константа"):
            validate_detector('def detect(event):\n    rule = ' + repr('X' * 150) + '\n    return 0\n')
        with self.assertRaises(ValueError):
            validate_detector('def detect(event):\n    return int(event["label"])\n')


if __name__ == "__main__":
    unittest.main()
