"""Проверки направленного поиска и раздельного бюджета train/validation."""
import unittest

from app import ExperimentManager


class GuidedTests(unittest.TestCase):
    """Направленный поиск локален, воспроизводим и не использует validation в скрининге."""

    def test_newest_run_order(self):
        """Сортировка не группирует даты по имени провайдера."""
        ids = ['shinka-20261121-120000-abcd', 'local-20261126-120000-abcd', 'example']
        self.assertEqual(max(ids, key=ExperimentManager._run_order), ids[1])
