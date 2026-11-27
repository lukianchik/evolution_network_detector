"""Регрессии контейнерного транспорта, миграции и разнообразия train."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from detectors.metrics import measure
from detectors.rule_engine import RuleEvaluator, seed_rule
from integrations import mcp_bridge
from tools.migrate_docker import copy_missing


class ContainerTransportTests(unittest.TestCase):
    """Проверяет совместимость без настоящего Docker и внешних API."""

    def test_http_transport(self):
        """URL передаётся официальному Client вместо запуска stdio-процесса."""
        targets = []

        class Client:
            """Имитирует контракт SDK, не имитирует успешный сетевой тест."""
            def __init__(self, target):
                targets.append(target)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def call_tool(self, name, args):
                return SimpleNamespace(is_error=False, structured_content={"tool": name, "args": args})

        def no_stdio(**_kwargs):
            """В HTTP-режиме создание дочернего процесса запрещено."""
            raise AssertionError('stdio')

        url = 'http://metrics:8090/mcp'
        with patch.dict(os.environ, {'EVO_MCP_URL': url}), patch.dict(sys.modules, {
                'mcp': SimpleNamespace(Client=Client, StdioServerParameters=no_stdio)}):
            result = mcp_bridge.call_tools([('evaluate_rule', {})])
        self.assertEqual(targets, [url])
        self.assertEqual(result[0]['tool'], 'evaluate_rule')

    def test_migration_idempotent(self):
        """Существующие запуски и база квот не перезаписываются."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dst = root / 'src', root / 'dst'
            (src / 'run').mkdir(parents=True)
            (src / 'run' / 'history.json').write_text('first')
            self.assertEqual(copy_missing(src, dst), 1)
            (src / 'run' / 'history.json').write_text('second')
            self.assertEqual(copy_missing(src, dst), 0)
            self.assertEqual((dst / 'run' / 'history.json').read_text(), 'first')
            (src / 'quota.db').write_text('state')
            self.assertEqual(copy_missing(src, dst, whole=True), 0)
            self.assertFalse((dst / 'quota.db').exists())

    def test_vector_metrics_equivalent(self):
        """Векторный подсчёт сохраняет точные исходные метрики."""
        ev = RuleEvaluator('synthetic', .5)
        rule = seed_rule('synthetic')
        y = ev.validation[1].tolist()
        p = ev._predictions(rule, 'validation').astype(int).tolist()
        self.assertEqual(ev.evaluate(rule), measure(y, p, .5))
