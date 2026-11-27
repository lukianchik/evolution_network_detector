"""Проверяет реальный MCP-канал без обращений к внешней LLM."""

import json

from integrations.mcp_bridge import call_tools
from detectors.rule_engine import identity, seed_rule


def main() -> None:
    """Отправляет исходное правило через MCP-клиент и печатает метрики."""
    rule = identity(seed_rule("synthetic"))
    outcome = call_tools([
        ("evaluate_rule", {"rule_json": rule, "dataset": "synthetic", "split": "validation"}),
        ("compare_rules", {"first_json": rule, "second_json": rule, "dataset": "synthetic", "split": "validation"}),
    ])
    assert outcome[0]["metrics"]["fitness"] >= 0
    assert abs(outcome[1]["delta_f1"]) < 1e-9
    print(json.dumps({"status": "ok", "mcp_calls": 2,
                      "f1": outcome[0]["metrics"]["f1"],
                      "fpr": outcome[0]["metrics"]["fpr"],
                      "fitness": outcome[0]["metrics"]["fitness"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
