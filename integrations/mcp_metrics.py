"""Локальный MCP-сервер метрик самоэволюционирующего сетевого детектора.

Запуск: python -m integrations.mcp_metrics. Использует официальный MCP SDK v2 и stdio.
MCP-инструменты считают метрики только по train/validation, не по test.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from artifacts.history_store import candidate_path, read_history_payload, selected_row
from detectors.rule_engine import RuleEvaluator, complexity, normalize

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")
@lru_cache(maxsize=4)
def evaluator_for(dataset: str, penalty: float = 0.2, train_size: int | None = None,
                  validation_size: int = 4096, feature_set: str = "expanded") -> RuleEvaluator:
    """Повторно использует предварительно загруженные матрицы признаков."""
    if (train_size is not None and not 0 <= train_size <= 200000 or not 30 <= validation_size <= 50000
            or feature_set not in ("legacy", "expanded")):
        raise ValueError("Некорректная конфигурация выборки")
    return RuleEvaluator(dataset, penalty, train_size=train_size,
                         validation_size=validation_size, feature_set=feature_set)


def evaluate_rule(rule_json: str, dataset: str = "unsw", split: str = "validation", fpr_penalty: float = 0.2,
                  train_size: int | None = None, validation_size: int = 4096,
                  feature_set: str = "expanded") -> dict:
    """Возвращает F1, FPR и fitness для JSON-правила на выбранной части train."""
    if len(rule_json) > 10000:
        raise ValueError("Правило слишком длинное")
    if split not in ("train", "validation"):
        raise ValueError("Разрешены только train и validation")
    rule = normalize(json.loads(rule_json), dataset)
    evaluator = evaluator_for(dataset, fpr_penalty, train_size, validation_size, feature_set)
    return {"dataset": dataset, "split": split, "rule": rule,
            "metrics": evaluator.evaluate(rule, split), "complexity": complexity(rule),
            "sample_id": evaluator.sample["id"]}


def compare_rules(first_json: str, second_json: str, dataset: str = "unsw",
                  split: str = "validation", fpr_penalty: float = 0.2, train_size: int | None = None,
                  validation_size: int = 4096, feature_set: str = "expanded") -> dict:
    """Сравнивает два правила по одному и тому же фиксированному набору."""
    first = evaluate_rule(first_json, dataset, split, fpr_penalty, train_size, validation_size, feature_set)
    second = evaluate_rule(second_json, dataset, split, fpr_penalty, train_size, validation_size, feature_set)
    return {"first": first["metrics"], "second": second["metrics"],
            "sample_id": first["sample_id"], "complexity": [first["complexity"], second["complexity"]],
            "delta_fitness": round(second["metrics"]["fitness"] - first["metrics"]["fitness"], 6),
            "delta_f1": round(second["metrics"]["f1"] - first["metrics"]["f1"], 6),
            "delta_fpr": round(second["metrics"]["fpr"] - first["metrics"]["fpr"], 6)}


def _read_run(run: str) -> tuple[Path, dict]:
    """Читает только безопасно названную локальную папку с историей эксперимента."""
    if not RUN_NAME.fullmatch(run):
        raise ValueError("Некорректное имя эксперимента")
    folder = (ROOT / "results" / run).resolve()
    if not folder.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Выход за пределы папки экспериментов")
    data = read_history_payload(folder)
    if not isinstance(data.get("evaluations"), list):
        raise ValueError("Некорректная история")
    return folder, data


def get_best_rule(run: str) -> dict:
    """Возвращает лучший структурированный детектор из локального архива."""
    folder, data = _read_run(run)
    candidates = data["evaluations"]
    if not candidates:
        raise ValueError("Нет оценённых кандидатов")
    best = selected_row(candidates, data)
    rule_path = candidate_path(folder, best["path"]).with_suffix(".json").resolve()
    if not rule_path.is_relative_to(folder / "candidates"):
        raise ValueError("Некорректный путь в истории")
    return {"id": best["id"], "fitness": best["fitness"], "f1": best["f1"],
            "fpr": best["fpr"], "recall": best.get("recall"), "selection_score": best.get("selection_score"),
            "rule": json.loads(rule_path.read_text(encoding="utf-8"))}


def get_search_summary(run: str) -> dict:
    """Показывает прогресс без передачи исходников и меток сетевых событий."""
    _folder, data = _read_run(run)
    rows = data["evaluations"]
    best = selected_row(rows, data) if rows else None
    return {"run": run, "mode": data.get("mode"), "dataset": data.get("dataset"),
            "evaluated": len(rows), "rejected": data.get("novelty_rejected", 0),
            "best_f1": best["f1"] if best else None,
            "best_fitness": best["fitness"] if best else None,
            "winner_id": best["id"] if best else None, "pareto_ids": data.get("pareto_ids", []),
            "statistics": data.get("statistics", {})}


def create_server():
    """Регистрирует функции как полноценные MCP tools через официальный SDK v2."""
    try:
        from mcp.server import MCPServer
    except ImportError as error:
        raise RuntimeError("Установи MCP SDK: python -m pip install -r requirements-mcp.txt") from error
    server = MCPServer("network-evolution-metrics")
    server.tool()(evaluate_rule)
    server.tool()(compare_rules)
    server.tool()(get_best_rule)
    server.tool()(get_search_summary)
    return server


def main() -> None:
    """Запускает локальный stdio или закрытый HTTP-сервис Compose."""
    parser = argparse.ArgumentParser(description="MCP metrics service")
    parser.add_argument("--http", action="store_true", help="Внутренний HTTP-транспорт Compose")
    args = parser.parse_args()
    server = create_server()
    if args.http:
        from mcp.server.transport_security import TransportSecuritySettings
        server.run("streamable-http", host="0.0.0.0", port=8090,
                   streamable_http_path="/mcp", json_response=True, stateless_http=True,
                   max_request_body_size=65536,
                   transport_security=TransportSecuritySettings(
                       allowed_hosts=["metrics:8090", "localhost:8090", "127.0.0.1:8090"],
                       allowed_origins=[]))
    else:
        server.run()


if __name__ == "__main__":
    main()
