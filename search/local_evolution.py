"""Быстрый локальный и гибридный эволюционный поиск сетевых правил.

В обычном режиме API не вызывается. В гибридном LLM предлагает только
компактный JSON, причём лишь после серии локальных попыток без улучшения.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

from search.evolution import ROOT, load_env
from search.strategies import (PROFILES, Archive, Coordinator, Objective,
                               TrainSearch, choose_promotions)
from artifacts.history_store import save_history_payload
from integrations.mcp_bridge import call_tools
from integrations.provider_api import request_completion
from detectors.rule_engine import RuleEvaluator, complexity, identity, measure_predictions, normalize, seed_rule, to_python
from ui.visualize import create_report

MODES = ("local", "hybrid")
BUDGETS = {"quick": (5, 10, 10000, 30), "standard": (12, 15, 20000, 120),
           "deep": (24, 24, 50000, 300)}


def proposal_model(provider: str) -> str:
    """Возвращает настроенную модель Groq."""
    if provider != "groq":
        raise ValueError("Поддерживается Groq")
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def propose_rule(provider: str, evaluator: RuleEvaluator, rule: dict, train_score: dict,
                 *, mcp_summary: dict | None = None) -> dict:
    """Передаёт LLM только краткие признаки, правило и агрегированные метрики."""
    model = proposal_model(provider)
    task = {"task": "Suggest ONE structurally different JSON rule set for network anomaly detection.",
            "features": list(evaluator.features),
            "schema": {"rules": [{"feature": "known name", "operator": ">= or <=",
                                  "threshold": "finite number", "weight": "integer 1..3"}],
                       "decision_threshold": "integer 1..sum(weights)"},
            "constraints": "Max 6 rules. No Python, no labels, no prose, no markdown.",
            "objective": "Raise precision and recall while reducing false positives and rule complexity.",
            "current_rule": rule,
            "train_metrics": train_score}
    if mcp_summary:
        # В LLM уходят только агрегированные числа; события, разметка и исходники — локально.
        task["mcp_feedback"] = mcp_summary
    answer = request_completion(provider, model,
                                [{"role": "user", "content": json.dumps(task, ensure_ascii=False, separators=(",", ":"))}],
                                max_tokens=480, temperature=0.5, max_attempts=1, max_wait=20)
    try:
        content = answer["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Нужен строковый JSON-ответ")
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.I | re.S)
        rule = json.loads((match.group(1) if match else content).strip())
        return normalize(rule, evaluator.dataset)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Модель не вернула допустимое JSON-правило: {error}") from error


def validation_folds(evaluator: RuleEvaluator) -> list[np.ndarray]:
    """Стратифицирует внутреннюю проверку по классу и категории."""
    y = evaluator.validation[1]
    categories = evaluator.validation_categories
    strata = y.astype(str) if categories is None else np.char.add(y.astype(str), categories)
    assignment = np.empty(len(y), dtype=np.uint8)
    rng = np.random.default_rng(377)
    for label in np.unique(strata):
        rows = rng.permutation(np.flatnonzero(strata == label))
        assignment[rows] = np.arange(len(rows)) % 3
    return [assignment == index for index in range(3) if np.any(assignment == index)]


def peak_rss_mib() -> float:
    """Читает пиковую резидентную память текущего процесса."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        ("sizes", ctypes.c_size_t * 8)]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        getter = ctypes.windll.psapi.GetProcessMemoryInfo
        getter.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
        if not getter(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        value = counters.sizes[0]
    else:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return round(value / 1024**2, 2)


def evolve_local(mode: str, dataset: str, generations: int, folder: Path, *, seed: int = 42,
                 per_generation: int = 15, stagnation_limit: int = 20, max_llm_calls: int = 0,
                 mcp_check: bool = True, fpr_penalty: float = 0.2, train_size: int = 20000,
                 validation_size: int = 4096, profile: str = "balanced", max_fpr: float | None = None,
                 min_recall: float | None = None, time_limit: float = 120,
                 feature_set: str = "expanded", ablations: tuple[str, ...] = (), stop=lambda: False) -> Path:
    """Выполняет одну эволюцию с ограниченным числом обращений к validation."""
    if mode not in MODES or not 1 <= generations <= 30 or not 1 <= per_generation <= 30:
        raise ValueError("Некорректный режим или размеры эксперимента")
    if (not 1 <= stagnation_limit <= 500 or not 0 <= max_llm_calls <= 5 or not 0 <= seed < 2**32
            or not 0 <= train_size <= 200000 or not 30 <= validation_size <= 50000
            or not 1 <= time_limit <= 3600 or profile not in PROFILES
            or feature_set not in ("legacy", "expanded")
            or any(value is not None and not 0 <= value <= 1 for value in (max_fpr, min_recall))
            or set(ablations) - {"error", "adaptive", "pareto", "cache"}):
        raise ValueError("Недопустимые параметры поиска")
    started = time.monotonic()
    evaluator = RuleEvaluator(dataset, fpr_penalty, train_size=train_size, validation_size=validation_size,
                              feature_set=feature_set, cache="cache" not in ablations)
    objective = Objective(profile, max_fpr, min_recall, "pareto" in ablations)
    archive = Archive(objective)
    coordinator = Coordinator(ablations)
    rng = random.Random(seed)
    search = TrainSearch(evaluator, objective, rng)
    folds = validation_folds(evaluator)
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "candidates").mkdir()
    (folder / "dataset.txt").write_text(dataset, encoding="utf-8")
    stats = {
        "engine": "v1.0.0", "seed": seed, "dataset": dataset, "sample": evaluator.sample,
        "profile": profile, "max_fpr": objective.max_fpr, "min_recall": objective.min_recall,
        "numeric_threads": {key: os.getenv(key) for key in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "time_limit": time_limit, "ablations": list(ablations), "generations": generations,
        "offspring": per_generation, "fpr_penalty": fpr_penalty, "stagnation_limit": stagnation_limit,
        "local_attempts": 0, "local_evaluations": 0, "train_evaluations": 0,
        "guided_train_evaluations": 0, "error_stat_checks": 0, "validation_evaluations": 0,
        "validation_folds": len(folds), "train_behaviors": 0, "semantic_rejected": 0,
        "llm_calls": 0, "llm_errors": 0, "llm_accepted": 0, "llm_improvements": 0,
        "max_llm_calls": max_llm_calls if mode == "hybrid" else 0,
        "mcp_enabled": mcp_check, "mcp_calls": 0, "mcp_verified": 0, "mcp_errors": 0,
        "mcp_last_status": "не проверялся", "search_status": "running",
        "preparation_seconds": round(time.monotonic() - started, 3),
    }
    history, mcp_events, rules = [], [], {}
    seen, seen_behaviors = set(), {}
    rejected = 0
    last_llm = 0
    mcp_disabled = False

    def save():
        """Сохраняет атомарный снимок поколения и зафиксированного лидера."""
        stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
        stats["strategy_attempts"] = dict(coordinator.attempts)
        stats["strategy_improvements"] = dict(coordinator.gains)
        stats["strategy_train_checks"] = dict(coordinator.costs)
        stats["train_behaviors"] = len(seen_behaviors)
        save_history_payload(folder, {"mode": mode, "dataset": dataset, "evaluations": history,
                                      "winner_id": archive.best()["id"], "pareto_ids": [r["id"] for r in archive.rows],
                                      "novelty_rejected": rejected, "statistics": stats, "mcp_events": mcp_events})

    def screen(rule, kind, parent=None, second=None):
        """Проверяет уникальность и метрики только на train."""
        nonlocal rejected
        try:
            rule = normalize(rule, dataset)
            key = identity(rule)
            if key in seen:
                rejected += 1
                return None
            seen.add(key)
            prediction = evaluator.scores(rule, "train") >= rule["decision_threshold"]
        except ValueError:
            rejected += 1
            return None
        stats["train_evaluations"] += 1
        signature = hashlib.blake2b(np.packbits(prediction), digest_size=16).digest()
        size = complexity(rule)
        if seen_behaviors.get(signature, size + 1) <= size:
            stats["semantic_rejected"] += 1
            rejected += 1
            return None
        seen_behaviors[signature] = size
        train = measure_predictions(prediction, evaluator.train[1], fpr_penalty)
        return {"key": key, "rule": rule, "train": train, "kind": kind, "complexity": size,
                "parent": parent, "second_parent": second, "utility": objective.utility(train, size)}

    def promote(item, generation):
        """Оценивает перспективное правило на validation и сохраняет его код."""
        rule = item["rule"]
        train = item["train"]
        prediction = evaluator.scores(rule, "validation") >= rule["decision_threshold"]
        y = evaluator.validation[1]
        score = measure_predictions(prediction, y, fpr_penalty)
        checks = [measure_predictions(prediction[mask], y[mask], fpr_penalty) for mask in folds]
        size = item["complexity"]
        code = f"r{len(history)}"
        path = Path("candidates") / f"g{generation:03d}_{code}.py"
        (folder / path).write_text(to_python(rule, dataset), encoding="utf-8")
        (folder / path.with_suffix(".json")).write_text(item["key"], encoding="utf-8")
        row = {"id": code, "generation": generation, "kind": item["kind"], "parent": item["parent"],
               "second_parent": item["second_parent"], "path": path.as_posix(), "complexity": size,
               "model": proposal_model("groq") if item["kind"] == "LLM" else None,
               "train_f1": train["f1"], "train_fpr": train["fpr"], "train_recall": train["recall"],
               "instability": round(float(np.std([objective.utility(check, size) for check in checks])), 6),
               "generalization_gap": round(max(0.0, train["f1"] - score["f1"], score["fpr"] - train["fpr"],
                                               train["recall"] - score["recall"]), 6), **score}
        row["selection_score"] = round(objective.rank(row), 8)
        history.append(row)
        rules[code] = rule
        archive.add(row)
        stats["validation_evaluations"] += 1
        print(f"  {code} · {item['kind']}: F1={row['f1']:.3f} FPR={row['fpr']:.3f}", flush=True)
        return row

    def checkpoint(generation, former):
        """Сверяет метрики лидера через MCP после локального отбора."""
        nonlocal mcp_disabled
        best = archive.best()
        args = {"first_json": identity(rules[former["id"]]), "second_json": identity(rules[best["id"]]),
                "dataset": dataset, "split": "validation", "fpr_penalty": fpr_penalty}
        if dataset == "unsw":
            args.update(train_size=train_size, validation_size=validation_size, feature_set=feature_set)
        stats["mcp_calls"] += 1
        event = {"kind": "checkpoint", "generation": generation, "tool": "compare_rules", "best": best["id"]}
        try:
            result = call_tools([("compare_rules", args)])[0]
            for expected, actual in ((former, result["first"]), (best, result["second"])):
                if any(expected[key] != actual[key] for key in
                       ("tp", "fn", "fp", "tn", "f1", "precision", "recall", "fpr", "fitness")):
                    raise ValueError("Различаются локальные метрики и метрики MCP")
            if dataset == "unsw" and result.get("sample_id") != evaluator.sample["id"]:
                raise ValueError("Идентификаторы выборок различаются")
            stats["mcp_verified"] += 1
            event.update(ok=True, delta_fitness=result["delta_fitness"])
        except Exception as error:
            stats["mcp_errors"] += 1
            mcp_disabled = stats["mcp_errors"] >= 2
            event.update(ok=False, error=str(error)[:150])
            print(f"  MCP: {event['error']}", flush=True)
        stats["mcp_last_status"] = "проверка пройдена" if event["ok"] else "ошибка проверки"
        mcp_events.append(event)

    def ask_groq(generation, before):
        """Передаёт агрегированные ошибки и полезность локальных стратегий."""
        stats["llm_calls"] += 1
        best = archive.best()
        rule = rules[best["id"]]
        train = evaluator.evaluate(rule, "train")
        feedback = {"objective": {"max_fpr": objective.max_fpr, "min_recall": objective.min_recall},
                    "train_errors": {key: train[key] for key in ("fp", "fn", "tp", "tn")},
                    "strategy_attempts": dict(coordinator.attempts), "strategy_gains": dict(coordinator.gains)}
        try:
            proposal = propose_rule("groq", evaluator, rule,
                                    {key: train[key] for key in ("f1", "fpr", "recall")}, mcp_summary=feedback)
            candidate = screen(proposal, "LLM", best["id"])
            if candidate is not None:
                promote(candidate, generation)
                stats["llm_accepted"] += 1
                stats["llm_improvements"] += objective.rank(archive.best()) > before + 1e-4
        except (ValueError, RuntimeError, OSError) as error:
            stats["llm_errors"] += 1
            print(f"  Groq: {type(error).__name__}. Локальный поиск продолжается.", flush=True)

    def stop_reason():
        """Проверяет отмену и бюджет времени перед следующей порцией работы."""
        if stop():
            return "stopped"
        if time.monotonic() - started >= time_limit:
            return "time-limit"
        return None

    promote(screen(seed_rule(dataset), "seed"), 0)
    save()
    stalled = 0
    for generation in range(1, generations + 1):
        reason = stop_reason()
        if reason:
            break
        former = archive.best()
        old_rank = objective.rank(former)
        batch = []
        for attempt in range(per_generation):
            if stop_reason():
                break
            strategy = coordinator.choose(per_generation - attempt, stalled >= stagnation_limit,
                                          len(archive.rows), adaptive="adaptive" not in ablations)
            parent = archive.parent(rng, strategy)
            second = rng.choice([row for row in archive.rows if row is not parent]) if strategy == "crossover" else None
            stats["local_attempts"] += 1
            coordinator.attempts[strategy] += 1
            coordinator.last_used[strategy] = stats["local_attempts"]
            extra = search.initialize() if strategy in ("guided", "error") else 0
            stats["guided_train_evaluations"] += extra
            try:
                proposals, error_checks = search.candidates(strategy, rules[parent["id"]],
                                                            rules[second["id"]] if second else None,
                                                            coordinator.attempts[strategy] - 1)
            except ValueError:
                continue
            stats["error_stat_checks"] += error_checks
            before = stats["train_evaluations"]
            entries = [entry for proposal in proposals
                       if (entry := screen(proposal, strategy, parent["id"], second["id"] if second else None))]
            coordinator.costs[strategy] += stats["train_evaluations"] - before + error_checks + extra
            batch.extend(choose_promotions(entries, 3))
            stats["local_evaluations"] += bool(entries)
        for item in choose_promotions(batch, min(4, per_generation)):
            before = objective.rank(archive.best())
            row = promote(item, generation)
            if row in archive.rows:
                coordinator.accepted[item["kind"]] += 1
            coordinator.gains[item["kind"]] += max(0.0, objective.rank(archive.best()) - before)
        if objective.rank(archive.best()) > old_rank + 1e-4:
            stalled = 0
        else:
            stalled += per_generation
        now = stats["local_attempts"]
        if (not stop_reason() and stats["llm_calls"] < stats["max_llm_calls"]
                and stalled >= stagnation_limit and now - last_llm >= stagnation_limit
                and all(coordinator.attempts[s] for s in coordinator.strategies if s != "crossover")):
            last_llm = now
            ask_groq(generation, objective.rank(archive.best()))
            if objective.rank(archive.best()) > old_rank + 1e-4:
                stalled = 0
        save()
        if mcp_check and not mcp_disabled and not stop_reason():
            checkpoint(generation, former)
            save()
        print(f"Поколение {generation}: лидер {archive.best()['id']} · ожидание улучшения {stalled}", flush=True)
        if stalled >= max(5 * per_generation, 2 * stagnation_limit):
            break
    stats["search_status"] = stop_reason() or ("early-stop" if stalled >= max(5 * per_generation, 2 * stagnation_limit)
                                              else "completed")
    if evaluator.validation_categories is not None:
        winner = rules[archive.best()["id"]]
        scores = evaluator.scores(winner, "validation") >= winner["decision_threshold"]
        categories = evaluator.validation_categories
        labels = evaluator.validation[1]
        stats["winner_categories"] = {}
        for category in np.unique(categories):
            mask = categories == category
            stats["winner_categories"][str(category)] = measure_predictions(scores[mask], labels[mask], fpr_penalty)
    try:
        stats["peak_rss_mib"] = peak_rss_mib()
    except OSError:
        stats["peak_rss_mib"] = None
    save()
    return folder


def main() -> int:
    """Запускает движок и записывает HTML-отчёт."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="local")
    parser.add_argument("--dataset", choices=("synthetic", "unsw"), default=os.getenv("EVO_DATASET", "unsw"))
    parser.add_argument("--budget", choices=BUDGETS, default="standard")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--offspring", type=int)
    parser.add_argument("--train-size", type=int, help="0 — весь доступный train после отделения проверок")
    parser.add_argument("--validation-size", type=int, default=4096)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--profile", choices=PROFILES, default="balanced")
    parser.add_argument("--max-fpr", type=float)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--stagnation", type=int, default=20)
    parser.add_argument("--llm-budget", type=int, default=0)
    parser.add_argument("--no-mcp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fpr-penalty", type=float, default=0.2)
    parser.add_argument("--feature-set", choices=("expanded", "legacy"), default="expanded")
    parser.add_argument("--ablate", action="append", choices=("error", "adaptive", "pareto", "cache"), default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = BUDGETS[args.budget]
    if args.output is None:
        from datetime import datetime
        from uuid import uuid4
        args.output = ROOT / "results" / f"{args.mode}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    if args.mode == "hybrid" and args.llm_budget:
        load_env(ROOT / ".env")
    stopped = False

    def cancel(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    from integrations.usage import initialize
    if args.output.exists() and any(item.name not in {"run.log", "stop.requested"} for item in args.output.iterdir()):
        parser.error("Каталог запуска должен быть пустым")
    initialize(args.output)

    def cancelled():
        return stopped or (args.output / "stop.requested").is_file()

    try:
        folder = evolve_local(args.mode, args.dataset, config[0] if args.generations is None else args.generations,
                              args.output, per_generation=config[1] if args.offspring is None else args.offspring,
                              train_size=config[2] if args.train_size is None else args.train_size,
                              validation_size=args.validation_size, time_limit=config[3] if args.time_limit is None
                              else args.time_limit, profile=args.profile, max_fpr=args.max_fpr,
                              min_recall=args.min_recall, seed=args.seed, stagnation_limit=args.stagnation,
                              max_llm_calls=args.llm_budget, mcp_check=not args.no_mcp, fpr_penalty=args.fpr_penalty,
                              feature_set=args.feature_set, ablations=tuple(args.ablate), stop=cancelled)
        print("HTML-отчёт", create_report(folder), flush=True)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 130 if cancelled() else 0


if __name__ == "__main__":
    raise SystemExit(main())
