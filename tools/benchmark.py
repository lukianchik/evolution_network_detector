"""Сравнение локального и гибридного поиска и внутренних абляций."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from artifacts.history_store import candidate_path, has_history, read_history_payload, selected_row
from integrations.usage import read_usage
from search.evolution import ROOT
from ui.visualize import load_history

MODES = ("local", "hybrid", "shinka")
SIZES = {"train-10000": 10000, "train-20000": 20000, "train-50000": 50000, "full-train": 0}
ABLATIONS = {"no-error": "error", "no-adaptive": "adaptive", "scalar": "pareto", "no-cache": "cache"}
VARIANTS = ("unified", "legacy-features", *SIZES, *ABLATIONS)
METRICS = ("f1", "precision", "recall", "fpr", "fitness")
COUNTERS = ("local_attempts", "train_evaluations", "guided_train_evaluations", "error_stat_checks",
            "llm_improvements", "preparation_seconds", "elapsed_seconds", "peak_rss_mib")


def run_suite(output: Path, dataset: str, modes: list[str], seeds: list[int], generations: int,
              offspring: int, llm_budget: int, penalty: float, *, final: bool = False,
              variants: list[str] | None = None, train_size=20000, validation_size=4096,
              workers=1, timeout=300, reference_summary: Path | None = None) -> dict:
    """Фиксирует лидеров до доступа к test и сохраняет индивидуальные исходы."""
    variants = variants or ["unified"]
    if (not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES)
            or not seeds or len(set(seeds)) != len(seeds) or not 1 <= generations <= 30
            or not 1 <= offspring <= 30 or not 0 <= llm_budget <= 5 or not 0 <= penalty <= 2
            or not 1 <= workers <= 4 or not 1 <= timeout <= 3600 or dataset not in ("synthetic", "unsw")
            or not 0 <= train_size <= 200000 or not 30 <= validation_size <= 50000
            or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS)):
        raise ValueError("Некорректные параметры benchmark")
    if "shinka" in modes and (llm_budget < 2 or variants != ["unified"]):
        raise ValueError("Shinka требует явный бюджет 2–5 и вариант unified")
    reference = json.loads(reference_summary.read_text(encoding="utf-8")) if reference_summary else None
    if reference and reference.get("manifest", {}).get("dataset") != dataset:
        raise ValueError("Несовпадение датасетов контрольного и текущего экспериментов")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"dataset": dataset, "modes": modes, "seeds": seeds, "variants": variants,
                "train_size": train_size, "validation_size": validation_size, "workers": workers,
                "generations": generations, "offspring": offspring, "llm_budget": llm_budget,
                "fpr_penalty": penalty, "timeout": timeout, "final": final}
    env = {**os.environ, "EVO_DATASET": dataset, "EVO_COMMON_VALIDATION": "1",
           "EVO_FPR_PENALTY": str(penalty), "PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("EVO_MCP_URL", None)
    if llm_budget == 0:
        env.pop("GROQ_API_KEY", None)

    def execute(task):
        """Запускает независимый процесс и сохраняет его измерения."""
        seed, mode, variant = task
        folder = output / f"{mode}-{variant}-{seed}"
        if mode == "shinka":
            command = ["-m", "search.run_shinka", "--generations", str(min(generations, 20)),
                       "--llm-budget", str(llm_budget)]
        else:
            command = ["-m", "search.local_evolution", "--mode", mode, "--seed", str(seed), "--no-mcp",
                       "--generations", str(generations), "--offspring", str(offspring),
                       "--llm-budget", str(llm_budget if mode == "hybrid" else 0), "--fpr-penalty", str(penalty),
                       "--train-size", str(SIZES.get(variant, train_size)), "--validation-size",
                       str(validation_size), "--time-limit", str(timeout)]
            if variant == "legacy-features":
                command.extend(["--feature-set", "legacy"])
            if variant in ABLATIONS:
                command.extend(["--ablate", ABLATIONS[variant]])
        started = time.monotonic()
        with (output / f"{folder.name}.log").open("wb") as log:
            try:
                process = subprocess.run([sys.executable, *command, "--dataset", dataset, "--output", str(folder)],
                                         env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                status, code = ("completed" if process.returncode == 0 else "failed"), process.returncode
            except subprocess.TimeoutExpired:
                status, code = "timeout", None
        entry = {"mode": mode, "variant": variant, "repeat_seed": seed,
                 "search_seed": None if mode == "shinka" else seed, "run": folder.name,
                 "status": status, "return_code": code, "wall_seconds": round(time.monotonic() - started, 3),
                 **read_usage(folder)}
        try:
            _, history, rejected = load_history(folder)
            payload = read_history_payload(folder) if has_history(folder) else {}
            best = selected_row(history, payload)
            stats = payload.get("statistics", {})
            entry.update({"evaluated": len(history), "rejected": rejected, "winner": best["id"],
                          "seed_path": history[0]["path"], "winner_path": best["path"],
                          "winner_source": best.get("model") or best.get("kind"), "sample": stats.get("sample"),
                          "seed_metrics": {name: history[0].get(name) for name in METRICS},
                          "validation_categories": stats.get("winner_categories"),
                          **{f"validation_{name}": best.get(name) for name in METRICS},
                          **{name: best.get(name) for name in ("complexity", "generalization_gap", "instability",
                                                              "train_f1", "train_fpr", "train_recall")},
                          **{name: stats.get(name) for name in (*COUNTERS, "strategy_attempts", "search_status")}})
            if stats.get("search_status") == "time-limit":
                entry["status"] = "time-limit"
        except (OSError, ValueError, KeyError) as error:
            entry["history_error"] = str(error)
            if status == "completed":
                entry["status"] = "no_history"
        print(f"{mode}/{variant}/{seed}: {entry['status']} · F1={entry.get('validation_f1', '—')}", flush=True)
        return entry

    tasks = [(seed, mode, variant) for seed in seeds for mode in modes for variant in variants]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(execute, tasks))
    (output / "selected.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if final:
        from evaluation.final_test import test_program
        previous = os.environ.get("EVO_DATASET")
        os.environ["EVO_DATASET"] = dataset
        try:
            for entry in rows:
                if entry["status"] == "completed":
                    try:
                        result = test_program(candidate_path(output / entry["run"], entry["winner_path"]),
                                              fpr_penalty=penalty)
                        entry.update({f"test_{key}": result[key] for key in METRICS})
                        entry["test_count"] = result["count"]
                        entry["test_categories"] = result.get("categories")
                    except (OSError, ValueError) as error:
                        entry["test_error"] = str(error)
        finally:
            if previous is None:
                os.environ.pop("EVO_DATASET", None)
            else:
                os.environ["EVO_DATASET"] = previous

    groups = {}
    for mode in modes:
        for variant in variants:
            completed = [row for row in rows if row["mode"] == mode and row["variant"] == variant
                         and row["status"] == "completed"]
            group = {"completed": len(completed), "requested": len(seeds)}
            fields = [*(f"validation_{name}" for name in METRICS), *(f"test_{name}" for name in METRICS),
                      "train_f1", "train_fpr", "train_recall", "complexity", "generalization_gap", "instability",
                      "wall_seconds", *COUNTERS, "http_attempts", "total_tokens"]
            for metric in fields:
                values = [row[metric] for row in completed if row.get(metric) is not None and (
                    metric != "total_tokens" or (row["usage_available"] and
                                                row["http_attempts"] == row["known_usage_responses"]))]
                if values:
                    group[metric] = {"n": len(values), "mean": round(statistics.mean(values), 6),
                                     "sd": round(statistics.stdev(values), 6) if len(values) > 1 else None}
            groups[f"{mode}/{variant}"] = group
    report = {"manifest": manifest, "runs": rows, "groups": groups}
    if reference is not None:
        report["reference_summary"] = reference
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сводка: {output / 'summary.json'}", flush=True)
    return report


def main():
    """Читает параметры воспроизводимого эксперимента."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "unsw"), default="synthetic")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["local"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["unified"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--offspring", type=int, default=15)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=4096)
    parser.add_argument("--llm-budget", type=int, default=0)
    parser.add_argument("--fpr-penalty", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--final", action="store_true", help="Открыть test после фиксации всех лидеров")
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / datetime.now().strftime("benchmark-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    report = run_suite(args.output, args.dataset, args.modes, args.seeds, args.generations,
                       args.offspring, args.llm_budget, args.fpr_penalty, final=args.final, variants=args.variants,
                       train_size=args.train_size, validation_size=args.validation_size, workers=args.workers,
                       timeout=args.timeout, reference_summary=args.reference_summary)
    if any(row["status"] != "completed" or row.get("test_error") for row in report["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
