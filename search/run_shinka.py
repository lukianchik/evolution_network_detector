"""Опциональный сравнительный запуск ShinkaEvolve через Groq."""

import argparse
import os
import secrets
import sys
import threading
from pathlib import Path

from integrations.embedding_server import start_server
from integrations.quota_proxy import start_proxy
from integrations.provider_api import request_completion
from traffic_data.dataset_real import selected_dataset, check_files, feature_names
from search.evolution import ROOT, load_env
from search.local_evolution import proposal_model
from ui.visualize import create_report, load_history


def ensure_local_bypass() -> None:
    """Не направляет запросы к 127.0.0.1 через системный HTTP-прокси."""
    for name in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]
        known = {item.lower() for item in existing}
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in known:
                existing.append(host)
        os.environ[name] = ",".join(existing)


def check_groq(model: str) -> None:
    """Проверяет генерацию до старта Shinka, предотвращая часы повторов 503."""
    print("Проверяем доступность Groq для ShinkaEvolve...", flush=True)
    payload = request_completion(
        "groq", model,
        [{"role": "user", "content": "Return only this valid Python code: def detect(event): return 0"}],
        max_tokens=256, temperature=0.1, max_attempts=1, max_wait=10,
    )
    choices = payload.get("choices", [])
    if not choices or not isinstance(choices[0].get("message", {}).get("content"), str) \
            or not choices[0]["message"]["content"].strip():
        raise RuntimeError("Groq вернул пустой ответ. Проверь выбранную модель.")
    print("Groq отвечает. Начинаем ShinkaEvolve.", flush=True)


def main() -> int:
    """Создаёт конфигурацию ShinkaEvolve и последовательно выполняет поколения."""
    parser = argparse.ArgumentParser(description="Запуск ShinkaEvolve через Groq")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "results" / "shinka"))
    parser.add_argument("--llm-budget", type=int, required=True, help="Явный предел HTTP-попыток, включая проверку")
    parser.add_argument("--dataset", choices=("synthetic", "unsw"), default=None)
    args = parser.parse_args()
    if not 1 <= args.generations <= 20:
        parser.error("Допускается от 1 до 20 поколений за запуск")
    if not 2 <= args.llm_budget <= 30:
        parser.error("Бюджет: 2–30 HTTP-попыток")
    load_env(ROOT / ".env")
    os.environ["SHINKA_PRICING_MODE"] = "offline"
    if args.dataset:
        os.environ["EVO_DATASET"] = args.dataset
    if not os.environ.get("GROQ_API_KEY"):
        print("Не указан GROQ_API_KEY в .env", file=sys.stderr)
        return 2
    try:
        model = proposal_model("groq")
        from shinka.core import ShinkaEvolveRunner, EvolutionConfig
        from shinka.database import DatabaseConfig
        from shinka.launch import LocalJobConfig
    except ImportError as error:
        print(f"Не удалось загрузить ShinkaEvolve: {error}. Установи: pip install shinka-evolve", file=sys.stderr)
        return 2
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        parser.error("Каталог запуска должен быть пустым")
    folder.mkdir(parents=True, exist_ok=True)
    from integrations.usage import initialize
    initialize(folder)
    (folder / "dataset.txt").write_text(selected_dataset(), encoding="utf-8")
    os.environ["EVO_HISTORY_DIR"] = str(folder)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), os.getenv("PYTHONPATH"))))
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if selected_dataset() == "unsw":
        check_files("train")
    ensure_local_bypass()
    try:
        check_groq(model)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ShinkaEvolve не запущен: проверка Groq не прошла: {error}", file=sys.stderr, flush=True)
        return 3
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    server = start_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    # Для Shinka модель выглядит как локальный OpenAI-сервис, но прокси
    # передаёт запрос в Groq с ожиданием квоты и контролем токенов.
    token = secrets.token_urlsafe(24)
    os.environ["EVO_SHINKA_GATEWAY_TOKEN"] = token
    proxy = start_proxy(model, args.llm_budget - 1, token)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    proxy_port = proxy.server_address[1]
    llm_models = [f"local/network-evolution@http://127.0.0.1:{proxy_port}/v1?api_key_env=EVO_SHINKA_GATEWAY_TOKEN"]
    print("Модель Groq", model)
    print("Локальные эмбеддинги:", f"http://127.0.0.1:{port}/v1")
    print("Параллельность: 1 генерация / 1 оценка; GPU не требуется")
    print("Шлюз ограничивает обращения к Groq и учитывает фактические токены")
    print("Набор данных:", selected_dataset())
    try:
        evo_config = EvolutionConfig(
            init_program_path=str(ROOT / "seeds" / ("initial_unsw.py" if selected_dataset() == "unsw" else "initial.py")),
            language="python", num_generations=args.generations,
            results_dir=str(folder / "shinka_engine"),
            llm_models=llm_models,
            llm_dynamic_selection="fixed",
            llm_kwargs={"temperatures": [0.35], "max_tokens": 1800},
            patch_types=["full"], patch_type_probs=[1.0],
            meta_rec_interval=None, novelty_llm_models=None,
            embedding_model=f"local/hash-embedding@http://127.0.0.1:{port}/v1",
            code_embed_sim_threshold=0.998,
            max_novelty_attempts=1, max_patch_resamples=1,
            max_patch_attempts=1, use_text_feedback=True,
            task_sys_msg=(
                "Improve the Python detect(event) function for binary network anomaly detection. "
                "Return 0 or 1 only. "
                "Only modify code inside EVOLVE-BLOCK; keep run_experiment unchanged. "
                "Use simple if statements, comparisons, assignments and arithmetic, no imports, loops, "
                "file access or networking. Keep docstrings to one short sentence. "
                "First propose a simple improvement by changing one threshold or adding one rule. "
                f"Optimize fitness = F1 - {os.getenv('EVO_FPR_PENALTY', '0.2')} * FPR. "
                f"Available features: {', '.join(feature_names())}. "
                "Do not use dataset labels. Evaluation cases are not included in the prompt."
            ),
        )
        job_config = LocalJobConfig(
            eval_program_path=str(ROOT / "evaluation" / "evaluate.py"),
            python_executable=sys.executable,
            numeric_threads_per_job=2,
            time="00:02:00",
        )
        db_config = DatabaseConfig(
            archive_size=25, num_archive_inspirations=1, num_islands=1,
        )
        runner = ShinkaEvolveRunner(
            evo_config=evo_config, job_config=job_config, db_config=db_config,
            max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1,
        )
        runner.run()
        _, rows, _ = load_history(folder)
        if len(rows) <= 1:
            raise RuntimeError("ShinkaEvolve не создал ни одного оценённого потомка")
        print("Эволюция ShinkaEvolve завершена")
        return_code = 0
    except Exception as error:
        print(f"ShinkaEvolve завершился с ошибкой: {error}", file=sys.stderr)
        print("Данные уже выполненных оценок сохранены в", folder)
        return_code = 1
    finally:
        proxy.shutdown()
        proxy.server_close()
        server.shutdown()
        server.server_close()
        os.environ.pop("EVO_SHINKA_GATEWAY_TOKEN", None)
    if (folder / "history.jsonl").exists():
        report = create_report(folder)
        print("HTML-отчёт:", report)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
