"""Локальная панель управления экспериментами и просмотра эволюции кода.

Запуск: python app.py. Панель доступна только на собственном компьютере.
Она запускает существующие CLI-режимы без передачи ключей в браузер.
"""

from collections import deque
import difflib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from search.evolution import ROOT
from search.local_evolution import BUDGETS
from search.strategies import PROFILES
from artifacts.history_store import candidate_path, has_history, read_history_payload, selected_row
from traffic_data.dataset_real import TRAIN, check_files
from integrations.quotas import QuotaLedger, limits_for, quota_key
from integrations.usage import read_usage
from ui.visualize import create_report, load_history

PAGE = ROOT / "ui" / "panel.html"
RESULTS = ROOT / "results"
MODES = {"local": "Local", "hybrid": "Hybrid"}
RUN_IDS = re.compile(r"^[a-z0-9_-]{1,64}$")


class ExperimentManager:
    """Хранит состояние единственного активного эксперимента и его историю."""

    def __init__(self):
        """Инициализирует потокобезопасное состояние панели."""
        self.lock = threading.RLock()
        self.process = None
        self.active = None
        self.runs = {}
        self.token = secrets.token_urlsafe(24)
        example = RESULTS / "demo"
        if (example / "history.json").exists():
            self.runs["example"] = {"id": "example", "mode": "demo", "generations": 8,
                                    "folder": example, "status": "completed", "logs": [
                                        "Пример создан автономными мутациями без LLM."],
                                    "return_code": 0}
            self.active = "example"

        self.restore_runs()

    def restore_runs(self) -> None:
        """Восстанавливает архивные запуски без ложного статуса completed."""
        if not RESULTS.is_dir():
            return
        for folder in sorted(RESULTS.iterdir(), key=lambda path: path.name):
            if not folder.is_dir() or folder.is_symlink() or folder.name == "demo" or not RUN_IDS.fullmatch(folder.name):
                continue
            try:
                mode, rows, _ = load_history(folder)
                marker = folder / "dataset.txt"
                dataset = marker.read_text(encoding="utf-8").strip() if marker.is_file() else "unknown"
                if not rows:
                    continue
                saved = read_history_payload(folder) if has_history(folder) else {}
                search_status = saved.get("statistics", {}).get("search_status")
                status = {"completed": "completed", "early-stop": "completed", "time-limit": "completed",
                          "running": "stopped", "stopped": "stopped"}.get(search_status, "archived")
                logs = []
                if (folder / "run.log").is_file():
                    with (folder / "run.log").open(encoding="utf-8", errors="replace") as log_file:
                        logs = list(deque((line.rstrip() for line in log_file), maxlen=120))
                status_file = folder / "run_status.json"
                if status_file.is_file():
                    metadata = json.loads(status_file.read_text(encoding="utf-8"))
                    if metadata.get("status") in ("completed", "failed", "stopped"):
                        status = metadata["status"]
                self.runs[folder.name] = {"id": folder.name, "mode": mode, "dataset": dataset,
                                          "generations": max(row["generation"] for row in rows),
                                          "folder": folder, "status": status, "logs": logs, "return_code": None}
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if self.runs:
            self.active = max(self.runs, key=self._run_order)

    @staticmethod
    def _run_order(run_id: str) -> tuple[str, str]:
        """Сортирует идентификаторы по дате независимо от режима."""
        match = re.search(r"\d{8}-\d{6}", run_id)
        return (match.group() if match else "", run_id)

    def comparisons(self) -> list[dict]:
        """Возвращает сводку архивов с настройками и метриками лидера."""
        with self.lock:
            runs = sorted(self.runs.values(), key=lambda run: self._run_order(run["id"]), reverse=True)
        result = []
        for run in runs:
            try:
                folder = run["folder"]
                _, rows, _ = load_history(folder)
                if not rows:
                    continue
                saved = read_history_payload(folder) if has_history(folder) else {}
                best = selected_row(rows, saved)
                stats = saved.get("statistics", {})
                result.append({"id": run["id"], "mode": run["mode"], "dataset": run.get("dataset", "unknown"),
                               "status": run["status"], "evaluated": len(rows), "seed": stats.get("seed"),
                               "penalty": stats.get("fpr_penalty", 0.2), "f1": best["f1"], "fpr": best["fpr"],
                               "fitness": best["fitness"], "source": best.get("model") or best.get("kind"),
                               "local_attempts": stats.get("local_attempts"), "seconds": stats.get("elapsed_seconds"),
                               "profile": stats.get("profile"), "recall": best.get("recall"),
                               "sample": stats.get("sample", {}).get("id"),
                               "train_evaluations": stats.get("train_evaluations", stats.get("guided_train_evaluations")),
                               "report_ready": (folder / "report.html").is_file(),
                               "usage": read_usage(folder)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return result

    def start(self, mode: str, generations: int | None = None, dataset: str = "synthetic",
              offspring: int | None = None, llm_budget: int = 0, seed: int = 42, *,
              budget: str = "standard", profile: str = "balanced", train_size: int | None = None,
              validation_size: int = 4096, time_limit: float | None = None, max_fpr: float | None = None,
              min_recall: float | None = None, numeric_threads: int = 2) -> dict:
        """Запускает локальный или гибридный эксперимент в отдельном процессе."""
        if mode not in MODES or dataset not in ("synthetic", "unsw") or budget not in BUDGETS or profile not in PROFILES:
            raise ValueError("Недопустимый режим, датасет, бюджет или профиль")
        defaults = BUDGETS[budget]
        generations = defaults[0] if generations is None else generations
        offspring = defaults[1] if offspring is None else offspring
        train_size = defaults[2] if train_size is None else train_size
        time_limit = defaults[3] if time_limit is None else time_limit
        if not (1 <= generations <= 30 and 1 <= offspring <= 30 and 0 <= llm_budget <= 5 and 0 <= seed < 2**32
                and 0 <= train_size <= 200000 and 30 <= validation_size <= 50000 and 1 <= time_limit <= 3600
                and 1 <= numeric_threads <= 4
                and all(value is None or 0 <= value <= 1 for value in (max_fpr, min_recall))):
            raise ValueError("Параметры запуска вне допустимого диапазона")
        if dataset == "unsw":
            check_files("train")
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("Уже выполняется эксперимент")
            name = f"{mode}-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
            folder = RESULTS / name
            folder.mkdir(parents=True)
            cmd = [sys.executable, "-u", "-m", "search.local_evolution",
                   "--generations", str(generations), "--output", str(folder), "--mode", mode,
                   "--dataset", dataset, "--offspring", str(offspring), "--llm-budget", str(llm_budget),
                   "--seed", str(seed), "--profile", profile, "--train-size", str(train_size),
                   "--validation-size", str(validation_size), "--time-limit", str(time_limit)]
            for option, value in (("--max-fpr", max_fpr), ("--min-recall", min_recall)):
                if value is not None:
                    cmd.extend([option, str(value)])
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                        "EVO_DATASET": dataset, "OMP_NUM_THREADS": str(numeric_threads),
                        "MKL_NUM_THREADS": str(numeric_threads), "OPENBLAS_NUM_THREADS": str(numeric_threads)})
            if mode == "local" or llm_budget == 0:
                env.pop("GROQ_API_KEY", None)
            self.process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                            errors="replace", bufsize=1)
            run = {"id": name, "mode": mode, "dataset": dataset, "generations": generations, "folder": folder,
                   "status": "running", "logs": [], "return_code": None}
            self.runs[name] = run
            self.active = name
            process = self.process
        threading.Thread(target=self._watch, args=(run, process), daemon=True).start()
        return {"id": name, "mode": mode, "status": "running"}

    def _watch(self, run: dict, process: subprocess.Popen) -> None:
        """Считывает журнал, завершает процесс и создаёт автономный HTML-отчёт."""
        folder = run["folder"]
        try:
            with (folder / "run.log").open("a", encoding="utf-8", buffering=1) as log:
                for line in process.stdout:
                    log.write(line)
                    with self.lock:
                        run["logs"].append(line.rstrip()[:500])
                        if len(run["logs"]) > 120:
                            del run["logs"][:-120]
            code = process.wait()
        except Exception as error:
            code = 1
            with self.lock:
                run["logs"].append(f"Ошибка чтения журнала: {error}")
        if has_history(folder) and not (folder / "report.html").is_file():
            try:
                create_report(folder)
            except Exception as error:
                with self.lock:
                    run["logs"].append(f"Отчёт не создан: {error}")
        with self.lock:
            run["return_code"] = code
            run["status"] = "completed" if code == 0 else "stopped" if code == 130 or run.get("stopping") else "failed"
            try:
                temporary = folder / "run_status.tmp"
                temporary.write_text(json.dumps({"status": run["status"], "return_code": code}), encoding="utf-8")
                temporary.replace(folder / "run_status.json")
            except OSError:
                pass

    def stop(self) -> dict:
        """Останавливает выполняющийся процесс по запросу пользователя."""
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise ValueError("Сейчас нет активного эксперимента")
            self.runs[self.active]["stopping"] = True
            (self.runs[self.active]["folder"] / "stop.requested").touch()
            return {"status": "stopping"}

    def snapshot(self, run_id: str | None = None) -> dict:
        """Возвращает статус, оценки и агрегированные метрики выбранного запуска."""
        with self.lock:
            if not run_id:
                run_id = self.active
            if not run_id or run_id not in self.runs:
                return {"status": "idle", "evaluations": [], "logs": []}
            run = self.runs[run_id]
            data = {"id": run_id, "mode": run["mode"], "dataset": run.get("dataset", "synthetic"),
                    "status": run["status"],
                    "generations": run["generations"], "logs": run["logs"][-40:],
                    "return_code": run["return_code"]}
            folder = run["folder"]
        try:
            payload = read_history_payload(folder) if has_history(folder) else None
            _mode, rows, rejected = load_history(folder, payload)
        except (OSError, ValueError, json.JSONDecodeError):
            rows, rejected = [], 0
            payload = None
        data["evaluations"] = [
            {key: row.get(key) for key in ("id", "generation", "parent", "second_parent", "fitness", "f1", "fpr", "recall", "model", "kind", "selection_score")}
            for row in rows
        ]
        data["rejected"] = rejected
        data["usage"] = read_usage(folder)
        payload = payload or {}
        data["statistics"] = payload.get("statistics", {})
        data["winner_id"] = payload.get("winner_id")
        data["pareto_ids"] = payload.get("pareto_ids", [])
        try:
            if run["mode"] not in ("hybrid", "hybrid-groq", "groq"):
                data["quota"] = None
                data["report_ready"] = (folder / "report.html").is_file()
                return data
            provider = "groq"
            model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            data["quota"] = QuotaLedger().snapshot(quota_key(provider, model), limits_for(provider))
        except (OSError, ValueError):
            data["quota"] = None
        data["report_ready"] = (folder / "report.html").is_file()
        return data

    def code(self, run_id: str, candidate_id: str, compare: str) -> dict:
        """Сравнивает код выбранного кандидата с родителем или исходной программой."""
        with self.lock:
            if run_id not in self.runs:
                raise ValueError("Неизвестный запуск")
            folder = self.runs[run_id]["folder"]
        _mode, rows, _ = load_history(folder)
        indexed = {str(row["id"]): row for row in rows}
        if candidate_id not in indexed:
            raise ValueError("Кандидат не найден")
        index = next(i for i, row in enumerate(rows) if str(row["id"]) == candidate_id)
        target = indexed[candidate_id]
        if compare == "seed":
            original = rows[0]
            title = "Исходная версия"
        elif compare == "previous" and index > 0:
            original = rows[index-1]
            title = "Предыдущая оценка"
        elif target.get("parent") in indexed:
            original = indexed[target["parent"]]
            title = "Родитель"
        elif index > 0:
            original = rows[index - 1]
            title = "Предыдущая оценка (родитель неизвестен)"
        else:
            original = target
            title = "Исходная версия"
        before = self._read_source(folder, original["path"])
        after = self._read_source(folder, target["path"])
        delta = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                         fromfile=str(original["id"]), tofile=str(target["id"]), lineterm=""))
        return {"before": before, "after": after, "diff": delta,
                "base": str(original["id"]), "title": title, "candidate": candidate_id}

    @staticmethod
    def _read_source(folder: Path, relative_path: str) -> str:
        """Читает только сохранённый Python-файл внутри папки эксперимента."""
        path = candidate_path(folder, relative_path)
        return path.read_text(encoding="utf-8")


class PanelHandler(BaseHTTPRequestHandler):
    """Обрабатывает запросы панели на локальном HTTP-сервере."""

    manager: ExperimentManager = None

    def parse_request(self):
        """Accept requests addressed to this loopback listener only."""
        if not super().parse_request():
            return False
        port = self.server.server_port
        ports = {port}
        if os.getenv("EVO_CONTAINER") == "1":
            ports.add(int(os.getenv("EVO_WEB_PORT", str(port))))
        if self.headers.get("Host") not in {f"{host}:{value}" for value in ports
                                            for host in ("127.0.0.1", "localhost")}:
            self.send_error(403)
            return False
        return True

    def log_message(self, format_string, *args):
        """Не засоряет консоль регулярными запросами состояния."""
        if not self.path.startswith("/api/state"):
            super().log_message(format_string, *args)

    def _json(self, code: int, data: dict) -> None:
        """Отправляет структурированный JSON-ответ без кеширования."""
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        """Показывает панель, состояние эксперимента, изменения кода и отчёты."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                raw = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if parsed.path == "/panel.js":
                raw = (ROOT / "ui" / "panel.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if parsed.path == "/api/config":
                return self._json(200, {"token": self.manager.token,
                                        "initial_run": self.manager.active,
                                        "modes": MODES,
                                        "budgets": BUDGETS,
                                        "unsw_available": TRAIN.is_file()})
            if parsed.path == "/api/runs":
                return self._json(200, {"runs": self.manager.comparisons(), "active": self.manager.active,
                                        "running": self.manager.process is not None and self.manager.process.poll() is None})
            if parsed.path == "/api/state":
                return self._json(200, self.manager.snapshot(query.get("run", [None])[0]))
            if parsed.path == "/api/code":
                result = self.manager.code(query.get("run", [""])[0], query.get("id", [""])[0],
                                           query.get("compare", ["parent"])[0])
                return self._json(200, result)
            if parsed.path == "/report":
                run_id = query.get("run", [""])[0]
                if run_id not in self.manager.runs or not RUN_IDS.fullmatch(run_id):
                    raise ValueError("Неизвестный запуск")
                file = self.manager.runs[run_id]["folder"] / "report.html"
                if file.is_symlink():
                    raise ValueError("Некорректный путь отчёта")
                raw = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Security-Policy", "sandbox; default-src 'none'; style-src 'unsafe-inline'")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self._json(404, {"error": "Страница не найдена"})
        except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})

    def do_POST(self):
        """Принимает запуск или остановку только из локальной панели."""
        origin = self.headers.get("Origin", "")
        host = self.headers.get("Host", "")
        if not origin or origin not in ("http://" + host,):
            return self._json(403, {"error": "Запрос должен исходить из локальной панели"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("Неверный размер запроса")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Ожидается JSON-объект")
            if request.get("token") != self.manager.token:
                return self._json(403, {"error": "Неверный токен панели"})
            if self.path == "/api/start":
                optional = {key: cast(request[key]) for key, cast in {
                    "generations": int, "offspring": int, "train_size": int, "validation_size": int,
                    "time_limit": float, "max_fpr": float, "min_recall": float, "numeric_threads": int,
                }.items() if request.get(key) is not None}
                result = self.manager.start(request.get("mode", ""), dataset=request.get("dataset", "synthetic"),
                                            llm_budget=int(request.get("llm_budget", 0)),
                                            seed=int(request.get("seed", 42)), budget=request.get("budget", "standard"),
                                            profile=request.get("profile", "balanced"), **optional)
            elif self.path == "/api/stop":
                result = self.manager.stop()
            else:
                return self._json(404, {"error": "Команда не найдена"})
            self._json(200, result)
        except (ValueError, TypeError, OverflowError, OSError) as error:
            self._json(400, {"error": str(error)})


def prepare_metrics_train() -> None:
    """Копирует только официальный train в том сервера метрик."""
    destination = os.getenv("EVO_METRICS_TRAIN")
    if not destination:
        return
    target = Path(destination)
    if not TRAIN.is_file():
        target.unlink(missing_ok=True)
        return
    source_info = TRAIN.stat()
    if target.is_file():
        target_info = target.stat()
        if (target_info.st_size, target_info.st_mtime_ns) == (source_info.st_size, source_info.st_mtime_ns):
            return
    temporary = target.with_suffix(".tmp")
    shutil.copy2(TRAIN, temporary)
    temporary.replace(target)


def main() -> None:
    """Открывает локальный веб-интерфейс и обрабатывает нажатия пользователя."""
    prepare_metrics_train()
    manager = ExperimentManager()
    # Передаём случайный токен только браузеру, открывшему страницу панели.
    class BoundHandler(PanelHandler):
        """Связывает обработчик HTTP с конкретным экспериментальным менеджером."""
        pass
    BoundHandler.manager = manager
    container = os.getenv("EVO_CONTAINER") == "1"
    server = ThreadingHTTPServer(("0.0.0.0", 8080) if container else ("127.0.0.1", 0), BoundHandler)
    port = server.server_address[1]
    print(f"Панель запущена: http://127.0.0.1:{port}")
    print("Остановка: Ctrl+C")
    if container:
        def stop_signal(_signum, _frame):
            """Завершает панель при остановке контейнера."""
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop_signal)
    else:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            manager.stop()
            try:
                manager.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                manager.process.terminate()
        except ValueError:
            pass
        server.server_close()


if __name__ == "__main__":
    main()
