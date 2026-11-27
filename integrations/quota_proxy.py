"""Локальный OpenAI-совместимый шлюз для контролируемых вызовов ShinkaEvolve.

Библиотека Shinka обращается сюда по HTTP; шлюз ждёт сброса квот, затем
пересылает запрос настоящему Groq. API-ключ браузеру не передаётся.
"""

import json
import secrets
from threading import Lock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from integrations.provider_api import request_completion


class Handler(BaseHTTPRequestHandler):
    """Преобразует формат OpenAI API и передаёт запрос провайдеру."""

    def do_POST(self):
        """Обрабатывает только Chat Completions с фиксированной моделью."""
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            self.send_error(403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 150000:
                raise ValueError("Слишком большой запрос")
            body = json.loads(self.rfile.read(size))
            if body.get("stream"):
                raise ValueError("Потоковые ответы не поддерживаются")
            with self.server.budget_lock:
                if self.server.remaining <= 0:
                    raise RuntimeError("Исчерпан лимит HTTP-попыток")
                self.server.remaining -= 1
            result = request_completion("groq", self.server.model, body.get("messages", []),
                                        max_tokens=min(int(body.get("max_tokens") or
                                                           body.get("max_completion_tokens") or 1800), 2500),
                                        temperature=float(body.get("temperature") or 0.4),
                                        max_attempts=1, max_wait=10)
            raw = json.dumps(result, ensure_ascii=False).encode("utf-8")
            status = 200
        except Exception as error:
            # ShinkaEvolve по умолчанию многократно повторяет даже ошибки 4xx.
            # Явный маркер позволяет панели остановить безнадёжный прогон.
            message = str(error).replace("\r", " ").replace("\n", " ")[:400]
            print(f"[API-FATAL] Ошибка шлюза Groq: {message}", flush=True)
            raw = json.dumps({"error": {"message": message, "type": "proxy_error"}},
                             ensure_ascii=False).encode("utf-8")
            status = 403
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format, *_args):
        """Не выводит секреты и HTTP-параметры в терминал."""
        return


def start_proxy(model: str, remaining: int, token: str) -> ThreadingHTTPServer:
    """Открывает шлюз только на loopback, чтобы сервис не был доступен из LAN."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.remaining = remaining
    server.budget_lock = Lock()
    server.model = model
    server.token = token
    return server
