"""Локальный OpenAI-совместимый API хэш-эмбеддингов для ShinkaEvolve.

Используется только localhost. Не загружает нейросеть и не требует GPU.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from detectors.novelty import embed


class EmbeddingHandler(BaseHTTPRequestHandler):
    """Отвечает на запросы /v1/embeddings в формате OpenAI API."""

    def do_POST(self):
        """Вычисляет векторы для одного или нескольких фрагментов кода."""
        if self.path not in ("/v1/embeddings", "/embeddings"):
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if not (0 < length < 100_000):
            self.send_error(413, "Invalid content length")
            return
        try:
            payload = json.loads(self.rfile.read(length))
            values = payload["input"]
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list) or len(values) > 30:
                raise ValueError("Invalid input")
            if not all(isinstance(item, str) and len(item) < 15000 for item in values):
                raise ValueError("Invalid input items")
            reply = {"object": "list", "data": [
                {"object": "embedding", "index": index, "embedding": embed(item)}
                for index, item in enumerate(values)],
                "model": "hash-embedding", "usage": {"prompt_tokens": 0, "total_tokens": 0}}
            raw = json.dumps(reply).encode("utf-8")
            self.send_response(200)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            raw = b'{"error":{"message":"Invalid embedding request"}}'
            self.send_response(400)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        """Убирает шумные сообщения HTTP-сервера из консоли эксперимента."""
        return


def start_server() -> ThreadingHTTPServer:
    """Поднимает сервер только на 127.0.0.1 со свободным системным портом."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    server.daemon_threads = True
    return server
