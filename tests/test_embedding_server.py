"""Интеграционный тест локального HTTP-сервера эмбеддингов."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from integrations.embedding_server import start_server


class EmbeddingApiTests(unittest.TestCase):
    """Проверяет совместимый формат векторного API и его ограничения."""

    @classmethod
    def setUpClass(cls):
        """Запускает сервер на свободном локальном порту только для тестов."""
        cls.server = start_server()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_address[1]}/v1/embeddings"

    @classmethod
    def tearDownClass(cls):
        """Корректно выключает тестовый HTTP-сервер."""
        cls.server.shutdown()
        cls.server.server_close()

    def test_embedding_response(self):
        """Ответ содержит единообразные векторы для нескольких документов."""
        request = urllib.request.Request(self.endpoint, method="POST",
            data=json.dumps({"model": "hash-embedding", "input": ["test program", "another program"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.load(response)
        self.assertEqual(len(result["data"]), 2)
        self.assertEqual(len(result["data"][0]["embedding"]), 192)

    def test_bad_request(self):
        """Слишком большие запросы отклоняются без выделения большого объёма RAM."""
        request = urllib.request.Request(self.endpoint, method="POST",
            data=json.dumps({"input": ["s"] * 31}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(err.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
