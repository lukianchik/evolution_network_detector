"""Начальный детектор для эволюции ShinkaEvolve.

LLM должна изменять только функцию detect() внутри EVOLVE-BLOCK.
Функция получает словарь сетевых признаков и возвращает метку 0 или 1.
"""

# EVOLVE-BLOCK-START

def detect(event: dict) -> int:
    """Считает подозрительными события с большим числом неудачных входов."""
    return int(event["failed_logins"] >= 9)

# EVOLVE-BLOCK-END


def run_experiment(events: list[dict]) -> list[int]:
    """Применяет детектор к событиям без передачи ему правильных ответов."""
    return [int(detect(event)) for event in events]
