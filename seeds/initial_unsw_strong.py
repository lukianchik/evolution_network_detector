"""Простой исходный детектор для реальных записей UNSW-NB15."""

# EVOLVE-BLOCK-START

def detect(event: dict) -> int:
    """Помечает поток подозрительным при высоком TTL или частых повторениях адресов."""
    return int(event["sttl"] >= 120 or event["ct_dst_src_ltm"] >= 15)

# EVOLVE-BLOCK-END


def run_experiment(events: list[dict]) -> list[int]:
    """Применяет функцию обнаружения к списку сетевых потоков."""
    return [int(detect(event)) for event in events]
