"""Простой стартовый детектор UNSW-NB15 для эволюционного эксперимента."""

# EVOLVE-BLOCK-START

def detect(event: dict) -> int:
    """Считает поток подозрительным при большом числе повторных обращений."""
    return int(event["ct_dst_src_ltm"] >= 20)

# EVOLVE-BLOCK-END


def run_experiment(events: list[dict]) -> list[int]:
    """Применяет детектор к потокам без доступа к правильным ответам."""
    return [int(detect(event)) for event in events]
