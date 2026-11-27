"""Генератор небольшого искусственного набора сетевых событий для учебного эксперимента.

Метки создаются только для оценки качества; детектор получает события без меток.
Эти данные не являются настоящими логами IDS и не доказывают качество на реальном трафике.
"""

import random

FEATURES = (
    "failed_logins",
    "connections_per_minute",
    "unique_ports",
    "outbound_mb",
    "interval_std",
    "risky_port",
)


def build_dataset(seed: int, size: int = 450) -> tuple[list[dict], list[int]]:
    """Создаёт воспроизводимый набор событий и эталонных меток 0/1.

    Генерируются четыре типа подозрительного поведения и несколько сложных
    нормальных случаев. Используется локальный генератор, не глобальный random.
    """
    if size < 1:
        raise ValueError("Размер выборки должен быть положительным")
    rng = random.Random(seed)
    events, labels = [], []
    for index in range(size):
        suspicious = rng.random() < 0.30
        event = {
            "failed_logins": rng.randint(0, 4),
            "connections_per_minute": rng.randint(3, 62),
            "unique_ports": rng.randint(1, 10),
            "outbound_mb": round(rng.uniform(0.2, 38), 2),
            "interval_std": round(rng.uniform(1.2, 12), 3),
            "risky_port": int(rng.random() < 0.06),
        }
        if suspicious:
            kind = rng.choice(("bruteforce", "scan", "beacon", "exfil"))
            if kind == "bruteforce":
                event["failed_logins"] = rng.randint(7, 26)
                event["connections_per_minute"] = rng.randint(28, 130)
            elif kind == "scan":
                event["unique_ports"] = rng.randint(16, 80)
                event["connections_per_minute"] = rng.randint(36, 150)
            elif kind == "beacon":
                event["interval_std"] = round(rng.uniform(0.02, 0.85), 3)
                event["risky_port"] = int(rng.random() < 0.55)
            else:
                event["outbound_mb"] = round(rng.uniform(70, 360), 2)
                event["risky_port"] = int(rng.random() < 0.38)
            # Вносим небольшую долю неполных / нечётких сигналов.
            if rng.random() < 0.06:
                feature = rng.choice(("failed_logins", "unique_ports", "outbound_mb"))
                event[feature] = rng.choice((4, 8, 30))
        else:
            # Законные резервные копии, администрирование, мониторинг.
            if rng.random() < 0.09:
                event["outbound_mb"] = round(rng.uniform(70, 145), 2)
            if rng.random() < 0.07:
                event["unique_ports"] = rng.randint(16, 25)
            if rng.random() < 0.08:
                event["interval_std"] = round(rng.uniform(0.18, 0.90), 3)
        events.append(event)
        labels.append(int(suspicious))
    return events, labels
