"""Загрузка открытого UNSW-NB15 без передачи правильных ответов детектору.

Файлы CSV скачиваются отдельно; проект не распространяет чужой датасет.
Официальный train используется для эволюции, официальный test — для итоговой оценки.
"""

import csv
import os
import random
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "unsw"
TRAIN = Path(os.getenv("EVO_UNSW_TRAIN", str(DATA_DIR / "UNSW_NB15_training-set.csv")))
TEST = DATA_DIR / "UNSW_NB15_testing-set.csv"
FEATURES = ("dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl",
            "rate", "ct_dst_src_ltm", "ct_srv_dst", "ct_state_ttl")
FLOW_NUMERIC = FEATURES + (
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
)
CATEGORIES = {
    "proto": ("tcp", "udp", "arp", "icmp", "ospf", "sctp"),
    "state": ("fin", "int", "con", "req", "rst", "acc", "clo"),
    "service": ("-", "dns", "http", "ftp", "ftp-data", "smtp", "ssh", "ssl", "dhcp", "snmp"),
}
EXPANDED_FEATURES = FLOW_NUMERIC + tuple(
    f"{field}_{value.replace('-', '_')}" for field, values in CATEGORIES.items() for value in values
) + ("response_byte_fraction", "response_packet_fraction")


def available() -> bool:
    """Проверяет наличие двух исходных CSV-файлов."""
    return TRAIN.is_file() and TEST.is_file()


def check_files(part: str) -> None:
    """Проверяет наличие только нужной части датасета без смешивания train и test."""
    if part not in ("train", "test"):
        raise ValueError("Раздел: train/test")
    path = TRAIN if part == "train" else TEST
    if not path.is_file():
        raise FileNotFoundError(f"Не найден {path.name}. Выполни: python -m traffic_data.download_unsw "
                                "(или положи официальный CSV в data/unsw/)")


def _as_float(value: str) -> float:
    """Преобразует числовое поле и отклоняет некорректные записи."""
    number = float(value)
    if not (-1e12 <= number <= 1e12):
        raise ValueError("Число вне диапазона")
    return number


def flow_features(row: dict, *, defaults: bool = False) -> dict:
    """Кодирует наблюдаемые признаки потока по фиксированной схеме."""
    values = {}
    for name in FLOW_NUMERIC:
        if name in row or defaults:
            value = row.get(name, 0)
            if value != "" or defaults:
                values[name] = _as_float(value if value != "" else 0)
    for field, categories in CATEGORIES.items():
        if field not in row and not defaults:
            continue
        category = str(row.get(field, "")).strip().lower()
        for value in categories:
            name = f"{field}_{value.replace('-', '_')}"
            values[name] = float(category == value)
    for suffix, source, destination in (("byte", "sbytes", "dbytes"), ("packet", "spkts", "dpkts")):
        if source in values and destination in values:
            total = values[source] + values[destination]
            values[f"response_{suffix}_fraction"] = values[destination] / total if total > 0 else 0.0
    return values


@lru_cache(maxsize=4)
def load_unsw(part: str, size: int = 1600, *, expanded: bool = False) -> tuple[tuple[dict, ...], tuple[int, ...]]:
    """Формирует воспроизводимую ограниченную выборку из официального CSV.

    Используется детерминированная reservoir-выборка отдельно по классам,
    затем перемешивание. Это сохраняет приблизительное соотношение классов
    без загрузки всего файла в оперативную память. Ни label, ни attack_cat,
    ни id не попадают в словари признаков.
    """
    if part not in ("train", "test") or not 1 <= size <= 5000:
        raise ValueError("Раздел: train/test, размер: 1–5000")
    check_files(part)
    rng = random.Random(20261121 + (1 if part == "test" else 0))
    pools: dict[int, list[dict]] = {0: [], 1: []}
    counts = {0: 0, 1: 0}
    cap = size  # Резерв каждого класса: не более size словарей.
    path = TRAIN if part == "train" else TEST
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = set(FEATURES) | {"label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"В {path.name} нет необходимых полей: "
                             f"{', '.join(sorted(required - set(reader.fieldnames or [])))}")
        for row in reader:
            try:
                label = int(row["label"])
                if label not in (0, 1):
                    continue
                event = (flow_features(row, defaults=True) if expanded
                         else {name: _as_float(row[name]) for name in FEATURES})
            except (ValueError, TypeError, KeyError):
                continue
            counts[label] += 1
            bucket = pools[label]
            if len(bucket) < cap:
                bucket.append(event)
            else:
                index = rng.randrange(counts[label])
                if index < cap:
                    bucket[index] = event
    total = sum(counts.values())
    if total == 0 or not counts[0] or not counts[1]:
        raise ValueError("Датасет пуст или в нём отсутствует один из классов")
    # Доля атаки соответствует выборке в оригинальном CSV (без искусственного 50/50).
    sample_size = min(size, total)
    max_attacks = sample_size - 1 if sample_size > 1 else 1
    attack_count = min(counts[1], max_attacks, max(1, round(sample_size * counts[1] / total)))
    normal_count = min(counts[0], size - attack_count)
    items = [(event, label) for label, count in ((0, normal_count), (1, attack_count))
             for event in rng.sample(pools[label], count)]
    rng.shuffle(items)
    return tuple(item[0] for item in items), tuple(item[1] for item in items)


def selected_dataset() -> str:
    """Возвращает тип данных из переменной окружения."""
    name = os.getenv("EVO_DATASET", "synthetic").strip().lower()
    if name not in ("synthetic", "unsw"):
        raise ValueError("EVO_DATASET должен быть synthetic или unsw")
    return name


def feature_names() -> tuple[str, ...]:
    """Предоставляет список доступных детектору полей текущей задачи."""
    if selected_dataset() == "unsw":
        return EXPANDED_FEATURES
    from traffic_data.dataset import FEATURES as synthetic_features
    return synthetic_features


def iter_full_test(*, categories: bool = False, control_csv: Path | None = None):
    """Потоково читает итоговый набор; ошибка данных не скрывается."""
    if control_csv is None:
        check_files("test")
    with (TEST if control_csv is None else control_csv).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not (set(FEATURES) | {"label"}).issubset(reader.fieldnames or []):
            raise ValueError("В тестовом CSV отсутствуют необходимые столбцы")
        for row in reader:
            try:
                label = int(row["label"])
                if label not in (0, 1):
                    raise ValueError("Метка не равна 0/1")
                event = flow_features(row, defaults=True)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Некорректная тестовая строка {reader.line_num}") from error
            if categories:
                yield event, label, row.get("attack_cat", "").strip()
            else:
                yield event, label
