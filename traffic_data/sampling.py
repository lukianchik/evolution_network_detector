"""Фиксированные групповые выборки UNSW из официального train."""

import csv
import hashlib
from array import array
from functools import lru_cache
from pathlib import Path

import numpy as np

from traffic_data import dataset_real

SPLIT_SEED = 20261127
CONTROL_SIZE = 4096


@lru_cache(maxsize=2)
def read_flow_table(path: Path, stamp: int) -> tuple:
    """Читает числовую матрицу и хранит разметку отдельно от признаков."""
    numeric = array("d")
    labels, categories, raw_categories = [], [], []
    missing = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, ())
        column = {name: index for index, name in enumerate(header)}
        required = set(dataset_real.FEATURES) | {"label"}
        if not required.issubset(column):
            raise ValueError(f"Отсутствуют обязательные поля CSV: {path.name}")
        numeric_columns = [(name, column.get(name)) for name in dataset_real.FLOW_NUMERIC]
        category_columns = [(name, column.get(name)) for name in dataset_real.CATEGORIES]
        label_column = column["label"]
        attack_column = column.get("attack_cat")
        for row in reader:
            if not row:
                continue
            try:
                label = int(row[label_column])
                if label not in (0, 1):
                    raise ValueError("Метка должна быть 0 или 1")
                for name, index in numeric_columns:
                    value = row[index] if index is not None else ""
                    if not value:
                        missing[name] = missing.get(name, 0) + 1
                        if name in dataset_real.FEATURES:
                            raise ValueError(f"Пустое обязательное поле {name}")
                    numeric.append(dataset_real._as_float(value) if value else 0.0)
                category_values = []
                for name, index in category_columns:
                    value = row[index] if index is not None else ""
                    if not value:
                        missing[name] = missing.get(name, 0) + 1
                    category_values.append(value.strip().lower())
                raw_categories.append(category_values)
                labels.append(label)
                categories.append(row[attack_column].strip() if attack_column is not None else "")
            except (ValueError, IndexError) as error:
                raise ValueError(f"{path.name}, строка {reader.line_num}: {error}") from error
    if len(labels) < 3 or len(set(labels)) != 2:
        raise ValueError("Для анализа нужны оба класса и минимум три строки")
    values = np.frombuffer(numeric, dtype=np.float64).reshape(-1, len(numeric_columns))
    raw_categories = np.asarray(raw_categories)
    encoded = [raw_categories[:, i] == value for i, levels in enumerate(dataset_real.CATEGORIES.values())
               for value in levels]
    ratios = []
    for source, destination in (("sbytes", "dbytes"), ("spkts", "dpkts")):
        outgoing, incoming = values[:, dataset_real.FLOW_NUMERIC.index(source)], values[:, dataset_real.FLOW_NUMERIC.index(destination)]
        total = outgoing + incoming
        ratios.append(np.divide(incoming, total, out=np.zeros_like(total), where=total > 0))
    matrix = np.column_stack([values, *encoded, *ratios])
    target = np.asarray(labels, dtype=np.uint8)
    attacks = np.asarray(categories)
    for result in (matrix, target, attacks, raw_categories):
        result.setflags(write=False)
    with path.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    return matrix, target, attacks, fingerprint, missing, raw_categories


def group_indices(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Группирует строки с одинаковыми признаками без идентификаторов и меток."""
    return np.unique(matrix, axis=0, return_index=True, return_inverse=True, return_counts=True)[1:]


def cut_groups(groups, counts, target):
    """Выбирает ближайший к целевому размеру префикс целых групп."""
    boundary = int(np.abs(np.r_[0, np.cumsum(counts[groups])] - target).argmin())
    return groups[:boundary], groups[boundary:]


@lru_cache(maxsize=2)
def partition(path: Path, stamp: int, train_size: int, validation_size: int) -> tuple:
    """Создаёт стратифицированные группы train, validation и резерва."""
    x, y, attacks, fingerprint, _, _ = read_flow_table(path, stamp)
    representatives, membership, counts = group_indices(x)
    rng = np.random.default_rng(SPLIT_SEED)
    assignment = np.full(len(counts), -1, dtype=np.int8)
    control_size = min(CONTROL_SIZE, len(x) // 8)
    validation_size = min(validation_size, len(x) // 3)
    available = len(x) - control_size - validation_size
    if train_size > available:
        raise ValueError(f"Запрошено {train_size} строк train, доступно около {available}")
    strata = np.char.add(y[representatives].astype(str), attacks[representatives])
    for stratum in np.unique(strata):
        groups = rng.permutation(np.flatnonzero(strata == stratum))
        fraction = counts[groups].sum() / len(x)
        control, rest = cut_groups(groups, counts, control_size * fraction)
        validation, rest = cut_groups(rest, counts, validation_size * fraction)
        subset, _ = cut_groups(rest, counts, train_size * fraction) if train_size else (rest, None)
        assignment[subset] = 0
        assignment[validation] = 1
        assignment[control] = 2
    indices = [np.flatnonzero(assignment[membership] == split) for split in range(3)]
    if any(len(np.unique(y[index])) != 2 for index in indices[:2]):
        raise ValueError("Групповое разделение не сохранило оба класса")
    return x, y, attacks, fingerprint, indices


def training_parts(train_size: int, validation_size: int, feature_set: str) -> tuple:
    """Возвращает только train и validation с отпечатком конфигурации."""
    dataset_real.check_files("train")
    path = dataset_real.TRAIN
    x, y, attacks, fingerprint, indices = partition(path, path.stat().st_mtime_ns, train_size, validation_size)
    train, validation = indices[:2]
    digest = hashlib.sha256(f"{fingerprint}:{feature_set}:{SPLIT_SEED}".encode())
    for index in (train, validation):
        digest.update(index.astype("<i8").tobytes())
    columns = len(dataset_real.FEATURES) if feature_set == "legacy" else x.shape[1]
    metadata = {
        "id": digest.hexdigest(), "source_sha256": fingerprint, "split_seed": SPLIT_SEED,
        "train_size": len(train), "validation_size": len(validation), "reserved_size": len(indices[2]),
        "requested_train_size": train_size, "requested_validation_size": validation_size,
        "feature_set": feature_set,
    }
    return (np.ascontiguousarray(x[train, :columns].T), y[train]), (
        np.ascontiguousarray(x[validation, :columns].T), y[validation]), attacks[validation], metadata


def reserved_flows():
    """Читает фиксированный внутренний резерв для итоговой оценки."""
    path = dataset_real.TRAIN
    x, y, attacks, _, indices = partition(path, path.stat().st_mtime_ns, 0, 4096)
    for index in indices[2]:
        event = dict(zip(dataset_real.EXPANDED_FEATURES, map(float, x[index])))
        yield event, int(y[index]), str(attacks[index])
