"""Проверяет качество CSV UNSW без передачи тестовой разметки поиску."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from traffic_data import dataset_real
from traffic_data.sampling import group_indices, read_flow_table, training_parts


def summarize(path):
    """Собирает распределения, пропуски, выбросы и коллизии признаков."""
    x, y, categories, fingerprint, missing, raw_categories = read_flow_table(path, path.stat().st_mtime_ns)
    _, groups, sizes = group_indices(x)
    mixed = np.bincount(groups, weights=y)
    quantiles = np.percentile(x, [0, 25, 50, 75, 99, 100], axis=0)
    summary = {"rows": len(x), "sha256": fingerprint, "classes": Counter(map(int, y)),
               "attack_categories": Counter(map(str, categories)), "missing": missing,
               "duplicate_feature_rows": int(len(x) - len(sizes)), "largest_group": int(max(sizes)),
               "conflicting_label_groups": int(np.count_nonzero((mixed > 0) & (mixed < sizes))),
               "categories": {name: dict(Counter(raw_categories[:, i]).most_common())
                              for i, name in enumerate(dataset_real.CATEGORIES)}, "features": {}}
    for index, name in enumerate(dataset_real.EXPANDED_FEATURES):
        values = quantiles[:, index]
        q1, q3 = values[1], values[3]
        width = q3 - q1
        outliers = None
        if width:
            outliers = int(np.count_nonzero((x[:, index] < q1 - 1.5 * width) |
                                           (x[:, index] > q3 + 1.5 * width)))
        summary["features"][name] = {"min_q25_median_q75_q99_max": values.tolist(),
                                     "iqr_outliers": outliers}
    return x, summary


def audit(include_test=False):
    """Измеряет сдвиг распределений только в отдельном режиме итогового аудита."""
    train, training = summarize(dataset_real.TRAIN)
    result = {"train": training, "allowed_features": dataset_real.EXPANDED_FEATURES,
              "excluded_columns": ["id", "label", "attack_cat"],
              "grouping": "exact equality of detector features; no host or session keys",
              "samples": [training_parts(size, 4096, "expanded")[3] for size in (10000, 20000, 50000, 0)]}
    if include_test:
        test, result["test"] = summarize(dataset_real.TEST)
        _, groups, sizes = group_indices(np.vstack([train, test]))
        by_train = np.bincount(groups[:len(train)], minlength=len(sizes))
        result["test_rows_with_matching_train_features"] = int(np.count_nonzero(by_train[groups[len(train):]]))
        shift = {}
        for index, name in enumerate(dataset_real.EXPANDED_FEATURES):
            a, b = np.sort(train[:, index]), np.sort(test[:, index])
            grid = np.union1d(a, b)
            shift[name] = round(float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                                                   - np.searchsorted(b, grid, side="right") / len(b)))), 6)
        result["ks_distance"] = dict(sorted(shift.items(), key=lambda pair: -pair[1]))
    return result


def main():
    """Записывает отчёт качества данных без отдельных событий."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-test", action="store_true", help="После фиксации правила и конфигурации")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.include_test), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
