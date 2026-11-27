"""Скачивание обучающей и тестовой частей открытого UNSW-NB15.

Официальное описание набора: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Файлы скачиваются из публичных репозиториев Hugging Face, не через LLM.
"""

import csv
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from traffic_data.dataset_real import FEATURES, TEST, TRAIN

# Тестовый CSV отсутствует в основном репозитории Mouwiya/UNSW-NB15,
# но опубликован в Mouwiya/UNSW-NB15-small (проверено по списку файлов 21.11.2026).
BASE_SMALL = "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/"
BASE_TRAIN = "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/"
SOURCES = {
    TRAIN.name: (BASE_SMALL + TRAIN.name, BASE_TRAIN + TRAIN.name),
    TEST.name: (BASE_SMALL + TEST.name,),
}
REQUIRED = set(FEATURES) | {"label"}


def validate_csv(path: Path) -> None:
    """Проверяет наличие необходимых полей и как минимум одной строки данных."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"В {path.name} отсутствуют поля: {', '.join(sorted(missing))}")
        row = next(reader, None)
        if row is None or row.get("label") not in ("0", "1"):
            raise ValueError(f"Файл {path.name} пуст или имеет неверную разметку")
        for feature in FEATURES:
            try:
                float(row[feature])
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Поле {feature} в {path.name} не является числом") from error


def download_file(path: Path) -> None:
    """Скачивает CSV из проверенного зеркала и заменяет файл только после проверки."""
    path.parent.mkdir(parents=True, exist_ok=True)
    problems = []
    for url in SOURCES[path.name]:
        temporary = None
        try:
            print(f"Загрузка {path.name} из {url.split('/datasets/')[-1].split('/resolve/')[0]} ...", flush=True)
            with tempfile.NamedTemporaryFile(prefix=".unsw-", suffix=".csv", dir=path.parent,
                                             delete=False) as target:
                temporary = Path(target.name)
                request = urllib.request.Request(url, headers={"User-Agent": "NetworkEvolutionLab/1.1"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
            validate_csv(temporary)
            os.replace(temporary, path)
            print(f"Сохранено: {path}", flush=True)
            return
        except (OSError, ValueError, urllib.error.URLError) as error:
            problems.append(f"{type(error).__name__}: {error}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Не удалось скачать {path.name}. {'; '.join(problems)}")


def main() -> int:
    """Проверяет ранее скачанные файлы и загружает только отсутствующие."""
    for path in (TRAIN, TEST):
        try:
            if path.is_file():
                validate_csv(path)
                print(f"Уже загружен и проверен: {path.name}", flush=True)
            else:
                download_file(path)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Ошибка: {error}", flush=True)
            print("Ручная загрузка: https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/tree/main")
            print("Помести официальные файлы с теми же именами в data/unsw/.")
            return 1
    print("UNSW-NB15 готов: обе выборки доступны.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
