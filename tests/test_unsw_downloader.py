"""Локальные тесты загрузки UNSW-NB15 без сетевых запросов."""

import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from traffic_data import dataset_real
from traffic_data import download_unsw


class DownloaderTests(unittest.TestCase):
    """Проверяет ссылки, валидацию CSV и независимые train/test."""

    def test_testing_source_exists_in_small_repository(self):
        """Тестовую часть запрашиваем не из каталога, где она отсутствует."""
        self.assertIn("UNSW-NB15-small", download_unsw.SOURCES[dataset_real.TEST.name][0])

    def test_download_and_validate_with_mock(self):
        """Скачанный файл проверяется до сохранения по целевому пути."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / dataset_real.TEST.name
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(dataset_real.FEATURES) + ["label"])
            writer.writeheader()
            writer.writerow({**{name: "2" for name in dataset_real.FEATURES}, "label": "1"})
            class Response(io.BytesIO):
                """Имитирует поток HTTP."""
                def __enter__(self):
                    """Возвращает себя."""
                    return self
                def __exit__(self, *_args):
                    """Закрывает буфер."""
                    self.close()
            with patch.object(download_unsw.urllib.request, "urlopen",
                              return_value=Response(buffer.getvalue().encode())):
                download_unsw.download_file(target)
            download_unsw.validate_csv(target)
            self.assertTrue(target.exists())

    def test_invalid_download_does_not_replace_target(self):
        """Неправильный ответ сервера не должен оставлять псевдо-CSV."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / dataset_real.TEST.name
            with patch.object(download_unsw.urllib.request, "urlopen",
                              return_value=io.BytesIO(b"<html>not dataset</html>")):
                with self.assertRaises(RuntimeError):
                    download_unsw.download_file(target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_train_does_not_require_test_file(self):
        """Валидация может работать до скачивания отложенной тестовой части."""
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.csv"
            train.write_text("sample", encoding="utf-8")
            with patch.object(dataset_real, "TRAIN", train), patch.object(
                    dataset_real, "TEST", Path(directory) / "missing.csv"):
                dataset_real.check_files("train")
                with self.assertRaises(FileNotFoundError):
                    dataset_real.check_files("test")


if __name__ == "__main__":
    unittest.main()
