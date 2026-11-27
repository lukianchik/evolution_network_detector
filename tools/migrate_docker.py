"""Однократный перенос старых результатов и квот в Linux volumes без перезаписи."""

from pathlib import Path
import shutil


def copy_missing(source: Path, target: Path, *, whole: bool = False) -> int:
    """Копирует только новые объекты; не следует по символическим ссылкам."""
    target.mkdir(parents=True, exist_ok=True)
    if not source.is_dir() or (whole and any(target.iterdir())):
        return 0
    count = 0
    for item in sorted(source.iterdir()):
        dest = target / item.name
        if item.is_symlink() or dest.exists() or dest.is_symlink():
            continue
        if item.is_dir():
            # Не импортируем поддеревья с symlink, чтобы не раскрыть сторонние файлы.
            if any(path.is_symlink() for path in item.rglob('*')):
                raise ValueError(f'Символическая ссылка в {item}')
            shutil.copytree(item, dest)
        elif item.is_file():
            shutil.copy2(item, dest)
        else:
            continue
        count += 1
    return count


def main() -> None:
    """Переносит архивы и состояние только при остановленном worker."""
    print('Импортировано объектов results:', copy_missing(Path('/legacy/results'), Path('/app/results')))
    print('Импортировано объектов state:', copy_missing(Path('/legacy/state'), Path('/app/.state'), whole=True))


if __name__ == '__main__':
    main()
