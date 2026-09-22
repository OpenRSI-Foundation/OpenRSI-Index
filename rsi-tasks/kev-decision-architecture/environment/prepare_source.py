"""Export only library source and its dependency/license metadata for the image."""
from pathlib import Path
import shutil
import sys


def export_source(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "kev").mkdir()
    for path in sorted((source / "kev").glob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unexpected source entry: {path}")
        shutil.copy2(path, destination / "kev" / path.name)
    for name in ("LICENSE", "README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(source / name, destination / name)


if __name__ == "__main__":
    export_source(*sys.argv[1:])
