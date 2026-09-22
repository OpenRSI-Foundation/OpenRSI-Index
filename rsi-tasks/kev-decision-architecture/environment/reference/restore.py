"""Restore reference code, preserving checkpoints and changed source in backups."""
from pathlib import Path
import shutil
import uuid


def files(path):
    if path.is_symlink():
        raise ValueError(f"refusing symlink code path: {path}")
    if path.is_file():
        return {"": path.read_bytes()}
    result = {}
    for item in path.rglob("*"):
        if "__pycache__" in item.parts:
            continue
        if item.is_symlink():
            raise ValueError(f"refusing symlink code path: {item}")
        if item.is_file():
            result[item.relative_to(path).as_posix()] = item.read_bytes()
    return result


def restore(reference, candidate, experiments):
    reference, candidate, experiments = map(Path, (reference, candidate, experiments))
    for root in (reference, candidate, experiments):
        if root.is_symlink():
            raise ValueError(f"refusing symlink root: {root}")
    candidate.mkdir(parents=True, exist_ok=True)
    changed = []
    for name in ("kev", "entrypoint.py", "train.py"):
        source, target = reference / name, candidate / name
        if not source.exists():
            raise FileNotFoundError(source)
        if target.is_symlink():
            raise ValueError(f"refusing symlink code path: {target}")
        if not target.exists() or files(target) != files(source):
            changed.append((source, target))
    backup = None
    for source, target in changed:
        if target.exists():
            if backup is None:
                backup = experiments / ("reference-code-backup-" + uuid.uuid4().hex)
                backup.mkdir(parents=True)
            shutil.move(str(target), backup / target.name)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    if backup is not None:
        print(f"Previous source preserved at {backup}")
    print("Reference code restored. Existing checkpoints and other experiments are unchanged.")
    print("Train in Work with: python /workspace/candidate/train.py --out /workspace/candidate/checkpoint")
    print("If that output exists, choose a fresh --out path. Judge never trains.")


if __name__ == "__main__":
    restore("/opt/kev-reference", "/workspace/candidate", "/workspace/experiments")
