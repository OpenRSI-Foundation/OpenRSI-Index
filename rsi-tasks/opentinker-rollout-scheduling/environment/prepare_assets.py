"""Build-only preparation; no repository entrypoint is executed by this script."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target, digest):
    with urllib.request.urlopen(url, timeout=120) as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
    if sha(target) != digest:
        raise RuntimeError("asset_digest: " + target.name)


def apply_patches():
    for patch in json.loads((HERE / "patches.json").read_text()):
        target = Path("/opt/src") / patch["path"]
        text = target.read_text()
        assert text.count(patch["old"]) == 1
        target.write_text(text.replace(patch["old"], patch["new"]))


def main():
    sources = json.loads((HERE / "vendor/sources.json").read_text())
    for name, spec in sources.items():
        archive = HERE / "vendor" / (name + ".tar.gz")
        if sha(archive) != spec["packaged_archive_sha256"]:
            raise RuntimeError("source_archive_digest")
        destination = Path("/opt/src") / name
        destination.mkdir(parents=True)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts or not member.isfile():
                    continue
                # Include only package code/data and licenses, never upstream example configurations.
                if parts[0] not in {name, "LICENSE", "LICENSE.md", "NOTICE"}:
                    continue
                if any(part in {"..", "__pycache__"} for part in parts):
                    raise ValueError("archive_path")
                if name == "opentinker" and ("config" in parts or "configs" in parts):
                    continue
                target = destination.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    shutil.copyfileobj(source.extractfile(member), stream)
    # A fixed .pth exposes only the three pinned source packages, never WORKDIR.
    import sysconfig
    (Path(sysconfig.get_paths()["purelib"]) / "opentinker_sources.pth").write_text(
        "/opt/src/opentinker\n/opt/src/verl\n/opt/src/alfworld\n")
    model = Path("/opt/models/qwen2.5-3b-instruct")
    model.mkdir(parents=True)
    prefix = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/aa8e72537993ba99e69dfaafa59ed015b17504d1/"
    for name, spec in json.loads((HERE / "model_identity.json").read_text()).items():
        download(prefix + name, model / name, spec["sha256"])
        if (model / name).stat().st_size != spec["size"]:
            raise RuntimeError("model_size")
    archive = Path("/tmp/alfworld-games.zip")
    download("https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip", archive,
             "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf")
    root = Path("/opt/data/alfworld")
    root.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            parts = Path(info.filename).parts
            if info.is_dir():
                continue
            if not parts or parts[0] != "json_2.1.1" or ".." in parts:
                raise RuntimeError("game_archive_path")
            target = root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    archive.unlink()
    (root / "logic").mkdir()
    for name in ("alfred.pddl", "alfred.twl2"):
        shutil.copyfile(Path("/opt/src/alfworld/alfworld/data") / name, root / "logic" / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-patches-only", action="store_true")
    args = parser.parse_args()
    if args.apply_patches_only:
        apply_patches()
    else:
        main()
