"""Apply checksum-pinned backend compatibility repairs during image build."""
import importlib.metadata
import json
from pathlib import Path
import sysconfig

from prepare_assets import sha


def main():
    manifest = Path(__file__).with_name("backend_patches.json")
    for patch in json.loads(manifest.read_text()):
        assert importlib.metadata.version(patch["distribution"]) == patch["version"]
        target = Path(sysconfig.get_paths()["purelib"]) / patch["path"]
        assert sha(target) == patch["before_sha256"]
        text = target.read_text()
        assert text.count(patch["old"]) == 1
        target.write_text(text.replace(patch["old"], patch["new"]))
        assert sha(target) == patch["after_sha256"]


if __name__ == "__main__":
    main()
