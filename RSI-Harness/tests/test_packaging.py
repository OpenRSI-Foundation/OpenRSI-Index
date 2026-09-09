from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib import metadata, resources
from pathlib import Path

import rsi_loop


def test_retired_sforge_namespace_is_not_importable(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "raise SystemExit(0 if "
                "importlib.util.find_spec('sforge') is None else 1)"
            ),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_rsi_harness_distribution_owns_complete_rsi_loop_package() -> None:
    root = Path(__file__).resolve().parents[1]
    package_path = Path(rsi_loop.__file__).resolve()

    assert package_path.is_relative_to((root / "src" / "rsi_loop").resolve())
    assert not (root / "src" / "sforge").exists()
    assert importlib.util.find_spec("sforge") is None

    requirements = metadata.requires("rsi-harness") or ()
    assert not any("edgebench" in item.lower() for item in requirements)
    assert not any("git+" in item.lower() for item in requirements)

    scripts = {
        entry.name: (entry.value, entry.dist.name)
        for entry in metadata.entry_points(group="console_scripts")
        if entry.name in {"rsi-harness", "rsi-submit", "sforge", "rsi-loop"}
    }
    assert scripts == {
        "rsi-harness": ("rsi_harness.cli:app", "rsi-harness"),
    }

    package = resources.files("rsi_loop")
    assert package.joinpath("LICENSE").is_file()
    assert package.joinpath("NOTICE").is_file()
    assert package.joinpath(
        "visualizer", "templates", "trajectory.html"
    ).is_file()

    cluster_package = resources.files("rsi_harness.cluster.bluevela")
    assert cluster_package.joinpath("profile.toml").is_file()
