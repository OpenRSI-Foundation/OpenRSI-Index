"""Offline contracts. Cloud acceptance uses fixtures/tasks/modal-native."""

from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.modal_runtime.config import AssetBind, ModalRunConfig, validate_task
from rsi_harness.modal_runtime.phases import copy_regular_tree
from rsi_harness.models import CompileOptions
from rsi_harness.task.compiler import HarborTaskCompiler

FIXTURE = Path(__file__).parent / "fixtures/tasks/modal-native"


def config(**changes):
    return ModalRunConfig(
        environment_name="test",
        profile="test",
        volume_name="test",
        max_compute_seconds=90000,
        **changes,
    )


def task():
    return HarborTaskCompiler().compile(FIXTURE, CompileOptions(agent_name="oracle"))


@pytest.mark.parametrize(
    "changes",
    [{"gpu": "H100:2"}, {"segment_seconds": 86400}, {"max_compute_seconds": 0}],
)
def test_resource_limits(changes):
    values = config().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError):
        ModalRunConfig(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"clean_image": False},
        {"visibility": "public"},
        {"model_access": True},
        {"gpu_count": 2},
    ],
)
def test_incompatible_judge_refused_before_allocation(change):
    t = task()
    t = t.model_copy(update={"verifier": t.verifier.model_copy(update=change)})
    with pytest.raises(SetupError):
        validate_task(t, config())


@pytest.mark.parametrize(
    "target", ["/", "/workspace/nested", "/etc", "/opt/rsi-isolate/evil"]
)
def test_assets_cannot_shadow_runtime(target):
    with pytest.raises((SetupError, ValidationError)):
        validate_task(
            task(),
            config(
                assets=(
                    AssetBind(
                        source=Path("/asset-store/data"),
                        target=PurePosixPath(target),
                        phase="judge",
                    ),
                )
            ),
        )


def test_oracle_no_provider_route():
    with pytest.raises(SetupError):
        validate_task(task(), config(provider_urls=("https://example.com",)))


@pytest.mark.parametrize("kind", ["file-link", "dir-link", "hardlink", "fifo"])
def test_candidate_cannot_import_other_files(tmp_path, kind):
    import os

    source = tmp_path / "source"
    source.mkdir()
    private = tmp_path / "private"
    private.write_text("not a candidate")
    if kind == "file-link":
        (source / "link").symlink_to(private)
    elif kind == "dir-link":
        (source / "link").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "hardlink":
        os.link(private, source / "link")
    else:
        os.mkfifo(source / "fifo")
    with pytest.raises(InfrastructureError):
        copy_regular_tree(source, tmp_path / "snapshot")


def test_snapshot_digest_tracks_content_and_executable_bit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    p = source / "candidate"
    p.write_text("v1")
    first = copy_regular_tree(source, tmp_path / "a")
    assert first == copy_regular_tree(source, tmp_path / "b")
    p.write_text("v2")
    second = copy_regular_tree(source, tmp_path / "c")
    assert first != second
    p.chmod(0o755)
    assert second != copy_regular_tree(source, tmp_path / "d")
    assert (tmp_path / "a/candidate").read_text() == "v1"


def test_cli_exposes_modal_without_importing_sdk():
    from typer.testing import CliRunner

    from rsi_harness.cli import app

    result = CliRunner().invoke(app, ["modal", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--max-submissions" in result.output


def test_volume_artifacts_preserve_full_bytes_without_hardlinks(tmp_path):
    from rsi_harness.modal_runtime.artifacts import ModalArtifactWriter

    source, target = tmp_path / "source", tmp_path / "target"
    content = b"complete output\n" * 100000
    source.write_bytes(content)
    writer = object.__new__(ModalArtifactWriter)
    writer._hardlink_output(source, target)
    assert target.read_bytes() == content
    assert source.stat().st_ino != target.stat().st_ino
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(OSError):
        writer._hardlink_output(link, target)
    assert target.read_bytes() == content
