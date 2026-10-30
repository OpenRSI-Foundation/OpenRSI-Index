"""Private volume export rejects unsafe artifacts and preserves recovery evidence."""

import errno
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.errors import InfrastructureError, RecoveryRequiredError, SetupError
from rsi_harness.runtime import desktop


@pytest.fixture(autouse=True)
def desktop_platform(monkeypatch):
    monkeypatch.setattr(desktop, "sys", SimpleNamespace(platform="darwin"))


def client_for(entries, mount_type="volume"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, kind, data in entries:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.size = len(data) if kind == tarfile.REGTYPE else 0
            entry.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
            archive.addfile(entry, io.BytesIO(data))
    container = SimpleNamespace(
        id="judge",
        reload=lambda: None,
        attrs={
            "Config": {
                "Labels": {
                    desktop.PRIVATE_LOG_LABEL: "volume",
                    "rsi-harness.role": "judge",
                }
            },
            "Mounts": [
                {"Destination": desktop.LOG_TARGET, "Type": mount_type, "RW": True}
            ],
        },
        get_archive=lambda path: (iter([buffer.getvalue()]), {}),
    )
    return SimpleNamespace(
        containers=SimpleNamespace(get=lambda _: container),
        info=lambda: {"OperatingSystem": "Docker Desktop"},
    )


def test_private_export(tmp_path):
    target = tmp_path / "logs"
    client = client_for([("verifier/reward.json", tarfile.REGTYPE, b'{"reward":1}')])
    desktop.export_private_logs(client, "judge", target)
    assert (target / "reward.json").read_bytes() == b'{"reward":1}'
    assert (target / "reward.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "name,kind",
    [
        ("/verifier/reward.json", tarfile.REGTYPE),
        ("verifier/../reward.json", tarfile.REGTYPE),
        ("verifier/link", tarfile.SYMTYPE),
        ("verifier/device", tarfile.CHRTYPE),
    ],
)
def test_unsafe_archive_preserves_previous_evidence(tmp_path, name, kind):
    target = tmp_path / "logs"
    target.mkdir()
    (target / "reward.json").write_text("previous")
    with pytest.raises(desktop.PrivateLogContentError):
        desktop.export_private_logs(client_for([(name, kind, b"bad")]), "judge", target)
    assert not (target / "reward.json").exists()
    previous = next(tmp_path.glob(".rejected-log-export-*/previous-export/reward.json"))
    assert previous.read_text() == "previous"
    assert not list(tmp_path.glob(".log-export-*"))
    assert (
        '"score": null' in (tmp_path / "logs.private-log-export-error.json").read_text()
    )


def test_bind_mount_is_not_private_volume(tmp_path):
    with pytest.raises(InfrastructureError, match="Linux volume"):
        desktop.export_private_logs(client_for([], "bind"), "judge", tmp_path / "logs")


def test_rejection_diagnostic_failure_retains_typed_recovery_authority(
    tmp_path, monkeypatch
):
    def fail_record(*_args):
        raise OSError(errno.ENOSPC, "storage unavailable")

    monkeypatch.setattr(desktop, "_record_content_rejection", fail_record)
    with pytest.raises(RecoveryRequiredError) as raised:
        desktop.export_private_logs(
            client_for([("verifier/link", tarfile.SYMTYPE, b"")]),
            "judge",
            tmp_path / "logs",
        )
    assert raised.value.code == "infrastructure_error"


def test_export_publish_failure_restores_directory(tmp_path, monkeypatch):
    target = tmp_path / "logs"
    (target / "evidence").mkdir(parents=True)
    (target / "evidence" / "old").write_text("keep")
    real_replace = desktop.os.replace

    def fail_publish(source, destination):
        if source.parent.name == "contents":
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(desktop.os, "replace", fail_publish)
    with pytest.raises(InfrastructureError, match="recovery_required"):
        desktop.export_private_logs(
            client_for([("verifier/evidence/new", tarfile.REGTYPE, b"new")]),
            "judge",
            target,
        )
    assert (target / "evidence" / "old").read_text() == "keep"


@pytest.mark.parametrize("directory", [False, True])
def test_export_replaces_max_length_basename(tmp_path, directory):
    target = tmp_path / "logs"
    name = "x" * 255
    relative = f"{name}/proof" if directory else name
    for contents in (b"previous", b"current"):
        desktop.export_private_logs(
            client_for([(f"verifier/{relative}", tarfile.REGTYPE, contents)]),
            "judge",
            target,
        )
        assert (target / relative).read_bytes() == contents
    assert not list(tmp_path.glob(".log-export-*"))


def test_partial_export_with_max_length_names_recovers(tmp_path, monkeypatch):
    from rsi_harness.runtime.production import ProductionRecoveryBackend

    target = tmp_path / "logs"
    names = ("a" * 255, "b" * 255)

    def archive(contents):
        return client_for(
            [(f"verifier/{name}", tarfile.REGTYPE, contents) for name in names]
        )

    desktop.export_private_logs(archive(b"previous"), "judge", target)
    # Controller output shares this destination and must survive re-export.
    (target / "controller.txt").write_text("keep")
    original = desktop.os.replace
    publications = 0

    def fail_second_publication(source, destination):
        nonlocal publications
        if source.parent.name == "contents":
            publications += 1
            if publications == 2:
                raise OSError(errno.ENOSPC, "injected publication failure")
        return original(source, destination)

    client = archive(b"current")
    with monkeypatch.context() as patch:
        patch.setattr(desktop.os, "replace", fail_second_publication)
        with pytest.raises(InfrastructureError, match="recovery_required"):
            desktop.export_private_logs(client, "judge", target)
    assert sorted((target / name).read_bytes() for name in names) == [
        b"current",
        b"previous",
    ]
    ProductionRecoveryBackend(client, None, None).collect_private_judge_logs(
        "judge", target
    )
    assert all((target / name).read_bytes() == b"current" for name in names)
    assert (target / "controller.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".log-export-*"))


def test_firewall_uses_daemon_namespaces_and_bounded_lock(monkeypatch):
    helper = desktop.DesktopHost(None)
    calls = []
    monkeypatch.setattr(
        helper, "_execute", lambda cmd, **kwargs: calls.append((cmd, kwargs))
    )
    helper.firewall(["iptables", "--wait", "-S", "DOCKER-USER"])
    assert calls == [
        (
            [
                "/usr/bin/nsenter",
                "-t",
                "1",
                "-m",
                "-n",
                "--",
                "/sbin/iptables",
                "--wait",
                "10",
                "-S",
                "DOCKER-USER",
            ],
            {"host": True},
        )
    ]
    with pytest.raises(SetupError):
        helper.firewall(["sh", "-c", "anything"])


@pytest.mark.parametrize("sandbox,operator_login", [(object(), None), (None, object())])
def test_desktop_refuses_linux_host_paths_before_execution(sandbox, operator_login):
    with pytest.raises(SetupError):
        desktop.validate_desktop_task(
            task_definition(sandbox=sandbox),
            SimpleNamespace(operator_login=operator_login),
        )


def test_desktop_accepts_api_or_lent_login():
    desktop.validate_desktop_task(
        task_definition(), SimpleNamespace(operator_login=None)
    )


def task_definition(*, sandbox=None, work_gpus=0, judge_gpus=0):
    return SimpleNamespace(
        sandbox=sandbox,
        gpu_requirement=SimpleNamespace(count=work_gpus),
        verifier=SimpleNamespace(gpu_count=judge_gpus),
    )


@pytest.mark.parametrize("work_gpus,judge_gpus", [(1, 0), (0, 1)])
def test_desktop_refuses_gpu(work_gpus, judge_gpus):
    with pytest.raises(SetupError, match="CPU-only"):
        desktop.validate_desktop_task(
            task_definition(work_gpus=work_gpus, judge_gpus=judge_gpus), None
        )


@pytest.mark.parametrize(
    "mode,path", [("full-rootfs", "/"), ("other", "/work"), ("split-workdir", "/")]
)
def test_desktop_refuses_unsupported_workspace(mode, path):
    with pytest.raises(SetupError, match="split WORKDIR"):
        desktop.validate_desktop_images(
            SimpleNamespace(rootfs_snapshot_mode=mode, workdir=path)
        )


@pytest.mark.parametrize("daemon", ["Colima", "OrbStack", "Ubuntu Linux", ""])
def test_macos_refuses_non_desktop_daemon(daemon):
    with pytest.raises(SetupError, match="requires the Docker Desktop"):
        desktop.validate_local_daemon(
            SimpleNamespace(info=lambda: {"OperatingSystem": daemon})
        )


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:2375", "https://remote:2376", "http+docker://ssh"]
)
def test_macos_refuses_non_unix_transport_even_for_desktop_daemon(endpoint):
    client = SimpleNamespace(
        info=lambda: {"OperatingSystem": "Docker Desktop"},
        api=SimpleNamespace(base_url=endpoint),
    )
    with pytest.raises(SetupError, match="Unix socket"):
        desktop.validate_local_daemon(client)


@pytest.mark.parametrize(
    "names", [("A/x", "a/y"), ("café", "cafe\u0301"), ("REWARD.json", "reward.json")]
)
def test_export_rejects_filesystem_aliases(tmp_path, names):
    client = client_for([(f"verifier/{name}", tarfile.REGTYPE, b"x") for name in names])
    with pytest.raises(desktop.PrivateLogContentError, match="colliding"):
        desktop.export_private_logs(client, "judge", tmp_path / "logs")


def test_content_limit_counts_file_bytes_not_tar_padding(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "MAX_LOG_BYTES", 3)
    desktop.export_private_logs(
        client_for([("verifier/file", tarfile.REGTYPE, b"abc")]),
        "judge",
        tmp_path / "logs",
    )
    with pytest.raises(desktop.PrivateLogContentError, match="contents exceed"):
        desktop.export_private_logs(
            client_for([("verifier/file", tarfile.REGTYPE, b"abcd")]),
            "judge",
            tmp_path / "logs",
        )


@pytest.mark.parametrize(
    "error_number,terminal",
    [(errno.ENAMETOOLONG, True), (errno.ENOSPC, False), (errno.EIO, False)],
)
def test_extraction_distinguishes_path_limits_from_storage_faults(
    tmp_path, monkeypatch, error_number, terminal
):
    original = Path.open

    def open_path(path, *args, **kwargs):
        if path.name == "payload" and path.parent.name == "contents":
            raise OSError(error_number, "simulated extraction failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    expected = desktop.PrivateLogContentError if terminal else RecoveryRequiredError
    with pytest.raises(expected):
        desktop.export_private_logs(
            client_for([("verifier/payload", tarfile.REGTYPE, b"x")]),
            "judge",
            tmp_path / "logs",
        )
    assert (tmp_path / "logs.private-log-export-error.json").exists() is terminal


def test_export_flushes_nested_directory_entries_before_return(tmp_path, monkeypatch):
    flushed = []
    actual = desktop.fsync_directory

    def flush(path):
        flushed.append(path)
        actual(path)

    monkeypatch.setattr(desktop, "fsync_directory", flush)
    target = tmp_path / "logs"
    desktop.export_private_logs(
        client_for([("verifier/nested/payload", tarfile.REGTYPE, b"x")]),
        "judge",
        target,
    )
    assert any(path.name == "nested" for path in flushed)
    assert any(path.name == "contents" for path in flushed)
    assert flushed[-2:] == [target, target.parent]


def test_linux_image_label_cannot_enable_private_log_recovery(monkeypatch):
    monkeypatch.setattr(desktop, "sys", SimpleNamespace(platform="linux"))
    client = client_for([], mount_type="bind")
    assert (
        desktop.has_private_log_volume(client, client.containers.get("judge")) is False
    )
    with pytest.raises(InfrastructureError, match="Desktop backend"):
        desktop.attest_private_log_volume(client, "judge")


@pytest.mark.parametrize("uid,gid", [(0, 0), (1001, 1002)])
def test_private_log_owner_uses_actual_judge_identity(tmp_path, uid, gid):
    from rsi_harness.models import AgentRunResult, ContainerRef
    from rsi_harness.runtime.docker import DockerContainerRuntime

    runtime = object.__new__(DockerContainerRuntime)
    runtime._client = client_for([])
    calls = []

    def execute(container, command, **options):
        calls.append((command, options))
        if len(calls) == 1:
            return AgentRunResult(exit_code=0, output=f"{uid}\n{gid}\n")
        return AgentRunResult(exit_code=0, output=f"700:{uid}:{gid}\n")

    runtime.exec = execute
    runtime._initialize_private_judge_logs(
        ContainerRef(container_id="judge", role="judge")
    )
    assert "user" not in calls[0][1]  # Docker resolves the configured Judge user.
    assert calls[0][1]["timeout_seconds"] == 10
    assert calls[1][1] == {"timeout_seconds": 10.0, "user": "root"}
    assert calls[1][0][-2:] == (desktop.LOG_TARGET, f"{uid}:{gid}")
    assert "chmod 0700" in calls[1][0][2]
    assert "chown -R" not in calls[1][0][2]


@pytest.mark.parametrize(
    "identity", ["judge\ngroup", "0\n0\nextra", "0\n4294967295", "", "0\n-1"]
)
def test_private_log_owner_rejects_unproven_identity(identity):
    from rsi_harness.models import AgentRunResult, ContainerRef
    from rsi_harness.runtime.docker import DockerContainerRuntime

    runtime = object.__new__(DockerContainerRuntime)
    runtime._client = client_for([])
    calls = []

    def execute(container, command, **options):
        calls.append(command)
        return AgentRunResult(exit_code=0, output=identity)

    runtime.exec = execute
    with pytest.raises(InfrastructureError, match="owner could not be resolved"):
        runtime._initialize_private_judge_logs(
            ContainerRef(container_id="judge", role="judge")
        )
    assert len(calls) == 1  # Never send an unproven identity into a privileged command.


def test_private_log_owner_requires_exact_mount_and_result():
    from rsi_harness.models import AgentRunResult, ContainerRef
    from rsi_harness.runtime.docker import DockerContainerRuntime

    runtime = object.__new__(DockerContainerRuntime)
    runtime._client = client_for([], "bind")
    calls = []

    def execute(container, command, **options):
        calls.append(command)
        return AgentRunResult(
            exit_code=0, output="1001\n1002" if len(calls) == 1 else "777:0:0"
        )

    runtime.exec = execute
    with pytest.raises(InfrastructureError, match="Linux volume"):
        runtime._initialize_private_judge_logs(
            ContainerRef(container_id="judge", role="judge")
        )
    assert not calls
    runtime._client = client_for([])
    with pytest.raises(InfrastructureError, match="ownership is unproven"):
        runtime._initialize_private_judge_logs(
            ContainerRef(container_id="judge", role="judge")
        )
