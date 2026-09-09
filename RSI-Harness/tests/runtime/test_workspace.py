from __future__ import annotations

import errno
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import docker
import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import ImagePlan, RootfsSnapshotMode, RunPaths
from rsi_harness.runtime.workspace import WorkspaceManager
from tests.factories import make_run_plan

BASE_REF = f"base@sha256:{'a' * 64}"
WORK_REF = f"work@sha256:{'b' * 64}"
BASE_DIGEST = f"sha256:{'a' * 64}"
WORK_DIGEST = f"sha256:{'b' * 64}"


class CopyingHelperContainers:
    def __init__(self, source: Path, *, reject_symlink: bool = False) -> None:
        self.source = source
        self.reject_symlink = reject_symlink
        self.runs: list[dict[str, Any]] = []

    def run(self, image: str, command: list[str], **kwargs: Any) -> bytes:
        self.runs.append({"image": image, "command": command, **kwargs})
        if self.reject_symlink:
            raise RuntimeError("image WORKDIR is a symlink")
        volumes = kwargs["volumes"]
        [(raw_target, mount)] = volumes.items()
        assert mount == {"bind": kwargs["environment"]["RSI_INIT_TARGET"], "mode": "rw"}
        target = Path(raw_target)
        if "find" in " ".join(command):
            for child in target.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            return b""
        if "managed workspace is a symlink" in " ".join(command):
            target.chmod(0o777)
            return b""
        shutil.copytree(
            self.source,
            target,
            symlinks=True,
            copy_function=shutil.copy2,
            dirs_exist_ok=True,
        )
        target.chmod(0o777)
        stat = self.source.stat()
        return f"{stat.st_uid}:{stat.st_gid}\n".encode()


class CopyingHelperClient:
    def __init__(self, source: Path, *, reject_symlink: bool = False) -> None:
        self.containers = CopyingHelperContainers(
            source, reject_symlink=reject_symlink
        )


def managed_plan(tmp_path: Path, *, work_user: str | None = None):
    data_root = tmp_path / "engine-data"
    run_root = data_root / "runs" / "run-7"
    plan = make_run_plan(tmp_path).model_copy(
        update={
            "paths": RunPaths(
                root=run_root,
                workspace=run_root / "workspace",
                logs=run_root / "logs",
            ),
            "images": ImagePlan(
                base_ref=BASE_REF,
                work_ref=WORK_REF,
                judge_ref=BASE_REF,
                workdir=PurePosixPath("/workspace"),
                rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
                work_user=work_user,
                base_digest=BASE_DIGEST,
                work_digest=WORK_DIGEST,
                judge_digest=BASE_DIGEST,
            ),
        }
    )
    return data_root, plan


def source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "image-workdir"
    source.mkdir(mode=0o751)
    executable = source / "run.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o751)
    payload = source / "payload.txt"
    payload.write_text("from clean base\n")
    (source / "payload-link").symlink_to("payload.txt")
    return source


def test_initialize_copies_clean_base_once_and_reuses_verified_manifest(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    source = source_tree(tmp_path)
    client = CopyingHelperClient(source)
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )

    first = manager.initialize()
    second = manager.initialize()

    assert first == second == plan.paths.workspace
    assert len(client.containers.runs) == 1
    helper = client.containers.runs[0]
    assert helper["image"] == BASE_REF
    assert helper["network_mode"] == "none"
    assert helper["user"] == "root"
    assert helper["remove"] is True
    assert helper["working_dir"] == "/"
    assert helper["privileged"] is False
    assert helper["entrypoint"] == ["/bin/sh", "-c"]
    assert helper["cap_drop"] == ["ALL"]
    assert helper["cap_add"] == ["CHOWN", "DAC_OVERRIDE", "FOWNER"]
    helper_target = helper["environment"]["RSI_INIT_TARGET"]
    assert helper["volumes"] == {
        str(plan.paths.workspace): {"bind": helper_target, "mode": "rw"}
    }
    command_text = " ".join(helper["command"])
    assert "cp -a --reflink=auto" in command_text
    assert str(plan.workdir) not in helper["volumes"].values()

    assert (first / "payload.txt").read_text() == "from clean base\n"
    assert (first / "payload-link").is_symlink()
    assert os.readlink(first / "payload-link") == "payload.txt"
    assert (first / "run.sh").stat().st_mode & 0o777 == 0o751
    assert first.stat().st_mode & 0o777 == 0o777
    assert (first / "payload.txt").stat().st_uid == source.stat().st_uid
    assert (first / "payload.txt").stat().st_gid == source.stat().st_gid
    assert source.stat().st_mode & 0o777 == 0o751

    manifest = json.loads(manager.manifest_path.read_text())
    assert manifest == {
        "completed": True,
        "gid": source.stat().st_gid,
        "image_digest": BASE_DIGEST,
        "schema_version": 1,
        "uid": source.stat().st_uid,
        "workdir": "/workspace",
    }
    assert not tuple(manager.manifest_path.parent.glob("*.tmp"))


def test_reuse_rejects_manifest_that_no_longer_proves_fixed_image(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    client = CopyingHelperClient(source_tree(tmp_path))
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )
    manager.initialize()
    changed_plan = plan.model_copy(
        update={
            "images": plan.images.model_copy(
                update={"base_digest": "sha256:different"}
            )
        }
    )

    with pytest.raises(SetupError, match="manifest.*image digest"):
        WorkspaceManager(
            client,
            plan=changed_plan,
            run_id="run-7",
            data_root=data_root,
        ).initialize()

    assert len(client.containers.runs) == 1


def test_incomplete_manifest_is_not_reinitialized_over_partial_workspace(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    client = CopyingHelperClient(source_tree(tmp_path))
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )
    plan.paths.workspace.mkdir(parents=True)
    manager.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_digest": BASE_DIGEST,
                "workdir": "/workspace",
                "uid": None,
                "gid": None,
                "completed": False,
            }
        )
    )

    with pytest.raises(SetupError, match="incomplete.*cleanup"):
        manager.initialize()

    assert client.containers.runs == []


def test_image_workdir_symlink_is_rejected_by_isolated_helper(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    client = CopyingHelperClient(source_tree(tmp_path), reject_symlink=True)

    with pytest.raises(SetupError, match="WORKDIR.*symlink"):
        WorkspaceManager(
            client, plan=plan, run_id="run-7", data_root=data_root
        ).initialize()


@pytest.mark.parametrize("unsafe_path", ("workspace", "root"))
def test_engine_workspace_paths_outside_data_root_are_rejected(
    tmp_path, unsafe_path
):
    data_root, plan = managed_plan(tmp_path)
    paths = plan.paths.model_copy(
        update={unsafe_path: tmp_path / "outside" / unsafe_path}
    )
    plan = plan.model_copy(update={"paths": paths})
    client = CopyingHelperClient(source_tree(tmp_path))

    with pytest.raises(SetupError, match="outside.*data root"):
        WorkspaceManager(
            client, plan=plan, run_id="run-7", data_root=data_root
        ).initialize()

    assert client.containers.runs == []


def test_workspace_is_retained_by_default_and_deleted_only_explicitly(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    client = CopyingHelperClient(source_tree(tmp_path))
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )
    workspace = manager.initialize()

    assert manager.retain() == workspace
    assert workspace.exists()
    assert manager.manifest_path.exists()

    manager.delete()

    assert not workspace.exists()
    assert not manager.manifest_path.exists()
    assert len(client.containers.runs) == 2
    cleanup = client.containers.runs[-1]
    assert cleanup["image"] == BASE_REF
    assert cleanup["network_mode"] == "none"
    cleanup_target = cleanup["environment"]["RSI_INIT_TARGET"]
    assert cleanup["entrypoint"] == ["/bin/sh", "-c"]
    assert cleanup["cap_drop"] == ["ALL"]
    assert cleanup["cap_add"] == ["CHOWN", "DAC_OVERRIDE", "FOWNER"]
    assert cleanup["volumes"] == {
        str(workspace): {"bind": cleanup_target, "mode": "rw"}
    }
    assert "find" in " ".join(cleanup["command"])


def test_judge_handoff_fails_closed_when_posix_acl_removal_is_unsupported(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rsi_harness.runtime.workspace as workspace_module

    data_root, plan = managed_plan(tmp_path)
    client = CopyingHelperClient(source_tree(tmp_path))
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )
    manager.initialize()

    def reject_acl_removal(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "ACL xattrs unsupported")

    monkeypatch.setattr(workspace_module.os, "removexattr", reject_acl_removal)

    with pytest.raises(SetupError, match="POSIX ACL.*unproven"):
        manager.attest_for_judge()


def test_host_workspace_symlink_is_never_followed(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    plan.paths.workspace.parent.mkdir(parents=True)
    plan.paths.workspace.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SetupError, match="symlink"):
        WorkspaceManager(
            CopyingHelperClient(source_tree(tmp_path)),
            plan=plan,
            run_id="run-7",
            data_root=data_root,
        ).initialize()

    assert outside.exists()
    assert list(outside.iterdir()) == []


def test_non_absolute_or_separator_run_id_is_rejected(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    client = CopyingHelperClient(source_tree(tmp_path))

    for run_id in ("../escape", "nested/run", "nested\\run", "", "."):
        with pytest.raises(SetupError, match="run ID"):
            WorkspaceManager(
                client, plan=plan, run_id=run_id, data_root=data_root
            )


def test_workdir_value_passed_to_helper_is_exact_posix_path(tmp_path):
    data_root, plan = managed_plan(tmp_path)
    client = CopyingHelperClient(source_tree(tmp_path))
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )

    manager.initialize()

    environment = client.containers.runs[0]["environment"]
    assert environment == {
        "NVIDIA_VISIBLE_DEVICES": "void",
        "RSI_SOURCE_WORKDIR": str(PurePosixPath("/workspace")),
        "RSI_INIT_TARGET": environment["RSI_INIT_TARGET"],
        "RSI_ENGINE_UID": str(os.getuid()),
        "RSI_ENGINE_GID": str(os.getgid()),
    }


@pytest.mark.integration
def test_real_helper_overrides_hostile_base_entrypoint(tmp_path):
    try:
        client = docker.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local Ubuntu Docker capability unavailable: {error}")
    context = tmp_path / "hostile-image"
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "RUN mkdir -p /workspace/sub && "
        "printf payload > /workspace/sub/payload && "
        "chmod 0755 /workspace/sub/payload && "
        "ln -s sub/payload /workspace/payload-link\n"
        "WORKDIR /workspace\n"
        'ENTRYPOINT ["/bin/sh", "-c", "exit 77"]\n'
    )
    image, _ = client.images.build(
        path=str(context), rm=True, forcerm=True, pull=False
    )
    data_root, plan = managed_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "images": ImagePlan(
                base_ref=image.id,
                work_ref=image.id,
                judge_ref=image.id,
                workdir=PurePosixPath("/workspace"),
                rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
                base_digest=image.id,
                work_digest=image.id,
                judge_digest=image.id,
            )
        }
    )
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )
    workspace = plan.paths.workspace
    try:
        workspace = manager.initialize()
        assert workspace.stat().st_uid == os.getuid()
        assert workspace.stat().st_gid == os.getgid()
        assert workspace.stat().st_mode & 0o777 == 0o777
        assert (workspace / "sub" / "payload").read_text() == "payload"
        assert (workspace / "sub" / "payload").stat().st_mode & 0o777 == 0o755
        assert (workspace / "payload-link").is_symlink()
    finally:
        if workspace.exists():
            manager.delete()
        image.remove(force=True)


@pytest.mark.integration
def test_real_non_host_uid_can_write_initialized_bind_workspace(tmp_path):
    try:
        client = docker.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local Ubuntu Docker capability unavailable: {error}")
    context = tmp_path / "non-host-user-image"
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "RUN mkdir -p /workspace && printf clean > /workspace/payload\n"
        "WORKDIR /workspace\n"
        "USER 2001:2002\n"
    )
    image, _ = client.images.build(path=str(context), rm=True, forcerm=True, pull=False)
    data_root, plan = managed_plan(tmp_path, work_user=image.attrs["Config"]["User"])
    plan = plan.model_copy(
        update={
            "images": plan.images.model_copy(
                update={
                    "base_ref": image.id,
                    "work_ref": image.id,
                    "judge_ref": image.id,
                    "base_digest": image.id,
                    "work_digest": image.id,
                    "judge_digest": image.id,
                    "judge_user": "2003:2004",
                }
            )
        }
    )
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )
    workspace = plan.paths.workspace
    try:
        workspace = manager.initialize()
        assert workspace.stat().st_uid == os.getuid()
        assert workspace.stat().st_gid == os.getgid()
        assert workspace.stat().st_mode & 0o777 == 0o777

        result = client.containers.run(
            image.id,
            [
                "/bin/sh",
                "-c",
                "printf writable > /workspace/agent-output; "
                "if chmod 000 /workspace 2>/dev/null; then exit 91; fi",
            ],
            user="2001:2002",
            network_mode="none",
            volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
            remove=True,
        )

        assert result == b""
        assert (workspace / "agent-output").read_text() == "writable"

        result = client.containers.run(
            image.id,
            ["/bin/sh", "-c", "printf judged > /workspace/judge-output"],
            user="2003:2004",
            network_mode="none",
            volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
            remove=True,
        )

        assert result == b""
        assert (workspace / "judge-output").read_text() == "judged"
        assert client.containers.list(
            all=True,
            filters={
                "label": [
                    "rsi-harness.run-id=run-7",
                    f"rsi-harness.task-id={plan.task.task_id}",
                    "rsi-harness.role=helper",
                ]
            },
        ) == []
    finally:
        if workspace.exists():
            manager.delete()
        image.remove(force=True)


@pytest.mark.integration
@pytest.mark.parametrize(
    "work_user",
    ("root", f"{os.getuid()}:{os.getgid()}"),
    ids=("root-cap-fowner", "engine-uid-owner"),
)
def test_real_paused_work_workspace_mode_is_restored_before_distinct_judge(
    tmp_path, work_user
):
    try:
        client = docker.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local Ubuntu Docker capability unavailable: {error}")
    setfacl = shutil.which("setfacl")
    if setfacl is None:
        pytest.skip("host setfacl binary unavailable for real ACL regression")
    context = tmp_path / "workspace-repair-image"
    context.mkdir()
    shutil.copy2(setfacl, context / "setfacl")
    (context / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "COPY setfacl /usr/local/bin/setfacl\n"
        "RUN mkdir -p /workspace\n"
        "WORKDIR /workspace\n"
    )
    image, _ = client.images.build(path=str(context), rm=True, forcerm=True, pull=False)
    data_root, plan = managed_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "images": plan.images.model_copy(
                update={
                    "base_ref": image.id,
                    "work_ref": image.id,
                    "judge_ref": image.id,
                    "base_digest": image.id,
                    "work_digest": image.id,
                    "judge_digest": image.id,
                }
            )
        }
    )
    manager = WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    )
    workspace = plan.paths.workspace
    try:
        workspace = manager.initialize()
        client.containers.run(
            image.id,
            [
                "/bin/sh",
                "-c",
                "setfacl -m u:2003:--- /workspace && "
                "chmod 0777 /workspace && test \"$(stat -c %a /workspace)\" = 777",
            ],
            user=work_user,
            network_mode="none",
            volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
            remove=True,
        )
        assert workspace.stat().st_mode & 0o777 == 0o777

        manager.attest_for_judge()

        stat = workspace.stat()
        assert stat.st_uid == os.getuid()
        assert stat.st_gid == os.getgid()
        assert stat.st_mode & 0o777 == 0o777
        result = client.containers.run(
            image.id,
            ["/bin/sh", "-c", "printf judged > /workspace/judge-output"],
            user="2003:2004",
            network_mode="none",
            volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
            remove=True,
        )
        assert result == b""
        assert (workspace / "judge-output").read_text() == "judged"
        assert client.containers.list(
            all=True,
            filters={
                "label": [
                    "rsi-harness.run-id=run-7",
                    f"rsi-harness.task-id={plan.task.task_id}",
                    "rsi-harness.role=helper",
                ]
            },
        ) == []
    finally:
        if workspace.exists():
            manager.delete()
        image.remove(force=True)


@pytest.mark.parametrize(
    "workdir",
    (
        PurePosixPath("/rsi-init-target"),
        PurePosixPath("/rsi-init-target/project"),
        PurePosixPath("/rsi-init"),
    ),
)
def test_helper_target_is_selected_disjoint_from_every_valid_workdir(
    tmp_path, workdir
):
    data_root, plan = managed_plan(tmp_path)
    plan = plan.model_copy(update={"workdir": workdir})
    client = CopyingHelperClient(source_tree(tmp_path))

    WorkspaceManager(
        client, plan=plan, run_id="run-7", data_root=data_root
    ).initialize()

    helper = client.containers.runs[0]
    target = PurePosixPath(helper["environment"]["RSI_INIT_TARGET"])
    assert target != workdir
    assert target not in workdir.parents
    assert workdir not in target.parents
