"""Content-addressed clean Base/Judge and independently derived Work images."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from docker.errors import APIError, BuildError, ImageNotFound

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopBackendBridge,
    RSILoopContainerHandle,
)
from rsi_harness.integrations.submit_client import generate_submit_client
from rsi_harness.models import (
    ContainerSpec,
    ImagePlan,
    RootfsSnapshotMode,
    TaskDefinition,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.gpu import (
    NVIDIA_VISIBLE_DEVICES_ENV,
    NVIDIA_VISIBLE_DEVICES_VOID,
)
from rsi_loop import __version__ as rsi_loop_version
from rsi_loop.harness.agent import Agent, create_agent, list_agent_classes
from rsi_loop.harness.config import RSILoopConfig

DEFAULT_RSI_LOOP_VERSION = rsi_loop_version
DEFAULT_BOOTSTRAP_VERSION = "1"
_SAFE_AGENT_LAUNCHER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_ORIGINAL_LAUNCHER_DIR = "/opt/rsi-harness/original-launchers"


def _bash_run_instruction(command: str) -> str:
    return "RUN " + json.dumps(["/bin/bash", "-lc", command])


def _registered_agent_launcher(agent: Agent) -> str:
    try:
        command = shlex.split(agent.run_cmd)
    except (AttributeError, TypeError, ValueError) as error:
        raise SetupError(
            "registered RSI Loop Agent has an invalid run command"
        ) from error
    if not command or _SAFE_AGENT_LAUNCHER.fullmatch(command[0]) is None:
        raise SetupError("registered RSI Loop Agent launcher is unsafe")
    return command[0]


def _drain_build_logs(logs: Any) -> None:
    for _event in logs:
        pass


def build_agent_identity_script(effective_user: str) -> str:
    """Create RSI Loop's ``agent`` alias at the effective Harbor UID/GID."""
    return f"""set -euo pipefail
effective={shlex.quote(effective_user)}
if [[ "$effective" == *:* ]]; then
    effective_user=${{effective%%:*}}
    effective_group=${{effective#*:}}
else
    effective_user=$effective
    effective_group=
fi
if [[ -z "$effective_user" || ( "$effective" == *:* && -z "$effective_group" ) ]]; then
    echo "invalid effective user specification" >&2
    exit 86
fi
if [[ "$effective_user" =~ ^[0-9]+$ ]]; then
    uid=$effective_user
else
    uid=$(id -u "$effective_user")
fi
if [[ -n "$effective_group" ]]; then
    if [[ "$effective_group" =~ ^[0-9]+$ ]]; then
        gid=$effective_group
    else
        gid=$(getent group "$effective_group" | cut -d: -f3)
        test -n "$gid"
    fi
elif [[ "$effective_user" =~ ^[0-9]+$ ]]; then
    gid=$(awk -F: -v uid="$uid" '$3 == uid {{ print $4; exit }}' /etc/passwd)
    gid=${{gid:-$uid}}
else
    gid=$(id -g "$effective_user")
fi
if getent group agent >/dev/null; then
    if [[ "$(getent group agent | cut -d: -f3)" != "$gid" ]]; then
        groupmod -o -g "$gid" agent
    fi
else
    groupadd -o -g "$gid" agent
fi
if getent passwd agent >/dev/null; then
    usermod -o -u "$uid" -g agent -d /home/agent agent
else
    useradd -o -u "$uid" -g agent -d /home/agent -M agent
fi
mkdir -p /home/agent
chown -R agent:agent /home/agent
chmod 0755 /home/agent
"""


def resolve_image_workdir(
    configured: PurePosixPath | None, image: Any
) -> PurePosixPath:
    """Resolve Harbor WORKDIR first, otherwise inspect pinned image metadata."""
    raw = (
        str(configured)
        if configured is not None
        else str(image.attrs.get("Config", {}).get("WorkingDir", ""))
    )
    if raw in {"", ".", "/"}:
        return PurePosixPath("/")
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise SetupError("image must provide an absolute WORKDIR without traversal")
    return path


def resolve_rootfs_snapshot_mode(
    workdir: PurePosixPath,
) -> RootfsSnapshotMode:
    return (
        RootfsSnapshotMode.FULL_ROOTFS
        if workdir == PurePosixPath("/")
        else RootfsSnapshotMode.SPLIT_WORKDIR
    )


class DockerImageBuilder:
    """Prepare an immutable clean Base and a generated Agent-only Work layer."""

    def __init__(
        self,
        client: Any,
        *,
        run_id: str = "image-preparation",
        rsi_loop_version: str = DEFAULT_RSI_LOOP_VERSION,
        bootstrap_version: str = DEFAULT_BOOTSTRAP_VERSION,
        rsi_loop_config: RSILoopConfig | None = None,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._rsi_loop_version = rsi_loop_version
        self._bootstrap_version = bootstrap_version
        self._rsi_loop_config = rsi_loop_config or RSILoopConfig()

    def prepare(self, task: TaskDefinition, *, agent: Agent) -> ImagePlan:
        base_image = self._prepare_base(task)
        volumes = base_image.attrs.get("Config", {}).get("Volumes")
        if volumes:
            raise SetupError(
                "Docker image declares volumes that cannot be included in a rootfs "
                "snapshot"
            )
        base_ref = self._immutable_ref(base_image)
        base_digest = self._image_id(base_image)
        workdir = resolve_image_workdir(task.service.workdir, base_image)
        base_user = str(base_image.attrs.get("Config", {}).get("User", "")).strip()
        effective_user = task.agent.user or task.service.user or base_user or "0"
        verifier_user = task.verifier.user or task.service.user or base_user or "0"
        submit_client = generate_submit_client()
        dockerfile = self._work_dockerfile(
            agent=agent,
            base_ref=base_ref,
            workdir=workdir,
            final_user=effective_user,
        )
        work_tag = self._work_tag(
            task,
            agent,
            base_digest=base_digest,
            effective_user=effective_user,
            workdir=workdir,
            dockerfile=dockerfile,
            submit_client=submit_client,
        )
        try:
            work_image = self._client.images.get(work_tag)
        except ImageNotFound:
            work_image = self._build_work(
                task,
                agent=agent,
                work_tag=work_tag,
                dockerfile=dockerfile,
                submit_client=submit_client,
            )
        except APIError as error:
            raise SetupError(
                f"failed to inspect Work image {work_tag}: {error}"
            ) from error
        work_ref = self._immutable_ref(work_image)
        work_digest = self._image_id(work_image)
        self._preflight(
            task,
            base_image=base_image,
            base_ref=base_ref,
            work_ref=work_ref,
            workdir=workdir,
        )
        if task.agent.install_stop_hook:
            self._preflight_hooks(task, agent=agent, work_ref=work_ref)
        return ImagePlan(
            base_ref=base_ref,
            work_ref=work_ref,
            judge_ref=base_ref,
            workdir=workdir,
            rootfs_snapshot_mode=resolve_rootfs_snapshot_mode(workdir),
            work_user=effective_user,
            judge_user=verifier_user,
            base_digest=base_digest,
            work_digest=work_digest,
            judge_digest=base_digest,
        )

    def _prepare_base(self, task: TaskDefinition) -> Any:
        if task.service.image:
            try:
                return self._client.images.get(task.service.image)
            except ImageNotFound:
                try:
                    return self._client.images.pull(task.service.image)
                except Exception as error:
                    raise SetupError(
                        f"failed to pull task Base image {task.service.image!r}: "
                        f"{error}"
                    ) from error
            except APIError as error:
                raise SetupError(
                    f"failed to inspect task Base image {task.service.image!r}: {error}"
                ) from error
        context = task.service.build_context
        if context is None:
            raise SetupError(
                "task has neither a prebuilt image nor an environment context"
            )
        expected = (task.source_dir / "environment").resolve()
        if context.resolve() != expected:
            raise SetupError(
                "Base image context must be exactly the task environment directory"
            )
        tag = f"rsi-harness-base:{self._tag_part(task.environment_digest or '')}"
        try:
            return self._client.images.get(tag)
        except ImageNotFound:
            try:
                image, logs = self._client.images.build(
                    path=str(expected),
                    tag=tag,
                    timeout=max(1, round(task.service.build_timeout_seconds)),
                    pull=True,
                    rm=True,
                    forcerm=True,
                    labels=self._labels(task, "judge"),
                )
                _drain_build_logs(logs)
                return image
            except BuildError as error:
                raise SetupError(
                    f"failed to build clean task Base image: {error}"
                ) from error
            except APIError as error:
                raise SetupError(
                    f"Docker failed while building clean task Base image: {error}"
                ) from error

    def _build_work(
        self,
        task: TaskDefinition,
        *,
        agent: Agent,
        work_tag: str,
        dockerfile: str,
        submit_client: str,
    ) -> Any:
        try:
            with tempfile.TemporaryDirectory(prefix="rsi-work-image-") as raw_context:
                context = Path(raw_context)
                (context / "Dockerfile").write_text(dockerfile)
                submit = context / "rsi-submit"
                submit.write_text(submit_client)
                submit.chmod(0o755)
                image, logs = self._client.images.build(
                    path=str(context),
                    tag=work_tag,
                    timeout=max(1, round(task.service.build_timeout_seconds)),
                    pull=False,
                    rm=True,
                    forcerm=True,
                    labels=self._labels(task, "work"),
                )
                _drain_build_logs(logs)
                return image
        except (BuildError, APIError) as error:
            raise SetupError(
                f"{agent.name} Work image setup failed while running pinned install "
                f"commands: {error}"
            ) from error

    def _work_dockerfile(
        self,
        *,
        agent: Agent,
        base_ref: str,
        workdir: PurePosixPath,
        final_user: str,
    ) -> str:
        launcher = _registered_agent_launcher(agent)
        quarantine_launcher = f"""set -euo pipefail
launcher={shlex.quote(launcher)}
existing=$(command -v "$launcher" || true)
if [[ -n "$existing" ]]; then
    case "$existing" in
        /*) ;;
        *) echo 'existing Agent launcher path is unsafe' >&2; exit 86 ;;
    esac
    install -d -m 0755 {_ORIGINAL_LAUNCHER_DIR}
    mv -f -- "$existing" {_ORIGINAL_LAUNCHER_DIR}/{shlex.quote(launcher)}
fi
"""
        lines = [
            f"FROM {base_ref}",
            "USER root",
            "RUN if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "bash ca-certificates curl passwd sudo tar xz-utils && "
            "rm -rf /var/lib/apt/lists/*; "
            "elif command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache bash ca-certificates curl shadow sudo tar xz; "
            "else echo 'setup command failed: supported package manager' >&2; "
            "exit 86; fi",
            _bash_run_instruction(build_agent_identity_script(final_user)),
            "COPY rsi-submit /usr/local/bin/rsi-submit",
            "RUN chmod 0755 /usr/local/bin/rsi-submit",
            _bash_run_instruction(quarantine_launcher),
        ]
        for command in agent.install_cmds:
            build_command = command.replace("sudo -E ", "")
            lines.append(_bash_run_instruction(build_command))
        lines.append(
            _bash_run_instruction(
                f"launcher=$(command -v {shlex.quote(launcher)}) && "
                'test -n "$launcher" && test -x "$launcher"'
            )
        )
        lines.extend(["ENV HOME=/home/agent", f"WORKDIR {workdir}"])
        if final_user:
            lines.append(f"USER {final_user}")
        return "\n".join(lines) + "\n"

    def _preflight(
        self,
        task: TaskDefinition,
        *,
        base_image: Any,
        base_ref: str,
        work_ref: str,
        workdir: PurePosixPath,
    ) -> None:
        image_user = str(base_image.attrs.get("Config", {}).get("User", "")).strip()
        agent_user = task.agent.user or task.service.user or image_user or None
        verifier_user = task.verifier.user or task.service.user or image_user or None
        try:
            self._client.containers.run(
                work_ref,
                [
                    "/bin/sh",
                    "-c",
                    'test "$(id -u agent)" = "$(id -u)" && '
                    'test "$(id -g agent)" = "$(id -g)" && '
                    f"test -w {shlex.quote(str(workdir))} && test -w /home/agent",
                ],
                user=agent_user,
                environment={
                    "HOME": "/home/agent",
                    NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID,
                },
                network_mode="none",
                labels=self._labels(task, "work"),
                remove=True,
            )
            self._client.containers.run(
                base_ref,
                ["/bin/sh", "-c", f"test -d {shlex.quote(str(workdir))}"],
                user=verifier_user,
                environment={
                    NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID
                },
                network_mode="none",
                labels=self._labels(task, "judge"),
                remove=True,
            )
        except Exception as error:
            raise SetupError(
                f"image user/WORKDIR dynamic preflight failed: {error}"
            ) from error

    def _preflight_hooks(
        self, task: TaskDefinition, *, agent: Agent, work_ref: str
    ) -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="rsi-hook-preflight-") as raw_root:
                root = Path(raw_root)
                staging = root / "staging"
                runtime = DockerContainerRuntime(
                    self._client,
                    run_id=self._run_id,
                    task_id=task.task_id,
                    role="work",
                    task_source_dir=task.source_dir,
                    allowed_mount_roots=(root,),
                    staging_dir=staging,
                )
                container = runtime.create(
                    ContainerSpec(
                        image=work_ref,
                        command=("sleep", "infinity"),
                        user="root",
                    )
                )
                self._client.containers.get(container.container_id).start()
                try:
                    for agent_class in list_agent_classes():
                        hook_agent = create_agent(
                            agent_class.name, self._rsi_loop_config
                        )
                        log_dir = root / "logs" / agent_class.name
                        log_dir.mkdir(parents=True)
                        hook_agent.install_stop_hook(
                            RSILoopBackendBridge(runtime),
                            RSILoopContainerHandle(container),
                            log_dir,
                            logging.getLogger(__name__),
                        )
                finally:
                    runtime.stop(container)
                    runtime.remove(container)
        except Exception as error:
            raise SetupError(
                f"real RSI Loop stop-hook preflight failed: {error}"
            ) from error

    def _work_tag(
        self,
        task: TaskDefinition,
        agent: Agent,
        *,
        base_digest: str,
        effective_user: str,
        workdir: PurePosixPath,
        dockerfile: str,
        submit_client: str,
    ) -> str:
        content_digest = hashlib.sha256(
            json.dumps(
                {
                    "base_digest": base_digest,
                    "effective_user": effective_user,
                    "workdir": str(workdir),
                    "dockerfile": dockerfile,
                    "submit_client": submit_client,
                    "image_labels": self._labels(task, "work"),
                    "rsi_loop_version": self._rsi_loop_version,
                    "bootstrap_version": self._bootstrap_version,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        values = (
            task.environment_digest or "missing-environment-digest",
            agent.name,
            self._rsi_loop_version,
            f"v{self._bootstrap_version}",
            content_digest,
        )
        return "rsi-harness-work:" + "-".join(self._tag_part(value) for value in values)

    @staticmethod
    def _tag_part(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip(".-")
        if not normalized:
            raise SetupError("image content tag component must not be empty")
        return normalized

    @staticmethod
    def _image_id(image: Any) -> str:
        image_id = str(image.attrs.get("Id") or image.id)
        if not image_id:
            raise SetupError("Docker image inspection returned no immutable ID")
        return image_id

    @classmethod
    def _immutable_ref(cls, image: Any) -> str:
        digests = tuple(image.attrs.get("RepoDigests") or ())
        return str(digests[0]) if digests else cls._image_id(image)

    def _labels(self, task: TaskDefinition, role: str) -> dict[str, str]:
        return {
            "rsi-harness.run-id": self._run_id,
            "rsi-harness.task-id": task.task_id,
            "rsi-harness.role": role,
        }


__all__ = [
    "DockerImageBuilder",
    "build_agent_identity_script",
    "resolve_image_workdir",
]
