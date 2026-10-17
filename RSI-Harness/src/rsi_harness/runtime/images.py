"""Content-addressed clean Base/Judge and independently derived Work images."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import tarfile
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from docker.errors import APIError, BuildError, ImageNotFound

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopBackendBridge,
    RSILoopContainerHandle,
)
from rsi_harness.integrations.submit_client import generate_submit_client
from rsi_harness.judge_agent import JUDGE_AGENT_UIDS
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
from rsi_loop.harness.agent import Agent, create_agent
from rsi_loop.harness.config import RSILoopConfig

DEFAULT_RSI_LOOP_VERSION = rsi_loop_version
DEFAULT_BOOTSTRAP_VERSION = "1"
_SAFE_AGENT_LAUNCHER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_SAFE_IMAGE_FILE_PATH = re.compile(r"[A-Za-z0-9._+/-]+")
_SAFE_BUILD_ARG = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ORIGINAL_LAUNCHER_DIR = "/opt/rsi-harness/original-launchers"
_AGENT_FILES_CONTEXT_DIR = "agent-files"
# Docker's image architecture -> the machine the Codex releases are named for.
_CODEX_ARCHITECTURES = {"amd64": "x86_64", "arm64": "aarch64"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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


def _quarantine_launcher_script(launcher: str) -> str:
    # A task image that already ships the launcher would make the pinned
    # install fail or shadow it on PATH, so move it aside first.
    return f"""set -euo pipefail
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


def _agent_build_args(agent: Agent) -> dict[str, str | None]:
    """Return the Agent's build arguments, refusing a name ARG cannot take."""
    build_args = agent.build_args()
    for name in build_args:
        if not isinstance(name, str) or _SAFE_BUILD_ARG.fullmatch(name) is None:
            raise SetupError(f"{agent.name} build argument is unsafe: {name!r}")
    return build_args


def _agent_launchers(agent: Agent) -> tuple[str, ...]:
    """Return the run command's launcher followed by the Agent's extra ones."""
    launchers = [_registered_agent_launcher(agent)]
    for launcher in agent.extra_launchers:
        if (
            not isinstance(launcher, str)
            or _SAFE_AGENT_LAUNCHER.fullmatch(launcher) is None
        ):
            raise SetupError(f"{agent.name} extra launcher is unsafe: {launcher!r}")
        if launcher not in launchers:
            launchers.append(launcher)
    return tuple(launchers)


@dataclass(frozen=True)
class _AgentImageFile:
    target: PurePosixPath
    context_name: str
    mode: int
    content: bytes


def _agent_image_files(agent: Agent) -> tuple[_AgentImageFile, ...]:
    """Resolve an Agent's packaged image files in a stable order."""
    files: list[_AgentImageFile] = []
    for index, (raw_target, (resource, mode)) in enumerate(
        sorted(agent.image_files.items())
    ):
        target = PurePosixPath(raw_target)
        if (
            not target.is_absolute()
            or ".." in target.parts
            or _SAFE_IMAGE_FILE_PATH.fullmatch(raw_target) is None
        ):
            raise SetupError(f"{agent.name} image file path is unsafe: {raw_target!r}")
        source = PurePosixPath(resource)
        if source.is_absolute() or ".." in source.parts:
            raise SetupError(f"{agent.name} image file source is unsafe: {resource!r}")
        if not isinstance(mode, int) or not 0 <= mode <= 0o777:
            raise SetupError(f"{agent.name} image file mode is invalid: {mode!r}")
        try:
            content = resources.files("rsi_loop").joinpath(resource).read_bytes()
        except OSError as error:
            raise SetupError(
                f"{agent.name} image file {resource!r} is missing from the "
                f"rsi_loop package: {error}"
            ) from error
        files.append(
            _AgentImageFile(
                target=target,
                context_name=f"{_AGENT_FILES_CONTEXT_DIR}/{index}",
                mode=mode,
                content=content,
            )
        )
    return tuple(files)


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
                for image_file in _agent_image_files(agent):
                    staged = context / image_file.context_name
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_bytes(image_file.content)
                # Only the values that are set; the Dockerfile declares every
                # name, so the tag does not depend on them.
                build_args = {
                    name: value
                    for name, value in _agent_build_args(agent).items()
                    if value
                }
                image, logs = self._client.images.build(
                    path=str(context),
                    tag=work_tag,
                    buildargs=build_args,
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
        launchers = _agent_launchers(agent)
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
        ]
        for launcher in launchers:
            lines.append(_bash_run_instruction(_quarantine_launcher_script(launcher)))
        # After the quarantine, which would otherwise move a baked-in
        # launcher aside, and before install_cmds, which may use the files.
        image_files = _agent_image_files(agent)
        for image_file in image_files:
            lines.append(
                "COPY " + json.dumps([image_file.context_name, str(image_file.target)])
            )
        if image_files:
            lines.append(
                _bash_run_instruction(
                    " && ".join(
                        f"chmod {image_file.mode:04o} "
                        f"{shlex.quote(str(image_file.target))}"
                        for image_file in image_files
                    )
                )
            )
        lines.extend(f"ARG {name}" for name in _agent_build_args(agent))
        for command in agent.install_cmds:
            build_command = command.replace("sudo -E ", "")
            lines.append(_bash_run_instruction(build_command))
        for launcher in launchers:
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
                environment={NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID},
                network_mode="none",
                labels=self._labels(task, "judge"),
                remove=True,
            )
        except Exception as error:
            raise SetupError(
                f"image user/WORKDIR dynamic preflight failed: {error}"
            ) from error
        if task.verifier.model_access or task.verifier.judge_model is not None:
            self._preflight_judge_agent(task, work_ref=work_ref)

    def prepare_judge_codex(
        self,
        task: TaskDefinition,
        *,
        images: ImagePlan,
        work_agent: Agent,
        cache_root: Path,
    ) -> Path:
        """The pinned Codex CLI for a Judge LLM, as a host directory to mount.

        The native binary comes from the Codex Work image the Harness builds
        for this task with its usual install commands (the pinned version,
        for the image's architecture): the run's own Work image when the Agent
        is Codex, otherwise that image built alongside, never used for Work.
        Those images start from the task's, so the bytes are trusted only when
        their SHA-256 is the release's, pinned in versions.py: checked before
        caching below ``cache_root`` and again on every later use.
        """
        from rsi_loop.harness.agent import versions

        try:
            architecture = self._client.images.get(images.base_ref).attrs.get(
                "Architecture"
            )
        except (APIError, ImageNotFound) as error:
            raise SetupError(f"cannot inspect the task image: {error}") from error
        machine = _CODEX_ARCHITECTURES.get(str(architecture))
        if machine is None:
            raise SetupError(
                f"the Judge LLM's Codex CLI has no release for {architecture}"
            )
        expected = getattr(versions, f"CODEX_SHA256_{machine.upper()}")
        directory = cache_root / f"codex-{versions.CODEX}-{machine}"
        binary = directory / "codex"
        if (
            not directory.is_symlink()
            and binary.is_file()
            and not binary.is_symlink()
            and binary.stat().st_mode & 0o777 == 0o755
            and _sha256_file(binary) == expected
        ):
            return directory
        source = images.work_ref
        if work_agent.name != "codex":
            source = self._codex_work_image(task, images)
        cache_root.mkdir(mode=0o755, parents=True, exist_ok=True)
        directory.mkdir(mode=0o755, exist_ok=True)
        os.chmod(directory, 0o755)
        path = self._find_codex_binary(source, task, versions.CODEX)
        container = self._client.containers.create(
            source,
            ["/bin/true"],
            network_mode="none",
            labels=self._labels(task, "judge"),
        )
        descriptor, raw_temporary = tempfile.mkstemp(prefix=".codex-", dir=directory)
        temporary = Path(raw_temporary)
        try:
            stream, _ = container.get_archive(path)
            with tempfile.TemporaryFile(dir=directory) as archive:
                for chunk in stream:
                    archive.write(chunk)
                archive.seek(0)
                with tarfile.open(fileobj=archive) as bundle:
                    members = [item for item in bundle.getmembers() if item.isfile()]
                    if len(members) != 1:
                        raise SetupError("the Codex CLI archive is not one file")
                    extracted = bundle.extractfile(members[0])
                    if extracted is None:
                        raise SetupError("the Codex CLI archive is unreadable")
                    digest = hashlib.sha256()
                    with os.fdopen(descriptor, "wb") as target:
                        descriptor = -1
                        while chunk := extracted.read(1 << 20):
                            digest.update(chunk)
                            target.write(chunk)
            if digest.hexdigest() != expected:
                raise SetupError(
                    f"the Codex CLI in {source} is not the pinned {versions.CODEX} "
                    f"release for {machine} (sha256 {digest.hexdigest()}, "
                    f"expected {expected})"
                )
            os.chmod(temporary, 0o755)
            # Concurrent runs may race here; each rename installs the same bytes.
            os.replace(temporary, binary)
        except (APIError, OSError, tarfile.TarError) as error:
            raise SetupError(f"cannot copy the Codex CLI: {error}") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            container.remove(force=True)
        return directory

    def _codex_work_image(self, task: TaskDefinition, images: ImagePlan) -> str:
        """The Codex Work image for this task: tagged, built and cached as one."""
        codex = create_agent("codex", self._rsi_loop_config)
        submit_client = generate_submit_client()
        dockerfile = self._work_dockerfile(
            agent=codex,
            base_ref=images.base_ref,
            workdir=images.workdir,
            final_user=images.work_user,
        )
        tag = self._work_tag(
            task,
            codex,
            base_digest=str(images.base_digest),
            effective_user=images.work_user,
            workdir=images.workdir,
            dockerfile=dockerfile,
            submit_client=submit_client,
        )
        try:
            image = self._client.images.get(tag)
        except ImageNotFound:
            image = self._build_work(
                task,
                agent=codex,
                work_tag=tag,
                dockerfile=dockerfile,
                submit_client=submit_client,
            )
        except APIError as error:
            raise SetupError(f"failed to inspect Codex image {tag}: {error}") from error
        return self._immutable_ref(image)

    def _find_codex_binary(self, image: str, task: TaskDefinition, version: str) -> str:
        """The path of the pinned native Codex binary npm installed in ``image``."""
        script = (
            "set -eu\n"
            "root=$(npm root -g)\n"
            'found=$(find "$root/@openai" -type f -path '
            "'*/vendor/*-unknown-linux-musl/bin/codex' 2>/dev/null || true)\n"
            'count=$(printf "%s" "$found" | grep -c . || true)\n'
            'if [ "$count" != 1 ]; then\n'
            '  echo "expected one Codex binary, found $count" >&2; exit 1\n'
            "fi\n"
            'reported=$("$found" --version)\n'
            f'if [ "$reported" != "codex-cli {version}" ]; then\n'
            '  echo "unexpected Codex version: $reported" >&2; exit 1\n'
            "fi\n"
            'printf "%s" "$found"\n'
        )
        try:
            output = self._client.containers.run(
                image,
                ["/bin/sh", "-c", script],
                user="0",
                environment={
                    "HOME": "/tmp",
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID,
                },
                network_mode="none",
                labels=self._labels(task, "judge"),
                remove=True,
            )
        except Exception as error:
            raise SetupError(f"cannot find the pinned Codex CLI: {error}") from error
        path = output.decode("utf-8", errors="replace").strip()
        if not path.startswith("/") or "\n" in path:
            raise SetupError("cannot find the pinned Codex CLI")
        return path

    def _preflight_judge_agent(self, task: TaskDefinition, *, work_ref: str) -> None:
        """A Judge with rsi-agent needs Python 3.8+ and free uids/gids."""
        first, last = JUDGE_AGENT_UIDS[0], JUDGE_AGENT_UIDS[-1]
        check = (
            "import ctypes, fcntl, grp, pwd, sys\n"
            "if sys.version_info < (3, 8):\n"
            "    sys.exit('python3 is older than 3.8')\n"
            f"ids = range({first}, {last + 1})\n"
            "taken = sorted({e.pw_uid for e in pwd.getpwall() if e.pw_uid in ids}\n"
            "               | {e.gr_gid for e in grp.getgrall() if e.gr_gid in ids})\n"
            "if taken:\n"
            f"    sys.exit('ids {first}-{last} are reserved for rsi-agent; '\n"
            "             'the image already uses ' + ', '.join(map(str, taken)))\n"
        )
        try:
            self._client.containers.run(
                work_ref,
                [
                    "/bin/sh",
                    "-c",
                    "command -v python3 >/dev/null 2>&1 || "
                    '{ echo "python3 is missing" >&2; exit 1; }; '
                    'exec python3 -I -S -B -c "$1"',
                    "rsi-agent-preflight",
                    check,
                ],
                user="0",
                environment={NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID},
                network_mode="none",
                labels=self._labels(task, "judge"),
                remove=True,
            )
        except Exception as error:
            raise SetupError(
                "model_access and judge_model need python3 3.8 or later in the "
                f"task image for rsi-agent: {error}"
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
                    log_dir = root / "logs" / agent.name
                    log_dir.mkdir(parents=True)
                    create_agent(agent.name, self._rsi_loop_config).install_stop_hook(
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
        identity: dict[str, Any] = {
            "base_digest": base_digest,
            "effective_user": effective_user,
            "workdir": str(workdir),
            "dockerfile": dockerfile,
            "submit_client": submit_client,
            "image_labels": self._labels(task, "work"),
            "rsi_loop_version": self._rsi_loop_version,
            "bootstrap_version": self._bootstrap_version,
        }
        image_files = _agent_image_files(agent)
        # Only when present, so Agents without files keep their existing tags.
        if image_files:
            identity["image_files"] = {
                str(image_file.target): hashlib.sha256(image_file.content).hexdigest()
                for image_file in image_files
            }
        content_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
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
