"""Agent-extensible, in-memory discovery of local login credentials."""

from __future__ import annotations

import errno
import json
import os
import pwd
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from rsi_harness.errors import SetupError
from rsi_harness.models import AgentAuthSource, ContainerTmpfs

_MAX_AUTH_BYTES = 1024 * 1024
_PRIVATE_TMPFS_OPTIONS = "rw,nosuid,nodev,noexec,mode=0700"


@dataclass(frozen=True, slots=True, repr=False)
class AgentAuthFile:
    """One secret file relative to an Agent-owned private tmpfs."""

    path: PurePosixPath
    content: bytes
    mode: int = 0o600

    def __post_init__(self) -> None:
        raw = str(self.path)
        if (
            raw in {"", "."}
            or self.path.is_absolute()
            or ".." in self.path.parts
            or len(self.path.parts) != 1
        ):
            raise SetupError("Agent auth file path must be one safe relative name")
        if not isinstance(self.content, bytes) or not self.content:
            raise SetupError("Agent auth file content must be non-empty bytes")
        if len(self.content) > _MAX_AUTH_BYTES:
            raise SetupError("Agent auth file exceeds the size limit")
        if self.mode not in {0o400, 0o600}:
            raise SetupError("Agent auth file mode must be private")

    def __repr__(self) -> str:
        return "AgentAuthFile(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class AgentAuthMount:
    """A private tmpfs and the in-memory credentials injected into it."""

    tmpfs: ContainerTmpfs
    files: tuple[AgentAuthFile, ...]

    def __post_init__(self) -> None:
        if (
            not self.tmpfs.target.is_absolute()
            or self.tmpfs.target == PurePosixPath("/")
            or self.tmpfs.options != _PRIVATE_TMPFS_OPTIONS
        ):
            raise SetupError("Agent auth requires an exact private tmpfs")
        if not self.files or len({entry.path for entry in self.files}) != len(
            self.files
        ):
            raise SetupError("Agent auth files must be non-empty and unique")

    def __repr__(self) -> str:
        return "AgentAuthMount(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class AgentAuthMaterial:
    """Resolved credentials returned by an Agent-specific auth provider."""

    agent_name: str
    mounts: tuple[AgentAuthMount, ...]
    secret_values: frozenset[str]
    provider_endpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.agent_name:
            raise SetupError("Agent auth material requires an Agent name")
        targets = [mount.tmpfs.target for mount in self.mounts]
        if len(set(targets)) != len(targets):
            raise SetupError("Agent auth tmpfs targets must be unique")

    def __repr__(self) -> str:
        return f"AgentAuthMaterial(agent_name={self.agent_name!r}, <redacted>)"


class LocalAgentAuthProvider(Protocol):
    """Extension point for one Agent's local-login file format."""

    def resolve(
        self,
        *,
        environ: Mapping[str, str] | None,
        process_uid: int | None,
        effective_uid: int | None,
        user_lookup: Callable[[int], Any],
    ) -> AgentAuthMaterial: ...


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory(
    path: Path, *, agent_label: str, label: str, dir_fd: int | None = None
) -> int:
    try:
        return os.open(path, _directory_flags(), dir_fd=dir_fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise SetupError(
                f"local {agent_label} {label} must not be a symlink"
            ) from None
        raise SetupError(f"local {agent_label} auth {label} was not found") from None


def _read_auth_file(
    directory_fd: int, *, filename: str, uid: int, agent_label: str
) -> bytes:
    auth_fd: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            auth_fd = os.open(filename, flags, dir_fd=directory_fd)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise SetupError(
                    f"local {agent_label} auth must be a non-symlink regular file"
                ) from None
            raise SetupError(f"local {agent_label} auth was not found") from None

        before = os.fstat(auth_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SetupError(f"local {agent_label} auth must be a regular file")
        if before.st_uid != uid:
            raise SetupError(
                f"local {agent_label} auth must be owned by the invoking user"
            )
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise SetupError(
                f"local {agent_label} auth permissions must deny group and other access"
            )
        if before.st_size > _MAX_AUTH_BYTES:
            raise SetupError(f"local {agent_label} auth exceeds the size limit")

        chunks: list[bytes] = []
        remaining = _MAX_AUTH_BYTES + 1
        while remaining:
            chunk = os.read(auth_fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > _MAX_AUTH_BYTES:
            raise SetupError(f"local {agent_label} auth exceeds the size limit")
        after = os.fstat(auth_fd)

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_uid,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
            )

        if identity(before) != identity(after) or len(content) != after.st_size:
            raise SetupError(
                f"local {agent_label} auth changed while it was being read"
            )
        return content
    finally:
        if auth_fd is not None:
            os.close(auth_fd)


def _read_standard_auth_file(
    home: Path,
    *,
    directory: str,
    filename: str,
    uid: int,
    agent_label: str,
) -> bytes:
    home_fd = _open_directory(
        home, agent_label=agent_label, label="home directory"
    )
    config_fd: int | None = None
    try:
        config_fd = _open_directory(
            Path(directory),
            agent_label=agent_label,
            label="directory",
            dir_fd=home_fd,
        )
        return _read_auth_file(
            config_fd, filename=filename, uid=uid, agent_label=agent_label
        )
    finally:
        if config_fd is not None:
            os.close(config_fd)
        os.close(home_fd)


def _invoking_user(
    *,
    environment: Mapping[str, str],
    process_uid: int | None,
    effective_uid: int | None,
    user_lookup: Callable[[int], Any],
    agent_label: str,
) -> tuple[int, Path]:
    real_uid = os.getuid() if process_uid is None else process_uid
    current_effective_uid = os.geteuid() if effective_uid is None else effective_uid
    selected_uid = real_uid
    sudo_uid = environment.get("SUDO_UID")
    if current_effective_uid == 0 and sudo_uid is not None:
        if not sudo_uid.isdecimal():
            raise SetupError(
                f"local {agent_label} auth has an invalid sudo user identity"
            )
        selected_uid = int(sudo_uid)

    try:
        user = user_lookup(selected_uid)
    except (KeyError, OverflowError, ValueError):
        raise SetupError(
            f"local {agent_label} auth invoking user was not found"
        ) from None
    if int(user.pw_uid) != selected_uid:
        raise SetupError(
            f"local {agent_label} auth invoking user identity is inconsistent"
        )
    home = Path(str(user.pw_dir))
    if not home.is_absolute():
        raise SetupError(f"local {agent_label} auth home directory must be absolute")
    return selected_uid, home


def _secret_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, dict):
        result: set[str] = set()
        for child in value.values():
            result.update(_secret_strings(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_secret_strings(child))
        return result
    return set()


class CodexLocalAuthProvider:
    """Resolve the invoking user's standard Codex login file."""

    def resolve(
        self,
        *,
        environ: Mapping[str, str] | None,
        process_uid: int | None,
        effective_uid: int | None,
        user_lookup: Callable[[int], Any],
    ) -> AgentAuthMaterial:
        environment = os.environ if environ is None else environ
        selected_uid, home = _invoking_user(
            environment=environment,
            process_uid=process_uid,
            effective_uid=effective_uid,
            user_lookup=user_lookup,
            agent_label="Codex",
        )
        content = _read_standard_auth_file(
            home,
            directory=".codex",
            filename="auth.json",
            uid=selected_uid,
            agent_label="Codex",
        )
        try:
            decoded = content.decode("utf-8")
            parsed = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SetupError("local Codex auth must contain valid JSON") from None
        if not isinstance(parsed, dict):
            raise SetupError("local Codex auth must contain a JSON object")
        secret_values = _secret_strings(parsed)
        secret_values.add(decoded)
        return AgentAuthMaterial(
            agent_name="codex",
            mounts=(
                AgentAuthMount(
                    tmpfs=ContainerTmpfs(
                        target=PurePosixPath("/home/agent/.codex"),
                        options=_PRIVATE_TMPFS_OPTIONS,
                    ),
                    files=(
                        AgentAuthFile(
                            path=PurePosixPath("auth.json"),
                            content=content,
                            mode=0o600,
                        ),
                    ),
                ),
            ),
            secret_values=frozenset(secret_values),
            provider_endpoints=(
                (
                    "https://chatgpt.com/backend-api/codex",
                    "https://auth.openai.com",
                )
                if parsed.get("auth_mode") == "chatgpt"
                else ()
            ),
        )


class ClaudeCodeLocalAuthProvider:
    """Resolve the invoking user's standard Claude Code login file."""

    def resolve(
        self,
        *,
        environ: Mapping[str, str] | None,
        process_uid: int | None,
        effective_uid: int | None,
        user_lookup: Callable[[int], Any],
    ) -> AgentAuthMaterial:
        environment = os.environ if environ is None else environ
        selected_uid, home = _invoking_user(
            environment=environment,
            process_uid=process_uid,
            effective_uid=effective_uid,
            user_lookup=user_lookup,
            agent_label="Claude Code",
        )
        configured = environment.get("CLAUDE_CONFIG_DIR")
        if configured:
            config_dir = Path(configured)
            if not config_dir.is_absolute():
                raise SetupError(
                    "local Claude Code auth config directory must be absolute"
                )
            config_fd = _open_directory(
                config_dir,
                agent_label="Claude Code",
                label="config directory",
            )
            try:
                content = _read_auth_file(
                    config_fd,
                    filename=".credentials.json",
                    uid=selected_uid,
                    agent_label="Claude Code",
                )
            finally:
                os.close(config_fd)
        else:
            content = _read_standard_auth_file(
                home,
                directory=".claude",
                filename=".credentials.json",
                uid=selected_uid,
                agent_label="Claude Code",
            )
        try:
            decoded = content.decode("utf-8")
            parsed = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SetupError(
                "local Claude Code auth must contain valid JSON"
            ) from None
        if not isinstance(parsed, dict):
            raise SetupError("local Claude Code auth must contain a JSON object")
        secret_values = _secret_strings(parsed)
        secret_values.add(decoded)
        return AgentAuthMaterial(
            agent_name="claude-code",
            mounts=(
                AgentAuthMount(
                    tmpfs=ContainerTmpfs(
                        target=PurePosixPath("/home/agent/.claude"),
                        options=_PRIVATE_TMPFS_OPTIONS,
                    ),
                    files=(
                        AgentAuthFile(
                            path=PurePosixPath(".credentials.json"),
                            content=content,
                            mode=0o600,
                        ),
                    ),
                ),
            ),
            secret_values=frozenset(secret_values),
        )


_LOCAL_PROVIDERS: Mapping[str, LocalAgentAuthProvider] = {
    "claude-code": ClaudeCodeLocalAuthProvider(),
    "codex": CodexLocalAuthProvider(),
}


def resolve_agent_auth(
    *,
    source: AgentAuthSource | None,
    agent_name: str,
    agent_api_key: str | None,
    providers: Mapping[str, LocalAgentAuthProvider] | None = None,
    environ: Mapping[str, str] | None = None,
    process_uid: int | None = None,
    effective_uid: int | None = None,
    user_lookup: Callable[[int], Any] = pwd.getpwuid,
) -> AgentAuthMaterial | None:
    """Resolve an explicit auth source through the selected Agent provider."""

    if source is None or agent_api_key:
        return None
    if source is not AgentAuthSource.LOCAL:
        raise SetupError(f"unsupported Agent auth source {source!r}")
    registry = _LOCAL_PROVIDERS if providers is None else providers
    provider = registry.get(agent_name)
    if provider is None:
        raise SetupError(
            f"local authentication is not supported for Agent {agent_name!r}"
        )
    material = provider.resolve(
        environ=environ,
        process_uid=process_uid,
        effective_uid=effective_uid,
        user_lookup=user_lookup,
    )
    if material.agent_name != agent_name:
        raise SetupError("Agent auth provider returned material for another Agent")
    return material


__all__ = [
    "AgentAuthFile",
    "AgentAuthMaterial",
    "AgentAuthMount",
    "ClaudeCodeLocalAuthProvider",
    "CodexLocalAuthProvider",
    "LocalAgentAuthProvider",
    "resolve_agent_auth",
]
