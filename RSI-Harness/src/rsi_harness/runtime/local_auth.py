"""Agent-extensible, in-memory discovery of local login credentials."""

from __future__ import annotations

import base64
import errno
import json
import math
import os
import pwd
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import AgentAuthSource, ContainerTmpfs
from rsi_harness.runtime.redaction import is_credential_name

_MAX_AUTH_BYTES = 1024 * 1024
MAX_AUTH_BYTES = _MAX_AUTH_BYTES
# The refresh token in every Judge's copy of a login: a Judge never refreshes
# (and so never rotates and invalidates) the login it was lent.
WITHHELD_REFRESH_TOKEN = "withheld-by-rsi-harness"
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
        # Non-blocking: opening a FIFO put in a login's place returns at once
        # (and is refused below) instead of waiting for a writer forever.
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
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
    home_fd = _open_directory(home, agent_label=agent_label, label="home directory")
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
    """Collect login credentials, not subscription names, scopes or timestamps."""
    if isinstance(value, dict):
        result: set[str] = set()
        for key, child in value.items():
            if isinstance(child, str):
                if child and (is_credential_name(key) or key.casefold() == "token"):
                    result.add(child)
            else:
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


def read_codex_login(
    codex_home: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    process_uid: int | None = None,
    effective_uid: int | None = None,
    user_lookup: Callable[[int], Any] = pwd.getpwuid,
) -> bytes:
    """Read a Codex login file (``auth.json``) with the local-login checks.

    ``codex_home`` defaults to ``$CODEX_HOME``, then the invoking user's
    ``~/.codex``. The file must be a private, non-symlink regular file owned
    by the invoking user.
    """
    environment = os.environ if environ is None else environ
    selected_uid, home = _invoking_user(
        environment=environment,
        process_uid=process_uid,
        effective_uid=effective_uid,
        user_lookup=user_lookup,
        agent_label="Codex",
    )
    directory = codex_home
    if directory is None and environment.get("CODEX_HOME"):
        directory = Path(environment["CODEX_HOME"])
    if directory is None:
        return _read_standard_auth_file(
            home,
            directory=".codex",
            filename="auth.json",
            uid=selected_uid,
            agent_label="Codex",
        )
    if not directory.is_absolute():
        raise SetupError("the Codex home directory must be absolute")
    directory_fd = _open_directory(directory, agent_label="Codex", label="home")
    try:
        return _read_auth_file(
            directory_fd, filename="auth.json", uid=selected_uid, agent_label="Codex"
        )
    finally:
        os.close(directory_fd)


def codex_login_owner(
    codex_home: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    process_uid: int | None = None,
    effective_uid: int | None = None,
    user_lookup: Callable[[int], Any] = pwd.getpwuid,
) -> tuple[Path, int, int]:
    """The directory ``read_codex_login`` reads, and its user's uid and gid."""
    environment = os.environ if environ is None else environ
    selected_uid, home = _invoking_user(
        environment=environment,
        process_uid=process_uid,
        effective_uid=effective_uid,
        user_lookup=user_lookup,
        agent_label="Codex",
    )
    directory = codex_home
    if directory is None and environment.get("CODEX_HOME"):
        directory = Path(environment["CODEX_HOME"])
    if directory is None:
        directory = home / ".codex"
    return directory, selected_uid, int(user_lookup(selected_uid).pw_gid)


def login_secret_values(value: Any) -> frozenset[str]:
    """The credential strings of a parsed login file, for redaction."""
    return frozenset(_secret_strings(value))


_CLAUDE_OAUTH_PROVIDER_ENDPOINTS = (
    "https://api.anthropic.com",
    "https://platform.claude.com",
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
            raise SetupError("local Claude Code auth must contain valid JSON") from None
        if not isinstance(parsed, dict):
            raise SetupError("local Claude Code auth must contain a JSON object")
        secret_values = _secret_strings(parsed)
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
            # A claude.ai OAuth login refreshes its short-lived access token at
            # platform.claude.com; a no-network Work container must reach it as
            # well as the inference API, which provider endpoints replace.
            provider_endpoints=(
                _CLAUDE_OAUTH_PROVIDER_ENDPOINTS
                if isinstance(parsed.get("claudeAiOauth"), dict)
                else ()
            ),
        )


# -- the Judge's copy of a Work Agent's login --------------------------------
#
# A Judge gets the Work Agent's current login before each round (Work refreshes
# its own copy during the run). The Work Agent can write those files, so what
# is read is checked against the run-start login: the same login kind and, for
# Codex, the same ChatGPT account. The account claim is read, not verified: it
# catches another login, not a forged one.


# Bounds on what the Work Agent writes. A real login nests a few levels; a
# real token is a long run of token characters. A new credential value joins
# the redaction set, so it must look like a token: a short or prose-like value
# could otherwise redact (garble or hide) ordinary text in Judge output and
# operator logs.
_MAX_LOGIN_DEPTH = 32
_MIN_NEW_CREDENTIAL = 32
_MAX_CREDENTIAL = 16 * 1024
_TOKEN_CHARACTERS = re.compile(r"[A-Za-z0-9._~+/=-]+")
# Epoch seconds: a sane expiry is a finite time before the year 5000.
_MAX_EPOCH_SECONDS = 1e11


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON number")


def _json_value(text: str | bytes) -> Any:
    """Strict JSON within a nesting bound; ValueError otherwise."""
    try:
        value = json.loads(text, parse_constant=_refuse_constant)
    except RecursionError:
        raise ValueError("JSON nests too deeply") from None
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        children = (
            item.values()
            if isinstance(item, dict)
            else item
            if isinstance(item, list)
            else ()
        )
        for child in children:
            if isinstance(child, (dict, list)):
                if depth >= _MAX_LOGIN_DEPTH:
                    raise ValueError("JSON nests too deeply")
                pending.append((child, depth + 1))
    return value


def _jwt_claims(token: object) -> dict[str, Any]:
    if not isinstance(token, str):
        return {}
    try:
        payload = token.split(".")[1]
        claims = _json_value(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
    except (ValueError, IndexError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _epoch_seconds(value: object, *, scale: float = 1.0) -> float | None:
    """An expiry as epoch seconds; ValueError for an unusable one."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("an expiry must be a number")
    try:
        seconds = float(value) / scale
    except OverflowError:
        raise ValueError("an expiry is out of range") from None
    if not math.isfinite(seconds) or not 0 < seconds < _MAX_EPOCH_SECONDS:
        raise ValueError("an expiry is out of range")
    return seconds


def _codex_identity(login: Mapping[str, Any]) -> tuple[str, str] | None:
    tokens = login.get("tokens")
    if isinstance(tokens, dict) and login.get("auth_mode") in (None, "chatgpt"):
        if (
            not isinstance(tokens.get("access_token"), str)
            or not tokens["access_token"]
        ):
            return None
        accounts = set()
        if isinstance(tokens.get("account_id"), str):
            accounts.add(tokens["account_id"])
        for name in ("id_token", "access_token"):
            claim = _jwt_claims(tokens.get(name)).get("https://api.openai.com/auth")
            if isinstance(claim, dict) and isinstance(
                claim.get("chatgpt_account_id"), str
            ):
                accounts.add(claim["chatgpt_account_id"])
        # One account, named consistently; none named is not checkable.
        if len(accounts) != 1:
            return None
        return ("chatgpt", accounts.pop())
    key = login.get("OPENAI_API_KEY")
    if isinstance(key, str) and key:
        return ("apikey", key)
    return None


def _codex_credentials(login: Mapping[str, Any]) -> tuple[object, ...]:
    tokens = login.get("tokens")
    names = ("access_token", "id_token", "refresh_token")
    named = (
        tuple(tokens.get(name) for name in names) if isinstance(tokens, dict) else ()
    )
    return (*named, login.get("OPENAI_API_KEY"))


def _codex_withheld(login: dict[str, Any]) -> dict[str, Any]:
    tokens = login.get("tokens")
    if isinstance(tokens, dict) and isinstance(tokens.get("refresh_token"), str):
        return {**login, "tokens": {**tokens, "refresh_token": WITHHELD_REFRESH_TOKEN}}
    return login


def _codex_expiry(login: Mapping[str, Any]) -> float | None:
    tokens = login.get("tokens")
    if not isinstance(tokens, dict):
        return None
    return _epoch_seconds(_jwt_claims(tokens.get("access_token")).get("exp"))


def _claude_identity(login: Mapping[str, Any]) -> tuple[str, str] | None:
    # The credentials file names no account: only the login kind is checked.
    oauth = login.get("claudeAiOauth")
    if isinstance(oauth, dict):
        token = oauth.get("accessToken")
        return ("claudeAiOauth", "") if isinstance(token, str) and token else None
    return None


def _claude_credentials(login: Mapping[str, Any]) -> tuple[object, ...]:
    oauth = login.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return ()
    return (oauth.get("accessToken"), oauth.get("refreshToken"))


def _claude_withheld(login: dict[str, Any]) -> dict[str, Any]:
    oauth = login.get("claudeAiOauth")
    if isinstance(oauth, dict) and isinstance(oauth.get("refreshToken"), str):
        return {
            **login,
            "claudeAiOauth": {**oauth, "refreshToken": WITHHELD_REFRESH_TOKEN},
        }
    return login


def _claude_expiry(login: Mapping[str, Any]) -> float | None:
    oauth = login.get("claudeAiOauth")
    expiry = oauth.get("expiresAt") if isinstance(oauth, dict) else None
    return _epoch_seconds(expiry, scale=1000.0)  # milliseconds


@dataclass(frozen=True, slots=True)
class _LoginRules:
    identity: Callable[[Mapping[str, Any]], tuple[str, str] | None]
    # The values a login authenticates with: the only ones redacted from a
    # login the Work Agent wrote.
    credentials: Callable[[Mapping[str, Any]], tuple[object, ...]]
    withheld: Callable[[dict[str, Any]], dict[str, Any]]
    # Raises ValueError for an unusable expiry.
    expiry: Callable[[Mapping[str, Any]], float | None]


_LOGIN_RULES: Mapping[str, _LoginRules] = {
    "codex": _LoginRules(
        _codex_identity, _codex_credentials, _codex_withheld, _codex_expiry
    ),
    "claude-code": _LoginRules(
        _claude_identity, _claude_credentials, _claude_withheld, _claude_expiry
    ),
}


def _parsed_login(content: bytes) -> dict[str, Any] | None:
    try:
        value = _json_value(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def withhold_refresh_token(agent_name: str, content: bytes) -> bytes:
    """One login file as a Judge gets it: its refresh token a placeholder."""
    rules = _LOGIN_RULES.get(agent_name)
    login = _parsed_login(content)
    if rules is None or login is None:
        return content
    withheld = rules.withheld(login)
    return content if withheld is login else json.dumps(withheld).encode("utf-8")


def withheld_values(agent_name: str, content: bytes) -> frozenset[str]:
    """The credentials of one login file that a Judge's copy lacks.

    Its refresh token: the Harness redacts it on the host, but it never
    reaches a Judge, not even in rsi-agent's redaction list.
    """
    login = _parsed_login(content)
    copy = _parsed_login(withhold_refresh_token(agent_name, content))
    if login is None or copy is None:
        return frozenset()
    return frozenset(_secret_strings(login) - _secret_strings(copy))


def login_withheld_values(material: AgentAuthMaterial) -> frozenset[str]:
    """Every login file's ``withheld_values``."""
    return frozenset().union(
        *(
            withheld_values(material.agent_name, entry.content)
            for mount in material.mounts
            for entry in mount.files
        )
    )


def judge_login_copy(material: AgentAuthMaterial) -> AgentAuthMaterial:
    """The Judge's copy of a login: no usable refresh token in any file."""
    mounts = tuple(
        replace(
            mount,
            files=tuple(
                replace(
                    entry,
                    content=withhold_refresh_token(material.agent_name, entry.content),
                )
                for entry in mount.files
            ),
        )
        for mount in material.mounts
    )
    return replace(material, mounts=mounts)


def login_expiry(material: AgentAuthMaterial) -> float | None:
    """When the earliest access token of a login expires (epoch seconds).

    An unusable expiry counts as none; a login read from Work has none
    (current_login refuses it).
    """
    rules = _LOGIN_RULES.get(material.agent_name)
    if rules is None:
        return None
    expiries = []
    for mount in material.mounts:
        for entry in mount.files:
            login = _parsed_login(entry.content)
            try:
                expiry = None if login is None else rules.expiry(login)
            except ValueError:
                expiry = None
            if expiry is not None:
                expiries.append(expiry)
    return min(expiries, default=None)


def login_paths(material: AgentAuthMaterial) -> tuple[PurePosixPath, ...]:
    """Every login file's absolute path inside the Work container."""
    return tuple(
        mount.tmpfs.target / entry.path
        for mount in material.mounts
        for entry in mount.files
    )


def read_login_store(
    material: AgentAuthMaterial, roots: Mapping[PurePosixPath, Path]
) -> dict[PurePosixPath, bytes]:
    """Work's login files from host directories bind-mounted into Work.

    ``roots`` maps each login tmpfs target to its host directory. The same
    checks as a local login: private regular files of this user (never a
    FIFO, which could block the read), no symlinks, bounded size. Raises
    SetupError otherwise.
    """
    files: dict[PurePosixPath, bytes] = {}
    for mount in material.mounts:
        root = roots.get(mount.tmpfs.target)
        if root is None:
            raise SetupError("the Work Agent's login store is unknown")
        directory = _open_directory(
            root, agent_label="Work Agent", label="login directory"
        )
        try:
            for entry in mount.files:
                files[mount.tmpfs.target / entry.path] = _read_auth_file(
                    directory,
                    filename=entry.path.name,
                    uid=os.getuid(),
                    agent_label="Work Agent",
                )
        finally:
            os.close(directory)
    return files


def _new_credentials(
    rules: _LoginRules, login: Mapping[str, Any], known: frozenset[str], path: object
) -> set[str]:
    """The login's credential values; a new one must look like a token."""
    values: set[str] = set()
    for value in rules.credentials(login):
        if not isinstance(value, str):
            continue  # absent, or nothing a provider would accept
        if len(value) > _MAX_CREDENTIAL or (
            value not in known
            and (
                len(value) < _MIN_NEW_CREDENTIAL
                or _TOKEN_CHARACTERS.fullmatch(value) is None
            )
        ):
            raise SetupError(
                f"the Work Agent's login file {path} holds a malformed credential"
            )
        values.add(value)
    return values


def current_login(
    start: AgentAuthMaterial, files: Mapping[PurePosixPath, bytes]
) -> AgentAuthMaterial:
    """Work's login as it is now, checked against ``start``, the last good one.

    ``files`` maps each login file's path in Work to its current bytes.
    Raises SetupError, naming no secret, for a missing, oversized or
    malformed file (a credential that is not token-shaped, an unusable
    expiry), another login kind or another account. The result's
    ``secret_values`` are its credential values alone, never other strings
    the Work Agent wrote.
    """
    rules = _LOGIN_RULES.get(start.agent_name)
    if rules is None:
        raise SetupError(
            f"the {start.agent_name} login cannot be checked; keeping the last copy"
        )
    mounts = []
    secret_values: set[str] = set()
    for mount in start.mounts:
        entries = []
        for entry in mount.files:
            path = mount.tmpfs.target / entry.path
            content = files.get(path)
            if content is None:
                raise SetupError(f"the Work Agent's login file {path} is missing")
            if not content or len(content) > _MAX_AUTH_BYTES:
                raise SetupError(
                    f"the Work Agent's login file {path} is empty or too large"
                )
            login = _parsed_login(content)
            if login is None:
                raise SetupError(
                    f"the Work Agent's login file {path} is not a JSON object"
                )
            before = _parsed_login(entry.content)
            identity = rules.identity(login)
            if identity is None:
                raise SetupError(
                    f"the Work Agent's login file {path} holds no usable login"
                )
            if before is None or identity != rules.identity(before):
                raise SetupError(
                    f"the Work Agent's login file {path} is another login or "
                    "account than at the run start"
                )
            try:
                rules.expiry(login)
            except ValueError:
                raise SetupError(
                    f"the Work Agent's login file {path} has an unusable expiry"
                ) from None
            secret_values |= _new_credentials(rules, login, start.secret_values, path)
            entries.append(replace(entry, content=content))
        mounts.append(replace(mount, files=tuple(entries)))
    return replace(start, mounts=tuple(mounts), secret_values=frozenset(secret_values))


def login_kept_reason(error: BaseException) -> str:
    """Why a round kept the last good login, naming none of the Agent's data.

    The caller still redacts it: a SetupError or InfrastructureError names
    paths and checks, never file contents; anything else (an unexpected
    failure on Agent-written data) is named by its type alone.
    """
    if isinstance(error, (SetupError, InfrastructureError)) and str(error).strip():
        return str(error).strip()
    return f"the Work Agent's login could not be used ({type(error).__name__})"


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
    "MAX_AUTH_BYTES",
    "WITHHELD_REFRESH_TOKEN",
    "codex_login_owner",
    "current_login",
    "judge_login_copy",
    "login_expiry",
    "login_kept_reason",
    "login_paths",
    "login_secret_values",
    "login_withheld_values",
    "read_codex_login",
    "read_login_store",
    "resolve_agent_auth",
    "withhold_refresh_token",
    "withheld_values",
]
