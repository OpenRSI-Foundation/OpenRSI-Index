from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import AgentAuthSource


def _write_auth(home: Path, value: object) -> Path:
    auth = home / ".codex" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps(value))
    auth.chmod(0o600)
    return auth


def _write_claude_auth(directory: Path, value: object) -> Path:
    auth = directory / ".credentials.json"
    directory.mkdir(parents=True, exist_ok=True)
    auth.write_text(json.dumps(value))
    auth.chmod(0o600)
    return auth


def _lookup(home: Path, uid: int):
    return lambda selected: SimpleNamespace(
        pw_dir=str(home), pw_uid=uid, pw_name=f"user-{selected}"
    )


def _resolve(home: Path, **overrides):
    from rsi_harness.runtime.local_auth import resolve_agent_auth

    uid = overrides.pop("process_uid", os.getuid())
    effective_uid = overrides.pop("effective_uid", os.geteuid())
    user_lookup = overrides.pop("user_lookup", _lookup(home, uid))
    return resolve_agent_auth(
        source=AgentAuthSource.LOCAL,
        agent_name="codex",
        agent_api_key=None,
        environ={},
        process_uid=uid,
        effective_uid=effective_uid,
        user_lookup=user_lookup,
        **overrides,
    )


def test_resolver_returns_secret_free_validated_in_memory_auth(tmp_path: Path) -> None:
    raw = {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "local-access-secret",
            "refresh_token": "local-refresh-secret",
        },
    }
    auth_file = _write_auth(tmp_path, raw)

    auth = _resolve(tmp_path)

    content = auth.mounts[0].files[0].content
    assert json.loads(content) == raw
    assert auth_file.read_bytes() == content
    assert "local-access-secret" in auth.secret_values
    assert "local-refresh-secret" in auth.secret_values
    assert content.decode() in auth.secret_values
    assert "local-access-secret" not in repr(auth)
    assert str(auth_file) not in repr(auth)
    assert auth.agent_name == "codex"
    assert len(auth.mounts) == 1
    assert auth.mounts[0].tmpfs.target.as_posix() == "/home/agent/.codex"
    assert auth.mounts[0].files[0].path.as_posix() == "auth.json"
    assert auth.provider_endpoints == (
        "https://chatgpt.com/backend-api/codex",
        "https://auth.openai.com",
    )


def test_resolver_dispatches_through_agent_provider_registry() -> None:
    from rsi_harness.runtime.local_auth import (
        AgentAuthMaterial,
        resolve_agent_auth,
    )

    expected = AgentAuthMaterial(
        agent_name="future-agent", mounts=(), secret_values=frozenset()
    )

    class Provider:
        def resolve(self, **_kwargs):
            return expected

    assert (
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL,
            agent_name="future-agent",
            agent_api_key=None,
            providers={"future-agent": Provider()},
        )
        is expected
    )


def test_resolver_uses_sudo_uid_home_instead_of_root_home(tmp_path: Path) -> None:
    uid = os.getuid()
    _write_auth(tmp_path, {"tokens": {"access_token": "sudo-user-secret"}})
    selected: list[int] = []

    def lookup(value: int):
        selected.append(value)
        return SimpleNamespace(pw_dir=str(tmp_path), pw_uid=uid, pw_name="nsl")

    from rsi_harness.runtime.local_auth import resolve_agent_auth

    auth = resolve_agent_auth(
        source=AgentAuthSource.LOCAL,
        agent_name="codex",
        agent_api_key=None,
        environ={"SUDO_UID": str(uid), "HOME": "/root"},
        process_uid=0,
        effective_uid=0,
        user_lookup=lookup,
    )

    assert selected == [uid]
    assert auth is not None
    assert "sudo-user-secret" in auth.secret_values


def test_disabled_or_api_key_auth_never_reads_local_file(tmp_path: Path) -> None:
    from rsi_harness.runtime.local_auth import resolve_agent_auth

    def forbidden(_uid: int):
        raise AssertionError("local user lookup must not run")

    common = {
        "agent_name": "codex",
        "environ": {},
        "process_uid": os.getuid(),
        "effective_uid": os.geteuid(),
        "user_lookup": forbidden,
    }
    assert (
        resolve_agent_auth(source=None, agent_api_key=None, **common)
        is None
    )
    assert (
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL,
            agent_api_key="configured-api-key",
            **common,
        )
        is None
    )


def test_claude_provider_reads_standard_login_into_private_work_tmpfs(
    tmp_path: Path,
) -> None:
    raw = {
        "claudeAiOauth": {
            "accessToken": "claude-local-access-secret",
            "refreshToken": "claude-local-refresh-secret",
        }
    }
    source = _write_claude_auth(tmp_path / ".claude", raw)

    from rsi_harness.runtime.local_auth import resolve_agent_auth

    auth = resolve_agent_auth(
        source=AgentAuthSource.LOCAL,
        agent_name="claude-code",
        agent_api_key=None,
        environ={},
        process_uid=os.getuid(),
        effective_uid=os.geteuid(),
        user_lookup=_lookup(tmp_path, os.getuid()),
    )

    assert auth is not None
    assert auth.agent_name == "claude-code"
    assert auth.mounts[0].tmpfs.target.as_posix() == "/home/agent/.claude"
    assert auth.mounts[0].files[0].path.as_posix() == ".credentials.json"
    assert auth.mounts[0].files[0].content == source.read_bytes()
    assert "claude-local-access-secret" in auth.secret_values
    assert "claude-local-refresh-secret" in auth.secret_values
    assert "claude-local-access-secret" not in repr(auth)


def test_claude_provider_reads_absolute_config_dir_without_forwarding_host_path(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "custom-claude-config"
    source = _write_claude_auth(
        config_dir, {"claudeAiOauth": {"accessToken": "custom-dir-secret"}}
    )

    from rsi_harness.runtime.local_auth import resolve_agent_auth

    auth = resolve_agent_auth(
        source=AgentAuthSource.LOCAL,
        agent_name="claude-code",
        agent_api_key=None,
        environ={"CLAUDE_CONFIG_DIR": str(config_dir)},
        process_uid=os.getuid(),
        effective_uid=os.geteuid(),
        user_lookup=_lookup(tmp_path, os.getuid()),
    )

    assert auth is not None
    assert auth.mounts[0].tmpfs.target.as_posix() == "/home/agent/.claude"
    assert auth.mounts[0].files[0].content == source.read_bytes()
    assert "custom-dir-secret" in auth.secret_values


def test_local_auth_option_rejects_unregistered_agent_before_lookup() -> None:
    from rsi_harness.runtime.local_auth import resolve_agent_auth

    with pytest.raises(SetupError, match="not supported.*future-agent"):
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL,
            agent_name="future-agent",
            agent_api_key=None,
            environ={},
            process_uid=os.getuid(),
            effective_uid=os.geteuid(),
            user_lookup=lambda _uid: (_ for _ in ()).throw(
                AssertionError("lookup must not run")
            ),
        )


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("missing", "not found"),
        ("symlink", "symlink|regular file"),
        ("owner", "owned"),
        ("permissions", "permissions"),
        ("oversize", "size limit"),
        ("array", "JSON object"),
    ),
)
def test_resolver_rejects_unsafe_or_invalid_auth_files(
    tmp_path: Path, kind: str, message: str
) -> None:
    uid = os.getuid()
    if kind == "symlink":
        target = tmp_path / "actual-auth.json"
        target.write_text("{}")
        target.chmod(0o600)
        auth = tmp_path / ".codex" / "auth.json"
        auth.parent.mkdir()
        auth.symlink_to(target)
    elif kind == "oversize":
        auth = _write_auth(tmp_path, {})
        auth.write_bytes(b"{" + b"x" * (1024 * 1024) + b"}")
    elif kind == "array":
        _write_auth(tmp_path, [])
    elif kind != "missing":
        auth = _write_auth(tmp_path, {})
        if kind == "permissions":
            auth.chmod(0o640)

    expected_uid = uid + 1 if kind == "owner" else uid
    with pytest.raises(SetupError, match=message) as caught:
        _resolve(
            tmp_path,
            process_uid=expected_uid,
            effective_uid=expected_uid,
            user_lookup=_lookup(tmp_path, expected_uid),
        )
    assert "actual-auth.json" not in str(caught.value)


def test_resolver_rejects_malformed_json_without_echoing_content(
    tmp_path: Path,
) -> None:
    secret = "MALFORMED-LOCAL-AUTH-SECRET"
    auth = _write_auth(tmp_path, {})
    auth.write_text(f'{{"access_token":"{secret}"')
    auth.chmod(0o600)

    with pytest.raises(SetupError, match="valid JSON") as caught:
        _resolve(tmp_path)

    assert secret not in str(caught.value)
