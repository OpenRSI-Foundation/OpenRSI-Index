from __future__ import annotations

import json
import os
import shutil
from pathlib import Path, PurePosixPath
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
        "last_refresh": "2026-09-26",
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
    assert "chatgpt" not in auth.secret_values
    assert "2026-09-26" not in auth.secret_values
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
    assert resolve_agent_auth(source=None, agent_api_key=None, **common) is None
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
            "subscriptionType": "max",
            "scopes": ["user:inference"],
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
    from rsi_harness.runtime.redaction import redact_exact_values

    assert (
        redact_exact_values(
            "softmax(x); max_tokens=2048; user:inference claude-local-access-secret",
            auth.secret_values,
        )
        == "softmax(x); max_tokens=2048; user:inference [REDACTED]"
    )
    safe_auth = json.loads(redact_exact_values(source.read_text(), auth.secret_values))
    assert safe_auth["claudeAiOauth"]["subscriptionType"] == "max"
    assert safe_auth["claudeAiOauth"]["accessToken"] == "[REDACTED]"
    assert "claude-local-access-secret" not in repr(auth)
    assert auth.provider_endpoints == (
        "https://api.anthropic.com",
        "https://platform.claude.com",
    )


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
    assert auth.provider_endpoints == (
        "https://api.anthropic.com",
        "https://platform.claude.com",
    )


def test_claude_provider_without_oauth_login_keeps_default_provider_endpoints(
    tmp_path: Path,
) -> None:
    _write_claude_auth(tmp_path / ".claude", {"apiKey": "static-key-secret"})

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
    assert auth.provider_endpoints == ()
    assert "static-key-secret" in auth.secret_values


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


@pytest.mark.parametrize(
    ("agent_name", "login", "sessions", "store"),
    [
        ("codex", "auth.json", "sessions", ".codex-sessions"),
        ("claude-code", ".credentials.json", "projects", ".claude-projects"),
    ],
)
def test_agent_sessions_outlive_the_private_login_tmpfs(
    tmp_path: Path, agent_name: str, login: str, sessions: str, store: str
) -> None:
    """The login stays on its tmpfs; the CLI's sessions go to Work's rootfs,
    where the retained image keeps them for the next attempt."""
    import subprocess

    from rsi_harness.runtime.local_auth import resolve_agent_auth
    from rsi_loop.harness.agent import create_agent
    from rsi_loop.harness.config import RSILoopConfig

    host = tmp_path / "host"
    if agent_name == "codex":
        _write_auth(host, {"tokens": {"access_token": "local-access-secret"}})
    else:
        _write_claude_auth(host / ".claude", {"claudeAiOauth": {"accessToken": "x"}})
    material = resolve_agent_auth(
        source=AgentAuthSource.LOCAL,
        agent_name=agent_name,
        agent_api_key=None,
        environ={},
        process_uid=os.getuid(),
        effective_uid=os.geteuid(),
        user_lookup=_lookup(host, os.getuid()),
    )
    assert material is not None
    [mount] = material.mounts
    tmpfs = mount.tmpfs.target.relative_to("/home/agent")
    assert [entry.path.as_posix() for entry in mount.files] == [login]
    assert not Path(store).is_relative_to(tmpfs)

    # Work's $HOME, with the tmpfs as the Harness fills it, and a CLI that
    # writes a session and lists the ones it can read.
    home = tmp_path / "work-home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli = "codex" if agent_name == "codex" else "claude"
    (bin_dir / cli).write_text(
        "#!/bin/sh\n"
        f'ls "$HOME/{tmpfs}/{sessions}/"\n'
        f'echo transcript > "$HOME/{tmpfs}/{sessions}/session-$$.jsonl"\n'
    )
    (bin_dir / cli).chmod(0o755)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Do the task.\n")
    command = create_agent(agent_name, RSILoopConfig()).format_run_cmd(
        str(prompt), model="m"
    )

    def launch() -> str:
        (home / tmpfs).mkdir(parents=True)
        (home / tmpfs / login).write_bytes(mount.files[0].content)
        completed = subprocess.run(
            ["/bin/bash", "-c", command],
            env={"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    assert launch() == ""
    assert (home / tmpfs / sessions).resolve() == (home / store).resolve()
    [saved] = (home / store).iterdir()
    # Work stops: the tmpfs and the login are gone, the rootfs is retained.
    shutil.rmtree(home / tmpfs)

    assert launch().split() == [saved.name]
    assert sorted(path.name for path in (home / tmpfs).iterdir()) == sorted(
        [login, sessions]
    )
    assert all(
        b"local-access-secret" not in path.read_bytes()
        for path in (home / store).iterdir()
    )


# -- Antigravity's Google login ------------------------------------------------------

AGY_ACCESS = "ya29.local-access-secret-for-antigravity"
AGY_REFRESH = "1//local-refresh-secret-for-antigravity"
AGY_EMAIL = "someone@example.test"


def _agy_login(**overrides: object) -> dict:
    return {
        "token": {
            "access_token": AGY_ACCESS,
            "token_type": "Bearer",
            "refresh_token": AGY_REFRESH,
            "expiry": "2026-11-17T23:53:26.123456789+08:00",
            "expires_in": 3599,
        },
        "auth_method": "consumer",
        "id_token_email": AGY_EMAIL,
        **overrides,
    }


def _write_agy_auth(home: Path, value: object) -> Path:
    auth = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text(json.dumps(value))
    auth.chmod(0o600)
    return auth


def _resolve_agy(home: Path, **overrides):
    from rsi_harness.runtime.local_auth import resolve_agent_auth

    uid = overrides.pop("process_uid", os.getuid())
    return resolve_agent_auth(
        source=AgentAuthSource.LOCAL,
        agent_name="antigravity",
        agent_api_key=overrides.pop("agent_api_key", None),
        environ=overrides.pop("environ", {}),
        process_uid=uid,
        effective_uid=overrides.pop("effective_uid", os.geteuid()),
        user_lookup=overrides.pop("user_lookup", _lookup(home, uid)),
    )


def test_antigravity_provider_reads_the_google_login_into_its_own_tmpfs(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.local_auth import (
        ANTIGRAVITY_LOGIN_DIR,
        ANTIGRAVITY_LOGIN_FILE,
    )
    from rsi_loop.harness.agent.antigravity import LOGIN_PATH

    auth = _write_agy_auth(tmp_path, _agy_login())
    uid = os.getuid()

    # Under sudo, the invoking user's home, not root's.
    material = _resolve_agy(
        tmp_path,
        environ={"SUDO_UID": str(uid), "HOME": "/root"},
        process_uid=0,
        effective_uid=0,
        user_lookup=_lookup(tmp_path, uid),
    )

    assert material is not None
    assert material.agent_name == "antigravity"
    [mount] = material.mounts
    assert mount.tmpfs.target == ANTIGRAVITY_LOGIN_DIR
    assert mount.tmpfs.options == "rw,nosuid,nodev,noexec,mode=0700"
    [entry] = mount.files
    assert (entry.path.as_posix(), entry.mode) == (ANTIGRAVITY_LOGIN_FILE, 0o600)
    assert entry.content == auth.read_bytes()
    # Where the Agent's launch looks for it.
    assert ANTIGRAVITY_LOGIN_DIR / ANTIGRAVITY_LOGIN_FILE == (
        PurePosixPath("/home/agent") / LOGIN_PATH
    )
    # Its credentials and the account's email, which agy logs; nothing else.
    assert material.secret_values == {AGY_ACCESS, AGY_REFRESH, AGY_EMAIL}
    assert material.provider_endpoints == (
        "https://daily-cloudcode-pa.googleapis.com",
        "https://oauth2.googleapis.com",
    )
    assert AGY_ACCESS not in repr(material)
    # An API key wins, and the file is not read.
    auth.chmod(0o644)
    assert _resolve_agy(tmp_path, agent_api_key="key") is None


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("missing", "not found"),
        # O_NOFOLLOW on a directory symlink fails as no directory.
        ("gemini-symlink", "directory was not found"),
        ("cli-symlink", "directory was not found"),
        ("file-symlink", "symlink|regular file"),
        ("permissions", "permissions"),
        ("array", "JSON object"),
        ("not-json", "JSON object"),
        ("no-token", "no Google login"),
        ("empty-access", "no Google login"),
        ("business", "only a personal Google login"),
        ("no-kind", "only a personal Google login"),
    ),
)
def test_antigravity_provider_rejects_unsafe_or_invalid_logins(
    tmp_path: Path, kind: str, message: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    elsewhere = tmp_path / "elsewhere"
    if kind == "gemini-symlink":
        _write_agy_auth(elsewhere, _agy_login())
        (home / ".gemini").symlink_to(elsewhere / ".gemini")
    elif kind == "cli-symlink":
        _write_agy_auth(elsewhere, _agy_login())
        (home / ".gemini").mkdir()
        (home / ".gemini" / "antigravity-cli").symlink_to(
            elsewhere / ".gemini" / "antigravity-cli"
        )
    elif kind == "file-symlink":
        target = _write_agy_auth(elsewhere, _agy_login())
        link = _write_agy_auth(home, {})
        link.unlink()
        link.symlink_to(target)
    elif kind == "permissions":
        _write_agy_auth(home, _agy_login()).chmod(0o640)
    elif kind == "array":
        _write_agy_auth(home, [])
    elif kind == "not-json":
        _write_agy_auth(home, {}).write_text(f'{{"token": "{AGY_ACCESS}"')
    elif kind == "no-token":
        _write_agy_auth(home, {"auth_method": "consumer"})
    elif kind == "empty-access":
        _write_agy_auth(home, _agy_login(token={"access_token": ""}))
    elif kind == "business":
        _write_agy_auth(home, _agy_login(auth_method="oauth-business"))
    elif kind == "no-kind":
        login = _agy_login()
        del login["auth_method"]
        _write_agy_auth(home, login)

    with pytest.raises(SetupError, match=message) as caught:
        _resolve_agy(home)
    assert AGY_ACCESS not in str(caught.value)
    assert "elsewhere" not in str(caught.value)


def test_antigravity_login_stays_on_its_tmpfs_and_sessions_on_the_rootfs(
    tmp_path: Path,
) -> None:
    """The launch links the tmpfs login and agy's log directory into agy's
    directory, which (with agy's sessions) stays on Work's rootfs; agy's
    refresh and logs, written through the links, stay on the tmpfs."""
    import subprocess

    from rsi_loop.harness.agent import create_agent
    from rsi_loop.harness.config import RSILoopConfig

    material = _resolve_agy(_write_agy_auth(tmp_path / "host", _agy_login()).parents[2])
    assert material is not None
    [mount] = material.mounts
    tmpfs = mount.tmpfs.target.relative_to("/home/agent")
    home = tmp_path / "work-home"
    agy_dir = home / ".gemini" / "antigravity-cli"
    login = agy_dir / "antigravity-oauth-token"
    agent = create_agent("antigravity", RSILoopConfig(agent_local_auth=True))
    prelude = agent.format_run_cmd("/tmp/prompt.md", model="m").rpartition("\n")[0]
    # agy: a session on its own directory, a log naming the account (and
    # cli.log linking to it), and a token refresh in place.
    agy = (
        'mkdir -p "$HOME/.gemini/antigravity-cli/conversations" && '
        'echo session > "$HOME/.gemini/antigravity-cli/conversations/c-$$.db" && '
        'cd "$HOME/.gemini/antigravity-cli" && mkdir -p log && '
        f'echo "authenticated successfully as {AGY_EMAIL}" > log/cli-$$.log && '
        "ln -sfn log/cli-$$.log cli.log && "
        'token="$HOME/.gemini/antigravity-cli/antigravity-oauth-token" && '
        'content=$(cat "$token") && '
        'printf %s "${content/local-access/refreshed-access}" > "$token"'
    )

    def launch(keep: bool = False) -> subprocess.CompletedProcess[str]:
        if not keep:
            (home / tmpfs).mkdir(parents=True)
            (home / tmpfs / mount.files[0].path).write_bytes(mount.files[0].content)
        return subprocess.run(
            ["/bin/bash", "-c", f"{prelude}\n{agy}"],
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=False,
        )

    assert launch().returncode == 0
    assert login.is_symlink()
    assert os.readlink(login) == "../antigravity-login/antigravity-oauth-token"
    assert (
        b"refreshed-access-secret" in (home / tmpfs / mount.files[0].path).read_bytes()
    )
    # The Agent puts a file in the link's place; Work stops (the tmpfs goes).
    login.unlink()
    login.write_text("left on the rootfs")
    shutil.rmtree(home / tmpfs)

    assert launch().returncode == 0
    assert login.is_symlink()
    assert len(list((agy_dir / "conversations").iterdir())) == 2
    # A log the Agent left on the rootfs goes too.
    (agy_dir / "log").unlink()
    (agy_dir / "log").mkdir()
    (agy_dir / "log" / "old.log").write_text(AGY_EMAIL)

    assert launch(keep=True).returncode == 0
    assert (agy_dir / "log").is_symlink()
    assert len(list((home / tmpfs / "log").iterdir())) == 2
    assert all(
        b"secret" not in path.read_bytes()
        and AGY_EMAIL.encode() not in path.read_bytes()
        for path in home.rglob("*")
        if not path.is_symlink()
        and path.is_file()
        and not path.is_relative_to(home / tmpfs)
    )
    # Without the login, the launch stops before agy starts.
    shutil.rmtree(home / tmpfs)
    (home / tmpfs).mkdir()
    completed = subprocess.run(
        ["/bin/bash", "-c", f"{prelude}\necho started"],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "started" not in completed.stdout
    assert "Antigravity login is missing" in completed.stderr
