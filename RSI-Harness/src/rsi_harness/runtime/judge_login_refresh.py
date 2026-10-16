"""Keep an operator's Codex login fresh for the Judge LLM, on the host.

``rsi-harness run --judge-llm-auth codex-login`` hands each Judge the access
token of the operator's ChatGPT login, never its refresh token, so a Judge
cannot refresh it. Before each Judge round the Harness instead re-reads the
login and, when its access token would not outlive the round, has Codex
itself refresh it: the pinned Codex CLI (the Judge's own, SHA-256-checked
again here) runs ``codex app-server`` and asks ``account/read`` with
``refreshToken: true``, which runs Codex's normal refresh flow and saves the
rotated tokens to the operator's ``auth.json`` in place, as any Codex session
does. The Harness never calls the OAuth endpoint itself. (In the pinned
release: ``GetAccountParams.refresh_token`` in app-server-protocol's
v2/account.rs; the app-server's ``refresh_token_if_requested`` calls
``AuthManager::refresh_token``, whose ``persist_tokens`` saves through the
file store before ``account/read`` answers. A failed refresh is only logged
there: the login left unchanged is what shows it.)

Codex runs as the operator (the ``sudo`` user under sudo), in a private
``CODEX_HOME`` holding nothing but a link to the operator's ``auth.json``, so
it reads no user config (a system-wide ``/etc/codex`` one still applies) and
leaves nothing behind, with every feature off, in an empty working directory,
for a bounded time. The login is read back with the run-start checks
whatever Codex answered: a refresh it saved is used even when it then failed
to answer. Other failures keep the Judge's current copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import select
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi_harness.judge_agent import rsi_agent

# A refresh happens when the access token would expire within the round's
# Judge timeout plus Codex's own refresh window plus this much: the Judge's
# setup (snapshot, container, rsi-agent) before its timeout starts. rsi-agent
# checks a call's --timeout only up to the round's end, so this suffices
# whatever --timeout a Judge passes.
JUDGE_LOGIN_MARGIN_SECONDS = 600
# Codex's start, its refresh and its answer.
REFRESH_TIMEOUT_SECONDS = 60
# Codex sends its refresh request with no timeout of its own: one still in
# flight when Codex does not answer in time gets this much longer to save its
# result, since the server may already have spent the old refresh token.
IN_FLIGHT_GRACE_SECONDS = 120
# After Codex answered: its own shutdown, before it is stopped.
_EXIT_GRACE_SECONDS = 5
# How often a late refresh is looked for.
_SAVED_POLL_SECONDS = 0.5
# Codex's longest run here, its stop included.
MAX_REFRESH_SECONDS = (
    REFRESH_TIMEOUT_SECONDS + IN_FLIGHT_GRACE_SECONDS + _EXIT_GRACE_SECONDS + 4
)
# Host variables Codex may need to reach the auth server (a corporate CA).
_PASSED_ENVIRONMENT = ("SSL_CERT_FILE", "SSL_CERT_DIR")
_MACHINES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}
_MAX_LINE = 1 << 20


class LoginRefreshError(Exception):
    """Codex could not be run to refresh the login (a one-line reason).

    ``in_flight``: Codex did not answer in time, so a refresh it sent may
    have reached the server and never been saved.
    """

    def __init__(self, reason: str, *, in_flight: bool = False) -> None:
        super().__init__(reason)
        self.in_flight = in_flight


@dataclass(frozen=True, slots=True, repr=False)
class OperatorCodexLogin:
    """The operator's Codex login: where it lives, whose it is, how to read it.

    ``read`` re-reads ``auth.json`` with the run-start checks (a private,
    non-symlink regular file owned by ``uid``). ``proxy`` is the Judge LLM's
    proxy, which Codex uses here too.
    """

    codex_home: Path
    uid: int
    gid: int
    read: Callable[[], bytes]
    proxy: str | None = None

    def __repr__(self) -> str:
        return (
            f"OperatorCodexLogin(codex_home={str(self.codex_home)!r}, uid={self.uid})"
        )


@dataclass(frozen=True, slots=True)
class LoginRenewal:
    """A round's operator login: ``content`` to hand the Judge (None keeps its
    current copy) and a ``warning`` when the login was not kept fresh."""

    content: bytes | None
    warning: str | None = None


def needed_lifetime(round_seconds: float) -> float:
    """How long the access token must still live when a round starts."""
    return (
        round_seconds
        + rsi_agent.CODEX_REFRESH_WINDOW_SECONDS
        + JUDGE_LOGIN_MARGIN_SECONDS
    )


def renew_operator_login(
    login: OperatorCodexLogin,
    *,
    codex_cli: Path | None,
    round_seconds: float,
    now: Callable[[], float] = time.time,
    refresh: Callable[..., str | None] | None = None,
) -> LoginRenewal:
    """The operator's current login for a round, refreshed by Codex if needed."""
    try:
        content = login.read()
        expiry = _login_expiry(content)
    except Exception as error:  # the file may be mid-write by another Codex
        return LoginRenewal(None, f"the login could not be read: {_reason(error)}")
    lifetime = needed_lifetime(round_seconds)
    if expiry is None or expiry - now() > lifetime:
        return LoginRenewal(content)
    run = refresh_codex_login if refresh is None else refresh

    def saved() -> bool:
        try:
            later = _login_expiry(login.read())
        except Exception:  # absent, or mid-write
            return False
        return later is not None and later > expiry

    answer = failure = None
    try:
        answer = run(
            login,
            None if codex_cli is None else codex_cli / "codex",
            timeout=REFRESH_TIMEOUT_SECONDS,
            saved=saved,
        )
    except LoginRefreshError as error:
        failure = f"Codex could not refresh it: {_reason(error)}"
        if error.in_flight:
            failure += (
                "; a refresh it had sent may have spent the refresh token: if "
                "the next ones fail too, log in again with codex login"
            )
    except Exception as error:
        failure = f"Codex could not refresh it: {_reason(error)}"
    # Whatever Codex answered: it saves a refresh before it answers.
    try:
        renewed = login.read()
        renewed_expiry = _login_expiry(renewed)
    except Exception as error:
        unread = f"the refreshed login could not be read: {_reason(error)}"
        return LoginRenewal(content, failure or unread)
    if renewed_expiry is None or renewed_expiry <= expiry:
        if failure is not None:
            return LoginRenewal(content, failure)
        # Codex answers account/read normally after a failed refresh: an
        # error here is about something else (its account request).
        detail = f" (Codex answered: {answer})" if answer else ""
        return LoginRenewal(content, f"Codex did not refresh it{detail}")
    left = renewed_expiry - now()
    if left <= lifetime:
        return LoginRenewal(
            renewed,
            f"even refreshed, its access token expires in {int(left)} seconds, "
            "before this round could end",
        )
    return LoginRenewal(renewed)


def refresh_codex_login(
    login: OperatorCodexLogin,
    codex_binary: Path | None,
    *,
    timeout: float = REFRESH_TIMEOUT_SECONDS,
    saved: Callable[[], bool] | None = None,
    environment: Mapping[str, str] | None = None,
    effective_uid: int | None = None,
) -> str | None:
    """Run the pinned Codex CLI so that it refreshes ``login`` in place.

    Returns the error Codex answered ``account/read`` with (None when it
    answered normally). Codex reports no failed refresh there: whether the
    login changed is for the caller to read back. When Codex does not answer
    within ``timeout`` it still gets ``IN_FLIGHT_GRACE_SECONDS``, until
    ``saved()`` says the refreshed login is on disk, before it is stopped.
    ``environment`` adds variables to Codex's (tests only).
    """
    if codex_binary is None:
        raise LoginRefreshError("the pinned Codex CLI was not prepared")
    expected = host_codex_sha256()
    if expected is None:
        raise LoginRefreshError(
            f"the pinned Codex CLI has no release for this host ({platform.machine()})"
        )
    root_uid = os.geteuid() if effective_uid is None else effective_uid
    as_operator = root_uid == 0 and login.uid != 0
    binary_fd = _open_verified(codex_binary, expected)
    scratch = tempfile.mkdtemp(prefix="rsi-judge-login-")
    try:
        home = os.path.join(scratch, "codex-home")
        work = os.path.join(scratch, "work")
        os.mkdir(home, 0o700)
        os.mkdir(work, 0o700)
        # Codex reads and saves CODEX_HOME/auth.json: the operator's file.
        os.symlink(login.codex_home / "auth.json", os.path.join(home, "auth.json"))
        if as_operator:
            for path in (scratch, home, work):
                os.chown(path, login.uid, login.gid, follow_symlinks=False)
        argv = ["codex"]
        for item in rsi_agent.CODEX_LLM_CONFIG:
            argv += ["-c", item]
        argv.append("app-server")
        try:
            process = subprocess.Popen(
                argv,
                # The verified bytes themselves; the operator need not reach
                # the Harness's data root.
                executable=f"/proc/self/fd/{binary_fd}",
                pass_fds=(binary_fd,),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=work,
                env=_codex_environment(login, home, environment),
                start_new_session=True,
                **(
                    {"user": login.uid, "group": login.gid, "extra_groups": []}
                    if as_operator
                    else {}
                ),
            )
        except OSError as error:
            raise LoginRefreshError(
                f"cannot start the Codex CLI: {error.strerror}"
            ) from None
        try:
            return _ask_for_refresh(process, time.monotonic() + timeout)
        except LoginRefreshError as error:
            if error.in_flight:
                _await_saved(process, saved, time.monotonic() + IN_FLIGHT_GRACE_SECONDS)
            raise
        finally:
            _stop(process, time.monotonic() + _EXIT_GRACE_SECONDS)
    finally:
        os.close(binary_fd)
        shutil.rmtree(scratch, ignore_errors=True)


def host_codex_sha256() -> str | None:
    """The pinned Codex release's SHA-256 for this host, if it has one."""
    from rsi_loop.harness.agent import versions

    machine = _MACHINES.get(platform.machine().lower())
    if machine is None:
        return None
    return getattr(versions, f"CODEX_SHA256_{machine.upper()}", None)


def _open_verified(path: Path, expected: str) -> int:
    """An open descriptor of ``path`` whose bytes are the pinned release."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise LoginRefreshError(
            f"the pinned Codex CLI is not available: {error.strerror}"
        ) from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LoginRefreshError("the pinned Codex CLI is not a file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        if digest.hexdigest() != expected:
            raise LoginRefreshError(
                "the prepared Codex CLI is not the pinned release for this host"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _codex_environment(
    login: OperatorCodexLogin, home: str, extra: Mapping[str, str] | None
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": home,
        "CODEX_HOME": home,
        "NO_COLOR": "1",
    }
    environment.update(
        (name, os.environ[name]) for name in _PASSED_ENVIRONMENT if os.environ.get(name)
    )
    if login.proxy:
        environment.update(https_proxy=login.proxy, HTTPS_PROXY=login.proxy)
    environment.update(extra or {})
    return environment


def _ask_for_refresh(process: subprocess.Popen, deadline: float) -> str | None:
    """initialize, then account/read with refreshToken; Codex's error, if any."""
    lines = _Lines(process, deadline)
    lines.send(
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "rsi_harness", "title": None, "version": "1"}
            },
        }
    )
    _answer(lines, 1)
    lines.send({"method": "initialized"})
    # Codex's managed refresh flow, then its account; saved before it answers.
    lines.send({"id": 2, "method": "account/read", "params": {"refreshToken": True}})
    error = _answer(lines, 2)
    if error is None:
        return None
    message = error.get("message") if isinstance(error, dict) else None
    return str(message)[:200] if message else "an error"


def _answer(lines: _Lines, identifier: int) -> Any:
    """Wait for the answer to request ``identifier``; its error or None."""
    while True:
        message = lines.receive()
        if "method" in message and "id" in message:
            # A request from Codex: nothing here may act or ask.
            lines.send(
                {
                    "id": message["id"],
                    "error": {"code": -32601, "message": "not supported"},
                }
            )
            continue
        if message.get("id") == identifier and "method" not in message:
            return message.get("error")


class _Lines:
    """JSON lines from Codex's stdout, each by ``deadline``."""

    def __init__(self, process: subprocess.Popen, deadline: float) -> None:
        self.process = process
        self.deadline = deadline
        self.buffer = b""

    def send(self, message: dict) -> None:
        try:
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            self.process.stdin.flush()
        except OSError:
            raise LoginRefreshError("the Codex CLI exited early") from None

    def receive(self) -> dict:
        while True:
            line, newline, rest = self.buffer.partition(b"\n")
            if newline:
                self.buffer = rest
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    return message
                continue
            if len(self.buffer) > _MAX_LINE:
                raise LoginRefreshError("the Codex CLI wrote an oversized line")
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise LoginRefreshError(
                    "the Codex CLI did not answer in time", in_flight=True
                )
            stdout = self.process.stdout
            ready, _, _ = select.select([stdout], [], [], left)
            if not ready:
                continue
            chunk = os.read(stdout.fileno(), 65536)
            if not chunk:
                code = self.process.poll()
                how = "" if code is None else f" with status {code}"
                raise LoginRefreshError(f"the Codex CLI exited{how} before answering")
            self.buffer += chunk


def _await_saved(
    process: subprocess.Popen, saved: Callable[[], bool] | None, deadline: float
) -> None:
    """Until ``deadline``: wait for Codex to save a refresh, or to exit."""
    stdout = process.stdout
    while process.poll() is None and (left := deadline - time.monotonic()) > 0:
        if saved is not None and saved():
            return
        ready, _, _ = select.select([stdout], [], [], min(left, _SAVED_POLL_SECONDS))
        # Drain, so a full pipe never keeps Codex from going on.
        if ready and not os.read(stdout.fileno(), 65536):
            return


def _stop(process: subprocess.Popen, deadline: float) -> None:
    """Let Codex shut down on stdin's end, then stop its session."""
    try:
        process.stdin.close()
    except OSError:
        pass
    stdout = process.stdout
    while process.poll() is None and (left := deadline - time.monotonic()) > 0:
        # Drain, so a full pipe never keeps it from exiting.
        ready, _, _ = select.select([stdout], [], [], left)
        if ready and not os.read(stdout.fileno(), 65536):
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            break
    if process.poll() is None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except OSError:
                pass
            try:
                process.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                continue
    stdout.close()


def _login_expiry(content: bytes) -> float | None:
    """The access token's expiry of a ChatGPT login; ValueError otherwise."""
    login = json.loads(content.decode("utf-8"))
    tokens = login.get("tokens") if isinstance(login, dict) else None
    if (
        not isinstance(tokens, dict)
        or login.get("auth_mode") not in (None, "chatgpt")
        or not isinstance(tokens.get("access_token"), str)
        or not isinstance(tokens.get("refresh_token"), str)
    ):
        raise ValueError("it is no longer a ChatGPT login")
    return rsi_agent.access_token_expiry(content.decode("utf-8"))


def _reason(error: BaseException) -> str:
    text = str(error).strip().splitlines()
    return text[0][:200] if text else type(error).__name__


__all__ = [
    "IN_FLIGHT_GRACE_SECONDS",
    "JUDGE_LOGIN_MARGIN_SECONDS",
    "LoginRefreshError",
    "LoginRenewal",
    "MAX_REFRESH_SECONDS",
    "OperatorCodexLogin",
    "REFRESH_TIMEOUT_SECONDS",
    "host_codex_sha256",
    "needed_lifetime",
    "refresh_codex_login",
    "renew_operator_login",
]
