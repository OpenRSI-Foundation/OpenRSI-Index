"""The operator's Codex login for the Judge LLM, kept fresh on the host by the
pinned Codex CLI itself; a Judge never refreshes it."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from rsi_harness.judge_agent import rsi_agent
from rsi_harness.models import JudgeLLMAuth, JudgeLLMRequest, SubmissionStatus
from rsi_harness.runtime import judge_login_refresh
from rsi_harness.runtime.judge import JudgeRunner
from rsi_harness.runtime.judge_llm import resolve_judge_llm
from rsi_harness.runtime.judge_login_refresh import (
    JUDGE_LOGIN_MARGIN_SECONDS,
    LoginRefreshError,
    OperatorCodexLogin,
    needed_lifetime,
    refresh_codex_login,
    renew_operator_login,
)
from rsi_harness.runtime.judge_model_access import resolve_judge_model_access
from rsi_harness.runtime.local_auth import WITHHELD_REFRESH_TOKEN, read_codex_login
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan
from tests.runtime.test_judge import evaluate, make_harness, managed_workdir_volume
from tests.runtime.test_judge_model_access import ENDPOINT

ROUND = 1800.0


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"h.{payload.decode()}.s"


def login(generation: str, expires_at: float) -> bytes:
    claims = {
        "exp": int(expires_at),
        "generation": generation,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acc-1"},
    }
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": _jwt({**claims, "kind": "id"}),
                "access_token": _jwt(claims),
                "refresh_token": f"REFRESH-{generation}-synthetic-refresh-token",
                "account_id": "acc-1",
            },
            "last_refresh": "2026-10-14T00:00:00Z",
        }
    ).encode()


def access_token(content: bytes | str) -> str:
    return json.loads(content)["tokens"]["access_token"]


def codex_home(tmp_path: Path, content: bytes) -> Path:
    home = tmp_path / "operator-codex"
    home.mkdir(mode=0o700)
    (home / "auth.json").write_bytes(content)
    (home / "auth.json").chmod(0o600)
    return home


def operator(home: Path, **options) -> OperatorCodexLogin:
    return OperatorCodexLogin(
        codex_home=home,
        uid=os.getuid(),
        gid=os.getgid(),
        read=lambda: read_codex_login(home, environ={}),
        **options,
    )


class Refresh:
    """A refresh that rewrites the login as Codex would, and records calls."""

    def __init__(self, home: Path, renewed: bytes | None, answer=None, error=None):
        self.home, self.renewed, self.answer, self.error = home, renewed, answer, error
        self.calls: list = []

    def __call__(self, login, binary, *, timeout, saved=None):
        self.calls.append((login, binary, timeout))
        self.saved = saved
        if self.renewed is not None:
            (self.home / "auth.json").write_bytes(self.renewed)
        if self.error is not None:
            raise self.error
        return self.answer


# -- when a refresh happens --------------------------------------------------------


def test_the_needed_lifetime_is_the_round_codexs_window_and_a_margin():
    assert needed_lifetime(ROUND) == (
        ROUND + rsi_agent.CODEX_REFRESH_WINDOW_SECONDS + JUDGE_LOGIN_MARGIN_SECONDS
    )


@pytest.mark.parametrize(
    ("left", "refreshed"),
    [
        (needed_lifetime(ROUND) + 60, False),
        (86400, False),
        (needed_lifetime(ROUND) - 60, True),
        (ROUND + 60, True),  # outlives the round, not Codex's window and setup
        (rsi_agent.CODEX_REFRESH_WINDOW_SECONDS - 1, True),  # Codex's own window
        (-60, True),  # already expired
    ],
)
def test_a_refresh_happens_only_when_the_token_would_not_outlive_the_round(
    tmp_path, left, refreshed
):
    now = time.time()
    current = login("1", now + left)
    home = codex_home(tmp_path, current)
    renewed = login("2", now + 86400)
    refresh = Refresh(home, renewed)
    cli = tmp_path / "judge-codex"

    renewal = renew_operator_login(
        operator(home),
        codex_cli=cli,
        round_seconds=ROUND,
        now=lambda: now,
        refresh=refresh,
    )

    assert renewal.warning is None
    if refreshed:
        [(_, binary, timeout)] = refresh.calls
        assert binary == cli / "codex"
        assert timeout == judge_login_refresh.REFRESH_TIMEOUT_SECONDS
        assert renewal.content == renewed
    else:
        assert refresh.calls == []
        assert renewal.content == current


def test_a_fresh_login_from_the_operators_other_sessions_is_picked_up(tmp_path):
    # Another Codex session refreshed the file: no refresh, its tokens used.
    home = codex_home(tmp_path, login("1", time.time() + 86400))
    (home / "auth.json").write_bytes(login("other", time.time() + 86400))
    refresh = Refresh(home, None)

    renewal = renew_operator_login(
        operator(home), codex_cli=None, round_seconds=ROUND, refresh=refresh
    )

    assert refresh.calls == [] and renewal.warning is None
    assert access_token(renewal.content) == access_token(
        login("other", time.time() + 86400)
    )


# -- failures keep the current copy -----------------------------------------------


def test_a_failed_refresh_keeps_the_copy_with_a_warning(tmp_path):
    current = login("1", time.time() + 120)
    home = codex_home(tmp_path, current)
    refresh = Refresh(home, None, error=LoginRefreshError("it did not answer in time"))

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND, refresh=refresh
    )

    assert renewal.content == current
    assert renewal.warning == "Codex could not refresh it: it did not answer in time"


def test_a_refresh_that_may_have_been_in_flight_says_how_to_recover(tmp_path):
    current = login("1", time.time() + 120)
    home = codex_home(tmp_path, current)
    error = LoginRefreshError("it did not answer in time", in_flight=True)
    refresh = Refresh(home, None, error=error)

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND, refresh=refresh
    )

    assert renewal.content == current
    assert renewal.warning.startswith(
        "Codex could not refresh it: it did not answer in time; "
    )
    assert "log in again with codex login" in renewal.warning
    # What tells Codex's run that a late refresh was saved: the login on disk.
    assert refresh.saved() is False
    (home / "auth.json").write_bytes(b'{"tokens": ')  # mid-write
    assert refresh.saved() is False
    (home / "auth.json").write_bytes(login("2", time.time() + 86400))
    assert refresh.saved() is True


def test_a_refresh_codex_saved_before_failing_is_used(tmp_path):
    # Codex saves the rotated tokens, then stalls on its account request.
    home = codex_home(tmp_path, login("1", time.time() + 120))
    renewed = login("2", time.time() + 86400)
    error = LoginRefreshError("the Codex CLI did not answer in time", in_flight=True)
    refresh = Refresh(home, renewed, error=error)

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND, refresh=refresh
    )

    assert renewal == judge_login_refresh.LoginRenewal(renewed)


def test_a_refresh_that_changed_nothing_keeps_the_copy_with_a_warning(tmp_path):
    current = login("1", time.time() + 120)
    home = codex_home(tmp_path, current)
    refresh = Refresh(home, None, answer="workspace routing discovery failed")

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND, refresh=refresh
    )

    assert renewal.content == current
    assert renewal.warning == (
        "Codex did not refresh it (Codex answered: workspace routing discovery failed)"
    )


def test_codexs_error_answer_is_no_failure_once_the_login_was_refreshed(tmp_path):
    # account/read refreshes first and may then fail on the account itself.
    home = codex_home(tmp_path, login("1", time.time() + 120))
    renewed = login("2", time.time() + 86400)
    refresh = Refresh(home, renewed, answer="workspace routing discovery failed")

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND, refresh=refresh
    )

    assert renewal == judge_login_refresh.LoginRenewal(renewed)


def test_an_unreadable_login_keeps_the_copy_with_a_warning(tmp_path):
    home = codex_home(tmp_path, login("1", time.time() + 120))
    (home / "auth.json").chmod(0o644)
    refresh = Refresh(home, None)

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND, refresh=refresh
    )

    assert renewal.content is None and refresh.calls == []
    assert renewal.warning.startswith("the login could not be read: ")
    assert "group and other access" in renewal.warning


def test_a_login_that_is_no_longer_chatgpt_is_not_used(tmp_path):
    home = codex_home(tmp_path, b'{"auth_mode": "apikey", "OPENAI_API_KEY": "sk-x"}')

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND
    )

    assert renewal.content is None
    assert "no longer a ChatGPT login" in renewal.warning


def test_a_refresh_too_short_for_the_round_is_used_with_a_warning(tmp_path):
    home = codex_home(tmp_path, login("1", time.time() + 120))
    renewed = login("2", time.time() + 900)
    refresh = Refresh(home, renewed)

    renewal = renew_operator_login(
        operator(home), codex_cli=tmp_path, round_seconds=ROUND, refresh=refresh
    )

    assert renewal.content == renewed
    assert "before this round could end" in renewal.warning


# -- running Codex -------------------------------------------------------------------


@pytest.fixture
def popen(monkeypatch):
    """Record Popen's arguments; Codex itself never starts."""
    calls: list = []

    def fake(argv, **options):
        calls.append((argv, options))
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(judge_login_refresh.subprocess, "Popen", fake)
    return calls


@pytest.fixture
def pinned_binary(tmp_path, monkeypatch):
    binary = tmp_path / "judge-codex" / "codex"
    binary.parent.mkdir()
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr(judge_login_refresh, "host_codex_sha256", lambda: digest)
    return binary


def test_codex_runs_as_the_operator_never_root(
    tmp_path, monkeypatch, popen, pinned_binary
):
    home = codex_home(tmp_path, login("1", time.time() + 120))
    owners: list = []
    monkeypatch.setattr(
        judge_login_refresh.os,
        "chown",
        lambda path, uid, gid, follow_symlinks=True: owners.append((uid, gid)),
    )
    login_ = replace(
        operator(home, proxy="http://proxy.example:3128"), uid=1234, gid=5678
    )

    with pytest.raises(LoginRefreshError, match="cannot start the Codex CLI"):
        refresh_codex_login(login_, pinned_binary, effective_uid=0)

    [(argv, options)] = popen
    assert (options["user"], options["group"], options["extra_groups"]) == (
        1234,
        5678,
        [],
    )
    assert owners == [(1234, 5678)] * 3  # the scratch dir, its home and cwd
    # The verified bytes, by descriptor; no user config; an empty cwd.
    assert options["executable"] == f"/proc/self/fd/{options['pass_fds'][0]}"
    assert argv[0] == "codex" and argv[-1] == "app-server"
    assert 'cli_auth_credentials_store="file"' in argv
    assert "analytics.enabled=false" in argv
    environment = options["env"]
    assert environment["CODEX_HOME"] == environment["HOME"]
    assert Path(options["cwd"]).parent == Path(environment["CODEX_HOME"]).parent
    assert environment["https_proxy"] == "http://proxy.example:3128"
    assert set(environment) <= {
        "PATH",
        "HOME",
        "CODEX_HOME",
        "NO_COLOR",
        "https_proxy",
        "HTTPS_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    assert options["start_new_session"] is True
    # The scratch directory is gone afterwards.
    assert not Path(environment["CODEX_HOME"]).exists()


def test_codex_runs_as_the_invoking_user_when_not_root(tmp_path, popen, pinned_binary):
    home = codex_home(tmp_path, login("1", time.time() + 120))

    with pytest.raises(LoginRefreshError):
        refresh_codex_login(operator(home), pinned_binary, effective_uid=os.getuid())

    [(_, options)] = popen
    assert "user" not in options and "group" not in options


def test_a_binary_that_is_not_the_pinned_release_never_runs(
    tmp_path, monkeypatch, popen, pinned_binary
):
    home = codex_home(tmp_path, login("1", time.time() + 120))
    monkeypatch.setattr(judge_login_refresh, "host_codex_sha256", lambda: "0" * 64)

    with pytest.raises(LoginRefreshError, match="not the pinned release"):
        refresh_codex_login(operator(home), pinned_binary)
    with pytest.raises(LoginRefreshError, match="was not prepared"):
        refresh_codex_login(operator(home), None)
    monkeypatch.setattr(judge_login_refresh, "host_codex_sha256", lambda: None)
    with pytest.raises(LoginRefreshError, match="no release for this host"):
        refresh_codex_login(operator(home), pinned_binary)
    assert popen == []


def test_the_host_release_is_the_pinned_one_for_its_machine(monkeypatch):
    from rsi_loop.harness.agent import versions

    for machine, expected in (
        ("x86_64", versions.CODEX_SHA256_X86_64),
        ("aarch64", versions.CODEX_SHA256_AARCH64),
        ("arm64", versions.CODEX_SHA256_AARCH64),
        ("riscv64", None),
    ):
        monkeypatch.setattr(
            judge_login_refresh.platform, "machine", lambda m=machine: m
        )
        assert judge_login_refresh.host_codex_sha256() == expected


# -- a fake Codex CLI that refreshes the way Codex does -----------------------------

FAKE_CODEX = r'''#!{python} -I
"""A stand-in for codex app-server: account/read with refreshToken rotates the
tokens in CODEX_HOME/auth.json in place (truncate and write, as Codex's file
store does), then answers."""
import base64, json, os, sys, time

MODE = {mode!r}
RECORD = {record!r}


def jwt(claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return "h." + payload.decode() + ".s"


home = os.environ["CODEX_HOME"]
with open(RECORD, "w") as stream:
    json.dump(
        {{
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "cwd_entries": os.listdir("."),
            "home_entries": sorted(os.listdir(home)),
            "auth_link": os.readlink(os.path.join(home, "auth.json")),
            "uid": os.getuid(),
            "environment": sorted(os.environ),
        }},
        stream,
    )
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "initialize":
        answer = {{"id": message["id"], "result": {{"userAgent": "fake"}}}}
    elif message.get("method") == "account/read":
        if MODE == "hang":
            time.sleep(60)
        if MODE == "slow":  # the refresh request takes a while
            time.sleep(4)
        if MODE == "refuse" or not message["params"].get("refreshToken"):
            error = {{"code": -32603, "message": "refresh token reused"}}
            answer = {{"id": message["id"], "error": error}}
        else:
            path = os.path.join(home, "auth.json")
            with open(path) as stream:
                auth = json.load(stream)
            claims = {{"exp": int(time.time()) + 86400, "generation": "refreshed",
                      "https://api.openai.com/auth": {{"chatgpt_account_id": "acc-1"}}}}
            auth["tokens"].update(
                id_token=jwt(dict(claims, kind="id")),
                access_token=jwt(claims),
                refresh_token="REFRESH-rotated-synthetic-refresh-token",
            )
            auth["last_refresh"] = "2026-10-14T12:00:00Z"
            descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(auth, stream, indent=2)
            if MODE == "save-then-hang":  # its account request stalls
                time.sleep(60)
            result = {{"account": None, "requiresOpenaiAuth": True}}
            answer = {{"id": message["id"], "result": result}}
    else:
        continue
    sys.stdout.write(json.dumps({{"method": "configWarning", "params": {{}}}}) + "\n")
    sys.stdout.write(json.dumps(answer) + "\n")
    sys.stdout.flush()
'''


def fake_codex(tmp_path: Path, monkeypatch, mode: str = "ok") -> tuple[Path, Path]:
    """The fake as the prepared Codex CLI directory, pinned for this host."""
    directory = tmp_path / "judge-codex" / "codex-0.0.0-x86_64"
    directory.mkdir(parents=True)
    record = tmp_path / "fake-codex-record.json"
    script = FAKE_CODEX.format(python=sys.executable, mode=mode, record=str(record))
    (directory / "codex").write_text(script)
    (directory / "codex").chmod(0o755)
    digest = hashlib.sha256(script.encode()).hexdigest()
    monkeypatch.setattr(judge_login_refresh, "host_codex_sha256", lambda: digest)
    return directory, record


def test_the_fake_codex_refreshes_the_operators_file_in_place(tmp_path, monkeypatch):
    home = codex_home(tmp_path, login("1", time.time() + 120))
    inode = (home / "auth.json").stat().st_ino
    directory, record = fake_codex(tmp_path, monkeypatch)

    answer = refresh_codex_login(operator(home), directory / "codex", timeout=30)

    assert answer is None
    saved = json.loads((home / "auth.json").read_bytes())
    assert saved["tokens"]["refresh_token"] == "REFRESH-rotated-synthetic-refresh-token"
    stat = (home / "auth.json").stat()
    assert (stat.st_ino, stat.st_mode & 0o777) == (inode, 0o600)
    seen = json.loads(record.read_text())
    # Nothing of the operator's but auth.json; no config; an empty cwd.
    assert seen["auth_link"] == str(home / "auth.json")
    assert seen["home_entries"] == ["auth.json"]
    assert seen["cwd_entries"] == []
    assert seen["argv"][-1] == "app-server"
    assert seen["uid"] == os.getuid()
    assert "CODEX_REFRESH_TOKEN_URL_OVERRIDE" not in seen["environment"]


@pytest.mark.parametrize(
    ("mode", "match"),
    [("refuse", None), ("hang", "did not answer in time")],
)
def test_the_fake_codex_failing_leaves_the_file_alone(
    tmp_path, monkeypatch, mode, match
):
    monkeypatch.setattr(judge_login_refresh, "IN_FLIGHT_GRACE_SECONDS", 1)
    current = login("1", time.time() + 120)
    home = codex_home(tmp_path, current)
    directory, _record = fake_codex(tmp_path, monkeypatch, mode)
    started = time.monotonic()

    if match is None:
        assert (
            refresh_codex_login(operator(home), directory / "codex", timeout=30)
            == "refresh token reused"
        )
    else:
        with pytest.raises(LoginRefreshError, match=match):
            refresh_codex_login(operator(home), directory / "codex", timeout=2)
    assert time.monotonic() - started < 20
    assert (home / "auth.json").read_bytes() == current


@pytest.mark.parametrize("mode", ["save-then-hang", "slow"])
def test_a_refresh_codex_saves_after_its_answer_deadline_is_used(
    tmp_path, monkeypatch, mode
):
    """save-then-hang: saved, then no answer; slow: the refresh request
    outlasts the answer deadline. Either way Codex gets the in-flight grace
    to save it, is stopped once it did, and the Judge gets the new login."""
    monkeypatch.setattr(judge_login_refresh, "REFRESH_TIMEOUT_SECONDS", 2)
    home = codex_home(tmp_path, login("1", time.time() + 120))
    directory, _record = fake_codex(tmp_path, monkeypatch, mode)
    started = time.monotonic()

    renewal = renew_operator_login(
        operator(home), codex_cli=directory, round_seconds=ROUND
    )

    assert time.monotonic() - started < 20
    saved = json.loads((home / "auth.json").read_bytes())
    assert saved["tokens"]["refresh_token"] == "REFRESH-rotated-synthetic-refresh-token"
    assert renewal.content == (home / "auth.json").read_bytes()
    assert renewal.warning is None


def test_the_refresh_bound_counts_every_wait():
    assert judge_login_refresh.MAX_REFRESH_SECONDS == (
        judge_login_refresh.REFRESH_TIMEOUT_SECONDS
        + judge_login_refresh.IN_FLIGHT_GRACE_SECONDS
        + 5
        + 4
    )


# -- a Judge round ---------------------------------------------------------------------


def operator_access(tmp_path: Path, content: bytes, cli: Path | None):
    definition = make_run_plan(tmp_path / "plan").task
    definition = definition.model_copy(
        update={
            "verifier": definition.verifier.model_copy(
                update={"judge_model": "gpt-judge", "clean_image": True}
            )
        }
    )
    home = codex_home(tmp_path, content)
    llm = resolve_judge_llm(
        definition,
        JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
        RSILoopConfig(),
        None,
        pin=lambda _targets: (ENDPOINT,),
        environ={},
    )
    access = resolve_judge_model_access(definition, RSILoopConfig(), None, (), llm)
    return replace(access, codex_cli=cli), home


def judge_round(tmp_path: Path, access, events: list):
    _, runtime, snapshot, artifacts, request = make_harness(
        tmp_path, event_callback=lambda name, value: events.append((name, value))
    )
    plan, work, log_dir = request
    verifier = plan.task.verifier.model_copy(
        update={"judge_model": "gpt-judge", "judge_model_effort": "medium"}
    )
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"verifier": verifier})}
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(plan),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        event_callback=lambda name, value: events.append((name, value)),
        model_access=access,
    )
    return runner, runtime, (plan, work, log_dir)


def plan_timeout(request) -> float:
    return request[0].task.verifier.timeout_seconds


def launched_llm(runtime) -> dict:
    [launch, _owner] = runtime.injected_judge_agent[-1]
    return json.loads(launch)


def test_a_round_hands_the_judge_the_login_codex_refreshed(tmp_path, monkeypatch):
    directory, record = fake_codex(tmp_path, monkeypatch)
    access, home = operator_access(tmp_path, login("1", time.time() + 120), directory)
    events: list = []
    runner, runtime, request = judge_round(tmp_path, access, events)

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert record.exists()
    saved = (home / "auth.json").read_bytes()
    launch = launched_llm(runtime)
    handed = json.loads(launch["llm"]["auth_json"])
    assert handed["tokens"]["access_token"] == access_token(saved)
    assert handed["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN
    # The new tokens are redacted: on the host all of them; in the Judge the
    # ones it holds. No refresh token reaches a Judge, not even to redact.
    assert access_token(saved) in launch["redact"]
    assert {
        access_token(saved),
        "REFRESH-rotated-synthetic-refresh-token",
        "REFRESH-1-synthetic-refresh-token",
    } <= runner.model_access.secret_values
    launch_bytes = runtime.injected_judge_agent[-1][0]
    assert b"REFRESH-" not in launch_bytes
    assert WITHHELD_REFRESH_TOKEN not in launch["redact"]
    # The round's end, past which rsi-agent need not check the login.
    assert (
        time.time()
        < launch["round_ends_at"]
        <= (time.time() + plan_timeout(request) + 60)
    )
    assert not [name for name, _ in events if name == "judge_login_refresh_failed"]

    # The next round needs no refresh: the same copy, Codex not run again.
    record.unlink()
    report = evaluate(runner, request, round_id="agent-2")
    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert not record.exists()
    assert json.loads(launched_llm(runtime)["llm"]["auth_json"]) == handed


def test_a_round_whose_refresh_failed_keeps_the_copy_and_warns(tmp_path, monkeypatch):
    directory, _record = fake_codex(tmp_path, monkeypatch, "refuse")
    current = login("1", time.time() + 120)
    access, home = operator_access(tmp_path, current, directory)
    events: list = []
    runner, runtime, request = judge_round(tmp_path, access, events)

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert runner.model_access is access
    handed = json.loads(launched_llm(runtime)["llm"]["auth_json"])
    assert handed["tokens"]["access_token"] == access_token(current)
    [(_, warning)] = [
        event for event in events if event[0] == "judge_login_refresh_failed"
    ]
    assert warning == {
        "round_id": "agent-1",
        "reason": "Codex did not refresh it (Codex answered: refresh token reused)",
    }


def test_any_failure_on_the_operators_login_keeps_the_copy(tmp_path, monkeypatch):
    from rsi_harness.runtime import judge

    access, _home = operator_access(tmp_path, login("1", time.time() + 86400), tmp_path)
    events: list = []
    runner, _runtime, request = judge_round(tmp_path, access, events)

    def broken(*args, **options):
        raise RecursionError("deep")

    monkeypatch.setattr(judge, "renew_operator_login", broken)
    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert runner.model_access is access
    [(_, warning)] = [
        event for event in events if event[0] == "judge_login_refresh_failed"
    ]
    assert warning["reason"] == "the login could not be used (RecursionError)"


def test_the_judge_never_gets_a_refreshable_copy(tmp_path):
    access, _home = operator_access(tmp_path, login("1", time.time() + 86400), None)

    with pytest.raises(Exception, match="could not be withheld"):
        access.with_operator_login(b'{"tokens": {"refresh_token": 1}, "x": NaN}')


def test_no_run_start_expiry_check_remains():
    root = Path(judge_login_refresh.__file__).parents[3]
    for name in ("judge_llm.py", "judge_model_access.py", "judge.py"):
        text = (root / "src" / "rsi_harness" / "runtime" / name).read_text()
        assert "_MIN_LOGIN_SECONDS" not in text and "expires within" not in text
    for document in ("README.md", "docs/harbor-task-authoring/task-toml.md"):
        assert "35 minutes" not in (root / document).read_text()
