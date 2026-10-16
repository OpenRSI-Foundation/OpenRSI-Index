"""Each Judge round gets the login Work's Agent keeps refreshing, never usable
for a refresh of its own."""

from __future__ import annotations

import base64
import functools
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    ContainerRef,
    ContainerTmpfs,
    JudgeLLMAuth,
    JudgeLLMRequest,
    SubmissionStatus,
)
from rsi_harness.runtime.judge import JudgeRunner
from rsi_harness.runtime.judge_llm import resolve_judge_llm
from rsi_harness.runtime.judge_model_access import resolve_judge_model_access
from rsi_harness.runtime.local_auth import (
    WITHHELD_REFRESH_TOKEN,
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
    current_login,
    judge_login_copy,
    login_expiry,
    login_secret_values,
    read_login_store,
)
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan
from tests.fakes import FakeDockerClient
from tests.runtime.test_docker import make_runtime
from tests.runtime.test_judge import evaluate, make_harness, managed_workdir_volume
from tests.runtime.test_judge_model_access import ENDPOINT, declare_model_access

OPTIONS = "rw,nosuid,nodev,noexec,mode=0700"
CODEX_LOGIN = PurePosixPath("/home/agent/.codex/auth.json")
CLAUDE_LOGIN = PurePosixPath("/home/agent/.claude/.credentials.json")


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"h.{payload.decode()}.s"


def refresh_token(generation: int) -> str:
    return f"REFRESH-{generation}-synthetic-refresh-token"


def _raw_jwt(claims_json: str) -> str:
    payload = base64.urlsafe_b64encode(claims_json.encode()).rstrip(b"=")
    return f"h.{payload.decode()}.s"


def codex_login(
    generation: int,
    *,
    account: str = "acc-1",
    expires_in: float = 86400,
    exp: str | None = None,
) -> bytes:
    """A synthetic ChatGPT login; each generation has its own tokens.

    ``exp`` is the access token's raw ``exp`` JSON text, for hostile values.
    The same login however it is asked for (``codex_login(2)`` and
    ``codex_login(2, account="acc-1")`` alike), however often it is read.
    """
    return _codex_login(generation, account, expires_in, exp)


@functools.cache  # one generation, one set of tokens, however often it is read
def _codex_login(
    generation: int, account: str, expires_in: float, exp: str | None
) -> bytes:
    claims = {
        "exp": int(time.time() + expires_in),
        "generation": generation,
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }
    access = _jwt(claims)
    if exp is not None:
        access = _raw_jwt(
            f'{{"exp": {exp}, "generation": {generation}, '
            f'"https://api.openai.com/auth": {{"chatgpt_account_id": "{account}"}}}}'
        )
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": _jwt({**claims, "kind": "id"}),
                "access_token": access,
                "refresh_token": refresh_token(generation),
                "account_id": account,
            },
            "last_refresh": "2026-10-14T00:00:00Z",
        }
    ).encode()


def _with_tokens(content: bytes, **tokens: object) -> bytes:
    login = json.loads(content)
    login["tokens"].update(tokens)
    return json.dumps(login).encode()


@functools.cache
def claude_login(generation: int) -> bytes:
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": f"sk-ant-oat-ACCESS-{generation}-synthetic-token",
                "refreshToken": f"sk-ant-ort-REFRESH-{generation}-synthetic-token",
                "expiresAt": int((time.time() + 3600) * 1000),
                "scopes": ["user:inference"],
                "subscriptionType": "max",
            }
        }
    ).encode()


def material(agent: str = "codex", content: bytes | None = None) -> AgentAuthMaterial:
    path = CODEX_LOGIN if agent == "codex" else CLAUDE_LOGIN
    content = content or (codex_login(1) if agent == "codex" else claude_login(1))
    return AgentAuthMaterial(
        agent_name=agent,
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(target=path.parent, options=OPTIONS),
                files=(AgentAuthFile(path=PurePosixPath(path.name), content=content),),
            ),
        ),
        secret_values=login_secret_values(json.loads(content)),
    )


def _content(login: AgentAuthMaterial) -> dict:
    [entry] = [entry for mount in login.mounts for entry in mount.files]
    return json.loads(entry.content)


# -- reading and checking Work's current login --------------------------------------


def test_the_current_login_replaces_the_run_start_one():
    start = material()

    current = current_login(start, {CODEX_LOGIN: codex_login(2)})

    assert _content(current)["tokens"]["refresh_token"] == refresh_token(2)
    assert refresh_token(2) in current.secret_values
    assert refresh_token(1) not in current.secret_values
    assert current.agent_name == "codex"
    assert current.mounts[0].tmpfs == start.mounts[0].tmpfs


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        ({}, "missing"),
        ({CODEX_LOGIN: b"{not json"}, "not a JSON object"),
        ({CODEX_LOGIN: b"[1, 2]"}, "not a JSON object"),
        ({CODEX_LOGIN: b"x" * (1024 * 1024 + 1)}, "too large"),
        ({CODEX_LOGIN: b'{"auth_mode": "chatgpt"}'}, "no usable login"),
        ({CODEX_LOGIN: codex_login(2, account="acc-other")}, "another login"),
        (
            {CODEX_LOGIN: json.dumps({"OPENAI_API_KEY": "sk-other"}).encode()},
            "another login",
        ),
    ],
)
def test_a_missing_malformed_or_foreign_login_is_refused(files, reason):
    with pytest.raises(SetupError, match=reason) as error:
        current_login(material(), files)
    assert "REFRESH" not in str(error.value)


def test_a_codex_login_naming_two_accounts_is_refused():
    login = json.loads(codex_login(2))
    login["tokens"]["account_id"] = "acc-other"

    with pytest.raises(SetupError, match="no usable login"):
        current_login(material(), {CODEX_LOGIN: json.dumps(login).encode()})


def test_a_claude_login_is_checked_for_its_kind():
    start = material("claude-code")

    current = current_login(start, {CLAUDE_LOGIN: claude_login(2)})
    assert (
        _content(current)["claudeAiOauth"]["accessToken"]
        == "sk-ant-oat-ACCESS-2-synthetic-token"
    )
    with pytest.raises(SetupError, match="no usable login"):
        current_login(start, {CLAUDE_LOGIN: b'{"other": {}}'})


def test_the_login_store_on_a_cluster_is_read_without_symlinks(tmp_path):
    store = tmp_path / "store"
    store.mkdir(mode=0o700)
    (store / "auth.json").write_bytes(codex_login(2))
    (store / "auth.json").chmod(0o600)
    roots = {CODEX_LOGIN.parent: store}

    assert read_login_store(material(), roots) == {CODEX_LOGIN: codex_login(2)}
    (store / "auth.json").unlink()
    (tmp_path / "elsewhere.json").write_bytes(codex_login(3))
    os.symlink(tmp_path / "elsewhere.json", store / "auth.json")
    with pytest.raises(SetupError, match="symlink"):
        read_login_store(material(), roots)


@pytest.mark.parametrize("kind", ["fifo", "socket", "directory"])
def test_a_login_store_entry_that_is_no_file_is_refused_at_once(kind):
    """The Work Agent writes its login store: a FIFO must not block the read."""
    store = Path(tempfile.mkdtemp(prefix="ls-"))  # short: a socket path
    try:
        path = store / "auth.json"
        if kind == "fifo":
            os.mkfifo(path, 0o600)
        elif kind == "socket":
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(path))
            listener.close()
            path.chmod(0o600)
        else:
            path.mkdir(mode=0o700)
        outcome: list[BaseException] = []

        def read() -> None:
            try:
                read_login_store(material(), {CODEX_LOGIN.parent: store})
            except BaseException as error:
                outcome.append(error)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(10)
        assert not reader.is_alive(), "the login store read blocked"
        [error] = outcome
        assert isinstance(error, SetupError)
    finally:
        shutil.rmtree(store)


# -- what the Work Agent writes never breaks a round or the redaction --------------


@pytest.mark.parametrize(
    "exp", ["1" + "0" * 400, "1e400", "-5", '"soon"', "true", "99999999999999"]
)
def test_an_unusable_codex_expiry_is_refused(exp):
    with pytest.raises(SetupError, match="unusable expiry"):
        current_login(material(), {CODEX_LOGIN: codex_login(2, exp=exp)})


@pytest.mark.parametrize("exp", ["Infinity", "-Infinity", "NaN"])
def test_a_non_json_expiry_leaves_the_token_without_one(tmp_path, exp):
    """No JSON number: the claims are unreadable, so no expiry is known."""
    current = current_login(material(), {CODEX_LOGIN: codex_login(2, exp=exp)})

    assert login_expiry(current) is None
    renewed = _access(tmp_path).with_work_login(current)
    assert json.loads(renewed.launch)["credentials_expire_at"] is None


@pytest.mark.parametrize("expires_at", ["1" + "0" * 400, "1e400", "-1", '"soon"'])
def test_an_unusable_claude_expiry_is_refused(expires_at):
    content = claude_login(2).replace(
        b'"expiresAt": ', b'"expiresAt": ' + expires_at.encode() + b', "was": '
    )

    with pytest.raises(SetupError, match="unusable expiry"):
        current_login(material("claude-code"), {CLAUDE_LOGIN: content})


@pytest.mark.parametrize(
    "content",
    [
        b'{"auth_mode": "chatgpt", "x": NaN}',
        b'{"a": ' * 40 + b"1" + b"}" * 40,
        b"[" * 100_000 + b"]" * 100_000,
        b'{"a": ' * 100_000 + b"1" + b"}" * 100_000,
    ],
    ids=["nan", "nested-40", "nested-list", "nested-object"],
)
def test_non_standard_or_deep_json_is_malformed(content):
    assert len(content) < 1024 * 1024
    with pytest.raises(SetupError, match="not a JSON object"):
        current_login(material(), {CODEX_LOGIN: content})


def test_only_a_logins_credentials_join_the_redaction_set():
    login = json.loads(codex_login(2))
    login["secret"] = "e"
    login["tokens"]["api_key"] = "FAIL"
    login["extra"] = {"password": "score never exceeded baseline"}

    current = current_login(material(), {CODEX_LOGIN: json.dumps(login).encode()})

    tokens = login["tokens"]
    assert current.secret_values == {
        tokens["access_token"],
        tokens["id_token"],
        tokens["refresh_token"],
    }


@pytest.mark.parametrize(
    "refresh",
    [
        "e",
        "FAIL",
        "too-short-to-be-a-token",
        "score never exceeded the baseline",
        "x" * (16 * 1024 + 1),
    ],
)
def test_a_new_credential_must_look_like_a_token(refresh):
    with pytest.raises(SetupError, match="malformed credential") as error:
        current_login(
            material(),
            {CODEX_LOGIN: _with_tokens(codex_login(2), refresh_token=refresh)},
        )
    assert str(error.value) == (
        f"the Work Agent's login file {CODEX_LOGIN} holds a malformed credential"
    )


def test_a_claude_credential_must_look_like_a_token():
    content = claude_login(2).replace(b"sk-ant-ort-REFRESH-2-synthetic-token", b"e")

    with pytest.raises(SetupError, match="malformed credential"):
        current_login(material("claude-code"), {CLAUDE_LOGIN: content})


def test_an_unchanged_credential_is_accepted_whatever_its_shape():
    start = material(content=_with_tokens(codex_login(1), refresh_token="rt-1"))

    current = current_login(
        start, {CODEX_LOGIN: _with_tokens(codex_login(2), refresh_token="rt-1")}
    )

    assert "rt-1" in current.secret_values


def test_the_redaction_set_keeps_the_run_start_values_and_the_last_logins(tmp_path):
    from rsi_harness.runtime.judge_model_access import KEPT_LOGIN_GENERATIONS

    access = _access(tmp_path)
    start = access.secret_values
    last = 10
    for generation in range(2, last + 1):
        access = access.with_work_login(
            current_login(access.work_login, {CODEX_LOGIN: codex_login(generation)})
        )

    kept = range(last - KEPT_LOGIN_GENERATIONS + 1, last + 1)
    spec = json.loads(access.launch)
    # The Judge redacts all but the refresh tokens, which it never holds.
    assert set(spec["redact"]) == access.secret_values - access.withheld_values
    assert access.withheld_values == {
        refresh_token(generation) for generation in (1, *kept)
    }
    assert start <= access.secret_values and refresh_token(1) in start
    assert {refresh_token(generation) for generation in kept} <= access.secret_values
    assert (
        not {refresh_token(generation) for generation in range(2, kept.start)}
        & access.secret_values
    )
    assert len(access.secret_values) == len(start) + 3 * KEPT_LOGIN_GENERATIONS


# -- the Judge's copy never refreshes ----------------------------------------------


@pytest.mark.parametrize("agent", ["codex", "claude-code"])
def test_the_judge_copy_has_no_usable_refresh_token(agent):
    start = material(agent)

    copy = judge_login_copy(start)

    content = _content(copy)
    original = _content(start)
    if agent == "codex":
        assert content["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN
        assert content["tokens"]["access_token"] == original["tokens"]["access_token"]
    else:
        assert content["claudeAiOauth"]["refreshToken"] == WITHHELD_REFRESH_TOKEN
        assert (
            content["claudeAiOauth"]["accessToken"]
            == original["claudeAiOauth"]["accessToken"]
        )
    assert copy.secret_values == start.secret_values


def test_model_access_hands_exec_a_copy_without_the_refresh_token(tmp_path):
    definition = declare_model_access(make_run_plan(tmp_path)).task
    start = material(content=codex_login(1, expires_in=7200))

    access = resolve_judge_model_access(definition, RSILoopConfig(), start, (ENDPOINT,))

    assert access.work_login is start
    assert _content(access.auth)["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN
    spec = json.loads(access.launch)
    assert spec["credentials_expire_at"] == login_expiry(start)
    # Redacted on the host; never in the Judge, not even to redact.
    assert refresh_token(1) in access.secret_values
    assert refresh_token(1).encode() not in access.launch
    assert refresh_token(1) not in json.dumps(_content(access.auth))


def test_a_renewed_login_reaches_exec_llm_and_the_redaction_set(tmp_path):
    definition = declare_model_access(make_run_plan(tmp_path)).task
    definition = definition.model_copy(
        update={
            "verifier": definition.verifier.model_copy(
                update={"judge_model": "gpt-judge"}
            )
        }
    )
    start = material()
    llm = resolve_judge_llm(
        definition,
        None,
        RSILoopConfig(),
        start,
        pin=lambda _targets: (ENDPOINT,),
        environ={},
    )
    access = resolve_judge_model_access(
        definition, RSILoopConfig(), start, (ENDPOINT,), llm
    )
    assert access.llm_reuses_login and access.work_login is start

    renewed = access.with_work_login(
        current_login(start, {CODEX_LOGIN: codex_login(2)})
    )

    spec = json.loads(renewed.launch)
    access_token = json.loads(codex_login(2))["tokens"]["access_token"]
    for login in (_content(renewed.auth), json.loads(spec["llm"]["auth_json"])):
        assert login["tokens"]["access_token"] == access_token
        assert login["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN
    # Old and new values alike stay redacted; refresh tokens on the host only.
    assert access_token in spec["redact"]
    assert b"REFRESH-" not in renewed.launch
    assert {refresh_token(1), refresh_token(2), access_token} <= renewed.secret_values
    assert renewed.work_login.secret_values == login_secret_values(
        json.loads(codex_login(2))
    )


def test_no_login_has_a_run_start_expiry_check(tmp_path):
    definition = make_run_plan(tmp_path).task
    definition = definition.model_copy(
        update={
            "verifier": definition.verifier.model_copy(
                update={"judge_model": "gpt-judge", "clean_image": True}
            )
        }
    )
    soon = codex_login(1, expires_in=60)

    lent = resolve_judge_llm(
        definition,
        None,
        RSILoopConfig(),
        material(content=soon),
        pin=lambda _targets: (ENDPOINT,),
        environ={},
    )
    assert lent.reused and lent.expires_at is None and lent.operator_login is None

    home = tmp_path / "codex-home"
    home.mkdir(mode=0o700)
    (home / "auth.json").write_bytes(soon)
    (home / "auth.json").chmod(0o600)
    # The operator's login is refreshed before each round instead.
    operator = resolve_judge_llm(
        definition,
        JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
        RSILoopConfig(),
        None,
        pin=lambda _targets: (ENDPOINT,),
        environ={},
    )
    assert operator.expires_at is not None and operator.operator_login is not None


# -- a Judge round -----------------------------------------------------------------


def _runner(tmp_path: Path, access, events):
    _, runtime, snapshot, artifacts, request = make_harness(
        tmp_path, event_callback=lambda name, value: events.append((name, value))
    )
    plan, work, log_dir = request
    plan = declare_model_access(plan)
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


def _access(tmp_path: Path):
    definition = declare_model_access(make_run_plan(tmp_path / "plan")).task
    return resolve_judge_model_access(
        definition, RSILoopConfig(), material(), (ENDPOINT,)
    )


def _injected_token(runtime, index: int = -1) -> str:
    login, _owner = runtime.injected_auth[index]
    return _content(login)["tokens"]["access_token"]


def test_each_round_reads_works_current_login_before_pausing_it(tmp_path):
    events: list = []
    runner, runtime, request = _runner(tmp_path, _access(tmp_path), events)
    second = json.loads(codex_login(2))["tokens"]["access_token"]
    runtime.work_login_files = {CODEX_LOGIN: codex_login(2)}

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED, report.error
    names = [name for name, _ in runtime.events]
    assert names.index("auth_read") < names.index("pause")
    assert _injected_token(runtime) == second
    login, _ = runtime.injected_auth[-1]
    assert _content(login)["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN
    [launch, _] = runtime.injected_judge_agent[-1]
    assert second in json.loads(launch)["redact"]
    # This round's output, feedback and events never show the new values.
    assert {second, refresh_token(2)} <= set(
        runtime.exec_options[-1]["output_redact_values"]
    )
    assert runner.model_access.work_login.secret_values >= {refresh_token(2)}

    # The next round is checked against this one, and moves on again.
    runtime.work_login_files = {CODEX_LOGIN: codex_login(3)}
    report = evaluate(runner, request, round_id="agent-2")
    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert (
        _injected_token(runtime) == json.loads(codex_login(3))["tokens"]["access_token"]
    )
    assert not [name for name, _ in events if name == "judge_login_kept"]


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        ({}, "missing"),
        ({CODEX_LOGIN: b"{truncated"}, "not a JSON object"),
        ({CODEX_LOGIN: codex_login(2, account="acc-other")}, "another login"),
        ({CODEX_LOGIN: b"[" * 100_000 + b"]" * 100_000}, "not a JSON object"),
        ({CODEX_LOGIN: codex_login(3, exp="1" + "0" * 400)}, "unusable expiry"),
        (
            {CODEX_LOGIN: _with_tokens(codex_login(3), refresh_token="e")},
            "malformed credential",
        ),
    ],
)
def test_an_unusable_login_keeps_the_last_good_copy_with_a_warning(
    tmp_path, files, reason
):
    events: list = []
    access = _access(tmp_path)
    runner, runtime, request = _runner(tmp_path, access, events)
    runtime.work_login_files = {CODEX_LOGIN: codex_login(2)}
    evaluate(runner, request)
    good = runner.model_access
    runtime.work_login_files = files

    report = evaluate(runner, request, round_id="agent-2")

    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert runner.model_access is good
    assert (
        _injected_token(runtime) == json.loads(codex_login(2))["tokens"]["access_token"]
    )
    [(_, warning)] = [value for value in events if value[0] == "judge_login_kept"]
    assert warning["round_id"] == "agent-2" and reason in warning["reason"]


@pytest.mark.parametrize(
    "error", [RecursionError("deep"), OverflowError("big"), ValueError("nan")]
)
def test_any_failure_on_the_login_keeps_the_last_good_copy(
    tmp_path, monkeypatch, error
):
    from rsi_harness.runtime import judge

    events: list = []
    access = _access(tmp_path)
    runner, _runtime, request = _runner(tmp_path, access, events)

    def broken(start, files):
        raise error

    monkeypatch.setattr(judge, "current_login", broken)
    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert runner.model_access is access
    [(_, warning)] = [value for value in events if value[0] == "judge_login_kept"]
    assert warning["reason"] == (
        f"the Work Agent's login could not be used ({type(error).__name__})"
    )


def test_a_failed_read_keeps_the_last_good_copy(tmp_path, monkeypatch):
    events: list = []
    access = _access(tmp_path)
    runner, runtime, request = _runner(tmp_path, access, events)

    def broken(container, material):
        raise InfrastructureError("failed to read the Work Agent's login file")

    monkeypatch.setattr(runtime, "read_agent_auth", broken)
    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert runner.model_access is access
    assert [name for name, _ in events].count("judge_login_kept") == 1


def test_an_api_key_judge_reads_nothing_from_work(tmp_path):
    events: list = []
    definition = declare_model_access(make_run_plan(tmp_path / "plan")).task
    access = resolve_judge_model_access(
        definition, RSILoopConfig(agent_api_key="sk-x"), None, (ENDPOINT,)
    )
    runner, runtime, request = _runner(tmp_path, access, events)

    evaluate(runner, request)

    assert access.work_login is None
    assert "auth_read" not in [name for name, _ in runtime.events]


# -- reading from a Docker Work container -------------------------------------------


class _Socket:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def settimeout(self, value: float) -> None:
        assert value > 0

    def recv(self, size: int) -> bytes:
        chunk, self.data = self.data[:size], self.data[size:]
        return chunk


class _Stream:
    def __init__(self, data: bytes) -> None:
        self._sock = _Socket(data)
        self._response = self
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _frame(stream: int, data: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + len(data).to_bytes(4, "big") + data


def _docker_work(tmp_path, monkeypatch, output: bytes, exit_code: int = 0):
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path)
    calls = []
    streams = []

    def exec_create(container, command, **options):
        calls.append((command, options))
        return {"Id": "read-1"}

    def exec_start(exec_id, **options):
        assert options == {"detach": False, "tty": False, "socket": True}
        streams.append(_Stream(output))
        return streams[-1]

    monkeypatch.setattr(client.api, "exec_create", exec_create)
    monkeypatch.setattr(client.api, "exec_start", exec_start)
    monkeypatch.setattr(
        client.api,
        "exec_inspect",
        lambda exec_id: {"Running": False, "ExitCode": exit_code},
    )
    return runtime, calls, streams


def test_docker_reads_works_login_from_stdout_frames(tmp_path, monkeypatch):
    content = codex_login(2)
    runtime, calls, streams = _docker_work(
        tmp_path,
        monkeypatch,
        _frame(1, content[:10]) + _frame(2, b"noise") + _frame(1, content[10:]),
    )

    files = runtime.read_agent_auth(
        ContainerRef(container_id="work-1", role="work"), material()
    )

    assert files == {CODEX_LOGIN: content}
    [(command, options)] = calls
    assert command[-1] == str(CODEX_LOGIN)
    assert '[ ! -L "$1" ]' in command[2] and "/bin/cat" in command[2]
    assert options["user"] == "root" and options["stderr"] is False
    assert streams[0].closed


@pytest.mark.parametrize(
    ("output", "exit_code"),
    [
        (_frame(1, b"{}"), 1),
        (_frame(1, b"x" * (1024 * 1024 + 1)), 0),
    ],
)
def test_docker_leaves_out_a_file_it_may_not_read(
    tmp_path, monkeypatch, output, exit_code
):
    runtime, _, _ = _docker_work(tmp_path, monkeypatch, output, exit_code)

    files = runtime.read_agent_auth(
        ContainerRef(container_id="work-1", role="work"), material()
    )

    assert files == {}


def test_docker_reads_only_works_login(tmp_path):
    client = FakeDockerClient()
    judge = make_runtime(client, tmp_path, role="judge")

    with pytest.raises(SetupError, match="only Work"):
        judge.read_agent_auth(
            ContainerRef(container_id="judge-1", role="judge"), material()
        )


def test_docker_truncated_frames_fail_the_read(tmp_path, monkeypatch):
    runtime, _, _ = _docker_work(tmp_path, monkeypatch, _frame(1, b"abc")[:-1])

    with pytest.raises(InfrastructureError, match="login file"):
        runtime.read_agent_auth(
            ContainerRef(container_id="work-1", role="work"), material()
        )
