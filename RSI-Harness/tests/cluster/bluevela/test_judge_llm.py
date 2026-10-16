"""The Judge LLM (judge_model) on cluster runs.

The submit host checks the credential and finds the pinned Codex CLI; the
job resolves the credential again, copies and verifies the Codex CLI on its
node, refreshes an operator's login only in place, and gives each Judge round
rsi-agent llm with the network of cluster_judge_egress. Apptainer is not run
here: the tests check the argv and the per-round files the runtime prepares.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela import adapter as adapter_module
from rsi_harness.cluster.bluevela import judge_llm as cluster_judge_llm
from rsi_harness.cluster.bluevela import runtime as runtime_module
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.bluevela.judge_llm import (
    LOGIN_LOCK,
    ClusterJudgeLLM,
    OperatorLoginFile,
    cluster_judge_egress,
    cluster_login_refresh,
    filesystem_type,
    install_pinned_codex,
    judge_llm_for_egress,
    locate_pinned_codex,
    preflight_cluster_judge_llm,
    require_login_in_place,
)
from rsi_harness.cluster.bluevela.runtime import (
    NativeEngineComposition,
    NativeJudgeEvaluator,
    require_cluster_judge_model_access,
)
from rsi_harness.errors import SetupError
from rsi_harness.models import JudgeLLMAuth, JudgeLLMRequest
from rsi_harness.runtime.judge_llm import WITHHELD_REFRESH_TOKEN, resolve_judge_llm
from rsi_harness.runtime.judge_login_refresh import (
    LoginRefreshError,
    LoginRenewal,
    OperatorCodexLogin,
    renew_operator_login,
)
from rsi_harness.runtime.judge_model_access import resolve_judge_model_access
from rsi_loop.harness.config import RSILoopConfig
from tests.cluster.bluevela.test_adapter import (
    RecordingScheduler,
    _profile,
    _request,
)
from tests.cluster.bluevela.test_judge_model_access import (
    _binds,
    _envs,
    _judge,
    _plan,
    _runtime,
)
from tests.factories import write_harbor_task
from tests.runtime.test_judge_llm import (
    ACCESS_TOKEN,
    API_KEY,
    CHATGPT_LOGIN,
    REFRESH_TOKEN,
    codex_home,
    definition,
    login_material,
    pin,
)

PROXY = "http://127.0.0.1:3128"
KEY_ENV = {"RSI_JUDGE_LLM_API_KEY": API_KEY}


def _llm_plan(tmp_path: Path, **options):
    plan = _plan(tmp_path, **{"model_access": False, **options})
    verifier = plan.task.verifier.model_copy(
        # Cluster plans record the clean Judge they always use.
        update={
            "judge_model": "gpt-judge",
            "judge_model_effort": "high",
            "clean_image": True,
        }
    )
    return plan.model_copy(
        update={"task": plan.task.model_copy(update={"verifier": verifier})}
    )


def _task(*, verifier="no-network", agent="public", model_access=False, judge=True):
    task = definition(
        judge_model="gpt-judge" if judge else None, model_access=model_access
    )
    task.verifier.network.mode = verifier
    task.agent.network.mode = agent
    return task


def _api(task, *, egress, environ=KEY_ENV, **request):
    access = resolve_judge_llm(
        task,
        JudgeLLMRequest(auth=JudgeLLMAuth.API, **request),
        RSILoopConfig(),
        None,
        pin=pin,
        environ=environ,
    )
    return judge_llm_for_egress(access, egress)


def _codex(task, tmp_path, *, egress, **request):
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    access = resolve_judge_llm(
        task,
        JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home, **request),
        RSILoopConfig(),
        None,
        pin=pin,
        environ={},
    )
    return judge_llm_for_egress(access, egress)


def _fake_codex(directory: Path, content: bytes = b"#!/bin/sh\necho codex\n") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "codex"
    binary.write_bytes(content)
    binary.chmod(0o755)
    return binary


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _login_file(home: Path) -> OperatorLoginFile:
    found = (home / "auth.json").stat()
    return OperatorLoginFile(
        inode=found.st_ino,
        uid=found.st_uid,
        filesystem=filesystem_type(home / "auth.json"),
    )


# -- what a cluster serves ---------------------------------------------------------


@pytest.mark.parametrize("verifier", ["public", "no-network"])
@pytest.mark.parametrize("agent", ["public", "no-network", "allowlist"])
def test_a_judge_llm_runs_on_a_single_node_judge_whatever_the_agent(verifier, agent):
    task = _task(verifier=verifier, agent=agent)
    require_cluster_judge_model_access(task, multi_node=False)
    assert cluster_judge_egress(task) == ("site" if verifier == "public" else "pinned")


def test_a_cluster_refuses_only_what_it_cannot_serve():
    with pytest.raises(SetupError, match="A Judge LLM on a multi-node Judge"):
        require_cluster_judge_model_access(_task(), multi_node=True)
    with pytest.raises(SetupError, match="public or no-network Verifier"):
        require_cluster_judge_model_access(
            _task(verifier="allowlist"), multi_node=False
        )
    # With model access too, its own conditions still hold.
    with pytest.raises(SetupError, match="public or no-network Agent"):
        require_cluster_judge_model_access(
            _task(agent="allowlist", model_access=True), multi_node=False
        )
    with pytest.raises(SetupError, match="pinned endpoints needs"):
        require_cluster_judge_model_access(
            _task(agent="no-network", verifier="public", model_access=True),
            multi_node=False,
        )
    # And model access decides the network: the Work Agent's.
    both = _task(agent="public", verifier="no-network", model_access=True)
    require_cluster_judge_model_access(both, multi_node=False)
    assert cluster_judge_egress(both) == "site"
    require_cluster_judge_model_access(_task(judge=False), multi_node=True)


# -- the network ---------------------------------------------------------------------


def test_site_egress_keeps_the_judge_llm_as_on_docker(tmp_path):
    task = _task(verifier="public")
    for access in (
        _api(task, egress="site", proxy="http://proxy.example:3128"),
        _codex(task, tmp_path, egress="site"),
    ):
        assert judge_llm_for_egress(access, "site") is access
    assert _api(task, egress="site", proxy="http://proxy.example:3128").spec[
        "proxy"
    ] == ("http://proxy.example:3128")


def test_pinned_egress_goes_through_the_judges_own_provider_proxy(tmp_path):
    task = _task()
    api = _api(task, egress="pinned")
    assert api.spec["proxy"] == PROXY
    assert api.spec["url"] == "https://api.openai.com/v1/responses"
    assert [(item.hostname, item.port) for item in api.endpoints] == [
        ("api.openai.com", 443)
    ]
    codex = _codex(task, tmp_path, egress="pinned")
    assert codex.spec["environment"]["HTTPS_PROXY"] == PROXY
    assert codex.spec["environment"]["NO_PROXY"] == "127.0.0.1,localhost"
    assert [(item.hostname, item.port) for item in codex.endpoints] == [
        ("chatgpt.com", 443)
    ]


def test_pinned_egress_refuses_an_upstream_proxy_and_plain_http(tmp_path):
    task = _task()
    with pytest.raises(SetupError, match="upstream proxy"):
        _api(task, egress="pinned", proxy="http://proxy.example:3128")
    with pytest.raises(SetupError, match="upstream proxy"):
        _codex(task, tmp_path, egress="pinned", proxy="http://proxy.example:3128")
    with pytest.raises(SetupError, match="https API base URL"):
        _api(task, egress="pinned", base_url="http://models.example")


# -- the submit host ---------------------------------------------------------------


def test_preflight_checks_an_api_key_and_keeps_only_where_it_comes_from():
    prepared, access = preflight_cluster_judge_llm(
        _task(),
        JudgeLLMRequest(auth=JudgeLLMAuth.API),
        agent_auth=None,
        config=RSILoopConfig(),
        environ=KEY_ENV,
    )
    assert access.kind is JudgeLLMAuth.API and not access.reused
    assert prepared == ClusterJudgeLLM(request=JudgeLLMRequest(auth=JudgeLLMAuth.API))
    assert API_KEY not in prepared.model_dump_json()
    with pytest.raises(SetupError, match="needs an API key"):
        preflight_cluster_judge_llm(
            _task(),
            JudgeLLMRequest(auth=JudgeLLMAuth.API),
            agent_auth=None,
            config=RSILoopConfig(),
            environ={},
        )


def test_preflight_ignores_a_task_without_a_judge_model():
    assert (
        preflight_cluster_judge_llm(
            _task(judge=False),
            JudgeLLMRequest(auth=JudgeLLMAuth.API),
            agent_auth=None,
            config=RSILoopConfig(),
            environ={},
        )
        is None
    )


def test_preflight_fixes_an_operator_login_and_finds_the_pinned_codex(tmp_path):
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    binary = _fake_codex(tmp_path / "bin")
    prepared, access = preflight_cluster_judge_llm(
        _task(verifier="public"),
        JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN),
        agent_auth=None,
        config=RSILoopConfig(),
        environ={"CODEX_HOME": str(home)},
        which=lambda name: str(binary) if name == "codex" else None,
        digests={_digest(binary): "x86_64"},
    )
    assert access.needs_codex_cli and access.operator_login is not None
    # The job reads exactly this file, wherever its CODEX_HOME points.
    assert prepared.request.codex_home == home
    assert prepared.login == _login_file(home)
    assert prepared.codex == {"x86_64": binary}
    dumped = prepared.model_dump_json()
    assert ACCESS_TOKEN not in dumped and REFRESH_TOKEN not in dumped


def test_preflight_refuses_what_the_job_could_not_use(tmp_path):
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    other = _fake_codex(tmp_path / "bin")
    options = dict(agent_auth=None, config=RSILoopConfig())
    with pytest.raises(SetupError, match="pinned Codex CLI"):
        preflight_cluster_judge_llm(
            _task(verifier="public"),
            JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
            environ={},
            which=lambda _name: str(other),
            digests={"0" * 64: "x86_64"},
            **options,
        )
    with pytest.raises(SetupError, match="proxy environment"):
        preflight_cluster_judge_llm(
            _task(verifier="public"),
            JudgeLLMRequest(auth=JudgeLLMAuth.API, proxy="http://u:p@proxy:3128"),
            environ=KEY_ENV,
            **options,
        )
    # The base URL is kept in the run directory too.
    with pytest.raises(SetupError, match="--judge-llm-api-key-env"):
        preflight_cluster_judge_llm(
            _task(verifier="public"),
            JudgeLLMRequest(
                auth=JudgeLLMAuth.API, base_url="https://u:secret@llm.example"
            ),
            environ=KEY_ENV,
            **options,
        )
    with pytest.raises(SetupError, match="upstream proxy"):
        preflight_cluster_judge_llm(
            _task(verifier="no-network"),
            JudgeLLMRequest(auth=JudgeLLMAuth.API, proxy="http://proxy:3128"),
            environ=KEY_ENV,
            **options,
        )
    # The operator's login may not be the Work Agent's own.
    with pytest.raises(SetupError, match="Work Agent's own Codex login"):
        preflight_cluster_judge_llm(
            _task(verifier="public"),
            JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
            environ={},
            which=lambda _name: str(other),
            digests={_digest(other): "x86_64"},
            agent_auth=login_material(CHATGPT_LOGIN),
            config=RSILoopConfig(),
        )


def test_a_work_agents_lent_login_needs_the_pinned_codex_too(tmp_path):
    binary = _fake_codex(tmp_path / "bin")
    prepared, access = preflight_cluster_judge_llm(
        _task(verifier="public"),
        None,
        agent_auth=login_material(CHATGPT_LOGIN),
        config=RSILoopConfig(),
        environ={},
        which=lambda _name: str(binary),
        digests={_digest(binary): "aarch64"},
    )
    assert access.reused and access.needs_codex_cli
    assert prepared == ClusterJudgeLLM(codex={"aarch64": binary})


def test_the_pinned_codex_is_found_beside_an_npm_install(tmp_path):
    prefix = tmp_path / "prefix"
    package = prefix / "lib/node_modules/@openai/codex"
    (package / "bin").mkdir(parents=True)
    (package / "bin/codex.js").write_text("// launcher\n")
    vendor = (
        package
        / "node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin"
    )
    native = _fake_codex(vendor, b"\x7fELF pinned codex")
    (prefix / "bin").mkdir()
    (prefix / "bin/codex").symlink_to(package / "bin/codex.js")

    found = locate_pinned_codex(
        lambda _name: str(prefix / "bin/codex"), {_digest(native): "x86_64"}
    )

    assert found == {"x86_64": native}
    # A shared prefix with both packages: each architecture's, for the job to
    # pick its own (codex-linux-arm64 sorts first).
    arm_package = package / "node_modules/@openai/codex-linux-arm64"
    arm = _fake_codex(
        arm_package / "vendor/aarch64-unknown-linux-musl/bin",
        b"\x7fELF pinned codex arm",
    )
    both = locate_pinned_codex(
        lambda _name: str(prefix / "bin/codex"),
        {_digest(native): "x86_64", _digest(arm): "aarch64"},
    )
    assert both == {"x86_64": native, "aarch64": arm}
    with pytest.raises(SetupError, match="npm install -g @openai/codex@"):
        locate_pinned_codex(lambda _name: str(prefix / "bin/codex"), {"0" * 64: "x"})
    with pytest.raises(SetupError, match="pinned Codex CLI"):
        locate_pinned_codex(lambda _name: None, {_digest(native): "x86_64"})


# -- the compute node ----------------------------------------------------------------


def test_the_node_copies_the_codex_cli_only_when_it_is_its_release(tmp_path):
    source = _fake_codex(tmp_path / "shared")
    directory = install_pinned_codex(
        source, tmp_path / "judge-codex", expected=_digest(source)
    )
    copy = directory / "codex"
    assert copy.read_bytes() == source.read_bytes()
    assert copy.stat().st_mode & 0o777 == 0o755 and not copy.is_symlink()

    with pytest.raises(SetupError, match="not the pinned release"):
        install_pinned_codex(source, tmp_path / "other", expected="0" * 64)
    assert not (tmp_path / "other" / "codex").exists()
    with pytest.raises(SetupError, match="not readable on this compute node"):
        install_pinned_codex(
            tmp_path / "missing", tmp_path / "third", expected=_digest(source)
        )


def test_an_operator_login_is_used_only_in_place(tmp_path):
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    request = JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home)
    recorded = _login_file(home)
    require_login_in_place(request, recorded)
    # The same inode number on another filesystem, or another owner's: a
    # node-local file, never refreshed.
    for other in (
        recorded.model_copy(update={"filesystem": "not-" + recorded.filesystem}),
        recorded.model_copy(update={"uid": recorded.uid + 1}),
    ):
        with pytest.raises(SetupError, match="has another file there"):
            require_login_in_place(request, other)
    # Another file at the same path (a new codex login, or a copy).
    other = home / "auth.json.other"
    shutil.copy2(home / "auth.json", other)
    os.replace(other, home / "auth.json")
    with pytest.raises(SetupError, match="cannot be refreshed in place") as error:
        require_login_in_place(request, recorded)
    assert "a new codex login since submission" in str(error.value)
    assert "--judge-llm-auth api" in str(error.value)
    (home / "auth.json").unlink()
    with pytest.raises(SetupError, match="it is not there"):
        require_login_in_place(request, recorded)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes any file")
def test_an_operator_login_the_node_cannot_save_is_refused(tmp_path):
    """The same file, read-only here (a home read-only on compute nodes): a
    refresh would spend the refresh token and then fail to save the new one."""
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    request = JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home)
    recorded = _login_file(home)
    (home / "auth.json").chmod(0o400)
    with pytest.raises(SetupError, match="it is not writable here"):
        require_login_in_place(request, recorded)
    (home / "auth.json").chmod(0o600)
    home.chmod(0o500)  # no room for the refresh lock
    try:
        with pytest.raises(SetupError, match="is not writable here"):
            require_login_in_place(request, recorded)
    finally:
        home.chmod(0o700)


def test_filesystem_type_reads_the_mount_above_a_path(tmp_path):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "22 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
        "40 22 0:40 / /home rw - autofs auto.home rw\n"
        "41 40 0:41 /alice /home rw - nfs4 server:/home rw\n"
        "42 22 0:42 / /scratch\\040space rw - lustre fs rw\n"
    )
    assert filesystem_type(Path("/home/alice/.codex/auth.json"), mountinfo) == "nfs4"
    assert filesystem_type(Path("/etc/passwd"), mountinfo) == "ext4"
    assert filesystem_type(Path("/scratch space/x"), mountinfo) == "lustre"
    assert filesystem_type(Path("/x"), tmp_path / "missing") == "unknown"


# -- refreshing an operator's login on a node ---------------------------------------


def _operator(home: Path) -> OperatorCodexLogin:
    return OperatorCodexLogin(
        codex_home=home,
        uid=os.getuid(),
        gid=os.getgid(),
        read=lambda: (home / "auth.json").read_bytes(),
    )


def test_a_cluster_refresh_runs_codex_under_the_login_lock(tmp_path):
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    calls = []

    def run(login, binary, *, timeout, saved):
        # The lock is held: another process cannot take it now.
        held = subprocess.run(
            [sys.executable, "-c", _TRY_LOCK, str(home / LOGIN_LOCK)], check=False
        )
        calls.append((login.codex_home, binary, timeout, held.returncode))
        return None

    answer = cluster_login_refresh(
        _operator(home), tmp_path / "codex", timeout=60, saved=lambda: False, run=run
    )
    assert answer is None
    assert calls == [(home, tmp_path / "codex", 60, 1)]
    assert (home / LOGIN_LOCK).stat().st_mode & 0o777 == 0o600

    # Another job refreshed it while this one waited: no second refresh.
    calls.clear()
    cluster_login_refresh(
        _operator(home), tmp_path / "codex", timeout=60, saved=lambda: True, run=run
    )
    assert calls == []


_TRY_LOCK = """\
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
"""

_HOLD_LOCK = """\
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.lockf(fd, fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(30)
"""


def test_a_cluster_refresh_waits_for_another_jobs_refresh(tmp_path):
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(home / LOGIN_LOCK)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline() == "locked\n"
        ran = []
        started = time.monotonic()
        with pytest.raises(LoginRefreshError, match="another run is still"):
            cluster_login_refresh(
                _operator(home),
                None,
                timeout=60,
                saved=lambda: False,
                run=lambda *args, **options: ran.append(args),
                lock_timeout=1,
            )
        assert ran == [] and time.monotonic() - started >= 1
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes any file")
def test_a_node_that_cannot_save_the_login_never_refreshes_it(tmp_path):
    """Per round too: the round keeps its copy and says why; Codex never runs."""
    from tests.runtime.test_judge_login import codex_login

    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_bytes(codex_login(5, expires_in=60))
    (home / "auth.json").chmod(0o400)
    ran = []

    def refresh(login, binary, *, timeout, saved):
        return cluster_login_refresh(
            login,
            binary,
            timeout=timeout,
            saved=saved,
            run=lambda *args, **options: ran.append(args),
        )

    renewal = renew_operator_login(
        _operator(home), codex_cli=None, round_seconds=600, refresh=refresh
    )
    assert renewal.content == codex_login(5, expires_in=60)
    assert "not writable here, so this node does not refresh it" in renewal.warning
    assert ran == []
    assert not (home / LOGIN_LOCK).exists()


def _composition(tmp_path, monkeypatch, plan, prepared, *, environ=KEY_ENV):
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    runtime = _runtime(tmp_path, monkeypatch, plan)
    composition = object.__new__(NativeEngineComposition)
    composition.plan = plan
    composition.runtime = runtime
    composition.payload = SimpleNamespace(
        profile=runtime.profile, run_id="native-run", judge_llm=prepared
    )
    composition.rsi_loop_config = RSILoopConfig()
    composition.provider_urls = ()
    return composition


def test_the_job_resolves_an_api_key_judge_llm_again(tmp_path, monkeypatch, capsys):
    plan = _llm_plan(tmp_path, verifier_network="no-network")
    prepared = ClusterJudgeLLM(
        request=JudgeLLMRequest(
            auth=JudgeLLMAuth.API, base_url="https://198.51.100.8:8443"
        )
    )
    composition = _composition(tmp_path, monkeypatch, plan, prepared)

    access = composition.judge_model_access()

    assert access.egress == "pinned" and access.codex_cli is None
    assert [(item.hostname, item.port) for item in access.endpoints] == [
        ("198.51.100.8", 8443)
    ]
    llm = json.loads(access.launch)["llm"]
    assert (llm["api_key"], llm["proxy"]) == (API_KEY, PROXY)
    assert json.loads(access.launch)["unsupported"]  # no rsi-agent exec
    assert API_KEY in access.secret_values
    out = capsys.readouterr().out
    assert "judge_llm:" in out and "'egress': 'pinned'" in out
    assert API_KEY not in out


def test_the_job_prepares_the_codex_cli_and_checks_the_login_in_place(
    tmp_path, monkeypatch
):
    plan = _llm_plan(tmp_path, verifier_network="public")
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    source = _fake_codex(tmp_path / "shared")
    monkeypatch.setattr(cluster_judge_llm, "host_codex_sha256", lambda: _digest(source))
    prepared = ClusterJudgeLLM(
        request=JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
        codex={"x86_64": source, "aarch64": tmp_path / "unused"},
        login=_login_file(home),
    )
    composition = _composition(tmp_path, monkeypatch, plan, prepared, environ={})
    monkeypatch.setattr(cluster_judge_llm.platform, "machine", lambda: "x86_64")

    access = composition.judge_model_access()

    assert access.egress == "site"
    assert access.codex_cli == composition.runtime.node_tmp / "judge-codex"
    assert (access.codex_cli / "codex").read_bytes() == source.read_bytes()
    assert access.operator_login is not None
    assert access.operator_login.codex_home == home
    llm = json.loads(access.launch)["llm"]
    assert llm["codex"] == "/opt/rsi-agent-codex/codex"
    login = json.loads(llm["auth_json"])
    assert login["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN

    # A different file at that path on this node: refused before Work starts.
    login = prepared.login.model_copy(update={"inode": prepared.login.inode + 1})
    composition.payload.judge_llm = prepared.model_copy(update={"login": login})
    shutil.rmtree(access.codex_cli)
    with pytest.raises(SetupError, match="cannot be refreshed in place"):
        composition.judge_model_access()


def test_a_codex_cli_for_another_architecture_is_refused_on_the_node(
    tmp_path, monkeypatch
):
    plan = _llm_plan(tmp_path, verifier_network="public")
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    source = _fake_codex(tmp_path / "shared")
    monkeypatch.setattr(cluster_judge_llm, "host_codex_sha256", lambda: "0" * 64)
    monkeypatch.setattr(cluster_judge_llm.platform, "machine", lambda: "x86_64")
    prepared = ClusterJudgeLLM(
        request=JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
        codex={"x86_64": source},
        login=_login_file(home),
    )
    composition = _composition(tmp_path, monkeypatch, plan, prepared, environ={})
    with pytest.raises(SetupError, match="not the pinned release for this compute"):
        composition.judge_model_access()
    # The submit host found only another architecture's release.
    composition.payload.judge_llm = prepared.model_copy(
        update={"codex": {"aarch64": source}}
    )
    with pytest.raises(SetupError, match=r"for aarch64, not for this compute node"):
        composition.judge_model_access()


def test_the_agent_and_the_judge_llm_on_one_host_get_one_pinned_endpoint(
    tmp_path, monkeypatch
):
    plan = _llm_plan(
        tmp_path,
        model_access=True,
        agent_network="no-network",
        verifier_network="no-network",
    )
    import ipaddress

    from rsi_harness.runtime.network import PinnedEndpoint

    prepared = ClusterJudgeLLM(
        request=JudgeLLMRequest(auth=JudgeLLMAuth.API, base_url="https://LLM.example")
    )
    composition = _composition(tmp_path, monkeypatch, plan, prepared)
    composition.rsi_loop_config = RSILoopConfig(agent_api_key="SYNTHETIC-AGENT-KEY")
    composition.provider_urls = ("https://llm.example/v1",)
    # Each resolution of the shared host answers with another address.
    answers = iter(("198.51.100.8", "198.51.100.9"))

    def pin_endpoints(_enforcer, urls):
        return tuple(
            PinnedEndpoint(
                hostname=cluster_judge_llm.urlsplit(url).hostname,
                port=443,
                addresses=(ipaddress.ip_address(next(answers)),),
            )
            for url in urls
        )

    monkeypatch.setattr(
        runtime_module.NetworkPolicyEnforcer, "pin_endpoints", pin_endpoints
    )

    access = composition.judge_model_access()

    assert access.egress == "pinned"
    assert [(item.hostname, item.port) for item in access.endpoints] == [
        ("llm.example", 443)
    ]


# -- a Judge round -------------------------------------------------------------------


def _round_access(plan, judge_llm, *, egress, codex_cli=None):
    access = resolve_judge_model_access(
        plan.task, RSILoopConfig(), None, (), judge_llm, egress=egress
    )
    return replace(access, codex_cli=codex_cli)


def test_an_api_key_judge_round_gets_rsi_agent_llm_and_never_the_key(
    tmp_path, monkeypatch
):
    plan = _llm_plan(tmp_path, verifier_network="public")
    access = _round_access(plan, _api(plan.task, egress="site"), egress="site")

    runtime, capture, _ = _judge(tmp_path, monkeypatch, plan, access)

    binds = _binds(capture.argv)
    agent = runtime.control / "judge-agent"
    assert f"{agent}:/opt/rsi-agent:ro" in binds
    assert f"{agent / 'rsi-agent'}:/usr/local/bin/rsi-agent:ro" in binds
    # Neither the Work Agent's CLI (no model access) nor a Codex CLI.
    assert not any(":/usr/local/bin/codex" in bind for bind in binds)
    assert not any("/opt/rsi-agent-codex" in bind for bind in binds)
    assert capture.options["network_mode"] == "public"
    launch = capture.launch
    assert launch["llm"]["api_key"] == API_KEY
    assert launch["llm"]["model"] == "gpt-judge"
    assert launch["round_ends_at"] > 0
    assert capture.launch_mode == 0o600
    environment = _envs(capture.argv)
    assert environment["RSI_JUDGE_MODEL"] == "gpt-judge"
    assert environment["RSI_JUDGE_MODEL_EFFORT"] == "high"
    assert environment["RSI_JUDGE_ISOLATION_FALLBACK"] == "none"
    assert API_KEY not in " ".join(capture.argv)
    assert API_KEY in capture.options["output_redact_values"]
    assert list((runtime.node_tmp / "judge").iterdir()) == []


def test_a_codex_login_judge_round_is_pinned_with_the_codex_cli_read_only(
    tmp_path, monkeypatch
):
    plan = _llm_plan(tmp_path, verifier_network="no-network")
    codex_cli = tmp_path / "node-codex"
    _fake_codex(codex_cli)
    judge_llm = _codex(plan.task, tmp_path, egress="pinned")
    access = _round_access(plan, judge_llm, egress="pinned", codex_cli=codex_cli)

    _runtime_, capture, _ = _judge(tmp_path, monkeypatch, plan, access)

    binds = _binds(capture.argv)
    assert f"{codex_cli}:/opt/rsi-agent-codex:ro" in binds
    assert capture.options["network_mode"] == "no-network"
    assert capture.command[2] == "/usr/local/bin/rsi-netns-relay"
    assert capture.command[-2:] == ("/bin/bash", "/tests/test.sh")
    llm = capture.launch["llm"]
    assert llm["environment"]["HTTPS_PROXY"] == PROXY
    assert json.loads(llm["auth_json"])["tokens"]["refresh_token"] == (
        WITHHELD_REFRESH_TOKEN
    )
    assert REFRESH_TOKEN not in json.dumps(capture.launch)
    assert ACCESS_TOKEN not in " ".join(capture.argv)
    assert {ACCESS_TOKEN, REFRESH_TOKEN} <= set(capture.options["output_redact_values"])


def test_a_codex_login_needs_the_prepared_codex_cli(tmp_path, monkeypatch):
    plan = _llm_plan(tmp_path, verifier_network="public")
    access = _round_access(
        plan, _codex(plan.task, tmp_path, egress="site"), egress="site"
    )
    runtime = _runtime(tmp_path, monkeypatch, plan)
    with pytest.raises(SetupError, match="Codex CLI, which was not prepared"):
        runtime.configure_judge_model_access(access)
    with pytest.raises(SetupError, match="differs"):
        runtime.configure_judge_model_access(None)


def _later_login() -> bytes:
    later = json.loads(json.dumps(CHATGPT_LOGIN))
    later["tokens"]["access_token"] = "SYNTHETIC-ACCESS-LATER-77aa"
    later["tokens"]["refresh_token"] = "SYNTHETIC-REFRESH-LATER-1b2c"
    return json.dumps(later).encode()


def test_each_round_gets_the_operators_refreshed_login(tmp_path, monkeypatch, capsys):
    plan = _llm_plan(tmp_path, verifier_network="public")
    codex_cli = tmp_path / "node-codex"
    _fake_codex(codex_cli)
    access = _round_access(
        plan,
        _codex(plan.task, tmp_path, egress="site"),
        egress="site",
        codex_cli=codex_cli,
    )
    runtime = _runtime(tmp_path, monkeypatch, plan)
    runtime.configure_judge_model_access(access)
    calls = []

    def renew(login, *, codex_cli, round_seconds, refresh):
        calls.append((login, codex_cli, round_seconds, refresh))
        return LoginRenewal(_later_login())

    monkeypatch.setattr(runtime_module, "renew_operator_login", renew)

    runtime.renew_operator_login("agent-1")

    # Locked and only where it can be saved (cluster_login_refresh).
    assert calls == [
        (
            access.operator_login,
            codex_cli,
            float(plan.task.verifier.timeout_seconds),
            cluster_login_refresh,
        )
    ]
    renewed = runtime.judge_model_access
    login = json.loads(json.loads(renewed.launch)["llm"]["auth_json"])
    assert login["tokens"]["access_token"] == "SYNTHETIC-ACCESS-LATER-77aa"
    assert login["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN
    assert "SYNTHETIC-REFRESH-LATER-1b2c" in renewed.secret_values
    assert "SYNTHETIC-ACCESS-LATER-77aa" in runtime.auth_secret_values
    assert "judge_login_refresh_failed" not in capsys.readouterr().out

    # A failed refresh keeps the copy and says why, redacted.
    monkeypatch.setattr(
        runtime_module,
        "renew_operator_login",
        lambda *_args, **_options: LoginRenewal(
            None, f"Codex could not refresh it: {ACCESS_TOKEN}"
        ),
    )
    runtime.renew_operator_login("agent-2")
    assert runtime.judge_model_access is renewed
    out = capsys.readouterr().out
    assert "judge_login_refresh_failed: agent-2" in out
    assert ACCESS_TOKEN not in out

    # Never fatal: whatever it raises, the round keeps its copy.
    def broken(*_args, **_options):
        raise OSError(f"stale file handle {ACCESS_TOKEN}")

    monkeypatch.setattr(runtime_module, "renew_operator_login", broken)
    runtime.renew_operator_login("agent-3")
    assert runtime.judge_model_access is renewed
    out = capsys.readouterr().out
    assert "judge_login_refresh_failed: agent-3: the login could not be used " in out
    assert "(OSError)" in out and ACCESS_TOKEN not in out


def test_a_lent_work_login_is_renewed_from_work_each_round(tmp_path, monkeypatch):
    """A Codex Work Agent's ChatGPT login: each round's Judge LLM gets the
    login Work's Agent keeps refreshing, read while Work is stopped."""
    from tests.runtime.test_judge_login import codex_login, material, refresh_token

    plan = _llm_plan(tmp_path, verifier_network="public")
    start = material()
    judge_llm = resolve_judge_llm(
        plan.task, None, RSILoopConfig(), start, pin=pin, environ={}
    )
    assert judge_llm.reused and judge_llm.needs_codex_cli
    codex_cli = tmp_path / "node-codex"
    _fake_codex(codex_cli)
    access = replace(
        resolve_judge_model_access(
            plan.task, RSILoopConfig(), start, (), judge_llm, egress="site"
        ),
        codex_cli=codex_cli,
    )
    assert access.llm_reuses_login and access.operator_login is None
    store = tmp_path / "work-login"
    store.mkdir(mode=0o700)
    (store / "auth.json").write_bytes(codex_login(2))
    (store / "auth.json").chmod(0o600)
    make_runtime = _runtime

    def with_store(*args, **options):
        runtime = make_runtime(*args, **options)
        runtime._agent_auth_roots[PurePosixPath("/home/agent/.codex")] = store
        return runtime

    monkeypatch.setattr(
        "tests.cluster.bluevela.test_judge_model_access._runtime", with_store
    )

    _runtime_, capture, _ = _judge(tmp_path, monkeypatch, plan, access)

    login = json.loads(capture.launch["llm"]["auth_json"])
    renewed = json.loads(codex_login(2))
    assert login["tokens"]["access_token"] == renewed["tokens"]["access_token"]
    assert login["tokens"]["refresh_token"] == WITHHELD_REFRESH_TOKEN
    assert refresh_token(2) not in json.dumps(capture.launch)
    assert refresh_token(2) in capture.options["output_redact_values"]
    # No exec: the Work Agent's own login files never reach this Judge.
    assert capture.auth_files == {}


def test_the_refresh_happens_before_work_pauses(tmp_path):
    events = []

    class Stop(Exception):
        pass

    def pause(_work):
        events.append("pause")
        raise Stop

    apptainer = SimpleNamespace(
        rounds=tmp_path,
        pause=pause,
        renew_operator_login=lambda round_id: events.append(f"renew {round_id}"),
    )
    observer = SimpleNamespace(resource_event=lambda *_args, **_kwargs: None)
    request = SimpleNamespace(
        round_id="agent-1",
        work_container=SimpleNamespace(container_id="work"),
    )
    with pytest.raises(Stop):
        NativeJudgeEvaluator(apptainer, None).evaluate(request, observer)
    assert events == ["renew agent-1", "pause"]


# -- the adapter ---------------------------------------------------------------------


_JUDGE_LLM_TOML = """\
schema_version = "1.4"

[task]
name = "rsi/minimal-gpu"

[environment]
os = "linux"
workdir = "/workspace"
gpus = 2
network_mode = "no-network"

[agent]
timeout_sec = 3600
network_mode = "public"

[verifier]
timeout_sec = 3600
network_mode = "no-network"

[metadata.rsi_harness.verifier]
judge_model = "gpt-judge"
"""


def _llm_request(tmp_path, *, dry_run, judge_llm):
    task_dir = write_harbor_task(tmp_path, task_toml=_JUDGE_LLM_TOML).resolve()
    return _request(tmp_path, dry_run=dry_run).model_copy(
        update={"task_dir": task_dir, "judge_llm": judge_llm}
    )


@pytest.fixture
def _fake_submit_host(monkeypatch, tmp_path):
    """No Work login, and a pinned Codex CLI where `codex` resolves."""
    monkeypatch.setattr(adapter_module, "load_config", RSILoopConfig)
    monkeypatch.setattr(adapter_module, "resolve_agent_auth", lambda **_kwargs: None)
    binary = _fake_codex(tmp_path / "submit-bin")
    monkeypatch.setattr(
        cluster_judge_llm.shutil,
        "which",
        lambda name: str(binary) if name == "codex" else None,
    )
    monkeypatch.setattr(
        cluster_judge_llm, "pinned_codex_digests", lambda: {_digest(binary): "x86_64"}
    )
    return binary


def test_the_adapter_refuses_a_missing_credential_before_submission(
    tmp_path, monkeypatch, _fake_submit_host
):
    monkeypatch.delenv("RSI_JUDGE_LLM_API_KEY", raising=False)
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(_profile(tmp_path), scheduler=scheduler)
    with pytest.raises(SetupError, match="needs an API key"):
        adapter.run(
            _llm_request(
                tmp_path, dry_run=True, judge_llm=JudgeLLMRequest(auth=JudgeLLMAuth.API)
            )
        )
    assert scheduler.events == [] and scheduler.specs == []


def test_the_adapter_hands_the_job_where_the_credential_is_never_the_key(
    tmp_path, monkeypatch, _fake_submit_host
):
    monkeypatch.setenv("RSI_JUDGE_LLM_API_KEY", API_KEY)
    events = []
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(
        _profile(tmp_path),
        scheduler=scheduler,
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.162.1",
    )

    result = adapter.run(
        _llm_request(
            tmp_path, dry_run=False, judge_llm=JudgeLLMRequest(auth=JudgeLLMAuth.API)
        )
    )

    assert result.status.value == "completed"
    payload_path = scheduler.specs[-1].script_path.with_suffix(".json")
    payload = load_engine_payload(payload_path)
    assert payload.judge_llm == ClusterJudgeLLM(
        request=JudgeLLMRequest(auth=JudgeLLMAuth.API)
    )
    assert API_KEY not in payload_path.read_text()
    run_dir = payload.run_plan.paths.root
    assert not any(
        API_KEY in path.read_text(errors="replace")
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    [(_, judge)] = [item for item in events if item[0] == "judge_llm"]
    assert judge["model"] == "gpt-judge" and judge["credential"] == "api"


def test_the_adapter_ignores_judge_llm_options_for_a_task_without_one(
    tmp_path, _fake_submit_host
):
    events = []
    adapter = BlueVelaClusterAdapter(
        _profile(tmp_path),
        scheduler=RecordingScheduler(),
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.162.1",
    )
    request = _request(tmp_path, dry_run=True).model_copy(
        update={"judge_llm": JudgeLLMRequest(auth=JudgeLLMAuth.API)}
    )
    adapter.run(request)
    assert ("judge_llm_ignored", None) in events


def test_a_multinode_judge_llm_is_refused_before_submission(
    tmp_path, monkeypatch, _fake_submit_host
):
    monkeypatch.setenv("RSI_JUDGE_LLM_API_KEY", API_KEY)
    toml = _JUDGE_LLM_TOML.replace("gpus = 2", "gpus = 16").replace(
        'judge_model = "gpt-judge"', 'judge_model = "gpt-judge"\ngpus = 8'
    )
    task_dir = write_harbor_task(tmp_path, task_toml=toml).resolve()
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(_profile(tmp_path), scheduler=scheduler)
    request = _request(tmp_path, dry_run=True).model_copy(
        update={
            "task_dir": task_dir,
            "judge_llm": JudgeLLMRequest(auth=JudgeLLMAuth.API),
        }
    )
    with pytest.raises(SetupError, match="multi-node Judge"):
        adapter.run(request)
    assert scheduler.events == []


@pytest.mark.parametrize("cls", ["bluevela", "slurm"])
def test_the_adapter_hands_the_job_an_operator_login_by_place_never_its_tokens(
    tmp_path, monkeypatch, _fake_submit_host, cls
):
    from rsi_harness.cluster.config import load_cluster_profile
    from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
    from tests.cluster.slurm.test_adapter import RecordingSlurmScheduler

    home = codex_home(tmp_path, CHATGPT_LOGIN)
    monkeypatch.setenv("CODEX_HOME", str(home))
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    adapter_class = BlueVelaClusterAdapter
    if cls == "slurm":
        profile = load_cluster_profile("slurm", {"USER": "alice"}).model_copy(
            update={
                "storage": profile.storage,
                "apptainer": profile.apptainer,
                "builder": profile.builder,
            }
        )
        scheduler = RecordingSlurmScheduler()
        adapter_class = SlurmClusterAdapter
    events = []
    adapter = adapter_class(
        profile,
        scheduler=scheduler,
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.162.1",
    )
    toml = _JUDGE_LLM_TOML.replace(
        'network_mode = "no-network"\n\n[metadata',
        ('network_mode = "public"\n\n[metadata'),
    )
    task_dir = write_harbor_task(tmp_path, task_toml=toml).resolve()
    request = _request(tmp_path, dry_run=False).model_copy(
        update={
            "task_dir": task_dir,
            "judge_llm": JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN),
        }
    )

    adapter.run(request)

    payload_path = scheduler.specs[-1].script_path.with_suffix(".json")
    payload = load_engine_payload(payload_path)
    assert payload.judge_llm == ClusterJudgeLLM(
        request=JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
        codex={"x86_64": _fake_submit_host},
        login=_login_file(home),
    )
    text = payload_path.read_text()
    assert ACCESS_TOKEN not in text and REFRESH_TOKEN not in text
    [(_, judge)] = [item for item in events if item[0] == "judge_llm"]
    assert judge["credential"] == "codex-login" and not judge["reused"]
