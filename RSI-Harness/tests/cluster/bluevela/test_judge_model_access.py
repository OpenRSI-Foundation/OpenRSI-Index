"""Cluster Judges: model access through rsi-agent with the network the Work
Agent already uses for its model, and the operator's isolation fallback.
Apptainer is not run here: the tests check the argv and the per-round files
the runtime prepares."""

from __future__ import annotations

import json
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela.runtime import (
    ApptainerAgentRuntime,
    NativeEngineComposition,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentRunResult,
    ContainerRef,
    EvaluationRequest,
    NetworkPolicy,
)
from rsi_harness.runtime.judge_model_access import resolve_judge_model_access
from rsi_harness.runtime.network import NetworkPolicyEnforcer
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.config import RSILoopConfig
from tests.cluster.bluevela.test_runtime import _multi_pools
from tests.factories import make_run_plan
from tests.runtime.test_judge_model_access import (
    LOGIN_SECRET,
    auth_material,
    declare_model_access,
)

API_KEY = "sk-synthetic-cluster-judge-key-5e1d"
request_cleanup: list[Path] = []


@pytest.fixture(autouse=True)
def _remove_node_tmp():
    yield
    while request_cleanup:
        shutil.rmtree(request_cleanup.pop(), ignore_errors=True)


def _plan(
    tmp_path: Path,
    *,
    agent_network: str = "public",
    verifier_network: str = "no-network",
    model_access: bool = True,
    visibility: str = "hidden",
):
    plan = declare_model_access(make_run_plan(tmp_path), model_access)
    task = plan.task
    task = task.model_copy(
        update={
            "agent": task.agent.model_copy(
                update={"network": NetworkPolicy(mode=agent_network)}
            ),
            "verifier": task.verifier.model_copy(
                update={
                    "network": NetworkPolicy(mode=verifier_network),
                    "visibility": visibility,
                }
            ),
        }
    )
    tests = tmp_path / "task" / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test.sh").write_text("#!/bin/bash\n")
    task = task.model_copy(update={"tests_digest": hash_tree(tests)})
    return plan.model_copy(update={"task": task})


def _runtime(tmp_path: Path, monkeypatch, plan, *, profile=None, pools=None):
    # Short, as on a node: the Judge's provider socket lives below it.
    node_tmp = Path(tempfile.mkdtemp(prefix="nt-"))
    request_cleanup.append(node_tmp)
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    binary = tmp_path / "codex-binary"
    binary.write_text("#!/bin/sh\n")
    companion = tmp_path / "codex-helper"
    companion.write_text("#!/bin/sh\n")
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=profile or load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
            agent_binary=binary,
            agent_launcher="codex",
            agent_companions=(companion,),
            agent_auth=None,
            agent_version="0.149.0",
        ),
        plan,
        allocated_pools=pools,
    )
    bin_dir = PurePosixPath("/usr/local/bin")
    runtime._copies.update(
        {
            bin_dir / "codex": binary,
            bin_dir / "codex-helper": companion,
            bin_dir / "rsi-netns-relay": Path(sys.executable),
            bin_dir / "rsi-submit": tmp_path / "rsi-submit",
        }
    )
    runtime._agent_cli_targets = [bin_dir / "codex", bin_dir / "codex-helper"]
    return runtime


def _access(plan, *, egress: str, endpoints=(), auth=None, fallback="none"):
    from rsi_harness.cluster.bluevela.network import judge_proxy_environment

    return resolve_judge_model_access(
        plan.task,
        RSILoopConfig(agent_api_key=API_KEY),
        auth,
        endpoints,
        egress=egress,
        isolation_fallback=fallback,
        extra_agent_environment=judge_proxy_environment()
        if egress == "pinned"
        else None,
    )


def _request(tmp_path: Path, plan) -> EvaluationRequest:
    return EvaluationRequest(
        run_plan=plan,
        work_container=ContainerRef(container_id="work", role="work"),
        round_id="agent-1",
        verifier_logs=tmp_path / "verifier",
        verifier_output=tmp_path / "feedback.log",
    )


def _binds(argv: tuple[str, ...]) -> list[str]:
    return [argv[i + 1] for i, item in enumerate(argv) if item == "--bind"]


class _Capture:
    """Stands in for _run: builds the real Apptainer argv and inspects the
    per-round files while they exist."""

    def __init__(self, runtime: ApptainerAgentRuntime) -> None:
        self.runtime = runtime
        self.argv: tuple[str, ...] = ()
        self.environment: dict[str, str] = {}
        self.command: tuple[str, ...] = ()
        self.options: dict = {}
        self.launch: dict | None = None
        self.launch_mode = None
        self.auth_files: dict[str, bytes] = {}

    def __call__(self, command, **options):
        self.command = tuple(command)
        self.options = options
        self.argv = self.runtime._base_command(
            devices=options["devices"],
            extra_binds=options.get("extra_binds", ()),
            mount_workspace=options.get("mount_workspace", True),
            mount_agent_home=options.get("mount_agent_home", True),
            network_mode=options["network_mode"],
            phase=options.get("phase", "work"),
            declared_network_mode=options.get("declared_network_mode"),
            image_tmp=options.get("image_tmp", False),
        )
        self.environment = self.runtime._container_environment(
            devices=options["devices"],
            environment=options.get("environment"),
            mount_agent_home=options.get("mount_agent_home", True),
            declared=options.get("declared_network_mode") or options["network_mode"],
            phase=options.get("phase", "work"),
        )
        for source, target, _ro in options.get("extra_binds", ()):
            if target == PurePosixPath("/run/rsi-agent"):
                launch = Path(source) / "private" / "launch.json"
                self.launch = json.loads(launch.read_bytes())
                self.launch_mode = stat.S_IMODE(launch.stat().st_mode)
                self.private_mode = stat.S_IMODE(launch.parent.stat().st_mode)
                self.state_mode = stat.S_IMODE(Path(source).stat().st_mode)
            if str(target).startswith("/home/agent/"):
                for path in Path(source).rglob("*"):
                    if path.is_file():
                        self.auth_files[str(target / path.relative_to(source))] = (
                            path.read_bytes()
                        )
        return AgentRunResult(exit_code=0, output="passed\n")


def _judge(tmp_path, monkeypatch, plan, access, **runtime_options):
    runtime = _runtime(tmp_path, monkeypatch, plan, **runtime_options)
    runtime.configure_judge_model_access(access)
    capture = _Capture(runtime)
    monkeypatch.setattr(runtime, "_run", capture)
    workspace = runtime.rounds / "agent-1" / "workspace"
    workspace.mkdir(parents=True)
    trajectory = tmp_path / "judge-trajectory" / "agent-1"
    trajectory.mkdir(parents=True)
    runtime.task_assets.mkdir(parents=True, exist_ok=True)
    result = runtime.run_judge(
        workspace,
        _request(tmp_path, plan),
        {"TASK_VALUE": "1"},
        trajectory=trajectory,
    )
    assert result.exit_code == 0
    return runtime, capture, trajectory


def test_a_model_access_judge_gets_rsi_agent_and_never_the_credentials(
    tmp_path, monkeypatch
):
    plan = _plan(tmp_path, agent_network="public")
    access = _access(plan, egress="site", auth=auth_material())

    runtime, capture, trajectory = _judge(tmp_path, monkeypatch, plan, access)

    binds = _binds(capture.argv)
    agent = runtime.control / "judge-agent"
    for bind in (
        f"{agent}:/opt/rsi-agent:ro",
        f"{agent / 'rsi-agent'}:/usr/local/bin/rsi-agent:ro",
        f"{tmp_path / 'codex-binary'}:/usr/local/bin/codex:ro",
        f"{tmp_path / 'codex-helper'}:/usr/local/bin/codex-helper:ro",
        f"{trajectory}:/logs/agent:ro",
    ):
        assert bind in binds, bind
    # rsi-agent's state is writable; the login copies are read-only.
    state = [bind for bind in binds if bind.endswith(":/run/rsi-agent")]
    assert len(state) == 1 and Path(state[0].split(":")[0]).name == "state"
    assert any(bind.endswith(":/home/agent/.codex:ro") for bind in binds)
    assert (capture.launch_mode, capture.private_mode, capture.state_mode) == (
        0o600,
        0o700,
        0o711,
    )
    assert capture.launch["version"] == 2
    assert capture.launch["credentials"] == [
        {"source": "/home/agent/.codex/auth.json", "target": ".codex/auth.json"}
    ]
    assert "/home/agent/.codex" in capture.launch["hidden"]
    assert capture.auth_files == {
        "/home/agent/.codex/auth.json": auth_material().mounts[0].files[0].content
    }
    # The Judge never sees Work's home, its /tmp or the submit client.
    assert not any(
        ":/home/agent:" in bind or bind.endswith(":/home/agent") for bind in binds
    )
    assert not any(str(runtime.agent_home) in bind for bind in binds)
    assert not any(str(runtime.work_tmp) in bind for bind in binds)
    assert not any("rsi-submit" in bind for bind in binds)
    # The procedure sees the Agent's name, never a credential.
    environment = capture.environment
    assert environment["RSI_AGENT_NAME"] == "codex"
    assert environment["TASK_VALUE"] == "1"
    # Always explicit, so an ENV in the task's image never decides.
    assert environment["RSI_JUDGE_ISOLATION_FALLBACK"] == "none"
    assert API_KEY not in " ".join(capture.argv)
    assert LOGIN_SECRET not in " ".join(capture.argv)
    assert set(capture.options["output_redact_values"]) >= {API_KEY, LOGIN_SECRET}
    # Every per-round file is gone after the round.
    assert list((runtime.node_tmp / "judge").iterdir()) == []


def test_site_egress_uses_the_work_agents_network(tmp_path, monkeypatch, capsys):
    plan = _plan(tmp_path, agent_network="public", verifier_network="no-network")
    access = _access(plan, egress="site")

    _runtime_, capture, _ = _judge(tmp_path, monkeypatch, plan, access)

    assert capture.options["network_mode"] == "public"
    assert "--net" not in capture.argv
    assert capture.command == ("/bin/bash", "/tests/test.sh")
    assert "HTTP_PROXY" not in capture.launch["environment"]
    assert "judge_model_egress: site" in capsys.readouterr().out


@pytest.mark.parametrize("mount_policy", ["legacy", "scoped"])
def test_site_egress_changes_only_the_network(tmp_path, monkeypatch, mount_policy):
    """A no-network Judge on the site network keeps its own binds, caches
    and HF environment; it never gets Work's public binds or shared caches."""
    plan = _plan(tmp_path, agent_network="public", verifier_network="no-network")
    profile = load_cluster_profile("bluevela")
    extra = tmp_path / "site-data"
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={
                    "mount_policy": mount_policy,
                    # A scoped profile forbids extra_binds.
                    "extra_binds": (extra,) if mount_policy == "legacy" else (),
                }
            ),
        }
    )
    access = _access(plan, egress="site")

    runtime, capture, _ = _judge(tmp_path, monkeypatch, plan, access, profile=profile)

    assert capture.options["network_mode"] == "public"
    assert capture.options["declared_network_mode"] == "no-network"
    assert "--net" not in capture.argv
    binds = _binds(capture.argv)
    assert not any(bind.startswith(f"{extra}:") for bind in binds)
    assert not any("/rsi-cache/" in bind for bind in binds)
    environment = capture.environment
    assert environment["HF_HOME"] == "/tmp/.cache/huggingface"
    assert environment["HF_DATASETS_CACHE"] == "/tmp/.cache/huggingface/datasets"
    # The same round, declared public, does get them: the guard is real.
    public = runtime._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="public",
        phase="judge",
    )
    if mount_policy == "scoped":
        assert any("/rsi-cache/" in bind for bind in _binds(public))
    else:
        assert any(bind.startswith(f"{extra}:") for bind in _binds(public))


def test_pinned_egress_runs_the_judge_behind_its_own_provider_proxy(
    tmp_path, monkeypatch
):
    plan = _plan(tmp_path, agent_network="no-network", verifier_network="no-network")
    endpoints = NetworkPolicyEnforcer(run_id="native-run").pin_endpoints(
        ("https://198.51.100.8:8443/v1",)
    )
    access = _access(plan, egress="pinned", endpoints=endpoints)

    _runtime_, capture, _ = _judge(tmp_path, monkeypatch, plan, access)

    assert capture.options["network_mode"] == "no-network"
    argv = capture.argv
    assert ("--net", "--network", "none") == argv[2:5]
    # --register: the proxy serves only the relay's process, never a route.
    assert capture.command[:9] == (
        "/usr/bin/env",
        "python3",
        "/usr/local/bin/rsi-netns-relay",
        "--provider-socket",
        "/run/rsi-harness/network/provider.sock",
        "--provider-port",
        "3128",
        "--register",
        "--",
    )
    assert "--submit-socket" not in capture.command
    assert capture.command[9:] == ("/bin/bash", "/tests/test.sh")
    binds = _binds(argv)
    assert any(bind.endswith(":/run/rsi-harness/network:ro") for bind in binds)
    assert any(bind.endswith(":/usr/local/bin/rsi-netns-relay:ro") for bind in binds)
    proxy = "http://127.0.0.1:3128"
    environment = capture.launch["environment"]
    assert environment["HTTPS_PROXY"] == environment["https_proxy"] == proxy
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"


def test_pinned_egress_needs_a_no_network_verifier(tmp_path, monkeypatch):
    plan = _plan(tmp_path, agent_network="no-network", verifier_network="public")
    endpoints = NetworkPolicyEnforcer(run_id="native-run").pin_endpoints(
        ("https://198.51.100.8:8443/v1",)
    )
    runtime = _runtime(tmp_path, monkeypatch, plan)
    with pytest.raises(SetupError, match="no-network"):
        runtime.configure_judge_model_access(
            _access(plan, egress="pinned", endpoints=endpoints)
        )


def test_model_access_must_match_the_task(tmp_path, monkeypatch):
    plan = _plan(tmp_path, model_access=False)
    runtime = _runtime(tmp_path, monkeypatch, plan)
    runtime.configure_judge_model_access(None)
    with pytest.raises(SetupError, match="differs"):
        runtime.configure_judge_model_access(_access(_plan(tmp_path), egress="site"))


def test_a_multinode_judge_with_model_access_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    profile = load_cluster_profile("bluevela")
    profile = profile.model_copy(
        update={"resources": profile.resources.model_copy(update={"gpus_per_node": 4})}
    )
    runtime = _runtime(
        tmp_path, monkeypatch, plan, profile=profile, pools=_multi_pools()
    )
    with pytest.raises(SetupError, match="multi-node Judge is not supported"):
        runtime.initialize()


def test_the_isolation_fallback_comes_only_from_the_operator(tmp_path, monkeypatch):
    profile = load_cluster_profile("bluevela")
    fallback = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={"judge_isolation_fallback": "same-user"}
            )
        }
    )
    plan = _plan(tmp_path, model_access=False)

    _runtime_, capture, _ = _judge(tmp_path, monkeypatch, plan, None, profile=fallback)

    assert capture.environment["RSI_JUDGE_ISOLATION_FALLBACK"] == "same-user"
    # A profile cannot set it through an environment table either.
    with pytest.raises(ValueError, match="judge_isolation_fallback"):
        type(profile.apptainer).model_validate(
            {
                **profile.apptainer.model_dump(),
                "judge_environment": {"RSI_JUDGE_ISOLATION_FALLBACK": "same-user"},
            }
        )


def test_the_composition_resolves_egress_from_the_work_agents_network(
    tmp_path, monkeypatch
):
    for agent_network, egress in (("public", "site"), ("no-network", "pinned")):
        plan = _plan(tmp_path / agent_network, agent_network=agent_network)
        runtime = _runtime(tmp_path / agent_network, monkeypatch, plan)
        composition = object.__new__(NativeEngineComposition)
        composition.plan = plan
        composition.runtime = runtime
        composition.payload = runtime.payload
        composition.rsi_loop_config = RSILoopConfig(agent_api_key=API_KEY)
        composition.provider_urls = ("https://198.51.100.8:8443/v1",)
        access = composition.judge_model_access()
        assert access.egress == egress
        launch = json.loads(access.launch)
        assert ("HTTPS_PROXY" in launch["environment"]) == (egress == "pinned")
        assert launch["isolation_fallback"] == "none"


def test_the_operators_fallback_reaches_rsi_agents_launch_spec(tmp_path, monkeypatch):
    profile = load_cluster_profile("bluevela")
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={"judge_isolation_fallback": "same-user"}
            )
        }
    )
    plan = _plan(tmp_path)
    runtime = _runtime(tmp_path, monkeypatch, plan, profile=profile)
    composition = object.__new__(NativeEngineComposition)
    composition.plan = plan
    composition.runtime = runtime
    composition.payload = SimpleNamespace(profile=profile, run_id="native-run")
    composition.rsi_loop_config = RSILoopConfig(agent_api_key=API_KEY)
    composition.provider_urls = ()

    access = composition.judge_model_access()

    assert json.loads(access.launch)["isolation_fallback"] == "same-user"


@pytest.mark.parametrize("case", ["renewed", "other-account", "not-a-file"])
def test_each_round_gets_the_login_work_keeps_refreshing(
    tmp_path, monkeypatch, capsys, case
):
    """Work is stopped during the round, so its login store is read directly."""
    from tests.runtime.test_judge_login import codex_login, material, refresh_token

    plan = _plan(tmp_path, agent_network="public")
    start = material()
    access = _access(plan, egress="site", auth=start)
    store = tmp_path / "work-login"
    store.mkdir(mode=0o700)
    if case == "not-a-file":
        (store / "auth.json").mkdir(mode=0o700)
    else:
        account = "acc-1" if case == "renewed" else "acc-other"
        (store / "auth.json").write_bytes(codex_login(2, account=account))
        (store / "auth.json").chmod(0o600)
    make_runtime = _runtime

    def with_store(*args, **options):
        runtime = make_runtime(*args, **options)
        runtime._agent_auth_roots[PurePosixPath("/home/agent/.codex")] = store
        return runtime

    monkeypatch.setitem(globals(), "_runtime", with_store)
    runtime, capture, _ = _judge(tmp_path, monkeypatch, plan, access)

    login = json.loads(capture.auth_files["/home/agent/.codex/auth.json"])
    expected = json.loads(
        codex_login(2) if case == "renewed" else start.mounts[0].files[0].content
    )
    assert login["tokens"]["access_token"] == expected["tokens"]["access_token"]
    assert login["tokens"]["refresh_token"] == "withheld-by-rsi-harness"
    if case == "renewed":
        assert refresh_token(2) not in json.dumps(capture.launch)
        assert refresh_token(2) in capture.options["output_redact_values"]
        assert runtime.judge_model_access.work_login.secret_values >= {refresh_token(2)}
        assert "judge_login_kept" not in capsys.readouterr().out
    else:
        assert runtime.judge_model_access is access
        assert "judge_login_kept: agent-1" in capsys.readouterr().out


def test_a_renewed_login_joins_the_redaction_of_works_later_output(
    tmp_path, monkeypatch
):
    """After a round, the composition redacts the renewed tokens from Work's
    Agent output too, and the next round starts from the renewed login."""
    from dataclasses import replace

    from tests.runtime.test_judge_login import material

    plan = _plan(tmp_path, agent_network="public")
    access = _access(plan, egress="site", auth=material())
    renewed = replace(access, secret_values=access.secret_values | {"RENEWED-TOKEN"})
    runtime = _runtime(tmp_path, monkeypatch, plan)
    runtime.configure_judge_model_access(access)
    composition = object.__new__(NativeEngineComposition)
    composition.runtime = runtime
    composition.secrets = set()

    class Stop(Exception):
        pass

    class Evaluator:
        def evaluate(self, request, observer):
            runtime.judge_model_access = renewed
            raise Stop

    composition.evaluator = Evaluator()

    with pytest.raises(Stop):
        composition.evaluate(_request(tmp_path, plan), observer=None)
    assert "RENEWED-TOKEN" in composition.secrets
    assert access.secret_values <= composition.secrets


@pytest.mark.parametrize("cluster_name", ["bluevela", "slurm"])
@pytest.mark.parametrize("allow", [False, True])
def test_signal_compatibility_is_explicit_in_judge_and_launch_spec(
    tmp_path, monkeypatch, cluster_name, allow
):
    profile = load_cluster_profile(cluster_name)
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={
                    "judge_allow_missing_landlock_signal": allow,
                }
            ),
        }
    )
    # Ambient host environment never decides the operator policy.
    monkeypatch.setenv("RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL", "0" if allow else "1")
    plan = _plan(tmp_path)
    runtime = _runtime(tmp_path, monkeypatch, plan, profile=profile)
    composition = object.__new__(NativeEngineComposition)
    composition.plan = plan
    composition.runtime = runtime
    composition.payload = SimpleNamespace(profile=profile, run_id="native-run")
    composition.rsi_loop_config = RSILoopConfig(agent_api_key=API_KEY)
    composition.provider_urls = ()
    access = composition.judge_model_access()
    assert json.loads(access.launch)["allow_missing_landlock_signal"] is allow

    _, capture, _ = _judge(tmp_path, monkeypatch, plan, access, profile=profile)
    assert capture.environment["RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL"] == (
        "1" if allow else "0"
    )
    assert capture.launch["allow_missing_landlock_signal"] is allow


@pytest.mark.parametrize(
    "table", ["environment", "work_environment", "judge_environment"]
)
def test_signal_policy_cannot_be_set_in_profile_environment_tables(table):
    profile = load_cluster_profile("slurm").apptainer
    with pytest.raises(ValueError, match="judge_allow_missing_landlock_signal"):
        type(profile).model_validate(
            {
                **profile.model_dump(),
                table: {"RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL": "1"},
            }
        )


@pytest.mark.parametrize("value", ["1", "true", 1, None])
def test_signal_policy_profile_requires_boolean(value):
    profile = load_cluster_profile("slurm").apptainer
    with pytest.raises(ValueError):
        type(profile).model_validate(
            {
                **profile.model_dump(),
                "judge_allow_missing_landlock_signal": value,
            }
        )
