"""Judge model access: only on task opt-in, only the Agent's pinned model."""

from __future__ import annotations

import ipaddress
import json
import time
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import rsi_loop_agent_environment
from rsi_harness.models import (
    AgentRunResult,
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerTmpfs,
    SubmissionStatus,
)
from rsi_harness.runtime.judge import (
    JUDGE_ROUND_END_SLACK_SECONDS,
    DockerJudgeRuntimeFactory,
    JudgeRunner,
    _auth_owner,
)
from rsi_harness.runtime.judge_model_access import (
    JudgeModelAccess,
    resolve_judge_model_access,
)
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
)
from rsi_harness.runtime.network import (
    UNREACHABLE_DNS_SERVER,
    NetworkPolicyEnforcer,
    PinnedEndpoint,
)
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan
from tests.fakes import FakeDockerClient, FakeFirewallBackend
from tests.runtime.test_docker import make_runtime
from tests.runtime.test_judge import (
    RecordingJudgeResourceObserver,
    evaluate,
    install_managed_workdir_volume,
    install_work_volume_reference,
    make_harness,
    managed_workdir_volume,
)

API_KEY = "sk-fake-judge-model-key"
LOGIN_SECRET = "LOCAL-LOGIN-SECRET"
AUTH_OPTIONS = "rw,nosuid,nodev,noexec,mode=0700"
ENDPOINT = PinnedEndpoint(
    hostname="model.test",
    port=8443,
    addresses=(ipaddress.ip_address("198.51.100.8"),),
)


def auth_material() -> AgentAuthMaterial:
    return AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options=AUTH_OPTIONS,
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=b'{"tokens":{"access_token":"LOCAL-LOGIN-SECRET"}}',
                    ),
                ),
            ),
        ),
        secret_values=frozenset({LOGIN_SECRET}),
    )


LAUNCH = json.dumps({"version": 1, "redact": [API_KEY]}).encode()


def launched(runtime) -> list:
    """The launch specs injected, each without its round's end."""
    return [
        ({k: v for k, v in json.loads(launch).items() if k != "round_ends_at"}, owner)
        for launch, owner in runtime.injected_judge_agent
    ]


def model_access(*, auth: AgentAuthMaterial | None = None) -> JudgeModelAccess:
    return JudgeModelAccess(
        endpoints=(ENDPOINT,),
        environment=(
            ("RSI_AGENT_MODEL", "gpt-test"),
            ("RSI_AGENT_NAME", "codex"),
        ),
        auth=auth,
        secret_values=frozenset({API_KEY, *(() if auth is None else (LOGIN_SECRET,))}),
        launch=LAUNCH,
    )


def declare_model_access(plan, declared: bool = True):
    return plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={"model_access": declared}
                    )
                }
            )
        }
    )


def model_access_harness(tmp_path: Path, access: JudgeModelAccess | None, **plan):
    _, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    run_plan, work, log_dir = request
    run_plan = declare_model_access(run_plan, plan.get("declared", True))
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(run_plan),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        model_access=access,
    )
    return runner, runtime, (run_plan, work, log_dir)


def test_resolve_grants_nothing_without_the_task_opt_in(tmp_path):
    definition = make_run_plan(tmp_path).task

    assert (
        resolve_judge_model_access(
            definition, RSILoopConfig(agent_api_key=API_KEY), None, (ENDPOINT,)
        )
        is None
    )


def test_resolve_requires_a_pinned_model_endpoint(tmp_path):
    definition = declare_model_access(make_run_plan(tmp_path)).task

    with pytest.raises(SetupError, match="pinned Agent provider"):
        resolve_judge_model_access(
            definition, RSILoopConfig(agent_api_key=API_KEY), None, ()
        )


def test_resolve_reuses_the_work_agent_environment_and_credentials(tmp_path):
    definition = declare_model_access(make_run_plan(tmp_path)).task
    definition = definition.model_copy(
        update={
            "agent": definition.agent.model_copy(
                update={"model": "gpt-test", "reasoning_effort": "high"}
            )
        }
    )
    auth = auth_material()

    access = resolve_judge_model_access(
        definition,
        RSILoopConfig(
            agent_api_key=API_KEY,
            agent_api_base_url="https://model.test:8443/v1",
        ),
        auth,
        (ENDPOINT,),
    )

    assert access is not None
    # The procedure sees only which Agent, model and effort: no credential,
    # base URL or Agent HOME.
    assert dict(access.environment) == {
        "RSI_AGENT_NAME": "codex",
        "RSI_AGENT_MODEL": "gpt-test",
        "RSI_AGENT_REASONING_EFFORT": "high",
    }
    # rsi-agent's launch spec carries the Work Agent's own environment and
    # one-shot command, and copies the login files into the Agent's HOME.
    config = RSILoopConfig(
        agent_api_key=API_KEY, agent_api_base_url="https://model.test:8443/v1"
    )
    agent = create_agent("codex", config)
    spec = json.loads(access.launch)
    assert spec["environment"] == rsi_loop_agent_environment(config, agent, "gpt-test")
    assert spec["environment"]["OPENAI_API_KEY"] == API_KEY
    assert spec["agent"] == "codex" and spec["model"] == "gpt-test"
    assert spec["reasoning_effort"] == "high"
    assert spec["command"].startswith(
        "codex exec -c 'model_reasoning_effort=\"high\"' "
    )
    assert "--output-last-message {output_file}" in spec["command"]
    assert spec["schema_option"] == "--output-schema {schema_file}"
    assert spec["output"] == {"mode": "file", "fields": [], "error_field": None}
    assert spec["credentials"] == [
        {"source": "/home/agent/.codex/auth.json", "target": ".codex/auth.json"}
    ]
    assert {"/logs/verifier", "/tests", "/run/rsi-agent/private"} <= set(
        spec["protected"]
    )
    assert set(spec["redact"]) == {API_KEY, LOGIN_SECRET}
    assert spec["uid"] == spec["gid"] == 61000
    assert spec["slots"] == 16
    # The Judge's copy (no refresh token here to withhold); Work's is kept to
    # check the next round's read against.
    assert access.auth == auth and access.work_login is auth
    assert access.endpoints == (ENDPOINT,)
    assert access.extra_hosts == (("model.test", "198.51.100.8"),)
    assert {API_KEY, LOGIN_SECRET} <= access.secret_values
    assert API_KEY not in repr(access) and LOGIN_SECRET not in repr(access)


def test_an_agent_without_a_one_shot_command_is_named_in_the_spec(
    tmp_path, monkeypatch
):
    from rsi_loop.harness.agent.codex import CodexAgent

    monkeypatch.setattr(CodexAgent, "format_oneshot_cmd", lambda self, **options: None)
    definition = declare_model_access(make_run_plan(tmp_path)).task

    access = resolve_judge_model_access(
        definition, RSILoopConfig(agent_api_key=API_KEY), None, (ENDPOINT,)
    )

    spec = json.loads(access.launch)
    assert spec["command"] is None
    assert spec["unsupported"] == (
        "the run's Agent 'codex' cannot run one-shot in a Judge"
    )


def test_model_access_requires_a_launch_spec():
    with pytest.raises(SetupError, match="launch spec"):
        JudgeModelAccess(
            endpoints=(ENDPOINT,),
            environment=(),
            auth=None,
            secret_values=frozenset(),
            launch=b"",
        )


@pytest.mark.parametrize(
    ("user", "owner"),
    [
        (None, None),
        ("0", None),
        ("root", None),
        ("0:0", None),
        ("1001:1002", "1001:1002"),
        ("agent", "agent"),
    ],
)
def test_injected_credentials_belong_to_the_judge_user(user, owner):
    assert _auth_owner(user) == owner


def test_judge_receives_agent_credentials_and_redacts_them(tmp_path):
    auth = auth_material()
    runner, runtime, request = model_access_harness(tmp_path, model_access(auth=auth))
    runtime.exec_result = AgentRunResult(
        exit_code=0,
        output=f"key {API_KEY} login {LOGIN_SECRET}\n",
        full_output_captured=True,
    )

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED
    spec = runtime.created_specs[0]
    assert spec.extra_hosts == (("model.test", "198.51.100.8"),)
    # No external DNS: only the pinned names resolve.
    assert spec.dns == (UNREACHABLE_DNS_SERVER,)
    assert (
        ContainerTmpfs(target=PurePosixPath("/home/agent/.codex"), options=AUTH_OPTIONS)
        in spec.tmpfs
    )
    # The verifier user alone reads the credentials it was handed.
    assert runtime.injected_auth == [(auth, "1001:1002")]
    assert launched(runtime) == [(json.loads(LAUNCH), "1001:1002")]
    # rsi-agent checks the login against a call only up to the round's end.
    [(launch, _owner)] = runtime.injected_judge_agent
    timeout = request[0].task.verifier.timeout_seconds
    assert (
        time.time()
        < json.loads(launch)["round_ends_at"]
        <= (time.time() + timeout + JUDGE_ROUND_END_SLACK_SECONDS)
    )
    names = [name for name, _ in runtime.events]
    assert (
        names.index("judge_start")
        < names.index("auth_inject")
        < names.index("judge_agent_inject")
        < names.index("tests_inject")
    )
    environment = runtime.exec_environments[0]
    assert "OPENAI_API_KEY" not in environment
    assert environment["RSI_AGENT_MODEL"] == "gpt-test"
    assert environment["FIXED"] == "yes"
    assert {API_KEY, LOGIN_SECRET} <= set(
        runtime.exec_options[0]["output_redact_values"]
    )
    assert API_KEY not in report.output and LOGIN_SECRET not in report.output


def test_api_key_access_needs_no_credential_files(tmp_path):
    runner, runtime, request = model_access_harness(tmp_path, model_access())

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED
    assert runtime.injected_auth == []
    assert launched(runtime) == [(json.loads(LAUNCH), "1001:1002")]
    spec = runtime.created_specs[0]
    assert [str(tmpfs.target) for tmpfs in spec.tmpfs] == [
        "/tests",
        "/run/rsi-agent",
        "/run/rsi-agent/private",
    ]
    mounts = {str(mount.target): mount for mount in spec.mounts}
    code = mounts["/opt/rsi-agent"]
    launcher = mounts["/usr/local/bin/rsi-agent"]
    assert code.read_only and launcher.read_only
    assert code.source.name == "judge-agent"
    assert launcher.source == code.source / "rsi-agent"
    assert (code.source / "rsi_agent.py").is_file()
    assert "OPENAI_API_KEY" not in runtime.exec_environments[0]


def test_hidden_default_judge_gets_no_model_access(tmp_path):
    runner, runtime, request = model_access_harness(tmp_path, None, declared=False)

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.COMPLETED
    spec = runtime.created_specs[0]
    assert spec.extra_hosts == ()
    assert spec.dns == ()
    assert runtime.injected_auth == []
    assert runtime.injected_judge_agent == []
    assert [str(tmpfs.target) for tmpfs in spec.tmpfs] == ["/tests"]
    assert not any("rsi-agent" in str(mount.target) for mount in spec.mounts)
    assert "RSI_AGENT_NAME" not in runtime.exec_environments[0]
    assert "output_redact_values" not in runtime.exec_options[0]


@pytest.mark.parametrize("clean_image", [False, True])
@pytest.mark.parametrize(
    ("judge_user", "mode"), [("0", "0700"), (None, "0700"), ("1001:1002", "0755")]
)
def test_a_root_hidden_judge_keeps_its_tests_from_the_agent(
    tmp_path, judge_user, mode, clean_image
):
    runner, runtime, request = model_access_harness(tmp_path, model_access())
    run_plan, work, log_dir = request
    images = run_plan.images.model_copy(update={"judge_user": judge_user})
    verifier = {"user": None, "clean_image": clean_image}
    task = run_plan.task.model_copy(
        update={
            "verifier": run_plan.task.verifier.model_copy(update=verifier),
            "service": run_plan.task.service.model_copy(update={"user": None}),
        }
    )
    run_plan = run_plan.model_copy(update={"images": images, "task": task})
    runtime.snapshot_images[images.work_digest] = {}

    report = evaluate(runner, (run_plan, work, log_dir))

    assert report.status is SubmissionStatus.COMPLETED, report.error
    assert (runtime.created_specs[0].image == images.work_digest) is clean_image
    [tests] = [
        tmpfs
        for tmpfs in runtime.created_specs[0].tmpfs
        if str(tmpfs.target) == "/tests"
    ]
    assert tests.options == f"rw,exec,nosuid,nodev,mode={mode}"


@pytest.mark.parametrize(
    ("visibility", "clean_image"),
    [("hidden", True), ("hidden", False), ("public", True)],
)
@pytest.mark.parametrize("access", [True, False], ids=["model", "no-model"])
def test_every_judge_image_gets_the_same_rsi_agent_and_isolation_fallback(
    tmp_path, visibility, clean_image, access
):
    """rsi-agent, rsi-isolate and RSI_JUDGE_ISOLATION_FALLBACK come from the
    Harness, not the image: a clean hidden Judge, a Work-snapshot hidden
    Judge and a read-only public Judge all get the same mounts, tmpfs and
    value; rsi-isolate also without model access."""
    runner, runtime, (plan, work, log_dir) = model_access_harness(
        tmp_path, model_access() if access else None, declared=access
    )
    verifier = {"visibility": visibility, "clean_image": clean_image}
    task = plan.task.model_copy(
        update={
            "verifier": plan.task.verifier.model_copy(update=verifier),
            "tests_digest": hash_tree(plan.task.source_dir / "tests"),
        }
    )
    plan = plan.model_copy(update={"task": task})
    runtime.snapshot_images[plan.images.work_digest] = {}

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status is SubmissionStatus.COMPLETED, report.error
    spec = runtime.created_specs[0]
    assert (spec.image == plan.images.work_digest) is clean_image
    assert spec.read_only_rootfs is (visibility == "public")
    mounts = {str(mount.target): mount for mount in spec.mounts}
    isolate = mounts["/opt/rsi-isolate"]
    assert isolate.read_only and isolate.source.name == "judge-isolate"
    assert (isolate.source / "rsi_isolate.py").is_file()
    assert (isolate.source / "judge_isolation.py").is_file()
    launcher = mounts["/usr/local/bin/rsi-isolate"]
    assert launcher.read_only and launcher.source == isolate.source / "rsi-isolate"
    assert runtime.exec_environments[0]["RSI_JUDGE_ISOLATION_FALLBACK"] == "none"
    if not access:
        assert "/opt/rsi-agent" not in mounts
        return
    assert mounts["/opt/rsi-agent"].read_only
    assert mounts["/usr/local/bin/rsi-agent"].read_only
    assert {"/run/rsi-agent", "/run/rsi-agent/private"} <= {
        str(tmpfs.target) for tmpfs in spec.tmpfs
    }
    assert launched(runtime) == [(json.loads(LAUNCH), "1001:1002")]


@pytest.mark.parametrize("clean_image", [True, False])
@pytest.mark.parametrize("visibility", ["hidden", "public"])
def test_launch_spec_hides_tests_by_visibility_not_by_judge_image(
    tmp_path, visibility, clean_image
):
    definition = declare_model_access(make_run_plan(tmp_path)).task
    definition = definition.model_copy(
        update={
            "verifier": definition.verifier.model_copy(
                update={"visibility": visibility, "clean_image": clean_image}
            )
        }
    )

    access = resolve_judge_model_access(
        definition, RSILoopConfig(agent_api_key=API_KEY), None, (ENDPOINT,)
    )

    assert access is not None
    spec = json.loads(access.launch)
    # Docker has no operator fallback, whichever image the Judge starts from.
    assert spec["isolation_fallback"] == "none"
    assert ("/tests" in spec["hidden"]) is (visibility == "hidden")
    assert {"/logs/verifier", "/run/rsi-agent"} <= set(spec["hidden"])


@pytest.mark.parametrize(
    ("access", "declared", "message"),
    [
        (None, True, "declares Judge model access"),
        (model_access(), False, "task without it"),
    ],
)
def test_model_access_must_match_the_task_declaration(
    tmp_path, access, declared, message
):
    runner, runtime, request = model_access_harness(tmp_path, access, declared=declared)

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.INFRASTRUCTURE_ERROR
    assert message in (report.error or "")
    # Refused before Work was paused or any Judge resource existed.
    assert not any(name in {"pause", "judge_create"} for name, _ in runtime.events)


def test_failed_credential_injection_removes_the_judge(tmp_path):
    runner, runtime, request = model_access_harness(
        tmp_path, model_access(auth=auth_material())
    )
    runtime.fail_at = "auth"

    report = evaluate(runner, request)

    assert report.status is SubmissionStatus.INFRASTRUCTURE_ERROR
    names = [name for name, _ in runtime.events]
    assert "judge_exec" not in names
    assert "judge_remove" in names and "unpause" in names


def _factory(tmp_path, plan, access):
    (plan.task.source_dir / "tests").mkdir(parents=True)
    (plan.task.source_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    firewall = FakeFirewallBackend()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=firewall
        ),
        lifecycle_observer=RecordingJudgeResourceObserver(),
        work_container=install_work_volume_reference(client, plan),
        model_access=access,
    )
    return client, firewall, factory


def test_model_access_judge_bridge_routes_only_to_pinned_endpoints(tmp_path):
    plan = declare_model_access(make_run_plan(tmp_path))
    client, firewall, factory = _factory(tmp_path, plan, model_access())
    logs = tmp_path / "logs" / "verifier"
    logs.mkdir(parents=True)

    runtime = factory(plan, "agent-1")
    judge = runtime.create(
        ContainerSpec(
            image=plan.images.judge_ref,
            mounts=(
                ContainerMount(source=logs, target=PurePosixPath("/logs/verifier")),
            ),
        )
    )

    assert plan.task.verifier.network.mode == "no-network"
    assert client.networks.created[0]["internal"] is False
    (rules,) = firewall.installed.values()
    assert rules.role == "judge" and rules.mode == "no-network"
    assert rules.exact_endpoints == (ENDPOINT,)
    assert rules.allow_networks == ()
    runtime.remove(judge)
    runtime.close()


def test_judge_without_model_access_keeps_its_internal_bridge(tmp_path):
    plan = make_run_plan(tmp_path)
    client, firewall, factory = _factory(tmp_path, plan, None)

    runtime = factory(plan, "agent-1")
    judge = runtime.create(ContainerSpec(image=plan.images.judge_ref))

    assert client.networks.created[0]["internal"] is True
    (rules,) = firewall.installed.values()
    assert rules.exact_endpoints == ()
    runtime.remove(judge)
    runtime.close()


def test_factory_refuses_undeclared_model_access(tmp_path):
    plan = make_run_plan(tmp_path)
    client, _, factory = _factory(tmp_path, plan, model_access())

    with pytest.raises(SetupError, match="task without it"):
        factory(plan, "agent-1")
    assert client.networks.created == []


def _judge_with_auth_tmpfs(client, tmp_path, **options):
    runtime = make_runtime(client, tmp_path, role="judge", **options)
    ref = runtime.create(
        ContainerSpec(
            image="judge-image",
            tmpfs=(
                ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options=AUTH_OPTIONS,
                ),
            ),
        )
    )
    container = client.containers.by_id[ref.container_id]
    container.attrs["HostConfig"] = {"Tmpfs": {"/home/agent/.codex": AUTH_OPTIONS}}
    container.attrs["Mounts"].append(
        {"Type": "tmpfs", "Destination": "/home/agent/.codex", "RW": True}
    )
    return runtime, ref


def test_docker_hands_judge_credentials_to_the_verifier_user(tmp_path, monkeypatch):
    from rsi_harness.runtime.docker import DockerContainerRuntime

    client = FakeDockerClient()
    runtime, ref = _judge_with_auth_tmpfs(client, tmp_path, judge_model_access=True)
    streamed = []
    monkeypatch.setattr(
        DockerContainerRuntime,
        "_stream_agent_auth_file",
        lambda self, container, *, target, content, mode: streamed.append(target),
    )

    runtime.inject_agent_auth(ref, auth_material(), owner="1001:1002")

    assert streamed == [PurePosixPath("/home/agent/.codex/auth.json")]
    commands = [call["command"] for call in client.api.exec_create_calls]
    assert commands[0] == ("/bin/chmod", "0700", "--", "/home/agent/.codex")
    assert commands[1] == (
        "/bin/chown",
        "-R",
        "--",
        "1001:1002",
        "/home/agent/.codex",
    )
    assert commands[2][-2:] == ("/home/agent/.codex", "1001")
    assert '"700:$uid"' in commands[2][2]
    assert all(call["user"] == "root" for call in client.api.exec_create_calls)
    assert LOGIN_SECRET.encode() not in repr(client.api.exec_create_calls).encode()


def test_docker_judge_without_declared_access_gets_no_credentials(tmp_path):
    client = FakeDockerClient()
    runtime, ref = _judge_with_auth_tmpfs(client, tmp_path)

    with pytest.raises(SetupError, match="model-access Judge"):
        runtime.inject_agent_auth(ref, auth_material(), owner=None)
    assert client.api.exec_create_calls == []


@pytest.mark.parametrize("owner", ("", "1001;id", "-R", "a b", "../x"))
def test_docker_rejects_an_unsafe_credential_owner(tmp_path, owner):
    client = FakeDockerClient()
    runtime, ref = _judge_with_auth_tmpfs(client, tmp_path, judge_model_access=True)

    with pytest.raises(SetupError, match="owner"):
        runtime.inject_agent_auth(ref, auth_material(), owner=owner)


def test_docker_model_access_flag_is_judge_only(tmp_path):
    with pytest.raises(SetupError, match="limited to a Judge"):
        make_runtime(FakeDockerClient(), tmp_path, judge_model_access=True)
    client = FakeDockerClient()
    work = make_runtime(client, tmp_path)
    ref = work.create(ContainerSpec(image="work-image"))
    with pytest.raises(SetupError, match="owner"):
        work.inject_agent_auth(ref, auth_material(), owner="1001")


def test_container_ref_roles_cannot_borrow_judge_credentials(tmp_path):
    client = FakeDockerClient()
    runtime, _ = _judge_with_auth_tmpfs(client, tmp_path, judge_model_access=True)

    with pytest.raises(SetupError, match="limited to Work"):
        runtime.inject_agent_auth(
            ContainerRef(container_id="other", role="work"), auth_material()
        )


def _production_composition(tmp_path: Path, *, declared: bool):
    import shutil

    from rsi_harness.models import RunRequest, RunResult, RunStatus
    from rsi_harness.runtime.production import ProductionRuntimeServices
    from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    if declared:
        config = task / "task.toml"
        config.write_text(
            config.read_text() + "\n[metadata.rsi_harness.verifier]\n"
            "model_access = true\n"
        )
    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request):
            del request
            return RunResult(run_id="model-access", status=RunStatus.COMPLETED)

    ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        # An address literal: pinned without DNS.
        rsi_loop_config=RSILoopConfig(
            agent_api_key=API_KEY,
            agent_api_base_url="http://198.51.100.8:8443/v1",
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    ).run(RunRequest(task_dir=task, agent_name="codex", model="gpt-test"))
    return captured[0].work_creator.__self__


def test_production_hands_declared_model_access_to_every_judge_round(
    tmp_path, monkeypatch
):
    from rsi_harness.runtime import production

    composition = _production_composition(tmp_path, declared=True)
    access = composition.judge_model_access
    assert [(e.hostname, e.port) for e in access.endpoints] == [("198.51.100.8", 8443)]
    # The Work Agent's pinned endpoints, not a second, wider set.
    assert access.endpoints == composition.api_endpoints
    assert "OPENAI_API_KEY" not in dict(access.environment)
    assert dict(access.environment)["RSI_AGENT_MODEL"] == "gpt-test"
    assert json.loads(access.launch)["environment"]["OPENAI_API_KEY"] == API_KEY

    seen = {}

    class Stop(Exception):
        pass

    class Factory:
        def __init__(self, *args, **kwargs):
            seen["factory"] = kwargs["model_access"]

    class Runner:
        def __init__(self, **kwargs):
            seen["runner"] = kwargs["model_access"]
            self.model_access = kwargs["model_access"]

        def evaluate(self, request):
            raise Stop

    monkeypatch.setattr(production, "DockerJudgeRuntimeFactory", Factory)
    monkeypatch.setattr(production, "JudgeRunner", Runner)
    composition.work_runtime = object()
    composition.enforcer = object()
    composition.artifacts = object()
    # The Agent's transient secrets are cleared when it exits; a drained
    # Judge round keeps its model access.
    composition._clear_agent_bindings()
    request = type("Request", (), {"round_id": "agent-1", "work_container": None})

    with pytest.raises(Stop):
        composition.evaluate(request, observer=None)
    assert seen == {"factory": access, "runner": access}


def test_production_carries_a_renewed_login_to_the_next_round(tmp_path, monkeypatch):
    """Production builds a JudgeRunner per round: the access a round renewed
    is what the next round starts from, and its tokens are redacted from
    Work's later Agent output."""
    from dataclasses import replace

    from rsi_harness.runtime import production

    composition = _production_composition(tmp_path, declared=True)
    access = composition.judge_model_access
    renewed = replace(access, secret_values=access.secret_values | {"RENEWED-TOKEN"})
    built = []

    class Stop(Exception):
        pass

    class Factory:
        def __init__(self, *args, **kwargs):
            pass

    class Runner:
        def __init__(self, **kwargs):
            built.append(kwargs["model_access"])
            self.model_access = kwargs["model_access"]

        def evaluate(self, request):
            # As JudgeRunner._renew_work_login does, before the round fails.
            self.model_access = renewed
            raise Stop

    monkeypatch.setattr(production, "DockerJudgeRuntimeFactory", Factory)
    monkeypatch.setattr(production, "JudgeRunner", Runner)
    composition.work_runtime = object()
    composition.enforcer = object()
    composition.artifacts = object()
    request = type("Request", (), {"round_id": "agent-1", "work_container": None})

    for _round in range(2):
        with pytest.raises(Stop):
            composition.evaluate(request, observer=None)

    assert built == [access, renewed]
    assert composition.judge_model_access is renewed
    assert "RENEWED-TOKEN" in composition.agent_output_secrets


def test_production_without_opt_in_grants_the_judge_nothing(tmp_path):
    composition = _production_composition(tmp_path, declared=False)

    assert composition.judge_model_access is None


# -- rsi-agent in the Docker runtime ---------------------------------------------


def _agent_mounts(tmp_path: Path, **overrides) -> tuple[ContainerMount, ...]:
    from rsi_harness.judge_agent import install_judge_agent

    run_root = tmp_path / "engine-root" / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    directory = install_judge_agent(run_root)
    code = {
        "source": directory,
        "target": PurePosixPath("/opt/rsi-agent"),
        "read_only": True,
    }
    launcher = {
        "source": directory / "rsi-agent",
        "target": PurePosixPath("/usr/local/bin/rsi-agent"),
        "read_only": True,
    }
    code.update(overrides.get("code", {}))
    launcher.update(overrides.get("launcher", {}))
    return ContainerMount(**code), ContainerMount(**launcher)


@pytest.mark.parametrize(
    ("install", "program", "launcher"),
    [
        ("install_judge_agent", "rsi_agent.py", "rsi-agent"),
        ("install_judge_isolate", "rsi_isolate.py", "rsi-isolate"),
    ],
)
def test_install_is_idempotent_and_repairs_a_changed_copy(
    tmp_path, install, program, launcher
):
    from rsi_harness import judge_agent

    install = getattr(judge_agent, install)
    source = judge_agent.judge_agent_source
    directory = install(tmp_path)
    files = {
        directory / program: 0o644,
        directory / "judge_isolation.py": 0o644,
        directory / launcher: 0o755,
    }
    for path, mode in files.items():
        assert path.read_bytes() == source(path.name)
        assert path.stat().st_mode & 0o777 == mode
    (directory / program).write_text("changed")
    shared = directory / "judge_isolation.py"
    shared.unlink()
    shared.symlink_to(directory / program)
    (directory / launcher).chmod(0o777)

    assert install(tmp_path) == directory
    for path, mode in files.items():
        assert not path.is_symlink()
        assert path.read_bytes() == source(path.name)
        assert path.stat().st_mode & 0o777 == mode


def _isolate_mounts(tmp_path: Path, **overrides) -> tuple[ContainerMount, ...]:
    from rsi_harness.judge_agent import install_judge_isolate

    run_root = tmp_path / "engine-root" / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    directory = install_judge_isolate(run_root)
    code = {
        "source": directory,
        "target": PurePosixPath("/opt/rsi-isolate"),
        "read_only": True,
    }
    launcher = {
        "source": directory / "rsi-isolate",
        "target": PurePosixPath("/usr/local/bin/rsi-isolate"),
        "read_only": True,
    }
    code.update(overrides.get("code", {}))
    launcher.update(overrides.get("launcher", {}))
    return ContainerMount(**code), ContainerMount(**launcher)


@pytest.mark.parametrize("model", [True, False])
def test_docker_mounts_rsi_isolate_into_every_judge_only(tmp_path, model):
    judge = make_runtime(
        FakeDockerClient(), tmp_path, role="judge", judge_model_access=model
    )
    assert judge.create(
        ContainerSpec(image="judge-image", mounts=_isolate_mounts(tmp_path))
    )
    work = make_runtime(FakeDockerClient(), tmp_path, role="work")
    with pytest.raises(SetupError, match="rsi-isolate mounts"):
        work.create(ContainerSpec(image="image", mounts=_isolate_mounts(tmp_path)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"code": {"read_only": False}},
        {"launcher": {"read_only": False}},
        {"launcher": {"target": PurePosixPath("/opt/rsi-isolate")}},
        {"code": {"source": "elsewhere"}},
        {"code": {"source": "symlink"}},
    ],
)
def test_docker_refuses_an_inexact_rsi_isolate_mount(tmp_path, overrides):
    judge = make_runtime(FakeDockerClient(), tmp_path, role="judge")
    source = overrides.get("code", {}).get("source")
    if source in ("elsewhere", "symlink"):
        elsewhere = tmp_path / "engine-root" / "elsewhere"
        elsewhere.mkdir(parents=True)
        path = elsewhere
        if source == "symlink":
            # A link named like the copy, to a directory that is not it.
            path = tmp_path / "engine-root" / "judge-isolate"
            path.symlink_to(elsewhere)
        overrides = {"code": {"source": path}}

    with pytest.raises(SetupError, match="rsi-isolate mounts"):
        judge.create(
            ContainerSpec(
                image="judge-image", mounts=_isolate_mounts(tmp_path, **overrides)
            )
        )


def test_docker_mounts_rsi_agent_only_into_a_model_access_judge(tmp_path):
    from rsi_harness.judge_agent import JUDGE_AGENT_TMPFS

    client = FakeDockerClient()
    judge = make_runtime(client, tmp_path, role="judge", judge_model_access=True)
    ref = judge.create(
        ContainerSpec(
            image="judge-image",
            mounts=_agent_mounts(tmp_path),
            tmpfs=JUDGE_AGENT_TMPFS,
        )
    )
    assert ref.role == "judge"
    for runtime in (
        make_runtime(FakeDockerClient(), tmp_path, role="judge"),
        make_runtime(FakeDockerClient(), tmp_path, role="work"),
    ):
        with pytest.raises(SetupError, match="rsi-agent mounts"):
            runtime.create(ContainerSpec(image="image", mounts=_agent_mounts(tmp_path)))
        with pytest.raises(SetupError, match="rsi-agent state"):
            runtime.create(ContainerSpec(image="image", tmpfs=JUDGE_AGENT_TMPFS))


@pytest.mark.parametrize(
    "overrides",
    [
        {"code": {"read_only": False}},
        {"launcher": {"read_only": False}},
        {"code": {"target": PurePosixPath("/opt/other")}},
        {"launcher": {"target": PurePosixPath("/opt/rsi-agent")}},
        {"code": {"source": "elsewhere"}},
    ],
)
def test_docker_refuses_an_inexact_rsi_agent_mount(tmp_path, overrides):
    client = FakeDockerClient()
    judge = make_runtime(client, tmp_path, role="judge", judge_model_access=True)
    if overrides.get("code", {}).get("source") == "elsewhere":
        elsewhere = tmp_path / "engine-root" / "elsewhere"
        elsewhere.mkdir(parents=True)
        overrides = {"code": {"source": elsewhere}}

    with pytest.raises(SetupError):
        judge.create(
            ContainerSpec(
                image="judge-image", mounts=_agent_mounts(tmp_path, **overrides)
            )
        )


def test_docker_refuses_rsi_agent_state_with_other_options(tmp_path):
    client = FakeDockerClient()
    judge = make_runtime(client, tmp_path, role="judge", judge_model_access=True)

    with pytest.raises(SetupError, match="exact tmpfs"):
        judge.create(
            ContainerSpec(
                image="judge-image",
                tmpfs=(
                    ContainerTmpfs(
                        target=PurePosixPath("/run/rsi-agent"),
                        options="rw,mode=0777",
                    ),
                ),
            )
        )


def _judge_with_agent_state(client, tmp_path, **options):
    from rsi_harness.judge_agent import JUDGE_AGENT_TMPFS

    runtime = make_runtime(client, tmp_path, role="judge", **options)
    ref = runtime.create(ContainerSpec(image="judge-image"))
    container = client.containers.by_id[ref.container_id]
    container.attrs["HostConfig"] = {
        "Tmpfs": {str(tmpfs.target): tmpfs.options for tmpfs in JUDGE_AGENT_TMPFS}
    }
    return runtime, ref


@pytest.mark.parametrize("owner", [None, "1001:1002"])
def test_docker_streams_the_launch_spec_into_the_private_tmpfs(
    tmp_path, monkeypatch, owner
):
    from rsi_harness.runtime.docker import DockerContainerRuntime

    client = FakeDockerClient()
    runtime, ref = _judge_with_agent_state(client, tmp_path, judge_model_access=True)
    streamed = []
    monkeypatch.setattr(
        DockerContainerRuntime,
        "_stream_agent_auth_file",
        lambda self, container, *, target, content, mode: streamed.append(
            (target, content, mode)
        ),
    )

    runtime.inject_judge_agent(ref, LAUNCH, owner=owner)

    assert streamed == [
        (PurePosixPath("/run/rsi-agent/private/launch.json"), LAUNCH, 0o600)
    ]
    commands = [call["command"] for call in client.api.exec_create_calls]
    assert commands[0][-2:] == ("/run/rsi-agent", "/run/rsi-agent/private")
    if owner is None:
        assert len(commands) == 1
    else:
        assert commands[1] == ("/bin/chown", "-R", "--", owner, "/run/rsi-agent")
    assert all(call["user"] == "root" for call in client.api.exec_create_calls)
    assert API_KEY.encode() not in repr(client.api.exec_create_calls).encode()


def test_docker_gives_no_launch_spec_without_model_access(tmp_path):
    client = FakeDockerClient()
    runtime, ref = _judge_with_agent_state(client, tmp_path)

    with pytest.raises(SetupError, match="model-access Judge"):
        runtime.inject_judge_agent(ref, LAUNCH)
    assert client.api.exec_create_calls == []


def test_docker_refuses_a_launch_spec_without_its_exact_tmpfs(tmp_path):
    from rsi_harness.errors import InfrastructureError

    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge", judge_model_access=True)
    ref = runtime.create(ContainerSpec(image="judge-image"))
    client.containers.by_id[ref.container_id].attrs["HostConfig"] = {"Tmpfs": {}}

    with pytest.raises(InfrastructureError, match="not installed exactly"):
        runtime.inject_judge_agent(ref, LAUNCH)
