"""E2B env sandboxes on the cluster backends, over the in-process E2B fake.

The broker, lifecycle, journal and endpoint server are the real ones; the
E2B SDK (tests/remote_fake.py), the scheduler and Apptainer are faked.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rsi_harness.cluster.bluevela import engine
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.bluevela.runtime import (
    ApptainerAgentRuntime,
    NativeJudgeEvaluator,
)
from rsi_harness.cluster.bluevela.sandbox import (
    ClusterSandbox,
    recover_cluster_sandboxes,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentRunResult,
    ContainerRef,
    EvaluationRequest,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_contracts import SandboxEnvGrant
from rsi_harness.runtime.sandbox_policy import (
    validate_cluster_sandbox,
    validate_sandbox_policy,
)
from rsi_harness.runtime.sandbox_server import SANDBOX_TARGET
from tests.cluster.bluevela.test_adapter import (
    RecordingScheduler,
    _profile,
    _request,
)
from tests.cluster.bluevela.test_adapter import (
    _fake_codex_runtime as _fake_codex_runtime,
)
from tests.cluster.bluevela.test_engine import _multi_payload, _payload
from tests.factories import make_run_plan, write_harbor_task
from tests.remote_fake import FAKES, RemoteFake
from tests.runtime.test_sandbox_budget import authority, make_lease
from tests.runtime.test_sandbox_e2b import (
    KEY_ENV,
    TASK,
    Kit,
    _crashed_run,
    e2b_grant,
    e2b_policy_text,
    pull,
    ready,
    sandbox_of,
    sdk_installed,
)
from tests.sandbox_helpers import (
    env_policy_toml,
    load_policy_text,
    make_env_task,
    make_sandbox_task,
)

KEY = "e2b_cluster_test_key_0123456789"
FIXTURE = Path(__file__).parents[2] / "fixtures/tasks/minimal-bluevela-multinode"
WORK_ONLY = TASK[: TASK.index("[metadata.rsi_harness.sandbox.environments.judge]")]


def _cluster_task(tmp_path, sandbox=TASK, *, multi_node=False):
    text = (FIXTURE / "task.toml").read_text()
    if not multi_node:
        text = (
            text.replace("gpus = 16", "gpus = 2")
            .replace("gpus = 8", "gpus = 2")
            .replace("timeout_sec = 180", "timeout_sec = 3600")
            .replace("timeout_sec = 30", "timeout_sec = 3600")
        )
    return write_harbor_task(tmp_path, task_toml=text + sandbox).resolve()


def _e2b_policy(tmp_path):
    return load_policy_text(tmp_path, e2b_policy_text(), "e2b-policy.toml")


@pytest.fixture
def short_root():
    # Unix socket paths must stay under 108 bytes; pytest's tmp_path may not.
    root = Path(tempfile.mkdtemp(prefix="rsi-cl-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- policy ---------------------------------------------------------------------


def test_a_cluster_accepts_only_environment_tasks_with_the_e2b_backend(tmp_path):
    task = make_env_task(TASK)
    validate_sandbox_policy(task, _e2b_policy(tmp_path), "cluster")
    for policy in (None, load_policy_text(tmp_path, env_policy_toml())):
        with pytest.raises(SetupError, match="sandbox.*local.*Docker.*e2b"):
            validate_sandbox_policy(task, policy, "cluster")
    with pytest.raises(SetupError, match="sandbox.*local.*Docker"):
        validate_sandbox_policy(make_sandbox_task(), _e2b_policy(tmp_path), "cluster")
    # The Engine re-checks the frozen plan: only an E2B env grant passes.
    grant = e2b_grant(tmp_path)
    assert validate_cluster_sandbox(task, grant, multi_node=False) is grant
    assert validate_cluster_sandbox(None, None, multi_node=True) is None
    with pytest.raises(SetupError, match="local.*Docker"):
        validate_cluster_sandbox(task, None, multi_node=False)
    with pytest.raises(SetupError, match="judge: unsupported on multi-node"):
        validate_cluster_sandbox(task, grant, multi_node=True)
    work_only = e2b_grant(tmp_path, task=WORK_ONLY)
    assert (
        validate_cluster_sandbox(make_env_task(WORK_ONLY), work_only, multi_node=True)
        is work_only
    )


# -- submit host ----------------------------------------------------------------


@pytest.mark.parametrize("cls", [BlueVelaClusterAdapter, SlurmClusterAdapter])
def test_the_adapter_freezes_the_e2b_grant_in_the_engine_payload(
    tmp_path, monkeypatch, cls
):
    from tests.cluster.slurm.test_adapter import RecordingSlurmScheduler

    monkeypatch.setenv(KEY_ENV, KEY)
    sdk_installed(monkeypatch)
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    if cls is SlurmClusterAdapter:
        profile = load_cluster_profile("slurm", {"USER": "alice"}).model_copy(
            update={
                "storage": profile.storage,
                "apptainer": profile.apptainer,
                "builder": profile.builder,
            }
        )
        scheduler = RecordingSlurmScheduler()
    adapter = cls(profile, scheduler=scheduler, agent_version_resolver=lambda _: "1")
    task = _cluster_task(tmp_path)
    request = _request(tmp_path, dry_run=False).model_copy(
        update={"task_dir": task, "sandbox_policy": _e2b_policy(tmp_path)}
    )

    adapter.run(request)

    run_spec = scheduler.specs[-1]
    payload_path = run_spec.script_path.with_suffix(".json")
    payload = load_engine_payload(payload_path)
    grant = payload.run_plan.sandbox
    assert isinstance(grant, SandboxEnvGrant)
    assert grant.environments.host.backend == "e2b"
    assert grant.environments.host.e2b.api_key_env == KEY_ENV
    assert KEY not in payload_path.read_text()


@pytest.mark.parametrize("refusal", ["sdk", "variable", "key", "auth"])
def test_the_submit_host_checks_its_e2b_setup_before_submit(
    tmp_path, monkeypatch, refusal
):
    """The Engine runs the submit host's interpreter and environment."""
    from tests.cluster.slurm.test_adapter import RecordingSlurmScheduler

    monkeypatch.setenv(KEY_ENV, KEY)
    if refusal == "sdk":
        sdk_installed(monkeypatch, None)
    else:
        sdk_installed(monkeypatch)
    if refusal == "variable":
        monkeypatch.setenv("E2B_SANDBOX_URL", "https://gw.e2b.example")
    if refusal == "key":
        monkeypatch.delenv(KEY_ENV)
    policy = _e2b_policy(tmp_path)
    if refusal == "auth":
        auth = tmp_path / "auth.json"
        auth.write_text("{")
        text = e2b_policy_text(
            e2b=f'api_key_env = "{KEY_ENV}"\nregistry_auth_file = "{auth}"'
        )
        policy = load_policy_text(tmp_path, text, "e2b-policy.toml")
    profile = load_cluster_profile("slurm", {"USER": "alice"}).model_copy(
        update={"storage": _profile(tmp_path).storage}
    )
    scheduler = RecordingSlurmScheduler()
    adapter = SlurmClusterAdapter(profile, scheduler=scheduler)
    task = _cluster_task(tmp_path)
    request = _request(tmp_path, dry_run=False).model_copy(
        update={"task_dir": task, "sandbox_policy": policy}
    )
    expected = {
        "sdk": r"e2b==.*found none",
        "variable": "E2B_SANDBOX_URL",
        "auth": "registry_auth_file .* is unreadable",
    }
    for dry_run in (False, True):
        with pytest.raises(SetupError, match=expected.get(refusal, "holds no key")):
            adapter.run(request.model_copy(update={"dry_run": dry_run}))
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()


def test_resume_checks_the_submit_hosts_e2b_setup(tmp_path, monkeypatch):
    from rsi_harness.runtime.recovery import AttemptGrants
    from tests.cluster.bluevela.test_resume import ResumeScheduler

    monkeypatch.setenv(KEY_ENV, KEY)
    sdk_installed(monkeypatch)
    profile = _profile(tmp_path)
    scheduler = ResumeScheduler()
    adapter = BlueVelaClusterAdapter(
        profile, scheduler=scheduler, agent_version_resolver=lambda _: "1"
    )
    task = _cluster_task(tmp_path)
    request = _request(tmp_path, dry_run=False).model_copy(
        update={"task_dir": task, "sandbox_policy": _e2b_policy(tmp_path)}
    )
    run_id = adapter.run(request).run_id
    submitted = len(scheduler.specs)

    sdk_installed(monkeypatch, None)
    with pytest.raises(SetupError, match="found none"):
        adapter.plan_resume(run_id, AttemptGrants(), dry_run=True)
    assert len(scheduler.specs) == submitted


def test_the_adapter_refuses_docker_envs_and_a_multinode_judge_before_submit(
    tmp_path,
):
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(profile, scheduler=scheduler)
    docker = load_policy_text(tmp_path, env_policy_toml())
    task = _cluster_task(tmp_path)
    request = _request(tmp_path, dry_run=True).model_copy(
        update={"task_dir": task, "sandbox_policy": docker}
    )
    with pytest.raises(SetupError, match="local.*Docker"):
        adapter.run(request)

    multi = tmp_path / "multi"
    multi.mkdir()
    request = request.model_copy(
        update={
            "task_dir": _cluster_task(multi, multi_node=True),
            "sandbox_policy": _e2b_policy(tmp_path),
        }
    )
    with pytest.raises(SetupError, match="judge: unsupported on multi-node"):
        adapter.run(request)
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()


def test_the_cli_hands_an_e2b_policy_to_the_cluster_adapter(tmp_path, monkeypatch):
    from rsi_harness import cli
    from rsi_harness.cluster.base import ClusterRunResult
    from rsi_harness.models import RunStatus

    monkeypatch.chdir(tmp_path)
    (tmp_path / "policy.toml").write_text(e2b_policy_text())
    captured = []

    class Adapter:
        def run(self, request):
            captured.append(request)
            return ClusterRunResult(
                run_id="r", status=RunStatus.PREPARING, log_dir=tmp_path
            )

    monkeypatch.setattr(cli, "build_cluster_adapter", lambda *a, **k: Adapter())
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            str(tmp_path),
            "--cluster",
            "bluevela",
            "--sandbox-policy",
            "policy.toml",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    [request] = captured
    assert request.sandbox_policy.environments.host.backend == "e2b"


# -- compute node ---------------------------------------------------------------


def test_the_engine_admits_an_e2b_plan_and_refuses_a_multinode_judge(
    tmp_path, monkeypatch
):
    class Reached(Exception):
        pass

    def reached(*_args, **_kwargs):
        raise Reached

    grant = e2b_grant(tmp_path)
    sandbox = {"sandbox": make_env_task(TASK)}
    payload = _payload(tmp_path)
    plan = payload.run_plan
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update=sandbox), "sandbox": grant}
    )
    monkeypatch.setattr(engine, "bind_lsf_devices", reached)
    with pytest.raises(Reached):
        engine.run_engine_payload(payload.model_copy(update={"run_plan": plan}))

    multi = _multi_payload(tmp_path)
    multi_task = multi.run_plan.task.model_copy(update=sandbox)
    multi_plan = multi.run_plan.model_copy(
        update={"task": multi_task, "sandbox": grant}
    )
    monkeypatch.setattr(engine, "parse_lsb_mcpu_hosts", reached)
    with pytest.raises(SetupError, match="judge: unsupported on multi-node"):
        engine.run_engine_payload(multi.model_copy(update={"run_plan": multi_plan}))


def _bound(tmp_path, root, monkeypatch, fake):
    monkeypatch.setenv(KEY_ENV, KEY)
    monkeypatch.setenv("RSI_TEST_MODAL_ID", "ak-cluster-id")
    monkeypatch.setenv("RSI_TEST_MODAL_SECRET", KEY)
    monkeypatch.setenv("RSI_TEST_DAYTONA_KEY", KEY)
    grant = e2b_grant(tmp_path, backend=fake.name)
    plan = make_run_plan(tmp_path)
    verifier = plan.task.verifier.model_copy(update={"timeout_seconds": 600})
    plan = plan.model_copy(
        update={
            "sandbox": grant,
            "task": plan.task.model_copy(update={"verifier": verifier}),
        }
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(make_lease("run-1").model_copy(update={"task_id": plan.task.task_id}))
    sandbox = ClusterSandbox(
        plan, "run-1", root / "sb", client_factory=lambda credentials, settings: fake
    )
    work_dir = sandbox.bind(authority(store))
    return sandbox, plan, store, work_dir


class _Apptainer:
    """The Apptainer ports NativeJudgeEvaluator drives, recording the order."""

    def __init__(self, root: Path, kit: Kit, work_env: str) -> None:
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.rounds = root / "rounds"
        self.rounds.mkdir()
        self.payload = SimpleNamespace(run_id="run-1")
        self.events: list[str] = []
        self.kit = kit
        self.work_env = work_env
        self.judge_env: str | None = None

    def renew_operator_login(self, _round_id):
        pass

    def pause(self, _work):
        self.events.append("pause")

    def unpause(self, _work):
        self.events.append("unpause")

    def require_work_idle(self):
        self.events.append("idle")

    def run_judge(
        self, workspace, request, environment, *, sandbox_endpoint, trajectory=None
    ):
        del workspace, environment, trajectory
        self.events.append("judge")
        # Work's envs are frozen before Work is paused and the round starts.
        assert sandbox_of(self.kit, self.work_env).paused
        assert (sandbox_endpoint.directory / "s").is_socket()
        assert sandbox_endpoint.owner.round_id == request.round_id
        judge = SimpleNamespace(
            credential=sandbox_endpoint.environment["RSI_SANDBOX_TOKEN"]
        )
        self.judge_env = ready(
            self.kit, judge, pull(self.kit, judge, "judge-pull"), "judge-env"
        )
        (request.verifier_logs / "reward.txt").write_text("1\n")
        request.verifier_output.write_text("passed\n")
        return AgentRunResult(exit_code=0, output="passed\n", full_output_captured=True)


@pytest.mark.parametrize("backend", ["e2b", "modal", "daytona"])
def test_a_judge_round_freezes_work_envs_and_closes_its_own(
    tmp_path, short_root, monkeypatch, backend
):
    fake = RemoteFake(tmp_path / backend, **FAKES[backend])
    try:
        sandbox, plan, store, work_dir = _bound(tmp_path, short_root, monkeypatch, fake)
        lifecycle = sandbox.lifecycle
        try:
            # Where the sandboxes are is durable; the key is only a secret.
            host = plan.sandbox.environments.host
            assert getattr(store.read("run-1"), f"sandbox_{backend}") == host.remote
            assert KEY in sandbox.secrets
            assert (work_dir / "s").is_socket()
            assert work_dir.parent == short_root / "sb"
            environment = sandbox.work_environment()
            assert environment["RSI_SANDBOX_SOCKET"] == SANDBOX_TARGET + "/s"
            assert environment["RSI_SANDBOX_TOKEN"] in sandbox.secrets
            lifecycle.activate_work(time.monotonic() + 600)
            kit = Kit(lifecycle.broker, fake, store, plan.sandbox)
            work = SimpleNamespace(credential=environment["RSI_SANDBOX_TOKEN"])
            work_env = ready(kit, work, pull(kit, work))

            writer = RunArtifactWriter(plan, run_id="run-1")
            writer.start()
            apptainer = _Apptainer(tmp_path, kit, work_env)
            request = EvaluationRequest(
                run_plan=plan,
                work_container=ContainerRef(container_id="work", role="work"),
                round_id="agent-1",
                verifier_logs=writer.root / "verifier/agent-1",
                verifier_output=writer.feedback_root / "agent-1.log",
            )
            observer = SimpleNamespace(resource_event=lambda *_a, **_k: None)
            report = NativeJudgeEvaluator(apptainer, writer, lifecycle).evaluate(
                request, observer
            )

            assert report.score == 1.0
            assert apptainer.events == ["pause", "idle", "judge", "unpause"]
            assert apptainer.judge_env is not None
            # The round's env is killed; Work's resumed and usable again.
            assert [box.sandbox_id for box in fake.live()] == [
                sandbox_of(kit, work_env).sandbox_id
            ]
            assert sandbox_of(kit, work_env).state == "running"
            status = lifecycle.broker.env_status(work.credential, work_env)
            assert status["state"] == "ready"
        finally:
            lifecycle.close()
            lifecycle.release_resources()
        assert fake.live() == []
        assert fake.closed
        assert not (short_root / "sb").exists()
    finally:
        fake.shutdown()


def test_busy_work_envs_make_the_submission_retryable_before_any_pause(
    tmp_path, short_root, monkeypatch
):
    from rsi_harness.errors import RetryableSubmissionError

    fake = RemoteFake(tmp_path / "e2b")
    try:
        sandbox, plan, _store, _ = _bound(tmp_path, short_root, monkeypatch, fake)
        events = []

        def busy():
            events.append("freeze")
            raise RetryableSubmissionError("sandbox operation busy")

        monkeypatch.setattr(sandbox.lifecycle, "freeze_work", busy)
        apptainer = SimpleNamespace(
            rounds=tmp_path,
            pause=lambda _w: events.append("pause"),
            renew_operator_login=lambda _round_id: None,
        )
        request = SimpleNamespace(round_id="agent-1")
        with pytest.raises(RetryableSubmissionError):
            NativeJudgeEvaluator(apptainer, None, sandbox.lifecycle).evaluate(
                request, None
            )
        assert events == ["freeze"]
        sandbox.lifecycle.close()
    finally:
        fake.shutdown()


def test_the_cluster_broker_keeps_its_ledger_beside_the_runs_lease(
    tmp_path, short_root, monkeypatch
):
    """On shared storage, where recover --cluster reads it after the job;
    the broker's own root is node-local scratch."""
    fake = RemoteFake(tmp_path / "daytona", **FAKES["daytona"])
    fake.ledger = None  # a client that keeps one
    try:
        sandbox, plan, _store, _ = _bound(tmp_path, short_root, monkeypatch, fake)
        expected = plan.paths.root / "leases" / "run-1.sandboxes.jsonl"
        assert (fake.ledger.path, fake.ledger.run_id) == (expected, "run-1")
        sandbox.lifecycle.close()
    finally:
        fake.shutdown()


def test_apptainer_mounts_each_phase_endpoint_and_starts_the_work_deadline(
    tmp_path, monkeypatch
):
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    apptainer = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )
    endpoint_dir = tmp_path / "endpoint"
    apptainer.work_sandbox = endpoint_dir
    bind = f"{endpoint_dir}:{SANDBOX_TARGET}:ro"

    def binds(phase):
        command = apptainer._base_command(
            devices=(),
            extra_binds=(),
            mount_workspace=True,
            mount_agent_home=phase == "work",
            network_mode="no-network",
            phase=phase,
        )
        return [b for o, b in zip(command, command[1:], strict=False) if o == "--bind"]

    assert bind in binds("work")
    assert bind not in binds("judge")

    calls = []
    monkeypatch.setattr(
        apptainer, "_run", lambda command, **kwargs: calls.append(kwargs) or None
    )
    started = []
    before = time.monotonic()
    apptainer.exec(
        apptainer.work_ref, ("true",), timeout_seconds=60, on_exec_start=started.append
    )
    assert before + 60 <= started[0] <= time.monotonic() + 60

    judge_dir = tmp_path / "judge-endpoint"
    endpoint = SimpleNamespace(
        directory=judge_dir,
        environment={"RSI_SANDBOX_TOKEN": "judge-token", "RSI_SANDBOX_SOCKET": "x"},
    )
    (plan.task.source_dir / "tests").mkdir(parents=True)
    (tmp_path / "assets").mkdir()
    apptainer.task_assets = tmp_path / "assets"
    apptainer.judge_tmp_root.mkdir()
    request = SimpleNamespace(
        round_id="agent-1",
        run_plan=plan,
        verifier_output=tmp_path / "out.log",
        verifier_logs=tmp_path / "logs",
    )
    apptainer.run_judge(tmp_path, request, {"A": "1"}, sandbox_endpoint=endpoint)
    judge = calls[-1]
    assert (judge_dir, PurePosixPath(SANDBOX_TARGET), True) in judge["extra_binds"]
    assert judge["environment"]["RSI_SANDBOX_TOKEN"] == "judge-token"
    assert judge["environment"]["A"] == "1"
    assert judge["output_redact_values"] == ("judge-token",)


def test_apptainer_puts_the_endpoint_rsi_sandbox_first_on_path(tmp_path, monkeypatch):
    """As on Docker: whichever phase gets an endpoint finds rsi-sandbox on
    PATH, after the image's environment (Apptainer's PREPEND_PATH)."""
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    apptainer = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        make_run_plan(tmp_path),
    )
    apptainer.work_sandbox = tmp_path / "endpoint"
    endpoint = {"RSI_SANDBOX_SOCKET": SANDBOX_TARGET + "/s", "RSI_SANDBOX_TOKEN": "t"}

    def environment(phase, values, *, broker=False):
        apptainer.work_broker = (
            SimpleNamespace(root=tmp_path / "torchrun", local_world_size=2)
            if broker
            else None
        )
        return apptainer._container_environment(
            devices=(),
            environment=values,
            mount_agent_home=phase == "work",
            declared="no-network",
            phase=phase,
        )

    assert environment("work", endpoint)["PREPEND_PATH"] == SANDBOX_TARGET
    assert environment("judge", endpoint)["PREPEND_PATH"] == SANDBOX_TARGET
    assert environment("work", endpoint, broker=True)["PREPEND_PATH"] == (
        SANDBOX_TARGET + ":/usr/local/bin"
    )
    # Without an endpoint the PATH stays the image's own.
    assert "PREPEND_PATH" not in environment("work", {"A": "1"})
    assert "PREPEND_PATH" not in environment("judge", None)


# -- recovery -------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["e2b", "modal", "daytona"])
def test_cluster_recovery_kills_the_runs_sandboxes_by_metadata(tmp_path, backend):
    made, lease, other = _crashed_run(tmp_path, backend)
    try:
        run_root = tmp_path / "runs"
        store = LeaseStore(run_root / "run-1" / "leases")
        store.write(lease)
        (run_root / "unrelated").mkdir()

        with store.lock("run-1"), pytest.raises(RuntimeError, match="still live"):
            recover_cluster_sandboxes(
                run_root, "run-1", remote_client=lambda _: made.fake
            )
        assert len(made.fake.live()) == 3

        made.fake.ledger = None  # a client that keeps one
        recovered = recover_cluster_sandboxes(
            run_root, remote_client=lambda _: made.fake
        )

        assert recovered == ("run-1",)
        # It read the ledger the run's broker kept beside the lease.
        assert made.fake.ledger.path == store.root / "run-1.sandboxes.jsonl"
        assert [box.sandbox_id for box in made.fake.live()] == [other]
        assert store.read("run-1").sandbox_envs == ()
        # Nothing left to kill: a second pass is a no-op that still succeeds.
        assert recover_cluster_sandboxes(
            run_root, "run-1", remote_client=lambda _: made.fake
        ) == ("run-1",)
    finally:
        made.fake.shutdown()


@pytest.mark.parametrize("command", ["recover", "cleanup"])
def test_the_cli_recovers_cluster_runs_through_the_profile(
    tmp_path, monkeypatch, command
):
    from rsi_harness import cli
    from rsi_harness.cluster.bluevela import sandbox as cluster_sandbox

    calls = []
    monkeypatch.setattr(
        cluster_sandbox,
        "recover_cluster_sandboxes",
        lambda root, run_id: calls.append((root, run_id)) or ("run-1",),
    )
    roots = ["--data-root", str(tmp_path), "--logs-root", str(tmp_path)]
    result = CliRunner().invoke(
        cli.app, [command, "run-1", "--cluster", "bluevela", *roots]
    )
    assert result.exit_code == 0, result.output
    profile = load_cluster_profile("bluevela")
    assert calls == [(profile.storage.run_root, "run-1")]
    refused = CliRunner().invoke(
        cli.app,
        ["cleanup", "run-1", "--cluster", "bluevela", "--delete-workspace", *roots],
    )
    assert refused.exit_code == 2


def test_compute_nodes_may_reach_e2b_through_a_credential_free_proxy():
    from pydantic import ValidationError

    from rsi_harness.runtime.sandbox_contracts import EnvE2BHost

    settings = EnvE2BHost(api_key_env=KEY_ENV, proxy="http://proxy.example:3128")
    for secret in ("http://user:pass@proxy.example:3128", "ftp://proxy.example"):
        with pytest.raises(ValidationError):
            EnvE2BHost(api_key_env=KEY_ENV, proxy=secret)
    socks = EnvE2BHost(api_key_env=KEY_ENV, proxy="socks5h://127.0.0.1:1080")
    pytest.importorskip("e2b")
    import httpx

    from rsi_harness.runtime.sandbox_e2b import e2b_client

    client = e2b_client(settings, KEY)
    assert client._opts["proxy"] == "http://proxy.example:3128"
    # The extra brings SOCKS to the registry probe's httpx, as to the SDK.
    httpx.Client(proxy=socks.proxy).close()


def test_run_native_engine_wires_the_sandbox_into_the_coordinator(
    tmp_path, short_root, monkeypatch
):
    """The Engine hands the coordinator the bind hook and the lifecycle (so
    run close kills the sandboxes), and Work and Judge see the endpoint."""
    from dataclasses import dataclass

    from rsi_harness.cluster.bluevela import runtime
    from rsi_harness.runtime import sandbox_e2b

    @dataclass(frozen=True)
    class Prepared:
        agent_name: str = "codex"
        environment: tuple[tuple[str, str], ...] = (("A", "1"),)

    class Stop(Exception):
        pass

    captured = {}

    def coordinator(**kwargs):
        captured.update(kwargs)
        raise Stop

    fake = RemoteFake(tmp_path / "e2b")
    monkeypatch.setenv(KEY_ENV, KEY)
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(short_root))
    # The backend's own client, as the Engine makes it.
    backend = sandbox_e2b.BACKEND._replace(client=lambda *_: fake)
    monkeypatch.setattr(sandbox_e2b, "BACKEND", backend)
    monkeypatch.setattr(runtime, "RunCoordinator", coordinator)
    # Provider URLs come from the Agent login, which CI runners do not have.
    monkeypatch.setattr(runtime, "_agent_provider_urls", lambda *a, **k: ())
    payload = _payload(tmp_path)
    plan = payload.run_plan.model_copy(
        update={
            "task": payload.run_plan.task.model_copy(
                update={"sandbox": make_env_task(TASK)}
            ),
            "sandbox": e2b_grant(tmp_path),
        }
    )
    payload = payload.model_copy(update={"run_plan": plan})
    try:
        with pytest.raises(Stop):
            runtime.run_native_engine(payload, plan)
        bind = captured["on_lease_ready"]
        composition = bind.__self__
        lifecycle = captured["sandbox_lifecycle"]
        assert isinstance(composition, runtime.NativeEngineComposition)
        assert lifecycle is composition.sandbox.lifecycle
        try:
            store = LeaseStore(tmp_path / "leases")
            store.write(
                make_lease(payload.run_id).model_copy(
                    update={"task_id": plan.task.task_id}
                )
            )
            bind(authority(store, payload.run_id))
            work_dir = composition.runtime.work_sandbox
            assert work_dir.parent == short_root / "sb"
            assert (work_dir / "s").is_socket()
            assert KEY in composition.secrets

            composition.start_artifacts(plan, payload.run_id)
            assert composition.evaluator.sandbox is lifecycle
            composition.agent = SimpleNamespace(prepare=lambda _request: Prepared())
            prepared = composition.prepare_agent(plan, 4)
            environment = dict(prepared.environment)
            assert environment["A"] == "1"
            assert environment["RSI_SANDBOX_SOCKET"] == SANDBOX_TARGET + "/s"
            token = environment["RSI_SANDBOX_TOKEN"]
            assert token in composition.secrets

            runs = []
            composition.agent = SimpleNamespace(
                run=lambda request: (
                    runs.append(request)
                    or AgentRunResult(exit_code=0, output="", full_output_captured=True)
                )
            )
            composition.run_agent(prepared, composition.runtime.work_ref, 600)
            assert runs[0].on_exec_start == lifecycle.activate_work
            assert token in runs[0].output_redact_values
            with pytest.raises(SetupError, match="finite Work"):
                composition.run_agent(prepared, composition.runtime.work_ref, None)

            # Run close through the coordinator's lifecycle kills Work's env.
            lifecycle.activate_work(time.monotonic() + 600)
            kit = Kit(lifecycle.broker, fake, store, plan.sandbox)
            work = SimpleNamespace(credential=token)
            ready(kit, work, pull(kit, work))
            assert len(fake.live()) == 1
        finally:
            lifecycle.close()
            lifecycle.release_resources()
        assert fake.live() == []
        assert not (short_root / "sb").exists()
    finally:
        fake.shutdown()


def test_cluster_salvage_kills_the_runs_sandboxes_and_closes_the_run(tmp_path):
    """resume --cluster closes a run whose job died: its E2B sandboxes go and
    the lease ends interrupted, as local recovery leaves one."""
    from rsi_harness.models import RunStatus
    from rsi_harness.runtime.recovery import RecoveryManager

    made, lease, other = _crashed_run(tmp_path)
    try:
        run_dir = tmp_path / "runs" / "run-1"
        store = LeaseStore(run_dir / "leases")
        store.write(
            lease.model_copy(update={"recovery_required": True, "error": "lost"})
        )
        manager = RecoveryManager(
            store=store,
            backend=None,  # type: ignore[arg-type]
            managed_root=run_dir,
            remote_client=lambda _: made.fake,
        )

        with store.lock("run-1"):
            manager.salvage_cluster("run-1", ended_at=None)

        assert [box.sandbox_id for box in made.fake.live()] == [other]
        final = store.read("run-1")
        assert final.status is RunStatus.CANCELLED
        assert final.phase == "interrupted"
        assert not final.recovery_required
        assert final.work.container_id is None
        assert all(env.state == "removed" for env in final.sandbox_envs)
    finally:
        made.fake.shutdown()
