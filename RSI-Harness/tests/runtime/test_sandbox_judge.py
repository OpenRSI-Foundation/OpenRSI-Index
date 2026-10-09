"""Judge owns the sandbox family barrier and shares its verifier deadline."""

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.errors import (
    InfrastructureError,
    RetryableSubmissionError,
    SubmissionError,
)
from rsi_harness.models import (
    AgentRunResult,
    ContainerMount,
    JudgeGPUMode,
    SubmissionStatus,
)
from rsi_harness.runtime.judge import (
    DockerJudgeRoundRuntime,
    DockerJudgeRuntimeFactory,
    JudgeRunner,
)
from rsi_harness.runtime.network import NetworkPolicyEnforcer
from rsi_harness.runtime.submissions import SubmissionService
from tests.fakes import FakeClock, FakeDockerClient, FakeFirewallBackend
from tests.runtime.test_judge import (
    RecordingJudgeResourceObserver,
    evaluate,
    make_harness,
    managed_workdir_volume,
)
from tests.runtime.test_sandbox import kit as kit


class RecordingSandboxLifecycle:
    enabled = True

    def __init__(self, root, events):
        self.events = events
        self.can_resume = True
        self.work_ended_normally = False
        self.failures = {}
        self.deadline = None
        self.frozen = False
        self.admission_open = True
        self.endpoint = SimpleNamespace(
            directory=root / "judge-sandbox",
            environment={
                "RSI_SANDBOX_SOCKET": "/run/rsi-harness/sandbox/broker.sock",
                "RSI_SANDBOX_TOKEN": "judge-sandbox-private-token",
            },
            mount=ContainerMount(
                source=root / "judge-sandbox",
                target=PurePosixPath("/run/rsi-harness/sandbox"),
                read_only=True,
            ),
        )
        self.endpoint.directory.mkdir()

    def _event(self, name, value=None):
        self.events.append((name, value))
        if name in self.failures:
            raise self.failures[name]

    def freeze_work(self):
        self._event("family_freeze")
        self.frozen = True
        self.admission_open = False

    def prepare_judge(self, round_id):
        assert self.frozen, "Judge endpoint prepared before family freeze"
        self._event("family_prepare", round_id)
        return self.endpoint

    def activate_judge(self, deadline):
        self.deadline = deadline
        self._event("family_activate", deadline)

    def close_judge(self):
        self._event("family_close")

    def resume_work(self):
        self._event("family_resume")
        self.frozen = False

    def resume_parent(self, runtime, container):
        if not self.can_resume:
            raise InfrastructureError("sandbox parent resume denied; recovery required")
        runtime.unpause(container)

    def reopen_work(self):
        self._event("family_reopen")
        self.admission_open = True

    def contain_work(self):
        self._event("family_contain")
        self.frozen = True
        self.admission_open = False


class RecordingObserver:
    def __init__(self, events):
        self.events = events

    def resource_event(self, name, **values):
        self.events.append((name, values))


def sandbox_harness(tmp_path, *, monotonic=None, event_callback=None):
    _old, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    lifecycle = RecordingSandboxLifecycle(tmp_path, runtime.events)
    observer = RecordingObserver(runtime.events)
    options = {} if monotonic is None else {"monotonic": monotonic}
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
        sandbox_lifecycle=lifecycle,
        event_callback=event_callback,
        **options,
    )
    return runner, lifecycle, runtime, snapshot, artifacts, request


def test_family_freeze_and_cleanup_bracket_parent_judge_lifecycle(tmp_path):
    """A child must never run across the Work snapshot or resume barrier."""
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    names = [name for name, _ in runtime.events]
    assert names.index("family_freeze") < names.index("work_pause_planned")
    assert names.index("pause") < names.index("snapshot_acquire")
    assert names.index("snapshot_acquire") < names.index("family_prepare")
    assert names.index("family_prepare") < names.index("judge_create")
    assert names.index("family_close") < names.index("judge_remove")
    assert names.index("judge_close") < names.index("snapshot_release")
    assert names.index("snapshot_release") < names.index("family_resume")
    assert names.index("family_resume") < names.index("unpause")
    assert names.index("work_unpaused") < names.index("family_reopen")
    assert lifecycle.admission_open
    assert not runtime.work_paused


def test_sandbox_endpoint_mount_and_credentials_are_exec_only(tmp_path):
    """Container metadata must not retain the per-round capability token."""
    runner, lifecycle, runtime, _snapshot, artifacts, request = sandbox_harness(
        tmp_path
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    spec = runtime.created_specs[0]
    assert lifecycle.endpoint.mount in spec.mounts
    assert spec.environment == ()
    assert runtime.exec_environments[0] == {
        "FIXED": "yes",
        "NVIDIA_VISIBLE_DEVICES": "GPU-c,GPU-d",
        "RSI_HARNESS_EXPECTED_GPU_UUIDS": "GPU-c,GPU-d",
        **lifecycle.endpoint.environment,
    }
    token = lifecycle.endpoint.environment["RSI_SANDBOX_TOKEN"]
    assert token not in repr(spec)
    assert token not in request[0].model_dump_json()
    assert token not in repr(runtime.events)
    assert token not in artifacts.reports[0].model_dump_json()


def test_sandbox_cli_prefix_preserves_task_verifier_path(tmp_path):
    """Adding the client must not hide a task's selected virtual environment."""
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    plan, work, logs = request
    verifier = plan.task.verifier.model_copy(
        update={"environment": (("PATH", "/opt/task-venv/bin:/usr/bin"),)}
    )
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"verifier": verifier})}
    )

    report = evaluate(runner, (plan, work, logs))

    assert report.status == SubmissionStatus.COMPLETED
    assert runtime.exec_environments[0]["PATH"] == "/opt/task-venv/bin:/usr/bin"
    assert "PATH" not in lifecycle.endpoint.environment


def test_verifier_deadline_starts_after_test_injection_and_is_shared(tmp_path):
    """Sandbox startup must spend the exact parent exec deadline."""
    clock = [10.0]
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path, monotonic=lambda: clock[0]
    )
    inject = runtime.inject_tests
    execute = runtime.exec
    captured = {}

    def delayed_injection(*args):
        inject(*args)
        clock[0] = 100.0

    def capture_exec(*args, **kwargs):
        captured.update(kwargs)
        return execute(*args, **kwargs)

    runtime.inject_tests = delayed_injection
    runtime.exec = capture_exec

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert lifecycle.deadline == 100.0 + request[0].task.verifier.timeout_seconds
    assert captured["deadline"] == lifecycle.deadline
    names = [name for name, _ in runtime.events]
    assert names.index("tests_inject") < names.index("family_activate")
    assert names.index("family_activate") < names.index("judge_exec")


def test_busy_work_family_retries_without_pause_artifact_or_round_consumption(
    tmp_path,
):
    """Busy children must reject before any parent or round mutation."""
    runner, lifecycle, runtime, snapshot, artifacts, request = sandbox_harness(tmp_path)
    lifecycle.failures["family_freeze"] = RetryableSubmissionError(
        "sandbox operation busy; retry submission after it completes"
    )
    service = SubmissionService(
        evaluator=runner, artifact_writer=artifacts, clock=FakeClock()
    )
    token = service.register(
        run_id="run-1",
        run_plan=request[0],
        work_container=request[1],
        max_submissions=1,
    )

    with pytest.raises(RetryableSubmissionError, match="sandbox operation busy") as exc:
        service.submit(token)

    assert "GPU" not in str(exc.value)
    assert service.session_state.rounds_allocated == 0
    assert artifacts.reports == []
    assert snapshot.acquired == []
    assert runtime.events == [("family_freeze", None)]
    assert not runner.recovery_required
    assert not runner.submission_closed


def test_partial_family_freeze_failure_contains_parent_and_requires_recovery(tmp_path):
    """A failed partial child pause must not leave ordinary Work running."""
    runner, lifecycle, runtime, snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    lifecycle.failures["family_freeze"] = InfrastructureError(
        "sandbox family freeze failed; recovery required"
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert runner.recovery_required
    assert "recovery_required" in report.error
    assert runtime.work_paused
    assert snapshot.acquired == []
    names = [name for name, _ in runtime.events]
    assert "family_contain" in names
    assert "recovery_required" in names
    assert "unpause" not in names


def test_child_cleanup_failure_retains_snapshot_and_work_pause(tmp_path):
    """Failed child destruction cannot authorize parent isolation release."""
    runner, lifecycle, runtime, snapshot, artifacts, request = sandbox_harness(tmp_path)
    lifecycle.failures["family_close"] = InfrastructureError("child still running")

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert report.score is None
    assert "child still running" in report.error
    assert runner.recovery_required
    assert runtime.work_paused
    assert snapshot.acquired[0].image_id in runtime.snapshot_images
    assert artifacts.reports == [report]
    names = [name for name, _ in runtime.events]
    assert names.index("family_close") < names.index("judge_remove")
    assert "snapshot_release" not in names
    assert "family_resume" not in names
    assert "unpause" not in names
    assert "recovery_required" in names


def test_expired_or_cancelled_work_family_is_never_resumed(tmp_path):
    """A completed Judge cannot reopen an expired Work phase."""
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    execute = runtime.exec

    def cancel_during_exec(*args, **kwargs):
        result = execute(*args, **kwargs)
        lifecycle.can_resume = False
        return result

    runtime.exec = cancel_during_exec

    report = evaluate(runner, request)

    names = [name for name, _ in runtime.events]
    assert "family_close" in names
    assert "family_resume" not in names
    assert "family_reopen" not in names
    assert "unpause" not in names
    assert runtime.work_paused
    assert runner.submission_closed
    assert runner.recovery_required
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR


@pytest.mark.parametrize("moment", ["exec", "remove"])
def test_work_ending_normally_mid_round_keeps_reward_and_parent_paused(
    tmp_path, moment
):
    """An accepted round outliving Work's end is judged, never resumed.

    Work may end while the verifier runs or later, during Judge cleanup.
    """
    runner, lifecycle, runtime, _snapshot, artifacts, request = sandbox_harness(
        tmp_path
    )
    step = getattr(runtime, moment)

    def work_deadline_passes(*args, **kwargs):
        result = step(*args, **kwargs)
        lifecycle.can_resume = False
        lifecycle.work_ended_normally = True
        return result

    setattr(runtime, moment, work_deadline_passes)

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert report.rewards == {"reward": 1}
    assert report.error is None
    assert artifacts.reports == [report]
    assert runner.submission_closed
    assert not runner.recovery_required
    assert runtime.work_paused
    names = [name for name, _ in runtime.events]
    # The coordinator's stream closes: a later round would re-pause Work.
    assert "submission_closed" in names
    assert names.index("family_close") < names.index("judge_remove")
    assert names.index("judge_close") < names.index("snapshot_release")
    assert not {
        "family_resume",
        "unpause",
        "family_reopen",
        "family_contain",
        "recovery_required",
    } & set(names)


def test_gpu_release_rejection_after_work_end_leaves_work_paused(tmp_path):
    """Ended Work can never retry, so its rejection must not resume it."""
    runner, lifecycle, runtime, snapshot, artifacts, request = sandbox_harness(
        tmp_path
    )
    plan, work, logs = request
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "judge_mode": JudgeGPUMode.RELEASE_ALL,
                    "judge": plan.gpu_plan.work,
                }
            )
        }
    )

    def work_deadline_while_gpus_busy(_allocation, _work):
        lifecycle.can_resume = False
        lifecycle.work_ended_normally = True
        raise SubmissionError("GPU is not quiescent")

    runner._quiescence_checker = work_deadline_while_gpus_busy

    with pytest.raises(
        RetryableSubmissionError, match="release all Work GPU processes"
    ):
        evaluate(runner, (plan, work, logs))

    assert runtime.work_paused
    assert artifacts.reports == []
    assert snapshot.acquired == []
    assert runner.submission_closed
    assert not runner.recovery_required
    names = [name for name, _ in runtime.events]
    assert "submission_closed" in names
    assert not {
        "family_resume",
        "unpause",
        "family_reopen",
        "recovery_required",
    } & set(names)


@pytest.mark.parametrize("end", ["deadline", "cancel_work", "deadline_in_cleanup"])
def test_round_activated_before_work_deadline_outlives_it_on_real_broker(
    tmp_path, kit, end
):
    """Judge 30 s before Work's end keeps its 900 s verifier deadline.

    Work holds a live child that the round's freeze pauses. Work's end, even
    during Judge cleanup, must neither fail the run closed nor resume Work.
    """
    import tempfile
    from pathlib import Path

    from rsi_harness.runtime.sandbox_contracts import SandboxError
    from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle

    broker, backend, clock = kit
    runner, _fake, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path, monotonic=clock
    )
    plan, work, logs = request
    verifier = plan.task.verifier.model_copy(update={"timeout_seconds": 900.0})
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"verifier": verifier})}
    )
    execute, remove = runtime.exec, runtime.remove
    observed = {}

    def work_ends():
        if end == "cancel_work":
            lifecycle.cancel_work()
        else:
            clock.now = 1001.0
        # The broker watchdog runs this every 0.1 s.
        broker.sweep_expired()

    def verifier_across_work_end(*args, **kwargs):
        token = kwargs["environment"]["RSI_SANDBOX_TOKEN"]
        child = broker.create(token, "offline", 120, "before")
        if end != "deadline_in_cleanup":
            work_ends()
        observed["now"] = clock.now
        observed["status"] = broker.status(token, child.child_id)
        broker.execute(token, child.child_id, ["true"], "/workspace", {}, 60)
        observed["deadline"] = backend.last_deadline
        # The paused Work child holds the other run-wide live slot.
        broker.destroy(token, child.child_id)
        observed["late"] = broker.create(token, "offline", 30, "after").state
        return execute(*args, **kwargs)

    def judge_removal_across_work_end(container):
        remove(container)
        if end == "deadline_in_cleanup":
            work_ends()

    runtime.exec = verifier_across_work_end
    runtime.remove = judge_removal_across_work_end
    with tempfile.TemporaryDirectory(prefix="rsi-m0-") as root:
        lifecycle = SandboxLifecycle()
        lifecycle.configure(broker, Path(root), "run-1", "task")
        try:
            work_token = lifecycle.prepare_work().environment["RSI_SANDBOX_TOKEN"]
            lifecycle.activate_work(1000.0)
            clock.now = 950.0
            work_child = broker.create(work_token, "offline", 120, "work-child")
            runner._sandbox_lifecycle = lifecycle
            clock.now = 970.0

            report = evaluate(runner, (plan, work, logs))

            assert report.status == SubmissionStatus.COMPLETED
            assert report.rewards == {"reward": 1}
            assert report.error is None
            assert not runner.recovery_required
            assert runner.submission_closed
            now = observed["now"]
            assert observed["status"]["expired"] is False
            assert observed["status"]["remaining_sec"] == 970.0 + 120 - now
            assert observed["late"] == "running"
            assert observed["deadline"] == now + 60
            assert not broker.recovery_required
            assert lifecycle.work_ended_normally
            assert not lifecycle.can_resume
            with pytest.raises(
                SandboxError,
                match="permission" if end == "cancel_work" else "expired",
            ):
                broker.capabilities(work_token)
            # Judge children are gone; the paused Work child waits for close().
            assert set(backend.states) == {work_child.child_id}
            assert backend.states[work_child.child_id]["Paused"]
            assert runtime.work_paused
            names = [name for name, _ in runtime.events]
            assert "submission_closed" in names
            assert "unpause" not in names
            # The paused killer ends the paused Work child without a thaw:
            # close() removes it (M0's gap, closed in M5).
            lifecycle.close()
            assert not broker.recovery_required
            assert work_child.child_id not in backend.states
            assert "unpause" not in [name for name, _ in runtime.events]
        finally:
            # These are trusted fake children; release them for cleanup.
            for state in backend.states.values():
                state["Paused"] = False
            lifecycle.close()


def test_cancellation_during_child_resume_prevents_parent_unpause(tmp_path):
    """A cancellation between the two resume steps must recontain children."""
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    resume = lifecycle.resume_work

    def cancel_during_resume():
        resume()
        lifecycle.can_resume = False

    lifecycle.resume_work = cancel_during_resume

    report = evaluate(runner, request)

    names = [name for name, _ in runtime.events]
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "unpause" not in names
    assert "family_contain" in names
    assert runtime.work_paused
    assert not lifecycle.admission_open


def test_pre_pause_observer_failure_does_not_leave_work_children_frozen(tmp_path):
    """A frozen family needs recovery even when parent pause was never invoked."""
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    observer = runner._lifecycle_observer
    record = observer.resource_event

    def fail_pause_plan(name, **values):
        if name == "work_pause_planned":
            raise OSError("pause plan persistence failed before parent mutation")
        record(name, **values)

    observer.resource_event = fail_pause_plan

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert not runtime.work_paused
    assert not lifecycle.frozen
    assert lifecycle.admission_open
    assert ("pause", "work-1") not in runtime.events


@pytest.mark.parametrize("before_parent_pause", (False, True))
def test_artifact_failure_retains_frozen_family_even_before_parent_pause(
    tmp_path,
    before_parent_pause,
):
    """Failed durable reporting must not reopen a successfully frozen family."""
    runner, lifecycle, runtime, snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )

    class FailingWriter:
        def record_submission(self, report):
            raise OSError("report fsync failed")

    runner._artifact_writer = FailingWriter()
    if before_parent_pause:
        record = runner._lifecycle_observer.resource_event

        def fail_pause_plan(name, **values):
            if name == "work_pause_planned":
                raise OSError("pause plan persistence failed")
            record(name, **values)

        runner._lifecycle_observer.resource_event = fail_pause_plan

    with pytest.raises(InfrastructureError, match="report fsync failed"):
        evaluate(runner, request)

    names = [name for name, _ in runtime.events]
    assert runner.recovery_required
    assert runner.submission_closed
    assert lifecycle.frozen
    assert not lifecycle.admission_open
    assert runtime.work_paused
    assert "family_close" in names
    assert "family_resume" not in names
    assert "unpause" not in names
    assert "recovery_required" in names
    if not before_parent_pause:
        assert "judge_contain" in names
        assert "judge_remove" not in names
        assert snapshot.acquired[0].image_id in runtime.snapshot_images


@pytest.mark.parametrize("exec_started", (False, True))
def test_pre_start_timeout_alone_waives_full_output_artifact(tmp_path, exec_started):
    """Never confuse pre-start expiry with a started verifier's lost output."""
    from rsi_harness.runtime.artifacts import RunArtifactWriter

    runner, _lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    writer = RunArtifactWriter(request[0], run_id="run-1")
    writer.start()
    runner._artifact_writer = writer
    runtime.exec_result = AgentRunResult(
        exit_code=None,
        output="",
        timed_out=True,
        exec_started=exec_started,
        full_output_captured=False,
    )

    if exec_started:
        with pytest.raises(InfrastructureError, match="complete verifier output"):
            evaluate(runner, request)
        assert runner.recovery_required
        assert runtime.work_paused
    else:
        report = evaluate(runner, request)
        assert report.status == SubmissionStatus.VERIFIER_TIMEOUT
        assert not report.verifier_output_required
        assert not runner.recovery_required
        assert not runtime.work_paused


@pytest.mark.parametrize("failure", ("family_resume", "unpause", "family_reopen"))
def test_partial_resume_failure_recontains_entire_work_family(tmp_path, failure):
    """Failures after any resume step must close admission and re-pause both sides."""
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    if failure == "unpause":
        runtime.fail_at = "unpause"
    else:
        lifecycle.failures[failure] = InfrastructureError("partial family resume")

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert runner.recovery_required
    assert runtime.work_paused
    assert lifecycle.frozen
    assert not lifecycle.admission_open
    names = [name for name, _ in runtime.events]
    assert names.index("family_contain") > names.index(failure)
    assert "work_recovery_state" in names


def test_sandbox_token_is_redacted_from_errors_and_progress(tmp_path):
    """Register the grant token before any verifier setup can report it."""
    progress = []
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path, event_callback=lambda name, value: progress.append((name, value))
    )
    token = lifecycle.endpoint.environment["RSI_SANDBOX_TOKEN"]
    lifecycle.failures["family_close"] = InfrastructureError(f"close failed {token}")
    runtime.exec_result = AgentRunResult(
        exit_code=0, output=f"echoed {token}", full_output_captured=True
    )

    report = evaluate(runner, request)

    assert token not in report.error
    assert token not in repr(progress)


def test_sandbox_token_is_redacted_from_report_without_changing_verifier_literals(
    tmp_path,
):
    runner, lifecycle, runtime, _snapshot, artifacts, request = sandbox_harness(
        tmp_path
    )
    token = lifecycle.endpoint.environment["RSI_SANDBOX_TOKEN"]
    runtime.exec_result = AgentRunResult(
        exit_code=0, output=f"token=task-literal\nechoed {token}\n",
        full_output_captured=True,
    )

    report = evaluate(runner, request)

    assert report.output == "token=task-literal\nechoed [REDACTED]\n"
    assert token not in repr(artifacts.reports)


def test_judge_round_redacts_sandbox_token_in_real_atomic_output_capture(tmp_path):
    """Exercise the Judge wrapper with the real Docker output-capture implementation."""
    from rsi_harness.models import ContainerRef
    from tests.fakes import FakeDockerContainer
    from tests.runtime.test_docker import make_runtime

    client = FakeDockerClient(chunks=(b"ordinary token=literal\nsandbox-", b"secret\n"))
    client.containers.by_id["judge"] = FakeDockerContainer("judge")
    concrete = make_runtime(client, tmp_path, role="judge")
    runtime = DockerJudgeRoundRuntime(
        runtime=concrete, provisioner=None, enforcer=None, network=None, policy=None,
        round_id="r1", planned_container_name="judge", observer=None,
    )
    output = tmp_path / "engine-root" / "judge.log"

    result = runtime.exec(
        ContainerRef(container_id="judge", role="judge"), ("true",),
        output_path=output, output_redact_values=("sandbox-secret",),
    )

    assert result.output == "ordinary token=literal\n[REDACTED]\n"
    assert output.read_bytes() == b"ordinary token=literal\n[REDACTED]\n"


def test_judge_runner_supplies_only_new_sandbox_token_for_output_redaction(tmp_path):
    runner, lifecycle, runtime, _snapshot, _artifacts, request = sandbox_harness(
        tmp_path
    )
    original = runtime.exec
    observed = []

    def execute(*args, **kwargs):
        observed.append(kwargs.get("output_redact_values"))
        return original(*args, **kwargs)

    runtime.exec = execute
    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert observed == [(lifecycle.endpoint.environment["RSI_SANDBOX_TOKEN"],)]


def test_disabled_judge_does_not_pass_deadline_or_sandbox_mount(tmp_path):
    """Legacy round ports must still work without new keyword arguments."""
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    execute = runtime.exec

    def legacy_exec(container, command, *, timeout_seconds, environment, output_path):
        return execute(
            container,
            command,
            timeout_seconds=timeout_seconds,
            environment=environment,
            output_path=output_path,
        )

    runtime.exec = legacy_exec

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert len(runtime.created_specs[0].mounts) == 1
    assert "RSI_SANDBOX_TOKEN" not in runtime.exec_environments[0]


@pytest.mark.parametrize("deadline", (None, 42.0))
def test_round_port_forwards_absolute_deadline_only_when_provided(deadline):
    """The wrapper must preserve one deadline instead of resetting timeout."""
    captured = {}

    class Runtime:
        def exec(self, *args, **kwargs):
            captured.update(kwargs)
            return AgentRunResult(exit_code=0, output="done")

    runtime = DockerJudgeRoundRuntime(
        runtime=Runtime(),
        provisioner=None,
        enforcer=None,
        network=None,
        policy=None,
        round_id="agent-1",
        planned_container_name="judge",
        observer=None,
    )
    options = {} if deadline is None else {"deadline": deadline}

    result = runtime.exec(None, ("/bin/true",), **options)

    assert result.output == "done"
    if deadline is None:
        assert "deadline" not in captured
    else:
        assert captured["deadline"] == 42.0


@pytest.mark.parametrize("enabled", (False, True))
def test_factory_passes_socket_authority_only_for_enabled_endpoint(
    tmp_path,
    monkeypatch,
    enabled,
):
    """Only an enabled phase endpoint may broaden the exact Judge mount authority."""
    import rsi_harness.runtime.judge as judge_module

    _runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    lifecycle = RecordingSandboxLifecycle(tmp_path, runtime.events)
    lifecycle.enabled = enabled
    lifecycle.frozen = True
    client = FakeDockerClient()
    enforcer = NetworkPolicyEnforcer(run_id="run-1", firewall=FakeFirewallBackend())
    observed = []
    concrete = judge_module.DockerContainerRuntime

    class RecordingRuntime(concrete):
        def __init__(self, *args, **kwargs):
            observed.append(dict(kwargs))
            # Root implements the concrete mount boundary independently.
            kwargs.pop("sandbox_socket_dir", None)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(judge_module, "DockerContainerRuntime", RecordingRuntime)
    plan = request[0]
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=enforcer,
        lifecycle_observer=RecordingJudgeResourceObserver(),
        sandbox_lifecycle=lifecycle,
    )

    round_runtime = factory(plan, "agent-1")
    round_runtime.close()

    if enabled:
        assert ("family_prepare", "agent-1") in runtime.events
        assert observed[-1]["sandbox_socket_dir"] == lifecycle.endpoint.directory
    else:
        assert ("family_prepare", "agent-1") not in runtime.events
        assert all("sandbox_socket_dir" not in values for values in observed)
