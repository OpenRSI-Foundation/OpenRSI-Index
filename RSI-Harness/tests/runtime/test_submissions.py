from __future__ import annotations

import json
import socket
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi_harness.errors import RetryableSubmissionError
from rsi_harness.models import ContainerRef, SubmissionReport, SubmissionStatus
from rsi_harness.runtime.submissions import (
    EmbeddedJudgeServer,
    SubmissionClosedError,
    SubmissionService,
    SubmissionShutdownTimeout,
    SubmissionStartupTimeout,
    create_submission_app,
)
from tests.factories import make_run_plan
from tests.fakes import FakeArtifactWriter, FakeClock, FakeEvaluator


def make_report(
    round_id: str = "ignored",
    *,
    status: SubmissionStatus = SubmissionStatus.COMPLETED,
    output: str = "task-authored feedback\n",
    rewards: dict[str, float] | None = None,
    score: float | None = 0.5,
    error: str | None = None,
) -> SubmissionReport:
    return SubmissionReport(
        round_id=round_id,
        status=status,
        output=output,
        rewards={"reward": 0.5} if rewards is None else rewards,
        score=score,
        exit_code=3,
        timed_out=False,
        duration_seconds=1.2,
        error=error,
    )


def registered_service(
    tmp_path,
    *,
    report: SubmissionReport | None = None,
    evaluator: FakeEvaluator | None = None,
    clock: FakeClock | None = None,
    max_submissions: int | None = None,
    cooldown_seconds: float = 0.0,
) -> tuple[SubmissionService, str, FakeEvaluator, FakeArtifactWriter]:
    writer = FakeArtifactWriter()
    selected_evaluator = evaluator or FakeEvaluator(
        report or make_report(), artifact_writer=writer
    )
    if isinstance(selected_evaluator, FakeEvaluator):
        selected_evaluator.artifact_writer = writer
    service = SubmissionService(
        evaluator=selected_evaluator,
        artifact_writer=writer,
        clock=clock or FakeClock(),
    )
    token = service.register(
        run_id="run-a",
        run_plan=make_run_plan(tmp_path),
        work_container=ContainerRef(container_id="work-a", role="work"),
        max_submissions=max_submissions,
        cooldown_seconds=cooldown_seconds,
    )
    return service, token, selected_evaluator, writer


def test_real_app_requires_this_runs_bearer_token_before_evaluation(tmp_path) -> None:
    service, token, evaluator, _ = registered_service(tmp_path / "a")
    other, other_token, _, _ = registered_service(tmp_path / "b")
    client = TestClient(create_submission_app(service))

    assert client.post("/api/v1/submit").status_code == 401
    assert (
        client.post(
            "/api/v1/submit", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/submit", headers={"Authorization": f"Bearer {other_token}"}
        ).status_code
        == 401
    )
    assert evaluator.events == []

    response = client.post(
        "/api/v1/submit", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert evaluator.events == [("evaluate", "agent-1")]
    other.close()


def test_unproven_judge_containment_closes_further_submissions(tmp_path) -> None:
    report = make_report(
        status=SubmissionStatus.INFRASTRUCTURE_ERROR,
        rewards={},
        score=None,
        error="recovery_required: live Judge containment unproven",
    )
    service, token, evaluator, writer = registered_service(tmp_path, report=report)

    feedback = service.submit(token)

    assert "status: infrastructure_error" in feedback
    assert len(writer.reports) == 1
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)
    assert evaluator.round_ids == ["agent-1"]


def test_recovery_marked_evaluator_exception_closes_further_submissions(
    tmp_path,
) -> None:
    class RecoveryMarkedEvaluator:
        recovery_required = True

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request):
            del request
            self.calls += 1
            raise OSError("submission artifact fsync failed")

    evaluator = RecoveryMarkedEvaluator()
    service = SubmissionService(
        evaluator=evaluator,
        artifact_writer=FakeArtifactWriter(),
        clock=FakeClock(),
    )
    token = service.register(
        run_id="run-a",
        run_plan=make_run_plan(tmp_path),
        work_container=ContainerRef(container_id="work-a", role="work"),
        max_submissions=2,
    )

    with pytest.raises(OSError, match="artifact fsync"):
        service.submit(token)
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)
    assert evaluator.calls == 1


def test_submission_service_never_duplicates_evaluator_owned_report_write(
    tmp_path,
) -> None:
    class DuplicateRejectingWriter:
        def __init__(self) -> None:
            self.round_ids: list[str] = []

        def record_submission(self, report) -> None:
            if report.round_id in self.round_ids:
                raise OSError("duplicate report write rejected")
            self.round_ids.append(report.round_id)

    class PersistingEvaluator:
        recovery_required = False

        def __init__(self, writer) -> None:
            self.writer = writer

        def evaluate(self, request):
            report = make_report(round_id=request.round_id)
            self.writer.record_submission(report)
            return report

    writer = DuplicateRejectingWriter()
    service = SubmissionService(
        evaluator=PersistingEvaluator(writer),
        artifact_writer=writer,
        clock=FakeClock(),
    )
    token = service.register(
        run_id="run-a",
        run_plan=make_run_plan(tmp_path),
        work_container=ContainerRef(container_id="work-a", role="work"),
    )

    feedback = service.submit(token)

    assert "round: agent-1" in feedback
    assert writer.round_ids == ["agent-1"]


@pytest.mark.parametrize(
    ("payload", "raw_json"),
    (
        pytest.param({"path": "/workspace/answer.py"}, False, id="path"),
        pytest.param({"command": ["bash", "test.sh"]}, False, id="command"),
        pytest.param({"file": "checkpoint.bin"}, False, id="file"),
        pytest.param({"image": "judge:latest"}, False, id="image"),
        pytest.param({"gpu": "GPU-secret"}, False, id="gpu"),
        pytest.param({"payload": "workspace bytes"}, False, id="payload"),
        pytest.param(None, True, id="null"),
        pytest.param("workspace-path", True, id="string"),
        pytest.param(["file-bytes"], True, id="list"),
    ),
)
def test_real_app_rejects_payload_before_evaluation(
    tmp_path, payload, raw_json
) -> None:
    service, token, evaluator, _ = registered_service(tmp_path)
    headers = {
        "Authorization": f"Bearer {token}",
    }
    request = {"content": json.dumps(payload)} if raw_json else {"json": payload}
    if raw_json:
        headers["Content-Type"] = "application/json"
    response = TestClient(create_submission_app(service)).post(
        "/api/v1/submit", headers=headers, **request
    )

    assert response.status_code == 422
    assert evaluator.events == []


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("path", "/workspace/private"),
        ("command", "bash /tests/test.sh"),
        ("file", "checkpoint.bin"),
        ("image", "private-judge-image"),
        ("gpu", "GPU-private"),
        ("payload", "workspace-bytes"),
        ("arbitrary", "control-data"),
    ),
)
def test_submit_rejects_every_query_parameter_without_consuming_round(
    tmp_path, key, value
) -> None:
    service, token, evaluator, _ = registered_service(tmp_path, max_submissions=1)
    client = TestClient(create_submission_app(service))

    rejected = client.post(
        "/api/v1/submit",
        params={key: value},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "query parameters are not allowed"
    assert evaluator.events == []
    assert service.session_state.rounds_allocated == 0
    assert service.history(token) == "No submissions yet.\n"
    accepted = client.post(
        "/api/v1/submit", headers={"Authorization": f"Bearer {token}"}
    )
    assert accepted.status_code == 200
    assert evaluator.round_ids == ["agent-1"]


@pytest.mark.parametrize(
    ("query", "body"),
    (
        ({"path": "/workspace/private"}, {"command": "test"}),
        ({"arbitrary": "value"}, "file-bytes"),
        ({"gpu": "GPU-private"}, None),
    ),
)
def test_query_is_rejected_before_auth_body_and_service(tmp_path, query, body) -> None:
    service, token, evaluator, _ = registered_service(tmp_path)
    client = TestClient(create_submission_app(service))

    rejected = client.post(
        "/api/v1/submit",
        params=query,
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "query parameters are not allowed"
    assert evaluator.events == []
    assert service.session_state.rounds_allocated == 0
    assert service.history(token) == "No submissions yet.\n"


def test_history_rejects_query_before_auth_without_changing_history(tmp_path) -> None:
    service, token, evaluator, _ = registered_service(tmp_path)
    service.submit(token)
    expected = service.history(token)

    rejected = TestClient(create_submission_app(service)).get(
        "/api/v1/history", params={"path": "/other/run"}
    )

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "query parameters are not allowed"
    assert evaluator.round_ids == ["agent-1"]
    assert service.session_state.rounds_allocated == 1
    assert service.history(token) == expected


def test_service_stores_only_a_one_way_hash_of_generated_token(
    tmp_path, monkeypatch
) -> None:
    generated = "raw-control-secret"
    calls: list[int] = []

    def token_urlsafe(size: int) -> str:
        calls.append(size)
        return generated

    monkeypatch.setattr(
        "rsi_harness.runtime.submissions.secrets.token_urlsafe", token_urlsafe
    )
    service, token, _, _ = registered_service(tmp_path)

    assert token == generated
    assert calls == [32]
    state = service.session_state
    assert state.token_hash != generated
    assert generated not in state.model_dump_json()


def test_two_concurrent_submissions_are_serialized_with_stable_rounds(tmp_path) -> None:
    evaluator = FakeEvaluator(make_report(), delay_seconds=0.05)
    service, token, _, writer = registered_service(tmp_path, evaluator=evaluator)
    barrier = threading.Barrier(3)
    results: list[str] = []

    def submit() -> None:
        barrier.wait()
        results.append(service.submit(token))

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert evaluator.round_ids == ["agent-1", "agent-2"]
    assert evaluator.max_concurrency == 1
    assert [report.round_id for report in writer.reports] == ["agent-1", "agent-2"]
    assert sum("round: agent-1" in result for result in results) == 1
    assert sum("round: agent-2" in result for result in results) == 1


def test_retryable_submission_refunds_round_and_cooldown_before_agent_one_retries(
    tmp_path,
) -> None:
    class RetryOnceEvaluator:
        recovery_required = False

        def __init__(self) -> None:
            self.round_ids: list[str] = []

        def evaluate(self, request):
            self.round_ids.append(request.round_id)
            if len(self.round_ids) == 1:
                raise RetryableSubmissionError(
                    "release all Work GPU processes before retrying"
                )
            return make_report(round_id=request.round_id)

    evaluator = RetryOnceEvaluator()
    service = SubmissionService(
        evaluator=evaluator, artifact_writer=FakeArtifactWriter(), clock=FakeClock()
    )
    token = service.register(
        run_id="run-a",
        run_plan=make_run_plan(tmp_path),
        work_container=ContainerRef(container_id="work-a", role="work"),
        max_submissions=1,
        cooldown_seconds=60,
    )

    with pytest.raises(RetryableSubmissionError):
        service.submit(token)

    assert service.session_state.rounds_allocated == 0
    assert service.session_state.last_allocated_at is None
    assert service.reports == ()
    assert service.history(token) == "No submissions yet.\n"
    assert "round: agent-1" in service.submit(token)
    assert evaluator.round_ids == ["agent-1", "agent-1"]
    assert service.session_state.rounds_allocated == 1


def test_retryable_submission_keeps_round_lock_until_reservation_is_refunded(
    tmp_path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class RetryOnceEvaluator:
        recovery_required = False

        def __init__(self) -> None:
            self.round_ids: list[str] = []

        def evaluate(self, request):
            self.round_ids.append(request.round_id)
            if len(self.round_ids) == 1:
                entered.set()
                release.wait(timeout=2)
                raise RetryableSubmissionError(
                    "release all Work GPU processes before retrying"
                )
            return make_report(round_id=request.round_id)

    evaluator = RetryOnceEvaluator()
    service = SubmissionService(
        evaluator=evaluator, artifact_writer=FakeArtifactWriter(), clock=FakeClock()
    )
    token = service.register(
        run_id="run-a",
        run_plan=make_run_plan(tmp_path),
        work_container=ContainerRef(container_id="work-a", role="work"),
        max_submissions=1,
    )
    errors: list[BaseException] = []

    def submit() -> None:
        try:
            service.submit(token)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=submit)
    second = threading.Thread(target=submit)
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert all(not thread.is_alive() for thread in (first, second))
    assert [type(error) for error in errors] == [RetryableSubmissionError]
    assert evaluator.round_ids == ["agent-1", "agent-1"]
    assert service.session_state.rounds_allocated == 1


def test_first_round_verifier_logs_are_isolated_between_runs(tmp_path) -> None:
    plan = make_run_plan(tmp_path)
    observed: list[Path] = []

    class DirectoryClaimingEvaluator:
        recovery_required = False

        def evaluate(self, request):
            request.verifier_logs.mkdir(parents=True, mode=0o700)
            observed.append(request.verifier_logs)
            return make_report(round_id=request.round_id)

    for run_id in ("run-a", "run-b"):
        service = SubmissionService(
            evaluator=DirectoryClaimingEvaluator(),
            artifact_writer=FakeArtifactWriter(),
            clock=FakeClock(),
        )
        token = service.register(
            run_id=run_id,
            run_plan=plan,
            work_container=ContainerRef(container_id=f"work-{run_id}", role="work"),
        )
        service.submit(token)

    assert observed == [
        tmp_path.resolve() / "logs/runs/run-a/minimal-gpu/verifier/agent-1",
        tmp_path.resolve() / "logs/runs/run-b/minimal-gpu/verifier/agent-1",
    ]
    assert all(path.is_dir() for path in observed)


def test_cooldown_and_maximum_budget_share_round_allocation_authority(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    service, token, evaluator, _ = registered_service(
        tmp_path,
        clock=clock,
        max_submissions=2,
        cooldown_seconds=10,
    )

    first = service.submit(token)
    with pytest.raises(SubmissionClosedError, match="cooldown"):
        service.submit(token)
    clock.current += timedelta(seconds=10)
    second = service.submit(token)
    clock.current += timedelta(seconds=10)
    with pytest.raises(SubmissionClosedError, match="budget"):
        service.submit(token)

    assert evaluator.round_ids == ["agent-1", "agent-2"]
    assert "remaining: 1" in first
    assert "remaining: 0" in second


def test_history_requires_token_and_contains_only_this_run(tmp_path) -> None:
    service, token, _, _ = registered_service(tmp_path / "a")
    other, other_token, _, _ = registered_service(
        tmp_path / "b", report=make_report(output="other-run-secret\n")
    )
    service.submit(token)
    other.submit(other_token)

    assert (
        service.history(token).count(
            "Full verifier output: /run/rsi-harness/feedback/agent-1.log\n"
        )
        == 1
    )
    assert "task-authored feedback" not in service.history(token)
    assert "other-run-secret" not in service.history(token)
    with pytest.raises(PermissionError):
        service.history(other_token)


def test_feedback_points_to_complete_output_inside_work_without_inlining_it(
    tmp_path,
) -> None:
    service, token, _, writer = registered_service(tmp_path, max_submissions=3)

    feedback = service.submit(token)

    assert feedback == (
        "Full verifier output: /run/rsi-harness/feedback/agent-1.log\n"
        "--- rsi-harness submission ---\n"
        "round: agent-1\n"
        "status: completed\n"
        'reward: {"reward": 0.5}\n'
        "score: 0.5\n"
        "exit_code: 3\n"
        "timed_out: false\n"
        "duration_seconds: 1.2\n"
        "remaining: 2\n"
    )
    assert "task-authored feedback" not in feedback
    assert writer.reports == [make_report().model_copy(update={"round_id": "agent-1"})]


def test_submission_assigns_the_complete_output_host_path_to_the_judge(
    tmp_path,
) -> None:
    service, token, evaluator, _ = registered_service(tmp_path)

    service.submit(token)

    request = evaluator.requests[0]
    assert getattr(request, "verifier_output", None) == (
        request.run_plan.paths.logs / "runs/run-a/minimal-gpu/feedback/agent-1.log"
    )


@pytest.mark.parametrize(
    "output", ("task-authored feedback\n", "task-authored feedback")
)
def test_feedback_never_inlines_output_and_has_one_metadata_footer(
    tmp_path, output
) -> None:
    service, token, _, _ = registered_service(
        tmp_path, report=make_report(output=output)
    )

    feedback = service.submit(token)

    assert output not in feedback
    assert feedback.startswith(
        "Full verifier output: /run/rsi-harness/feedback/agent-1.log\n"
    )
    assert feedback.count("--- rsi-harness submission ---\n") == 1


def test_invalid_reward_is_null_with_parser_error_and_no_fake_score(tmp_path) -> None:
    report = make_report(
        status=SubmissionStatus.VERIFIER_ERROR,
        rewards={},
        score=None,
        error="reward.json is malformed",
    )
    service, token, _, _ = registered_service(tmp_path, report=report)

    feedback = service.submit(token)

    assert "reward: null\n" in feedback
    assert "error: reward.json is malformed\n" in feedback
    assert "score:" not in feedback
    assert "reward: 0" not in feedback


def test_close_rejects_new_submissions_and_drains_current_round(tmp_path) -> None:
    evaluator = FakeEvaluator(make_report(), delay_seconds=0.08)
    service, token, _, writer = registered_service(tmp_path, evaluator=evaluator)
    worker = threading.Thread(target=service.submit, args=(token,))
    worker.start()
    assert evaluator.started.wait(timeout=1)

    service.close(timeout_seconds=1)
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(writer.reports) == 1
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)


def test_close_timeout_is_typed_without_corrupting_inflight_report(tmp_path) -> None:
    release = threading.Event()
    evaluator = FakeEvaluator(make_report(), release=release)
    service, token, _, writer = registered_service(tmp_path, evaluator=evaluator)
    worker = threading.Thread(target=service.submit, args=(token,))
    worker.start()
    assert evaluator.started.wait(timeout=1)

    with pytest.raises(SubmissionShutdownTimeout):
        service.close(timeout_seconds=0.01)
    release.set()
    worker.join(timeout=1)

    assert len(writer.reports) == 1


def test_embedded_server_uses_selected_bind_and_bridge_address(tmp_path) -> None:
    service, token, evaluator, _ = registered_service(tmp_path)
    server = EmbeddedJudgeServer(
        service,
        bind_host="127.0.0.1",
        port=0,
        bridge_gateway="127.0.0.1",
        startup_timeout_seconds=2,
        shutdown_timeout_seconds=2,
    )

    endpoint = server.start()
    try:
        assert endpoint.url.startswith("http://127.0.0.1:")
        import httpx

        response = httpx.post(
            f"{endpoint.url}/api/v1/submit",
            headers={"Authorization": f"Bearer {token}"},
            timeout=2,
        )
        assert response.status_code == 200
        assert evaluator.events == [("evaluate", "agent-1")]
    finally:
        server.stop()


def test_embedded_server_stop_rejects_future_requests(tmp_path) -> None:
    service, token, _, _ = registered_service(tmp_path)
    server = EmbeddedJudgeServer(
        service,
        bind_host="127.0.0.1",
        port=0,
        bridge_gateway="127.0.0.1",
    )
    endpoint = server.start()
    server.stop()

    with pytest.raises(SubmissionClosedError):
        service.submit(token)
    assert endpoint.url.startswith("http://127.0.0.1:")


def test_embedded_server_stop_drains_response_and_artifact_before_join(
    tmp_path,
) -> None:
    import httpx

    release = threading.Event()
    evaluator = FakeEvaluator(make_report(), release=release)
    service, token, _, writer = registered_service(tmp_path, evaluator=evaluator)
    server = EmbeddedJudgeServer(
        service,
        bind_host="127.0.0.1",
        port=0,
        bridge_gateway="127.0.0.1",
        shutdown_timeout_seconds=1,
    )
    endpoint = server.start()
    responses: list[httpx.Response] = []
    stop_errors: list[BaseException] = []
    requester = threading.Thread(
        target=lambda: responses.append(
            httpx.post(
                f"{endpoint.url}/api/v1/submit",
                headers={"Authorization": f"Bearer {token}"},
                timeout=2,
            )
        )
    )

    def stop() -> None:
        try:
            server.stop()
        except BaseException as error:
            stop_errors.append(error)

    requester.start()
    assert evaluator.started.wait(timeout=1)
    stopper = threading.Thread(target=stop)
    stopper.start()
    stopper.join(timeout=0.02)
    assert stopper.is_alive()
    release.set()
    requester.join(timeout=2)
    stopper.join(timeout=2)

    assert stop_errors == []
    assert responses[0].status_code == 200
    assert len(writer.reports) == 1
    assert not requester.is_alive()
    assert not stopper.is_alive()


def test_embedded_server_reports_thread_failure_as_typed_startup_failure(
    tmp_path, monkeypatch
) -> None:
    service, _, _, _ = registered_service(tmp_path)

    def fail_run(self, sockets=None) -> None:
        del self, sockets
        raise RuntimeError("uvicorn thread failed")

    monkeypatch.setattr("rsi_harness.runtime.submissions._ReadyServer.run", fail_run)
    server = EmbeddedJudgeServer(
        service,
        bind_host="127.0.0.1",
        port=0,
        bridge_gateway="127.0.0.1",
        startup_timeout_seconds=1,
    )

    with pytest.raises(SubmissionStartupTimeout) as captured:
        server.start()

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_occupied_port_is_typed_startup_failure_without_lifecycle_leak(
    tmp_path,
) -> None:
    service, _, _, _ = registered_service(tmp_path)
    occupied = socket.socket()
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    port = int(occupied.getsockname()[1])
    server = EmbeddedJudgeServer(
        service,
        bind_host="127.0.0.1",
        port=port,
        bridge_gateway="127.0.0.1",
    )

    try:
        with pytest.raises(
            SubmissionStartupTimeout,
            match=rf"127\.0\.0\.1:{port}",
        ):
            server.start()
        server.stop()
    finally:
        occupied.close()

    replacement = socket.socket()
    replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        replacement.bind(("127.0.0.1", port))
    finally:
        replacement.close()
