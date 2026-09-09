from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rsi_harness.errors import RetryableSubmissionError
from rsi_harness.integrations.submit_client import generate_submit_client
from rsi_harness.models import ContainerRef, SubmissionReport, SubmissionStatus
from rsi_harness.runtime.submissions import EmbeddedJudgeServer, SubmissionService
from tests.factories import make_run_plan
from tests.fakes import FakeArtifactWriter, FakeClock, FakeEvaluator


@dataclass(frozen=True)
class Capture:
    method: str
    authorization: str | None
    content_type: str | None
    body: bytes
    path: str


def run_submit_client_against_capture_server(
    tmp_path: Path,
    *,
    args: tuple[str, ...] = (),
    cwd_with_files: dict[str, bytes] | None = None,
    path_prefix: Path | None = None,
    path_override: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], tuple[Capture, ...], tuple[str, ...]]:
    captures: list[Capture] = []

    class Handler(BaseHTTPRequestHandler):
        def _capture(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            captures.append(
                Capture(
                    method=self.command,
                    authorization=self.headers.get("Authorization"),
                    content_type=self.headers.get("Content-Type"),
                    body=self.rfile.read(length),
                    path=self.path,
                )
            )
            response = b"judge response\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        do_GET = _capture
        do_POST = _capture

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    for name, data in (cwd_with_files or {}).items():
        (cwd / name).write_bytes(data)
    script = tmp_path / "rsi-submit"
    script.write_text(generate_submit_client())
    script.chmod(0o755)
    trace_path = tmp_path / "trace"
    env = {
        **os.environ,
        "RSI_JUDGE_URL": f"http://127.0.0.1:{server.server_port}",
        "RSI_TOKEN": "control-token",
    }
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    if path_override is not None:
        env["PATH"] = str(path_override)
    try:
        result = subprocess.run(
            (
                "strace",
                "-f",
                "-e",
                "trace=open,openat",
                "-o",
                str(trace_path),
                script,
                *args,
            ),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
    trace = trace_path.read_text()
    opened_workdir_files = tuple(
        name
        for name in (cwd_with_files or {})
        if str(cwd / name) in trace or f'"{name}"' in trace
    )
    return result, tuple(captures), opened_workdir_files


def test_submit_client_carries_control_only(tmp_path: Path) -> None:
    result, captures, opened = run_submit_client_against_capture_server(
        tmp_path,
        cwd_with_files={"checkpoint.bin": b"must-not-be-read"},
    )

    assert result.returncode == 0
    assert result.stdout == "judge response\n"
    assert len(captures) == 1
    capture = captures[0]
    assert capture.method == "POST"
    assert capture.authorization == "Bearer control-token"
    assert capture.content_type != "multipart/form-data"
    assert capture.body == b""
    assert opened == ()


def test_details_is_rejected_before_any_submission(tmp_path: Path) -> None:
    result, captures, _ = run_submit_client_against_capture_server(
        tmp_path, args=("--details",)
    )

    assert result.returncode == 2
    assert captures == ()


def test_help_is_local_and_does_not_require_control_environment(
    tmp_path: Path,
) -> None:
    """Help must remain usable before a Work control token is installed."""
    script = tmp_path / "rsi-submit"
    script.write_text(generate_submit_client())
    script.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RSI_JUDGE_URL", "RSI_TOKEN"}
    }

    result = subprocess.run(
        (script, "--help"),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        "Usage: rsi-submit [--list|--help]\n"
        "\n"
        "Options:\n"
        "  --list  Show previous submissions for this run without submitting\n"
        "  --help  Show this help message\n"
    )


def test_list_gets_authenticated_history(tmp_path: Path) -> None:
    result, captures, _ = run_submit_client_against_capture_server(
        tmp_path, args=("--list",)
    )

    assert result.returncode == 0
    assert result.stdout == "judge response\n"
    assert tuple((capture.method, capture.path) for capture in captures) == (
        ("GET", "/api/v1/history"),
    )
    assert captures[0].authorization == "Bearer control-token"


def test_unknown_flag_exits_two_before_request(tmp_path: Path) -> None:
    result, captures, _ = run_submit_client_against_capture_server(
        tmp_path, args=("--archive",)
    )

    assert result.returncode == 2
    assert captures == ()


def test_client_does_not_depend_on_curl(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_marker = tmp_path / "curl-was-called"
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        f"#!/bin/sh\nprintf called > {curl_marker!s}\nexit 99\n"
    )
    fake_curl.chmod(0o755)
    result, captures, _ = run_submit_client_against_capture_server(
        tmp_path,
        path_prefix=bin_dir,
    )

    assert result.returncode == 0
    assert len(captures) == 1
    assert not curl_marker.exists()


def test_client_falls_back_to_curl_when_python_is_unavailable(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("curl", "strace"):
        executable = shutil.which(name)
        assert executable is not None
        bin_dir.joinpath(name).symlink_to(executable)

    result, captures, opened = run_submit_client_against_capture_server(
        tmp_path,
        cwd_with_files={"checkpoint.bin": b"must-not-be-read"},
        path_override=bin_dir,
    )

    assert result.returncode == 0
    assert result.stdout == "judge response\n"
    assert len(captures) == 1
    capture = captures[0]
    assert capture.method == "POST"
    assert capture.authorization == "Bearer control-token"
    assert capture.content_type != "multipart/form-data"
    assert capture.body == b""
    assert opened == ()


def test_generated_client_runs_against_real_submission_app(tmp_path: Path) -> None:
    report = SubmissionReport(
        round_id="ignored",
        status=SubmissionStatus.COMPLETED,
        rewards={"reward": 0.75},
        score=0.75,
        output="real app feedback\n",
        exit_code=0,
        duration_seconds=0.25,
    )
    evaluator = FakeEvaluator(report)
    service = SubmissionService(
        evaluator=evaluator,
        artifact_writer=FakeArtifactWriter(),
        clock=FakeClock(),
    )
    token = service.register(
        run_id="client-run",
        run_plan=make_run_plan(tmp_path / "plan"),
        work_container=ContainerRef(container_id="work-client", role="work"),
    )
    server = EmbeddedJudgeServer(
        service,
        bind_host="127.0.0.1",
        port=0,
        bridge_gateway="127.0.0.1",
    )
    endpoint = server.start()
    workdir = tmp_path / "workdir-real"
    workdir.mkdir()
    forbidden_workspace_data = {
        "solution-path.txt": b"/workspace/private-answer.py",
        "command.txt": b"bash /tests/test.sh",
        "checkpoint.bin": b"file-bytes-must-not-cross",
        "image.txt": b"private-judge-image",
        "gpu.txt": b"GPU-private-identifier",
    }
    for name, content in forbidden_workspace_data.items():
        (workdir / name).write_bytes(content)
    script = tmp_path / "rsi-submit-real"
    script.write_text(generate_submit_client())
    script.chmod(0o755)
    environment = {
        **os.environ,
        "RSI_JUDGE_URL": endpoint.url,
        "RSI_TOKEN": token,
    }
    try:
        submitted = subprocess.run(
            (script,),
            cwd=workdir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        listed = subprocess.run(
            (script, "--list"),
            cwd=workdir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    finally:
        server.stop()

    assert submitted.returncode == 0
    assert submitted.stdout.startswith(
        "Full verifier output: /run/rsi-harness/feedback/agent-1.log\n"
    )
    assert "real app feedback" not in submitted.stdout
    assert "round: agent-1\n" in submitted.stdout
    assert listed.returncode == 0
    assert listed.stdout == submitted.stdout
    assert evaluator.events == [("evaluate", "agent-1")]
    for forbidden in forbidden_workspace_data.values():
        assert forbidden.decode() not in submitted.stdout
        assert forbidden.decode() not in listed.stdout


def test_generated_client_prints_retryable_gpu_release_detail_without_workspace_data(
    tmp_path: Path,
) -> None:
    class RetryableEvaluator:
        recovery_required = False

        def evaluate(self, _request):
            raise RetryableSubmissionError(
                "release all Work GPU processes before retrying: PID 42"
            )

    service = SubmissionService(
        evaluator=RetryableEvaluator(),
        artifact_writer=FakeArtifactWriter(),
        clock=FakeClock(),
    )
    token = service.register(
        run_id="client-run",
        run_plan=make_run_plan(tmp_path / "plan"),
        work_container=ContainerRef(container_id="work-client", role="work"),
    )
    server = EmbeddedJudgeServer(
        service, bind_host="127.0.0.1", port=0, bridge_gateway="127.0.0.1"
    )
    endpoint = server.start()
    workdir = tmp_path / "workdir-retry"
    workdir.mkdir()
    (workdir / "checkpoint.bin").write_bytes(b"must-not-be-read")
    script = tmp_path / "rsi-submit-retry"
    script.write_text(generate_submit_client())
    script.chmod(0o755)
    try:
        result = subprocess.run(
            (script,),
            cwd=workdir,
            env={
                **os.environ,
                "RSI_JUDGE_URL": endpoint.url,
                "RSI_TOKEN": token,
            },
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    finally:
        server.stop()

    assert result.returncode != 0
    assert "release all Work GPU processes before retrying: PID 42" in result.stdout
