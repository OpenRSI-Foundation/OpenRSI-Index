"""Real Docker: ``rsi-harness resume`` continues a local run as its next attempt.

Run as a non-root user with the fake firewall, on the production run path:
the oracle Agent for runs killed hard mid-Agent or mid-Judge (the Engine runs
in a child process this test SIGKILLs) and for a finished run, and every
Agent with a session of its own, on its real pinned CLI against
tests/fakes/fake_model_server.py, for native session resume.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    CompileOptions,
    RunRequest,
    RunStatus,
    SubmissionStatus,
)
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_harness.runtime.recovery import AttemptGrants, LeaseStore
from rsi_harness.runtime.rootfs_snapshot import rootfs_image_ref
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_driven_agents_docker import (
    API_KEY,
    CASES,
    DRIVER_TIMINGS,
    MAX_OUTPUT_TOKENS,
    MODEL,
    DrivenCase,
    _tls_context,
)
from tests.integration.test_driven_agents_docker import _docker as _driven_docker
from tests.integration.test_driven_agents_docker import _task as _driven_task
from tests.integration.test_public_tests_docker import _docker
from tests.integration.test_run_engine import _OneDeviceInventory
from tests.integrations.test_agent_driver import _load_fake_model_server

pytestmark = pytest.mark.integration

REPO = Path(__file__).parents[2]
RUN_LABEL = "rsi-harness.run-id"
# Long enough for any attempt here; the kill comes well before.
AGENT_TIMEOUT_SECONDS = 300
CHILD_WAIT_SECONDS = 900

_TASK_TOML = """\
schema_version = "1.4"

[task]
name = "rsi/resume-check"

[environment]
os = "linux"
workdir = "{workdir}"
gpus = 0
network_mode = "no-network"

[agent]
timeout_sec = {agent_timeout}
network_mode = "no-network"
{agent_user}
[verifier]
timeout_sec = 300
network_mode = "no-network"
"""

# Attempt 1 writes files in WORKDIR and in $HOME (outside WORKDIR), submits
# once, says so and sleeps until it is killed. Attempt 2 finds them.
_SOLVE_KILLED_MID_AGENT = """\
#!/bin/bash
set -u
marker="$HOME/rsi-resume-home-marker"
if [ ! -e "$marker" ]; then
    echo 42 > answer.txt
    echo kept-in-workdir > workdir-file.txt
    echo kept-in-home > "$marker"
    sync
    rsi-submit
    echo "rsi-resume: attempt-1 sleeping"
    sleep 900
    exit 9
fi
echo "rsi-resume: workdir-file=$(cat workdir-file.txt) home=$(cat "$marker") \
answer=$(cat answer.txt) solution=$(ls "$(dirname "$0")" | tr '\\n' ,)"
"""

# Attempt 1 asks for a slow Judge round and ends; the Engine is killed while
# the Judge runs. Attempt 2 lists the feedback Work can read.
_SOLVE_KILLED_MID_JUDGE = """\
#!/bin/bash
set -u
marker="$HOME/rsi-resume-home-marker"
if [ ! -e "$marker" ]; then
    echo 42 > answer.txt
    touch slow-judge
    echo kept-in-home > "$marker"
    sync
    exit 0
fi
rm -f slow-judge
echo "rsi-resume: feedback=[$(ls -A /run/rsi-harness/feedback | tr '\\n' ' ')]"
if grep -rq judge-partial-output /run/rsi-harness/feedback; then
    echo "rsi-resume: partial feedback visible"
else
    echo "rsi-resume: no partial feedback"
fi
"""

_SOLVE_PLAIN = """\
#!/bin/bash
set -eu
echo 42 > answer.txt
echo "rsi-resume: solved"
"""

_TEST = """\
#!/bin/bash
set -eu
if [ -e {workdir_slash}slow-judge ]; then
    # More than an output buffer, so some of it reaches the disk.
    for i in $(seq 2000); do echo "rsi-resume: judge-partial-output $i"; done
    sleep 900
fi
answer=$(cat {workdir_slash}answer.txt)
if [ "$answer" = 42 ]; then reward=1; else reward=0; fi
echo "answer=$answer"
printf '{{"reward": %s}}\\n' "$reward" > /logs/verifier/reward.json
"""


def _task(root: Path, solve: str, *, workdir: str = "/workspace") -> Path:
    task = root / "resume-check"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    non_root = workdir != "/"
    (task / "task.toml").write_text(
        _TASK_TOML.format(
            workdir=workdir,
            agent_timeout=AGENT_TIMEOUT_SECONDS,
            agent_user='user = "1000"\n' if non_root else "",
        )
    )
    (task / "instruction.md").write_text("Write 42 to answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        + (
            "RUN mkdir -p /workspace && chown 1000:1000 /workspace\n"
            if non_root
            else ""
        )
        + f"WORKDIR {workdir}\n"
    )
    (task / "tests" / "test.sh").write_text(
        _TEST.format(workdir_slash=workdir.rstrip("/") + "/")
    )
    (task / "solution" / "solve.sh").write_text(solve)
    return task


def _build_services(root: Path) -> ProductionRuntimeServices:
    return ProductionRuntimeServices(
        data_root=root / "data",
        logs_root=root / "logs",
        docker_client=_docker(),
        firewall_backend=FakeFirewallBackend(),
        rsi_loop_config=RSILoopConfig(),
        omit_gpu_device_requests_for_tests=True,
        quiescence_checker=lambda _allocation, _container: None,
    )


def _services(root: Path, request: pytest.FixtureRequest) -> ProductionRuntimeServices:
    services = _build_services(root)
    _cleanup_at_end(services, root / "data", request)
    return services


def _cleanup_at_end(services, data: Path, request: pytest.FixtureRequest) -> None:
    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)


def _oracle_request(task: Path, max_submissions: int | None) -> RunRequest:
    return RunRequest(
        task_dir=task,
        agent_name="oracle",
        options=CompileOptions(agent_name="oracle", max_submissions=max_submissions),
    )


def _child_main(spec_path: str) -> None:
    """The Engine of attempt 1, in a process of its own that the test kills."""
    spec = json.loads(Path(spec_path).read_text())
    root = Path(spec["root"])
    if "agent" in spec:
        services = _build_driven_services(root, spec["base_url"], spec["extra_env"])
        services.run(_driven_request(Path(spec["task"]), spec["agent"]))
    else:
        services = _build_services(root)
        services.run(_oracle_request(Path(spec["task"]), spec["max_submissions"]))


def _start_child(root: Path, request, **spec: Any) -> subprocess.Popen:
    spec_path = root / "child.json"
    spec_path.write_text(json.dumps({"root": str(root), **spec}, default=str))
    log = (root / "child.log").open("w")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO), environment.get("PYTHONPATH", "")]
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.integration.test_resume_docker", str(spec_path)],
        cwd=REPO,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def stop() -> None:
        if process.poll() is None:
            process.kill()
            process.wait()
        log.close()

    request.addfinalizer(stop)
    return process


def _wait_for(
    what: str,
    found: Callable[[], Any],
    process: subprocess.Popen,
    root: Path,
    timeout: float = CHILD_WAIT_SECONDS,
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = found()
        if value:
            return value
        if process.poll() is not None:
            pytest.fail(
                f"the Engine exited ({process.returncode}) before {what}:\n"
                + (root / "child.log").read_text()[-6000:]
            )
        time.sleep(0.5)
    pytest.fail(f"timed out waiting for {what}")


def _run_root(root: Path) -> Path | None:
    found = sorted((root / "logs" / "runs").glob("*/*/agent_output.txt"))
    return found[0].parent if found else None


def _output_has(root: Path, text: str) -> Path | None:
    run_root = _run_root(root)
    if run_root is None:
        return None
    output = (run_root / "agent_output.txt").read_text(errors="replace")
    return run_root if text in output else None


def _kill(process: subprocess.Popen) -> None:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)
    assert process.returncode == -signal.SIGKILL


def _labeled(kind: str, run_id: str) -> list[Any]:
    client = _docker()
    filters = {"label": f"{RUN_LABEL}={run_id}"}
    if kind == "containers":
        return client.containers.list(all=True, filters=filters)
    if kind == "images":
        return client.images.list(all=True, filters=filters)
    if kind == "volumes":
        return client.volumes.list(filters=filters)
    return client.networks.list(filters=filters)


def _retained_ref(run_id: str, task_id: str, round_id: str) -> str:
    return rootfs_image_ref(
        run_id=run_id, task_id=task_id, round_id=round_id, purpose="retained-work"
    )


def _image_exists(ref: str) -> bool:
    try:
        _docker().images.get(ref)
    except Exception:
        return False
    return True


def _assert_cleanup_removes_everything(
    services, run_id: str, refs: tuple[str, ...]
) -> None:
    assert all(_image_exists(ref) for ref in refs), refs
    services.cleanup(run_id, delete_workspace=True)
    assert not [ref for ref in refs if _image_exists(ref)]
    for kind in ("containers", "images", "volumes", "networks"):
        assert _labeled(kind, run_id) == [], kind


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


@pytest.mark.parametrize("workdir", ["/workspace", "/"], ids=["split", "full-rootfs"])
def test_oracle_run_killed_mid_agent_resumes_with_its_files(
    tmp_path: Path, request: pytest.FixtureRequest, workdir: str
) -> None:
    task = _task(tmp_path, _SOLVE_KILLED_MID_AGENT, workdir=workdir)
    child = _start_child(tmp_path, request, task=task, max_submissions=3)
    services = _services(tmp_path, request)

    run_root = _wait_for(
        "attempt 1 sleeps",
        lambda: _output_has(tmp_path, "rsi-resume: attempt-1 sleeping"),
        child,
        tmp_path,
    )
    _kill(child)
    run_id, task_id = run_root.parent.name, run_root.name
    [work] = [
        c
        for c in _labeled("containers", run_id)
        if c.labels["rsi-harness.role"] == "work"
    ]
    assert work.status == "running"
    assert not (run_root / "final_result.json").exists()
    split = workdir != "/"
    if split:
        assert len(_labeled("volumes", run_id)) == 1

    dry = services.plan_resume(run_id, AttemptGrants(), dry_run=True)
    assert dry.salvage_pending and dry.attempt == 2
    context = services.plan_resume(run_id, AttemptGrants())

    # Salvage stopped and committed Work; attempt 1 is closed.
    assert not context.salvage_pending
    assert _labeled("containers", run_id) == []
    salvage_ref = _retained_ref(run_id, task_id, "salvage")
    assert context.base_image_ref == salvage_ref
    assert context.base_image_id == _docker().images.get(salvage_ref).id
    assert (context.workdir_volume is not None) is split
    lease = LeaseStore(tmp_path / "data" / "leases").read(run_id)
    assert lease is not None
    [first] = lease.attempts
    assert first.ended_at is not None and first.end_basis == "running"
    assert 0 < context.agent_seconds_used < AGENT_TIMEOUT_SECONDS
    assert context.agent_seconds_left == pytest.approx(
        AGENT_TIMEOUT_SECONDS - context.agent_seconds_used
    )
    assert (context.charged_rounds, context.next_round_id) == (1, "agent-2")
    assert (context.max_submissions, context.submissions_left) == (3, 2)
    assert "Submissions left: 2." in context.message

    result = services.resume(context)

    output = (run_root / "agent_output.txt").read_text()
    assert result.status is RunStatus.COMPLETED, output[-4000:]
    assert result.run_id == run_id
    rounds = [report.round_id for report in result.reports]
    assert rounds == ["agent-1", "agent-2", "agent-3"], rounds
    assert all(r.status is SubmissionStatus.COMPLETED for r in result.reports)
    assert result.best_score == 1
    # Files in WORKDIR and outside it survived; the solution was staged again.
    assert (
        "rsi-resume: workdir-file=kept-in-workdir home=kept-in-home answer=42 "
        "solution=solve.sh," in output
    )
    assert output.count("rsi-resume: attempt-1 sleeping") == 1
    assert "\n--- rsi-harness: attempt 2 ---\n" in output
    assert output.index("attempt-1 sleeping") < output.index("attempt 2 ---")
    assert "rsi-oracle: submission 2/2" in output
    final = _read(run_root / "final_result.json")
    assert final["attempts"] == 2
    assert final["total_rounds"] == 3
    assert final["agent_runtime_seconds"] > context.agent_seconds_used
    # The killed attempt wrote no result to move.
    assert not (run_root / "attempts").exists()
    state = _read(run_root / "evolve_state.json")
    assert [entry["round"] for entry in state["submissions"]] == rounds
    lease = LeaseStore(tmp_path / "data" / "leases").read(run_id)
    assert lease is not None
    assert [a.attempt for a in lease.attempts] == [1, 2]
    assert all(a.ended_at is not None for a in lease.attempts)

    _assert_cleanup_removes_everything(
        services,
        run_id,
        (salvage_ref, _retained_ref(run_id, task_id, "final-a2")),
    )


def test_finished_run_resumes_only_with_more_submissions(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness import cli

    services = _services(tmp_path, request)
    task = _task(tmp_path, _SOLVE_PLAIN)
    first = services.run(_oracle_request(task, 1))
    assert first.status is RunStatus.COMPLETED
    run_id = first.run_id
    run_root = first.artifacts_dir
    assert run_root is not None
    task_id = run_root.name

    with pytest.raises(SetupError, match="no submissions left"):
        services.plan_resume(run_id, AttemptGrants())

    def build(*, roots, event_callback=None, engine_config=None):
        del event_callback, engine_config
        assert roots.data == (tmp_path / "data").resolve()
        return _build_services(tmp_path)

    monkeypatch.setattr(cli, "build_runtime_services", build)
    roots = [
        "--data-root",
        str(tmp_path / "data"),
        "--logs-root",
        str(tmp_path / "logs"),
    ]
    refused = CliRunner().invoke(cli.app, ["resume", run_id, *roots])
    assert refused.exit_code != 0
    assert "no submissions left" in refused.output
    dry = CliRunner().invoke(
        cli.app, ["resume", run_id, "--add-submissions", "1", "--dry-run", *roots]
    )
    assert dry.exit_code == 0, dry.output
    assert "Next round:" in dry.output and "agent-2" in dry.output
    resumed = CliRunner().invoke(
        cli.app, ["resume", run_id, "--add-submissions", "1", *roots]
    )

    assert resumed.exit_code == 0, resumed.output
    assert f"Resuming run {run_id} as attempt 2" in resumed.output
    final = _read(run_root / "final_result.json")
    assert final["attempts"] == 2 and final["total_rounds"] == 2
    earlier = _read(run_root / "attempts" / "1" / "final_result.json")
    assert "attempts" not in earlier and earlier["total_rounds"] == 1
    output = (run_root / "agent_output.txt").read_text()
    assert output.count("rsi-resume: solved") == 2
    assert "\n--- rsi-harness: attempt 2 ---\n" in output
    state = _read(run_root / "evolve_state.json")
    assert [entry["round"] for entry in state["submissions"]] == ["agent-1", "agent-2"]
    # Nothing is left now, the grant spent.
    with pytest.raises(SetupError, match="no submissions left"):
        services.plan_resume(run_id, AttemptGrants())

    # A third attempt stacks a third retained image on the chain.
    third = services.resume(
        services.plan_resume(run_id, AttemptGrants(add_submissions=1))
    )
    assert third.status is RunStatus.COMPLETED
    assert [r.round_id for r in third.reports] == ["agent-1", "agent-2", "agent-3"]
    assert _read(run_root / "final_result.json")["attempts"] == 3
    assert (run_root / "attempts" / "2" / "final_result.json").exists()
    chain = tuple(
        _retained_ref(run_id, task_id, name)
        for name in ("final", "final-a2", "final-a3")
    )
    images = [_docker().images.get(ref) for ref in chain]
    # Each is committed from a Work started from the one before.
    assert [image.attrs["Parent"] for image in images[1:]] == [
        images[0].id,
        images[1].id,
    ]

    _assert_cleanup_removes_everything(services, run_id, chain)


def test_run_killed_mid_judge_charges_the_round_and_withholds_its_feedback(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    task = _task(tmp_path, _SOLVE_KILLED_MID_JUDGE)
    child = _start_child(tmp_path, request, task=task, max_submissions=None)
    services = _services(tmp_path, request)

    def judge_sleeping() -> Path | None:
        run_root = _run_root(tmp_path)
        if run_root is None:
            return None
        judges = [
            c
            for c in _labeled("containers", run_root.parent.name)
            if c.labels["rsi-harness.role"] == "judge" and c.status == "running"
        ]
        if judges and any(
            "sleep 900" in " ".join(process) for process in judges[0].top()["Processes"]
        ):
            return run_root
        return None

    run_root = _wait_for("the Judge's test.sh", judge_sleeping, child, tmp_path)
    # Its output so far is flushed.
    time.sleep(2)
    _kill(child)
    run_id = run_root.parent.name
    roles = sorted(c.labels["rsi-harness.role"] for c in _labeled("containers", run_id))
    assert roles == ["judge", "work"], roles
    assert not (run_root / "feedback" / "agent-1.log").exists()

    context = services.plan_resume(run_id, AttemptGrants())

    # The Judge is gone and its partial output moved out of feedback/.
    assert _labeled("containers", run_id) == []
    assert not (run_root / "feedback" / "agent-1.log").exists()
    hidden = run_root / "verifier" / "agent-1" / "interrupted-feedback.log"
    assert "judge-partial-output" in hidden.read_text()
    assert (context.charged_rounds, context.next_round_id) == (1, "agent-2")
    assert context.reports == ()
    assert "Round agent-1 was in flight" in context.message

    result = services.resume(context)

    output = (run_root / "agent_output.txt").read_text()
    assert result.status is RunStatus.COMPLETED, output[-4000:]
    assert [r.round_id for r in result.reports] == ["agent-2"]
    assert result.best_score == 1
    assert "rsi-resume: feedback=[]" in output
    assert "rsi-resume: no partial feedback" in output
    assert "judge-partial-output" not in output
    assert (
        "judge-partial-output"
        not in (run_root / "feedback" / "agent-2.log").read_text()
    )
    final = _read(run_root / "final_result.json")
    assert final["attempts"] == 2
    # Rounds with a report; the interrupted one has none.
    assert final["total_rounds"] == 1
    services.cleanup(run_id, delete_workspace=True)
    for kind in ("containers", "images", "volumes", "networks"):
        assert _labeled(kind, run_id) == [], kind


# Native session resume: attempt 1 runs one turn with a tool call whose
# command carries a string no fresh session has; attempt 2's first model
# request must carry it, with the resume message as its last user text.
SESSION_CASES = tuple(case for case in CASES if case.agent != "oracle")
WATCH = "rsi-resume-watch-6c1f"


def _fake_model(
    tmp_path: Path, request: pytest.FixtureRequest, case: DrivenCase, scenario
) -> tuple[Any, str, dict[str, str]]:
    """The fake model server, its base URL for Work and the Agent's extra env."""
    client = _driven_docker()
    gateway = next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )
    module = _load_fake_model_server()
    server = module.FakeModelServer(
        tmp_path / "model-requests.jsonl",
        {**scenario, **case.scenario},
        host="0.0.0.0",
    )
    scheme = "http"
    extra_env = {"RSI_AGENT_DRIVER_TIMINGS": json.dumps(DRIVER_TIMINGS)}
    if case.tls:
        scheme = "https"
        server.httpd.socket = _tls_context(tmp_path).wrap_socket(
            server.httpd.socket, server_side=True, do_handshake_on_connect=False
        )
        extra_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    threading.Thread(target=server.httpd.serve_forever, daemon=True).start()
    request.addfinalizer(server.stop)
    port = server.httpd.server_address[1]
    return server, f"{scheme}://{gateway}:{port}{case.base_path}", extra_env


def _build_driven_services(
    root: Path, base_url: str, extra_env: dict[str, str]
) -> ProductionRuntimeServices:
    return ProductionRuntimeServices(
        data_root=root / "data",
        logs_root=root / "logs",
        docker_client=_driven_docker(),
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=API_KEY,
            agent_api_base_url=base_url,
            agent_context_window="200000",
            agent_max_output_tokens=str(MAX_OUTPUT_TOKENS),
            agent_extra_env=extra_env,
        ),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: RSILoopAgentAdapter(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
    )


def _driven_request(task: Path, agent: str) -> RunRequest:
    # No stop hook: the CLI ends after one turn, in either attempt.
    return RunRequest(
        task_dir=task,
        agent_name=agent,
        model=MODEL,
        gpu_selectors=("0",),
        options=CompileOptions(
            agent_name=agent, primary_reward="reward", disable_stop_hook=True
        ),
    )


def _session_scenario(**extra: Any) -> dict[str, Any]:
    return {
        # A tool call, then a text turn end: one turn, and the CLI exits.
        # Attempt 2 does the same.
        "cycle": ["mixed", "text"],
        "commands": [
            f"echo {WATCH} > /workspace/attempt-1.txt; cat /workspace/attempt-1.txt",
            'echo "attempt-2-sees:[$(cat /workspace/attempt-1.txt)]"',
        ],
        "watch": [WATCH],
        **extra,
    }


def _assert_session_continued(
    server, attempt_1: list[dict[str, Any]], context, run_root: Path
) -> None:
    output = (run_root / "agent_output.txt").read_text()
    attempt_2 = server.requests()[len(attempt_1) :]
    assert attempt_2, output[-4000:]
    opening = attempt_2[0]
    # The continued session: attempt 1's history, then the resume message.
    assert opening["watched"] == [WATCH], (opening, output[-4000:])
    assert opening["n_messages"] > attempt_1[-1]["n_messages"], (attempt_1, opening)
    assert opening["last_role"] == "user", opening
    assert (opening["last_user"] or "").strip() == context.message.strip(), opening
    # The continued session ran its tools in the kept Work.
    assert any(
        f"attempt-2-sees:[{WATCH}]" in (r["last_tool"] or "") for r in attempt_2
    ), attempt_2
    assert "\n--- rsi-harness: attempt 2 ---\n" in output
    assert _read(run_root / "final_result.json")["attempts"] == 2


@pytest.mark.parametrize("case", SESSION_CASES, ids=str)
def test_agent_continues_its_own_session_on_resume(
    tmp_path: Path, request: pytest.FixtureRequest, case: DrivenCase
) -> None:
    server, base_url, extra_env = _fake_model(
        tmp_path, request, case, _session_scenario()
    )
    services = _build_driven_services(tmp_path, base_url, extra_env)
    _cleanup_at_end(services, tmp_path / "data", request)
    task = _driven_task(tmp_path, timeout_seconds=AGENT_TIMEOUT_SECONDS)
    first = services.run(_driven_request(task, case.agent))
    run_root = tmp_path / "logs" / "runs" / first.run_id / "minimal-gpu"
    attempt_1 = server.requests()
    assert len(attempt_1) >= 2, (first, run_root)
    assert attempt_1[0]["watched"] == []
    assert any(WATCH in (r["last_tool"] or "") for r in attempt_1), attempt_1

    context = services.plan_resume(first.run_id, AttemptGrants())
    assert not context.salvage_pending and context.attempt == 2
    services.resume(context)

    _assert_session_continued(server, attempt_1, context, run_root)
    assert (run_root / "attempts" / "1" / "final_result.json").exists()


@pytest.mark.parametrize("case", SESSION_CASES, ids=str)
def test_agent_continues_its_session_after_a_crash_mid_turn(
    tmp_path: Path, request: pytest.FixtureRequest, case: DrivenCase
) -> None:
    """The Engine is killed while the CLI waits for the model, mid-turn."""
    server, base_url, extra_env = _fake_model(
        tmp_path, request, case, _session_scenario(at={"2": "hang"})
    )
    services = _build_driven_services(tmp_path, base_url, extra_env)
    _cleanup_at_end(services, tmp_path / "data", request)
    task = _driven_task(tmp_path, timeout_seconds=AGENT_TIMEOUT_SECONDS)
    child = _start_child(
        tmp_path,
        request,
        task=task,
        agent=case.agent,
        base_url=base_url,
        extra_env=extra_env,
    )

    attempt_1 = _wait_for(
        "the model request after the tool call",
        lambda: [r for r in server.requests() if r["n"] == 2] and server.requests(),
        child,
        tmp_path,
    )
    _kill(child)
    run_root = _run_root(tmp_path)
    assert run_root is not None
    assert attempt_1[-1]["watched"] == [WATCH]

    context = services.plan_resume(run_root.parent.name, AttemptGrants())
    assert context.attempt == 2
    services.resume(context)

    _assert_session_continued(server, attempt_1, context, run_root)
    assert not (run_root / "attempts").exists()


if __name__ == "__main__":
    _child_main(sys.argv[1])
