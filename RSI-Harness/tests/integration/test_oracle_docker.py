"""Real Docker: the oracle Agent runs solve.sh and the production Judge path.

Run as a non-root user with the fake firewall; the real RSI Loop adapter and
the oracle's real Work image, launcher and solution staging are used.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rsi_harness.config import EngineConfig
from rsi_harness.models import (
    CompileOptions,
    RunRequest,
    RunStatus,
    SubmissionStatus,
)
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_contracts import SandboxPolicy
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.sandbox_support import (
    assert_no_sandbox_resources,
    require_sandbox_authority,
)
from tests.integration.test_public_tests_docker import _docker

pytestmark = pytest.mark.integration

SANDBOX_FIXTURE = Path(__file__).parents[1] / "fixtures/tasks/minimal-sandbox"

_TASK_TOML = """\
schema_version = "1.4"

[task]
name = "rsi/oracle-check"

[environment]
os = "linux"
workdir = "{workdir}"
gpus = 0
network_mode = "no-network"

[agent]
timeout_sec = 300
network_mode = "no-network"
{agent_user}
[verifier]
timeout_sec = 60
network_mode = "no-network"
"""

_SOLVE = """\
#!/bin/bash
set -eu
# A file next to solve.sh, staged with it.
cat "$(dirname "$0")/data/answer" > answer.txt
echo "solved as $(id -u) in $(pwd)"
exit {status}
"""

_TEST = """\
#!/bin/bash
set -eu
if [ -e /solution ]; then echo "the solution reached the Judge"; exit 5; fi
answer=$(cat {answer})
if [ "$answer" = 42 ]; then reward=1; else reward=0; fi
echo "answer=$answer"
printf '{{"reward": %s}}\\n' "$reward" > /logs/verifier/reward.json
"""


def _task(
    root: Path, *, workdir: str = "/workspace", status: int = 0, answer: str = "42"
) -> Path:
    task = root / "oracle-check"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution" / "data").mkdir(parents=True)
    non_root = workdir != "/"
    (task / "task.toml").write_text(
        _TASK_TOML.format(
            workdir=workdir,
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
    answer_path = "/answer.txt" if workdir == "/" else f"{workdir}/answer.txt"
    (task / "tests" / "test.sh").write_text(_TEST.format(answer=answer_path))
    solve = task / "solution" / "solve.sh"
    solve.write_text(_SOLVE.format(status=status))
    # Not executable on the host: staging makes it so.
    solve.chmod(0o644)
    (task / "solution" / "data" / "answer").write_text(answer)
    return task


def _services(root: Path, request: pytest.FixtureRequest, **options):
    client = _docker()
    data = root / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=root / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        rsi_loop_config=RSILoopConfig(
            # Never reaches Work: the oracle gets no credential.
            agent_api_key="sk-oracle-must-not-see",
        ),
        omit_gpu_device_requests_for_tests=True,
        quiescence_checker=lambda _allocation, _container: None,
        **options,
    )

    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)
    return services


def _oracle_run(services, task: Path, max_submissions: int | None = None):
    return services.run(
        RunRequest(
            task_dir=task,
            agent_name="oracle",
            options=CompileOptions(
                agent_name="oracle", max_submissions=max_submissions
            ),
        )
    )


def _artifacts(result) -> tuple[Path, dict, str, str]:
    root = result.artifacts_dir
    assert root is not None
    final = json.loads((root / "final_result.json").read_text())
    output = (root / "agent_output.txt").read_text()
    feedback = "".join(
        path.read_text() for path in sorted((root / "feedback").glob("*.log"))
    )
    return root, final, output, feedback


@pytest.mark.parametrize(
    ("workdir", "clean_image"), [("/workspace", True), ("/", False)]
)
def test_oracle_solution_scores_on_the_real_hidden_judge(
    tmp_path: Path, request: pytest.FixtureRequest, workdir: str, clean_image: bool
) -> None:
    services = _services(tmp_path, request)

    result = _oracle_run(services, _task(tmp_path, workdir=workdir))

    root, final, output, feedback = _artifacts(result)
    assert result.status is RunStatus.COMPLETED, feedback + output
    assert result.best_score == 1, feedback
    assert result.agent_exit_code == 0
    assert [report.status for report in result.reports] == [SubmissionStatus.COMPLETED]
    assert final["agent"] == "oracle"
    assert final["agent_exit_code"] == 0
    assert final["first_submission_seconds"] > 0
    assert final["agent_runtime_seconds"] >= final["first_submission_seconds"]
    plan = json.loads((root / "run-plan.json").read_text())
    assert plan["task"]["verifier"]["clean_image"] is clean_image
    report = json.loads((root / "submissions/agent-1/report.json").read_text())
    assert report["status"] == "completed" and report["exit_code"] == 0
    assert report["runtime_seconds"] > 0
    assert "answer=42" in feedback
    assert "rsi-oracle: solve.sh exited 0" in output
    # Whatever exec user every Agent gets; not pinned to one uid.
    assert re.search(rf"solved as \d+ in {re.escape(workdir)}\n", output)
    assert "sk-oracle-must-not-see" not in output


def test_oracle_submits_n_times(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    services = _services(tmp_path, request)

    result = _oracle_run(services, _task(tmp_path), max_submissions=2)

    _root, final, output, feedback = _artifacts(result)
    assert result.status is RunStatus.COMPLETED, feedback + output
    assert result.total_rounds == 2
    assert all(r.status is SubmissionStatus.COMPLETED for r in result.reports)
    assert final["total_rounds"] == 2
    assert final["first_submission_seconds"] > 0
    assert "rsi-oracle: submission 2/2" in output


def test_a_failing_solve_still_reaches_the_judge(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    services = _services(tmp_path, request)

    result = _oracle_run(services, _task(tmp_path, status=3, answer="41"))

    _root, final, output, feedback = _artifacts(result)
    assert result.agent_exit_code == 3
    assert final["agent_exit_code"] == 3
    assert result.total_rounds == 1
    assert result.reports[0].status is SubmissionStatus.COMPLETED
    assert result.best_score == 0
    assert "answer=41" in feedback
    assert "rsi-oracle: solve.sh exited 3" in output


def test_cli_oracle_exit_code_follows_the_real_run(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness import cli

    def build(*, roots, event_callback=None, engine_config=None):
        del event_callback, engine_config
        return _services(roots.data.parent, request)

    monkeypatch.setattr(cli, "build_runtime_services", build)
    passing = _task(tmp_path / "pass")
    failing = _task(tmp_path / "fail", answer="41")

    def invoke(task: Path, root: Path):
        return CliRunner().invoke(
            cli.app,
            [
                "run",
                str(task),
                "--agent",
                "oracle",
                "--data-root",
                str(root / "data"),
                "--logs-root",
                str(root / "logs"),
            ],
        )

    ok = invoke(passing, tmp_path / "pass")
    # A completed round scores 0 here; the Judge itself worked.
    scored_zero = invoke(failing, tmp_path / "fail")

    assert ok.exit_code == 0, ok.output
    assert "Oracle check passed" in ok.output
    assert "Artifacts:" in ok.output
    assert "exit_code=0 runtime=" in ok.output
    assert scored_zero.exit_code == 0, scored_zero.output
    assert "score=0.0" in scored_zero.output


def test_oracle_drives_a_managed_sandbox_through_work_and_judge(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """minimal-sandbox with solve.sh: the Work broker for solve.sh, a Judge
    broker for test.sh, as the operator policy grants."""
    client, image = require_sandbox_authority()
    client.close()
    # The broker's socket path must fit in sockaddr_un.
    tmp_path = Path(tempfile.mkdtemp(prefix="rsi-o-"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    task = tmp_path / "task"
    shutil.copytree(SANDBOX_FIXTURE, task)
    config = task / "task.toml"
    config.write_text(config.read_text().replace("__APPROVED_IMAGE_ID__", image.id))
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        # The image's own PATH entry, as for the Agent exec.
        "task-python --version\n"
        "child=$(rsi-sandbox create offline --lifetime 100 --json |\n"
        "    python3 -c 'import json,sys; print(json.load(sys.stdin)[\"child_id\"])')\n"
        "rsi-sandbox exec \"$child\" --json -- python3 -c 'print(6 * 7)' "
        "> /tmp/r.json\n"
        'python3 -c \'import json; print(json.load(open("/tmp/r.json"))'
        '["stdout"].strip())\' > answer.txt\n'
        'rsi-sandbox destroy "$child" --json\n'
        "echo sandbox-solve-done\n"
    )
    declaration = HarborTaskCompiler().compile(task, CompileOptions()).sandbox
    policy = SandboxPolicy(
        profiles=declaration.profiles,
        work=declaration.work,
        judge=declaration.judge,
        run_limits=declaration.judge.limits,
        pool_cpus=4,
        pool_memory_mb=4096,
    )
    services = _services(
        tmp_path,
        request,
        engine_config=EngineConfig(
            data_root=tmp_path / "data",
            logs_root=tmp_path / "logs",
            sandbox_policy=policy,
        ),
    )

    result = _oracle_run(services, task)

    store = LeaseStore(tmp_path / "data" / "leases")
    lease = store.read(result.run_id)
    assert result.artifacts_dir is not None, lease.error
    _root, final, output, feedback = _artifacts(result)
    assert result.status is RunStatus.COMPLETED, feedback + output
    assert result.best_score == 1.0, feedback
    assert final["agent_exit_code"] == 0
    assert "sandbox-solve-done" in output
    lease = store.read(result.run_id)
    phases = sorted(child.owner.phase for child in lease.sandboxes)
    assert phases == ["judge", "work"]
    assert all(child.state == "removed" for child in lease.sandboxes)
    services.cleanup(result.run_id, delete_workspace=True)
    assert_no_sandbox_resources(_docker(), store, result.run_id)
