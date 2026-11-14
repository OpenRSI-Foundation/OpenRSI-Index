"""Real CPU Work/Judge lifecycle, without model credentials or task downloads.

Opt in with RSI_ACCEPTANCE=1, RSI_APPTAINER_TEST_PROFILE (site TOML), and
RSI_APPTAINER_TEST_IMAGE (SIF containing bash and the profile's Python).
Run inside the site's CPU allocation. This tests the native evaluator and
runtime, not image building, the model CLI, or the Engine resume command.
The Judge reads Work's paused workspace through a read-only bind; no copy.
"""

import errno
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela.runtime import (
    ApptainerAgentRuntime,
    NativeJudgeEvaluator,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import InfrastructureError, RetryableSubmissionError
from rsi_harness.models import EvaluationRequest, GPURequirement, SubmissionStatus
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.coordinator import _cancellation_signals, _RunCancelled
from tests.factories import make_run_plan

pytestmark = pytest.mark.acceptance


@pytest.fixture
def native(tmp_path, monkeypatch):
    profile_path = os.environ.get("RSI_APPTAINER_TEST_PROFILE")
    image = os.environ.get("RSI_APPTAINER_TEST_IMAGE")
    if not profile_path or not image:
        pytest.skip("set RSI_APPTAINER_TEST_PROFILE and RSI_APPTAINER_TEST_IMAGE")
    profile = load_cluster_profile(profile_path)
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "gpu_requirement": GPURequirement(count=0),
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "timeout_seconds": 20,
                        }
                    ),
                }
            ),
        }
    )
    writer = RunArtifactWriter(plan, run_id="snapshot-acceptance")
    writer.start()
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=profile, run_id="snapshot-acceptance", sif_path=Path(image)
        ),
        plan,
    )
    for directory in (
        runtime.workspace,
        runtime.control,
        runtime.agent_home,
        runtime.task_assets,
        runtime.work_tmp,
        runtime.judge_tmp_root,
        runtime.feedback,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    tests = plan.task.source_dir / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text(
        "#!/bin/bash\nset -eu\n"
        "echo judge-started\n"
        # The Judge reads Work's workspace, read-only.
        "cat /workspace/candidate.txt\n"
        "if mkdir /workspace/readonly 2>/dev/null; then\n"
        "  echo judge-wrote-workspace\n"
        "else echo judge-workspace-read-only; fi\n"
        "echo judge > /workspace/candidate.txt 2>/dev/null || true\n"
        "case $(cat /workspace/mode) in\n"
        "  error) exit 3;;\n"
        "  timeout|interrupt) sleep 120;;\n"
        "  *) printf '0.75\\n' > /logs/verifier/reward.txt;;\n"
        "esac\n"
    )
    (runtime.workspace / "candidate.txt").write_text("candidate-source\n")
    runtime._require_site_isolation()
    ready = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(
            runtime._run,
            (
                str(profile.apptainer.container_python),
                "-u",
                "-c",
                "import time; print('work-ready', flush=True); time.sleep(300)",
            ),
            devices=(),
            network_mode="no-network",
            box="work",
            timeout_seconds=300,
            track_work=True,
            output_path=writer.root / "agent_output.txt",
            output_callback=lambda line: ready.set() if "work-ready" in line else None,
        )
        try:
            if not ready.wait(45):
                if work.done():
                    pytest.fail(f"Work failed: {work.result()}")
                pytest.fail("Work did not become ready")
            yield runtime, writer, work
        finally:
            if not work.done():
                runtime.stop(runtime.work_ref)
            try:
                work.result(timeout=45)
            except InfrastructureError:
                if runtime._work_failure is None:
                    raise


class Observer:
    def __init__(self, runtime, path):
        self.runtime, self.path, self.events = runtime, path, []

    def resource_event(self, name, **values):
        if name == "judge_created":
            assert self.runtime._paused
        self.events.append(name)
        with self.path.open("a") as stream:
            stream.write(json.dumps({"event": name, **values}) + "\n")


def evaluate(runtime, writer, name, observer, evaluator=None):
    request = EvaluationRequest(
        run_plan=runtime.plan,
        work_container=runtime.work_ref,
        round_id=name,
        verifier_logs=writer.root / "verifier" / name,
        verifier_output=writer.feedback_root / f"{name}.log",
    )
    evaluator = evaluator or NativeJudgeEvaluator(runtime, writer)
    return evaluator.evaluate(request, observer)


def test_real_apptainer_judge_reads_the_paused_workspace_read_only(native):
    runtime, writer, work = native
    before = sorted(path.name for path in runtime.plan.paths.root.iterdir())
    evaluator = NativeJudgeEvaluator(runtime, writer)
    # A FIFO in the workspace: refused, not counted, and Work resumes.
    os.mkfifo(runtime.workspace / "pipe")
    (runtime.workspace / "mode").write_text("success")
    observer = Observer(runtime, writer.root / "acceptance-events.jsonl")
    with pytest.raises(RetryableSubmissionError, match="pipe"):
        evaluate(runtime, writer, "agent-1", observer, evaluator)
    assert "judge_created" not in observer.events
    assert not runtime._paused
    (runtime.workspace / "pipe").unlink()
    for index, mode in enumerate(("success", "success", "error", "timeout")):
        name = f"agent-{index + 1}"
        (runtime.workspace / "mode").write_text(mode)
        observer = Observer(runtime, writer.root / "acceptance-events.jsonl")
        report = evaluate(runtime, writer, name, observer, evaluator)
        expected = {
            "success": SubmissionStatus.COMPLETED,
            "error": SubmissionStatus.VERIFIER_ERROR,
            "timeout": SubmissionStatus.VERIFIER_TIMEOUT,
        }[mode]
        assert report.status == expected
        if mode == "success":
            assert report.score == 0.75
        assert (writer.root / "submissions" / name / "report.json").is_file()
        feedback = (writer.feedback_root / f"{name}.log").read_text()
        assert "judge-started" in feedback
        assert "candidate-source" in feedback
        assert "judge-workspace-read-only" in feedback
        assert not any(event.startswith("snapshot_") for event in observer.events)
        assert observer.events.index("judge_removed") < observer.events.index(
            "work_unpaused"
        )
        assert not work.done()
    # No copy anywhere under the run directory.
    assert sorted(path.name for path in runtime.plan.paths.root.iterdir()) == before
    assert "work-ready" in (writer.root / "agent_output.txt").read_text()
    assert (runtime.workspace / "candidate.txt").read_text() == "candidate-source\n"
    assert not (runtime.workspace / "readonly").exists()


def test_real_apptainer_interruption_keeps_evidence_without_resuming_work(native):
    runtime, writer, work = native
    (runtime.workspace / "mode").write_text("interrupt")
    observer = Observer(runtime, writer.root / "acceptance-events.jsonl")
    output = writer.feedback_root / "agent-1.log"
    cancelled = threading.Event()

    def interrupt_when_judge_starts():
        deadline = time.monotonic() + 60
        while not cancelled.wait(0.05) and time.monotonic() < deadline:
            if output.exists() and "judge-started" in output.read_text():
                os.kill(os.getpid(), signal.SIGTERM)
                return

    thread = threading.Thread(target=interrupt_when_judge_starts)
    with _cancellation_signals():
        thread.start()
        try:
            with pytest.raises(_RunCancelled, match="received signal"):
                evaluate(runtime, writer, "agent-1", observer)
        finally:
            cancelled.set()
            thread.join()
    with pytest.raises(InfrastructureError, match="recovery_required"):
        work.result(timeout=20)
    assert runtime._process.poll() is not None
    assert not (runtime.plan.paths.root / "judge-rounds").exists()
    assert (runtime.workspace / "candidate.txt").read_text() == "candidate-source\n"
    assert "recovery_required" in observer.events
    assert "judge_removed" not in observer.events
    assert "work_unpaused" not in observer.events
    assert not (writer.root / "submissions/agent-1/report.json").exists()
    assert "judge-started" in output.read_text()


def test_real_apptainer_output_failure_requires_recovery(native, monkeypatch):
    runtime, writer, work = native
    (runtime.workspace / "mode").write_text("interrupt")
    output = writer.feedback_root / "agent-1.log"
    original = Path.open

    class FullSink:
        def __init__(self, stream):
            self.stream = stream

        def write(self, _data):
            raise OSError(errno.EDQUOT, "injected Judge output quota")

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def open_sink(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return FullSink(stream) if path == output and mode == "a" else stream

    monkeypatch.setattr(Path, "open", open_sink)
    observer = Observer(runtime, writer.root / "acceptance-events.jsonl")
    started = time.monotonic()
    with pytest.raises(InfrastructureError, match="output capture failed"):
        evaluate(runtime, writer, "agent-1", observer)
    assert time.monotonic() - started < runtime.plan.task.verifier.timeout_seconds
    with pytest.raises(InfrastructureError, match="recovery_required"):
        work.result(timeout=20)
    assert runtime._process.poll() is not None
    assert not (runtime.plan.paths.root / "judge-rounds").exists()
    assert (runtime.workspace / "candidate.txt").read_text() == "candidate-source\n"
    assert "recovery_required" in observer.events
    assert "judge_removed" not in observer.events
    assert "work_unpaused" not in observer.events
    assert not (writer.root / "submissions/agent-1/report.json").exists()
