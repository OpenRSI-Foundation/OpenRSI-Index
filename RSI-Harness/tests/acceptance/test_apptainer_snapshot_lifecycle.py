"""Real CPU Work/Judge lifecycle, without model credentials or task downloads.

Opt in with RSI_ACCEPTANCE=1, RSI_APPTAINER_TEST_PROFILE (site TOML), and
RSI_APPTAINER_TEST_IMAGE (SIF containing bash and the profile's Python).
Run inside the site's CPU allocation. This tests the native evaluator and
runtime, not image building, the model CLI, or the Engine resume command.
The Judge reads a read-only copy; a later round links unchanged files.
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
from rsi_harness.cluster.snapshots import RETAINED, can_link
from rsi_harness.errors import InfrastructureError
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
        runtime.rounds,
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
        # The Judge's copy is read-only.
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
        if name in ("snapshot_acquired", "judge_created", "snapshot_released"):
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


def test_real_apptainer_keeps_one_copy_and_links_unchanged_files(native):
    runtime, writer, work = native
    # Existing interrupted evidence is not swept by a later round.
    old = runtime.rounds / "previous-attempt" / "workspace"
    old.mkdir(parents=True)
    (old / "keep").write_text("recovery evidence")
    evaluator = NativeJudgeEvaluator(runtime, writer)
    # Only a verified full stop on one node, off tmpfs, links unchanged files.
    reuse = runtime.judge_copy_reuse and can_link(runtime.workspace, runtime.rounds)
    copied = None
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
        assert not (runtime.rounds / name).exists()
        assert sorted(path.name for path in runtime.rounds.iterdir()) == (
            [RETAINED, old.parent.name] if reuse else [old.parent.name]
        )
        assert (old / "keep").read_text() == "recovery evidence"
        assert (writer.root / "submissions" / name / "report.json").is_file()
        feedback = (writer.feedback_root / f"{name}.log").read_text()
        assert "judge-started" in feedback
        assert "judge-workspace-read-only" in feedback
        assert observer.events.index("snapshot_released") < observer.events.index(
            "work_unpaused"
        )
        if reuse:
            candidate = (runtime.rounds / RETAINED / "candidate.txt").stat()
            work_candidate = (runtime.workspace / "candidate.txt").stat()
            assert candidate.st_ino != work_candidate.st_ino
            if copied is not None:
                # Unchanged in Work: linked from the previous round's copy.
                assert candidate.st_ino == copied
            copied = candidate.st_ino
        assert not work.done()
    evaluator.close()
    assert not (runtime.rounds / RETAINED).exists()
    assert "work-ready" in (writer.root / "agent_output.txt").read_text()
    assert (runtime.workspace / "candidate.txt").read_text() == "candidate-source\n"
    assert not (runtime.workspace / "readonly").exists()


def test_real_apptainer_interruption_retains_uncommitted_snapshot(native):
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
    assert (runtime.rounds / "agent-1/workspace/candidate.txt").is_file()
    assert (runtime.workspace / "candidate.txt").read_text() == "candidate-source\n"
    assert "recovery_required" in observer.events
    assert "snapshot_released" not in observer.events
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
    assert (runtime.rounds / "agent-1/workspace/candidate.txt").is_file()
    assert (runtime.workspace / "candidate.txt").read_text() == "candidate-source\n"
    assert "recovery_required" in observer.events
    assert "snapshot_released" not in observer.events
    assert "work_unpaused" not in observer.events
    assert not (writer.root / "submissions/agent-1/report.json").exists()
