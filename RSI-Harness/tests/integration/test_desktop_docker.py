"""Opt-in real Mac controller, Docker Desktop, and production firewall acceptance."""

from __future__ import annotations

import os
import sys

import pytest

from rsi_harness.models import CompileOptions, RunRequest, RunStatus
from rsi_harness.runtime.production import ProductionRuntimeServices
from tests.integration.test_oracle_docker import _task

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform != "darwin" or os.environ.get("RSI_TEST_MAC_DESKTOP") != "1",
        reason="requires explicit real Docker Desktop firewall acceptance",
    ),
]


def test_native_desktop_recovers_partial_export_with_max_length_names(
    tmp_path, monkeypatch
):
    """Real private volume and recovery; no image build or model invocation."""
    import errno
    from types import SimpleNamespace

    import docker
    from docker.types import Mount

    from rsi_harness.errors import RecoveryRequiredError
    from rsi_harness.runtime import desktop
    from rsi_harness.runtime.production import ProductionRecoveryBackend
    from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager

    client = docker.from_env()
    try:
        image = client.images.get("ubuntu:24.04")
    except docker.errors.ImageNotFound:
        pytest.skip("requires a cached ubuntu:24.04 image; this test never pulls")
    desktop.validate_local_daemon(client)
    root = tmp_path / "runs" / "export-recovery" / "private-logs"
    target = root / "verifier" / "agent-1"
    names = ("a" * 255, "b" * 255)
    container = client.containers.create(
        image.id,
        ["sleep", "300"],
        network_mode="none",
        read_only=True,
        cap_drop=["ALL"],
        mem_limit="64m",
        nano_cpus=500_000_000,
        mounts=[Mount(desktop.LOG_TARGET, source=None, type="volume", no_copy=True)],
        labels={
            desktop.PRIVATE_LOG_LABEL: "volume",
            "rsi-harness.role": "judge",
            "rsi-harness.round-id": "agent-1",
        },
    )
    try:
        container.start()
        container.reload()
        volume = next(
            mount["Name"]
            for mount in container.attrs["Mounts"]
            if mount["Destination"] == desktop.LOG_TARGET
        )

        def write(contents):
            for name in names:
                result = container.exec_run(
                    [
                        "/bin/sh",
                        "-c",
                        'printf %s "$2" > "$1"',
                        "write-proof",
                        f"{desktop.LOG_TARGET}/{name}",
                        contents,
                    ]
                )
                assert result.exit_code == 0, result.output

        write("previous")
        desktop.export_private_logs(client, container.id, target)
        (target / "controller.txt").write_text("keep")
        write("current")
        replace = desktop.os.replace
        publications = 0

        def fail_second_publication(source, destination):
            nonlocal publications
            if source.parent.name == "contents":
                publications += 1
                if publications == 2:
                    raise OSError(errno.ENOSPC, "injected publication failure")
            return replace(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(desktop.os, "replace", fail_second_publication)
            with pytest.raises(RecoveryRequiredError):
                desktop.export_private_logs(client, container.id, target)
        assert sorted((target / name).read_text() for name in names) == [
            "current",
            "previous",
        ]
        backend = ProductionRecoveryBackend(client, None, None)
        manager = RecoveryManager(
            store=LeaseStore(tmp_path / "leases"),
            backend=backend,
            managed_root=tmp_path / "data",
        )
        lease = SimpleNamespace(
            artifact_root=root, run_id="export-recovery", task_id="private-logs"
        )
        manager._stop_and_remove_judge(
            lease, container.id, backend.inspect_container(container.id)
        )
        assert all((target / name).read_text() == "current" for name in names)
        assert (target / "controller.txt").read_text() == "keep"
        assert not list(target.parent.glob(".log-export-*"))
        with pytest.raises(docker.errors.NotFound):
            client.containers.get(container.id)
        with pytest.raises(docker.errors.NotFound):
            client.volumes.get(volume)
    finally:
        try:
            container.remove(force=True, v=True)
        except docker.errors.NotFound:
            pass


@pytest.mark.parametrize("public", [False, True])
def test_native_desktop_two_submissions_and_private_logs(tmp_path, public):
    task = _task(tmp_path)
    if public:
        config = task / "task.toml"
        config.write_text(
            config.read_text()
            + '\n[metadata.rsi_harness.verifier]\nvisibility = "public"\n'
        )
    checker = task / "tests/test.sh"
    probe = """\
mkdir -p /logs/verifier/private
chmod 700 /logs/verifier/private
printf secret > /logs/verifier/private/secret
chmod 600 /logs/verifier/private/secret
for uid in 1000 61000; do
  test "$(setpriv --reuid=$uid --regid=$uid --clear-groups id -u)" = "$uid"
  if setpriv --reuid=$uid --regid=$uid --clear-groups cat /logs/verifier/private/secret; then exit 90; fi
  if setpriv --reuid=$uid --regid=$uid --clear-groups sh -c 'echo forged > /logs/verifier/private/forged'; then exit 91; fi
done
"""  # noqa: E501
    checker.write_text(checker.read_text().replace("set -eu\n", "set -eu\n" + probe))
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data", logs_root=tmp_path / "logs"
    )
    result = services.run(
        RunRequest(
            task_dir=task,
            agent_name="oracle",
            options=CompileOptions(agent_name="oracle", max_submissions=2),
        )
    )
    assert result.status == RunStatus.COMPLETED
    rewards = list((tmp_path / "logs").glob("runs/*/*/verifier/*/reward.json"))
    assert len(rewards) == 2
    import json

    assert all(json.loads(path.read_text())["reward"] == 1 for path in rewards)
    assert not list((tmp_path / "logs").rglob("forged"))


def test_native_desktop_recovers_private_evidence_after_controller_kill(tmp_path):
    """Kill the actual CLI during Judge; recover evidence before volume removal."""
    import json
    import signal
    import subprocess
    import time

    import docker

    from rsi_harness.runtime.recovery import AttemptGrants
    from tests.integration.test_resume_docker import (
        _SOLVE_KILLED_MID_JUDGE,
    )
    from tests.integration.test_resume_docker import (
        _task as resume_task,
    )

    task = resume_task(tmp_path, _SOLVE_KILLED_MID_JUDGE)
    checker = task / "tests/test.sh"
    checker.write_text(
        checker.read_text().replace(
            "    sleep 900",
            "    echo durable-private-evidence > /logs/verifier/partial.txt\n"
            "    sleep 900",
        )
    )
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data", logs_root=tmp_path / "logs"
    )
    client = docker.from_env()
    run_id = None
    with (tmp_path / "child.log").open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rsi_harness.cli",
                "run",
                str(task),
                "--agent",
                "oracle",
                "--data-root",
                str(tmp_path / "data"),
                "--logs-root",
                str(tmp_path / "logs"),
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            judge = None
            while time.monotonic() < deadline:
                leases = list((tmp_path / "data/leases").glob("*.json"))
                if leases:
                    run_id = leases[0].stem
                    judges = client.containers.list(
                        filters={
                            "label": [
                                f"rsi-harness.run-id={run_id}",
                                "rsi-harness.role=judge",
                            ]
                        }
                    )
                    if judges:
                        judge = judges[0]
                        if (
                            judge.exec_run(
                                ["test", "-f", "/logs/verifier/partial.txt"]
                            ).exit_code
                            == 0
                        ):
                            break
                assert process.poll() is None, (tmp_path / "child.log").read_text()[
                    -4000:
                ]
                time.sleep(0.5)
            else:
                pytest.fail("Judge did not reach crash checkpoint")
            assert judge is not None
            judge.reload()
            volume = next(
                m["Name"]
                for m in judge.attrs["Mounts"]
                if m["Destination"] == "/logs/verifier"
            )
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            context = services.plan_resume(run_id, AttemptGrants())
            run_root = tmp_path / "logs/runs" / run_id / "resume-check"
            assert (
                run_root / "verifier/agent-1/partial.txt"
            ).read_text().strip() == "durable-private-evidence"
            assert not (run_root / "feedback/agent-1.log").exists()
            assert context.charged_rounds == 1
            with pytest.raises(docker.errors.NotFound):
                client.volumes.get(volume)
            result = services.resume(context)
            assert result.status == RunStatus.COMPLETED
            assert result.best_score == 1
            assert [r.round_id for r in result.reports] == ["agent-2"]
            assert (
                json.loads((run_root / "final_result.json").read_text())["attempts"]
                == 2
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            if run_id:
                services.cleanup(run_id, delete_workspace=True)
    assert not client.containers.list(
        all=True, filters={"label": f"rsi-harness.run-id={run_id}"}
    )


def test_native_desktop_judge_api_and_network_boundary(tmp_path, monkeypatch):
    """Real rsi-agent LLM path with a synthetic endpoint; no provider/model call."""
    import threading
    from http.server import ThreadingHTTPServer

    import docker

    from rsi_harness.models import JudgeLLMAuth, JudgeLLMRequest
    from rsi_harness.runtime.desktop import DesktopHost
    from tests.integration.test_judge_llm_docker import API_KEY, _Responses

    _Responses.seen = []
    server = ThreadingHTTPServer(("0.0.0.0", 0), _Responses)
    forbidden = ThreadingHTTPServer(("0.0.0.0", 0), _Responses)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=forbidden.serve_forever, daemon=True).start()
    gateway = DesktopHost(docker.from_env()).gateway()
    base_url = f"http://{gateway}:{server.server_port}"
    monkeypatch.setenv("RSI_JUDGE_LLM_API_KEY", API_KEY)
    task = _task(tmp_path)
    dockerfile = task / "environment/Dockerfile"
    dockerfile.write_text(
        "FROM python:3.12-slim-bookworm\n"
        "RUN mkdir /workspace && chown 1000:1000 /workspace\nWORKDIR /workspace\n"
    )
    config = task / "task.toml"
    config.write_text(
        config.read_text()
        + '\n[metadata.rsi_harness.verifier]\nvisibility = "public"\n'
        'judge_model = "synthetic-judge"\njudge_model_effort = "medium"\n'
    )
    probe = f"""
python3 - <<'CHECK'
import socket
try:
    s = socket.create_connection(('{gateway}', {forbidden.server_port}), timeout=2)
except OSError:
    pass
else:
    s.close()
    raise SystemExit('unapproved host port reachable')
CHECK
"""
    solve = task / "solution/solve.sh"
    solve.write_text(solve.read_text().replace("set -eu\n", "set -eu\n" + probe))
    checker = task / "tests/test.sh"
    checker.write_text(
        """#!/bin/bash
set -eu
"""
        + probe
        + """
mkdir -p /tmp/review
printf 'Return a synthetic verdict.' > /tmp/review/prompt.txt
rsi-agent llm --prompt-file /tmp/review/prompt.txt \\
  --output /tmp/review/answer.json --timeout 20
python3 - <<'CHECK'
import json
value=json.load(open('/tmp/review/answer.json'))
assert value == {'verdict': 'clear', 'echo': '[REDACTED]'}, value
json.dump({'reward': 1}, open('/logs/verifier/reward.json', 'w'))
CHECK
"""
    )
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data", logs_root=tmp_path / "logs"
    )
    result = None
    try:
        result = services.run(
            RunRequest(
                task_dir=task,
                agent_name="oracle",
                options=CompileOptions(agent_name="oracle", max_submissions=1),
                judge_llm=JudgeLLMRequest(auth=JudgeLLMAuth.API, base_url=base_url),
            )
        )
        assert result.status == RunStatus.COMPLETED, result
        assert result.best_score == 1, result
        assert len(_Responses.seen) == 1
        assert _Responses.seen[0][0] == f"Bearer {API_KEY}"
        assert _Responses.seen[0][1]["model"] == "synthetic-judge"
        assert _Responses.seen[0][1]["reasoning"] == {"effort": "medium"}
        for path in (tmp_path / "logs").rglob("*"):
            if path.is_file():
                assert API_KEY not in path.read_text(errors="replace"), path
    finally:
        server.shutdown()
        forbidden.shutdown()
        server.server_close()
        forbidden.server_close()
        if result:
            services.cleanup(result.run_id, delete_workspace=True)


@pytest.mark.parametrize("invalid", ["symlink", "case-collision"])
def test_native_desktop_rejected_archive_is_unscored_and_cleaned(tmp_path, invalid):
    import docker

    from rsi_harness.models import SubmissionStatus

    task = _task(tmp_path)
    checker = task / "tests/test.sh"
    checker.write_text(
        "#!/bin/bash\nset -eu\nprintf '{\"reward\": 1}' > /logs/verifier/reward.json\n"
        + (
            "ln -s /etc/passwd /logs/verifier/link\n"
            if invalid == "symlink"
            else "echo a > /logs/verifier/Case\necho b > /logs/verifier/case\n"
        )
    )
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data", logs_root=tmp_path / "logs"
    )
    result = None
    try:
        result = services.run(
            RunRequest(
                task_dir=task,
                agent_name="oracle",
                options=CompileOptions(agent_name="oracle", max_submissions=1),
            )
        )
        assert result.best_score is None, result
        assert result.reports[0].status == SubmissionStatus.VERIFIER_ERROR, result
        assert result.reports[0].score is None
        assert list((tmp_path / "logs").rglob("agent-1.private-log-export-error.json"))
        assert not docker.from_env().containers.list(
            all=True,
            filters={
                "label": [
                    f"rsi-harness.run-id={result.run_id}",
                    "rsi-harness.role=judge",
                ]
            },
        )
    finally:
        if result:
            services.cleanup(result.run_id, delete_workspace=True)


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_native_desktop_failed_judge_keeps_evidence_without_score(tmp_path, failure):
    task = _task(tmp_path)
    checker = task / "tests/test.sh"
    checker.write_text(
        "#!/bin/bash\nset -eu\necho partial-evidence > /logs/verifier/partial.txt\n"
        + ("exit 7\n" if failure == "exit" else "sleep 60\n")
    )
    if failure == "timeout":
        config = task / "task.toml"
        config.write_text(
            config.read_text().replace("timeout_sec = 60", "timeout_sec = 1")
        )
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data", logs_root=tmp_path / "logs"
    )
    result = None
    try:
        result = services.run(
            RunRequest(
                task_dir=task,
                agent_name="oracle",
                options=CompileOptions(agent_name="oracle", max_submissions=1),
            )
        )
        assert result.status != RunStatus.COMPLETED
        assert result.best_score is None
        assert len(result.reports) == 1
        report = result.reports[0]
        assert report.score is None
        evidence = (
            tmp_path / "logs/runs" / result.run_id / "oracle-check/verifier/agent-1"
        )
        assert (evidence / "partial.txt").read_text().strip() == "partial-evidence"
        assert not (evidence / "reward.json").exists()
    finally:
        if result:
            services.cleanup(result.run_id, delete_workspace=True)


@pytest.mark.parametrize("public", [False, True])
def test_native_desktop_nonroot_judge_writes_private_volume(tmp_path, public):
    """Named Judge users keep their UID/GID and own only their private log volume."""
    task = _task(tmp_path)
    dockerfile = task / "environment/Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text()
        + "RUN groupadd --gid 1002 judge && useradd --uid 1001 --gid 1002 judge\n"
    )
    config = task / "task.toml"
    body = config.read_text().replace("[verifier]\n", '[verifier]\nuser = "judge"\n')
    if public:
        body += '\n[metadata.rsi_harness.verifier]\nvisibility = "public"\n'
    config.write_text(body)
    checker = task / "tests/test.sh"
    checker.write_text(
        checker.read_text().replace(
            "set -eu\n",
            "set -eu\n"
            'test "$(id -u):$(id -g)" = "1001:1002"\n'
            'test "$(stat -c %a:%u:%g /logs/verifier)" = "700:1001:1002"\n'
            "echo nonroot-private-evidence > /logs/verifier/nonroot.txt\n",
        )
    )
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data", logs_root=tmp_path / "logs"
    )
    result = None
    try:
        result = services.run(
            RunRequest(
                task_dir=task,
                agent_name="oracle",
                options=CompileOptions(agent_name="oracle", max_submissions=1),
            )
        )
        assert result.status == RunStatus.COMPLETED, result
        assert result.best_score == 1, result
        evidence = (
            tmp_path / "logs/runs" / result.run_id / "oracle-check/verifier/agent-1"
        )
        assert (
            evidence / "nonroot.txt"
        ).read_text().strip() == "nonroot-private-evidence"
    finally:
        if result:
            services.cleanup(result.run_id, delete_workspace=True)
