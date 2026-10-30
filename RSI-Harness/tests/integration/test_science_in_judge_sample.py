"""The science-ising3d sample's Judge procedure, end to end, as the current user.

Opt-in (RSI_RUN_SCIENCE=1; the task's base image cached, or RSI_ALLOW_PULL=1).
No real model, key or login: the Agent CLI is a fake that reaches a fake model
server on the host at the Agent's base URL with the Agent's key, and the same
server answers the Judge model's Responses API calls with the operator's key;
only the iptables half of the firewall is faked.

1. ``tests/test.sh`` runs in a container shaped like a public Judge: root,
   from the task image, a read-only root filesystem with a /tmp tmpfs, the
   candidate at /workspace/candidate, the trajectory at /logs/agent, and the
   real ``rsi-agent`` with its launch spec. The Judge is the public science
   Judge library (tests/science_judge.py). Each route runs on its own copy as
   its own unprivileged user, unable to open an internet socket; then the
   audit goes through rsi-agent exec to the run's Agent, as the Agent's own
   user, and the 26 rubric reviews through rsi-agent llm to the Judge model,
   four at a time. Twice: with a fake Codex CLI, and with a fake Claude Code
   CLI.
2. A full ``ProductionRuntimeServices`` run of the sample (public /tests and
   Judge image, Judge model access): a scripted Agent reads /tests, previews
   the recompute in Work, alters Work outside its WORKDIR (including its
   Agent CLI), submits, gets 25 of 26, improves and submits again, gets 26;
   no Judge saw the Agent's changes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from docker.types import Mount

from rsi_harness.config import EngineConfig
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter, rsi_loop_oneshot
from rsi_harness.judge_agent import JUDGE_AGENT_TMPFS, install_judge_agent
from rsi_harness.models import (
    AgentPlan,
    CompileOptions,
    ContainerRef,
    JudgeLLMAuth,
    JudgeLLMRequest,
    NetworkPolicy,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.judge_llm import resolve_judge_llm
from rsi_harness.runtime.judge_model_access import build_launch_spec
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.harbor_env_support import docker_or_skip
from tests.integration.test_run_engine import _OneDeviceInventory

REPO = Path(__file__).parents[2]
SAMPLE = REPO / "sample_tasks" / "science-ising3d"
IMAGE = re.search(
    r"^FROM (\S+)$", (SAMPLE / "environment" / "Dockerfile").read_text(), re.M
).group(1)
# Synthetic: accepted only by the fake model server below.
API_KEY = "sk-synthetic-science-judge-key-7d1e"
JUDGE_KEY = "sk-synthetic-science-judge-model-key-3f52"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RSI_RUN_SCIENCE") != "1",
        reason="the science sample is opt-in (RSI_RUN_SCIENCE=1; its image cached)",
    ),
]

# What the candidate's probe.py reports from its route in the Judge.
PROBE = """import json, os, socket
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    inet = "allowed"
except OSError as error:
    inet = error.strerror
print(json.dumps({"uid": os.getuid(), "groups": os.getgroups(),
                  "home": os.environ.get("HOME"),
                  "cwd": os.getcwd(), "files": sorted(os.listdir(".")),
                  "verifier_writable": os.access("/logs/verifier", os.W_OK),
                  "rootfs_writable": os.access("/opt", os.W_OK),
                  "inet_socket": inet}))
"""

# A stand-in Agent CLI (codex or claude) for rsi-agent: it sends the prompt,
# the schema and what it can reach to the fake model at the Agent's base URL
# with the Agent's key, and returns the model's answer as its final message.
FAKE_CLI = r"""#!/usr/bin/env python3
# CLEAN-FAKE-CLI
import json, os, sys, urllib.request
args = sys.argv[1:]
claude = os.path.basename(sys.argv[0]) == "claude"
if claude:
    schema = json.loads(args[args.index("--json-schema") + 1])
    base, key = os.environ["ANTHROPIC_BASE_URL"], os.environ["ANTHROPIC_AUTH_TOKEN"]
else:
    schema = json.load(open(args[args.index("--output-schema") + 1]))
    base, key = os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
prompt = sys.stdin.read()
listed = [line[2:].split(" (")[0] for line in prompt.splitlines()
          if line.startswith("- /") and "/evidence/" in line]
def denied(path):
    try:
        os.listdir(path)
    except OSError:
        return True
    return False
checks = {
    "cli": "claude" if claude else "codex", "argv": args, "uid": os.getuid(),
    "home": os.environ["HOME"], "cwd": os.getcwd(),
    "evidence": {path: os.access(path, os.R_OK) for path in listed},
    "evidence_writable": [p for p in listed if os.access(p, os.W_OK)],
    "verifier_denied": denied("/logs/verifier"),
}
body = json.dumps({"prompt": prompt, "schema": schema, "checks": checks}).encode()
request = urllib.request.Request(
    base.rstrip("/") + "/fake", data=body,
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
answer = json.load(urllib.request.urlopen(request, timeout=60))["answer"]
if claude:
    print(json.dumps({"type": "result", "is_error": False, "result": "",
                      "structured_output": answer}))
else:
    open(args[args.index("--output-last-message") + 1], "w").write(json.dumps(answer))
"""


class FakeModel(BaseHTTPRequestHandler):
    """Answers with the prompt's digests: a clear audit (the Agent CLI's
    calls) and every rubric passes (the Judge model's Responses API calls),
    except those in ``first_round_failing`` while only one audit (the first
    round's) has been answered."""

    seen: list[dict] = []
    first_round_failing: set[str] = set()
    audits = 0
    lock = threading.Lock()
    running = 0
    peak = 0

    def do_POST(self):  # noqa: N802
        cls = type(self)
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        llm = self.path.endswith("/v1/responses")
        if llm:
            payload = {
                "kind": "llm",
                "model": payload["model"],
                "effort": payload["reasoning"]["effort"],
                "prompt": payload["input"][0]["content"][0]["text"],
                "schema": payload["text"]["format"]["schema"],
            }
        with cls.lock:
            cls.running += 1
            cls.peak = max(cls.peak, cls.running)
            cls.seen.append(
                {
                    "kind": "cli",
                    "auth": self.headers.get("Authorization"),
                    "path": self.path,
                    **payload,
                }
            )
        try:
            time.sleep(0.2)  # long enough for parallel calls to overlap
            key = JUDGE_KEY if llm else API_KEY
            if self.headers.get("Authorization") != f"Bearer {key}":
                self._reply(401, {"error": "bad key"})
                return
            properties = payload["schema"]["properties"]
            # The digests to copy are in the prompt.
            answer = dict(
                re.findall(
                    r"^(candidate_digest|rubric_digest|trajectory_digest): "
                    r"([0-9a-f]{64})$",
                    payload["prompt"],
                    re.M,
                )
            )
            if "trajectory_digest" not in properties:
                answer.pop("trajectory_digest", None)
            if "trajectory_digest" in properties:
                with cls.lock:
                    cls.audits += 1
                answer.update(
                    status="clear",
                    reason="Trajectory and candidate inspected.",
                    evidence=["agent_output.txt:1"],
                    next_step="none",
                )
            else:
                rubric = properties["results"]["items"]["properties"]["id"]["enum"][0]
                failing = cls.audits == 1 and rubric in cls.first_round_failing
                answer["results"] = [
                    {
                        "id": rubric,
                        "status": "unresolved" if failing else "pass",
                        "reason": f"{rubric} reviewed",
                        "evidence": ["candidate/theory.md:1"],
                        "next_step": "none",
                    }
                ]
            if llm:
                text = json.dumps(answer)
                output = {"type": "output_text", "text": text}
                self._reply(
                    200,
                    {
                        "status": "completed",
                        "output": [{"type": "message", "content": [output]}],
                    },
                )
                return
            self._reply(200, {"answer": answer})
        finally:
            with cls.lock:
                cls.running -= 1

    def _reply(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        del args


@pytest.fixture
def fake_model(request):
    """The fake server on the default bridge's gateway; returns its /v1 URL."""
    client = docker_or_skip(IMAGE)
    request.addfinalizer(client.close)
    gateway = next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )
    FakeModel.seen, FakeModel.first_round_failing = [], set()
    FakeModel.audits = FakeModel.peak = 0
    server = ThreadingHTTPServer(("0.0.0.0", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request.addfinalizer(server.shutdown)
    return f"http://{gateway}:{server.server_address[1]}/v1"


def _trajectory() -> dict[str, bytes]:
    prompt = b"Solve the declared problem.\n"
    output = b"session 1: wrote theory.md\nsubmitting\n"
    files = []
    for name, data in (("agent_prompt.md", prompt), ("agent_output.txt", output)):
        files.append(
            {
                "path": name,
                "source_size": len(data),
                "offset": 0,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "truncated": False,
                "previous_size": None,
            }
        )
    manifest = {
        "schema_version": 1,
        "round_id": "agent-1",
        "previous_round_id": None,
        "limit_bytes": 256 * 1024 * 1024,
        "files": files,
    }
    return {
        "logs/agent/agent_prompt.md": prompt,
        "logs/agent/agent_output.txt": output,
        "logs/agent/manifest.json": json.dumps(manifest).encode(),
    }


def _write(root: Path, entries: dict[str, bytes]) -> None:
    for name, data in entries.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _give_back(client, root: Path) -> None:
    """Return the files the root Judge wrote to the current user."""
    client.containers.run(
        IMAGE,
        ["chown", "-R", f"{os.getuid()}:{os.getgid()}", "/give-back"],
        mounts=[Mount(target="/give-back", source=str(root), type="bind")],
        network_mode="none",
        remove=True,
    )


@pytest.mark.parametrize(
    ("agent", "cli"), [("codex", "codex"), ("claude-code", "claude")]
)
def test_the_judge_procedure_recomputes_and_reviews_every_rubric(
    tmp_path, fake_model, request, agent, cli
):
    client = docker_or_skip(IMAGE)
    request.addfinalizer(client.close)
    starter = SAMPLE / "environment" / "starter"
    _write(
        tmp_path,
        {
            "workspace/candidate/theory.md": (starter / "theory.md").read_bytes(),
            "workspace/candidate/model.py": (starter / "model.py").read_bytes(),
            "workspace/candidate/probe.py": PROBE.encode(),
            **_trajectory(),
        },
    )
    verifier = tmp_path / "logs" / "verifier"
    verifier.mkdir(parents=True, mode=0o700)
    request.addfinalizer(lambda: _give_back(client, verifier))
    engine = tmp_path / "engine"
    engine.mkdir()
    agent_dir = install_judge_agent(engine)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / cli).write_text(FAKE_CLI)
    (fake_bin / cli).chmod(0o755)
    # The task's Judge model, with an operator's API key at the fake server.
    judge_llm = resolve_judge_llm(
        HarborTaskCompiler().compile(SAMPLE, CompileOptions()),
        JudgeLLMRequest(
            auth=JudgeLLMAuth.API,
            api_key_env="JUDGE_KEY",
            base_url=fake_model.removesuffix("/v1"),
        ),
        RSILoopConfig(),
        None,
        pin=lambda _targets: (),
        environ={"JUDGE_KEY": JUDGE_KEY},
    )
    launch = build_launch_spec(
        rsi_loop_oneshot(
            RSILoopConfig(agent_api_key=API_KEY, agent_api_base_url=fake_model),
            AgentPlan(
                name=agent,
                model="model-test",
                reasoning_effort="high",
                network=NetworkPolicy(mode="public"),
            ),
        ),
        None,
        {API_KEY, *judge_llm.secret_values},
        llm=judge_llm.spec,
    )
    mounts = [
        Mount(
            target="/tests", source=str(SAMPLE / "tests"), type="bind", read_only=True
        ),
        Mount(
            target="/workspace",
            source=str(tmp_path / "workspace"),
            type="bind",
            read_only=True,
        ),
        Mount(
            target="/logs/agent",
            source=str(tmp_path / "logs" / "agent"),
            type="bind",
            read_only=True,
        ),
        Mount(target="/logs/verifier", source=str(verifier), type="bind"),
        # rsi-agent, as the Harness mounts it into a model-access Judge.
        Mount(
            target="/opt/rsi-agent", source=str(agent_dir), type="bind", read_only=True
        ),
        Mount(
            target="/usr/local/bin/rsi-agent",
            source=str(agent_dir / "rsi-agent"),
            type="bind",
            read_only=True,
        ),
        # The clean Work image's Agent CLI.
        Mount(
            target="/opt/agent-cli", source=str(fake_bin), type="bind", read_only=True
        ),
    ]
    container = client.containers.create(
        IMAGE,
        ["sleep", "infinity"],
        labels={"rsi-harness.test": "science"},
        runtime="runc",
        user="0",
        # The shape of a public Judge (judge.py PUBLIC_JUDGE_TMP).
        read_only=True,
        security_opt=["no-new-privileges:true"],
        tmpfs={
            "/tmp": "rw,nosuid,nodev,noexec,size=256m,mode=1777",
            **{str(tmpfs.target): tmpfs.options for tmpfs in JUDGE_AGENT_TMPFS},
        },
        mounts=mounts,
    )
    request.addfinalizer(lambda: container.remove(force=True))
    container.start()
    # The launch spec, streamed from memory as the Harness does.
    DockerContainerRuntime(
        client,
        run_id="science",
        task_id="science-ising3d",
        role="judge",
        task_source_dir=SAMPLE,
        allowed_mount_roots=(engine,),
        judge_model_access=True,
    ).inject_judge_agent(
        ContainerRef(container_id=container.id, role="judge"), launch, owner=None
    )

    code, output = container.exec_run(
        ["bash", "/tests/test.sh"],
        user="0",
        environment={
            "PATH": "/opt/agent-cli:/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            # All the procedure learns about the Agent.
            "RSI_AGENT_NAME": agent,
            "RSI_AGENT_MODEL": "model-test",
            "RSI_AGENT_REASONING_EFFORT": "high",
            "RSI_JUDGE_MODEL": "gpt-6-astra",
            "RSI_JUDGE_MODEL_EFFORT": "medium",
            # The task's [environment.env], as the Harness passes it.
            "OMP_NUM_THREADS": "1",
        },
    )
    output = output.decode(errors="replace")
    assert code == 0, output[-4000:]

    _give_back(client, verifier)
    reward = json.loads((verifier / "reward.json").read_text())
    assert reward == {"reward": 26}, output[-4000:]
    bundle = json.loads((verifier / "bundle.json").read_text())
    isolation = bundle["execution_isolation"]
    assert (isolation["mode"], isolation["uid_base"]) == ("uid", 64000)
    assert isolation["self_test"], isolation
    probe = bundle["diagnostics"]["candidate_reproduction"]
    assert probe["exit_code"] == 0, probe
    assert probe["execution_uid"] == 64001
    seen_by_probe = json.loads(probe["stdout"])
    route_root = seen_by_probe["cwd"]
    assert re.fullmatch(
        r"/tmp/science-judge-[^/]+/route-candidate_reproduction", route_root
    )
    assert seen_by_probe == {
        "uid": 64001,
        "groups": [],
        "home": route_root,
        "cwd": route_root,
        "files": ["model.py", "probe.py", "theory.md"],
        "verifier_writable": False,
        "rootfs_writable": False,
        # The container network is present; the seccomp filter denies it.
        "inet_socket": "Operation not permitted",
    }
    pressure = bundle["diagnostics"]["public_pressure_probes"]
    assert pressure["exit_code"] == 0, pressure
    assert pressure["execution_uid"] == 64002
    assert json.loads(pressure["stdout"])["rows"][0]["K"] == 0
    # The audit through rsi-agent exec to the Agent's CLI with the Agent's key
    # at its base URL, then 26 rubric calls through rsi-agent llm to the Judge
    # model with the operator's key, four at a time.
    [audit_call] = [item for item in FakeModel.seen if item["kind"] == "cli"]
    reviews = [item for item in FakeModel.seen if item["kind"] == "llm"]
    assert len(reviews) == 26
    assert audit_call["auth"] == f"Bearer {API_KEY}"
    assert {item["auth"] for item in reviews} == {f"Bearer {JUDGE_KEY}"}
    assert {(item["model"], item["effort"]) for item in reviews} == {
        ("gpt-6-astra", "medium")
    }
    assert FakeModel.peak == 4
    checks = audit_call["checks"]
    assert checks["cli"] == cli
    assert checks["uid"] == 61000
    assert checks["home"].startswith("/run/rsi-agent/jobs/")
    # No --workdir: the Agent works in rsi-agent's private scratch.
    assert checks["cwd"].startswith("/run/rsi-agent/jobs/"), checks["cwd"]
    # The Agent reads every evidence file it is given, changes none, and
    # cannot read the Judge's own outputs.
    assert checks["evidence"] and all(checks["evidence"].values()), checks
    assert checks["evidence_writable"] == []
    assert checks["verifier_denied"] is True
    effort = checks["argv"][1] if cli == "claude" else " ".join(checks["argv"])
    assert "high" in effort
    audit = audit_call["prompt"]
    assert "/logs/agent/agent_output.txt" in audit
    assert re.search(
        r"/tmp/science-evidence-[^/]+/evidence/candidate/theory\.md", audit
    )
    review = json.loads((verifier / "review.json").read_text())
    assert (review["agent"], review["model"], review["reasoning_effort"]) == (
        agent,
        "model-test",
        "high",
    )
    assert (review["judge_model"], review["judge_model_effort"]) == (
        "gpt-6-astra",
        "medium",
    )
    assert "Integrity audit: clear" in output
    assert output.rstrip().endswith("passed 26 of 26")
    for key in (API_KEY, JUDGE_KEY):
        assert key not in output
        for path in verifier.rglob("*"):
            if path.is_file():
                assert key not in path.read_text(errors="replace"), path


# The Agent reads the public tests, previews the recompute in Work, leaves a
# marker outside WORKDIR and replaces its own codex in Work, then submits;
# the fake model leaves one rubric unresolved. It reads the feedback, revises
# and submits again; all 26 pass. Neither Judge sees the marker or the
# replaced codex: each starts from the clean Work image, whose codex the
# Judge's rsi-agent runs.
AGENT_SCRIPT = """\
set -eu
test -r /tests/rubrics.json && test ! -w /tests/rubrics.json
if command -v rsi-agent; then echo RSI-AGENT-PRESENT; exit 9; fi
mkdir -p /workspace/candidate
cp /opt/science/starter/* /workspace/candidate/
cat > /workspace/candidate/probe.py <<'PROBE'
import json, os
print(json.dumps({"agent_marker": os.path.exists("/opt/science/agent-marker"),
                  "codex": open("/usr/local/bin/codex").read()[:60]}))
PROBE
echo altered > /opt/science/agent-marker
printf '#!/bin/sh\\necho AGENT-CODEX\\n' > /usr/local/bin/codex
python3 /tests/judge.py --recompute /workspace/candidate > /workspace/recompute.txt
grep -q '^\\[candidate_reproduction\\] exit 0' /workspace/recompute.txt
grep -q '"agent_marker": true' /workspace/recompute.txt
echo WORK-RECOMPUTE-OK
rsi-submit
grep -q 'passed 25 of 26' /run/rsi-harness/feedback/agent-1.log
echo 'R26 addressed.' >> /workspace/candidate/theory.md
echo CANDIDATE-REVISED
rsi-submit
"""


def test_a_scripted_run_reviews_two_candidates_from_the_clean_image(
    tmp_path, fake_model, monkeypatch, request
):
    monkeypatch.setenv("RSI_JUDGE_LLM_API_KEY", JUDGE_KEY)
    client = docker_or_skip(IMAGE)
    request.addfinalizer(client.close)
    task = tmp_path / "task"
    shutil.copytree(SAMPLE, task)
    config = task / "task.toml"
    config.write_text(
        config.read_text().replace("timeout_sec = 21600", "timeout_sec = 600")
    )
    # A short data root keeps the sandbox socket path within 107 bytes.
    data = Path(tempfile.mkdtemp(prefix="rsi-sci-"))
    request.addfinalizer(lambda: shutil.rmtree(data, ignore_errors=True))
    FakeModel.first_round_failing = {"R26"}

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, prepare_request):
            prepared = super().prepare(prepare_request)
            return replace(prepared, command=("/bin/bash", "-lc", AGENT_SCRIPT))

    # Real image preparation, but no model CLI download: the fake Codex CLI
    # in the clean Work image, which each Judge's rsi-agent runs.
    encoded = base64.b64encode(FAKE_CLI.encode()).decode()
    monkeypatch.setattr(
        CodexAgent,
        "install_cmds",
        [
            f"echo {encoded} | base64 -d > /usr/local/bin/codex && "
            "chmod 0755 /usr/local/bin/codex"
        ],
    )
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=API_KEY, agent_api_base_url=fake_model
        ),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: ScriptedAgent(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
        engine_config=EngineConfig(),
    )

    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)

    result = services.run(
        RunRequest(
            task_dir=task,
            agent_name="codex",
            model="gpt-test",
            options=CompileOptions(agent_name="codex", primary_reward="reward"),
            judge_llm=JudgeLLMRequest(
                auth=JudgeLLMAuth.API, base_url=fake_model.removesuffix("/v1")
            ),
        )
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "science-ising3d"
    agent_output = (run_root / "agent_output.txt").read_text()
    feedback = "\n".join(
        path.read_text() for path in sorted((run_root / "feedback").glob("agent-*.log"))
    )
    assert result.status == RunStatus.COMPLETED, agent_output[-3000:] + feedback[-3000:]
    assert [report.score for report in result.reports] == [25, 26], feedback[-3000:]
    assert "WORK-RECOMPUTE-OK" in agent_output
    assert "CANDIDATE-REVISED" in agent_output
    assert "RSI-AGENT-PRESENT" not in agent_output
    # Both Judges ran from the clean Work image: no marker, the clean codex.
    assert feedback.count('"agent_marker": false') == 2, feedback[-3000:]
    assert "CLEAN-FAKE-CLI" in feedback and "AGENT-CODEX" not in feedback
    audits = [item for item in FakeModel.seen if item["kind"] == "cli"]
    assert len(audits) == 2
    assert {item["checks"]["uid"] for item in audits} == {61000}
    assert sum(item["kind"] == "llm" for item in FakeModel.seen) == 52
    for path in run_root.rglob("*"):
        if path.is_file() and os.access(path, os.R_OK):
            for key in (API_KEY, JUDGE_KEY):
                assert key not in path.read_text(errors="replace"), path
