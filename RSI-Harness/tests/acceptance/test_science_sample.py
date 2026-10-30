"""The science-ising3d sample: a public, model-access Judge built on the
public science Judge library.

The sample's ``tests/science_judge.py`` is a byte-for-byte copy of OpenRSI-X's
science-task-agent ``assets/science_judge.py``; that library's own unit tests
live with it in OpenRSI-X. These tests check the sample's configuration, that
the library agrees with what RSI Harness provides a Judge (rsi-agent, its exit
statuses and users, the trajectory manifest, the paths), and whole Judge
rounds of the sample through the real ``rsi-agent`` program.

Everything here runs without Docker, a GPU, network, root or an Agent login:
routes run as the current user, ``rsi-agent`` confines its fake Codex CLI
(the audit) and its Judge model calls (the reviews, to a fake Responses API
on loopback) with Landlock, as it does for a Judge that is not root, and the
trajectory comes from the Harness's own capture. Root, the uid drop and the Judge
container run in the opt-in Docker test
(tests/integration/test_science_in_judge_sample.py).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import ipaddress
import json
import os
import re
import shlex
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rsi_harness.integrations.rsi_loop import rsi_loop_oneshot
from rsi_harness.judge_agent import (
    JUDGE_AGENT_LAUNCHER_TARGET,
    JUDGE_AGENT_SLOTS,
    JUDGE_AGENT_UIDS,
    rsi_agent,
)
from rsi_harness.models import (
    JUDGE_TRAJECTORY_ROOT,
    AgentPlan,
    CompileOptions,
    JudgeLLMAuth,
    JudgeLLMRequest,
    NetworkPolicy,
)
from rsi_harness.runtime.judge import PUBLIC_JUDGE_TMP
from rsi_harness.runtime.judge_llm import resolve_judge_llm
from rsi_harness.runtime.judge_model_access import build_launch_spec
from rsi_harness.runtime.network import PinnedEndpoint
from rsi_harness.runtime.trajectory import (
    TRAJECTORY_FILE_LIMIT_BYTES,
    TRAJECTORY_SCHEMA_VERSION,
    capture_round_trajectory,
)
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.config import RSILoopConfig
from tests.landlock_helpers import native_signal_compatibility

REPO = Path(__file__).parents[2]
SAMPLE = REPO / "sample_tasks" / "science-ising3d"
TESTS = SAMPLE / "tests"
STARTER = SAMPLE / "environment" / "starter"
RSI_AGENT_PROGRAM = REPO / "src" / "rsi_harness" / "judge_agent" / "rsi_agent.py"
# sha256 of OpenRSI-X science-task-agent assets/science_judge.py, version 9.4.
LIBRARY_SHA256 = "5d5bca020962e41b2a0cdafaddd128b2fe9f3aa2ead47f407c70c48a32f30a8e"
API_KEY = "sk-synthetic-science-sample-key-4b7a"
JUDGE_KEY = "sk-synthetic-science-judge-model-key-91c0"


def _load_judge():
    """Import judge.py (and science_judge.py) without writing __pycache__
    into the task's tests/, which would change what the Judge runs."""
    spec = importlib.util.spec_from_file_location(
        "science_ising3d_judge", TESTS / "judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


JUDGE = _load_judge()
CONFIG = JUDGE.CONFIG
SJ = sys.modules["science_judge"]
RUBRICS = json.loads((TESTS / "rubrics.json").read_text())
IDS = [item["id"] for item in RUBRICS["rubrics"]]


# -- the sample's configuration ----------------------------------------------


def test_the_sample_declares_every_public_judge_capability():
    definition = HarborTaskCompiler().compile(SAMPLE, CompileOptions())

    verifier = definition.verifier
    assert verifier.visibility == "public"
    assert verifier.model_access is True
    assert (verifier.judge_model, verifier.judge_model_effort) == (
        "gpt-6-astra",
        "medium",
    )
    assert IDS == [f"R{n:02d}" for n in range(1, 27)]
    assert definition.gpu_requirement.count == verifier.gpu_count == 0
    # A public Judge reads only Work's WORKDIR: the candidate lives there.
    assert str(definition.workdir) == CONFIG.workdir == "/workspace"
    # Nothing is brokered: the recompute runs in the Judge itself.
    assert definition.sandbox is None
    # The Judge takes no backend option and no model credential.
    # [environment.env] reaches the Judge as usual: the thread count routes get.
    assert dict(verifier.environment) == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
    }


def test_the_judge_is_the_unchanged_library_with_this_tasks_config():
    library = (TESTS / "science_judge.py").read_bytes()
    assert hashlib.sha256(library).hexdigest() == LIBRARY_SHA256
    assert SJ.__version__ == "9.4"
    assert CONFIG.problems() == []
    assert SJ.validate_rules(TESTS) == []
    assert [(route.name, route.timeout_sec) for route in CONFIG.routes] == [
        ("candidate_reproduction", 60),
        ("public_pressure_probes", 60),
    ]
    assert CONFIG.checkers == ()
    assert (CONFIG.audit_seconds, CONFIG.review_seconds) == (1200, 1800)
    assert CONFIG.review_workers == 4
    # judge.py leaves the Judge's own paths to the library.
    defaults = SJ.JudgeConfig(
        workdir="/w", routes=(), audit_seconds=60, review_seconds=60
    )
    for name in ("rsi_agent", "tests_dir", "logs_dir", "trajectory_dir", "tmp_dir"):
        assert getattr(CONFIG, name) == getattr(defaults, name), name
    # The verifier timeout covers every route, the audit and the review.
    definition = HarborTaskCompiler().compile(SAMPLE, CompileOptions())
    used = sum(r.timeout_sec for r in CONFIG.routes)
    used += CONFIG.audit_seconds + CONFIG.review_seconds
    assert definition.verifier.timeout_seconds - used >= 300


def test_the_task_calls_models_only_through_rsi_agent():
    for path in (TESTS / "judge.py", TESTS / "test.sh", SAMPLE / "task.toml"):
        text = path.read_text()
        for forbidden in (
            "OPENAI",
            "ANTHROPIC",
            "codex",
            "claude",
            "chat/completions",
            "urllib",
            "SCIENCE_",
            "API_KEY",
        ):
            assert forbidden not in text, (path.name, forbidden)
    assert (TESTS / "test.sh").read_text() == (
        "#!/bin/bash\nset -euo pipefail\nexec python3 -I -B /tests/judge.py\n"
    )
    assert os.access(TESTS / "test.sh", os.X_OK)
    assert not (TESTS / "__pycache__").exists()


def test_the_task_image_places_the_starter_as_the_candidate():
    dockerfile = (SAMPLE / "environment" / "Dockerfile").read_text()
    base = re.search(r"^FROM (\S+)$", dockerfile, re.MULTILINE).group(1)

    assert re.fullmatch(r"python@sha256:[0-9a-f]{64}", base)
    assert "COPY starter/ /opt/science/starter/" in dockerfile
    assert "COPY starter/ /workspace/candidate/" in dockerfile
    assert re.search(r"^WORKDIR /workspace$", dockerfile, re.MULTILINE)
    # The rules are public through /tests, never copied into the image.
    commands = [line for line in dockerfile.splitlines() if not line.startswith("#")]
    assert not any("tests" in line or "solution" in line for line in commands)
    assert not (SAMPLE / "operator-policy.toml").exists()
    instruction = (SAMPLE / "instruction.md").read_text()
    assert "/tests/rubrics.json" in instruction
    assert "python3 /tests/judge.py --recompute /workspace/candidate" in instruction


# -- the library against what RSI Harness gives a Judge -------------------------


def test_the_library_matches_rsi_agent():
    assert SJ.RSI_AGENT == str(JUDGE_AGENT_LAUNCHER_TARGET)
    assert SJ.AGENT_UIDS == JUDGE_AGENT_UIDS
    assert SJ.MAX_AGENT_CALLS == JUDGE_AGENT_SLOTS
    assert SJ.MAX_PROMPT_BYTES == rsi_agent.MAX_PROMPT_BYTES
    # Each exit status rsi-agent defines maps onto the documented code.
    assert SJ.RSI_AGENT_EXIT == {
        rsi_agent.EXIT_AGENT: "model_error",
        rsi_agent.EXIT_USAGE: "infrastructure",
        rsi_agent.EXIT_UNAVAILABLE: "infrastructure",
        rsi_agent.EXIT_TIMEOUT: "model_timeout",
        rsi_agent.EXIT_INTERNAL: "infrastructure",
        rsi_agent.EXIT_TERMINATED: "infrastructure",
    }
    # rsi-agent's own SIGTERM, grace and SIGKILL fit before the deadline.
    assert SJ.AGENT_STOP_MARGIN > rsi_agent.KILL_GRACE_SECONDS
    # The library's call is one rsi-agent accepts: no --workdir, and only
    # PATH and LANG in its environment.
    call = SJ.AgentCall("/usr/local/bin/rsi-agent", {"PATH": "/usr/bin"})
    argv = call.argv(Path("/logs/verifier/model/audit"), 1190)
    options = rsi_agent._parser().parse_args(argv[1:])
    assert (options.command, options.timeout, options.workdir) == ("exec", 1190, None)
    assert set(call.env) == {"PATH", "LANG"}


def test_the_library_reads_the_harness_trajectory(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "agent_prompt.md").write_text("Solve the declared problem.\n")
    (run_root / "agent_output.txt").write_text("round one\n")
    capture_round_trajectory(run_root, "agent-1")
    with (run_root / "agent_output.txt").open("a") as stream:
        stream.write("round two\n")
    second = capture_round_trajectory(run_root, "agent-2")

    trajectory = SJ.load_trajectory(second)

    manifest = json.loads((second / "manifest.json").read_text())
    assert set(manifest) == SJ.TRAJECTORY_KEYS
    assert {key for item in manifest["files"] for key in item} == (
        SJ.TRAJECTORY_ENTRY_KEYS
    )
    assert manifest["schema_version"] == TRAJECTORY_SCHEMA_VERSION == 1
    assert manifest["limit_bytes"] == TRAJECTORY_FILE_LIMIT_BYTES
    output = next(e for e in trajectory.entries if e.path == "agent_output.txt")
    assert (output.new_start, output.bytes) == (10, 20)
    assert (
        trajectory.digest
        == hashlib.sha256((second / "manifest.json").read_bytes()).hexdigest()
    )
    assert CONFIG.trajectory_dir == str(JUDGE_TRAJECTORY_ROOT)


def test_the_library_fits_the_public_judge_filesystem():
    options = PUBLIC_JUDGE_TMP.options.split(",")
    assert options.count("size=256m") == 1
    # /tmp is noexec, so every route starts an interpreter on its /tmp copy
    # instead of executing a file there.
    assert "noexec" in options and "exec" not in options
    assert all(not route.argv[0].startswith("{root}") for route in CONFIG.routes)
    assert CONFIG.tmp_dir == str(PUBLIC_JUDGE_TMP.target)
    assert CONFIG.logs_dir == "/logs/verifier"
    assert CONFIG.tests_dir == "/tests"
    # The route uid block stays clear of rsi-agent's Agent users.
    routes = range(CONFIG.route_uid_base, CONFIG.route_uid_base + CONFIG.uid_count())
    assert not set(routes) & set(JUDGE_AGENT_UIDS)


# -- whole rounds through the real rsi-agent ----------------------------------------

# A stand-in Codex CLI for rsi-agent exec (the audit): answers from the schema
# and the digests in the prompt, as the plan file says, and records the call.
FAKE_CODEX = r"""#!{python}
import json, os, re, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
schema = json.load(open(args[args.index("--output-schema") + 1]))
props = schema["properties"]
job = ("audit" if "trajectory_digest" in props
       else props["results"]["items"]["properties"]["id"]["enum"][0])
plan = json.load(open(os.environ["FAKE_PLAN"]))
action = plan.get(job, "clear" if job == "audit" else "pass")
with open(os.environ["FAKE_RECORD"], "a") as log:
    log.write(json.dumps({{"job": job, "argv": args, "cwd": os.getcwd(),
                          "prompt": prompt, "schema": schema}}) + "\n")
if action == "crash":
    sys.stderr.write("the model refused\n")
    sys.exit(7)
digests = dict(re.findall(
    r"^(candidate_digest|rubric_digest|trajectory_digest): ([0-9a-f]{{64}})$",
    prompt, re.M))
if job == "audit":
    answer = {{**digests, "status": action, "reason": "trajectory read",
              "evidence": ["/logs/agent/agent_output.txt"], "next_step": "none"}}
else:
    answer = {{**digests, "results": [{{"id": job, "status": action,
              "reason": job + " reviewed", "evidence": ["candidate/theory.md"],
              "next_step": "none"}}]}}
open(args[args.index("--output-last-message") + 1], "w").write(json.dumps(answer))
"""


def _answer(job: str, action: str, prompt: str) -> dict:
    """A rubric review's answer, with the digests the prompt gives."""
    digests = dict(
        re.findall(r"^(candidate_digest|rubric_digest): ([0-9a-f]{64})$", prompt, re.M)
    )
    result = {
        "id": job,
        "status": action,
        "reason": job + " reviewed",
        "evidence": ["candidate/theory.md"],
        "next_step": "none",
    }
    return {**digests, "results": [result]}


class FakeJudgeModel(BaseHTTPRequestHandler):
    """The Judge model behind rsi-agent llm: a Responses API that answers each
    rubric as the plan file says and records each call."""

    def log_message(self, *args) -> None:
        del args

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server = self.server
        prompt = body["input"][0]["content"][0]["text"]
        schema = body["text"]["format"]["schema"]
        job = schema["properties"]["results"]["items"]["properties"]["id"]["enum"][0]
        action = json.loads(server.plan.read_text()).get(job, "pass")
        with server.lock:
            server.seen.append(
                {
                    "job": job,
                    "auth": self.headers.get("Authorization"),
                    "model": body["model"],
                    "effort": body["reasoning"]["effort"],
                    "prompt": prompt,
                    "schema": schema,
                }
            )
        text = json.dumps(_answer(job, action, prompt))
        response = {
            "id": "resp_1",
            "object": "response",
            "status": "failed" if action == "crash" else "completed",
            "error": {"message": "the model refused"} if action == "crash" else None,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        }
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _pin(endpoints: tuple[str, ...]) -> tuple[PinnedEndpoint, ...]:
    return tuple(
        PinnedEndpoint(
            hostname="127.0.0.1",
            port=int(url.rsplit(":", 1)[1].split("/")[0]),
            addresses=(ipaddress.ip_address("127.0.0.1"),),
        )
        for url in endpoints
    )


class Round:
    """A Judge round of the sample, as the current user, with the real
    rsi-agent and the Harness's trajectory capture."""

    def __init__(self, tmp_path: Path, model: ThreadingHTTPServer) -> None:
        self.tmp = tmp_path
        self.model = model
        self.workspace = tmp_path / "workspace"
        (self.workspace / "candidate").mkdir(parents=True)
        for path in STARTER.iterdir():
            (self.workspace / "candidate" / path.name).write_bytes(path.read_bytes())
        self.logs = tmp_path / "verifier"
        self.scratch = tmp_path / "scratch"
        self.scratch.mkdir()
        self.run_root = tmp_path / "run"
        self.run_root.mkdir()
        (self.run_root / "agent_prompt.md").write_text("Solve the declared problem.\n")
        (self.run_root / "agent_output.txt").write_text("wrote theory.md\n")
        self.rounds = 0
        self.record = tmp_path / "calls.jsonl"
        self.plan = tmp_path / "plan.json"
        self.plan.write_text("{}")
        model.plan = self.plan
        # rsi-agent's state root, with the launch spec the Harness writes.
        state = tmp_path / "state"
        (state / "private").mkdir(parents=True)
        state.chmod(0o711)
        (state / "private").chmod(0o700)
        oneshot = rsi_loop_oneshot(
            RSILoopConfig(
                agent_api_key=API_KEY,
                agent_api_base_url="http://model.test/v1",
                agent_extra_env={
                    "FAKE_RECORD": str(self.record),
                    "FAKE_PLAN": str(self.plan),
                },
            ),
            AgentPlan(
                name="codex",
                model="gpt-test",
                reasoning_effort="high",
                network=NetworkPolicy(mode="public"),
            ),
        )
        # The Judge model the task declares, with an operator's API key.
        judge_llm = resolve_judge_llm(
            HarborTaskCompiler().compile(SAMPLE, CompileOptions()),
            JudgeLLMRequest(
                auth=JudgeLLMAuth.API,
                api_key_env="JUDGE_KEY",
                base_url=f"http://127.0.0.1:{model.server_address[1]}",
            ),
            RSILoopConfig(),
            None,
            pin=_pin,
            environ={"JUDGE_KEY": JUDGE_KEY},
        )
        launch = json.loads(
            build_launch_spec(
                oneshot,
                None,
                {API_KEY, *judge_llm.secret_values},
                visibility="public",
                allow_missing_landlock_signal=native_signal_compatibility(),
                llm=judge_llm.spec,
            )
        )
        # The fake CLI's record: the only file outside its job a confined
        # call may write (the Harness grants none).
        self.record.touch()
        launch["writable"] = [str(self.record)]
        (state / "private" / "launch.json").write_text(json.dumps(launch))
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        codex = self.bin / "codex"
        codex.write_text(FAKE_CODEX.format(python=sys.executable))
        codex.chmod(0o755)
        # The library passes rsi-agent only PATH and LANG; this launcher
        # adds the test state root, as /run/rsi-agent is in a Judge.
        self.rsi_agent = tmp_path / "rsi-agent"
        self.rsi_agent.write_text(
            "#!/bin/sh\n"
            f"RSI_AGENT_STATE_ROOT={shlex.quote(str(state))} exec "
            f"{shlex.quote(sys.executable)} -I -S -B "
            f'{shlex.quote(str(RSI_AGENT_PROGRAM))} "$@"\n'
        )
        self.rsi_agent.chmod(0o755)

    def config(self, **changes):
        return dataclasses.replace(
            CONFIG,
            workdir=str(self.workspace),
            tests_dir=str(TESTS),
            logs_dir=str(self.logs),
            trajectory_dir=str(self.trajectory),
            tmp_dir=str(self.scratch),
            rsi_agent=str(self.rsi_agent),
            **changes,
        )

    def run(self, plan=None, environ=None, **changes):
        self.plan.write_text(json.dumps(plan or {}))
        self.record.write_text("")
        self.model.seen.clear()
        self.rounds += 1
        self.trajectory = capture_round_trajectory(
            self.run_root, f"agent-{self.rounds}"
        )
        stdout = io.StringIO()
        code = SJ.run_judge(
            self.config(**changes),
            environ={
                "PATH": f"{self.bin}:/usr/bin:/bin",
                "RSI_AGENT_NAME": "codex",
                "RSI_AGENT_MODEL": "gpt-test",
                "RSI_AGENT_REASONING_EFFORT": "high",
                "RSI_JUDGE_MODEL": "gpt-6-astra",
                "RSI_JUDGE_MODEL_EFFORT": "medium",
                "OMP_NUM_THREADS": "1",
                **(environ or {}),
            },
            stdout=stdout,
            isolation="current_user",
        )
        return code, stdout.getvalue()

    def calls(self):
        """The audit's Agent CLI calls, then the Judge model's calls."""
        audits = [json.loads(line) for line in self.record.read_text().splitlines()]
        return audits + list(self.model.seen)

    def read(self, name):
        return json.loads((self.logs / name).read_text())


@pytest.fixture
def judge_round(tmp_path):
    model = ThreadingHTTPServer(("127.0.0.1", 0), FakeJudgeModel)
    model.daemon_threads = True
    model.seen = []
    model.lock = threading.Lock()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    yield Round(tmp_path, model)
    model.shutdown()
    model.server_close()


def test_a_clear_round_rewards_the_rubrics_this_candidate_passes(judge_round):
    code, feedback = judge_round.run({"R26": "unresolved"})

    assert code == 0, feedback
    assert judge_round.read("reward.json") == {"reward": 25}
    review = judge_round.read("review.json")
    assert [item["id"] for item in review["results"]] == IDS
    assert (review["completed_rubrics"], review["total_rubrics"]) == (25, 26)
    assert review["scientific_complete"] is False
    assert (review["model_access"], review["agent"], review["model"]) == (
        "rsi-agent",
        "codex",
        "gpt-test",
    )
    assert (review["reasoning_effort"], review["review_workers"]) == ("high", 4)
    assert (review["judge_model"], review["judge_model_effort"]) == (
        "gpt-6-astra",
        "medium",
    )
    assert review["integrity_status"] == "clear"
    # Both routes ran on the starter; the probe claims nothing.
    diagnostics = judge_round.read("bundle.json")["diagnostics"]
    probe = diagnostics["candidate_reproduction"]
    assert probe["exit_code"] == 0, probe
    assert json.loads(probe["stdout"])["scientific_completion_claimed"] is False
    pressure = diagnostics["public_pressure_probes"]
    assert pressure["exit_code"] == 0, pressure
    assert [row["K"] for row in json.loads(pressure["stdout"])["rows"]] == [
        0,
        0.01,
        0.1,
        0.2,
        0.3,
        1,
        8,
    ]
    # One audit by the Agent's CLI in rsi-agent's private scratch, never in
    # the evidence; then one Judge model call per rubric, with the operator's
    # key, the task's model and effort, and the evidence in its prompt.
    calls = judge_round.calls()
    assert [call["job"] for call in calls][0] == "audit"
    assert "/evidence" not in calls[0]["cwd"]
    assert "do not search the web" in calls[0]["prompt"].lower()
    assert sorted(call["job"] for call in calls[1:]) == IDS
    for call in calls[1:]:
        assert call["auth"] == f"Bearer {JUDGE_KEY}"
        assert (call["model"], call["effort"]) == ("gpt-6-astra", "medium")
        assert "theory.md" in call["prompt"]
    audit = calls[0]["prompt"]
    assert "# Integrity audit" in audit
    assert str(judge_round.trajectory / "agent_output.txt") in audit
    assert "/evidence/candidate/theory.md" in audit
    assert "Rubric R26: unresolved" in feedback
    assert feedback.rstrip().endswith("passed 25 of 26")
    for key in (API_KEY, JUDGE_KEY):
        assert key not in feedback
        for path in judge_round.logs.rglob("*"):
            if path.is_file():
                assert key not in path.read_text(errors="replace"), path

    # The next round audits what is new since this one and passes all 26.
    with (judge_round.run_root / "agent_output.txt").open("a") as stream:
        stream.write("addressed R26\n")
    code, feedback = judge_round.run()

    assert code == 0, feedback
    assert judge_round.read("reward.json") == {"reward": 26}
    assert judge_round.read("review.json")["scientific_complete"] is True
    audit = judge_round.calls()[0]["prompt"]
    assert "previous round agent-1" in audit
    assert "new since the previous round: copy bytes 16-30" in audit


@pytest.mark.parametrize(
    ("status", "code"),
    [("violation", "integrity_violation"), ("unresolved", "integrity_unresolved")],
)
def test_an_audit_that_is_not_clear_writes_no_reward(judge_round, status, code):
    exit_code, feedback = judge_round.run({"audit": status})

    assert exit_code == 2
    assert not (judge_round.logs / "reward.json").exists()
    error = judge_round.read("review-error.json")
    assert (error["code"], error["stage"]) == (code, "audit")
    assert judge_round.read("integrity.json")["status"] == status
    assert [call["job"] for call in judge_round.calls()] == ["audit"]
    assert "not a scientific verdict" in feedback


def test_a_failed_agent_call_writes_no_reward(judge_round):
    exit_code, _feedback = judge_round.run({"R03": "crash"})

    assert exit_code == 2
    assert not (judge_round.logs / "reward.json").exists()
    error = judge_round.read("review-error.json")
    assert (error["code"], error["stage"]) == ("model_error", "review")
    call = json.loads(
        (judge_round.logs / "model" / "rubric-002-R03" / "call.json").read_text()
    )
    assert call["exit_code"] == rsi_agent.EXIT_AGENT


def test_without_model_access_the_round_stops_before_the_recompute(judge_round):
    exit_code, _feedback = judge_round.run(environ={"RSI_AGENT_NAME": ""})

    assert exit_code == 2
    error = judge_round.read("review-error.json")
    assert (error["code"], error["stage"]) == ("infrastructure", "validate")
    assert not (judge_round.logs / "bundle.json").exists()
    assert judge_round.calls() == []


def test_a_missing_candidate_says_how_to_restore_it(judge_round):
    for path in (judge_round.workspace / "candidate").iterdir():
        path.unlink()
    (judge_round.workspace / "candidate").rmdir()

    exit_code, _feedback = judge_round.run()

    assert exit_code == 2
    error = judge_round.read("review-error.json")
    assert error["code"] == "candidate_missing"
    assert "/opt/science/starter" in error["error"]


def test_work_can_preview_the_judge_recompute(tmp_path):
    stdout = io.StringIO()

    code = SJ.preview(
        dataclasses.replace(CONFIG, tmp_dir=str(tmp_path)), STARTER, stdout=stdout
    )

    output = stdout.getvalue()
    assert code == 0, output
    assert "[candidate_reproduction] exit 0" in output
    assert "[public_pressure_probes] exit 0" in output
    assert "no audit, no review, no reward." in output


@pytest.mark.parametrize(
    ("name", "script"),
    [
        ("writes_nothing", "pass\n"),
        ("fails", "raise SystemExit(3)\n"),
    ],
)
def test_a_candidate_file_never_stands_in_for_a_route_output(tmp_path, name, script):
    # Since library version 3: a file the candidate ships at a declared output path
    # is withheld from the route's copy, so a route that writes nothing, or
    # fails, has no output instead of the candidate's stale file.
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "run.py").write_text(script)
    (candidate / "out.json").write_text('{"stale": true}\n')
    route = SJ.Route(
        name=name,
        argv=(sys.executable, "-I", "-B", "{root}/run.py"),
        timeout_sec=30,
        requires=("run.py",),
        outputs=("out.json",),
    )
    config = dataclasses.replace(
        CONFIG, routes=(route,), checkers=(), tmp_dir=str(tmp_path)
    )
    files = SJ.candidate_map(
        candidate, max_files=config.max_files, max_bytes=config.max_bytes
    )

    bundle = SJ.recompute(
        config, candidate, files, isolation_mode="preview", outputs_dir=None
    )

    diagnostic = bundle["diagnostics"][name]
    assert diagnostic["withheld"] == ["out.json"]
    assert diagnostic["outputs"]["out.json"]["status"] == "missing"
    assert "out.json" in bundle["files"]
