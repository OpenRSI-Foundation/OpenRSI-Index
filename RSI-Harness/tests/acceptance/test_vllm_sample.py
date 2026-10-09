"""The vllm-in-judge sample: it compiles as CPU Work with a one-GPU Judge,
its operator policy grants the Judge's brokered environments once, its Work
agent places a checkpoint and submits, and its fixed procedure serves the
checkpoint with vLLM, runs Harbor 0.21's terminus-2 against it through the
plugin and reports. vLLM, nvidia-smi, Harbor and the Hub CLI are stand-ins
here; tests/integration/test_vllm_in_judge_sample.py runs the real ones
(RSI_RUN_VLLM=1) and scripts/operator/vllm_demo.sh the root demo."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions
from rsi_harness.runtime.sandbox_contracts import SandboxEnvTask
from rsi_harness.runtime.sandbox_policy import (
    load_sandbox_policy,
    resolve_env_grant,
    validate_sandbox_policy,
)
from rsi_harness.task.compiler import HarborTaskCompiler
from tests.acceptance.test_sample_task import TB2, assert_same_tree

REPO = Path(__file__).parents[2]
SAMPLE = REPO / "sample_tasks" / "vllm-in-judge"
TASKS = SAMPLE / "tests" / "harbor-tasks"
POLICY = REPO / "sample_tasks" / "vllm-in-judge" / "operator-policy.toml"
PLUGIN = "rsi_sandbox_harbor:ManagedSandboxEnvironment"
GPU = "GPU-00000000-0000-0000-0000-000000000004"


def demo_report():
    """The sample's reporter, loaded without leaving bytecode in its /tests."""
    spec = importlib.util.spec_from_file_location(
        "demo_report", SAMPLE / "tests" / "demo_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    return module


# -- the task and its grant -------------------------------------------------------


def test_the_sample_compiles_as_cpu_work_with_a_one_gpu_judge():
    definition = HarborTaskCompiler().compile(SAMPLE, CompileOptions())

    assert definition.gpu_requirement.count == 0
    assert definition.verifier.gpu_count == 1
    assert str(definition.workdir) == "/workspace"
    assert (definition.service.cpus, definition.service.memory_mb) == (8, 24576)
    sandbox = definition.sandbox
    assert isinstance(sandbox, SandboxEnvTask) and sandbox.version == 2
    assert sandbox.environments.work is None
    judge = sandbox.environments.judge
    assert judge.network == ("public",) and judge.pull and not judge.build
    assert dict(definition.verifier.environment) == {
        "RSI_VLLM_TASKS": "${RSI_VLLM_TASKS:-}",
        "RSI_VLLM_MAX_TURNS": "${RSI_VLLM_MAX_TURNS:-12}",
        "RSI_VLLM_CONCURRENCY": "${RSI_VLLM_CONCURRENCY:-2}",
    }
    config = tomllib.loads((SAMPLE / "task.toml").read_text())
    # Work fetches the checkpoint; the Judge has no network of its own.
    assert config["agent"]["network_mode"] == "public"
    assert config["verifier"]["network_mode"] == "no-network"


def test_the_operator_policy_grants_the_judge_once():
    definition = HarborTaskCompiler().compile(SAMPLE, CompileOptions())
    policy = load_sandbox_policy(POLICY)

    assert validate_sandbox_policy(definition.sandbox, policy, "docker")
    grant = resolve_env_grant(
        definition.sandbox,
        policy,
        {},
        definition.service.cpus,
        definition.service.memory_mb,
    )
    environments = grant.environments
    assert environments.work is None and environments.judge.build is None
    assert environments.judge.registries == ("docker.io",)
    assert (grant.reserved_cpus, grant.reserved_memory_mb) == (20, 59648)
    assert grant.reserved_disk_mb == 24576
    # The policy's header states the reservation it was reviewed with.
    assert "20 CPUs, 59648 MiB, 24576 MiB" in POLICY.read_text()
    # Both tasks fit at once (regex-log asks for 10 GiB of disk).
    judge = environments.judge
    assert judge.max_envs_live >= 2 and judge.max_disk_mb_live >= 10240 + 2048


def test_the_policy_refuses_a_build_or_work_environments(tmp_path):
    policy = load_sandbox_policy(POLICY)
    task = (SAMPLE / "task.toml").read_text()

    for index, (addition, refusal) in enumerate(
        (
            ("build = true\n", "judge.build: image build not approved"),
            (
                "\n[metadata.rsi_harness.sandbox.environments.work]\n"
                'network = ["public"]\n',
                "work: phase not approved",
            ),
        )
    ):
        copy = tmp_path / str(index)
        subprocess.run(["cp", "-a", str(SAMPLE), str(copy)], check=True)
        (copy / "task.toml").write_text(task + addition)
        sandbox = HarborTaskCompiler().compile(copy, CompileOptions()).sandbox
        with pytest.raises(SetupError, match=refusal):
            validate_sandbox_policy(sandbox, policy, "docker")


def test_the_work_image_pins_vllm_harbor_and_its_base():
    dockerfile = (SAMPLE / "environment" / "Dockerfile").read_text()
    pinned = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert "harbor==0.21.0" in pinned
    # vLLM's and Harbor's own dependencies are both resolved as of a fixed
    # date (Harbor's no earlier than Harbor 0.21.0's release, 2026-08-10).
    assert "--exclude-newer 2025-10-05T00:00:00Z vllm==0.11.0" in dockerfile
    assert "--exclude-newer 2026-10-01T00:00:00Z harbor==0.21.0 " in dockerfile
    assert "pip install --no-cache-dir harbor" not in dockerfile
    assert dockerfile.count("\nFROM python:3.13-slim-bookworm@sha256:") == 1
    assert "WORKDIR /workspace" in dockerfile
    for forbidden in ("docker.sock", "DOCKER_HOST", "docker-ce", "docker.io"):
        assert forbidden not in dockerfile


@pytest.mark.skipif(not TB2.is_dir(), reason="the TB2 source copy is not here")
def test_the_tasks_are_a_trivial_one_and_a_published_tb2_task():
    assert sorted(path.name for path in TASKS.iterdir()) == ["hello-file", "regex-log"]
    assert_same_tree(TB2 / "regex-log", TASKS / "regex-log")
    hello = tomllib.loads((TASKS / "hello-file" / "task.toml").read_text())
    assert hello["environment"]["docker_image"] == "python:3.13-slim-bookworm"
    # Harbor lists a dataset's task only with an environment directory.
    assert (TASKS / "hello-file" / "environment" / "Dockerfile").is_file()


def test_the_trivial_task_scores_its_solution_one_and_nothing_zero(tmp_path):
    def score(prepare):
        root = tmp_path / prepare
        (root / "logs" / "verifier").mkdir(parents=True)
        test = (TASKS / "hello-file" / "tests" / "test.sh").read_text()
        solve = (TASKS / "hello-file" / "solution" / "solve.sh").read_text()
        script = (solve if prepare == "solved" else "") + "\n" + test
        script = script.replace("/tmp/hello.txt", str(root / "hello.txt"))
        script = script.replace("/logs/verifier", str(root / "logs" / "verifier"))
        script = script.replace("set -eu\n", "")
        subprocess.run(["sh", "-c", script], check=True)
        return (root / "logs" / "verifier" / "reward.txt").read_text().strip()

    assert score("solved") == "1"
    assert score("untouched") == "0"


# -- the Work agent ---------------------------------------------------------------

STUB_HF = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_CALLS"
while [ $# -gt 0 ]; do
    [ "$1" = --local-dir ] && dir=$2
    shift
done
mkdir -p "$dir"
echo '{}' > "$dir/config.json"
[ -n "$STUB_NO_WEIGHTS" ] || echo weights > "$dir/model.safetensors"
"""
STUB_SUBMIT = '#!/bin/sh\necho submitted >> "$STUB_CALLS"\n'


def run_work(tmp_path, **environment):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in (("hf", STUB_HF), ("rsi-submit", STUB_SUBMIT)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", str(SAMPLE / "work" / "agent.sh")],
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STUB_CALLS": str(calls),
            "RSI_VLLM_CHECKPOINT": str(tmp_path / "workspace" / "checkpoint"),
            **environment,
        },
        capture_output=True,
        text=True,
    )
    return result, calls.read_text().splitlines() if calls.exists() else []


def test_the_work_agent_places_the_pinned_checkpoint_then_submits(tmp_path):
    result, calls = run_work(tmp_path)

    assert result.returncode == 0, result.stderr
    checkpoint = tmp_path / "workspace" / "checkpoint"
    assert calls == [
        "download Qwen/Qwen2.5-1.5B-Instruct --revision "
        f"989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --local-dir {checkpoint}",
        "submitted",
    ]
    assert "RSI-ACCEPTANCE checkpoint-download 0 Qwen/Qwen2.5-1.5B-Instruct@" in (
        result.stdout
    )
    assert "RSI-ACCEPTANCE checkpoint-files config.json model.safetensors" in (
        result.stdout
    )


def test_the_work_agent_never_submits_a_checkpoint_without_weights(tmp_path):
    result, calls = run_work(
        tmp_path, STUB_NO_WEIGHTS="1", RSI_VLLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    )

    assert result.returncode == 1
    # Another model than the default has no pinned commit of its own.
    assert calls[0].startswith("download Qwen/Qwen2.5-0.5B-Instruct --revision main ")
    assert "submitted" not in calls
    assert "RSI-ACCEPTANCE checkpoint-download 1 " in result.stdout


# -- the fixed procedure ----------------------------------------------------------

# vLLM's stand-in: `vllm serve CHECKPOINT ... --port P` answers /health,
# /v1/models, chat completions (counted, 100 prompt and 7 completion tokens
# each) and /metrics; STUB_VLLM_DIE exits before it is healthy.
STUB_VLLM = """#!{python}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
with open(os.environ["STUB_CALLS"], "a") as calls:
    calls.write(json.dumps({{"vllm": args,
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}}) + "\\n")
if os.environ.get("STUB_VLLM_DIE"):
    sys.exit(3)
value = lambda flag: args[args.index(flag) + 1]
served = 0

class Handler(BaseHTTPRequestHandler):
    def reply(self, body, kind="application/json"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply("")
        elif self.path == "/v1/models":
            self.reply(json.dumps({{"data": [{{"id": value("--served-model-name"),
                "root": args[1]}}]}}))
        elif self.path == "/metrics":
            self.reply(
                '# HELP vllm:request_success_total x\\n'
                'vllm:request_success_total{{finished_reason="stop"}} '
                f'{{served}}.0\\n'
                'vllm:request_success_total{{finished_reason="length"}} 0.0\\n'
                'vllm:request_success_total{{finished_reason="abort"}} 7.0\\n'
                f'vllm:prompt_tokens_total{{{{engine="0"}}}} {{100 * served}}.0\\n',
                "text/plain")

    def do_POST(self):
        global served
        self.rfile.read(int(self.headers["Content-Length"]))
        served += 1
        self.reply(json.dumps({{"model": value("--served-model-name"),
            "choices": [], "usage": {{"prompt_tokens": 100,
            "completion_tokens": 7}}}}))

    def log_message(self, format, *args):
        sys.stderr.write('INFO: "%s" %s\\n' % (self.requestline, args[1]))

HTTPServer(("127.0.0.1", int(value("--port"))), Handler).serve_forever()
"""
# Harbor's stand-in: records its argv, environment and start time, asks the
# model once per task and writes the trials a real job would (hello-file
# solved, regex-log not; STUB_ERROR makes regex-log raise in its
# environment, STUB_REGEX_LOG names regex-log's exception, STUB_TOKENS sets
# the agents' input tokens, STUB_BARE_BASE makes regex-log's agent ask
# outside its base path, STUB_SLEEP takes that long after the trials and
# records the end in STUB_CALLS.ended, STUB_HANG hangs after the first
# trial).
STUB_HARBOR = """#!{python}
import json, os, re, sys, time, tomllib, urllib.request
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["STUB_CALLS"], "a") as calls:
    calls.write(json.dumps({{"harbor": args, "started": time.time(),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "HARBOR_TELEMETRY": os.environ.get("HARBOR_TELEMETRY"),
        "LITELLM_LOCAL_MODEL_COST_MAP": os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"),
    }}) + "\\n")
values = lambda flag: [args[i + 1] for i, item in enumerate(args) if item == flag]
kwargs = dict(item.split("=", 1) for item in values("--ak"))

def name(task):
    config = tomllib.loads((task / "task.toml").read_text())
    return (config.get("task") or {{}}).get("name") or task.name

path = Path(values("--path")[0])
tasks = values("--include-task-name") or sorted(
    name(p) for p in path.iterdir() if (p / "environment").is_dir())
job = Path(values("--jobs-dir")[0]) / values("--job-name")[0]
job.mkdir(parents=True)
(job / "result.json").write_text("{{}}")
for task in tasks:
    base = kwargs["api_base"]
    if task == os.environ.get("STUB_BARE_BASE"):
        base = re.sub(r"/t/[^/]+/v1$", "/v1", base)
    urllib.request.urlopen(urllib.request.Request(
        base + "/chat/completions", data=b'{{"model": "rsi-checkpoint"}}',
        headers={{"Content-Type": "application/json"}}))
    error = None
    if task == "regex-log" and os.environ.get("STUB_ERROR"):
        error = {{"exception_type": "RuntimeError", "exception_message": "env"}}
    if task == "regex-log" and os.environ.get("STUB_REGEX_LOG"):
        error = {{"exception_type": os.environ["STUB_REGEX_LOG"],
            "exception_message": "x"}}
    trial = job / (task.replace("/", "-") + "__abc")
    trial.mkdir()
    (trial / "result.json").write_text(json.dumps({{
        "task_name": task, "trial_name": trial.name, "exception_info": error,
        "config": {{"environment": {{"import_path": values("--env")[0]}}}},
        "verifier_result": None if error else
            {{"rewards": {{"reward": 1.0 if "hello" in task else 0.0}}}},
        "agent_result": {{"n_input_tokens": int(os.environ.get("STUB_TOKENS", "10")),
            "n_output_tokens": 2}},
    }}))
    if os.environ.get("STUB_HANG"):
        time.sleep(600)
if os.environ.get("STUB_SLEEP"):
    time.sleep(float(os.environ["STUB_SLEEP"]))
    with open(os.environ["STUB_CALLS"] + ".ended", "a") as ended:
        ended.write(f"{{tasks[0]}} {{time.time()}}\\n")
"""
STUB_NVIDIA_SMI = """#!/bin/sh
# 0 MiB used before vLLM starts, 40000 MiB after.
used=0
[ -e "$STUB_SMI_COUNT" ] && used=40000
touch "$STUB_SMI_COUNT"
echo "0, {gpu}, NVIDIA H100 NVL, $used"
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def kill_tagged(tag: str) -> None:
    """SIGKILL every process whose environment holds ``tag`` (NAME=VALUE):
    test.sh, the metering proxy and the stand-ins, the setsid vLLM too."""
    needle = tag.encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if needle in environ.split(b"\0"):
            try:
                os.kill(int(entry.name), signal.SIGKILL)
            except OSError:
                pass


@pytest.fixture
def procedure(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (
        ("vllm", STUB_VLLM.format(python=sys.executable)),
        ("harbor", STUB_HARBOR.format(python=sys.executable)),
        ("nvidia-smi", STUB_NVIDIA_SMI.format(gpu=GPU)),
    ):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    out = tmp_path / "verifier"
    port = free_port()
    proxy_port = free_port()
    tag = f"STUB_CALLS={calls}"

    def run(**environment):
        env = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STUB_CALLS": str(calls),
            "STUB_SMI_COUNT": str(tmp_path / "smi-count"),
            "RSI_SANDBOX_PYTHONPATH": "/run/rsi-harness/sandbox/py",
            "RSI_HARNESS_EXPECTED_GPU_UUIDS": GPU,
            "RSI_VLLM_OUT": str(out),
            "RSI_VLLM_CHECKPOINT": "/workspace/checkpoint",
            "RSI_VLLM_PORT": str(port),
            "RSI_VLLM_PROXY_PORT": str(proxy_port),
            **environment,
        }
        env = {key: value for key, value in env.items() if value is not None}
        # test.sh stops the proxy and vLLM in its EXIT trap, which a timeout's
        # or an interrupt's SIGKILL skips: then kill whatever it left behind
        # (the teardown does too, after the test's own checks).
        process = subprocess.Popen(
            ["bash", str(SAMPLE / "tests" / "test.sh")],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=120)
        except BaseException:
            kill_tagged(tag)
            process.kill()
            process.wait()
            raise
        result = subprocess.CompletedProcess(
            process.args, process.returncode, stdout, stderr
        )
        recorded = (
            [json.loads(line) for line in calls.read_text().splitlines()]
            if calls.exists()
            else []
        )
        summary_path = out / "vllm-demo-summary.json"
        summary = (
            json.loads(summary_path.read_text()) if summary_path.exists() else None
        )
        return result, recorded, summary

    run.out = out
    run.port = port
    run.proxy_port = proxy_port
    run.ended = tmp_path / "calls.jsonl.ended"
    yield run
    kill_tagged(tag)


def listening(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def test_the_procedure_serves_runs_terminus_2_through_the_plugin_and_reports(
    procedure,
):
    result, calls, summary = procedure()

    assert result.returncode == 0, result.stderr
    [vllm, *harbors] = calls
    assert vllm["vllm"][:2] == ["serve", "/workspace/checkpoint"]
    served = vllm["vllm"]
    assert served[served.index("--host") + 1] == "127.0.0.1"
    assert served[served.index("--served-model-name") + 1] == "rsi-checkpoint"
    assert vllm["HF_HUB_OFFLINE"] == "1"
    port = procedure.port
    # A Harbor job of one trial per task, its agent's model the task's own
    # base path at the metering proxy.
    jobs = {}
    for harbor in harbors:
        argv = harbor["harbor"]
        assert argv[:7] == [
            "run",
            "--env",
            PLUGIN,
            "--agent",
            "terminus-2",
            "--model",
            "hosted_vllm/rsi-checkpoint",
        ]
        task = argv[argv.index("--include-task-name") + 1]
        key = task.replace("/", "-")
        jobs[task] = argv
        proxy = f"http://127.0.0.1:{procedure.proxy_port}"
        assert f"api_base={proxy}/t/{key}/v1" in argv
        assert argv[argv.index("--path") + 1] == str(TASKS)
        assert argv[argv.index("--jobs-dir") + 1] == str(
            procedure.out / "harbor-jobs" / "terminus-2"
        )
        assert argv[argv.index("--job-name") + 1] == key
        assert argv[argv.index("--n-concurrent") + 1] == "1"
        assert harbor["PYTHONPATH"] == "/run/rsi-harness/sandbox/py"
        assert harbor["HARBOR_TELEMETRY"] == "0"
        assert harbor["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert sorted(jobs) == ["regex-log", "rsi/hello-file"]
    argv = jobs["rsi/hello-file"]
    for task in jobs:
        assert f"harbor run terminus-2 {task}: exit 0" in result.stdout
    assert "harbor run terminus-2: exit 0" in result.stdout
    # vLLM and the proxy are stopped before the procedure ends.
    assert not listening(port) and not listening(procedure.proxy_port)

    assert summary["demo_ok"] is True
    assert summary["reward"] == 0.5
    assert json.loads((procedure.out / "reward.json").read_text()) == {"reward": 0.5}
    assert summary["vllm"]["serving_gpus"] == [GPU]
    assert summary["vllm"]["models"] == [
        {"id": "rsi-checkpoint", "root": "/workspace/checkpoint"}
    ]
    # Aborted requests are counted apart, never as served.
    assert summary["completions"]["requests_served"] == 2
    assert summary["completions"]["requests_aborted"] == 7
    assert summary["completions"]["prompt_tokens"] == 200.0
    assert summary["completions"]["access_log_200"] == 2
    assert [(item["task"], item["solved"]) for item in summary["trials"]] == [
        ("regex-log", False),
        ("rsi/hello-file", True),
    ]
    assert summary["vllm"]["checkpoint"] == "/workspace/checkpoint"
    assert summary["vllm"]["served_checkpoint"] is True
    assert {item["environment"] for item in summary["trials"]} == {PLUGIN}
    assert all(item["requested"] for item in summary["trials"])
    assert summary["infra_errors"] == [] and summary["model_outcomes"] == []
    # What is left of the default 3600 s budget less the 300 s reserve.
    limits = re.findall(r"harbor run terminus-2 \S+: limit (\d+)s", result.stdout)
    assert len(limits) == 2 and all(3200 <= int(limit) <= 3300 for limit in limits)

    # One usage line per request, each for the trial it came from; the
    # summary's tokens per trial and the score against them.
    usage = [
        json.loads(line)
        for line in (procedure.out / "usage.jsonl").read_text().splitlines()
    ]
    assert sorted(
        (item["trial"], item["model"], item["status"], item["completion_tokens"])
        for item in usage
    ) == [
        ("regex-log", "rsi-checkpoint", 200, 7),
        ("rsi-hello-file", "rsi-checkpoint", 200, 7),
    ]
    for item in summary["trials"]:
        metered = dict(item["metered"])
        assert metered.pop("latency_ms") >= 0
        assert metered == {
            "requests": 1,
            "failed": 0,
            "prompt_tokens": 100,
            "completion_tokens": 7,
        }
    assert summary["accuracy_vs_tokens"] == {
        "trials": 2,
        "solved": 1,
        "solved_fraction": 0.5,
        "requests": 2,
        "prompt_tokens": 200,
        "completion_tokens": 14,
        "completion_tokens_per_solved": 14.0,
        "per_task": {
            "regex-log": {"solved": False, "completion_tokens": 7},
            "rsi/hello-file": {"solved": True, "completion_tokens": 7},
        },
    }

    # Harbor 0.21.0's own CLI resolves this argv to terminus-2 against the
    # endpoint, the plugin as environment, and typed agent kwargs.
    printed = subprocess.run(
        [str(Path(sys.executable).parent / "harbor"), *argv, "--print-config"],
        capture_output=True,
        text=True,
        env={**os.environ, "HARBOR_TELEMETRY": "0"},
        check=True,
    )
    config = json.loads(printed.stdout)
    assert config["environment"] == {"import_path": PLUGIN}
    [agent] = config["agents"]
    assert agent["name"] == "terminus-2"
    assert agent["model_name"] == "hosted_vllm/rsi-checkpoint"
    kwargs = agent["kwargs"]
    assert kwargs["api_base"] == f"{proxy}/t/rsi-hello-file/v1"
    assert kwargs["max_turns"] == 12 and kwargs["record_terminal_session"] is False
    assert set(kwargs["model_info"]) == {
        "max_input_tokens",
        "max_output_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
    }


def test_the_procedure_narrows_the_tasks(procedure):
    result, calls, summary = procedure(RSI_VLLM_TASKS="rsi/hello-file")

    assert result.returncode == 0, result.stderr
    [_, harbor] = calls
    argv = harbor["harbor"]
    assert argv[argv.index("--include-task-name") + 1] == "rsi/hello-file"
    assert summary["expected_tasks"] == ["rsi/hello-file"]
    assert summary["reward"] == 1.0 and summary["demo_ok"] is True
    assert summary["accuracy_vs_tokens"]["completion_tokens_per_solved"] == 7.0


def test_an_unknown_task_runs_no_agent(procedure):
    result, calls, summary = procedure(RSI_VLLM_TASKS="nope")

    assert result.returncode == 0, result.stderr
    assert [list(call) for call in calls] == [["vllm", "HF_HUB_OFFLINE"]]
    assert "no task named nope" in summary["error"]
    assert summary["reward"] == 0.0 and summary["demo_ok"] is False


@pytest.mark.parametrize("concurrency", ["1", "2"])
def test_at_most_the_concurrency_of_jobs_run_at_once(procedure, concurrency):
    result, calls, _ = procedure(RSI_VLLM_CONCURRENCY=concurrency, STUB_SLEEP="2")

    assert result.returncode == 0, result.stderr
    started = sorted(call["started"] for call in calls[1:])
    ended = sorted(
        float(line.split()[1]) for line in procedure.ended.read_text().splitlines()
    )
    assert len(started) == len(ended) == 2
    # One at a time: the second starts after the first ended; two: at once.
    assert (started[1] >= ended[0]) is (concurrency == "1")


def test_a_metering_proxy_that_never_comes_up_runs_no_agent(procedure):
    # Its port is vLLM's: it cannot listen.
    result, calls, summary = procedure(RSI_VLLM_PROXY_PORT=str(procedure.port))

    assert result.returncode == 0, result.stderr
    assert [list(call) for call in calls] == [["vllm", "HF_HUB_OFFLINE"]]
    assert "exited before it was healthy" in result.stderr
    assert "harbor run terminus-2: exit 1" in result.stdout
    assert summary["demo_ok"] is False and summary["harbor_exit"] == 1
    assert (
        "Address already in use" in (procedure.out / "vllm" / "proxy.log").read_text()
    )


def test_a_request_outside_its_trials_base_path_fails_the_demo(procedure):
    result, _, summary = procedure(STUB_BARE_BASE="regex-log")

    assert result.returncode == 0, result.stderr
    assert summary["demo_ok"] is False
    assert summary["infra_errors"] == [
        "regex-log__abc: its agent's requests were not metered",
        "metering: 1 request(s) outside every trial's base path",
    ]
    [regex] = [item for item in summary["trials"] if item["task"] == "regex-log"]
    assert regex["metered"]["requests"] == 0


def test_an_infrastructure_error_fails_the_demo(procedure):
    result, _, summary = procedure(STUB_ERROR="1")

    assert result.returncode == 0, result.stderr
    assert summary["demo_ok"] is False
    assert summary["infra_errors"] == ["regex-log__abc: RuntimeError: env"]


def test_a_verifier_outcome_the_model_caused_is_completed_and_listed_apart(
    procedure,
):
    result, _, summary = procedure(STUB_REGEX_LOG="RewardFileNotFoundError")

    assert result.returncode == 0, result.stderr
    assert summary["demo_ok"] is True and summary["infra_errors"] == []
    assert summary["model_outcomes"] == [
        "regex-log__abc: verifier:RewardFileNotFoundError"
    ]
    [regex] = [item for item in summary["trials"] if item["task"] == "regex-log"]
    assert regex["completed"] and not regex["solved"]


def test_a_trial_whose_agent_never_asked_the_model_fails_the_demo(procedure):
    result, _, summary = procedure(STUB_TOKENS="0")

    assert result.returncode == 0, result.stderr
    assert summary["demo_ok"] is False
    assert not any(item["completed"] for item in summary["trials"])


def test_harbor_gets_the_budget_left_and_a_hang_still_reports(procedure):
    # 305 s of budget less the 300 s reserve: Harbor gets the 10 s floor.
    result, _, summary = procedure(STUB_HANG="1", RSI_VLLM_BUDGET_SEC="305")

    assert result.returncode == 0, result.stderr
    assert "harbor run terminus-2 regex-log: limit 10s" in result.stdout
    assert "harbor run terminus-2 rsi/hello-file: limit 10s" in result.stdout
    assert "harbor run terminus-2: exit 124" in result.stdout
    assert summary["harbor_exit"] == 124 and summary["demo_ok"] is False
    assert (
        "harbor run stopped by its time limit (exit 124)" in (summary["infra_errors"])
    )
    # Each job's trial was written before it hung: the one solved counts.
    assert json.loads((procedure.out / "reward.json").read_text()) == {"reward": 0.5}


def test_the_procedure_budget_is_the_verifier_timeout():
    text = (SAMPLE / "tests" / "test.sh").read_text()
    config = tomllib.loads((SAMPLE / "task.toml").read_text())

    budget = config["verifier"]["timeout_sec"]
    assert f"budget=${{RSI_VLLM_BUDGET_SEC:-{budget}}}\n" in text
    # Work's submit waits for the round: its time covers the round.
    assert config["agent"]["timeout_sec"] >= budget + 1800


def test_a_vllm_that_never_becomes_healthy_runs_no_agent(procedure):
    result, calls, summary = procedure(STUB_VLLM_DIE="1")

    assert result.returncode == 0, result.stderr
    assert [list(call) for call in calls] == [["vllm", "HF_HUB_OFFLINE"]]
    assert "exited before it was healthy" in result.stderr
    assert summary["demo_ok"] is False and summary["reward"] == 0.0
    assert summary["vllm"]["healthy"] is False


def test_the_procedure_needs_the_sandbox_endpoint(procedure):
    result, calls, summary = procedure(RSI_SANDBOX_PYTHONPATH=None)

    assert result.returncode != 0 and calls == [] and summary is None
    assert "no sandbox endpoint" in result.stderr


# -- demo_report ------------------------------------------------------------------


def trial(**changes):
    """A Harbor 0.21 trial result.json of the plugin, its agent having asked
    the model."""
    return {
        "trial_name": "t",
        "task_name": "x",
        "config": {"environment": {"import_path": PLUGIN}},
        "agent_result": {"n_input_tokens": 120},
        **changes,
    }


@pytest.mark.parametrize(
    ("changes", "outcome", "completed", "infra"),
    [
        ({"verifier_result": {"rewards": {"reward": 0.0}}}, "verified", True, False),
        (
            {
                "exception_info": {"exception_type": "AgentTimeoutError"},
                "verifier_result": {"rewards": {"reward": 0.0}},
            },
            "agent:AgentTimeoutError",
            True,
            False,
        ),
        (
            {"exception_info": {"exception_type": "ContextLengthExceededError"}},
            "agent:ContextLengthExceededError",
            True,
            False,
        ),
        # What the agent left in its environment can stop the verifier.
        (
            {"exception_info": {"exception_type": "VerifierTimeoutError"}},
            "verifier:VerifierTimeoutError",
            True,
            False,
        ),
        (
            {"exception_info": {"exception_type": "RewardFileNotFoundError"}},
            "verifier:RewardFileNotFoundError",
            True,
            False,
        ),
        (
            {"exception_info": {"exception_type": "RuntimeError"}},
            "infra:RuntimeError",
            False,
            True,
        ),
        (
            {"exception_info": {"exception_type": "EnvironmentStartTimeoutError"}},
            "infra:EnvironmentStartTimeoutError",
            False,
            True,
        ),
        (
            {"exception_info": {"exception_type": "DownloadVerifierDirError"}},
            "infra:DownloadVerifierDirError",
            False,
            True,
        ),
        ({"verifier_result": None}, "unverified", False, False),
    ],
)
def test_a_trial_is_complete_unless_the_infrastructure_failed_it(
    changes, outcome, completed, infra
):
    found = demo_report().trial_outcome(trial(**changes), PLUGIN)

    assert (found["outcome"], found["completed"], found["infra_error"]) == (
        outcome,
        completed,
        infra,
    )


@pytest.mark.parametrize(
    "changes",
    [
        # Its agent never asked the model (an agent timeout included).
        {"agent_result": {"n_input_tokens": 0}},
        {"agent_result": None},
        {
            "agent_result": {"n_input_tokens": 0},
            "exception_info": {"exception_type": "AgentTimeoutError"},
        },
        # Not in a broker-created environment.
        {"config": {"environment": {"type": "docker", "import_path": None}}},
    ],
)
def test_a_trial_counts_only_through_the_plugin_with_model_requests(changes):
    found = demo_report().trial_outcome(
        trial(verifier_result={"rewards": {"reward": 1.0}}, **changes), PLUGIN
    )

    assert found["completed"] is False and found["solved"] is True


def test_a_trial_outside_the_plugin_is_an_infrastructure_error(tmp_path):
    module = demo_report()
    job = tmp_path / "job"
    usage = tmp_path / "usage.jsonl"
    for task, environment in (("rsi/hello-file", PLUGIN), ("regex-log", None)):
        key = task.replace("/", "-")
        directory = job / key / f"{key}__abc"
        directory.mkdir(parents=True)
        with usage.open("a") as log:
            log.write(json.dumps({"trial": key, "status": 200}) + "\n")
        result = trial(
            trial_name=directory.name,
            task_name=task,
            config={"environment": {"type": "docker", "import_path": environment}},
            verifier_result={"rewards": {"reward": 1.0}},
        )
        (directory / "result.json").write_text(json.dumps(result))

    summary = module.report(
        tmp_path, job, TASKS, [], 0, checkpoint="/c", environment=PLUGIN, usage=usage
    )

    assert summary["infra_errors"] == [f"regex-log__abc: ran in docker, not {PLUGIN}"]
    assert summary["demo_ok"] is False


def test_vllm_counts_only_on_a_gpu_the_harness_gave_the_judge(tmp_path, monkeypatch):
    module = demo_report()
    (tmp_path / "healthy").touch()
    (tmp_path / "gpu-before.csv").write_text(f"0, {GPU}, H100, 0\n")
    (tmp_path / "gpu-after.csv").write_text(f"0, {GPU}, H100, 40000\n")

    def on_judge_gpu():
        return module.vllm_evidence(tmp_path, "/workspace/checkpoint")["on_judge_gpu"]

    monkeypatch.setenv("RSI_HARNESS_EXPECTED_GPU_UUIDS", GPU)
    assert on_judge_gpu() is True
    monkeypatch.setenv("RSI_HARNESS_EXPECTED_GPU_UUIDS", "GPU-other")
    assert on_judge_gpu() is False
    # A Judge the Harness named no GPU for proves nothing.
    monkeypatch.setenv("RSI_HARNESS_EXPECTED_GPU_UUIDS", "")
    assert on_judge_gpu() is False
    # Too little memory is no model on the GPU.
    monkeypatch.setenv("RSI_HARNESS_EXPECTED_GPU_UUIDS", GPU)
    (tmp_path / "gpu-after.csv").write_text(f"0, {GPU}, H100, 500\n")
    assert on_judge_gpu() is False


def test_vllm_must_serve_exactly_the_workdir_checkpoint(tmp_path):
    module = demo_report()

    def served(*roots):
        (tmp_path / "models.json").write_text(
            json.dumps({"data": [{"id": "m", "root": root} for root in roots]})
        )
        evidence = module.vllm_evidence(tmp_path, "/workspace/checkpoint")
        return evidence["served_checkpoint"]

    assert served("/workspace/checkpoint") is True
    assert served("/root/.cache/huggingface/hub/qwen") is False
    assert served("/workspace/checkpoint", "/other") is False
    assert served() is False


def test_metric_sums_every_series_or_some():
    module = demo_report()
    text = (
        "# TYPE vllm:request_success_total counter\n"
        'vllm:request_success_total{finished_reason="stop"} 3.0\n'
        'vllm:request_success_total{finished_reason="length"} 2.0\n'
        'vllm:request_success_total{finished_reason="abort"} 4.0\n'
        "vllm:request_success_created 17.0\n"
    )
    name = "vllm:request_success_total"

    assert module.metric_sum(text, name) == 9.0
    assert module.metric_sum(text, name, exclude=module.ABORTED) == 5.0
    assert module.metric_sum(text, name, only=module.ABORTED) == 4.0
    # Every request aborted: none served.
    aborted = 'vllm:request_success_total{finished_reason="abort"} 4.0\n'
    assert module.metric_sum(aborted, name, exclude=module.ABORTED) == 0.0
    assert module.metric_sum(text, "vllm:generation_tokens_total") is None


def test_a_missing_or_unknown_task_is_an_infrastructure_error(tmp_path):
    module = demo_report()
    job = tmp_path / "job"
    job.mkdir()

    def report(only):
        return module.report(
            tmp_path, job, TASKS, only, 0, checkpoint="/c", environment=PLUGIN
        )

    summary = report([])
    assert summary["demo_ok"] is False
    assert "expected one each of" in summary["infra_errors"][0]
    unknown = report(["nope"])
    assert unknown["reward"] == 0.0 and "no task named nope" in unknown["error"]


def test_the_judge_procedure_writes_its_reward_and_summary_to_the_verifier_logs():
    text = (SAMPLE / "tests" / "test.sh").read_text()
    assert "out=${RSI_VLLM_OUT:-/logs/verifier}" in text
    assert (
        'PYTHONPATH=$RSI_SANDBOX_PYTHONPATH timeout --kill-after=30 "${limit}s"'
        " harbor run \\\n" in text
    )
    assert f"plugin={PLUGIN}\n" in text and '--env "$plugin"' in text
    assert '--checkpoint "$checkpoint" --environment "$plugin"' in text
    assert '--reward-json "$out/reward.json"' in text
    assert '--summary "$out/vllm-demo-summary.json"' in text
