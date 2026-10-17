"""rsi-agent, run as the current (non-root) user against fake Agent CLIs.

The fake ``codex``, ``claude``, ``qwen``, ``opencode``, ``pi``, ``kimi`` and the
ZCode bundle (node modules, as the real ones are) record their argv, stdin or
prompt, environment and working directory, then answer from the schema they
were given. The uid drop and the
Judge filesystem run in the Docker tests
(tests/integration/test_judge_model_access_docker.py), the real CLIs in
tests/integration/test_agent_one_shot_docker.py.
"""

from __future__ import annotations

import ctypes
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import rsi_loop_oneshot
from rsi_harness.judge_agent import (
    JUDGE_AGENT_SLOTS,
    JUDGE_AGENT_UID,
    judge_agent_source,
)
from rsi_harness.models import AgentPlan, NetworkPolicy
from rsi_harness.runtime.judge_model_access import build_launch_spec
from rsi_loop.harness.agent.base import NODE_BIN
from rsi_loop.harness.config import RSILoopConfig

PROGRAM = Path(__file__).parents[2] / "src/rsi_harness/judge_agent/rsi_agent.py"
API_KEY = "sk-synthetic-rsi-agent-key-91c2"
MODELS = {
    "codex": "gpt-test",
    "claude-code": "claude-test",
    "qwen-code": "qwen-test",
    "kimi-code": "kimi-test",
    "opencode": "opencode-test",
    "zcode": "zcode-test",
    "pi": "pi-test",
}
NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
SCHEMA = {
    "type": "object",
    "required": ["verdict"],
    "properties": {"verdict": {"type": "string", "enum": ["clear"]}},
}

FAKE_CLI = r"""#!{python}
import json, os, subprocess, sys, time
args = sys.argv[1:]
cli = os.path.basename(sys.argv[0])
settings = os.path.join(os.environ.get("HOME", ""), ".qwen", "settings.json")
pi_dir = os.environ.get("PI_CODING_AGENT_DIR", "")
record = {{"cli": cli, "argv": args,
          "env": dict(os.environ), "cwd": os.getcwd(), "stdin": sys.stdin.read(),
          "uid": os.getuid(),
          "qwen_settings": json.load(open(settings)) if cli == "qwen" else None,
          "pi_config": {{name: json.load(open(os.path.join(pi_dir, name)))
                        for name in ("settings.json", "models.json")}}
                       if cli == "pi" else None,
          "home_files": sorted(
              os.path.relpath(os.path.join(root, name), os.environ.get("HOME", "/"))
              for root, _, names in os.walk(os.environ.get("HOME", "/nonexistent"))
              for name in names)}}
with open(os.environ["FAKE_RECORD"], "a") as log:
    log.write(json.dumps(record) + "\n")
mode = os.environ.get("FAKE_MODE", "ok")
schema = None
if "--output-schema" in args:
    schema = json.load(open(args[args.index("--output-schema") + 1]))
if "--json-schema" in args:
    value = args[args.index("--json-schema") + 1]
    # Qwen takes "@FILE", Claude the schema itself.
    schema = json.load(open(value[1:])) if value.startswith("@") else json.loads(value)
key = os.environ.get("OPENAI_API_KEY") or os.environ.get("RSI_AGENT_API_KEY", "")
answer = {{"verdict": "clear", "key": key}}
if mode == "fail":
    sys.stderr.write("model said no with key " + key + "\n")
    sys.exit(7)
if mode == "probe":
    # What a call's Agent may read and write: one more record line.
    def attempt(path, how):
        try:
            if how == "list":
                os.listdir(path)
            else:
                open(path, how).close()
            return "ok"
        except OSError as error:
            return error.strerror
    probe = json.loads(os.environ["FAKE_PROBE"])
    found = {{name: attempt(path, how) for name, (path, how) in probe.items()}}
    with open(os.environ["FAKE_RECORD"], "a") as log:
        log.write(json.dumps({{"probe": found}}) + "\n")
if mode == "daemon":
    # A double-forked daemon in a new session outlives the Agent's group.
    if os.fork() == 0:
        os.setsid()
        if os.fork() == 0:
            open(os.environ["FAKE_CHILD"], "w").write(str(os.getpid()))
            time.sleep(120)
        os._exit(0)
    while not open(os.environ["FAKE_CHILD"]).read():
        time.sleep(0.01)
if mode == "sleep":
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen(["sleep", "60"])
    open(os.environ["FAKE_CHILD"], "w").write(str(child.pid))
    time.sleep(60)
if cli == "qwen":
    # Its stream-json events, the result last.
    print(json.dumps({{"type": "system", "subtype": "init"}}))
    print(json.dumps({{"type": "assistant", "text": "thinking aloud"}}))
    result = {{"type": "result", "is_error": mode == "is_error",
              "result": "plain text answer" if mode != "is_error" else "boom"}}
    if schema is not None:
        result["structured_result"] = answer
    print(json.dumps(result))
    sys.exit(1 if mode == "is_error" else 0)
if cli == "opencode":
    # Its --format json events, compact as JSON.stringify prints them: every
    # finished text part, the final one last.
    def event(kind, **fields):
        fields = dict(type=kind, timestamp=1, sessionID="ses_1", **fields)
        print(json.dumps(fields, separators=(",", ":")))
    def text(value):
        event("text", part={{"id": "prt_1", "type": "text", "text": value}})
    event("step_start", part={{"type": "step-start"}})
    text("thinking aloud")
    event("tool_use", part={{"type": "tool", "tool": "bash"}})
    if mode == "error":
        event("error", error={{"name": "APIError",
                              "data": {{"message": "bad key " + key}}}})
        sys.exit(1)
    text(json.dumps(answer) if "JSON Schema" in record["stdin"]
         else "plain text answer")
    event("step_finish", part={{"type": "step-finish"}})
    sys.exit(0)
if cli == "pi":
    # Print mode: each text block of the final response, a newline after
    # each; a failed response is a message on stderr and status 1.
    if mode == "error":
        sys.stderr.write("401: bad key " + key + "\n")
        sys.exit(1)
    if mode != "empty":
        print(json.dumps(answer) if "JSON Schema" in record["stdin"]
              else "plain text answer\n\nwith a second paragraph")
    sys.exit(0)
if cli == "claude":
    result = {{"type": "result", "is_error": mode == "is_error",
              "result": "plain text answer" if mode != "is_error" else "boom"}}
    if schema is not None and mode != "result_only":
        result["structured_output"] = answer
    print(json.dumps(result))
    sys.exit(1 if mode == "is_error" else 0)
output = args[args.index("--output-last-message") + 1]
if mode == "symlink":
    os.symlink("/etc/passwd", output)
elif mode == "empty":
    open(output, "w").close()
elif mode != "nooutput":
    open(output, "w").write(json.dumps(answer))
"""

# Kimi's print mode takes the prompt only as --prompt; its stream-json prints
# one line per message, then meta lines.
FAKE_KIMI = r"""
import fs from "node:fs";
import path from "node:path";
const args = process.argv.slice(2);
const flag = (name) => args[args.indexOf(name) + 1];
const prompt = args.find((arg) => arg.startsWith("--prompt=")).slice(9);
const home = process.env.KIMI_CODE_HOME;
const record = {
  cli: path.basename(process.argv[1]),
  argv: args,
  cmdline: fs.readFileSync("/proc/self/cmdline", "utf8"),
  env: process.env,
  cwd: process.cwd(),
  prompt,
  config: fs.readFileSync(path.join(home, "config.toml"), "utf8"),
  profile: fs.readFileSync(flag("--agent-file"), "utf8"),
};
fs.appendFileSync(process.env.FAKE_RECORD, JSON.stringify(record) + "\n");
if (process.env.FAKE_MODE === "fail") {
  process.stderr.write("model said no with key " + process.env.OPENAI_API_KEY + "\n");
  process.exit(7);
}
const answer = prompt.includes("JSON Schema")
  ? JSON.stringify({ verdict: "clear", key: process.env.OPENAI_API_KEY })
  : "plain text answer";
const line = (value) => console.log(JSON.stringify(value));
line({ role: "meta", type: "system.version", version: "test" });
line({ role: "assistant", content: "thinking aloud",
       tool_calls: [{ type: "function", id: "c1",
                      function: { name: "Bash", arguments: "{}" } }] });
line({ role: "tool", tool_call_id: "c1", content: "tool output" });
line({ role: "assistant", content: answer });
line({ role: "meta", type: "session.resume_hint", content: "kimi -r s1" });
"""

# ZCode's print mode, loaded as its launcher loads the bundle: it takes the
# prompt only as --prompt and prints one JSON object, indented.
FAKE_ZCODE = r"""
const fs = require("node:fs");
const path = require("node:path");
const args = process.argv.slice(2);
const prompt = args.find((arg) => arg.startsWith("--prompt=")).slice(9);
const home = process.env.HOME;
const files = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap(
  (entry) => entry.isDirectory()
    ? files(path.join(dir, entry.name))
    : [path.relative(home, path.join(dir, entry.name))]);
const provider = process.env.ZCODE_PERSONAL_PROVIDER_CONFIG_FILE;
const read = (file) => JSON.parse(fs.readFileSync(file, "utf8"));
const record = {
  cli: path.basename(process.argv[1]),
  argv: args,
  cmdline: fs.readFileSync("/proc/self/cmdline", "utf8"),
  env: process.env,
  cwd: process.cwd(),
  prompt,
  home_files: files(home).sort(),
  cli_config: read(path.join(home, ".zcode", "cli", "config.json")),
  provider_file: provider,
  provider_mode: fs.statSync(provider).mode & 0o777,
  provider: read(provider),
};
fs.appendFileSync(process.env.FAKE_RECORD, JSON.stringify(record) + "\n");
const key = process.env.RSI_AGENT_API_KEY;
if (process.env.FAKE_MODE === "fail") {
  process.stderr.write("Error: Turn execution failed with key " + key + "\n");
  process.exit(1);
}
const response = process.env.FAKE_MODE === "empty" ? ""
  : prompt.includes("JSON Schema") ? JSON.stringify({ verdict: "clear", key })
  : "plain text answer";
const result = { sessionId: "sess_1", response, eventCount: 3 };
process.stdout.write(JSON.stringify(result, null, 2));
"""


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    for name in ("codex", "claude", "qwen", "opencode", "pi"):
        path = directory / name
        path.write_text(FAKE_CLI.format(python=sys.executable))
        path.chmod(0o755)
    if NODE is not None:
        # As npm installs it: the launcher links to the package's main module.
        main = tmp_path / "kimi-code" / "main.mjs"
        main.parent.mkdir()
        main.write_text(FAKE_KIMI)
        (directory / "kimi").symlink_to(main)
        (directory / "node").symlink_to(NODE)
        bundle = tmp_path / "zcode" / "zcode.cjs"
        bundle.parent.mkdir()
        bundle.write_text(FAKE_ZCODE)
    return directory


class Judge:
    """A state root with a launch spec, as the Harness leaves it in a Judge."""

    def __init__(
        self, tmp_path: Path, fake_bin: Path, root: Path | None = None
    ) -> None:
        self.root = root or tmp_path / "state"
        self.private = self.root / "private"
        self.private.mkdir(parents=True)
        self.root.chmod(0o711)
        self.private.chmod(0o700)
        self.fake_bin = fake_bin
        self.record = tmp_path / "record.jsonl"
        self.tmp = tmp_path
        self.store = tmp_path / "store"

    def spec(
        self,
        agent: str = "codex",
        *,
        mode: str = "ok",
        internet: bool = False,
        effort: str | None = "high",
        base_url: str | None = None,
        credentials: bool = False,
        **extra,
    ) -> dict:
        config = RSILoopConfig(
            agent_api_key=API_KEY,
            agent_api_base_url=base_url,
            agent_context_window="131072",
            agent_max_output_tokens="16384",
            agent_extra_env={"FAKE_RECORD": str(self.record), "FAKE_MODE": mode},
        )
        plan = AgentPlan(
            name=agent,
            model=MODELS[agent],
            reasoning_effort=effort,
            network=NetworkPolicy(mode="public" if internet else "no-network"),
        )
        oneshot = rsi_loop_oneshot(config, plan)
        spec = json.loads(build_launch_spec(oneshot, None, {API_KEY}))
        if NODE is not None:
            # The pinned node an Agent image installs is this host's node.
            spec["command"] = spec["command"].replace(NODE_BIN, NODE)
        spec["environment"].update(extra)
        if credentials:
            login = self.store / ".codex" / "auth.json"
            login.parent.mkdir(parents=True, exist_ok=True)
            login.write_text('{"tokens": {"access_token": "LOGIN-SECRET-77"}}')
            spec["credentials"] = [{"source": str(login), "target": ".codex/auth.json"}]
            spec["redact"].append("LOGIN-SECRET-77")
        spec["protected"] = [*spec["protected"], str(self.tmp / "verifier")]
        # The fake CLIs' own records: the only files outside its job a
        # Landlock-confined call may write (the Harness grants none).
        writable = [
            self.record,
            *(Path(extra[k]) for k in ("FAKE_CHILD",) if k in extra),
        ]
        for path in writable:
            path.touch()
        spec["writable"] = [str(path) for path in writable]
        (self.private / "launch.json").write_text(json.dumps(spec))
        return spec

    def run(self, *args: str, env: dict | None = None, timeout: float = 60):
        environment = {
            "PATH": f"{self.fake_bin}:/usr/bin:/bin",
            "RSI_AGENT_STATE_ROOT": str(self.root),
            "CALLER_ONLY_SECRET": "caller-secret-x",
            **(env or {}),
        }
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(PROGRAM), *args],
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def calls(self) -> list[dict]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]


@pytest.fixture
def judge(tmp_path: Path, fake_bin: Path) -> Judge:
    return Judge(tmp_path, fake_bin)


def _filesystem(path: Path | str) -> str:
    return subprocess.run(
        ["stat", "-f", "-c", "%T", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def tmpfs_judge(tmp_path: Path, fake_bin: Path):
    """A Judge whose state root, and so each call's HOME, is a tmpfs.

    As /run/rsi-agent is in a real Judge; ZCode's one-shot keeps its key there.
    """
    if not Path("/dev/shm").is_dir() or _filesystem("/dev/shm") != "tmpfs":
        pytest.skip("/dev/shm is not a tmpfs")
    root = Path(tempfile.mkdtemp(prefix="rsi-agent-test.", dir="/dev/shm"))
    try:
        yield Judge(tmp_path, fake_bin, root / "state")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _inputs(tmp_path: Path, prompt: str = "Review the evidence.\n"):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text(prompt)
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(SCHEMA, indent=1))
    workdir = tmp_path / "jobs" / "one"
    workdir.mkdir(parents=True)
    (workdir / "evidence.txt").write_text("evidence\n")
    return prompt_file, schema_file, workdir


def _exec(judge: Judge, tmp_path: Path, *extra: str, schema=True, workdir=True):
    prompt, schema_file, work = _inputs(tmp_path)
    args = ["exec", "--prompt-file", str(prompt), "--output", str(tmp_path / "out")]
    if schema:
        args += ["--schema", str(schema_file)]
    if workdir:
        args += ["--workdir", str(work)]
    return judge.run(*args, *extra), work


# -- the Agent's own command ------------------------------------------------------


def test_codex_runs_its_own_one_shot_command(judge, tmp_path):
    judge.spec("codex", base_url="http://model.test:8000/v1")

    result, work = _exec(judge, tmp_path, "--log", str(tmp_path / "agent.log"))

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    argv = call["argv"]
    assert argv[0] == "exec"
    assert argv[-1] == "-"
    for flag in (
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--dangerously-bypass-approvals-and-sandbox",
    ):
        assert flag in argv
    overrides = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]
    assert 'model_reasoning_effort="high"' in overrides
    # As in Work: bash strips the quotes; Codex reads a bare string.
    assert "web_search=disabled" in overrides
    assert "features.hooks=false" in overrides
    # Goal tools fail in an ephemeral thread; the call never offers them.
    assert "features.goals=false" in overrides
    assert "project_doc_max_bytes=0" in overrides
    # RSI_AGENT_API_BASE_URL reaches Codex as the launch-time provider; the
    # URL itself stays in the environment.
    assert "model_provider=rsi_proxy" in overrides
    assert 'model_providers.rsi_proxy.base_url="http://model.test:8000/v1"' in (
        overrides
    )
    assert argv[argv.index("--model") + 1] == "gpt-test"
    # The schema reaches Codex natively, byte for byte.
    schema_path = Path(argv[argv.index("--output-schema") + 1])
    assert schema_path.name == "schema.json"
    assert call["stdin"] == "Review the evidence.\n"
    assert call["cwd"] == str(work)
    # The answer is the final message, with the key redacted.
    answer = json.loads((tmp_path / "out").read_text())
    assert answer == {"verdict": "clear", "key": "[REDACTED]"}
    log = (tmp_path / "agent.log").read_text()
    assert "--- stdout ---" in log and API_KEY not in log
    # The job directory is gone; the workdir is handed back.
    assert list((judge.root / "jobs").iterdir()) == []
    # Not root: no Agent user, but the call still claimed one of the slots
    # that cap concurrent calls.
    assert sorted(path.name for path in judge.private.iterdir()) == sorted(
        ["launch.json"] + [f"slot-{n}.lock" for n in range(JUDGE_AGENT_SLOTS)]
    )
    assert work.stat().st_mode & 0o777 == 0o755 or os.geteuid() != 0


def test_the_agent_environment_is_exactly_the_runs_agent_environment(judge, tmp_path):
    spec = judge.spec("codex")

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    environment = call["env"]
    home = environment["HOME"]
    assert home.startswith(str(judge.root / "jobs")) and home.endswith("/home")
    expected = {
        **spec["environment"],
        "HOME": home,
        "TMPDIR": str(Path(home).parent / "tmp"),
        "USER": "rsi-agent",
        "LOGNAME": "rsi-agent",
        "PATH": f"{judge.fake_bin}:/usr/bin:/bin",
        "TERM": "dumb",
    }
    # bash and Python may add their own bookkeeping variables.
    added = set(environment) - set(expected)
    assert added <= {"PWD", "SHLVL", "_", "LC_CTYPE", "OLDPWD"}, added
    assert {key: environment[key] for key in expected} == expected
    assert environment["OPENAI_API_KEY"] == API_KEY
    # Without a base URL, Codex is keyed as RSI Loop keys it in Work.
    assert environment["CODEX_API_KEY"] == API_KEY
    assert "CALLER_ONLY_SECRET" not in environment
    assert "RSI_AGENT_STATE_ROOT" not in environment


def test_claude_takes_structured_output_and_its_own_flags(judge, tmp_path):
    judge.spec("claude-code", internet=False, effort="max")

    result, work = _exec(judge, tmp_path)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    argv = call["argv"]
    assert call["cli"] == "claude"
    assert argv[:2] == ["--effort", "max"]
    for flag in (
        "-p",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--safe-mode",
    ):
        assert flag in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--setting-sources") + 1] == "user"
    assert argv[argv.index("--model") + 1] == "claude-test"
    assert argv[argv.index("--disallowedTools") + 1] == "WebSearch,WebFetch"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    assert call["env"]["ANTHROPIC_AUTH_TOKEN"] == API_KEY
    assert call["env"]["IS_SANDBOX"] == "1"
    assert call["cwd"] == str(work)
    assert json.loads((tmp_path / "out").read_text()) == {
        "verdict": "clear",
        "key": "",
    }


def test_claude_falls_back_to_the_result_text(judge, tmp_path):
    judge.spec("claude-code", mode="result_only")

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out").read_text() == "plain text answer"


def test_claude_is_error_fails_the_call(judge, tmp_path):
    judge.spec("claude-code", mode="is_error")

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 1
    assert "the Agent reported an error: boom" in result.stderr
    assert not (tmp_path / "out").exists()


def test_qwen_code_answers_in_print_mode_with_its_own_settings(judge, tmp_path):
    from rsi_loop.harness.agent import create_agent

    judge.spec(
        "qwen-code", base_url="http://model.test:8000/v1", internet=False, effort=None
    )
    log = tmp_path / "agent.log"

    result, work = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    argv = call["argv"]
    assert call["cli"] == "qwen"
    for flag in ("--safe-mode", "--chat-recording=false"):
        assert flag in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--model") + 1] == "qwen-test"
    excluded = argv[argv.index("--exclude-tools") + 1].split(",")
    assert {"web_fetch", "skill"} <= set(excluded)
    # The schema reaches Qwen natively, as a file.
    schema_path = argv[argv.index("--json-schema") + 1]
    assert schema_path.startswith("@") and schema_path.endswith("/schema.json")
    # The settings Work writes, with folder trust on, in the private HOME;
    # the endpoint and key stay in the environment.
    agent = create_agent(
        "qwen-code",
        RSILoopConfig(agent_context_window="131072", agent_max_output_tokens="16384"),
    )
    assert call["qwen_settings"] == agent.oneshot_settings(internet=False)
    assert call["env"]["OPENAI_API_KEY"] == API_KEY
    assert call["env"]["OPENAI_BASE_URL"] == "http://model.test:8000/v1"
    assert call["env"]["QWEN_CODE_MODELS_DEV"] == "off"
    assert all(API_KEY not in arg for arg in argv)
    assert call["stdin"] == "Review the evidence.\n"
    assert call["cwd"] == str(work)
    # The structured answer, with the key redacted.
    assert json.loads((tmp_path / "out").read_text()) == {
        "verdict": "clear",
        "key": "[REDACTED]",
    }
    # Only the result line reaches rsi-agent.
    stdout = log.read_text().split("--- stderr ---")[0]
    assert '"structured_result"' in stdout and "thinking aloud" not in stdout


def test_qwen_code_falls_back_to_the_result_text(judge, tmp_path):
    judge.spec("qwen-code", effort=None)

    result, _ = _exec(judge, tmp_path, schema=False)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert "--json-schema" not in call["argv"]
    assert (tmp_path / "out").read_text() == "plain text answer"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("is_error", "the Agent reported an error: boom"),
        # Its status survives the pipe that keeps only the result line.
        ("fail", "the Agent exited with status 7"),
    ],
)
def test_a_failed_qwen_code_call_fails(judge, tmp_path, mode, reason):
    judge.spec("qwen-code", mode=mode, effort=None)

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 1
    assert reason in result.stderr
    assert not (tmp_path / "out").exists()


@requires_node
def test_kimi_code_answers_in_print_mode_with_its_own_config(judge, tmp_path):
    from rsi_loop.harness.agent import create_agent

    judge.spec("kimi-code", base_url="http://model.test:8000/v1", internet=False)
    log = tmp_path / "agent.log"

    result, work = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    argv = call["argv"]
    assert call["cli"] == "kimi"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    # Kimi has no schema option: rsi-agent states the schema in the prompt.
    assert call["prompt"].startswith("Review the evidence.\n\n")
    assert "JSON Schema" in call["prompt"]
    # The prompt reached Kimi's argv in its own process, not the process
    # table; the key stays in the environment.
    assert "Review the evidence" not in call["cmdline"]
    assert API_KEY not in call["cmdline"]
    # Work's config.toml with the reasoning effort, in the private HOME.
    home = call["env"]["HOME"]
    assert call["env"]["KIMI_CODE_HOME"] == f"{home}/.kimi-code"
    agent = create_agent(
        "kimi-code",
        RSILoopConfig(agent_context_window="131072", agent_max_output_tokens="16384"),
    )
    config = agent.config_toml(model="kimi-test", internet=False)
    assert call["config"] == config.replace(
        'capabilities = ["tool_use"]',
        'capabilities = ["tool_use", "thinking"]\n'
        'support_efforts = ["high"]\ndefault_effort = "high"',
    )
    # The default system prompt without subagents; no working-directory
    # skills.
    assert argv[argv.index("--agent-file") + 1].startswith(home)
    assert "${base_prompt}" in call["profile"]
    assert "disallowedTools: Agent, AgentSwarm" in call["profile"]
    assert argv[argv.index("--skills-dir") + 1] == f"{home}/.kimi-code/skills"
    assert call["env"]["OPENAI_API_KEY"] == API_KEY
    assert call["env"]["OPENAI_BASE_URL"] == "http://model.test:8000/v1"
    assert call["env"]["KIMI_DISABLE_TELEMETRY"] == "1"
    # A failed model request fails the call instead of retrying forever.
    assert "KIMI_CODE_INFINITE_RETRY" not in call["env"]
    assert call["cwd"] == str(work)
    # The final assistant message, with the key redacted.
    assert json.loads((tmp_path / "out").read_text()) == {
        "verdict": "clear",
        "key": "[REDACTED]",
    }
    # Only that line reaches rsi-agent.
    stdout = log.read_text().split("--- stderr ---")[0]
    [line] = [line for line in stdout.splitlines()[1:] if line]
    assert json.loads(line)["role"] == "assistant"


@requires_node
def test_kimi_code_answers_in_plain_text_without_a_schema(judge, tmp_path):
    judge.spec("kimi-code", effort=None)

    result, _ = _exec(judge, tmp_path, schema=False)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert call["prompt"] == "Review the evidence.\n"
    assert 'capabilities = ["tool_use"]\n' in call["config"]
    assert (tmp_path / "out").read_text() == "plain text answer"


@requires_node
def test_a_failed_kimi_code_call_fails(judge, tmp_path):
    judge.spec("kimi-code", mode="fail")

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 1
    # Its status survives the pipe that keeps only the final message.
    assert "the Agent exited with status 7" in result.stderr
    assert API_KEY not in result.stderr
    assert not (tmp_path / "out").exists()


def test_opencode_answers_in_run_mode_with_its_own_config(judge, tmp_path):
    from rsi_loop.harness.agent import create_agent

    judge.spec("opencode", base_url="http://model.test:8000/v1", internet=False)
    log = tmp_path / "agent.log"

    result, work = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert call["cli"] == "opencode"
    # The prompt on stdin, with the schema: OpenCode has no schema option.
    assert call["argv"] == ["run", "--format", "json"]
    assert call["stdin"].startswith("Review the evidence.\n\n")
    assert "JSON Schema" in call["stdin"]
    # Work's config with the reasoning effort; the endpoint and key stay in
    # the environment it names.
    env = call["env"]
    agent = create_agent(
        "opencode",
        RSILoopConfig(agent_context_window="131072", agent_max_output_tokens="16384"),
    )
    config = agent.config(model="opencode-test", internet=False)
    config["provider"]["rsi"]["models"]["opencode-test"]["options"] = {
        "reasoningEffort": "high"
    }
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"]) == config
    assert env["RSI_AGENT_API_KEY"] == API_KEY
    assert env["RSI_AGENT_API_BASE_URL"] == "http://model.test:8000/v1"
    assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
    for name in (
        "OPENCODE_DISABLE_PROJECT_CONFIG",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS",
        "OPENCODE_DISABLE_CLAUDE_CODE",
    ):
        assert env[name] == "1"
    # Work's ripgrep link and plugin install skip, and temporary files, in
    # the private HOME.
    assert call["home_files"] == [
        ".cache/opencode/bin/rg",
        ".config/opencode/package-lock.json",
    ]
    assert env["TMPDIR"] == env["HOME"] + "/.tmp"
    assert env["HOME"].startswith(str(judge.root / "jobs"))
    assert call["cwd"] == str(work)
    # The final text part, with the key redacted.
    assert json.loads((tmp_path / "out").read_text()) == {
        "verdict": "clear",
        "key": "[REDACTED]",
    }
    # Only that part reaches rsi-agent.
    stdout = log.read_text().split("--- stderr ---")[0]
    [line] = [line for line in stdout.splitlines()[1:] if line]
    assert json.loads(line)["type"] == "text"


def test_opencode_answers_in_plain_text_without_a_schema(judge, tmp_path):
    judge.spec("opencode", effort=None)

    result, _ = _exec(judge, tmp_path, schema=False)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert call["stdin"] == "Review the evidence.\n"
    assert "reasoningEffort" not in call["env"]["OPENCODE_CONFIG_CONTENT"]
    assert (tmp_path / "out").read_text() == "plain text answer"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        # Its status survives the pipe that keeps only the final text, and
        # its error event reaches the log.
        ("error", "the Agent exited with status 1"),
        ("fail", "the Agent exited with status 7"),
    ],
)
def test_a_failed_opencode_call_fails(judge, tmp_path, mode, reason):
    judge.spec("opencode", mode=mode)
    log = tmp_path / "agent.log"

    result, _ = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 1
    assert reason in result.stderr
    assert not (tmp_path / "out").exists()
    if mode == "error":
        stderr = log.read_text().split("--- stderr ---")[1]
        assert '"type":"error"' in stderr and "bad key [REDACTED]" in stderr
    assert API_KEY not in log.read_text()


PI_ONESHOT_FLAGS = [
    "--no-session",
    "--no-approve",
    "--no-context-files",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
]


@requires_node
def test_pi_answers_in_print_mode_with_its_own_config(judge, tmp_path):
    from rsi_loop.harness.agent import create_agent

    judge.spec("pi", base_url="http://model.test:8000/v1")
    log = tmp_path / "agent.log"

    result, work = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert call["cli"] == "pi"
    # Print mode with the effort, without Work's extension, and nothing from
    # the working directory; the prompt on stdin, with the schema: pi has no
    # schema option.
    assert call["argv"] == [
        "--thinking",
        "high",
        "--offline",
        "-p",
        *PI_ONESHOT_FLAGS,
        "--provider",
        "rsi",
        "--model",
        "pi-test",
    ]
    assert call["stdin"].startswith("Review the evidence.\n\n")
    assert "JSON Schema" in call["stdin"]
    # Work's settings and models in the private HOME, the effort turned on;
    # the endpoint written in, the key only named.
    agent = create_agent(
        "pi",
        RSILoopConfig(agent_context_window="131072", agent_max_output_tokens="16384"),
    )
    models = agent.models_config(model="pi-test")
    models["providers"]["rsi"]["baseUrl"] = "http://model.test:8000/v1"
    models["providers"]["rsi"]["models"][0]["reasoning"] = True
    assert call["pi_config"] == {
        "settings.json": agent.settings(),
        "models.json": models,
    }
    env = call["env"]
    assert env["PI_CODING_AGENT_DIR"] == env["HOME"] + "/.pi/agent"
    assert call["home_files"] == [".pi/agent/models.json", ".pi/agent/settings.json"]
    assert env["TMPDIR"] == env["HOME"] + "/.tmp"
    assert env["HOME"].startswith(str(judge.root / "jobs"))
    assert env["RSI_AGENT_API_KEY"] == API_KEY
    assert env["PI_OFFLINE"] == "1"
    assert call["cwd"] == str(work)
    # The final message without print mode's newline, the key redacted.
    assert (tmp_path / "out").read_text() == json.dumps(
        {"verdict": "clear", "key": "[REDACTED]"}
    )
    assert API_KEY not in log.read_text()


@requires_node
def test_pi_answers_in_plain_text_without_a_schema(judge, tmp_path):
    judge.spec("pi", base_url="http://model.test:8000/v1", effort=None)

    result, _ = _exec(judge, tmp_path, schema=False)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert call["stdin"] == "Review the evidence.\n"
    assert "--thinking" not in call["argv"]
    [model] = call["pi_config"]["models.json"]["providers"]["rsi"]["models"]
    assert model["reasoning"] is False
    # Only the trailing newline goes; the answer's own lines stay.
    assert (tmp_path / "out").read_text() == (
        "plain text answer\n\nwith a second paragraph"
    )


@requires_node
@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("error", "the Agent exited with status 1"),
        ("fail", "the Agent exited with status 7"),
        ("empty", "the Agent's final message is empty"),
    ],
)
def test_a_failed_pi_call_fails(judge, tmp_path, mode, reason):
    judge.spec("pi", base_url="http://model.test:8000/v1", mode=mode)
    log = tmp_path / "agent.log"

    result, _ = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 1
    assert reason in result.stderr
    assert not (tmp_path / "out").exists()
    if mode == "error":
        assert "401: bad key [REDACTED]" in log.read_text()
    assert API_KEY not in log.read_text() + result.stderr


def _zcode_spec(judge: Judge, tmp_path: Path, **options) -> dict:
    """The ZCode spec, its pinned node and bundle pointed at the fakes."""
    # ZCode has no default endpoint.
    options.setdefault("base_url", "http://model.test:8000/v1")
    spec = judge.spec("zcode", **options)
    spec["command"] = (
        spec["command"]
        .replace("/usr/local/bin/node", str(NODE))
        .replace("/opt/zcode/zcode.cjs", str(tmp_path / "zcode" / "zcode.cjs"))
    )
    (judge.private / "launch.json").write_text(json.dumps(spec))
    return spec


def _zcode_secret_dirs() -> set[Path]:
    return set(Path("/dev/shm").glob("rsi-zcode.*"))


@requires_node
def test_zcode_answers_in_print_mode_with_its_own_config(tmpfs_judge, tmp_path):
    from rsi_loop.harness.agent import create_agent

    judge = tmpfs_judge
    _zcode_spec(judge, tmp_path)
    log = tmp_path / "agent.log"
    before = _zcode_secret_dirs()

    result, work = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert call["cli"] == "zcode.cjs"
    # ZCode has no schema option: rsi-agent states the schema in the prompt.
    assert call["argv"][:-1] == [
        "--output-format",
        "json",
        "--mode",
        "yolo",
        "--disallowed-tools",
        "AskUserQuestion,EnterPlanMode,ExitPlanMode,CronList,CronUpdate,"
        "CronCreate,CronDelete,OffPeakCreate,Skill,Agent,SendMessage,WebFetch",
    ]
    assert call["prompt"].startswith("Review the evidence.\n\n")
    assert "JSON Schema" in call["prompt"]
    # The prompt reached ZCode's argv in its own process, not the process
    # table; the key never was on a command line.
    assert "Review the evidence" not in call["cmdline"]
    assert API_KEY not in call["cmdline"]
    # Work's CLI config and provider config, with the reasoning effort; the
    # provider config, with the key, in the private HOME on the tmpfs, and
    # nothing left under /dev/shm.
    home = call["env"]["HOME"]
    assert home.startswith(str(judge.root / "jobs"))
    assert call["home_files"] == [
        ".zcode/cli/config.json",
        ".zcode/provider_config.json",
    ]
    assert call["provider_file"] == f"{home}/.zcode/provider_config.json"
    assert call["provider_mode"] == 0o600
    agent = create_agent(
        "zcode",
        RSILoopConfig(agent_context_window="131072", agent_max_output_tokens="16384"),
    )
    assert call["cli_config"] == agent.cli_config()
    provider = agent.provider_config(model="zcode-test")
    rule = provider["config"]["providerConfigRules"]["providerRules"][0]["config"]
    rule["access"]["apiKey"] = API_KEY
    rule["api"]["baseUrl"] = "http://model.test:8000/v1"
    model = provider["config"]["modelConfigRules"]["manualProviderModelRules"][0]
    model["config"]["optionSpecs"]["reasoningLevel"] = {
        "values": ["high"],
        "map": '{"reasoning_effort": reasoningLevel}',
    }
    provider["config"]["defaultModelSelection"]["options"] = {"reasoningLevel": "high"}
    assert call["provider"] == provider
    assert _zcode_secret_dirs() == before
    # Work's environment; no proxy, CA, retry bound or storage the working
    # directory's .env could set.
    env = call["env"]
    assert env["RSI_AGENT_API_KEY"] == API_KEY
    assert env["ZCODE_BASE_URL"] == "http://127.0.0.1:9"
    assert env["ZCODE_MODEL_TELEMETRY_ENABLED"] == "false"
    for name in (
        "ZCODE_HTTP_PROXY",
        "ZCODE_NO_PROXY",
        "ZCODE_AGENT_CA_CERT",
        "NODE_TLS_REJECT_UNAUTHORIZED",
        "ZCODE_MODEL_RETRY_MAX_RETRIES",
        "ZCODE_BETA",
        "ZCODE_ENV",
    ):
        assert env[name] == ""
    assert env["ZCODE_STORAGE_DIR"] == env["ZCODE_HOME"] == f"{home}/.zcode"
    assert env["ZCODE_DATA_BASE_DIR"] == home
    assert env["ZCODE_LOG_DIR"] == f"{home}/.zcode/cli/log"
    assert call["cwd"] == str(work)
    # The response, with the key redacted.
    assert json.loads((tmp_path / "out").read_text()) == {
        "verdict": "clear",
        "key": "[REDACTED]",
    }
    assert API_KEY not in log.read_text()


@requires_node
def test_zcode_answers_in_plain_text_without_a_schema(tmpfs_judge, tmp_path):
    judge = tmpfs_judge
    # A proxy Work's own environment sets is Work's, and stays.
    _zcode_spec(
        judge, tmp_path, effort=None, internet=True, ZCODE_HTTP_PROXY="http://p:3128"
    )

    result, _ = _exec(judge, tmp_path, schema=False)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert call["prompt"] == "Review the evidence.\n"
    assert not call["argv"][-2].endswith(",WebFetch")
    selection = call["provider"]["config"]["defaultModelSelection"]
    assert selection["options"] == {"reasoningLevel": "default"}
    assert call["env"]["ZCODE_HTTP_PROXY"] == "http://p:3128"
    assert (tmp_path / "out").read_text() == "plain text answer"


@requires_node
@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("fail", "the Agent exited with status 1"),
        ("empty", "the Agent's final message is empty"),
    ],
)
def test_a_failed_zcode_call_fails(tmpfs_judge, tmp_path, mode, reason):
    judge = tmpfs_judge
    _zcode_spec(judge, tmp_path, mode=mode)

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 1
    assert reason in result.stderr
    assert API_KEY not in result.stderr
    assert not (tmp_path / "out").exists()


@requires_node
def test_zcode_never_writes_its_key_to_a_home_on_disk(judge, tmp_path):
    if _filesystem(tmp_path) == "tmpfs":
        pytest.skip("the test directory is a tmpfs")
    _zcode_spec(judge, tmp_path)

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 1
    assert "the Agent exited with status 1" in result.stderr
    assert judge.calls() == []


@requires_node
def test_a_missing_zcode_is_unavailable(tmpfs_judge, tmp_path):
    judge = tmpfs_judge
    _zcode_spec(judge, tmp_path)
    (tmp_path / "zcode" / "zcode.cjs").unlink()

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 3
    assert "zcode CLI is not installed" in result.stderr


def test_without_a_native_schema_option_the_prompt_states_the_schema(judge, tmp_path):
    spec = judge.spec("codex")
    spec["schema_option"] = None
    spec["command"] = spec["command"].replace(" {schema_option}", "")
    spec["command"] = spec["command"].replace(" -", " {schema_option} -")
    (judge.private / "launch.json").write_text(json.dumps(spec))

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert "--output-schema" not in call["argv"]
    stdin = call["stdin"]
    assert stdin.startswith("Review the evidence.\n\n")
    assert "conforms to this JSON Schema:\n" in stdin
    assert json.loads(stdin.split("JSON Schema:\n", 1)[1]) == SCHEMA


def test_no_schema_and_no_workdir_use_a_private_job_directory(judge, tmp_path):
    judge.spec("codex")

    result, _ = _exec(judge, tmp_path, schema=False, workdir=False)

    assert result.returncode == 0, result.stderr
    [call] = judge.calls()
    assert "--output-schema" not in call["argv"]
    assert call["cwd"].startswith(str(judge.root / "jobs"))
    assert call["cwd"].endswith("/work")


def test_login_files_are_copied_into_the_private_home(judge, tmp_path, fake_bin):
    judge.spec("codex", credentials=True)
    reader = fake_bin / "codex"
    reader.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "home = os.environ['HOME']\n"
        "login = open(os.path.join(home, '.codex', 'auth.json')).read()\n"
        "mode = oct(os.stat(os.path.join(home, '.codex', 'auth.json')).st_mode)\n"
        "out = args[args.index('--output-last-message') + 1]\n"
        "open(out, 'w').write(login + ' ' + mode)\n"
    )

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 0, result.stderr
    answer = (tmp_path / "out").read_text()
    # The copy is the Agent's own, private; its secret is redacted on output.
    assert "[REDACTED]" in answer and "LOGIN-SECRET-77" not in answer
    assert answer.endswith("0o100600")


@pytest.mark.parametrize(
    ("left", "timeout", "code"),
    [(-10, "60", 3), (200, "60", 3), (1000, "800", 3), (1000, "60", 0)],
)
def test_exec_refuses_a_call_the_login_would_not_outlive(
    judge, tmp_path, left, timeout, code
):
    """The login copies carry no usable refresh token: Codex's refresh window
    and the call's timeout must fit before the access token expires."""
    spec = judge.spec("codex", credentials=True)
    spec["credentials_expire_at"] = time.time() + left
    (judge.private / "launch.json").write_text(json.dumps(spec))

    result, _ = _exec(judge, tmp_path, "--timeout", timeout)

    assert result.returncode == code, result.stderr
    if code:
        assert "the Agent's login expires" in result.stderr
        assert "never refreshes it" in result.stderr
        assert judge.calls() == []


@pytest.mark.parametrize(("round_left", "code"), [(None, 3), (600, 0), (900, 3)])
def test_exec_needs_the_login_only_until_the_round_ends(
    judge, tmp_path, round_left, code
):
    """No call outlives the round: the default --timeout (1800 s) counts only
    up to ``round_ends_at``."""
    spec = judge.spec("codex", credentials=True)
    spec["credentials_expire_at"] = time.time() + 1000
    if round_left is not None:
        spec["round_ends_at"] = time.time() + round_left
    (judge.private / "launch.json").write_text(json.dumps(spec))

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == code, result.stderr


# -- failures ---------------------------------------------------------------------


def test_without_a_launch_spec_the_judge_has_no_model_access(judge, tmp_path):
    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 3
    assert "model_access = true" in result.stderr


def test_an_agent_without_a_one_shot_command_is_refused_by_name(judge, tmp_path):
    spec = judge.spec("codex")
    spec.update(command=None, unsupported="the run's Agent 'other' cannot run")
    (judge.private / "launch.json").write_text(json.dumps(spec))

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 3
    assert "'other'" in result.stderr
    assert judge.calls() == []


@pytest.mark.parametrize(
    "case",
    ["no-args", "bad-schema", "array-schema", "protected", "root", "timeout", "empty"],
)
def test_usage_errors_exit_2(judge, tmp_path, case):
    judge.spec("codex")
    prompt, schema, work = _inputs(tmp_path)
    args = ["exec", "--prompt-file", str(prompt), "--output", str(tmp_path / "out")]
    if case == "no-args":
        args = ["exec"]
    elif case == "bad-schema":
        schema.write_text("{not json")
        args += ["--schema", str(schema)]
    elif case == "array-schema":
        schema.write_text("[]")
        args += ["--schema", str(schema)]
    elif case == "protected":
        verifier = tmp_path / "verifier" / "inner"
        verifier.mkdir(parents=True)
        args += ["--workdir", str(verifier)]
    elif case == "root":
        args += ["--workdir", "/"]
    elif case == "timeout":
        args += ["--timeout", "0"]
    elif case == "empty":
        prompt.write_text("  \n")

    result = judge.run(*args)

    assert result.returncode == 2, result.stderr
    assert judge.calls() == []


@pytest.mark.parametrize("flag", ["--output", "--log"])
def test_nothing_rsi_agent_writes_may_lie_in_the_agents_workdir(judge, tmp_path, flag):
    judge.spec("codex")
    prompt, _, work = _inputs(tmp_path)
    args = {"--output": str(tmp_path / "out"), "--log": None}
    args[flag] = str(work / "inside")
    command = ["exec", "--prompt-file", str(prompt), "--workdir", str(work)]
    for name, value in args.items():
        if value is not None:
            command += [name, value]

    result = judge.run(*command)

    assert result.returncode == 2
    assert f"{flag} must lie outside the --workdir" in result.stderr
    assert judge.calls() == []


def _program():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rsi_agent_under_test", PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_concurrent_call_claims_its_own_uid_slot(tmp_path):
    import fcntl

    program = _program()
    held = os.open(tmp_path / "slot-0.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        slot, descriptor = program._claim_slot(str(tmp_path), 2, time.monotonic() + 5)
        assert slot == 1
        # Every slot taken: the call waits, then gives up at its deadline.
        with pytest.raises(program.Failure) as failure:
            program._claim_slot(str(tmp_path), 2, time.monotonic() + 0.3)
        assert failure.value.code == 124
        os.close(descriptor)
        # A finished (or killed) call frees its slot for the next one.
        slot, descriptor = program._claim_slot(str(tmp_path), 2, time.monotonic() + 5)
        assert slot == 1
        os.close(descriptor)
    finally:
        os.close(held)


def test_a_failing_agent_exits_1_with_a_redacted_reason(judge, tmp_path):
    judge.spec("codex", mode="fail")
    log = tmp_path / "agent.log"

    result, _ = _exec(judge, tmp_path, "--log", str(log))

    assert result.returncode == 1
    assert "the Agent exited with status 7" in result.stderr
    assert "model said no with key [REDACTED]" in result.stderr
    assert API_KEY not in result.stderr + result.stdout
    assert not (tmp_path / "out").exists()
    # The Agent printed its key; the log keeps the line, never the key.
    assert "model said no with key [REDACTED]" in log.read_text()
    assert API_KEY not in log.read_text()


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("nooutput", "no final message"),
        ("empty", "empty"),
        ("symlink", "symlink"),
    ],
)
def test_a_missing_empty_or_symlinked_answer_fails(judge, tmp_path, mode, reason):
    judge.spec("codex", mode=mode)
    (tmp_path / "out").write_text("previous")

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 1, result.stderr
    assert reason in result.stderr
    # The output is replaced only by a complete answer.
    assert (tmp_path / "out").read_text() == "previous"


@pytest.mark.parametrize(
    ("agent", "cli", "effort"),
    # Qwen Code's and Kimi Code's status 127 also survives their pipes, and
    # pi's its command substitution.
    [
        ("codex", "codex", "high"),
        ("qwen-code", "qwen", None),
        pytest.param("kimi-code", "kimi", "high", marks=requires_node),
        ("opencode", "opencode", "high"),
        # pi's prelude writes its models with node.
        pytest.param("pi", "pi", "high", marks=requires_node),
    ],
)
def test_a_missing_agent_cli_is_unavailable(judge, tmp_path, agent, cli, effort):
    # An endpoint, which pi needs before it starts.
    judge.spec(agent, effort=effort, base_url="http://model.test:8000/v1")
    (judge.fake_bin / cli).unlink()

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 3
    assert f"{agent} CLI is not installed" in result.stderr


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[0]
    except FileNotFoundError:
        return True
    return state in ("Z", "X")


def test_a_timeout_ends_the_whole_process_group(judge, tmp_path):
    child = tmp_path / "child.pid"
    judge.spec("codex", mode="sleep", FAKE_CHILD=str(child))

    started = time.monotonic()
    result, _ = _exec(judge, tmp_path, "--timeout", "1")

    assert result.returncode == 124, result.stderr
    assert time.monotonic() - started < 15
    pid = int(child.read_text())
    if not _gone(pid):
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the Agent's child survived the timeout")


def test_sigterm_stops_the_agent_and_exits_143(judge, tmp_path):
    child = tmp_path / "child.pid"
    judge.spec("codex", mode="sleep", FAKE_CHILD=str(child))
    prompt, schema, work = _inputs(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(PROGRAM),
            "exec",
            "--prompt-file",
            str(prompt),
            "--output",
            str(tmp_path / "out"),
            "--workdir",
            str(work),
        ],
        env={
            "PATH": f"{judge.fake_bin}:/usr/bin:/bin",
            "RSI_AGENT_STATE_ROOT": str(judge.root),
        },
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 20
    while not child.read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert child.read_text()
    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=20)

    assert process.returncode == 143, stderr
    pid = int(child.read_text())
    deadline = time.monotonic() + 5
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _gone(pid):
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the Agent's child survived SIGTERM to rsi-agent")
    assert not (tmp_path / "out").exists()


# -- the program and its spec -------------------------------------------------------


def test_the_launcher_runs_the_mounted_program_in_isolated_mode():
    launcher = judge_agent_source("rsi-agent").decode()
    assert launcher.startswith("#!/bin/sh\n")
    assert "python3 is required in the Judge image" in launcher
    assert 'exec python3 -I -S -B /opt/rsi-agent/rsi_agent.py "$@"' in launcher
    assert judge_agent_source("rsi_agent.py") == PROGRAM.read_bytes()


SHARED = PROGRAM.with_name("judge_isolation.py")


@pytest.mark.parametrize("path", [PROGRAM, SHARED], ids=["rsi_agent", "shared"])
def test_the_program_stays_python_3_8(path):
    import ast

    source = path.read_text()
    ast.parse(source, feature_version=(3, 8))
    # ast does not reject 3.10's parenthesized context managers.
    assert "with (" not in source


@pytest.mark.parametrize("path", [PROGRAM, SHARED], ids=["rsi_agent", "shared"])
def test_the_program_imports_nothing_from_rsi_harness(path):
    source = path.read_text()
    assert "import rsi_" not in source and "from rsi_" not in source


def test_the_launch_spec_names_the_agent_user_and_protected_paths():
    config = RSILoopConfig(agent_api_key=API_KEY)
    oneshot = rsi_loop_oneshot(config, AgentPlan(name="codex"))
    spec = json.loads(build_launch_spec(oneshot, None, {API_KEY}))

    assert spec["uid"] == spec["gid"] == JUDGE_AGENT_UID
    assert spec["slots"] == JUDGE_AGENT_SLOTS > 1
    assert spec["state_root"] == "/run/rsi-agent"
    assert {"/logs/verifier", "/tests", "/run/rsi-agent/private"} <= set(
        spec["protected"]
    )
    assert spec["redact"] == [API_KEY]
    assert shlex.split(spec["command"].replace("{", "").replace("}", ""))[:2] == [
        "codex",
        "exec",
    ]


def test_a_credential_never_reaches_the_agents_argv():
    # /proc/<pid>/cmdline is readable by every process of the Judge's uid.
    config = RSILoopConfig(agent_api_key=API_KEY)
    oneshot = rsi_loop_oneshot(config, AgentPlan(name="codex"))
    assert API_KEY not in oneshot["command"]
    for key in ("command", "schema_option"):
        leaky = {**oneshot, key: f"{oneshot[key]} --api-key {API_KEY}"}
        with pytest.raises(SetupError, match="must not carry a credential"):
            build_launch_spec(leaky, None, {API_KEY})


# -- isolation without root -------------------------------------------------------

LANDLOCK = _program().landlock_abi(ctypes.CDLL(None, use_errno=True))[0]
requires_landlock = pytest.mark.skipif(
    LANDLOCK < 1 or os.geteuid() == 0, reason="needs Landlock, without root"
)


@requires_landlock
def test_the_isolation_command_reports_landlock(judge):
    judge.spec("codex")

    result = judge.run("isolation")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "mode": "landlock",
        "landlock_abi": LANDLOCK,
        "fallback": "none",
    }


@requires_landlock
def test_a_confined_call_reads_and_writes_only_its_own(judge, tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    sibling = judge.root / "jobs" / "sibling-call"
    sibling.mkdir(parents=True)
    (sibling / "final-message").write_text("another call's answer")
    judge.root.joinpath("jobs").chmod(0o711)
    probe = {
        "launch_spec": [str(judge.private / "launch.json"), "rb"],
        "sibling_job": [str(sibling / "final-message"), "rb"],
        "verifier_write": [str(verifier / "reward.json"), "w"],
        "verifier_list": [str(verifier), "list"],
        "evidence": [str(tmp_path / "jobs" / "one" / "evidence.txt"), "rb"],
        "home_write": ["HOME/new.txt", "w"],
        "tmp_write": ["TMPDIR/new.txt", "w"],
    }
    spec = judge.spec("codex", mode="probe", FAKE_PROBE=json.dumps(probe))
    spec["hidden"] = [*spec["hidden"], str(verifier)]
    (judge.private / "launch.json").write_text(json.dumps(spec))
    script = judge.fake_bin / "codex"
    # Resolve HOME and TMPDIR inside the call.
    script.write_text(
        script.read_text().replace(
            'probe = json.loads(os.environ["FAKE_PROBE"])',
            'probe = json.loads(os.environ["FAKE_PROBE"].replace("HOME/", '
            'os.environ["HOME"] + "/").replace("TMPDIR/", os.environ["TMPDIR"] + "/"))',
        )
    )

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 0, result.stderr
    found = next(line["probe"] for line in judge.calls() if "probe" in line)
    denied = "Permission denied"
    assert found == {
        "launch_spec": denied,
        "sibling_job": denied,
        "verifier_write": denied,
        "verifier_list": denied,
        "evidence": "ok",
        "home_write": "ok",
        "tmp_write": "ok",
    }
    assert not (verifier / "reward.json").exists()


@requires_landlock
def test_a_daemon_the_agent_left_behind_is_killed(judge, tmp_path):
    child = tmp_path / "daemon.pid"
    judge.spec("codex", mode="daemon", FAKE_CHILD=str(child))

    result, _ = _exec(judge, tmp_path)

    assert result.returncode == 0, result.stderr
    pid = int(child.read_text())
    deadline = time.monotonic() + 5
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _gone(pid):
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the Agent's daemon survived the call")


def test_without_root_or_landlock_the_call_fails_closed(monkeypatch):
    program = _program()
    spec = {"uid": 61000, "gid": 61000, "slots": 16, "isolation_fallback": "none"}
    libc = ctypes.CDLL(None, use_errno=True)
    shared = program.isolation
    monkeypatch.setattr(shared, "landlock_abi", lambda _libc: (0, "ENOSYS"))
    monkeypatch.setattr(shared, "uid_mode_problem", lambda *_args: "euid 1000")
    # The Judge's environment never chooses: a task's image can set it.
    monkeypatch.setenv("RSI_JUDGE_ISOLATION_FALLBACK", "same-user")

    with pytest.raises(program.Failure) as failure:
        program.isolation_mode(spec, libc)

    assert failure.value.code == 3
    assert "Landlock is unavailable" in failure.value.message
    assert "ENOSYS" in failure.value.detail
    # Only the operator's fallback, from the launch spec, allows it.
    fallback = program.isolation_mode({**spec, "isolation_fallback": "same-user"}, libc)
    assert fallback["mode"] == "same_user"


def test_root_without_a_needed_capability_is_not_uid_mode(monkeypatch):
    """A root Judge without CAP_SETUID (bit 7) would fail at setuid: it must
    use Landlock instead of claiming uid mode."""
    shared = _program().isolation
    every = sum(1 << bit for bit in shared.UID_MODE_CAPABILITIES)
    monkeypatch.setattr(shared.os, "geteuid", lambda: 0)
    monkeypatch.setattr(shared, "id_map_covers", lambda *_args: True)

    monkeypatch.setattr(shared, "capabilities", lambda: every)
    assert shared.uid_mode_problem(61000, 16) is None
    monkeypatch.setattr(shared, "capabilities", lambda: every & ~(1 << 7))
    assert "capability" in shared.uid_mode_problem(61000, 16)


def test_the_same_user_fallback_runs_with_a_warning(judge, tmp_path):
    # A copy of the program whose shared module reports no Landlock.
    copy = tmp_path / "no-landlock"
    copy.mkdir()
    patched = copy / "rsi_agent.py"
    shutil.copy(PROGRAM, patched)
    shared = PROGRAM.with_name("judge_isolation.py").read_text()
    copy.joinpath("judge_isolation.py").write_text(
        shared.replace(
            "def landlock_abi(libc) -> tuple:",
            "def landlock_abi(libc) -> tuple:\n    return 0, 'ENOSYS'\n\n\n"
            "def _unused_landlock_abi(libc) -> tuple:",
        )
    )
    spec = judge.spec("codex")
    if os.geteuid() == 0:
        pytest.skip("root takes per-call uids")
    environment = {
        "PATH": f"{judge.fake_bin}:/usr/bin:/bin",
        "RSI_AGENT_STATE_ROOT": str(judge.root),
    }
    prompt, _schema, work = _inputs(tmp_path)
    argv = [sys.executable, "-I", "-S", "-B", str(patched), "exec", "--prompt-file"]
    argv += [str(prompt), "--output", str(tmp_path / "out"), "--workdir", str(work)]
    refused = subprocess.run(
        argv, env=environment, capture_output=True, text=True, check=False
    )
    assert refused.returncode == 3, refused.stderr
    assert "cannot isolate the Agent" in refused.stderr
    spec["isolation_fallback"] = "same-user"
    (judge.private / "launch.json").write_text(json.dumps(spec))
    allowed = subprocess.run(
        argv, env=environment, capture_output=True, text=True, check=False
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "same-user fallback" in allowed.stderr
    report = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(patched), "isolation"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert json.loads(report.stdout)["mode"] == "same_user"


def test_launch_spec_version_2_lists_what_a_confined_agent_never_reads():
    config = RSILoopConfig(agent_api_key=API_KEY)
    oneshot = rsi_loop_oneshot(config, AgentPlan(name="codex"))
    hidden = json.loads(build_launch_spec(oneshot, None, {API_KEY}))
    public = json.loads(
        build_launch_spec(
            oneshot,
            None,
            {API_KEY},
            visibility="public",
            isolation_fallback="same-user",
        )
    )

    assert hidden["version"] == public["version"] == 2
    assert hidden["hidden"] == ["/run/rsi-agent", "/logs/verifier", "/tests"]
    assert public["hidden"] == ["/run/rsi-agent", "/logs/verifier"]
    assert (hidden["isolation_fallback"], public["isolation_fallback"]) == (
        "none",
        "same-user",
    )
    assert "/logs/agent" not in hidden["hidden"]
    with pytest.raises(Exception, match="none or same-user"):
        build_launch_spec(oneshot, None, set(), isolation_fallback="yes")
