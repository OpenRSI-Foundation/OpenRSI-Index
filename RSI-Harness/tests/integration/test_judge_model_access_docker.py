"""Real Docker: a model-access Judge calls the run's Agent through rsi-agent."""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import threading
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter, rsi_loop_oneshot
from rsi_harness.judge_agent import JUDGE_AGENT_TMPFS, install_judge_agent
from rsi_harness.models import (
    AgentPlan,
    CompileOptions,
    ContainerMount,
    ContainerSpec,
    ContainerTmpfs,
    NetworkPolicy,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.judge import PUBLIC_JUDGE_TMP
from rsi_harness.runtime.judge_model_access import build_launch_spec
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
)
from rsi_harness.runtime.network import UNREACHABLE_DNS_SERVER, no_network_dns_servers
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.agent.claude_code import ClaudeCodeAgent
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

IMAGE = "ubuntu:24.04"
# rsi-agent needs python3 in the Judge image: the science sample's base.
PYTHON_IMAGE = (
    "python@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b"
)
# Synthetic: accepted only by the fake model server below.
API_KEY = "sk-synthetic-judge-model-key-5f0c"
LOGIN_SECRET = "SYNTHETIC-JUDGE-LOGIN-SECRET"

# A stand-in Agent CLI (codex or claude): reaches the fake model at the
# Agent's base URL with the Agent's key, and reports what it can reach.
FAKE_CLI = r"""#!/usr/bin/env python3
import json, os, sys, urllib.request
args = sys.argv[1:]
claude = os.path.basename(sys.argv[0]) == "claude"
base = os.environ["ANTHROPIC_BASE_URL" if claude else "OPENAI_BASE_URL"]
key = os.environ["ANTHROPIC_AUTH_TOKEN" if claude else "OPENAI_API_KEY"]
prompt = sys.stdin.read()
request = urllib.request.Request(
    base.rstrip("/") + "/fake", data=json.dumps({"prompt": prompt}).encode(),
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
reply = json.load(urllib.request.urlopen(request, timeout=30))

def denied(action):
    try:
        action()
    except OSError:
        return True
    return False

def write_verifier():
    open("/logs/verifier/agent-was-here", "w").close()

answer = {
    "answer": reply["answer"], "key": key, "uid": os.getuid(),
    "home": os.environ["HOME"], "cwd": os.getcwd(),
    "denied": {
        "write_verifier": denied(write_verifier),
        "read_launch_spec": denied(
            lambda: open("/run/rsi-agent/private/launch.json").read()),
        "list_tests": denied(lambda: os.listdir("/tests")),
    },
}
if claude:
    print(json.dumps({"type": "result", "is_error": False, "result": "",
                      "structured_output": answer}))
else:
    out = args[args.index("--output-last-message") + 1]
    open(out, "w").write(json.dumps(answer))
"""

JUDGE_SCRIPT = r"""#!/bin/bash
set -u
# The procedure itself holds no credential; only rsi-agent does.
for name in OPENAI_API_KEY OPENAI_BASE_URL CODEX_API_KEY ANTHROPIC_AUTH_TOKEN \
    ANTHROPIC_API_KEY ANTHROPIC_BASE_URL RSI_AGENT_HOME; do
    if [ -n "${!name:-}" ]; then
        echo "procedure sees $name"
        echo '{"reward": 0}' > /logs/verifier/reward.json
        exit 0
    fi
done
# The image says same-user; the Harness's explicit "none" wins.
if [ "${RSI_JUDGE_ISOLATION_FALLBACK:-unset}" != none ]; then
    echo "isolation fallback is ${RSI_JUDGE_ISOLATION_FALLBACK:-unset}"
    echo '{"reward": 0}' > /logs/verifier/reward.json
    exit 0
fi
# A task image must not enable signal-scope compatibility either.
if [ "${RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL:-unset}" != 0 ]; then
    echo "unexpected signal compatibility"
    echo '{"reward": 0}' > /logs/verifier/reward.json
    exit 0
fi
printf 'agent=%s model=%s\n' "$RSI_AGENT_NAME" "$RSI_AGENT_MODEL"
mkdir -p /tmp/review/work
printf 'Answer the question.\n' > /tmp/review/prompt.md
printf '{"type": "object"}' > /tmp/review/schema.json
rsi-agent exec --prompt-file /tmp/review/prompt.md \
    --output /tmp/review/answer.json --schema /tmp/review/schema.json \
    --workdir /tmp/review/work --timeout 120
echo "rsi-agent exit $?"
cat /tmp/review/answer.json; echo
python3 - <<'PY'
import json
a = json.load(open("/tmp/review/answer.json"))
ok = (a["answer"] == "fake-model-says-42" and a["key"] == "[REDACTED]"
      and a["uid"] == 61000 and a["home"].startswith("/run/rsi-agent/jobs/")
      and a["cwd"] == "/tmp/review/work" and all(a["denied"].values()))
json.dump({"reward": 1 if ok else 0}, open("/logs/verifier/reward.json", "w"))
PY
"""


def _docker(image: str = IMAGE):
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(image)
    except Exception as error:
        message = f"local cached {image} Docker authority unavailable: {error}"
        # scripts/bump_agent.py sets it: a bump must not pass on skipped cases.
        if os.environ.get("RSI_REQUIRE_DOCKER_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return client


def _install(cli: str) -> list[str]:
    encoded = base64.b64encode(FAKE_CLI.encode()).decode()
    return [
        f"echo {encoded} | base64 -d > /usr/local/bin/{cli} && "
        f"chmod 0755 /usr/local/bin/{cli}"
    ]


class _FakeModel(BaseHTTPRequestHandler):
    seen: list[tuple[str, dict]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        authorization = self.headers.get("Authorization", "")
        type(self).seen.append((authorization, payload))
        status = 200 if authorization == f"Bearer {API_KEY}" else 401
        body = json.dumps({"answer": "fake-model-says-42"}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body if status == 200 else b"{}")

    def log_message(self, *args) -> None:
        del args


@pytest.mark.parametrize(
    ("agent", "adapter", "cli"),
    [("codex", CodexAgent, "codex"), ("claude-code", ClaudeCodeAgent, "claude")],
)
# The default hidden Judge starts from the clean Work image; clean_image =
# false starts it from a snapshot of Work. rsi-agent and the explicit
# isolation fallback reach both the same way.
@pytest.mark.parametrize("clean_image", [True, False], ids=["clean", "snapshot"])
def test_real_judge_calls_the_agent_through_rsi_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    agent: str,
    adapter: type,
    cli: str,
    clean_image: bool,
) -> None:
    """Full production run as the current user; only iptables is faked."""
    client = _docker(PYTHON_IMAGE)
    gateway = next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )
    _FakeModel.seen = []
    server = ThreadingHTTPServer(("0.0.0.0", 0), _FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request.addfinalizer(server.shutdown)
    base_url = f"http://{gateway}:{server.server_address[1]}/v1"

    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PYTHON_IMAGE}\nRUN mkdir -p /workspace\nWORKDIR /workspace\n"
        # A task's image cannot choose the operator's isolation fallback.
        "ENV RSI_JUDGE_ISOLATION_FALLBACK=same-user\n"
        "ENV RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL=1\n"
    )
    config = task / "task.toml"
    config.write_text(
        config.read_text()
        + "\n[metadata.rsi_harness.verifier]\nmodel_access = true\n"
        + ("" if clean_image else "clean_image = false\n")
    )
    (task / "tests" / "test.sh").write_text(JUDGE_SCRIPT)
    data = tmp_path / "data"

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            # Work never has rsi-agent; then submit once.
            script = (
                "set -eu\n"
                "if command -v rsi-agent; then echo RSI-AGENT-PRESENT; fi\n"
                "test ! -e /opt/rsi-agent && echo NO-RSI-AGENT-IN-WORK\n"
                "rsi-submit\n"
            )
            return replace(prepared, command=("/bin/bash", "-lc", script))

    monkeypatch.setattr(adapter, "install_cmds", _install(cli))
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=API_KEY, agent_api_base_url=base_url
        ),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: ScriptedAgent(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
    )

    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)

    result = services.run(
        RunRequest(
            task_dir=task,
            agent_name=agent,
            model="model-test",
            reasoning_effort="high",
            gpu_selectors=("0",),
            options=CompileOptions(agent_name=agent, primary_reward="reward"),
        )
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    feedback = (run_root / "feedback" / "agent-1.log").read_text()
    agent_output = (run_root / "agent_output.txt").read_text()
    assert result.status == RunStatus.COMPLETED, feedback + agent_output[-2000:]
    assert result.best_score == 1, feedback
    plan = json.loads((run_root / "run-plan.json").read_text())
    assert plan["task"]["verifier"]["clean_image"] is clean_image
    assert "NO-RSI-AGENT-IN-WORK" in agent_output
    assert "RSI-AGENT-PRESENT" not in agent_output
    assert len(_FakeModel.seen) == 1
    assert _FakeModel.seen[0][0] == f"Bearer {API_KEY}"
    assert "Answer the question." in _FakeModel.seen[0][1]["prompt"]
    assert f"agent={agent} model=model-test" in feedback
    assert "rsi-agent exit 0" in feedback
    assert '"key": "[REDACTED]"' in feedback
    for path in run_root.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text(errors="replace"), path


# A fake Codex for the lower-level checks: its behaviour is named in the prompt.
FAKE_CODEX = r"""#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
args = sys.argv[1:]
prompt = sys.stdin.read().split()
mode = prompt[0]
out = args[args.index("--output-last-message") + 1]
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.Popen(["setsid", "sleep", "300"])
    time.sleep(300)
if mode == "sleep":
    time.sleep(float(prompt[1]))
home = os.environ["HOME"]
login = os.path.join(home, ".codex", "auth.json")
if mode == "leak":
    # A verbose (or hostile) Agent prints its login on both streams.
    sys.stdout.write(open(login).read() + "\n")
    sys.stderr.write(open(login).read() + "\n")
if mode == "victim":
    # Answer early, then stay alive while a sibling call looks for it.
    open(out, "w").write(json.dumps({"verdict": "honest"}))
    time.sleep(float(prompt[1]))
    sys.exit(0)
if mode == "probe":
    # What this call's Agent can reach of the Judge's private places.
    def attempt(action):
        try:
            action()
            return "done"
        except OSError as error:
            return error.strerror
    def touch(path):
        open(path, "w").close()
    found = {
        "launch_spec": attempt(
            lambda: open("/run/rsi-agent/private/launch.json").read()),
        "login_store": attempt(
            lambda: open("/home/agent/.codex/auth.json").read()),
        "write_verifier": attempt(lambda: touch("/logs/verifier/agent-was-here")),
        "list_jobs": attempt(lambda: os.listdir("/run/rsi-agent/jobs")),
        "list_tests": attempt(lambda: os.listdir("/tests")),
        "own_home": attempt(lambda: touch(os.path.join(home, "note"))),
        "own_tmp": attempt(lambda: touch(os.path.join(os.environ["TMPDIR"], "n"))),
    }
    open(out, "w").write(json.dumps({"uid": os.getuid(), "found": found}))
    sys.exit(0)
if mode == "attacker":
    # Find a concurrent call's final message and try to forge or end it.
    tried = {}
    deadline = time.time() + float(prompt[1])
    while time.time() < deadline and not tried:
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                argv = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
            except OSError:
                continue
            flag = b"--output-last-message"
            if flag not in argv or int(pid) == os.getpid():
                continue
            target = argv[argv.index(flag) + 1].decode()
            if target == out:
                continue
            def attempt(name, action):
                try:
                    action()
                    tried[name] = "done"
                except OSError as error:
                    tried[name] = error.strerror
            attempt("forge", lambda: open(target, "w").write('{"verdict": "FORGED"}'))
            attempt("list_job", lambda: os.listdir(os.path.dirname(target)))
            attempt("environ", lambda: open("/proc/%s/environ" % pid).read())
            attempt("cwd", lambda: os.listdir("/proc/%s/cwd" % pid))
            if "nokill" not in prompt:
                attempt("kill", lambda: os.kill(int(pid), signal.SIGKILL))
        time.sleep(0.05)
    open(out, "w").write(json.dumps({"uid": os.getuid(), "tried": tried}))
    sys.exit(0)
answer = {"uid": os.getuid(), "home": home, "login": open(login).read(),
          "login_mode": oct(os.stat(login).st_mode)}
open(out, "w").write(json.dumps(answer))
"""

# Live processes of the Agents' users (uids 61000-61015); the Judge's PID 1
# (sleep) reaps no zombies.
AGENT_UID_PROCESSES = (
    "python3",
    "-c",
    "import os\n"
    "def live(p):\n"
    "    try:\n"
    "        stat = open('/proc/' + p + '/stat').read().rpartition(') ')[2]\n"
    "        uid = os.stat('/proc/' + p).st_uid\n"
    "        return 61000 <= uid < 61016 and stat[0] not in 'ZX'\n"
    "    except OSError:\n"
    "        return False\n"
    "print([p for p in os.listdir('/proc') if p.isdigit() and live(p)])",
)
# In sh: the escaped `sleep 300` of a stubborn call, if any is still alive.
ESCAPEES = (
    'python3 -c "import os\n'
    "def argv(p):\n"
    "    try:\n"
    "        return open('/proc/' + p + '/cmdline', 'rb').read()\n"
    "    except OSError:\n"
    "        return b''\n"
    "found = [p for p in os.listdir('/proc') if p.isdigit()"
    " and argv(p) == b'sleep\\x00300\\x00']\n"
    "print('escapees=%d' % len(found))\""
)


@pytest.fixture
def isolated_judge(tmp_path: Path, request: pytest.FixtureRequest):
    """A root, read-only-rootfs Judge with rsi-agent and a local Codex login."""
    return _rsi_agent_judge(tmp_path, request, user=None)


# A Judge that is not root runs as the current user, so that only Landlock,
# not file modes, keeps each call's Agent out of the Judge's files.
NON_ROOT = f"{os.getuid()}:{os.getgid()}"


@pytest.fixture
def landlock_judge(tmp_path: Path, request: pytest.FixtureRequest):
    """The same Judge as a non-root user: rsi-agent confines with Landlock."""
    return _rsi_agent_judge(tmp_path, request, user=NON_ROOT)


def _rsi_agent_judge(tmp_path: Path, request: pytest.FixtureRequest, *, user):
    client = _docker(PYTHON_IMAGE)
    task_source = tmp_path / "task-source"
    engine_root = tmp_path / "engine-root"
    task_source.mkdir()
    engine_root.mkdir()
    verifier = engine_root / "verifier"
    verifier.mkdir(mode=0o700)
    agent_dir = install_judge_agent(engine_root)
    material = AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options="rw,nosuid,nodev,noexec,mode=0700",
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=json.dumps({"access_token": LOGIN_SECRET}).encode(),
                    ),
                ),
            ),
        ),
        secret_values=frozenset({LOGIN_SECRET}),
    )
    oneshot = rsi_loop_oneshot(
        RSILoopConfig(), AgentPlan(name="codex", network=NetworkPolicy())
    )
    launch = build_launch_spec(oneshot, material, {LOGIN_SECRET})
    if user is not None:
        # The Judge's own outputs: writable by its user, hidden by Landlock.
        verifier.chmod(0o777)
    runtime = DockerContainerRuntime(
        client,
        run_id=f"rsi-agent-{uuid.uuid4().hex[:8]}",
        task_id="isolation",
        role="judge",
        task_source_dir=task_source,
        allowed_mount_roots=(engine_root,),
        omit_gpu_device_requests_for_tests=True,
        judge_model_access=True,
    )
    judge = runtime.create(
        ContainerSpec(
            image=PYTHON_IMAGE,
            command=("sleep", "infinity"),
            user=user,
            # A public Judge's shape: read-only rootfs, no-new-privileges.
            read_only_rootfs=True,
            tmpfs=(
                # A hidden root Judge's tests: closed to the Agent's user.
                ContainerTmpfs(
                    target=PurePosixPath("/tests"),
                    options="rw,exec,nosuid,nodev,mode=0700",
                ),
                PUBLIC_JUDGE_TMP,
                # Where the fake Agent CLI lives; Docker's /tmp is noexec.
                ContainerTmpfs(
                    target=PurePosixPath("/opt/fake"),
                    options="rw,exec,nosuid,nodev,mode=0755",
                ),
                *(mount.tmpfs for mount in material.mounts),
                *JUDGE_AGENT_TMPFS,
            ),
            mounts=(
                ContainerMount(source=verifier, target=PurePosixPath("/logs/verifier")),
                ContainerMount(
                    source=agent_dir,
                    target=PurePosixPath("/opt/rsi-agent"),
                    read_only=True,
                ),
                ContainerMount(
                    source=agent_dir / "rsi-agent",
                    target=PurePosixPath("/usr/local/bin/rsi-agent"),
                    read_only=True,
                ),
            ),
        )
    )
    request.addfinalizer(lambda: runtime.remove(judge))
    container = client.containers.get(judge.container_id)
    # Started without a network policy: the container has no network at all.
    container.start()
    runtime.inject_agent_auth(judge, material, owner=user)
    runtime.inject_judge_agent(judge, launch, owner=user)
    encoded = base64.b64encode(FAKE_CODEX.encode()).decode()
    code, output = container.exec_run(
        (
            "sh",
            "-c",
            f"mkdir -p /opt/fake/bin /tmp/q && echo {encoded} | base64 -d > "
            "/opt/fake/bin/codex && chmod 0755 /opt/fake/bin/codex"
            + ("" if user is None else f" && chown {user} /tmp/q"),
        ),
        user="root",
    )
    assert code == 0, output
    return container


def _call(name: str, prompt: str, *options: str) -> str:
    """One rsi-agent call as a root Judge procedure makes it, in sh."""
    return (
        f"mkdir -p /tmp/q/{name} && echo '{prompt}' > /tmp/q/{name}.md && "
        "PATH=/opt/fake/bin:$PATH rsi-agent exec "
        f"--prompt-file /tmp/q/{name}.md --output /tmp/q/{name}.json "
        f"--workdir /tmp/q/{name} {' '.join(options)}"
    )


def test_rsi_agent_runs_the_agent_as_its_own_user_with_a_private_login(
    isolated_judge,
) -> None:
    container = isolated_judge

    code, output = container.exec_run(
        ("sh", "-c", _call("report", "report") + " && cat /tmp/q/report.json"),
        user="root",
    )

    assert code == 0, output
    answer = json.loads(output.decode().strip().splitlines()[-1])
    assert answer["uid"] == 61000
    assert answer["home"].startswith("/run/rsi-agent/jobs/")
    # The Agent's own copy of the login, private to it; redacted on output.
    assert answer["login"] == '{"access_token": "[REDACTED]"}'
    assert answer["login_mode"] == "0o100600"
    # Neither the login store nor the launch spec is readable by another user.
    for path in ("/home/agent/.codex/auth.json", "/run/rsi-agent/private/launch.json"):
        code, _ = container.exec_run(("cat", path), user="nobody")
        assert code != 0, path
        code, _ = container.exec_run(("cat", path), user="61000")
        assert code != 0, path
    code, _ = container.exec_run(("ls", "/tests"), user="61000")
    assert code != 0
    code, _ = container.exec_run(("touch", "/logs/verifier/x"), user="61000")
    assert code != 0
    # The per-call HOME is gone, and no Agent process remains.
    code, output = container.exec_run(("ls", "-A", "/run/rsi-agent/jobs"), user="root")
    assert code == 0 and output.decode().strip() == ""
    code, output = container.exec_run(AGENT_UID_PROCESSES, user="root")
    assert output.decode().strip() == "[]"


def test_rsi_agent_timeout_ends_every_agent_process(isolated_judge) -> None:
    container = isolated_judge

    code, output = container.exec_run(
        (
            "sh",
            "-c",
            "start=$(date +%s); "
            + _call("stubborn", "stubborn", "--timeout", "2")
            + '; echo "exit=$? seconds=$(( $(date +%s) - start ))"',
        ),
        user="root",
    )

    text = output.decode()
    assert "exit=124" in text, text
    seconds = int(text.rsplit("seconds=", 1)[1].split()[0])
    assert seconds < 20, text
    code, output = container.exec_run(AGENT_UID_PROCESSES, user="root")
    assert output.decode().strip() == "[]"


def test_concurrent_calls_survive_a_sibling_timeout(isolated_judge) -> None:
    container = isolated_judge
    script = " ".join(
        [
            *(
                f"({_call(f'slow{n}', 'sleep 10')}; echo slow{n}=$?) &"
                for n in range(3)
            ),
            f"({_call('stubborn', 'stubborn', '--timeout', '1')}; echo stubborn=$?) &",
            "wait",
        ]
    )

    code, output = container.exec_run(("sh", "-c", script), user="root")

    text = output.decode()
    assert "stubborn=124" in text, text
    for n in range(3):
        assert f"slow{n}=0" in text, text
    code, output = container.exec_run(AGENT_UID_PROCESSES, user="root")
    assert output.decode().strip() == "[]"


def test_an_escaped_agent_process_ends_with_its_own_call(isolated_judge) -> None:
    """A call's uid is its own, so its escapees die with it, not with the last."""
    container = isolated_judge
    script = " ".join(
        [
            f"({_call('slow', 'sleep 20')}; echo slow=$?) &",
            # The stubborn call ends after its 1 s timeout and 5 s grace; its
            # setsid child left the group but not the call's uid.
            f"({_call('stubborn', 'stubborn', '--timeout', '1')}; echo stubborn=$?;"
            f" {ESCAPEES}) &",
            "wait",
        ]
    )

    code, output = container.exec_run(("sh", "-c", script), user="root")

    text = output.decode()
    assert "stubborn=124" in text and "slow=0" in text, text
    # Checked while the slow sibling was still running.
    assert text.index("escapees=0") < text.index("slow=0"), text


def test_concurrent_calls_run_as_distinct_users_and_cannot_reach_each_other(
    isolated_judge,
) -> None:
    container = isolated_judge
    script = " ".join(
        [
            f"({_call('victim', 'victim 6')}; echo victim=$?) &",
            f"({_call('attacker', 'attacker 5')}; echo attacker=$?) &",
            *(f"({_call(f'peer{n}', 'sleep 2')}; echo peer{n}=$?) &" for n in range(2)),
            "wait; cat /tmp/q/victim.json; echo; cat /tmp/q/attacker.json; echo;"
            " for n in 0 1; do cat /tmp/q/peer$n.json; echo; done",
        ]
    )

    code, output = container.exec_run(("sh", "-c", script), user="root")

    text = output.decode()
    for name in ("victim", "attacker", "peer0", "peer1"):
        assert f"{name}=0" in text, text
    victim, attacker, *peers = (
        json.loads(line) for line in text.splitlines() if line.startswith("{")
    )
    assert victim == {"verdict": "honest"}
    # The attacker found its sibling's final message, and every move failed.
    assert set(attacker["tried"]) == {"forge", "list_job", "environ", "cwd", "kill"}
    assert "done" not in attacker["tried"].values(), attacker
    uids = {attacker["uid"], *(peer["uid"] for peer in peers)}
    assert len(uids) == 3 and all(61000 <= uid < 61016 for uid in uids), uids
    code, output = container.exec_run(AGENT_UID_PROCESSES, user="root")
    assert output.decode().strip() == "[]"


def test_the_agent_log_never_holds_a_credential(isolated_judge) -> None:
    container = isolated_judge

    code, output = container.exec_run(
        (
            "sh",
            "-c",
            _call("leak", "leak", "--log", "/logs/verifier/agent.log")
            + " && cat /logs/verifier/agent.log /tmp/q/leak.json",
        ),
        user="root",
    )

    text = output.decode()
    assert code == 0, text
    assert LOGIN_SECRET not in text
    # The login was printed on stdout and stderr, both kept, both redacted.
    stdout, stderr = text.split("--- stderr ---")
    redacted = '{"access_token": "[REDACTED]"}'
    assert redacted in stdout and redacted in stderr, text


def test_a_judge_that_is_not_root_isolates_with_landlock(landlock_judge) -> None:
    container = landlock_judge

    code, output = container.exec_run(("rsi-agent", "isolation"), user=NON_ROOT)

    assert code == 0, output
    report = json.loads(output)
    assert report["mode"] == "landlock" and report["landlock_abi"] >= 1


def test_a_confined_agent_reaches_only_its_own_call(landlock_judge) -> None:
    container = landlock_judge

    code, output = container.exec_run(
        ("sh", "-c", _call("probe", "probe") + " && cat /tmp/q/probe.json"),
        user=NON_ROOT,
    )

    assert code == 0, output
    answer = json.loads(output.decode().strip().splitlines()[-1])
    # The Judge's own uid: Landlock, not file modes, keeps it out.
    assert answer["uid"] == os.getuid()
    denied = "Permission denied"
    assert answer["found"] == {
        "launch_spec": denied,
        "login_store": denied,
        "write_verifier": denied,
        "list_jobs": denied,
        "list_tests": denied,
        "own_home": "done",
        "own_tmp": "done",
    }
    code, output = container.exec_run(("ls", "-A", "/logs/verifier"), user="root")
    assert output.decode().strip() == ""
    # The per-call job directory is gone.
    code, output = container.exec_run(("ls", "-A", "/run/rsi-agent/jobs"), user="root")
    assert code == 0 and output.decode().strip() == ""


def test_a_confined_agent_cannot_read_or_forge_a_sibling_call(landlock_judge) -> None:
    container = landlock_judge
    script = " ".join(
        [
            f"({_call('victim', 'victim 6')}; echo victim=$?) &",
            f"({_call('attacker', 'attacker 5 nokill')}; echo attacker=$?) &",
            "wait; cat /tmp/q/victim.json; echo; cat /tmp/q/attacker.json; echo",
        ]
    )

    code, output = container.exec_run(("sh", "-c", script), user=NON_ROOT)

    text = output.decode()
    assert "victim=0" in text and "attacker=0" in text, text
    victim, attacker = (
        json.loads(line) for line in text.splitlines() if line.startswith("{")
    )
    assert victim == {"verdict": "honest"}
    # Same uid, yet every move on the sibling's files and memory failed.
    assert set(attacker["tried"]) == {"forge", "list_job", "environ", "cwd"}, attacker
    assert "done" not in attacker["tried"].values(), attacker


def test_a_confined_agents_escaped_process_ends_with_its_call(landlock_judge) -> None:
    container = landlock_judge

    code, output = container.exec_run(
        (
            "sh",
            "-c",
            _call("stubborn", "stubborn", "--timeout", "1")
            + f"; echo stubborn=$?; {ESCAPEES}",
        ),
        user=NON_ROOT,
    )

    text = output.decode()
    assert "stubborn=124" in text and "escapees=0" in text, text


def test_real_judge_credentials_belong_to_the_verifier_user_alone(
    tmp_path: Path,
) -> None:
    client = _docker()
    task_source = tmp_path / "task-source"
    engine_root = tmp_path / "engine-root"
    task_source.mkdir()
    engine_root.mkdir()
    options = "rw,nosuid,nodev,noexec,mode=0700"
    material = AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"), options=options
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=json.dumps({"access_token": LOGIN_SECRET}).encode(),
                    ),
                ),
            ),
        ),
        secret_values=frozenset({LOGIN_SECRET}),
    )
    runtime = DockerContainerRuntime(
        client,
        run_id="judge-model-access",
        task_id="local-auth",
        role="judge",
        task_source_dir=task_source,
        allowed_mount_roots=(engine_root,),
        omit_gpu_device_requests_for_tests=True,
        judge_model_access=True,
    )
    judge = runtime.create(
        ContainerSpec(
            image=IMAGE,
            command=("sleep", "infinity"),
            user="ubuntu",
            tmpfs=tuple(mount.tmpfs for mount in material.mounts),
        )
    )
    try:
        # Start without a Judge network policy: this test only checks the
        # credential hand-over (the container has no network at all).
        container = client.containers.get(judge.container_id)
        container.start()
        runtime.inject_agent_auth(judge, material, owner="ubuntu")
        code, shown = container.exec_run(
            ("cat", "/home/agent/.codex/auth.json"), user="ubuntu"
        )
        assert code == 0 and LOGIN_SECRET.encode() in shown
        code, _ = container.exec_run(
            ("cat", "/home/agent/.codex/auth.json"), user="nobody"
        )
        assert code != 0
        code, mode = container.exec_run(
            (
                "stat",
                "-c",
                "%a %U",
                "/home/agent/.codex",
                "/home/agent/.codex/auth.json",
            )
        )
        assert mode.decode().split("\n")[:2] == ["700 ubuntu", "600 ubuntu"]
    finally:
        runtime.remove(judge)


def test_real_model_access_judge_resolves_only_the_pinned_names(
    tmp_path: Path,
) -> None:
    """A routable Judge bridge must not reach the host's resolver."""
    client = _docker()
    task_source = tmp_path / "task-source"
    engine_root = tmp_path / "engine-root"
    task_source.mkdir()
    engine_root.mkdir()
    common = {
        "run_id": f"judge-dns-{uuid.uuid4().hex[:8]}",
        "task_id": "dns",
        "role": "judge",
        "task_source_dir": task_source,
        "allowed_mount_roots": (engine_root,),
        "omit_gpu_device_requests_for_tests": True,
    }
    provisioner = DockerContainerRuntime(client, **common)
    network = provisioner.create_network("round-1", internal=False)
    runtime = DockerContainerRuntime(
        client, **common, network=network, judge_model_access=True
    )
    created = []
    try:

        def lookup(dns: tuple[str, ...], name: str) -> tuple[int, str]:
            judge = runtime.create(
                ContainerSpec(
                    image=IMAGE,
                    command=("sleep", "infinity"),
                    extra_hosts=(("model.test", "203.0.113.7"),),
                    dns=dns,
                )
            )
            created.append(judge)
            # Started without a policy: only name resolution is checked.
            container = client.containers.get(judge.container_id)
            container.start()
            code, shown = container.exec_run(("timeout", "60", "getent", "hosts", name))
            return code, shown.decode()

        servers = no_network_dns_servers("no-network", True)
        assert servers == (UNREACHABLE_DNS_SERVER,)
        code, shown = lookup(servers, "model.test")
        assert code == 0 and shown.split()[0] == "203.0.113.7"
        code, shown = lookup(servers, "example.com")
        assert code != 0, shown
        try:
            socket.getaddrinfo("example.com", None)
        except OSError:
            return  # The host itself resolves nothing: no leak to compare.
        # Control: Docker's default upstream does resolve external names.
        code, shown = lookup((), "example.com")
        assert code == 0, shown
    finally:
        for judge in created:
            runtime.remove(judge)
        provisioner.remove_network(network)
