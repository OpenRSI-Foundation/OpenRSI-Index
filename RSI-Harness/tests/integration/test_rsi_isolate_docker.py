"""Real Docker: rsi-isolate in a root Judge (uid mode) and a non-root one
(Landlock), and in a whole Harness round as each."""

from __future__ import annotations

import base64
import json
import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.judge_agent import (
    JUDGE_ISOLATE_CODE_TARGET,
    JUDGE_ISOLATE_LAUNCHER_TARGET,
    install_judge_isolate,
)
from rsi_harness.models import (
    CompileOptions,
    ContainerMount,
    ContainerSpec,
    ContainerTmpfs,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_judge_model_access_docker import PYTHON_IMAGE, _docker
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

NON_ROOT = "1000:1000"
# The command: who it is, and what it reaches of the Judge.
PROBE = r"""
import json, os, sys
def attempt(action):
    try:
        action()
        return "ok"
    except OSError as error:
        return error.strerror
def write(path):
    with open(path, "w") as stream:
        stream.write("x")
status = open("/proc/self/status").read()
found = {
    "uid": os.getuid(), "gid": os.getgid(), "groups": os.getgroups(),
    "no_new_privs": [l.split()[1] for l in status.splitlines()
                     if l.startswith("NoNewPrivs:")][0],
    "read_tests": attempt(lambda: open("/tests/secret").read()),
    "list_verifier": attempt(lambda: os.listdir("/logs/verifier")),
    "write_verifier": attempt(lambda: write("/logs/verifier/reward.json")),
    "read_input": attempt(lambda: open("/srv/inputs/data").read()),
    "read_other": attempt(lambda: open("/srv/other/data").read()),
    "write_scratch": attempt(lambda: write("/srv/scratch/out")),
    "scratch_owner": os.stat("/srv/scratch").st_uid,
    "list_tmp": attempt(lambda: os.listdir("/tmp")),
}
print(json.dumps(found))
"""
# A child in the command's group and a double-forked daemon; their pids.
SPAWN = r"""
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
read, write = os.pipe()
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        os.write(write, str(os.getpid()).encode())
        time.sleep(300)
    os._exit(0)
os.wait()
os.close(write)
daemon = os.read(read, 32).decode()
with open(sys.argv[1], "w") as stream:
    stream.write("%d %s" % (child.pid, daemon))
time.sleep(float(sys.argv[2]))
"""
ALIVE = (
    "import sys\n"
    "def alive(pid):\n"
    "    try:\n"
    "        return open('/proc/%s/stat' % pid).read().rpartition(') ')[2][0]"
    " not in 'ZX'\n"
    "    except OSError:\n"
    "        return False\n"
    "print([p for p in open(sys.argv[1]).read().split() if alive(p)])"
)
# Three calls at once: the uid each one ran as.
CONCURRENT = (
    "import subprocess\n"
    "calls = [subprocess.Popen(['rsi-isolate', 'run', '--', 'python3', '-c',"
    " 'import os, time; print(os.getuid()); time.sleep(2)'],"
    " stdout=subprocess.PIPE, text=True) for _ in range(3)]\n"
    "print(sorted(int(call.communicate()[0]) for call in calls))"
)
# An escapee that keeps moving: each process forks and exits, so no pid stays
# long enough for a scan of /proc to kill it.
HOPPER = (
    "import os\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    while True:\n"
    "        try:\n"
    "            if os.fork():\n"
    "                os._exit(0)\n"
    "        except OSError:\n"
    "            pass\n"
)
# Processes of the isolated users left in the Judge.
ISOLATED_LEFT = (
    "import os\n"
    "left = []\n"
    "for pid in filter(str.isdigit, os.listdir('/proc')):\n"
    "    try:\n"
    "        lines = open('/proc/%s/status' % pid).read().splitlines()\n"
    "    except OSError:\n"
    "        continue\n"
    "    fields = dict(line.split(':', 1) for line in lines if ':' in line)\n"
    "    uid = int(fields['Uid'].split()[0])\n"
    "    if 62000 <= uid < 62064 and fields['State'].split()[0] not in 'ZX':\n"
    "        left.append(pid)\n"
    "print(left)\n"
)
# Plants a setuid file and a hard link to a file outside the grant.
PLANT = (
    "import os\n"
    "os.link('/srv/shared/data', '/srv/handback/link')\n"
    "open('/srv/handback/planted', 'w').close()\n"
    "os.chmod('/srv/handback/planted', 0o6755)\n"
    "print(oct(os.stat('/srv/handback/planted').st_mode & 0o7777))\n"
)
NETWORK = (
    "import socket\n"
    "try:\n"
    "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    "    print('inet ok')\n"
    "except OSError as error:\n"
    "    print('inet', error.strerror)\n"
)


def _isolated_judge(tmp_path: Path, request: pytest.FixtureRequest, *, user):
    """A Judge without model access, as RSI Harness creates one: rsi-isolate
    mounted read-only, hidden tests, the Judge's outputs."""
    client = _docker(PYTHON_IMAGE)
    task_source = tmp_path / "task-source"
    engine_root = tmp_path / "engine-root"
    task_source.mkdir()
    engine_root.mkdir()
    verifier = engine_root / "verifier"
    verifier.mkdir(mode=0o700)
    if user is not None:
        verifier.chmod(0o777)
    isolate = install_judge_isolate(engine_root)
    runtime = DockerContainerRuntime(
        client,
        run_id=f"rsi-isolate-{uuid.uuid4().hex[:8]}",
        task_id="isolate",
        role="judge",
        task_source_dir=task_source,
        allowed_mount_roots=(engine_root,),
        omit_gpu_device_requests_for_tests=True,
    )
    judge = runtime.create(
        ContainerSpec(
            image=PYTHON_IMAGE,
            command=("sleep", "infinity"),
            user=user,
            tmpfs=(
                ContainerTmpfs(
                    target=PurePosixPath("/tests"),
                    # A hidden root Judge's tests; a non-root Judge's own.
                    options="rw,exec,nosuid,nodev,mode="
                    + ("0700" if user is None else "1777"),
                ),
            ),
            mounts=(
                ContainerMount(source=verifier, target=PurePosixPath("/logs/verifier")),
                ContainerMount(
                    source=isolate, target=JUDGE_ISOLATE_CODE_TARGET, read_only=True
                ),
                ContainerMount(
                    source=isolate / "rsi-isolate",
                    target=JUDGE_ISOLATE_LAUNCHER_TARGET,
                    read_only=True,
                ),
            ),
        )
    )
    request.addfinalizer(lambda: runtime.remove(judge))
    container = client.containers.get(judge.container_id)
    # No network policy: the container has no network at all.
    container.start()
    owner = "root" if user is None else user
    setup = (
        "set -eu\n"
        "mkdir -p /opt/probe /srv/inputs /srv/other /srv/scratch /srv/closed\n"
        "echo input > /srv/inputs/data; echo other > /srv/other/data\n"
        "chmod 0755 /srv/inputs; chmod 0700 /srv/closed\n"
        f"chown {owner} /srv/inputs /srv/other /srv/scratch /srv/closed\n"
    )
    _run(container, ("sh", "-c", setup), user="root")
    for name, content in (("probe.py", PROBE), ("spawn.py", SPAWN)):
        _put(container, f"/opt/probe/{name}", content)
    _run(
        container,
        ("sh", "-c", "echo hidden > /tests/secret; echo 0 > /tmp/judge-note"),
        user=owner,
    )
    return container, owner


def _put(container, path: str, content: str) -> None:
    _run(
        container,
        ("python3", "-c", "import sys; open(sys.argv[1], 'w').write(sys.argv[2])")
        + (path, content),
        user="root",
    )


def _run(container, argv, *, user, expect=0) -> str:
    code, output = container.exec_run(argv, user=user, demux=False)
    text = output.decode(errors="replace")
    if expect is not None:
        assert code == expect, text
    return text


def _isolate(container, owner, *arguments, expect=0) -> tuple[int, str, str]:
    code, (stdout, stderr) = container.exec_run(
        ("rsi-isolate", *arguments), user=owner, demux=True
    )
    stdout = (stdout or b"").decode(errors="replace")
    stderr = (stderr or b"").decode(errors="replace")
    if expect is not None:
        assert code == expect, stdout + stderr
    return code, stdout, stderr


def _probe(container, owner) -> dict:
    _, stdout, _ = _isolate(
        container,
        owner,
        "run",
        "--read",
        "/srv/inputs",
        "--write",
        "/srv/scratch",
        "--",
        "python3",
        "-I",
        "/opt/probe/probe.py",
    )
    return json.loads(stdout)


def _common_checks(container, owner) -> None:
    """What every mode gives: no network, a timeout and no survivors."""
    _, stdout, _ = _isolate(
        container, owner, "run", "--no-network", "--", "python3", "-c", NETWORK
    )
    assert stdout.strip() == "inet Operation not permitted"
    _, stdout, _ = _isolate(container, owner, "run", "--", "python3", "-c", NETWORK)
    assert stdout.strip() == "inet ok"
    for timeout, expect in ((None, 0), ("1", 124)):
        _run(container, ("rm", "-f", "/srv/scratch/pids"), user="root")
        arguments = ["run", "--write", "/srv/scratch"]
        if timeout is not None:
            arguments += ["--timeout", timeout]
        _isolate(
            container,
            owner,
            *arguments,
            "--",
            "python3",
            "-I",
            "/opt/probe/spawn.py",
            "/srv/scratch/pids",
            "0" if timeout is None else "300",
            expect=expect,
        )
        alive = _run(
            container,
            ("python3", "-c", ALIVE, "/srv/scratch/pids"),
            user="root",
        )
        assert alive.strip() == "[]", timeout


def test_a_root_judge_runs_each_command_as_its_own_user(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    container, owner = _isolated_judge(tmp_path, request, user=None)

    _, stdout, _ = _isolate(container, owner, "isolation")
    mode = json.loads(stdout)
    assert mode["mode"] == "uid" and mode["landlock_abi"] > 0
    found = _probe(container, owner)

    assert 62000 <= found["uid"] < 62064 and found["gid"] == found["uid"]
    assert found["groups"] == [] and found["no_new_privs"] == "1"
    denied = "Permission denied"
    assert {key: found[key] for key in found if key.startswith(("read", "list"))} == {
        "read_tests": denied,
        "list_verifier": denied,
        "read_input": "ok",
        "read_other": denied,
        "list_tmp": denied,
    }
    assert found["write_verifier"] == denied
    assert found["write_scratch"] == "ok"
    # The write grant was the command's during the call, and is root's again.
    assert found["scratch_owner"] == found["uid"]
    owners = _run(
        container, ("stat", "-c", "%u", "/srv/scratch", "/srv/scratch/out"), user="root"
    )
    assert owners.split() == ["0", "0"]
    # Another user cannot reach a 0700 input: the call is refused up front.
    code, _, stderr = _isolate(
        container, owner, "run", "--read", "/srv/closed", "--", "true", expect=125
    )
    assert "/srv/closed (mode 0700) is not open to other users" in stderr
    _common_checks(container, owner)
    concurrent = json.loads(
        _run(container, ("python3", "-c", CONCURRENT), user=owner).strip()
    )
    assert len(set(concurrent)) == 3
    assert all(62000 <= uid < 62064 for uid in concurrent)
    # No private HOME is left; only the slots' lock directory stays.
    left = _run(
        container,
        ("python3", "-c", "import os; print(sorted(os.listdir('/tmp')))"),
        user="root",
    )
    assert left.strip() == "['.rsi-isolate-slots', 'judge-note']"

    # An escapee that outruns /proc scans still ends with the call, and the
    # slot's uid stays usable.
    for _ in range(2):
        _isolate(container, owner, "run", "--", "python3", "-c", HOPPER)
        left = _run(container, ("python3", "-c", ISOLATED_LEFT), user="root")
        assert left.strip() == "[]"


def test_a_non_root_judge_confines_each_command_with_landlock(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    container, owner = _isolated_judge(tmp_path, request, user=NON_ROOT)

    _, stdout, _ = _isolate(container, owner, "isolation")
    mode = json.loads(stdout)
    assert mode["mode"] == "landlock" and mode["landlock_abi"] > 0
    found = _probe(container, owner)

    assert found["uid"] == 1000 and found["no_new_privs"] == "1"
    denied = "Permission denied"
    assert {key: found[key] for key in found if key.startswith(("read", "list"))} == {
        "read_tests": denied,
        "list_verifier": denied,
        "read_input": "ok",
        "read_other": denied,
        "list_tmp": denied,
    }
    assert found["write_verifier"] == denied
    assert found["write_scratch"] == "ok"
    _common_checks(container, owner)


# A seccomp profile that refuses the Landlock syscalls (444-446).
NO_LANDLOCK = {
    "defaultAction": "SCMP_ACT_ALLOW",
    "syscalls": [
        {
            "names": [
                "landlock_create_ruleset",
                "landlock_add_rule",
                "landlock_restrict_self",
            ],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 38,
        }
    ],
}


def _judge_without_landlock(tmp_path: Path, user: str | None):
    client = _docker(PYTHON_IMAGE)
    isolate = install_judge_isolate(tmp_path)
    return client.containers.run(
        PYTHON_IMAGE,
        ("sleep", "infinity"),
        user=user,
        detach=True,
        network_disabled=True,
        security_opt=["seccomp=" + json.dumps(NO_LANDLOCK)],
        tmpfs={"/tests": "mode=0755"},
        volumes={
            str(isolate): {"bind": str(JUDGE_ISOLATE_CODE_TARGET), "mode": "ro"},
            str(isolate / "rsi-isolate"): {
                "bind": str(JUDGE_ISOLATE_LAUNCHER_TARGET),
                "mode": "ro",
            },
        },
        labels={"rsi-harness.test": "rsi-isolate"},
    )


def test_without_landlock_a_non_root_judge_fails_closed(tmp_path: Path) -> None:
    container = _judge_without_landlock(tmp_path, NON_ROOT)
    try:
        for arguments in (("isolation",), ("run", "--", "true")):
            _, _, stderr = _isolate(container, NON_ROOT, *arguments, expect=123)
            assert "cannot isolate the command" in stderr
            assert "ENOSYS" in stderr
        code, (stdout, stderr) = container.exec_run(
            ("rsi-isolate", "run", "--", "python3", "-c", "print('ran')"),
            user=NON_ROOT,
            environment={"RSI_JUDGE_ISOLATION_FALLBACK": "same-user"},
            demux=True,
        )
        assert code == 0, stderr
        assert stdout.decode().strip() == "ran"
        assert b"same-user fallback" in stderr
    finally:
        container.remove(force=True)


def test_without_landlock_a_root_judge_needs_closed_tests(tmp_path: Path) -> None:
    """Only file permissions confine the isolated user then."""
    container = _judge_without_landlock(tmp_path, None)
    try:
        _run(container, ("sh", "-c", "echo hidden > /tests/secret"), user="root")
        for arguments in (("isolation",), ("run", "--", "true")):
            _, _, stderr = _isolate(container, "root", *arguments, expect=123)
            assert "/tests (mode 0755) is open to other users" in stderr
        _run(container, ("chmod", "0700", "/tests"), user="root")
        _, stdout, _ = _isolate(container, "root", "isolation")
        assert json.loads(stdout)["mode"] == "uid"
        assert json.loads(stdout)["landlock_abi"] == 0
        _, stdout, _ = _isolate(
            container,
            "root",
            "run",
            "--",
            "python3",
            "-c",
            "import os\ntry:\n    open('/tests/secret')\nexcept OSError as e:\n"
            "    print(os.getuid(), e.strerror)",
        )
        uid, _, reason = stdout.strip().partition(" ")
        assert 62000 <= int(uid) < 62064 and reason == "Permission denied"
        # Without Landlock a command can hard-link outside its grant. Handing
        # the grant back re-owns only the isolated user's entries, and
        # clears setuid and setgid bits.
        setup = (
            "set -eu\n"
            "mkdir /srv/handback /srv/shared; echo x > /srv/shared/data\n"
            "chmod 0666 /srv/shared/data; chown 1000:1000 /srv/handback\n"
        )
        _run(container, ("sh", "-c", setup), user="root")
        _, stdout, _ = _isolate(
            container,
            "root",
            "run",
            "--write",
            "/srv/handback",
            "--",
            "python3",
            "-c",
            PLANT,
        )
        assert stdout.strip() == "0o6755"
        found = _run(
            container,
            ("stat", "-c", "%n %u:%g %a", "/srv/shared/data", "/srv/handback/planted"),
            user="root",
        )
        assert found.split("\n")[:2] == [
            "/srv/shared/data 0:0 666",
            "/srv/handback/planted 1000:1000 755",
        ]
    finally:
        container.remove(force=True)


# -- a whole Harness round ------------------------------------------------------

# The Judge stages the candidate, runs it through rsi-isolate, and scores it.
JUDGE_SCRIPT = r"""#!/bin/bash
set -u
rsi-isolate isolation > /tmp/isolation.json || exit 1
stage=$(mktemp -d /tmp/stage.XXXXXX); chmod 0755 "$stage"
scratch=$(mktemp -d /tmp/scratch.XXXXXX)
cp /workspace/candidate.py "$stage/"
output=$(rsi-isolate run --read "$stage" --write "$scratch" --no-network \
    --timeout 60 --cwd "$scratch" -- python3 -I "$stage/candidate.py")
echo "rsi-isolate exit $? output $output"
cat /tmp/isolation.json
python3 - "$output" <<'PY'
import json, sys
found = json.loads(sys.argv[1])
mode = json.load(open("/tmp/isolation.json"))["mode"]
ok = (found["answer"] == 42 and all(found["denied"].values())
      and mode in ("uid", "landlock"))
json.dump({"reward": 1 if ok else 0}, open("/logs/verifier/reward.json", "w"))
PY
"""
CANDIDATE = r"""import json, os
def denied(action):
    try:
        action()
    except OSError:
        return True
    return False
print(json.dumps({"answer": 42, "denied": {
    "write_reward": denied(lambda: open("/logs/verifier/reward.json", "w")),
    "read_tests": denied(lambda: open("/tests/test.sh").read()),
    "list_tmp": denied(lambda: os.listdir("/tmp")),
}}))
"""


@pytest.mark.parametrize("verifier_user", [None, "judge"], ids=["root", "non-root"])
def test_a_harness_judge_runs_the_candidate_through_rsi_isolate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    verifier_user: str | None,
) -> None:
    """Full production run as the current user; only iptables is faked."""
    client = _docker(PYTHON_IMAGE)
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    # The non-root Verifier user has the invoking uid, which owns the round's
    # /logs/verifier.
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PYTHON_IMAGE}\n"
        f"RUN useradd --uid {os.getuid()} --create-home judge && mkdir -p /workspace\n"
        "WORKDIR /workspace\n"
    )
    if verifier_user is not None:
        config = task / "task.toml"
        config.write_text(
            config.read_text().replace(
                "[verifier]\n", f'[verifier]\nuser = "{verifier_user}"\n'
            )
        )
    (task / "tests" / "test.sh").write_text(JUDGE_SCRIPT)
    data = tmp_path / "data"
    candidate = base64.b64encode(CANDIDATE.encode()).decode()

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            # Work writes the candidate and submits once.
            script = (
                "set -eu\n"
                f"echo {candidate} | base64 -d > /workspace/candidate.py\n"
                "chmod 0644 /workspace/candidate.py\n"
                "rsi-submit\n"
            )
            return replace(prepared, command=("/bin/bash", "-lc", script))

    # Keep real image preparation, but no model CLI download or API call.
    monkeypatch.setattr(
        CodexAgent,
        "install_cmds",
        [
            "printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/codex && "
            "chmod 0755 /usr/local/bin/codex"
        ],
    )
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
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
            agent_name="codex",
            model="model-test",
            gpu_selectors=("0",),
            options=CompileOptions(agent_name="codex", primary_reward="reward"),
        )
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    feedback = (run_root / "feedback" / "agent-1.log").read_text()
    assert result.status == RunStatus.COMPLETED, feedback
    assert result.best_score == 1, feedback
    assert "rsi-isolate exit 0" in feedback
    mode = "uid" if verifier_user is None else "landlock"
    assert f'"mode": "{mode}"' in feedback
