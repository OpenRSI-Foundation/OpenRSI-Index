"""rsi-isolate, run as the current (non-root) user: Landlock mode.

The uid mode of a root Judge, and the program as RSI Harness mounts it, run
in the Docker tests (tests/integration/test_rsi_isolate_docker.py).
"""

from __future__ import annotations

import ast
import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rsi_harness.judge_agent import (
    JUDGE_AGENT_UIDS,
    JUDGE_ISOLATE_SLOTS,
    JUDGE_ISOLATE_UID,
    JUDGE_ISOLATE_UIDS,
    judge_agent_source,
    rsi_isolate,
)
from rsi_harness.judge_agent import judge_isolation as isolation
from tests.judge_agent.test_judge_isolation import _filter as _bpf
from tests.landlock_helpers import native_signal_compatibility

PROGRAM = Path(__file__).parents[2] / "src/rsi_harness/judge_agent/rsi_isolate.py"
SHARED = PROGRAM.with_name("judge_isolation.py")
# The commands' Python: the base interpreter, as a virtual environment's
# pyvenv.cfg lies outside what a command may read.
PYTHON = os.path.realpath(sys.executable)
LANDLOCK = isolation.landlock_abi(ctypes.CDLL(None, use_errno=True))[0]
requires_landlock = pytest.mark.skipif(
    LANDLOCK < 1 or os.geteuid() == 0, reason="needs Landlock, without root"
)
# What a command reports: each probe's outcome, "ok" or the error.
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
probes = json.loads(sys.argv[1])
found = {}
for name, (kind, path) in probes.items():
    path = path.replace("HOME/", os.environ.get("HOME", "?") + "/")
    if kind == "read":
        found[name] = attempt(lambda: open(path).read())
    elif kind == "list":
        found[name] = attempt(lambda: os.listdir(path))
    else:
        found[name] = attempt(lambda: write(path))
found["env"] = sorted(os.environ)
found["home"] = os.environ.get("HOME")
found["cwd"] = os.getcwd()
print(json.dumps(found))
"""


def _operator_environment(env=None):
    # This helper acts as the test operator. Explicit env overrides below
    # let policy tests exercise unset/disabled independently.
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        isolation.ALLOW_MISSING_SIGNAL_ENV: str(int(native_signal_compatibility())),
    }
    environment.update(env or {})
    environment = {
        key: value for key, value in environment.items() if value is not None
    }
    return environment


def _run(*arguments, env=None, program=PROGRAM, **options):
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(program), *arguments],
        env=_operator_environment(env),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        **options,
    )


def _probe(tmp_path, probes, *grants, **options):
    script = tmp_path / "bin" / "probe.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(PROBE)
    result = _run(
        "run",
        "--read",
        str(script.parent),
        *grants,
        "--",
        PYTHON,
        "-I",
        str(script),
        json.dumps(probes),
        **options,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[0]
    except FileNotFoundError:
        return True
    return state in ("Z", "X")


def _wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _gone(pid):
        return True
    os.kill(pid, signal.SIGKILL)
    return False


# -- the mode --------------------------------------------------------------------------


@requires_landlock
def test_isolation_reports_landlock_and_the_operator_fallback():
    result = _run("isolation")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "mode": "landlock",
        "landlock_abi": LANDLOCK,
        "fallback": "none",
        "landlock_scoped": (1 if native_signal_compatibility() else 3)
        if LANDLOCK >= 6
        else 0,
        "landlock_signal_scope": LANDLOCK >= 6 and not native_signal_compatibility(),
    }
    allowed = _run("isolation", env={"RSI_JUDGE_ISOLATION_FALLBACK": "same-user"})
    assert json.loads(allowed.stdout)["fallback"] == "same-user"
    assert json.loads(allowed.stdout)["mode"] == "landlock"


# -- what the command reaches ----------------------------------------------------------


@requires_landlock
def test_the_command_reads_and_writes_only_its_grants(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "data.txt").write_text("input")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "reference.json").write_text("{}")
    (tmp_path / "outputs").mkdir()
    home_file = next(
        (path for path in Path.home().iterdir() if path.is_file()), Path.home()
    )

    found = _probe(
        tmp_path,
        {
            "read_grant": ("read", str(inputs / "data.txt")),
            "write_into_read_grant": ("write", str(inputs / "new.txt")),
            "write_grant": ("write", str(scratch / "out.txt")),
            "read_sibling": ("read", str(tmp_path / "reference.json")),
            "write_sibling": ("write", str(tmp_path / "outputs" / "reward.json")),
            "list_tmp": ("list", "/tmp"),
            "home_dir": ("list", str(home_file.parent)),
            "usr": ("list", "/usr/lib"),
            "proc": ("read", "/proc/self/status"),
            "etc": ("read", "/etc/hostname"),
            "private_home": ("write", "HOME/note.txt"),
            "shm": ("write", f"/dev/shm/rsi-isolate-test-{os.getpid()}"),
        },
        "--read",
        str(inputs),
        "--write",
        str(scratch),
    )

    denied = "Permission denied"
    assert {key: value for key, value in found.items() if key != "env"} == {
        "read_grant": "ok",
        "write_into_read_grant": denied,
        "write_grant": "ok",
        "read_sibling": denied,
        "write_sibling": denied,
        "list_tmp": denied,
        "home_dir": denied,
        "usr": "ok",
        "proc": "ok",
        "etc": "ok",
        "private_home": "ok",
        "shm": "ok",
        "home": found["home"],
        "cwd": found["home"],
    }
    assert (scratch / "out.txt").read_text() == "x"
    assert not (tmp_path / "outputs" / "reward.json").exists()
    # The private HOME and TMPDIR are gone; what the command left in
    # /dev/shm stays until the Judge ends.
    assert found["home"].startswith("/tmp/rsi-isolate-")
    assert not os.path.exists(found["home"])
    Path(f"/dev/shm/rsi-isolate-test-{os.getpid()}").unlink()


@requires_landlock
def test_the_command_gets_only_path_home_tmpdir_and_its_env(tmp_path):
    found = _probe(
        tmp_path,
        {},
        "--env",
        "PASSED",
        "--env",
        "SET=value",
        "--env",
        "ABSENT",
        env={
            "PASSED": "1",
            "SECRET_TOKEN": "x",
            "RSI_JUDGE_ISOLATION_FALLBACK": "none",
        },
    )
    # LC_CTYPE: the command's own Python coerces the C locale.
    assert set(found["env"]) - {"LC_CTYPE"} == {
        "HOME",
        "PASSED",
        "PATH",
        "SET",
        "TMPDIR",
    }
    refused = _run(
        "run", "--env", "RSI_JUDGE_ISOLATION_FALLBACK=same-user", "--", "true"
    )
    assert refused.returncode == 125
    assert "never passed on" in refused.stderr


@requires_landlock
def test_the_working_directory_must_lie_inside_a_grant(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    found = _probe(tmp_path, {}, "--write", str(work), "--cwd", str(work))
    assert found["cwd"] == str(work)
    outside = _run("run", "--cwd", str(tmp_path), "--", "true")
    assert outside.returncode == 125
    assert "must lie inside a --read or --write" in outside.stderr


@requires_landlock
@pytest.mark.parametrize(
    ("grant", "reason"),
    [
        (("--read", "/"), "never granted"),
        (("--read", "/logs"), "never granted"),
        (("--read", "/logs/verifier"), "never granted"),
        (("--write", "/run/rsi-agent/private"), "never granted"),
        (("--write", "/usr/lib"), "never writable"),
        (("--write", "/etc/rsi-isolate-test"), "never writable"),
        (("--write", "/tests/output"), "never writable"),
        (("--write", "/home/rsi-isolate-test"), "does not exist"),
        (("--write", "/home"), "below a top-level directory"),
        (("--read", "/nonexistent/rsi-isolate"), "does not exist"),
    ],
)
def test_refused_grants_exit_125(grant, reason):
    result = _run("run", *grant, "--", "true")
    assert result.returncode == 125
    assert reason in result.stderr


@requires_landlock
def test_a_write_grant_with_hard_links_is_refused(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "reference.json").write_text("{}")
    os.link(tmp_path / "reference.json", scratch / "copy.json")
    result = _run("run", "--write", str(scratch), "--", "true")
    assert result.returncode == 125
    assert "has other hard links" in result.stderr


# -- network, processes and exit status ------------------------------------------------


NETWORK = r"""
import ctypes, json, os, socket
found = {}
for name, family, kind, protocol in (
    ("inet", socket.AF_INET, socket.SOCK_STREAM, 0),
    ("inet6", socket.AF_INET6, socket.SOCK_STREAM, 0),
    ("unix", socket.AF_UNIX, socket.SOCK_STREAM, 0),
    ("netlink_route", socket.AF_NETLINK, socket.SOCK_RAW, 0),
    ("netlink_usersock", socket.AF_NETLINK, socket.SOCK_RAW, 2),
):
    try:
        socket.socket(family, kind, protocol).close()
        found[name] = "ok"
    except OSError as error:
        found[name] = error.strerror
pairs = {"socketpair": socket.SOCK_STREAM, "socketpair_dgram": socket.SOCK_DGRAM}
for name, kind in pairs.items():
    try:
        socket.socketpair(socket.AF_UNIX, kind)
        found[name] = "ok"
    except OSError as error:
        found[name] = error.strerror
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
params = ctypes.create_string_buffer(120)
result = libc.syscall(ctypes.c_long(425), ctypes.c_long(1), params)
found["io_uring"] = "ok" if result >= 0 else os.strerror(ctypes.get_errno())
print(json.dumps(found))
"""


@requires_landlock
def test_no_network_refuses_internet_sockets_and_io_uring():
    private = {"RSI_JUDGE_NETNS": "private"}
    result = _run("run", "--no-network", "--", PYTHON, "-I", "-c", NETWORK, env=private)
    assert result.returncode == 0, result.stderr
    denied = "Operation not permitted"
    assert json.loads(result.stdout) == {
        "inet": denied,
        "inet6": denied,
        "unix": "ok",
        "netlink_route": "ok",
        "netlink_usersock": denied,
        "socketpair": "ok",
        "socketpair_dgram": "ok",
        "io_uring": denied,
    }
    open_network = json.loads(_run("run", "--", PYTHON, "-I", "-c", NETWORK).stdout)
    assert open_network["inet"] == "ok"
    # In every mode: no netlink socket but NETLINK_ROUTE.
    assert open_network["netlink_usersock"] == denied


@requires_landlock
@pytest.mark.parametrize("netns", [None, "shared", "PRIVATE"])
def test_no_network_in_a_shared_network_namespace_refuses_unix_sockets(netns):
    """Abstract sockets reach other jobs on the node: AF_UNIX goes too,
    unless Landlock (ABI 6) scopes them; a stream socketpair stays, a
    datagram one (it can reach any abstract name) goes too."""
    result = _run(
        "run",
        "--no-network",
        "--",
        PYTHON,
        "-I",
        "-c",
        NETWORK,
        env={"RSI_JUDGE_NETNS": netns},
    )
    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout)
    expected = "ok" if LANDLOCK >= 6 else "Operation not permitted"
    assert (found["unix"], found["socketpair_dgram"]) == (expected, expected)
    assert found["socketpair"] == "ok"
    # Without --no-network, AF_UNIX stays.
    open_network = _run(
        "run", "--", PYTHON, "-I", "-c", NETWORK, env={"RSI_JUDGE_NETNS": netns}
    )
    assert json.loads(open_network.stdout)["unix"] == "ok"


def test_unix_sockets_are_refused_only_where_the_judge_shares_a_netns(monkeypatch):
    class Ruleset:
        scoped = isolation.LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET

    monkeypatch.delenv("RSI_JUDGE_NETNS", raising=False)
    assert rsi_isolate._deny_unix(True, None)
    assert not rsi_isolate._deny_unix(False, None)
    assert not rsi_isolate._deny_unix(True, Ruleset())
    monkeypatch.setenv("RSI_JUDGE_NETNS", "private")
    assert not rsi_isolate._deny_unix(True, None)


@requires_landlock
def test_the_command_cannot_read_or_trace_rsi_isolate():
    """Outside the uid mode the command runs as the Judge's user: rsi-isolate
    is not dumpable, so its /proc files belong to root."""
    owner = "import os; print(os.stat('/proc/%d/environ' % os.getppid()).st_uid)"
    result = _run("run", "--", PYTHON, "-I", "-c", owner)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


# A child in the command's group, and a daemon that left it (double fork,
# new session); both pids go to the file named first.
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


@requires_landlock
@pytest.mark.parametrize("timeout", [None, 1], ids=["finished", "timeout"])
def test_everything_the_command_started_ends_with_it(tmp_path, timeout):
    pids = tmp_path / "out" / "pids"
    pids.parent.mkdir()
    arguments = ["run", "--write", str(pids.parent)]
    if timeout is not None:
        arguments += ["--timeout", str(timeout)]
    started = time.monotonic()
    result = _run(
        *arguments,
        "--",
        PYTHON,
        "-I",
        "-c",
        SPAWN,
        str(pids),
        "0" if timeout is None else "300",
    )

    assert result.returncode == (0 if timeout is None else 124), result.stderr
    if timeout is not None:
        assert "did not finish within 1 seconds" in result.stderr
        assert time.monotonic() - started < 15
    survivors = [
        int(pid) for pid in pids.read_text().split() if not _wait_gone(int(pid))
    ]
    assert survivors == []


@requires_landlock
@pytest.mark.parametrize(
    ("command", "status", "reason"),
    [
        ([PYTHON, "-c", "raise SystemExit(7)"], 7, None),
        (
            [PYTHON, "-c", "import os; os.kill(os.getpid(), 11)"],
            128 + 11,
            None,
        ),
        (["rsi-isolate-no-such-command"], 127, "command not found"),
        (["/etc/hostname"], 126, "cannot execute"),
    ],
)
def test_the_exit_status_is_the_commands_or_its_own(command, status, reason):
    result = _run("run", "--", *command)
    assert result.returncode == status, result.stderr
    if reason:
        assert reason in result.stderr
    else:
        assert "rsi-isolate:" not in result.stderr


@requires_landlock
def test_stdin_and_stdout_are_the_callers():
    result = _run(
        "run",
        "--",
        PYTHON,
        "-c",
        "import sys; sys.stdout.write(sys.stdin.read().upper())",
        input="through the pipe",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "THROUGH THE PIPE"


@requires_landlock
def test_a_passed_descriptor_stays_open(tmp_path):
    ours, theirs = socket.socketpair()
    with ours, theirs:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(PROGRAM),
                "run",
                "--pass-fd",
                str(theirs.fileno()),
                "--",
                PYTHON,
                "-c",
                "import socket, sys\n"
                "s = socket.socket(fileno=int(sys.argv[1]))\n"
                "s.sendall(s.recv(64).upper())",
                str(theirs.fileno()),
            ],
            pass_fds=(theirs.fileno(),),
            env=_operator_environment(),
            stderr=subprocess.PIPE,
            text=True,
        )
        theirs.close()
        ours.sendall(b"ping")
        ours.settimeout(20)
        assert ours.recv(64) == b"PING"
        _, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr
    closed = _run("run", "--pass-fd", "1000", "--", "true")
    assert closed.returncode == 125


@requires_landlock
def test_sigterm_ends_the_command_and_exits_143(tmp_path):
    pids = tmp_path / "out" / "pids"
    pids.parent.mkdir()
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", str(PROGRAM), "run", "--write"]
        + [str(pids.parent), "--", PYTHON, "-I", "-c", SPAWN]
        + [str(pids), "300"],
        env=_operator_environment(),
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 20
    while not (pids.exists() and pids.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=20)

    assert process.returncode == 143, stderr
    assert "terminated" in stderr
    assert [
        int(pid) for pid in pids.read_text().split() if not _wait_gone(int(pid))
    ] == []


# -- without Landlock, and below ABI 3 -------------------------------------------------


def _copy(tmp_path: Path, landlock: str) -> Path:
    """A copy of the program whose shared module reports ``landlock``."""
    copy = tmp_path / "program"
    copy.mkdir()
    shutil.copy(PROGRAM, copy / PROGRAM.name)
    (copy / SHARED.name).write_text(
        SHARED.read_text()
        .replace(
            "def landlock_abi(libc) -> tuple:",
            f"def landlock_abi(libc) -> tuple:\n    return {landlock}\n\n\n"
            "def _unused_landlock_abi(libc) -> tuple:",
        )
        .replace(
            "def userns_problem() -> str | None:",
            "def userns_problem() -> str | None:\n    return 'userns: off'\n\n\n"
            "def _unused_userns_problem() -> str | None:",
        )
    )
    return copy / PROGRAM.name


@requires_landlock
def test_without_landlock_it_fails_closed_unless_the_operator_allows(tmp_path):
    program = _copy(tmp_path, "0, 'ENOSYS'")

    for arguments in (["isolation"], ["run", "--", "true"]):
        refused = _run(*arguments, program=program)
        assert refused.returncode == 123, refused.stderr
        assert "cannot isolate the command" in refused.stderr
        assert "ENOSYS" in refused.stderr
        assert "user namespaces: userns: off" in refused.stderr
    allowed = _run(
        "run",
        "--",
        PYTHON,
        "-c",
        "import os; print(sorted(os.environ))",
        program=program,
        env={"RSI_JUDGE_ISOLATION_FALLBACK": "same-user"},
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "same-user fallback" in allowed.stderr
    assert "RSI_JUDGE_ISOLATION_FALLBACK" not in allowed.stdout


def test_a_root_judge_without_landlock_needs_closed_protected_paths(
    tmp_path, monkeypatch
):
    """Only file permissions confine the isolated user then."""
    tests = tmp_path / "tests"
    tests.mkdir(mode=0o755)
    tests.chmod(0o755)
    monkeypatch.setattr(rsi_isolate, "PROTECTED", (str(tests),))
    monkeypatch.setattr(isolation, "uid_mode_problem", lambda *_: None)
    monkeypatch.setattr(rsi_isolate, "_ids_in_use", lambda: None)
    monkeypatch.setattr(isolation, "landlock_abi", lambda _libc: (0, "ENOSYS"))
    monkeypatch.delenv("RSI_JUDGE_ISOLATION_FALLBACK", raising=False)

    with pytest.raises(isolation.Failure) as refused:
        rsi_isolate.isolation_mode(None)
    assert refused.value.code == 123
    assert f"{tests} (mode 0755) is open to other users" in refused.value.message
    tests.chmod(0o700)
    assert rsi_isolate.isolation_mode(None)["mode"] == "uid"
    # With Landlock the rules close it anyway.
    tests.chmod(0o755)
    monkeypatch.setattr(isolation, "landlock_abi", lambda _libc: (4, ""))
    assert rsi_isolate.isolation_mode(None)["mode"] == "uid"


TRUNCATE = r"""
import json, os, sys
found = {}
for name, action in (
    ("truncate", lambda: os.truncate(sys.argv[1], 0)),
    ("open_trunc", lambda: os.close(os.open(sys.argv[1], os.O_RDONLY | os.O_TRUNC))),
    ("read", lambda: open(sys.argv[1]).read()),
):
    try:
        action()
        found[name] = "ok"
    except OSError as error:
        found[name] = error.strerror
print(json.dumps(found))
"""


@requires_landlock
def test_below_landlock_abi_3_seccomp_guards_truncation(tmp_path):
    """Landlock ABI 1 and 2 cannot refuse truncating a readable file."""
    program = _copy(tmp_path, f"{min(LANDLOCK, 2)}, ''")
    reference = tmp_path / "reference" / "expected.json"
    reference.parent.mkdir()
    reference.write_text('{"answer": 42}')

    result = _run(
        "run",
        "--read",
        str(reference.parent),
        "--",
        PYTHON,
        "-I",
        "-c",
        TRUNCATE,
        str(reference),
        program=program,
    )

    assert result.returncode == 0, result.stderr
    denied = "Operation not permitted"
    assert json.loads(result.stdout) == {
        "truncate": denied,
        "open_trunc": denied,
        "read": "ok",
    }
    assert reference.read_text() == '{"answer": 42}'


# -- the program and its host side -----------------------------------------------------


def test_the_launcher_runs_the_mounted_program_in_isolated_mode():
    launcher = judge_agent_source("rsi-isolate").decode()
    assert launcher.startswith("#!/bin/sh\n")
    assert "python3 is required in the Judge image" in launcher
    assert "exit 125" in launcher
    assert 'exec python3 -I -S -B /opt/rsi-isolate/rsi_isolate.py "$@"' in launcher
    assert judge_agent_source("rsi_isolate.py") == PROGRAM.read_bytes()


def test_the_program_stays_python_3_8_and_imports_nothing_from_rsi_harness():
    source = PROGRAM.read_text()
    ast.parse(source, feature_version=(3, 8))
    assert "with (" not in source
    assert "import rsi_" not in source and "from rsi_" not in source


def test_the_isolated_users_are_the_harness_range_and_disjoint():
    assert (rsi_isolate.ISOLATE_UID, rsi_isolate.ISOLATE_SLOTS) == (
        JUDGE_ISOLATE_UID,
        JUDGE_ISOLATE_SLOTS,
    )
    assert not set(JUDGE_ISOLATE_UIDS) & set(JUDGE_AGENT_UIDS)
    # The OpenRSI-X science Judge library's default route uids start here.
    assert JUDGE_ISOLATE_UIDS[-1] < 64000


@requires_landlock
@pytest.mark.parametrize("policy", [None, "0"])
def test_native_strict_policy_is_not_enabled_by_functional_test_helper(policy):
    result = _run("isolation", env={isolation.ALLOW_MISSING_SIGNAL_ENV: policy})
    if native_signal_compatibility():
        assert result.returncode == 125
        assert "landlock_create_ruleset failed" in result.stderr
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["landlock_signal_scope"] is (LANDLOCK >= 6)


# -- the userns mode -------------------------------------------------------------


def test_userns_hides_all_top_level_entries_but_system_paths_and_grants(tmp_path):
    for name in ("usr", "etc", "dev", "tests", "data", "home", "logs"):
        (tmp_path / name).mkdir()
    (tmp_path / ".dockerenv").write_text("")
    (tmp_path / "bin").symlink_to("usr/bin")
    (tmp_path / "shortcut").symlink_to("tests")

    hidden = rsi_isolate._top_level([str(tmp_path / "data")], str(tmp_path))

    assert hidden == [
        str(tmp_path / name) for name in (".dockerenv", "home", "logs", "tests")
    ]


def test_the_userns_plan_grants_reads_writes_and_hides_aliases(monkeypatch):
    program = isolation.seccomp_program(
        no_network=True, truncate_guard=False, namespace_guard=True
    )
    monkeypatch.setattr(isolation, "mount_aliases", lambda paths: ["/srv/verifier"])
    monkeypatch.setattr(rsi_isolate, "_interpreter_paths", lambda: ["/usr", "/venv"])
    monkeypatch.setattr(rsi_isolate, "_top_level", lambda grants: ["/home", "/tmp"])

    plan = rsi_isolate._userns_plan(
        ["/data/in"],
        ["/data/out"],
        "/tmp/rsi-isolate-1",
        "/data/out",
        "/home/x/tool",
        program,
    )

    assert plan == {
        "hide": ["/home", "/tmp", "/srv/verifier"],
        "read": ["/data/in", "/venv", "/home/x/tool"],
        "write": ["/data/out", "/tmp/rsi-isolate-1"],
        "cwd": "/data/out",
        "shm": True,
        "filter": program.hex(),
    }
    # A grant on another mount of /logs/verifier would open it.
    with pytest.raises(isolation.Failure) as refused:
        rsi_isolate._userns_plan(
            ["/srv"], [], "/tmp/rsi-isolate-1", "/srv", None, program
        )
    assert refused.value.code == 125
    assert "another mount of a hidden path" in refused.value.message


def test_a_userns_call_runs_the_command_in_the_box(monkeypatch, tmp_path):
    """The mode as the probe found it; the box itself needs user namespaces."""
    started = []

    class Process:
        pid = 2**22 + 1
        returncode = 0

        def __init__(self, argv, **options):
            started.append((argv, options))

        def wait(self, timeout=None):
            return 0

    class Libc:
        def prctl(self, *args):
            return 0

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(isolation, "load_libc", Libc)
    monkeypatch.setattr(
        rsi_isolate,
        "isolation_mode",
        lambda libc: {"mode": "userns", "landlock_abi": 0, "fallback": "none"},
    )
    monkeypatch.setattr(isolation, "mount_aliases", lambda paths: [])
    monkeypatch.setattr(rsi_isolate.subprocess, "Popen", Process)
    # Nothing ran: leave this process's own children alone.
    monkeypatch.setattr(isolation, "end_descendants", lambda: True)
    options = rsi_isolate._parser().parse_args(["run", "--write", str(out)])

    assert rsi_isolate.run(options, ["true"]) == 0

    ((argv, options),) = started
    assert argv[:5] == [sys.executable, "-I", "-S", "-B", str(SHARED)]
    assert argv[-2:] == ["--", "true"]
    plan = json.loads(argv[argv.index("--plan") + 1])
    assert plan["write"][0] == str(out)
    assert plan["cwd"] == plan["write"][1] and plan["shm"] is True
    assert "/tmp" in plan["hide"] and "/usr" not in plan["hide"]
    # The namespace and trace guards; the box installs it, not the caller.
    nr = {"x86_64": (272, 101, 0xC000003E), "aarch64": (97, 117, 0xC00000B7)}
    unshare, ptrace, arch = nr[os.uname().machine]
    program = bytes.fromhex(plan["filter"])
    for number in (unshare, ptrace):
        assert _bpf(program, number, arch) == 0x00050000 | 1


def test_landlock_never_grants_another_mount_of_what_is_never_granted(
    monkeypatch, tmp_path
):
    """A read rule on /data, where /logs/verifier is also mounted, opens it."""
    monkeypatch.setattr(isolation, "mount_aliases", lambda paths: [str(tmp_path / "v")])

    with pytest.raises(isolation.Failure) as refused:
        rsi_isolate._ruleset(None, 4, [str(tmp_path)], [], "/tmp/private", None)

    assert refused.value.code == 125
    assert f"{tmp_path / 'v'} is another mount of a hidden path" in (
        refused.value.message
    )
