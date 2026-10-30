"""rsi_nonet.py, the host-side wrapper of a phase without network.

It runs here as it runs on a node: a forked child sets no_new_privs,
installs the network filter and execs the command in place.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

from rsi_harness.cluster.bluevela import rsi_nonet
from rsi_harness.judge_agent.judge_isolation import network_filter

WRAPPER = rsi_nonet.__file__
requires_seccomp_arch = pytest.mark.skipif(
    os.uname().machine not in ("x86_64", "aarch64"), reason="no network filter"
)
PROBE = r"""
import ctypes, json, os, socket
def attempt(action):
    try:
        action().close()
        return "ok"
    except OSError as error:
        return error.strerror
status = dict(line.split(":\t") for line in open("/proc/self/status"))
libc = ctypes.CDLL(None, use_errno=True)
add_key = {"x86_64": 248, "aarch64": 217}[os.uname().machine]
# The process keyring: gone with this process.
key = libc.syscall(add_key, b"user", b"rsi-nonet-probe", b"x", 1, -2)
print(json.dumps({
    "pid": os.getpid(),
    "no_new_privs": status["NoNewPrivs"].strip(),
    "seccomp": status["Seccomp"].strip(),
    "inet": attempt(lambda: socket.socket(socket.AF_INET)),
    "inet6": attempt(lambda: socket.socket(socket.AF_INET6)),
    "vsock": attempt(lambda: socket.socket(40, socket.SOCK_STREAM)),
    "unix": attempt(lambda: socket.socket(socket.AF_UNIX)),
    "socketpair": attempt(lambda: socket.socketpair()[0]),
    "netlink_route": attempt(lambda: socket.socket(16, socket.SOCK_RAW, 0)),
    "netlink_usersock": attempt(lambda: socket.socket(16, socket.SOCK_RAW, 2)),
    "add_key": "ok" if key >= 0 else os.strerror(ctypes.get_errno()),
}))
"""


@requires_seccomp_arch
def test_the_command_keeps_its_pid_and_gets_no_ip_sockets():
    process = subprocess.Popen(
        rsi_nonet.command([sys.executable, "-I", "-c", PROBE]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(timeout=60)

    assert process.returncode == 0, stderr
    denied = "Operation not permitted"
    found = json.loads(stdout)
    # The same process: pause and kill reach it unchanged.
    assert found.pop("pid") == process.pid
    assert found == {
        "no_new_privs": "1",
        "seccomp": "2",
        "inet": denied,
        "inet6": denied,
        "vsock": denied,
        "unix": "ok",
        "socketpair": "ok",
        "netlink_route": "ok",
        "netlink_usersock": denied,
        # Apptainer's user lookups may need them; the Judge layer denies them.
        "add_key": "ok",
    }


def test_the_wrapped_argv_carries_this_machines_filter():
    argv = rsi_nonet.command(["apptainer", "exec", "x.sif", "true"], python="py")

    assert argv[:5] == ["py", "-I", "-S", "-B", os.path.realpath(WRAPPER)]
    assert argv[6:] == ["--", "apptainer", "exec", "x.sif", "true"]
    if os.uname().machine in ("x86_64", "aarch64"):
        assert bytes.fromhex(argv[5]) == network_filter()


@pytest.mark.parametrize(
    ("arguments", "status", "reason"),
    [
        (["--", "true"], 125, "usage"),
        (["zz", "--", "true"], 125, "not a seccomp program"),
        (["00" * 7, "--", "true"], 125, "not a seccomp program"),
        (["FILTER", "--", "/nonexistent/apptainer"], 127, "cannot execute"),
    ],
)
@requires_seccomp_arch
def test_a_command_that_does_not_run_exits_125_to_127(arguments, status, reason):
    arguments = [network_filter().hex() if a == "FILTER" else a for a in arguments]
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", WRAPPER, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == status
    assert reason in result.stderr


def test_the_wrapper_runs_under_python_3_6_and_imports_rsi_harness_only_lazily():
    source = open(WRAPPER).read()
    tree = ast.parse(source, feature_version=(3, 6))
    assert "from __future__" not in source
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert sorted(alias.name for node in top for alias in node.names) == [
        "ctypes",
        "os",
        "sys",
    ]
