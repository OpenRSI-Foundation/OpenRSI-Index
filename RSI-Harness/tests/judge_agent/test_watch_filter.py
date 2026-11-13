"""Work's file-watch rule: inotify, fanotify, dnotify and leases are
unavailable.

The filters run through test_judge_isolation's BPF interpreter on both
machines, then on this kernel through ``rsi-box --work-filter``, as a
cluster Work exec runs.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from rsi_harness.judge_agent import JUDGE_BOX_TARGET, judge_box_command
from rsi_harness.judge_agent import judge_isolation as isolation
from tests.judge_agent.test_judge_isolation import (
    ALLOW,
    ARCH,
    ENOSYS,
    EPERM,
    KILL,
    MACHINES,
    NR,
    _filter,
)

EINVAL = 0x00050000 | errno.EINVAL
WATCH = {
    "x86_64": {
        "inotify_init": 253,
        "inotify_init1": 294,
        "inotify_add_watch": 254,
        "fanotify_init": 300,
        "fanotify_mark": 301,
    },
    "aarch64": {
        "inotify_init1": 26,
        "inotify_add_watch": 27,
        "fanotify_init": 262,
        "fanotify_mark": 263,
    },
}
FCNTL = {"x86_64": 72, "aarch64": 25}
INOTIFY_RM_WATCH = {"x86_64": 255, "aarch64": 28}
F_GETFL, F_SETFL, F_SETSIG, F_NOTIFY, F_DUPFD_CLOEXEC = 3, 4, 10, 1026, 1030
F_GETLK, F_SETLK, F_SETLKW, F_SETLEASE, F_GETLEASE = 5, 6, 7, 1024, 1025
F_GETDELEG, F_SETDELEG = 1039, 1040
F_RDLCK, F_WRLCK, F_UNLCK = 0, 1, 2
# 1042 is unassigned.
REFUSED = (F_SETLEASE, F_NOTIFY, F_SETDELEG, 1042)
ALLOWED = (F_GETFL, F_SETFL, F_SETSIG, F_DUPFD_CLOEXEC, F_GETLK, F_SETLK, F_SETLKW)
ALLOWED += (F_GETLEASE, F_GETDELEG, 1027, 1041, 1056, F_SETLEASE | 1 << 31)
ALLOWED += (1 << 32, 0)
# The 32-bit compat arch: i386 (int 0x80) and AArch32, as their kernel
# tables number them.
COMPAT = {
    "x86_64": {
        "arch": 0x40000003,
        "watch": {
            "inotify_init": 291,
            "inotify_init1": 332,
            "inotify_add_watch": 292,
            "fanotify_init": 338,
            "fanotify_mark": 339,
        },
        "fcntl": (55, 221),
        "other": {
            "read": 3,
            "getpid": 20,
            "mount": 21,
            "unshare": 310,
            "socket": 359,
            "inotify_rm_watch": 293,
        },
    },
    "aarch64": {
        "arch": 0x40000028,
        "watch": {
            "inotify_init": 316,
            "inotify_init1": 360,
            "inotify_add_watch": 317,
            "fanotify_init": 367,
            "fanotify_mark": 368,
        },
        "fcntl": (55, 221),
        "other": {
            "read": 3,
            "getpid": 20,
            "mount": 21,
            "unshare": 337,
            "socket": 281,
            "inotify_rm_watch": 318,
        },
    },
}


def _work_box_program(monkeypatch, *flags):
    """The program rsi-box builds for a layer, without building the box."""
    seen = []
    monkeypatch.setattr(
        isolation, "box", lambda plan, program, argv: seen.append(program) or 0
    )
    assert isolation.box_main([*flags, "--", "true"]) == 0
    return seen[0]


@MACHINES
def test_the_watch_filter_denies_watches_and_nothing_else(machine):
    program = isolation.watch_filter(machine=machine)
    arch, nr, fcntl = ARCH[machine], NR[machine], FCNTL[machine]

    for name, number in WATCH[machine].items():
        assert _filter(program, number, arch) == ENOSYS, name
    # The kernel truncates fcntl's command to 32 bits.
    for refused in REFUSED:
        for high in (0, 1, 0xFFFF, 0xFFFFFFFF):
            command = high << 32 | refused
            assert _filter(program, fcntl, arch, 3, command) == EINVAL, command
    for command in ALLOWED:
        assert _filter(program, fcntl, arch, 3, command) == ALLOW, command
    assert _filter(program, INOTIFY_RM_WATCH[machine], arch) == ALLOW
    # Work keeps what its other layers decide: sockets, io_uring, keyrings,
    # namespaces.
    for name in ("read", "io_uring_setup", "add_key", "unshare", "clone3"):
        assert _filter(program, nr[name], arch) == ALLOW, name
    assert _filter(program, nr["socket"], arch, 16, 3, 2) == ALLOW
    other = ARCH["aarch64" if machine == "x86_64" else "x86_64"]
    assert _filter(program, nr["read"], other) == KILL
    if machine == "x86_64":
        assert _filter(program, 0x40000000 | 294, arch) == EPERM


@MACHINES
def test_the_watch_filter_runs_32_bit_programs_without_watches(machine):
    """As Docker's profile does, the compat arch runs; its own numbers for
    the watch calls are denied, and its fcntl and fcntl64 refuse leases and
    dnotify."""
    program = isolation.watch_filter(machine=machine)
    compat = COMPAT[machine]
    arch = compat["arch"]
    for name, number in compat["watch"].items():
        assert _filter(program, number, arch) == ENOSYS, name
    for number in compat["fcntl"]:
        for command in REFUSED:
            assert _filter(program, number, arch, 3, command) == EINVAL, command
        for command in ALLOWED:
            assert _filter(program, number, arch, 3, command) == ALLOW, command
    for name, number in compat["other"].items():
        assert _filter(program, number, arch) == ALLOW, name
    # The native numbers mean other calls there.
    for number in WATCH[machine].values():
        if number not in compat["watch"].values():
            assert _filter(program, number, arch) == ALLOW, number
    # Any other arch is killed (x86_64 on aarch64 and the reverse).
    for foreign in (0x40000003, 0x40000028, 0xC000003E, 0xC00000B7, 0):
        if foreign not in (arch, ARCH[machine]):
            assert _filter(program, 3, foreign) == KILL, foreign


@MACHINES
def test_rsi_box_s_layers_still_kill_32_bit_programs(machine, monkeypatch):
    """The box's rules name native numbers only; the compat arch stays out."""
    monkeypatch.setattr(os, "uname", lambda: os.uname_result(("", "", "", "", machine)))
    arch = COMPAT[machine]["arch"]
    for flags in (("--work",), ("--judge",), ("--judge", "--no-unix")):
        program = _work_box_program(monkeypatch, *flags)
        for number in (3, 20, *COMPAT[machine]["watch"].values()):
            assert _filter(program, number, arch) == KILL, (flags, number)


@MACHINES
def test_rsi_box_s_work_layer_also_denies_watches(machine, monkeypatch):
    monkeypatch.setattr(os, "uname", lambda: os.uname_result(("", "", "", "", machine)))
    arch, nr, fcntl = ARCH[machine], NR[machine], FCNTL[machine]
    work = _work_box_program(monkeypatch, "--work")
    for name, number in WATCH[machine].items():
        assert _filter(work, number, arch) == ENOSYS, name
    for command in REFUSED:
        assert _filter(work, fcntl, arch, 3, command) == EINVAL, command
    for command in ALLOWED:
        assert _filter(work, fcntl, arch, 3, command) == ALLOW, command
    # Its own rules stay.
    assert _filter(work, nr["unshare"], arch) == EPERM
    assert _filter(work, nr["clone3"], arch) == ENOSYS
    assert _filter(work, nr["add_key"], arch) == EPERM
    # The Judge's layers keep watches.
    for flags in (("--judge",), ("--judge", "--no-unix")):
        judge = _work_box_program(monkeypatch, *flags)
        for name, number in WATCH[machine].items():
            assert _filter(judge, number, arch) == ALLOW, (flags, name)
        for command in REFUSED:
            assert _filter(judge, fcntl, arch, 3, command) == ALLOW, command


def test_only_the_work_filter_layer_runs_without_a_box():
    assert judge_box_command("/py", "work-filter", ["sh", "-c", "x"]) == [
        "/py",
        "-I",
        "-S",
        "-B",
        str(JUDGE_BOX_TARGET),
        "--work-filter",
        "--",
        "sh",
        "-c",
        "x",
    ]
    with pytest.raises(ValueError):
        judge_box_command("/py", "work-filter", ["sh"], no_unix=True)


PROBE = r"""
import ctypes, errno, fcntl, json, os, subprocess, sys, tempfile, time
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
def call(*args):
    result = libc.syscall(*(ctypes.c_long(value) for value in args))
    return "ok" if result >= 0 else errno.errorcode[ctypes.get_errno()]
calls = json.loads(sys.argv[1])
number = calls["fcntl_getfl"][0]
status = dict(line.split(":\t") for line in open("/proc/self/status"))
found = {name: call(*args) for name, args in calls.items()}
directory = tempfile.mkdtemp()
descriptor = os.open(directory, os.O_RDONLY)
for name, command in (("dnotify", 1026), ("dnotify_high", (1 << 32) | 1026)):
    found[name] = call(number, descriptor, command, 1)
found["getfl"] = fcntl.fcntl(descriptor, fcntl.F_GETFL) >= 0
# Leases on a file of Work's own: a write lease, a read lease, the command
# with high bits, and on a bad descriptor (EBADF from the kernel).
leased = os.path.join(directory, "leased")
open(leased, "w").close()
reader = os.open(leased, os.O_RDONLY)
for name, command, kind in (
    ("lease_write", 1024, 1),
    ("lease_read", 1024, 0),
    ("lease_high", (1 << 32) | 1024, 1),
):
    found[name] = call(number, reader, command, kind)
    call(number, reader, 1024, 2)
found["lease_bad_fd"] = call(number, -1, 1024, 1)
# F_SETDELEG (Linux 6.19+) takes a struct; a bad descriptor fails first.
found["deleg_bad_fd"] = call(number, -1, 1040, 0)
found["deleg_high"] = call(number, -1, (1 << 32) | 1040, 0)
found["getlease"] = fcntl.fcntl(reader, 1025)
writer = os.open(leased, os.O_RDWR)
fcntl.lockf(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
found["setlk"] = "ok"
path = os.path.join(directory, "log")
with open(path, "w") as stream:
    stream.write("a\n")
tail = subprocess.Popen(["tail", "-f", path], stdout=subprocess.PIPE, text=True)
time.sleep(0.5)
with open(path, "a") as stream:
    stream.write("b\n")
lines = [tail.stdout.readline(), tail.stdout.readline()]
tail.kill()
found["tail"] = lines
found["seccomp"] = status["Seccomp"].strip()
found["no_new_privs"] = status["NoNewPrivs"].strip()
found["pid"] = os.getpid()
print(json.dumps(found))
"""


def _run_probe(script: str, *args: str, filtered: bool = True):
    box = [sys.executable, "-I", "-S", "-B", isolation.__file__, "--work-filter"]
    process = subprocess.Popen(
        (box + ["--"] if filtered else [])
        + [sys.executable, "-I", "-c", script, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(timeout=60)
    return process, stdout, stderr


@pytest.mark.skipif(
    os.uname().machine not in WATCH or not Path("/usr/bin/tail").exists(),
    reason="no watch filter or tail here",
)
def test_a_command_under_the_work_filter_polls_instead_of_watching():
    """On this kernel: the command keeps its pid; watches and leases are
    unavailable, other fcntl commands work, and GNU tail -f follows the file
    by polling."""
    machine = os.uname().machine
    calls = {name: [number, 0, 0] for name, number in WATCH[machine].items()}
    calls["fcntl_getfl"] = [FCNTL[machine], 0, F_GETFL]
    process, stdout, stderr = _run_probe(PROBE, json.dumps(calls))

    assert process.returncode == 0, stderr
    found = json.loads(stdout)
    assert found.pop("pid") == process.pid
    for name in WATCH[machine]:
        assert found[name] == "ENOSYS", name
    assert (found["dnotify"], found["dnotify_high"]) == ("EINVAL", "EINVAL")
    for name in ("lease_write", "lease_read", "lease_high", "lease_bad_fd"):
        assert found[name] == "EINVAL", name
    assert (found["deleg_bad_fd"], found["deleg_high"]) == ("EINVAL", "EINVAL")
    assert found["getlease"] == F_UNLCK and found["setlk"] == "ok"
    assert found["fcntl_getfl"] == "ok" and found["getfl"] is True
    assert found["tail"] == ["a\n", "b\n"]
    assert (found["seccomp"], found["no_new_privs"]) == ("2", "1")

    # Without the filter this file system grants the write lease.
    with tempfile.TemporaryDirectory() as directory:
        leased = Path(directory) / "leased"
        leased.touch()
        reader = os.open(leased, os.O_RDONLY)
        try:
            assert fcntl.fcntl(reader, F_SETLEASE, F_WRLCK) == 0
            fcntl.fcntl(reader, F_SETLEASE, F_UNLCK)
        finally:
            os.close(reader)


@pytest.mark.skipif(os.uname().machine not in WATCH, reason="no watch filter here")
@pytest.mark.parametrize("locale", [{}, {"LANG": "C"}, {"LANG": "POSIX"}])
def test_the_work_filter_execs_with_default_signals_and_its_own_environment(locale):
    """Python ignores SIGPIPE and SIGXFSZ at start and coerces a C locale
    (LC_CTYPE=C.UTF-8); the command sees neither."""
    box = [sys.executable, "-I", "-S", "-B", isolation.__file__, "--work-filter"]
    script = (
        "grep ^SigIgn /proc/self/status; env | sort; "
        "set -o pipefail; yes | head -1 >/dev/null; echo pipe=$?"
    )
    environment = {"PATH": "/usr/bin:/bin", "MARK": "a=b", **locale}
    found = subprocess.run(
        [*box, "--", "/bin/bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.splitlines()

    assert found[0].split() == ["SigIgn:", "0" * 16]
    shown = {line.partition("=")[0]: line.partition("=")[2] for line in found[1:-1]}
    for name in ("PWD", "SHLVL", "_", "OLDPWD"):
        shown.pop(name, None)
    assert shown == environment
    assert found[-1] == "pipe=141"


# i386 syscalls from this x86_64 process: int 0x80 enters the kernel as a
# 32-bit program does (seccomp sees AUDIT_ARCH_I386).
IA32_PROBE = r"""
import ctypes, errno, json, mmap, os, sys, tempfile
# push rbx; mov eax, edi; mov ebx, esi; xchg ecx, edx; int 0x80; pop rbx; ret
code = bytes.fromhex("53" "89f8" "89f3" "87d1" "cd80" "5b" "c3")
prot = mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
page = mmap.mmap(-1, mmap.PAGESIZE, prot=prot)
page.write(code)
address = ctypes.addressof(ctypes.c_char.from_buffer(page))
int80 = ctypes.CFUNCTYPE(*[ctypes.c_int] * 5)(address)
def call(number, *args):
    result = int80(number, *(list(args) + [0, 0, 0])[:3])
    return "ok" if result >= 0 else errno.errorcode[-result]
directory = tempfile.mkdtemp()
leased = os.path.join(directory, "leased")
open(leased, "w").close()
reader = os.open(leased, os.O_RDONLY)
found = {"getpid": int80(20, 0, 0, 0) == os.getpid()}
for name, number in json.loads(sys.argv[1]).items():
    found[name] = call(number)
for name, number in (("fcntl", 55), ("fcntl64", 221)):
    found[name + "_lease"] = call(number, reader, 1024, 1)
    call(number, reader, 1024, 2)
    found[name + "_dnotify"] = call(number, os.open(directory, os.O_RDONLY), 1026, 1)
    found[name + "_getlease"] = int80(number, reader, 1025, 0)
    found[name + "_deleg_bad_fd"] = call(number, -1, 1040, 0)
    found[name + "_getfl"] = call(number, reader, 3)
print(json.dumps(found))
"""


@pytest.mark.skipif(os.uname().machine != "x86_64", reason="not x86_64")
def test_a_32_bit_program_runs_under_the_work_filter_without_watches():
    """On this kernel: i386 calls run (no SIGSYS), their watch calls get
    ENOSYS, and their fcntl and fcntl64 refuse leases and dnotify."""
    watch = json.dumps(COMPAT["x86_64"]["watch"])
    process, stdout, stderr = _run_probe(IA32_PROBE, watch, filtered=False)
    if process.returncode != 0 or not json.loads(stdout)["getpid"]:
        pytest.skip(f"no i386 syscalls here: {stderr.strip()[-200:]}")
    unfiltered = json.loads(stdout)
    assert unfiltered["inotify_init"] == "ok"
    assert unfiltered["fcntl64_lease"] == "ok"
    assert unfiltered["fcntl64_deleg_bad_fd"] == "EBADF"

    process, stdout, stderr = _run_probe(IA32_PROBE, watch)

    assert process.returncode == 0, stderr
    found = json.loads(stdout)
    assert found.pop("getpid") is True
    for name in COMPAT["x86_64"]["watch"]:
        assert found.pop(name) == "ENOSYS", name
    for name in ("fcntl", "fcntl64"):
        assert found[name + "_lease"] == "EINVAL"
        assert found[name + "_dnotify"] == "EINVAL"
        assert found[name + "_getlease"] == F_UNLCK
        assert found[name + "_deleg_bad_fd"] == "EINVAL"
        assert found[name + "_getfl"] == "ok"


@pytest.mark.parametrize(("name", "status"), [("missing", 127), ("plain", 126)])
def test_the_work_filter_reports_a_command_it_cannot_run(tmp_path, name, status):
    if os.uname().machine not in WATCH:
        pytest.skip("no watch filter here")
    (tmp_path / "plain").write_text("not a program\n")
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", isolation.__file__, "--work-filter", "--"]
        + [str(tmp_path / name)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status
    assert "rsi-box: cannot execute" in result.stderr
