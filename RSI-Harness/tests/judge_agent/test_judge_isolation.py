"""judge_isolation.py, the isolation rsi-agent and rsi-isolate share.

Mode selection with patched ids, capabilities, Landlock and user namespaces;
the seccomp filters run through a small BPF interpreter and in a child
process; mount aliases from synthetic mountinfo; the box's pieces; slots and
private directories. The box itself runs only where unprivileged user
namespaces work (not where AppArmor restricts them). The filters and
rulesets in force are exercised by tests/judge_agent/test_rsi_isolate.py
and the Docker tests.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rsi_harness.judge_agent import judge_isolation as isolation
from rsi_harness.judge_agent import rsi_isolate

KILL = 0x80000000
ALLOW = 0x7FFF0000
EPERM = 0x00050000 | errno.EPERM
ENOSYS = 0x00050000 | errno.ENOSYS
ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}
NR = {
    "x86_64": {
        "socket": 41,
        "socketpair": 53,
        "io_uring_setup": 425,
        "truncate": 76,
        "open": 2,
        "openat": 257,
        "openat2": 437,
        "read": 0,
        "add_key": 248,
        "request_key": 249,
        "keyctl": 250,
        "ptrace": 101,
        "process_madvise": 440,
        "move_pages": 279,
        "clone": 56,
        "clone3": 435,
        "unshare": 272,
        "setns": 308,
        "mount": 165,
        "fsopen": 430,
    },
    "aarch64": {
        "socket": 198,
        "socketpair": 199,
        "io_uring_setup": 425,
        "truncate": 45,
        "openat": 56,
        "openat2": 437,
        "read": 63,
        "add_key": 217,
        "request_key": 218,
        "keyctl": 219,
        "ptrace": 117,
        "process_madvise": 440,
        "move_pages": 239,
        "clone": 220,
        "clone3": 435,
        "unshare": 97,
        "setns": 268,
        "mount": 40,
        "fsopen": 430,
    },
}
O_TRUNC, O_WRONLY = 0o1000, 0o1
AF_UNIX, AF_INET, AF_INET6, AF_NETLINK, AF_PACKET, AF_VSOCK = 1, 2, 10, 16, 17, 40
SOCK_STREAM, SOCK_DGRAM, SOCK_RAW, SOCK_SEQPACKET = 1, 2, 3, 5
SOCK_NONBLOCK, SOCK_CLOEXEC = 0o4000, 0o2000000
NETLINK_ROUTE, NETLINK_USERSOCK = 0, 2
CLONE_NEWUSER, CLONE_NEWNET, SIGCHLD = 0x10000000, 0x40000000, 17
MACHINES = pytest.mark.parametrize("machine", ["x86_64", "aarch64"])


def _filter(program: bytes, nr: int, arch: int, *args: int) -> int:
    """Run a seccomp BPF program on one ``struct seccomp_data``."""
    data = struct.pack("<iIQ6Q", nr, arch, 0, *(list(args) + [0] * (6 - len(args))))
    code = [struct.unpack_from("<HBBI", program, i) for i in range(0, len(program), 8)]
    accumulator, pc = 0, 0
    while True:
        op, jt, jf, k = code[pc]
        if op == 0x20:  # BPF_LD | BPF_W | BPF_ABS
            accumulator = struct.unpack_from("<I", data, k)[0]
        elif op == 0x54:  # BPF_ALU | BPF_AND | BPF_K
            accumulator &= k
        elif op == 0x05:  # BPF_JMP | BPF_JA
            pc += k
        elif op == 0x15:  # BPF_JMP | BPF_JEQ | BPF_K
            pc += jt if accumulator == k else jf
        elif op == 0x35:  # BPF_JMP | BPF_JGE | BPF_K
            pc += jt if accumulator >= k else jf
        elif op == 0x45:  # BPF_JMP | BPF_JSET | BPF_K
            pc += jt if accumulator & k else jf
        elif op == 0x06:  # BPF_RET | BPF_K
            return k
        else:
            raise AssertionError(f"unexpected BPF op {op:#x}")
        pc += 1


# -- mode selection ------------------------------------------------------------


def test_a_usable_uid_range_is_uid_mode_whatever_landlock_says():
    mode = isolation.choose_mode(None, 0, "ENOSYS", "none", 123, "the command")
    assert mode == {"mode": "uid", "landlock_abi": 0, "fallback": "none"}


def test_without_the_uid_range_landlock_is_used():
    mode = isolation.choose_mode("euid 1000", 4, "", "none", 123, "the command")
    assert mode == {"mode": "landlock", "landlock_abi": 4, "fallback": "none"}


def test_without_both_only_the_operator_fallback_runs_same_user():
    mode = isolation.choose_mode("euid 1000", 0, "ENOSYS", "same-user", 3, "x")
    assert mode["mode"] == "same_user"
    with pytest.raises(isolation.Failure) as failure:
        isolation.choose_mode("euid 1000", 0, "ENOSYS", "none", 123, "the command")
    assert failure.value.code == 123
    assert "cannot isolate the command" in failure.value.message
    assert failure.value.detail == "euid 1000; Landlock: ENOSYS"


def test_uid_mode_needs_root_its_capabilities_and_mapped_ids(monkeypatch):
    every = sum(1 << bit for bit in isolation.UID_MODE_CAPABILITIES)
    maps = {}
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 1000)
    assert isolation.uid_mode_problem(62000, 64) == "euid 1000 is not root"
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolation, "capabilities", lambda: every)
    monkeypatch.setattr(
        isolation,
        "id_map_covers",
        lambda path, first, count: maps.get(path, (first, count)) == (first, count),
    )
    assert isolation.uid_mode_problem(62000, 64) is None
    maps["/proc/self/gid_map"] = None
    assert "gid_map does not map ids 62000-62063" in isolation.uid_mode_problem(
        62000, 64
    )
    monkeypatch.setattr(isolation, "capabilities", lambda: every & ~(1 << 6))
    assert "capability" in isolation.uid_mode_problem(62000, 64)


def test_id_maps_must_cover_the_whole_range(tmp_path):
    path = tmp_path / "uid_map"
    path.write_text("         0       1000          1\n     1   100000  65536\n")
    assert isolation.id_map_covers(str(path), 62000, 64)
    assert not isolation.id_map_covers(str(path), 65500, 64)
    assert not isolation.id_map_covers(str(tmp_path / "missing"), 1, 1)


def _forced(monkeypatch, *, root: bool, abi: int):
    monkeypatch.setattr(
        rsi_isolate.isolation,
        "uid_mode_problem",
        lambda *_args: None if root else "euid 1000 is not root",
    )
    monkeypatch.setattr(
        rsi_isolate.isolation, "landlock_abi", lambda _libc: (abi, "" if abi else "x")
    )
    monkeypatch.setattr(rsi_isolate.isolation, "userns_problem", lambda: "off")


def test_rsi_isolate_takes_landlock_when_the_image_uses_its_ids(monkeypatch):
    def missing(number):
        raise KeyError(number)

    def entry(number):
        if number == 62005:
            return ("someone",)
        raise KeyError(number)

    _forced(monkeypatch, root=True, abi=4)
    monkeypatch.setattr(rsi_isolate.pwd, "getpwuid", missing)
    monkeypatch.setattr(rsi_isolate.grp, "getgrgid", missing)
    assert rsi_isolate.isolation_mode(None)["mode"] == "uid"
    monkeypatch.setattr(rsi_isolate.grp, "getgrgid", entry)
    assert rsi_isolate._ids_in_use() == "the image uses gid 62005"
    monkeypatch.setattr(rsi_isolate.pwd, "getpwuid", entry)
    assert rsi_isolate._ids_in_use() == "the image uses uid 62005"
    assert rsi_isolate.isolation_mode(None)["mode"] == "landlock"


def test_rsi_isolate_reads_only_the_operator_fallback(monkeypatch):
    _forced(monkeypatch, root=False, abi=0)
    for value in ("none", "yes", "", "SAME-USER"):
        monkeypatch.setenv("RSI_JUDGE_ISOLATION_FALLBACK", value)
        with pytest.raises(isolation.Failure) as failure:
            rsi_isolate.isolation_mode(None)
        assert failure.value.code == 123
    monkeypatch.setenv("RSI_JUDGE_ISOLATION_FALLBACK", "same-user")
    assert rsi_isolate.isolation_mode(None) == {
        "mode": "same_user",
        "landlock_abi": 0,
        "fallback": "same-user",
    }


# -- seccomp -------------------------------------------------------------------


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_no_network_refuses_internet_sockets_and_io_uring(machine):
    program = isolation.seccomp_program(
        no_network=True, truncate_guard=False, machine=machine
    )
    arch, nr = ARCH[machine], NR[machine]

    assert _filter(program, nr["socket"], arch, 2) == EPERM  # AF_INET
    assert _filter(program, nr["socket"], arch, 10) == EPERM  # AF_INET6
    # Only the low 32 bits: the kernel truncates ``int family``.
    assert _filter(program, nr["socket"], arch, (1 << 32) | 2) == EPERM
    assert _filter(program, nr["socket"], arch, 1) == ALLOW  # AF_UNIX
    assert _filter(program, nr["io_uring_setup"], arch) == EPERM
    assert _filter(program, nr["read"], arch) == ALLOW
    other = ARCH["aarch64" if machine == "x86_64" else "x86_64"]
    assert _filter(program, nr["read"], other) == KILL
    if machine == "x86_64":
        assert _filter(program, 0x40000000 | 41, arch, 2) == EPERM  # x32


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_without_no_network_sockets_stay_but_io_uring_does_not(machine):
    program = isolation.seccomp_program(
        no_network=False, truncate_guard=True, machine=machine
    )
    arch, nr = ARCH[machine], NR[machine]

    assert _filter(program, nr["socket"], arch, 2) == ALLOW
    assert _filter(program, nr["io_uring_setup"], arch) == EPERM


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_the_truncate_guard_refuses_truncation_landlock_cannot_see(machine):
    program = isolation.seccomp_program(
        no_network=False, truncate_guard=True, machine=machine
    )
    arch, nr = ARCH[machine], NR[machine]

    assert _filter(program, nr["truncate"], arch) == EPERM
    assert _filter(program, nr["openat2"], arch) == ENOSYS
    assert _filter(program, nr["openat"], arch, 0, 0, O_TRUNC) == EPERM
    assert _filter(program, nr["openat"], arch, 0, 0, O_TRUNC | O_WRONLY) == ALLOW
    assert _filter(program, nr["openat"], arch, 0, 0, 0) == ALLOW
    if machine == "x86_64":
        assert _filter(program, nr["open"], arch, 0, O_TRUNC) == EPERM
        assert _filter(program, nr["open"], arch, 0, O_TRUNC | O_WRONLY) == ALLOW
    plain = isolation.seccomp_program(
        no_network=True, truncate_guard=False, machine=machine
    )
    assert _filter(plain, nr["truncate"], arch) == ALLOW


@pytest.mark.parametrize(
    "overrides",
    [
        {"machine": "riscv64"},
        {"machine": "x86_64", "pointer_bytes": 4},
        {"machine": "x86_64", "byteorder": "big"},
    ],
)
def test_seccomp_refuses_other_architectures(overrides):
    with pytest.raises(isolation.Failure) as failure:
        isolation.seccomp_program(no_network=True, truncate_guard=False, **overrides)
    assert failure.value.code == 125


# -- Landlock paths, slots and directories -------------------------------------


def test_readable_paths_cover_all_but_the_hidden_paths(tmp_path):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (tmp_path / "shown").mkdir()

    paths = isolation.readable_paths([str(hidden)])

    assert "/usr" in paths
    assert str(tmp_path / "shown") in paths
    assert str(hidden) not in paths
    assert not any(str(tmp_path).startswith(path + "/") for path in paths)
    assert isolation.readable_paths(["/"]) == []


def test_a_busy_slot_is_skipped_and_all_busy_times_out(tmp_path):
    held = os.open(tmp_path / "slot-0.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        slot, descriptor = isolation.claim_slot(
            str(tmp_path), 2, time.monotonic() + 5, "busy", "it"
        )
        assert slot == 1
        with pytest.raises(isolation.Failure) as failure:
            isolation.claim_slot(str(tmp_path), 2, time.monotonic() + 0.3, "busy", "it")
        assert (failure.value.code, failure.value.message) == (124, "busy")
        os.close(descriptor)
    finally:
        os.close(held)


def test_a_private_directory_is_a_real_unshared_directory(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    isolation.real_directory(str(private), os.geteuid())
    for case in ("group-writable", "symlink", "other-owner"):
        path = tmp_path / case
        if case == "symlink":
            path.symlink_to(private)
        else:
            path.mkdir(mode=0o700)
        if case == "group-writable":
            path.chmod(0o770)
        owner = os.geteuid() + (1 if case == "other-owner" else 0)
        with pytest.raises(isolation.Failure):
            isolation.real_directory(str(path), owner)


# -- the rules every filter carries --------------------------------------------


@MACHINES
@pytest.mark.parametrize("no_network", [False, True])
def test_every_filter_allows_netlink_route_only_and_no_keyrings(machine, no_network):
    """NETLINK_USERSOCK reaches a paused process, and a keyring outlives its
    writer: neither may carry data between Judge and Work."""
    program = isolation.seccomp_program(
        no_network=no_network, truncate_guard=False, machine=machine
    )
    arch, nr = ARCH[machine], NR[machine]

    def socket(domain, protocol):
        return _filter(program, nr["socket"], arch, domain, 3, protocol)

    assert socket(AF_NETLINK, NETLINK_ROUTE) == ALLOW
    # Only the low 32 bits: the kernel truncates ``int protocol``.
    assert socket(AF_NETLINK, 1 << 32) == ALLOW
    for protocol in (NETLINK_USERSOCK, 4, 15, 0x80000000, (1 << 32) | 2):
        assert socket(AF_NETLINK, protocol) == EPERM, protocol
    for name in ("add_key", "keyctl", "request_key"):
        assert _filter(program, nr[name], arch) == EPERM
    assert socket(AF_UNIX, 0) == ALLOW


@MACHINES
def test_no_network_allows_only_unix_and_netlink_sockets(machine):
    program = isolation.seccomp_program(
        no_network=True, truncate_guard=False, machine=machine
    )
    arch, nr = ARCH[machine], NR[machine]

    for domain in (AF_INET, AF_INET6, AF_PACKET, AF_VSOCK, 0x80000001, 0):
        assert _filter(program, nr["socket"], arch, domain) == EPERM, domain
    assert _filter(program, nr["socket"], arch, (1 << 32) | AF_UNIX) == ALLOW
    assert _filter(program, nr["socketpair"], arch, AF_UNIX) == ALLOW
    assert _filter(program, nr["socketpair"], arch, AF_INET) == EPERM
    open_network = isolation.seccomp_program(
        no_network=False, truncate_guard=False, machine=machine
    )
    for domain in (AF_INET, AF_VSOCK):
        assert _filter(open_network, nr["socket"], arch, domain) == ALLOW
    assert _filter(open_network, nr["socketpair"], arch, AF_INET) == ALLOW


@MACHINES
@pytest.mark.parametrize("no_network", [False, True])
def test_deny_unix_keeps_only_a_stream_or_seqpacket_socketpair(machine, no_network):
    """A datagram end can send to, or bind, any abstract name."""
    program = isolation.seccomp_program(
        no_network=no_network, truncate_guard=False, deny_unix=True, machine=machine
    )
    arch, nr = ARCH[machine], NR[machine]

    def pair(kind, domain=AF_UNIX):
        return _filter(program, nr["socketpair"], arch, domain, kind)

    assert _filter(program, nr["socket"], arch, AF_UNIX) == EPERM
    assert _filter(program, nr["socket"], arch, AF_NETLINK) == ALLOW
    for kind in (SOCK_STREAM, SOCK_SEQPACKET):
        assert pair(kind) == ALLOW
        assert pair(kind | SOCK_NONBLOCK | SOCK_CLOEXEC) == ALLOW
    for kind in (SOCK_DGRAM, SOCK_DGRAM | SOCK_CLOEXEC, SOCK_RAW, 0, 0xF):
        assert pair(kind) == EPERM, kind
    # Only with deny_unix: a plain filter leaves socketpair's type alone.
    assert pair(SOCK_STREAM, AF_INET) == (EPERM if no_network else ALLOW)
    plain = isolation.seccomp_program(
        no_network=no_network, truncate_guard=False, machine=machine
    )
    assert _filter(plain, nr["socketpair"], arch, AF_UNIX, SOCK_DGRAM) == ALLOW


@MACHINES
def test_the_namespace_guard_refuses_new_namespaces_and_mounts(machine):
    program = isolation.seccomp_program(
        no_network=False, truncate_guard=False, namespace_guard=True, machine=machine
    )
    arch, nr = ARCH[machine], NR[machine]

    assert _filter(program, nr["clone"], arch, SIGCHLD) == ALLOW
    assert _filter(program, nr["clone"], arch, CLONE_NEWUSER | SIGCHLD) == EPERM
    assert _filter(program, nr["clone"], arch, CLONE_NEWNET) == EPERM
    assert _filter(program, nr["clone3"], arch) == ENOSYS
    for name in ("unshare", "setns", "mount", "fsopen"):
        assert _filter(program, nr[name], arch) == EPERM, name
    assert _filter(program, nr["read"], arch) == ALLOW
    assert _filter(program, nr["socketpair"], arch, AF_UNIX) == ALLOW
    plain = isolation.seccomp_program(
        no_network=False, truncate_guard=False, machine=machine
    )
    assert _filter(plain, nr["unshare"], arch) == ALLOW
    assert _filter(plain, nr["clone3"], arch) == ALLOW


@MACHINES
def test_the_trace_guard_refuses_ptrace_and_reaching_into_memory(machine):
    arch, nr = ARCH[machine], NR[machine]
    for trace_guard, expected in ((True, EPERM), (False, ALLOW)):
        program = isolation.seccomp_program(
            no_network=False,
            truncate_guard=False,
            trace_guard=trace_guard,
            machine=machine,
        )
        for name in ("ptrace", "process_madvise", "move_pages"):
            assert _filter(program, nr[name], arch) == expected, name


@MACHINES
def test_the_network_filter_keeps_what_apptainer_needs(machine):
    program = isolation.network_filter(machine=machine)
    arch, nr = ARCH[machine], NR[machine]

    for domain, protocol in ((AF_INET, 0), (AF_INET6, 0), (AF_VSOCK, 0)):
        assert _filter(program, nr["socket"], arch, domain, 1, protocol) == EPERM
    assert _filter(program, nr["socket"], arch, AF_NETLINK, 3, 2) == EPERM
    assert _filter(program, nr["socket"], arch, AF_NETLINK, 3, 0) == ALLOW
    assert _filter(program, nr["socket"], arch, AF_UNIX) == ALLOW
    assert _filter(program, nr["socketpair"], arch, AF_INET) == EPERM
    assert _filter(program, nr["ptrace"], arch) == EPERM
    assert _filter(program, nr["io_uring_setup"], arch) == EPERM
    # NUMA libraries in Work move and advise their own pages.
    for name in ("move_pages", "process_madvise"):
        assert _filter(program, nr[name], arch) == ALLOW, name
    # Apptainer's user lookups (sssd, Kerberos) may use keyrings.
    assert _filter(program, nr["add_key"], arch) == ALLOW
    other = ARCH["aarch64" if machine == "x86_64" else "x86_64"]
    assert _filter(program, nr["read"], other) == KILL
    if machine == "x86_64":
        assert _filter(program, 0x40000000 | 41, arch, AF_UNIX) == EPERM


# A child under a filter: each probe's outcome, "ok" or the error.
FILTERED = r"""
import ctypes, json, os, socket, sys
sys.path.insert(0, sys.argv[1])
import judge_isolation as isolation
libc = isolation.load_libc()
libc.prctl(isolation.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
isolation.Seccomp(libc, bytes.fromhex(sys.argv[2])).install()
libc.syscall.restype = ctypes.c_long
found = {}
for name, args in json.loads(sys.argv[3]).items():
    result = libc.syscall(*(ctypes.c_long(value) for value in args))
    found[name] = "ok" if result >= 0 else os.strerror(ctypes.get_errno())
pairs = {"socketpair": socket.SOCK_STREAM, "socketpair_dgram": socket.SOCK_DGRAM}
for name, kind in pairs.items():
    try:
        socket.socketpair(socket.AF_UNIX, kind)
        found[name] = "ok"
    except OSError as error:
        found[name] = error.strerror
print(json.dumps(found))
"""


@pytest.mark.skipif(
    os.uname().machine != "x86_64", reason="x86_64 syscall numbers below"
)
@pytest.mark.parametrize("layer", ["no_network", "judge"])
def test_a_filtered_child_gets_eperm_for_what_the_rules_refuse(layer):
    """On the real kernel: the rsi-isolate --no-network rules, and the
    Judge layer of rsi-box."""
    program = isolation.seccomp_program(
        no_network=layer == "no_network",
        truncate_guard=False,
        deny_unix=layer == "judge",
    )
    calls = {
        "inet": [41, AF_INET, 1, 0],
        "vsock": [41, AF_VSOCK, 1, 0],
        "unix": [41, AF_UNIX, 1, 0],
        "netlink_route": [41, AF_NETLINK, 3, NETLINK_ROUTE],
        "netlink_usersock": [41, AF_NETLINK, 3, NETLINK_USERSOCK],
        "netlink_high_bit": [41, AF_NETLINK, 3, (1 << 32) | NETLINK_USERSOCK],
        "add_key": [248, 0, 0, 0, 0, 0],
    }
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            FILTERED,
            str(Path(isolation.__file__).parent),
            program.hex(),
            json.dumps(calls),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout)
    denied = "Operation not permitted"
    assert found["netlink_route"] == "ok"
    assert found["socketpair"] == "ok"
    for name in ("netlink_usersock", "netlink_high_bit", "add_key"):
        assert found[name] == denied, name
    if layer == "no_network":
        assert (found["inet"], found["vsock"], found["unix"]) == (denied,) * 2 + ("ok",)
        assert found["socketpair_dgram"] == "ok"
    else:
        assert (found["unix"], found["socketpair_dgram"]) == (denied, denied)
        assert found["inet"] == "ok"


# -- the userns mode -----------------------------------------------------------


def test_userns_comes_after_uid_and_landlock_and_before_same_user():
    probes = []

    def probe(problem):
        def userns():
            probes.append(problem)
            return problem

        return userns

    assert isolation.choose_mode(None, 0, "", "none", 3, "x", probe(None))["mode"] == (
        "uid"
    )
    assert isolation.choose_mode("u", 4, "", "none", 3, "x", probe(None))["mode"] == (
        "landlock"
    )
    assert probes == []
    assert isolation.choose_mode(
        "u", 0, "ENOSYS", "same-user", 3, "x", probe(None)
    ) == {
        "mode": "userns",
        "landlock_abi": 0,
        "fallback": "same-user",
        "namespaces": ["user", "mount", "pid", "ipc"],
        "proc": "private",
        "shm": "private",
    }
    assert isolation.choose_mode(
        "u", 0, "ENOSYS", "same-user", 3, "x", probe("unshare: EPERM")
    )["mode"] == ("same_user")
    with pytest.raises(isolation.Failure) as failure:
        isolation.choose_mode("u", 0, "ENOSYS", "none", 3, "x", probe("mount proc"))
    assert failure.value.detail == "u; Landlock: ENOSYS; user namespaces: mount proc"


def test_the_userns_probe_runs_the_box_once_and_reports_its_step(monkeypatch):
    runs = []

    def run(argv, **_options):
        runs.append(argv)
        return subprocess.CompletedProcess(
            argv, 125, b"", b"rsi-box: uid_map: Operation not permitted\n"
        )

    monkeypatch.setattr(isolation, "_USERNS", {})
    monkeypatch.setattr(isolation.subprocess, "run", run)

    assert isolation.userns_problem() == "rsi-box: uid_map: Operation not permitted"
    assert isolation.userns_problem() == "rsi-box: uid_map: Operation not permitted"
    assert len(runs) == 1
    plan = json.loads(runs[0][runs[0].index("--plan") + 1])
    assert plan["write"] == [plan["hide"][0] + "/grant"] and plan["shm"]
    # The probe's command runs behind the namespace guard, like a call's.
    program = bytes.fromhex(plan["filter"])
    machine = os.uname().machine
    if machine in ARCH:
        assert _filter(program, NR[machine]["unshare"], ARCH[machine]) == EPERM


MOUNTINFO = """\
22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw
30 22 8:1 /srv/run/verifier /logs/verifier rw,relatime - ext4 /dev/sda1 rw
31 22 8:1 /srv/run /data rw,relatime - ext4 /dev/sda1 rw
32 22 8:1 /srv/run/verifier /mnt/copy\\040two rw,relatime - ext4 /dev/sda1 rw
33 22 8:1 /srv/run/verifier/reward.json /etc/reward.json rw - ext4 /dev/sda1 rw
34 22 0:40 / /tests rw - tmpfs tmpfs rw
35 22 0:40 /cases /cases rw - tmpfs tmpfs rw
36 22 0:41 / /run/other rw - tmpfs tmpfs rw
"""


VERIFIER_ALIASES = [
    "/data/verifier",
    "/etc/reward.json",
    "/mnt/copy two",
    "/srv/run/verifier",
]


def test_mount_aliases_are_the_other_mounts_of_the_same_files():
    """Bound below, above and beside the hidden path, and the root's own
    view of it."""
    aliases = isolation.mount_aliases(["/logs/verifier"], MOUNTINFO)
    assert aliases == VERIFIER_ALIASES
    assert isolation.mount_aliases(["/tests"], MOUNTINFO) == ["/cases"]
    assert isolation.mount_aliases(["/tests/cases/one"], MOUNTINFO) == ["/cases/one"]
    assert isolation.mount_aliases(["/run/other"], MOUNTINFO) == []
    # Inside a hidden path is not an alias.
    assert "/logs/verifier" not in isolation.mount_aliases(
        ["/data", "/logs"], MOUNTINFO
    )


def test_mount_aliases_include_those_of_a_mount_below_a_hidden_path():
    """Hiding /logs hides /logs/verifier, a mount of another root: its
    other mounts are aliases too."""
    assert isolation.mount_aliases(["/logs"], MOUNTINFO) == VERIFIER_ALIASES
    assert isolation.mount_aliases(["/"], MOUNTINFO) == []


def test_a_grant_on_an_alias_of_a_hidden_path_is_refused(monkeypatch):
    parse = isolation.mounts
    monkeypatch.setattr(isolation, "mounts", lambda text=None: parse(MOUNTINFO))
    assert isolation.hidden_aliases(["/logs/verifier"], ["/tmp/x"], 2) == (
        VERIFIER_ALIASES
    )
    for grant in ("/data", "/data/verifier/sub", "/etc"):
        with pytest.raises(isolation.Failure) as failure:
            isolation.hidden_aliases(["/logs/verifier"], [grant], 2)
        assert failure.value.code == 2
        assert "is another mount of a hidden path" in failure.value.message


def test_a_bind_remount_keeps_the_locked_flags():
    assert isolation.remount_flags(os.ST_NOSUID | os.ST_NODEV | os.ST_RELATIME) == (
        isolation.MS_NOSUID | isolation.MS_NODEV | isolation.MS_RELATIME
    )
    assert isolation.remount_flags(os.ST_NOEXEC | os.ST_NOATIME | os.ST_NODIRATIME) == (
        isolation.MS_NOEXEC | isolation.MS_NOATIME | isolation.MS_NODIRATIME
    )
    # Neither noatime nor relatime: strictatime, said explicitly.
    assert isolation.remount_flags(os.ST_RDONLY) == isolation.MS_STRICTATIME


# -- the box -------------------------------------------------------------------


def test_pid_1_obeys_only_a_kill_from_outside_the_box():
    outside = {"code": 0, "sender": 0}
    assert isolation.control_action(signal.SIGUSR1, **outside) == signal.SIGSTOP
    assert isolation.control_action(signal.SIGUSR2, **outside) == signal.SIGCONT
    assert isolation.control_action(signal.SIGTERM, **outside) == signal.SIGTERM
    # A process in the box: its own pid; sigqueue() with a forged si_pid 0.
    assert isolation.control_action(signal.SIGUSR1, code=0, sender=7) is None
    assert isolation.control_action(signal.SIGUSR1, code=-1, sender=0) is None
    assert isolation.control_action(signal.SIGHUP, **outside) is None


def test_the_box_exits_with_the_commands_status():
    assert isolation.exit_status(3 << 8) == 3
    assert isolation.exit_status(signal.SIGKILL) == 128 + signal.SIGKILL


class _Libc:
    """A C library that records its calls; ``fail`` maps a call, or a call
    and its first argument, to an errno."""

    def __init__(self, fail=None) -> None:
        self.fail = fail or {}
        self.calls: list = []
        for name in ("unshare", "mount", "umount2", "prctl"):
            setattr(self, name, self._call(name))

    def _call(self, name):
        library = self

        class Call:
            argtypes = None

            def __call__(self, *args):
                args = tuple(
                    os.fsdecode(arg) if isinstance(arg, bytes) else arg for arg in args
                )
                library.calls.append((name, *args))
                error = library.fail.get((name, args[0]), library.fail.get(name))
                if error:
                    isolation.ctypes.set_errno(error)
                    return -1
                return 0

        return Call()


@pytest.mark.parametrize("detach", [0, errno.EINVAL, errno.EBUSY])
def test_proc_is_replaced_detached_or_covered(monkeypatch, tmp_path, detach):
    """MNT_LOCKED (EINVAL) leaves Apptainer's /proc covered; any other
    failure stops the box."""
    libc = _Libc({("umount2", "/proc"): detach})
    monkeypatch.setattr(isolation, "_PROC_STAGE", str(tmp_path))
    staged = str(tmp_path / "proc")
    hardened = isolation.MS_NOSUID | isolation.MS_NODEV | isolation.MS_NOEXEC

    if detach == errno.EBUSY:
        with pytest.raises(isolation.Failure) as failure:
            isolation._replace_proc(libc)
        assert failure.value.message == "detach /proc: " + os.strerror(errno.EBUSY)
        return
    isolation._replace_proc(libc)
    assert libc.calls == [
        ("mount", "tmpfs", str(tmp_path), "tmpfs", hardened, "mode=0700"),
        ("mount", "proc", staged, "proc", hardened, None),
        ("umount2", "/proc", isolation.MNT_DETACH),
        ("mount", staged, "/proc", None, isolation.MS_MOVE, None),
        ("umount2", str(tmp_path), 0),
    ]


def test_a_plan_makes_all_read_only_hides_then_binds_the_grants(monkeypatch, tmp_path):
    hidden = tmp_path / "hidden"
    (hidden / "grant").mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text("{}")
    shown = tmp_path / "shown"
    shown.mkdir()
    rdonly = {}

    class Libc(_Libc):
        def __init__(self):
            super().__init__()
            mount = self.mount

            def remount(source, target, fstype, flags, data):
                if flags & isolation.MS_REMOUNT and flags & isolation.MS_RDONLY:
                    rdonly[os.fsdecode(target)] = True
                return mount(source, target, fstype, flags, data)

            self.mount = remount

    def statvfs(path):
        if not os.path.lexists(path):
            raise FileNotFoundError(path)
        flags = os.ST_NOSUID | os.ST_RELATIME
        flags |= os.ST_RDONLY if rdonly.get(os.fspath(path)) else 0
        return os.statvfs_result((0,) * 8 + (flags, 255))

    libc = Libc()
    mounts = [("8:1", "/", "/"), ("0:5", "/", "/proc"), ("0:6", "/", str(tmp_path))]
    monkeypatch.setattr(isolation, "mounts", lambda text=None: mounts)
    monkeypatch.setattr(isolation.os, "statvfs", statvfs)
    plan = {
        "hide": [str(hidden), str(secret), str(hidden / "gone")],
        "read": [str(shown)],
        "write": [str(hidden / "grant")],
        "shm": True,
    }

    isolation._apply_plan(libc, plan)

    kept = isolation.MS_NOSUID | isolation.MS_RELATIME
    remount = isolation.MS_REMOUNT | isolation.MS_BIND | kept
    hardened = isolation.MS_NOSUID | isolation.MS_NODEV | isolation.MS_NOEXEC
    bind = isolation.MS_BIND | isolation.MS_REC
    ro = isolation.MS_RDONLY
    mounted = [call[1:] for call in libc.calls if call[0] == "mount"]
    grant, read = str(hidden / "grant"), str(shown)
    assert [(call[1], call[3]) for call in mounted] == [
        # Everything read-only but /proc, parents first.
        ("/", remount | ro),
        (str(tmp_path), remount | ro),
        # Hidden: a tmpfs, /dev/null for a file; what is gone is skipped.
        (str(hidden), hardened),
        (str(secret), isolation.MS_BIND),
        # The grants bound back, parents first, read-only or writable.
        (read, bind),
        (read, remount | ro),
        (grant, bind),
        (grant, remount),
        # The hiding mounts read-only too, then a fresh /dev/shm.
        (str(hidden), remount | ro),
        (str(secret), remount | ro),
        ("/dev/shm", isolation.MS_NOSUID | isolation.MS_NODEV),
    ]
    assert mounted[2][0] == "tmpfs" and mounted[3][0] == "/dev/null"
    assert mounted[4][0].startswith("/proc/self/fd/")


@pytest.mark.parametrize("error", [0, errno.EPERM])
def test_a_mount_that_stays_writable_stops_the_box(monkeypatch, error):
    writable = os.statvfs_result((0,) * 8 + (os.ST_NOSUID, 255))
    monkeypatch.setattr(isolation.os, "statvfs", lambda _path: writable)

    with pytest.raises(isolation.Failure) as failure:
        isolation._read_only(_Libc({"mount": error}), "/data")

    assert failure.value.message == f"remount /data read-only: {os.strerror(error)}"


def test_the_box_refuses_root_before_any_namespace(monkeypatch):
    libc = _Libc()
    monkeypatch.setattr(isolation, "load_libc", lambda: libc)
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 0)

    with pytest.raises(isolation.Failure, match="non-root"):
        isolation.box({}, b"", ["true"])
    assert libc.calls == []


class _Exited(Exception):
    pass


def _exit(code):
    raise _Exited(code)


@pytest.mark.parametrize("outer_alive", [False, True])
def test_pid_1_needs_its_outer_process_and_mounts_mqueue(
    monkeypatch, tmp_path, outer_alive
):
    """pid 1 stops at once when the outer process is gone; otherwise it
    sets the box up, with a private mqueue where /dev/mqueue exists."""
    libc = _Libc()
    alive, holder = os.pipe()
    if not outer_alive:
        os.close(holder)
    isdir = os.path.isdir
    monkeypatch.setattr(isolation, "_PROC_STAGE", str(tmp_path))
    monkeypatch.setattr(isolation.os, "_exit", _exit)
    monkeypatch.setattr(isolation.os, "fork", lambda: 4242)
    monkeypatch.setattr(
        isolation.os.path, "isdir", lambda path: path == "/dev/mqueue" or isdir(path)
    )
    monkeypatch.setattr(isolation.signal, "signal", lambda *_args: None)
    supervised = []
    monkeypatch.setattr(isolation, "_supervise", supervised.append)
    try:
        with pytest.raises(_Exited, match="125"):
            isolation._init(libc, {}, b"", ["true"], alive, set())
    finally:
        if outer_alive:
            os.close(holder)

    mounted = [call[1:4] for call in libc.calls if call[0] == "mount"]
    if not outer_alive:
        assert (mounted, supervised) == ([], [])
        return
    assert ("mqueue", "/dev/mqueue", "mqueue") in mounted
    assert supervised == [4242]


def test_pid_1_drops_python_handlers_before_setup_and_the_command(monkeypatch):
    """The kernel drops a signal from inside the box to pid 1 only at its
    default action: SIGINT's handler, or one the caller installed, would let
    the command interrupt pid 1. Ignored signals and the control handlers
    stay."""
    events = []
    installed = {signal.SIGINT: signal.default_int_handler, signal.SIGHUP: print}
    installed[signal.SIGPIPE] = signal.SIG_IGN

    def record(signum, handler):
        events.append((signum, "handler" if callable(handler) else handler))

    def fork():
        events.append("fork")
        return 4242

    alive, holder = os.pipe()
    monkeypatch.setattr(isolation.signal, "getsignal", lambda n: installed.get(n))
    monkeypatch.setattr(isolation.signal, "signal", record)
    monkeypatch.setattr(isolation, "_replace_proc", lambda libc: events.append("setup"))
    monkeypatch.setattr(isolation.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(isolation.os, "fork", fork)
    monkeypatch.setattr(isolation.os, "_exit", _exit)
    monkeypatch.setattr(isolation, "_supervise", lambda command: None)
    try:
        with pytest.raises(_Exited, match="125"):
            isolation._init(_Libc(), {}, b"", ["true"], alive, set())
    finally:
        os.close(holder)

    setup = events.index("setup")
    assert events.index("fork") > setup
    reset = {event for event in events[:setup] if event[1] == signal.SIG_DFL}
    assert reset == {(signal.SIGINT, signal.SIG_DFL), (signal.SIGHUP, signal.SIG_DFL)}
    handled = {event[0] for event in events[:setup] if event[1] == "handler"}
    assert handled == set(isolation.BOX_CONTROL) | {signal.SIGCHLD}


def test_the_command_starts_with_sigpipe_and_sigxfsz_at_their_defaults():
    """Python ignores both at start, and exec would keep them ignored."""
    script = (
        "import signal, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import judge_isolation as isolation\n"
        "mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])\n"
        "isolation._exec(['sh', '-c', 'grep ^SigIgn: /proc/self/status'], None, mask)\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(isolation.__file__).parent)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    ignored = int(result.stdout.split()[1], 16)
    assert not ignored & (1 << (signal.SIGPIPE - 1) | 1 << (signal.SIGXFSZ - 1))


def test_a_box_that_cannot_be_set_up_exits_125_with_the_step(monkeypatch, capsys):
    if os.geteuid() == 0:
        pytest.skip("the box refuses root")
    monkeypatch.setattr(isolation, "load_libc", lambda: _Libc({"unshare": errno.EPERM}))

    assert isolation.box_main(["--work", "--", "true"]) == 125
    assert capsys.readouterr().err == "rsi-box: unshare: Operation not permitted\n"


@pytest.mark.parametrize(
    "arguments",
    [["--work"], ["--work", "--"], ["--", "true"], ["--work", "--no-unix", "--", "x"]],
)
def test_box_usage_errors_exit_125(arguments, capsys):
    assert isolation.box_main(arguments) == 125
    assert capsys.readouterr().err.startswith("rsi-box: ")


def test_the_host_side_names_the_box_command(tmp_path):
    from rsi_harness.judge_agent import (
        JUDGE_BOX_TARGET,
        install_judge_box,
        judge_box_command,
    )

    assert judge_box_command("/usr/bin/python3", "work", ["sleep", "1"]) == [
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        str(JUDGE_BOX_TARGET),
        "--work",
        "--",
        "sleep",
        "1",
    ]
    judge = judge_box_command("python3", "judge", ["sh"], no_unix=True)
    assert judge[5:] == ["--judge", "--no-unix", "--", "sh"]
    for layer, no_unix in (("work", True), ("plan", False)):
        with pytest.raises(ValueError):
            judge_box_command("python3", layer, ["sh"], no_unix=no_unix)
    program = install_judge_box(tmp_path)
    assert program.read_bytes() == Path(isolation.__file__).read_bytes()


# -- the box on the real kernel (where unprivileged user namespaces work) ------

USERNS = os.geteuid() != 0 and isolation.userns_problem() is None
requires_userns = pytest.mark.skipif(
    not USERNS, reason="unprivileged user namespaces are unavailable here"
)
BOX = [sys.executable, "-I", "-S", "-B", isolation.__file__]


@requires_userns
def test_work_in_the_box_is_pid_2_sees_only_the_box_and_cannot_unshare():
    script = (
        "import ctypes, json, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "unshare = libc.unshare(0x10000000)\n"
        "print(json.dumps({'pid': os.getpid(), 'unshare': ctypes.get_errno(),\n"
        "    'proc': sorted(int(n) for n in os.listdir('/proc') if n.isdigit()),\n"
        "    'parent': os.path.exists('/proc/%d' % int(os.environ['OUTER']))}))\n"
    )
    result = subprocess.run(
        [*BOX, "--work", "--", sys.executable, "-I", "-c", script],
        env={"OUTER": str(os.getpid())},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout)
    assert found == {
        "pid": 2,
        "unshare": errno.EPERM,
        "proc": [1, 2],
        "parent": False,
    }


@requires_userns
def test_pause_and_resume_through_the_box_reach_a_setsid_daemon(tmp_path):
    ready = tmp_path / "ready"
    script = (
        "import os, sys, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    time.sleep(60)\n"
        "open(sys.argv[1], 'w').close()\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [*BOX, "--work", "--", sys.executable, "-I", "-c", script, str(ready)]
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        tree = _tree(process.pid)
        work = [pid for pid, parent in tree.items() if parent != process.pid]
        assert len(tree) == 3 and len(work) == 2
        os.kill(process.pid, signal.SIGUSR1)
        assert _states(work, "T")
        os.kill(process.pid, signal.SIGUSR2)
        assert _states(work, "S")
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) == 128 + signal.SIGTERM
    finally:
        process.kill()
        process.wait()


@requires_userns
def test_a_plan_hides_paths_keeps_grants_and_makes_the_rest_read_only(tmp_path):
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "flag").write_text("x")
    grant = secret / "grant"
    grant.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    plan = {
        "hide": [str(secret)],
        "write": [str(grant)],
        "cwd": str(grant),
        "shm": True,
        "filter": isolation.seccomp_program(
            no_network=False, truncate_guard=False, namespace_guard=True
        ).hex(),
    }
    script = (
        "import json, os, sys\n"
        "def attempt(action):\n"
        "    try:\n"
        "        action()\n"
        "        return 'ok'\n"
        "    except OSError as error:\n"
        "        return error.strerror\n"
        "secret, outside = sys.argv[1:]\n"
        "print(json.dumps({\n"
        "    'flag': attempt(lambda: open(secret + '/flag').read()),\n"
        "    'grant': attempt(lambda: open('new', 'w').close()),\n"
        "    'outside': attempt(lambda: open(outside + '/new', 'w').close()),\n"
        "}))\n"
    )
    result = subprocess.run(
        isolation.box_argv(
            plan, [sys.executable, "-I", "-c", script, str(secret), str(outside)]
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "flag": "No such file or directory",
        "grant": "ok",
        "outside": "Read-only file system",
    }
    assert (grant / "new").exists()


def _tree(pid: int) -> dict:
    """Every descendant of ``pid``, with its parent."""
    parents = {}
    for name in os.listdir("/proc"):
        if not name.isdecimal():
            continue
        try:
            stat = Path(f"/proc/{name}/stat").read_text().rpartition(") ")[2]
        except OSError:
            continue
        parents[int(name)] = int(stat.split()[1])
    found, stack = {}, [pid]
    while stack:
        parent = stack.pop()
        for child, its_parent in parents.items():
            if its_parent == parent:
                found[child] = parent
                stack.append(child)
    return found


def _states(pids, state: str) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        found = {
            Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[0]
            for pid in pids
        }
        if found == {state}:
            return True
        time.sleep(0.05)
    return False
