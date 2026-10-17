"""judge_isolation.py, the isolation rsi-agent and rsi-isolate share.

Mode selection with patched ids, capabilities and Landlock; the seccomp
filter run through a small BPF interpreter; slots and private directories.
The filters and rulesets in force are exercised by tests/judge_agent/
test_rsi_isolate.py and the Docker tests.
"""

from __future__ import annotations

import errno
import fcntl
import os
import struct
import time

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
        "io_uring_setup": 425,
        "truncate": 76,
        "open": 2,
        "openat": 257,
        "openat2": 437,
        "read": 0,
    },
    "aarch64": {
        "socket": 198,
        "io_uring_setup": 425,
        "truncate": 45,
        "openat": 56,
        "openat2": 437,
        "read": 63,
    },
}
O_TRUNC, O_WRONLY = 0o1000, 0o1


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
        elif op == 0x06:  # BPF_RET | BPF_K
            return k
        else:
            raise AssertionError(f"unexpected BPF op {op:#x}")
        pc += 1


# -- mode selection -------------------------------------------------------------


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


# -- seccomp ----------------------------------------------------------------------


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


# -- Landlock paths, slots and directories ----------------------------------------


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
