"""Isolation shared by rsi-agent and rsi-isolate, inside a Judge.

How a child is isolated, chosen automatically: when the Judge is root, the
child runs as a dedicated unprivileged uid; otherwise it keeps the Judge's
uid and is confined by Landlock, or else by its own user, mount, PID and IPC
namespaces (``userns``); without any of them it runs only under the
operator's same-user fallback. This module holds the pieces both programs
use: mode detection, the Landlock ruleset, the seccomp filters, uid slots,
the child's setup before exec, ending whatever a child left running, and
the namespace box.

Run as a program it is rsi-box, the box RSI Harness puts around Work and
the Judge where Apptainer gives them no PID or IPC namespace:

    rsi-box (--work | --judge [--no-unix]) -- COMMAND [ARG]...

COMMAND runs as pid 2 of new user, PID, IPC and mount namespaces, with a
fresh /proc and /dev/mqueue. ``--work`` denies new namespaces and mounts;
``--judge`` denies kernel keyrings, and with ``--no-unix`` AF_UNIX sockets
(a stream or seqpacket socketpair stays). SIGUSR1, SIGUSR2 and SIGTERM to
rsi-box stop, continue and terminate every process in the box. Exit status:
COMMAND's (128+N for signal N); 125 when the box cannot be set up, with the
failing step on stderr; 126 and 127 when COMMAND cannot be executed or is
not found.

Standard library only, Python 3.8 or later, no RSI Harness imports.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import os
import re
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time

EXIT_TIMEOUT = 124
EXIT_INTERNAL = 125
EXIT_CANNOT_EXECUTE = 126
EXIT_NOT_FOUND = 127
EXIT_TERMINATED = 143
KILL_GRACE_SECONDS = 5.0
SLOT_POLL_SECONDS = 0.2
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PR_SET_PDEATHSIG = 1
PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
PR_SET_CHILD_SUBREAPER = 36
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2
# Landlock: the same syscall numbers on x86_64 and aarch64.
LANDLOCK_CREATE_RULESET = 444
LANDLOCK_ADD_RULE = 445
LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_MAX_ABI = 6
ALLOW_MISSING_SIGNAL_ENV = "RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL"
LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET = 1
LANDLOCK_SCOPE_SIGNAL = 2
# Filesystem rights in bit order, each with the ABI that introduced it.
LANDLOCK_FS_RIGHTS = (1,) * 13 + (2, 3, 5)
FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1, 2, 4, 8
FS_TRUNCATE, FS_IOCTL_DEV = 1 << 14, 1 << 15
FS_FILE = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV
FS_READ = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
FS_DEVICE = FS_READ_FILE | FS_WRITE_FILE | FS_IOCTL_DEV | FS_TRUNCATE
# CHOWN, DAC_OVERRIDE, FOWNER, KILL, SETGID and SETUID: the uid mode's needs.
UID_MODE_CAPABILITIES = (0, 1, 3, 5, 6, 7)
O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

# seccomp, as the OpenRSI-X science Judge library builds it.
SECCOMP_ARCHES = {
    "x86_64": {
        "arch": 0xC000003E,
        "socket": 41,
        "socketpair": 53,
        "io_uring_setup": 425,
        "x32_mask": 0x40000000,
        "truncate": 76,
        "open": 2,
        "openat": 257,
        "openat2": 437,
        # add_key, request_key, keyctl
        "keyring": (248, 249, 250),
        # ptrace, process_vm_readv, process_vm_writev, pidfd_getfd
        "trace": (101, 310, 311, 438),
        # process_madvise, move_pages
        "trace_memory": (440, 279),
        # unshare, setns, mount, umount2, pivot_root, chroot, then the new
        # mount API: open_tree, move_mount, fsopen, fsconfig, fsmount,
        # fspick, mount_setattr
        "namespace": (272, 308, 165, 166, 155, 161, 428, 429, 430, 431, 432, 433, 442),
        "clone": 56,
        "clone3": 435,
    },
    "aarch64": {
        "arch": 0xC00000B7,
        "socket": 198,
        "socketpair": 199,
        "io_uring_setup": 425,
        "x32_mask": None,
        "truncate": 45,
        "open": None,
        "openat": 56,
        "openat2": 437,
        "keyring": (217, 218, 219),
        "trace": (117, 270, 271, 438),
        "trace_memory": (440, 239),
        "namespace": (97, 268, 40, 39, 41, 51, 428, 429, 430, 431, 432, 433, 442),
        "clone": 220,
        "clone3": 435,
    },
}
_BPF_LD_W_ABS = 0x20
_BPF_JEQ_K = 0x15
_BPF_JGE_K = 0x35
_BPF_RET_K = 0x06
_BPF_JA = 0x05
_BPF_JSET_K = 0x45
_BPF_AND_K = 0x54
_O_TRUNC_FLAG = 0o1000
_O_ACCMODE = 0o3
_RET_KILL_PROCESS = 0x80000000
_RET_ERRNO = 0x00050000
_RET_ALLOW = 0x7FFF0000
_AF_UNIX = 1
_AF_NETLINK = 16
_NETLINK_ROUTE = 0
_SOCK_TYPE_MASK = 0xF
_SOCK_STREAM = 1
_SOCK_SEQPACKET = 5
CLONE_NEWNS = 0x00020000
CLONE_NEWIPC = 0x08000000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
# Every namespace clone flag: NS, CGROUP, UTS, IPC, USER, PID and NET.
NAMESPACE_FLAGS = 0x7E020000


class Failure(Exception):
    """End the call with ``code`` and one reason line on stderr."""

    def __init__(self, code: int, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


class Stop:
    """The signal that asked this process to stop, if any."""

    signum: int | None = None


def on_signal(signum, frame) -> None:
    del frame
    Stop.signum = signum


def load_libc():
    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise Failure(EXIT_INTERNAL, f"cannot load the C library: {error}") from None


def syscall(libc, number: int, *args: int) -> int:
    libc.syscall.restype = ctypes.c_long
    return libc.syscall(ctypes.c_long(number), *(ctypes.c_long(a) for a in args))


# -- choosing the mode ------------------------------------------------------------


def landlock_abi(libc) -> tuple:
    """The kernel's Landlock ABI (0 when unavailable) and why it is 0."""
    if os.uname().machine not in SECCOMP_ARCHES:
        return 0, "not x86_64 or aarch64"
    value = syscall(libc, LANDLOCK_CREATE_RULESET, 0, 0, 1)
    if value < 0:
        code = ctypes.get_errno()
        return 0, errno.errorcode.get(code, str(code))
    return int(value), ""


def capabilities() -> int:
    try:
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("CapEff:"):
                    return int(line.split()[1], 16)
    except (OSError, ValueError, IndexError):
        pass
    return 0


def id_map_covers(path: str, first: int, count: int) -> bool:
    try:
        with open(path) as stream:
            for line in stream:
                inside, _outside, length = (int(value) for value in line.split())
                if inside <= first and first + count <= inside + length:
                    return True
    except (OSError, ValueError):
        return False
    return False


def uid_mode_problem(first: int, count: int, gid: int | None = None) -> str | None:
    """Why uids ``first`` .. ``first + count - 1`` cannot be used, or None.

    ``gid`` is the first gid when it differs from ``first``.
    """
    if os.geteuid() != 0:
        return f"euid {os.geteuid()} is not root"
    found = capabilities()
    if any(not found >> bit & 1 for bit in UID_MODE_CAPABILITIES):
        return f"root lacks a needed capability (CapEff {found:#x})"
    for start, path in (
        (first, "/proc/self/uid_map"),
        (first if gid is None else gid, "/proc/self/gid_map"),
    ):
        if not id_map_covers(path, start, count):
            return f"{path} does not map ids {start}-{start + count - 1}"
    return None


def choose_mode(
    problem: str | None,
    abi: int,
    reason: str,
    fallback: str,
    unavailable_code: int,
    what: str,
    userns=None,
) -> dict:
    """``uid`` (no ``problem``), ``landlock`` (``abi`` >= 1), ``userns`` (the
    probe ``userns()`` returns no problem), or ``same_user``.

    ``userns`` runs only when neither of the first two applies.
    ``same_user`` only with the operator's ``fallback`` "same-user";
    otherwise ``unavailable_code`` with the reasons.
    """
    report = {"landlock_abi": abi, "fallback": fallback}
    if problem is None:
        return dict(report, mode="uid")
    if abi >= 1:
        return dict(report, mode="landlock")
    userns_problem = None if userns is None else userns()
    if userns is not None and userns_problem is None:
        return dict(
            report,
            mode="userns",
            namespaces=list(BOX_NAMESPACES),
            proc="private",
            shm="private",
        )
    if fallback == "same-user":
        return dict(report, mode="same_user")
    detail = f"{problem}; Landlock: {reason}"
    if userns is not None:
        detail += f"; user namespaces: {userns_problem}"
    raise Failure(
        unavailable_code,
        f"cannot isolate {what}: the Judge is not root and Landlock is unavailable",
        detail,
    )


# -- Landlock -------------------------------------------------------------------


class Ruleset:
    """A Landlock ruleset built by the parent; the child enforces it.

    Every filesystem right of the ABI is handled, so a path without a rule
    grants nothing. From ABI 6, signals and abstract unix sockets stay inside
    the child's domain, except for an explicitly opted-in missing signal
    scope on an incomplete ABI 6 backport. The network is not handled.
    """

    def __init__(
        self, libc, abi: int, *, allow_missing_landlock_signal: bool = False
    ) -> None:
        if type(allow_missing_landlock_signal) is not bool:
            raise Failure(
                EXIT_INTERNAL, "allow_missing_landlock_signal must be a boolean"
            )
        self.libc = libc
        self.abi = min(abi, LANDLOCK_MAX_ABI)
        self.handled = sum(
            1 << bit
            for bit, since in enumerate(LANDLOCK_FS_RIGHTS)
            if since <= self.abi
        )
        self.scoped = 0b11 if self.abi >= 6 else 0
        size = 8 if self.abi < 4 else 16 if self.abi < 6 else 24

        def create(scoped):
            self._attr = ctypes.create_string_buffer(
                struct.pack("<QQQ", self.handled, 0, scoped)[:size], size
            )
            return syscall(
                libc, LANDLOCK_CREATE_RULESET, ctypes.addressof(self._attr), size, 0
            )

        fd = create(self.scoped)
        error = ctypes.get_errno() if fd < 0 else 0
        # RHEL 9 has shipped ABI 6 without LANDLOCK_SCOPE_SIGNAL. Never
        # silently drop a protection because a kernel rejects our ruleset.
        # The operator may opt into this one narrowly scoped compatibility
        # policy. Keep all filesystem rights and abstract-socket scoping.
        if fd < 0 and error == errno.EINVAL and abi == 6:
            if allow_missing_landlock_signal:
                fd = create(LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET)
                if fd >= 0:
                    self.scoped = LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET
                    sys.stderr.write(
                        "Landlock compatibility: signal scoping is unavailable; "
                        "operator explicitly allowed it. Filesystem and abstract "
                        "UNIX socket restrictions remain enabled.\n"
                    )
                else:
                    error = ctypes.get_errno()
        if fd < 0:
            detail = ""
            if error == errno.EINVAL and abi == 6:
                detail = (
                    "Kernel reports Landlock ABI 6 but rejected the ruleset. "
                    "Some RHEL 9 kernels lack signal scoping. Upgrade the kernel "
                    "or use another isolation backend; only if the task permits "
                    "missing signal isolation may the operator enable "
                    "apptainer.judge_allow_missing_landlock_signal. "
                    "No protection was disabled."
                )
            raise Failure(
                EXIT_INTERNAL,
                "landlock_create_ruleset failed: " + os.strerror(error),
                detail,
            )
        self.fd = int(fd)

    def allow(self, path: str, access: int) -> bool:
        try:
            descriptor = os.open(path, os.O_PATH | O_CLOEXEC)
        except OSError:
            return False
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                access &= FS_FILE
            access &= self.handled
            if not access:
                return False
            rule = ctypes.create_string_buffer(
                struct.pack("<Qi", access, descriptor), 12
            )
            if (
                syscall(
                    self.libc, LANDLOCK_ADD_RULE, self.fd, 1, ctypes.addressof(rule), 0
                )
                < 0
            ):
                raise Failure(
                    EXIT_INTERNAL,
                    f"landlock_add_rule failed for {path}: "
                    + os.strerror(ctypes.get_errno()),
                )
            return True
        finally:
            os.close(descriptor)

    def restrict(self) -> None:
        if syscall(self.libc, LANDLOCK_RESTRICT_SELF, self.fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def readable_paths(hidden) -> list:
    """Read rules covering everything but ``hidden`` (a walk from ``/``).

    Hidden paths and their ancestors get no rule (the walk descends into an
    ancestor so its other children do); symlinks and entries of a directory
    that cannot be listed get none either, so they stay unreadable.
    """
    hidden = {os.path.realpath(path) for path in hidden if path}
    if "/" in hidden:
        return []
    ancestors = set()
    for path in hidden:
        while path != "/":
            path = os.path.dirname(path)
            ancestors.add(path)
    readable = []

    def visit(directory: str) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            return
        for entry in entries:
            if entry.path in hidden:
                continue
            try:
                if entry.is_symlink():
                    continue
                descend = entry.path in ancestors and entry.is_dir(
                    follow_symlinks=False
                )
            except OSError:
                continue
            if descend:
                visit(entry.path)
            else:
                readable.append(entry.path)

    visit("/")
    return readable


def allow_devices(ruleset: Ruleset) -> None:
    """Read, write and ioctl on every character device (GPUs, /dev/null)."""
    for current, _dirs, files in os.walk("/dev", followlinks=False):
        for name in files:
            path = os.path.join(current, name)
            try:
                if stat.S_ISCHR(os.lstat(path).st_mode):
                    ruleset.allow(path, FS_DEVICE)
            except OSError:
                continue


# -- seccomp --------------------------------------------------------------------


def seccomp_program(
    *,
    no_network: bool,
    truncate_guard: bool,
    trace_guard: bool = False,
    namespace_guard: bool = False,
    deny_unix: bool = False,
    machine: str | None = None,
    pointer_bytes: int | None = None,
    byteorder: str | None = None,
) -> bytes:
    """A seccomp filter as packed ``struct sock_filter[]``.

    A foreign ``seccomp_data.arch`` is killed; x32 syscall numbers (x86_64),
    ``io_uring_setup``, the kernel keyrings (``add_key``, ``keyctl``,
    ``request_key``) and ``socket(AF_NETLINK)`` with any protocol but
    NETLINK_ROUTE get EPERM. Domains and protocols are compared on their low
    32 bits, as the kernel truncates ``int``. With ``no_network``, ``socket``
    is allowed only for AF_UNIX and AF_NETLINK, ``socketpair`` only for
    AF_UNIX. With ``deny_unix``, ``socket(AF_UNIX)`` gets EPERM, and so
    does a ``socketpair(AF_UNIX)`` that is not SOCK_STREAM or
    SOCK_SEQPACKET. With ``trace_guard``, ``ptrace``,
    ``process_vm_readv``/``writev``, ``process_madvise``, ``pidfd_getfd``
    and ``move_pages`` get EPERM. With
    ``namespace_guard``, ``unshare``, ``setns``, the mount calls,
    ``pivot_root``, ``chroot`` and ``clone`` with a namespace flag get EPERM
    and ``clone3`` ENOSYS (libc then falls back to ``clone``). With
    ``truncate_guard`` (Landlock ABI 1 and 2, which cannot control
    truncation) ``truncate`` and an ``open``/``openat`` with O_TRUNC and a
    read-only access mode get EPERM, and ``openat2`` ENOSYS. Everything else
    is allowed. Any machine other than x86_64 or aarch64, a 32-bit Python or
    a big-endian host raises :class:`Failure` (125).
    """
    return _filter(
        _arch(machine, pointer_bytes, byteorder),
        no_network=no_network,
        keyring=True,
        trace_guard=trace_guard,
        trace_memory=trace_guard,
        namespace_guard=namespace_guard,
        truncate_guard=truncate_guard,
        deny_unix=deny_unix,
    )


def network_filter(
    *,
    machine: str | None = None,
    pointer_bytes: int | None = None,
    byteorder: str | None = None,
) -> bytes:
    """The host-side filter of a phase without network: no IP sockets.

    It wraps ``apptainer`` itself, so it keeps what Apptainer's start and its
    user lookups need (AF_UNIX, NETLINK_ROUTE, the keyrings). As
    :func:`seccomp_program` with ``no_network`` and ``trace_guard``, without
    the keyring rule, and with ``move_pages`` and ``process_madvise``
    allowed: NUMA libraries in Work call them on their own pages.
    """
    return _filter(
        _arch(machine, pointer_bytes, byteorder),
        no_network=True,
        keyring=False,
        trace_guard=True,
        trace_memory=False,
        namespace_guard=False,
        truncate_guard=False,
        deny_unix=False,
    )


def _arch(machine, pointer_bytes, byteorder) -> dict:
    machine = machine if machine is not None else os.uname().machine
    if pointer_bytes is None:
        pointer_bytes = ctypes.sizeof(ctypes.c_void_p)
    byteorder = byteorder if byteorder is not None else sys.byteorder
    spec = SECCOMP_ARCHES.get(machine)
    if spec is None:
        raise Failure(
            EXIT_INTERNAL,
            f"seccomp filters support only x86_64 and aarch64, not {machine}",
        )
    if pointer_bytes != 8 or byteorder != "little":
        raise Failure(
            EXIT_INTERNAL, "seccomp filters need a 64-bit little-endian Python"
        )
    return spec


def _filter(
    spec: dict,
    *,
    no_network: bool,
    keyring: bool,
    trace_guard: bool,
    trace_memory: bool,
    namespace_guard: bool,
    truncate_guard: bool,
    deny_unix: bool,
) -> bytes:
    program: list = [
        (_BPF_LD_W_ABS, 0, 0, 4),
        (_BPF_JEQ_K, 1, 0, spec["arch"]),
        (_BPF_RET_K, 0, 0, _RET_KILL_PROCESS),
        (_BPF_LD_W_ABS, 0, 0, 0),
    ]
    if spec["x32_mask"] is not None:
        program.append((_BPF_JGE_K, "deny", 0, spec["x32_mask"]))
    denied = [spec["io_uring_setup"]]
    denied += list(spec["keyring"]) if keyring else []
    denied += list(spec["trace"]) if trace_guard else []
    denied += list(spec["trace_memory"]) if trace_memory else []
    denied += list(spec["namespace"]) if namespace_guard else []
    program += [(_BPF_JEQ_K, "deny", 0, number) for number in denied]
    if namespace_guard:
        program += [
            (_BPF_JEQ_K, "enosys", 0, spec["clone3"]),
            (_BPF_JEQ_K, "clone", 0, spec["clone"]),
        ]
    if truncate_guard:
        program += [
            (_BPF_JEQ_K, "deny", 0, spec["truncate"]),
            (_BPF_JEQ_K, "enosys", 0, spec["openat2"]),
            (_BPF_JEQ_K, "openat_flags", 0, spec["openat"]),
        ]
        if spec["open"] is not None:
            program.append((_BPF_JEQ_K, "open_flags", 0, spec["open"]))
    program.append((_BPF_JEQ_K, "socket", 0, spec["socket"]))
    if no_network or deny_unix:
        program.append((_BPF_JEQ_K, "socketpair", 0, spec["socketpair"]))
    unix = "deny" if deny_unix else "allow"
    # The low 32 bits of the domain, then of the protocol (args 0 and 2).
    program += [
        (_BPF_JA, 0, 0, "allow"),
        ("socket", None, None, None),
        (_BPF_LD_W_ABS, 0, 0, 16),
        (_BPF_JEQ_K, unix, 0, _AF_UNIX),
        (_BPF_JEQ_K, "netlink", "deny" if no_network else "allow", _AF_NETLINK),
        ("netlink", None, None, None),
        (_BPF_LD_W_ABS, 0, 0, 16 + 16),
        (_BPF_JEQ_K, "allow", "deny", _NETLINK_ROUTE),
    ]
    if no_network or deny_unix:
        program += [("socketpair", None, None, None), (_BPF_LD_W_ABS, 0, 0, 16)]
    if deny_unix:
        # A connected stream or seqpacket pair cannot be pointed elsewhere; a
        # datagram end can send to, or bind, any abstract name.
        program += [
            (_BPF_JEQ_K, 0, "deny" if no_network else "allow", _AF_UNIX),
            (_BPF_LD_W_ABS, 0, 0, 16 + 8),
            (_BPF_AND_K, 0, 0, _SOCK_TYPE_MASK),
            (_BPF_JEQ_K, "allow", 0, _SOCK_STREAM),
            (_BPF_JEQ_K, "allow", "deny", _SOCK_SEQPACKET),
        ]
    elif no_network:
        program.append((_BPF_JEQ_K, "allow", "deny", _AF_UNIX))
    if namespace_guard:
        program += [
            ("clone", None, None, None),
            (_BPF_LD_W_ABS, 0, 0, 16),
            (_BPF_JSET_K, "deny", "allow", NAMESPACE_FLAGS),
        ]
    if truncate_guard:
        # The low 32 bits of the flags argument: deny O_TRUNC with O_RDONLY.
        program += [
            ("open_flags", None, None, None),
            (_BPF_LD_W_ABS, 0, 0, 16 + 8),
            (_BPF_JA, 0, 0, "flags"),
            ("openat_flags", None, None, None),
            (_BPF_LD_W_ABS, 0, 0, 16 + 16),
            ("flags", None, None, None),
            (_BPF_AND_K, 0, 0, _O_TRUNC_FLAG | _O_ACCMODE),
            (_BPF_JEQ_K, "deny", "allow", _O_TRUNC_FLAG),
        ]
    program += [
        ("enosys", None, None, None),
        (_BPF_RET_K, 0, 0, _RET_ERRNO | errno.ENOSYS),
        ("allow", None, None, None),
        (_BPF_RET_K, 0, 0, _RET_ALLOW),
        ("deny", None, None, None),
        (_BPF_RET_K, 0, 0, _RET_ERRNO | errno.EPERM),
    ]
    return _assemble(program)


def _assemble(program) -> bytes:
    """Pack instructions; a string first field is a label, a string jump its target."""
    labels = {}
    index = 0
    for code, _jt, _jf, _k in program:
        if isinstance(code, str):
            labels[code] = index
        else:
            index += 1
    packed = bytearray()
    index = 0
    for code, jt, jf, k in program:
        if isinstance(code, str):
            continue
        offsets = [labels[j] - index - 1 if isinstance(j, str) else j for j in (jt, jf)]
        if isinstance(k, str):
            k = labels[k] - index - 1
        if not all(0 <= offset <= 255 for offset in offsets):
            raise ValueError("BPF jump out of range")
        packed += (
            code.to_bytes(2, "little") + bytes(offsets) + int(k).to_bytes(4, "little")
        )
        index += 1
    return bytes(packed)


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


class Seccomp:
    """A filter prepared by the parent (no allocation after the fork)."""

    def __init__(self, libc, program: bytes) -> None:
        self.libc = libc
        self._buffer = ctypes.create_string_buffer(program, len(program))
        self._fprog = _SockFprog(len(program) // 8, ctypes.addressof(self._buffer))
        self._address = ctypes.c_ulong(ctypes.addressof(self._fprog))

    def install(self) -> None:
        zero = ctypes.c_ulong(0)
        if (
            self.libc.prctl(
                ctypes.c_int(PR_SET_SECCOMP),
                ctypes.c_ulong(SECCOMP_MODE_FILTER),
                self._address,
                zero,
                zero,
            )
            != 0
        ):
            raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP failed")


# -- processes ------------------------------------------------------------------


def signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def group_alive(pgid: int, proc: str = "/proc") -> bool:
    try:
        names = os.listdir(proc)
    except OSError:
        return False
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc, name, "stat")) as stream:
                fields = stream.read().rpartition(") ")[2].split()
        except OSError:
            continue
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] not in "ZX":
            return True
    return False


def end_group(process: subprocess.Popen, grace: float) -> None:
    """SIGTERM the child's process group, then SIGKILL what remains.

    A signal to this process during the grace ends it at once.
    """
    signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and Stop.signum is None:
        if process.poll() is not None and not group_alive(process.pid):
            break
        time.sleep(0.05)
    signal_group(process.pid, signal.SIGKILL)
    process.wait()


def sweep_uid(uid: int, proc: str = "/proc") -> bool:
    """SIGKILL every process left running as ``uid``; True once none remain."""
    for _ in range(80):
        found = False
        try:
            names = os.listdir(proc)
        except OSError:
            return False
        for name in names:
            if not name.isdigit():
                continue
            try:
                with open(os.path.join(proc, name, "status")) as stream:
                    lines = stream.read().splitlines()
                state = next(line for line in lines if line.startswith("State:"))
                if "Z" in state.split()[1:2]:
                    continue
                uids = next(line for line in lines if line.startswith("Uid:"))
                if str(uid) in uids.split()[1:3]:
                    found = True
                    os.kill(int(name), signal.SIGKILL)
            except (OSError, StopIteration, ValueError, IndexError):
                continue
        if not found:
            return True
        time.sleep(0.025)
    return False


def kill_uid(uid: int) -> None:
    """SIGKILL every process of ``uid`` in this PID namespace at once (root
    only): a helper running as ``uid`` calls kill(-1), which signals them in
    one pass that no fork can outrun."""
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
            os.kill(-1, signal.SIGKILL)
        except BaseException:
            code = 1
        os._exit(code)
    os.waitpid(pid, 0)


def descendants() -> dict:
    """Every descendant of this process, with its /proc state."""
    children: dict = {}
    states: dict = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return {}
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join("/proc", name, "stat")) as stream:
                fields = stream.read().rpartition(")")[2].split()
            states[int(name)] = fields[0]
            children.setdefault(int(fields[1]), []).append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    found: dict = {}
    stack = [os.getpid()]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in found:
                found[child] = states.get(child, "?")
                stack.append(child)
    return found


def end_descendants() -> bool:
    """SIGKILL and reap everything this process started; True once none remain.

    The caller is a child subreaper, so a process that escaped its process
    group (a double fork, a new session) is still a descendant.
    """
    for _ in range(200):
        found = descendants()
        if not found:
            return True
        for pid, state in found.items():
            if state not in ("Z", "X"):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        while True:
            try:
                pid, _status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        time.sleep(0.025)
    return False


def real_directory(path: str, owner_uid: int) -> None:
    """Refuse ``path`` unless it is a real directory of ``owner_uid`` that
    nobody else can write."""
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise Failure(EXIT_INTERNAL, f"{path}: {error.strerror}") from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise Failure(EXIT_INTERNAL, f"{path} is not a private directory of the Judge")


def claim_slot(
    directory: str, slots: int, deadline: float, busy: str, what: str
) -> tuple:
    """Lock one free uid slot for the whole call; wait while all are taken.

    Returns (slot, lock descriptor). The lock goes with this process, so a
    killed call frees its slot. ``busy`` is the timeout's message.
    """
    descriptors = []
    try:
        for slot in range(slots):
            descriptors.append(
                os.open(
                    os.path.join(directory, f"slot-{slot}.lock"),
                    os.O_RDWR | os.O_CREAT | O_CLOEXEC | O_NOFOLLOW,
                    0o600,
                )
            )
        while True:
            for slot, descriptor in enumerate(descriptors):
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                descriptors[slot] = -1
                return slot, descriptor
            if Stop.signum is not None:
                raise Failure(EXIT_TERMINATED, f"terminated before {what} started")
            if time.monotonic() >= deadline:
                raise Failure(EXIT_TIMEOUT, busy)
            time.sleep(SLOT_POLL_SECONDS)
    finally:
        for descriptor in descriptors:
            if descriptor >= 0:
                os.close(descriptor)


def preexec(
    cwd_fd: int, owner, libc, ruleset=None, seccomp=None, umask=None, nproc=None
):
    """The child's setup before exec: its directory, user, no_new_privs,
    Landlock, seccomp and death with the parent. ``nproc`` caps the processes
    and threads of ``owner``'s uid."""

    def prepare() -> None:
        os.fchdir(cwd_fd)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if nproc is not None:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        if owner is not None:
            os.setgroups([])
            os.setgid(owner[1])
            os.setuid(owner[0])
            # Never exec the child with any privilege left.
            if (
                os.getresuid() != (owner[0],) * 3
                or os.getresgid() != (owner[1],) * 3
                or os.getgroups()
            ):
                raise OSError(errno.EPERM, "the child's user was not dropped")
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot set no_new_privs")
        if ruleset is not None:
            # The ruleset descriptor is still open: CPython closes inherited
            # descriptors only after this function returns.
            ruleset.restrict()
        if seccomp is not None:
            seccomp.install()
        # After the uid change, which would clear it: die with the parent.
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        if umask is not None:
            os.umask(umask)

    return prepare


def probe_ruleset(
    libc, mode: dict, *, allow_missing_landlock_signal: bool = False
) -> dict:
    """Check creation, not just the ABI query; do not restrict this process."""
    if mode["landlock_abi"] < 1:
        return mode
    ruleset = Ruleset(
        libc,
        mode["landlock_abi"],
        allow_missing_landlock_signal=allow_missing_landlock_signal,
    )
    try:
        return dict(
            mode,
            landlock_scoped=ruleset.scoped,
            landlock_signal_scope=bool(ruleset.scoped & LANDLOCK_SCOPE_SIGNAL),
        )
    finally:
        ruleset.close()


# -- the box: user, mount, PID and IPC namespaces ---------------------------------

BOX_NAMESPACES = ("user", "mount", "pid", "ipc")
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 1, 2, 4, 8
MS_REMOUNT, MS_NOATIME, MS_NODIRATIME, MS_BIND = 32, 1024, 2048, 4096
MS_MOVE, MS_REC, MS_PRIVATE = 8192, 16384, 1 << 18
MS_RELATIME, MS_STRICTATIME = 1 << 21, 1 << 24
MNT_DETACH = 2
# Where pid 1 mounts the fresh /proc before moving it into place.
_PROC_STAGE = "/tmp"
# A control signal to the box, and what pid 1 sends every process in it.
BOX_CONTROL = {
    signal.SIGUSR1: signal.SIGSTOP,
    signal.SIGUSR2: signal.SIGCONT,
    signal.SIGTERM: signal.SIGTERM,
}
_SI_USER = 0
_USERNS_PROBE_SECONDS = 60.0
_USERNS: dict = {}


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _rebase(path: str, old: str, new: str) -> str:
    """``path`` below ``old``, moved below ``new``."""
    rest = path[len(old.rstrip("/")) :]
    return (new.rstrip("/") + rest) or "/"


def mounts(text: str | None = None) -> list:
    """(device, root, mount point) of each mount, in mountinfo order."""
    if text is None:
        with open("/proc/self/mountinfo", errors="surrogateescape") as stream:
            text = stream.read()
    found = []
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) > 4:
            root, point = (
                re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)
                for field in fields[3:5]
            )
            found.append((fields[2], root, point))
    return found


def mount_aliases(paths, text: str | None = None) -> list:
    """Every other place that shows the files of ``paths`` (real paths).

    That is the same device mounted elsewhere with a root that overlaps the
    root-relative path of one of them or of a mount below one. Only the top
    mount at each point counts; places inside ``paths`` are left out.
    """
    paths = list(paths)
    top = {point: (device, root) for device, root, point in mounts(text)}
    found = set()
    below = [point for point in top if any(_within(point, path) for path in paths)]
    for path in dict.fromkeys(paths + below):
        holders = [point for point in top if _within(path, point)]
        if not holders:
            continue
        holder = max(holders, key=len)
        device, root = top[holder]
        inner = _rebase(path, holder, root)
        for point, (other_device, other_root) in top.items():
            if point == holder or other_device != device:
                continue
            if _within(inner, other_root):
                alias = _rebase(inner, other_root, point)
            elif _within(other_root, inner):
                alias = point
            else:
                continue
            if not any(_within(alias, item) for item in paths):
                found.add(alias)
    return sorted(found)


def hidden_aliases(hidden, grants, code: int) -> list:
    """The aliases of ``hidden``; an alias that overlaps a grant is refused
    with ``code``, as granting it would open the hidden path."""
    aliases = mount_aliases([os.path.realpath(path) for path in hidden])
    for alias in aliases:
        for grant in grants:
            if _within(alias, grant) or _within(grant, alias):
                raise Failure(
                    code,
                    f"refusing {grant}: {alias} is another mount of a hidden path",
                )
    return aliases


def remount_flags(statvfs_flags: int) -> int:
    """The mount flags a bind remount must keep: nosuid, nodev, noexec and
    the atime mode."""
    # The controller imports this module on macOS too; only Linux workers
    # remount filesystems. Do not resolve Linux-only os constants at import.
    if sys.platform != "linux":
        raise RuntimeError("bind remount flags require Linux")
    kept_flags = (
        (os.ST_NOSUID, MS_NOSUID),
        (os.ST_NODEV, MS_NODEV),
        (os.ST_NOEXEC, MS_NOEXEC),
        (os.ST_NOATIME, MS_NOATIME),
        (os.ST_NODIRATIME, MS_NODIRATIME),
        (os.ST_RELATIME, MS_RELATIME),
    )
    flags = sum(ms for st, ms in kept_flags if statvfs_flags & st)
    if not flags & (MS_NOATIME | MS_RELATIME):
        flags |= MS_STRICTATIME
    return flags


def box_argv(plan: dict, argv) -> list:
    """The command line that runs ``argv`` in a box built from ``plan``.

    ``hide``: paths covered by an empty read-only tmpfs (/dev/null for a
    file); ``read`` and ``write``: paths bound back read-only or writable,
    everything else is read-only; ``cwd``; ``shm``: a fresh /dev/shm;
    ``filter``: the seccomp program (hex). Exit status as rsi-box's.
    """
    return [
        sys.executable or "python3",
        "-I",
        "-S",
        "-B",
        os.path.realpath(__file__),
        "--plan",
        json.dumps(plan, sort_keys=True),
        "--",
        *argv,
    ]


def userns_problem() -> str | None:
    """Why the ``userns`` mode cannot run here, or None: one real box run,
    cached for the process."""
    if "problem" not in _USERNS:
        _USERNS["problem"] = _probe_box()
    return _USERNS["problem"]


def _probe_box() -> str | None:
    """Hide a scratch directory, write a grant below it, behind every guard."""
    scratch = None
    try:
        scratch = os.path.realpath(tempfile.mkdtemp(prefix=".rsi-box-"))
        grant = os.path.join(scratch, "grant")
        os.mkdir(grant)
        program = seccomp_program(
            no_network=False,
            truncate_guard=False,
            trace_guard=True,
            namespace_guard=True,
        )
        plan = {
            "hide": [scratch],
            "write": [grant],
            "cwd": grant,
            "shm": True,
            "filter": program.hex(),
        }
        command = [sys.executable or "python3", "-I", "-S", "-c", "open('ok', 'w')"]
        result = subprocess.run(
            box_argv(plan, command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=_USERNS_PROBE_SECONDS,
            check=False,
        )
        if result.returncode == 0 and os.path.isfile(os.path.join(grant, "ok")):
            return None
        lines = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return lines[-1] if lines else f"the box exited with {result.returncode}"
    except (Failure, OSError, subprocess.SubprocessError) as error:
        return f"the box did not run: {error}"
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def control_action(signum: int, code: int, sender: int) -> int | None:
    """What pid 1 sends every process in the box for a signal, or None.

    Only a kill() from outside the PID namespace counts: si_code SI_USER and
    si_pid 0. A process in the box sends with its own pid, and sigqueue()
    cannot claim SI_USER.
    """
    if code != _SI_USER or sender != 0:
        return None
    return BOX_CONTROL.get(signum)


def exit_status(status: int) -> int:
    """A wait status as a shell exit status (128+N for signal N)."""
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


def _check(result: int, step: str) -> None:
    if result != 0:
        raise Failure(EXIT_INTERNAL, f"{step}: {os.strerror(ctypes.get_errno())}")


def _mount(libc, source, target, fstype, flags: int, data, step: str) -> None:
    source, target, fstype, data = (
        None if value is None else os.fsencode(value)
        for value in (source, target, fstype, data)
    )
    _check(libc.mount(source, target, fstype, flags, data), step)


def _write(path: str, text: str, step: str) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | O_CLOEXEC)
        try:
            os.write(descriptor, text.encode())
        finally:
            os.close(descriptor)
    except OSError as error:
        raise Failure(EXIT_INTERNAL, f"{step}: {error.strerror}") from None


def _read_only(libc, point: str) -> None:
    """Remount the mount at ``point`` read-only; an unreachable point is
    skipped, a point that stays writable fails."""
    try:
        flags = os.statvfs(point).f_flag
    except OSError:
        return
    if flags & os.ST_RDONLY:
        return
    flags = MS_REMOUNT | MS_BIND | MS_RDONLY | remount_flags(flags)
    error = (
        ctypes.get_errno()
        if libc.mount(None, os.fsencode(point), None, flags, None)
        else 0
    )
    # A point covered by a later mount resolves to an already read-only one.
    if not os.statvfs(point).f_flag & os.ST_RDONLY:
        raise Failure(EXIT_INTERNAL, f"remount {point} read-only: {os.strerror(error)}")


def _replace_proc(libc) -> None:
    """A fresh /proc of the box's PID namespace: Apptainer's is detached
    where the kernel allows it, else covered (a locked mount, which a nested
    namespace cannot uncover either)."""
    staged = os.path.join(_PROC_STAGE, "proc")
    flags = MS_NOSUID | MS_NODEV | MS_NOEXEC
    _mount(libc, "tmpfs", _PROC_STAGE, "tmpfs", flags, "mode=0700", "stage /proc")
    try:
        os.mkdir(staged)
    except OSError as error:
        raise Failure(EXIT_INTERNAL, f"stage /proc: {error.strerror}") from None
    _mount(libc, "proc", staged, "proc", flags, None, "mount proc")
    if libc.umount2(b"/proc", MNT_DETACH) != 0 and ctypes.get_errno() != errno.EINVAL:
        _check(-1, "detach /proc")
    _mount(libc, staged, "/proc", None, MS_MOVE, None, "move proc")
    _check(libc.umount2(os.fsencode(_PROC_STAGE), 0), "unstage /proc")


def _apply_plan(libc, plan: dict) -> None:
    """Everything read-only, ``hide`` covered, the grants bound back."""
    hide = list(dict.fromkeys(plan.get("hide", [])))
    if "/" in hide:
        raise Failure(EXIT_INTERNAL, "cannot hide /")
    grants = [(path, False) for path in plan.get("read", [])]
    grants += [(path, True) for path in plan.get("write", [])]
    # Opened before anything is hidden; parents bound before children.
    sources = []
    for path, write in sorted(grants, key=lambda grant: grant[0].count("/")):
        try:
            descriptor = os.open(path, os.O_PATH | O_CLOEXEC)
            sources.append((path, write, descriptor, os.statvfs(path).f_flag))
        except OSError as error:
            raise Failure(EXIT_INTERNAL, f"grant {path}: {error.strerror}") from None
    points = {point for _device, _root, point in mounts()}
    for point in sorted(points, key=lambda point: (point.count("/"), point)):
        if not _within(point, "/proc"):
            _read_only(libc, point)
    hiding = MS_NOSUID | MS_NODEV | MS_NOEXEC
    for path in hide:
        # Gone when it lies in a path hidden before it.
        if not os.path.lexists(path):
            continue
        if os.path.isdir(path):
            _mount(libc, "tmpfs", path, "tmpfs", hiding, "mode=0755", f"hide {path}")
        else:
            _mount(libc, "/dev/null", path, None, MS_BIND, None, f"hide {path}")
    for path, write, descriptor, kept in sources:
        # A grant below a hidden path needs its mount point in the tmpfs.
        try:
            if os.path.lexists(path):
                pass
            elif stat.S_ISDIR(os.fstat(descriptor).st_mode):
                os.makedirs(path, 0o755)
            else:
                os.makedirs(os.path.dirname(path), 0o755, exist_ok=True)
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT | O_CLOEXEC, 0o644))
        except OSError as error:
            raise Failure(EXIT_INTERNAL, f"grant {path}: {error.strerror}") from None
        source = f"/proc/self/fd/{descriptor}"
        _mount(libc, source, path, None, MS_BIND | MS_REC, None, f"grant {path}")
        flags = MS_REMOUNT | MS_BIND | remount_flags(kept)
        if not write or kept & os.ST_RDONLY:
            flags |= MS_RDONLY
        _mount(libc, None, path, None, flags, None, f"grant {path}")
        os.close(descriptor)
    for path in hide:
        _read_only(libc, path)
    if plan.get("shm") and os.path.isdir("/dev/shm"):
        flags = MS_NOSUID | MS_NODEV
        _mount(libc, "tmpfs", "/dev/shm", "tmpfs", flags, "mode=1777", "mount /dev/shm")


def box(plan: dict, program: bytes, argv) -> int:
    """Run ``argv`` boxed (see the module docstring); its exit status."""
    libc = load_libc()
    uid, gid = os.geteuid(), os.getegid()
    if uid == 0:
        raise Failure(EXIT_INTERNAL, "the box needs a non-root user")
    libc.unshare.argtypes = (ctypes.c_int,)
    libc.umount2.argtypes = (ctypes.c_char_p, ctypes.c_int)
    libc.mount.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
    )
    flags = CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWIPC
    _check(libc.unshare(flags), "unshare")
    if os.path.exists("/proc/self/setgroups"):
        _write("/proc/self/setgroups", "deny\n", "setgroups")
    # The user's own ids: non-zero, so the command keeps no capability.
    _write("/proc/self/uid_map", f"{uid} {uid} 1\n", "uid_map")
    _write("/proc/self/gid_map", f"{gid} {gid} 1\n", "gid_map")
    # Only now: a non-dumpable process could not write its maps.
    libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    _mount(libc, None, "/", None, MS_REC | MS_PRIVATE, None, "make / private")
    watched = set(BOX_CONTROL) | {signal.SIGCHLD}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    # pid 1 sees EOF here once this process is gone (its ppid reads 0).
    alive, holder = os.pipe()
    init = os.fork()
    if init == 0:
        os.close(holder)
        _init(libc, plan, program, argv, alive, mask)
    os.close(alive)

    def forward(signum, frame) -> None:
        del frame
        try:
            os.kill(init, signum)
        except OSError:
            pass

    for signum in BOX_CONTROL:
        signal.signal(signum, forward)
    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    _pid, status = os.waitpid(init, 0)
    return exit_status(status)


def _init(libc, plan: dict, program: bytes, argv, alive: int, mask) -> None:
    """pid 1: set the box up, start the command as pid 2, relay control
    signals, reap; leave with the command's status. Never returns."""
    try:
        libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        os.set_blocking(alive, False)
        try:
            if os.read(alive, 1) == b"":
                os._exit(EXIT_INTERNAL)
        except BlockingIOError:
            pass
        os.close(alive)
        # The kernel drops a signal sent from inside the box to pid 1 only
        # when its action is the default: no Python handler (SIGINT's) stays,
        # so the command cannot interrupt pid 1.
        for signum in signal.valid_signals():
            if callable(signal.getsignal(signum)):
                signal.signal(signum, signal.SIG_DFL)
        # A handler: the kernel drops default-action signals to a pid 1.
        for signum in set(BOX_CONTROL) | {signal.SIGCHLD}:
            signal.signal(signum, lambda *_: None)
        _replace_proc(libc)
        if plan:
            _apply_plan(libc, plan)
        if os.path.isdir("/dev/mqueue"):
            flags = MS_NOSUID | MS_NODEV | MS_NOEXEC
            _mount(libc, "mqueue", "/dev/mqueue", "mqueue", flags, None, "mqueue")
        _check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
        if program:
            try:
                Seccomp(libc, program).install()
            except OSError as error:
                raise Failure(EXIT_INTERNAL, f"seccomp: {error.strerror}") from None
        command = os.fork()
        if command == 0:
            _exec(argv, plan.get("cwd"), mask)
        _supervise(command)
    except Failure as failure:
        sys.stderr.write(f"rsi-box: {failure.message}\n")
    except BaseException as error:
        sys.stderr.write(f"rsi-box: internal error: {error}\n")
    sys.stderr.flush()
    os._exit(EXIT_INTERNAL)


def _exec(argv, cwd, mask) -> None:
    """pid 2: the command, in ``cwd`` (looked up again inside the box)."""
    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    # Python ignores these at start; exec would keep them ignored.
    for signum in (signal.SIGPIPE, signal.SIGXFSZ):
        signal.signal(signum, signal.SIG_DFL)
    if cwd is not None:
        try:
            os.chdir(cwd)
        except OSError as error:
            raise Failure(EXIT_INTERNAL, f"cwd {cwd}: {error.strerror}") from None
    try:
        os.execvp(argv[0], argv)
    except OSError as error:
        sys.stderr.write(f"rsi-box: cannot execute {argv[0]}: {error.strerror}\n")
        sys.stderr.flush()
        found = not isinstance(error, FileNotFoundError)
        os._exit(EXIT_CANNOT_EXECUTE if found else EXIT_NOT_FOUND)


def _supervise(command: int) -> None:
    watched = set(BOX_CONTROL) | {signal.SIGCHLD}
    while True:
        info = signal.sigwaitinfo(watched)
        if info.si_signo != signal.SIGCHLD:
            action = control_action(info.si_signo, info.si_code, info.si_pid)
            if action is not None:
                try:
                    os.kill(-1, action)
                except OSError:
                    pass
            continue
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
            if pid == command:
                os._exit(exit_status(status))


class _BoxParser(argparse.ArgumentParser):
    def error(self, message):
        raise Failure(EXIT_INTERNAL, message)


def box_main(argv=None) -> int:
    """rsi-box (see the module docstring)."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--" not in arguments or arguments[-1] == "--":
            raise Failure(EXIT_INTERNAL, "give the command after --")
        index = arguments.index("--")
        parser = _BoxParser(prog="rsi-box", add_help=False)
        layer = parser.add_mutually_exclusive_group(required=True)
        layer.add_argument("--work", action="store_true")
        layer.add_argument("--judge", action="store_true")
        layer.add_argument("--plan")
        parser.add_argument("--no-unix", action="store_true")
        options = parser.parse_args(arguments[:index])
        if options.no_unix and not options.judge:
            raise Failure(EXIT_INTERNAL, "--no-unix needs --judge")
        if options.plan is not None:
            try:
                plan = json.loads(options.plan)
                program = bytes.fromhex(plan.get("filter", ""))
            except (ValueError, AttributeError):
                raise Failure(EXIT_INTERNAL, "--plan is not a valid plan") from None
        else:
            plan = {}
            program = seccomp_program(
                no_network=False,
                truncate_guard=False,
                namespace_guard=options.work,
                deny_unix=options.no_unix,
            )
        return box(plan, program, arguments[index + 1 :])
    except Failure as failure:
        sys.stderr.write(f"rsi-box: {failure.message}\n")
        return failure.code


if __name__ == "__main__":
    sys.exit(box_main())
