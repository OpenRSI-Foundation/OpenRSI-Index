"""Isolation shared by rsi-agent and rsi-isolate, inside a Judge.

How a child is isolated, chosen automatically: when the Judge is root, the
child runs as a dedicated unprivileged uid; otherwise it keeps the Judge's
uid and is confined by Landlock; without either it runs only under the
operator's same-user fallback. This module holds the pieces both programs
use: mode detection, the Landlock ruleset, the seccomp filter, uid slots,
the child's setup before exec, and ending whatever a child left running.

Standard library only, Python 3.8 or later, no RSI Harness imports.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import resource
import signal
import stat
import struct
import subprocess
import sys
import time

EXIT_TIMEOUT = 124
EXIT_INTERNAL = 125
EXIT_TERMINATED = 143
KILL_GRACE_SECONDS = 5.0
SLOT_POLL_SECONDS = 0.2
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PR_SET_PDEATHSIG = 1
PR_SET_SECCOMP = 22
PR_SET_CHILD_SUBREAPER = 36
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2
# Landlock: the same syscall numbers on x86_64 and aarch64.
LANDLOCK_CREATE_RULESET = 444
LANDLOCK_ADD_RULE = 445
LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_MAX_ABI = 6
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
        "io_uring_setup": 425,
        "x32_mask": 0x40000000,
        "truncate": 76,
        "open": 2,
        "openat": 257,
        "openat2": 437,
    },
    "aarch64": {
        "arch": 0xC00000B7,
        "socket": 198,
        "io_uring_setup": 425,
        "x32_mask": None,
        "truncate": 45,
        "open": None,
        "openat": 56,
        "openat2": 437,
    },
}
_BPF_LD_W_ABS = 0x20
_BPF_JEQ_K = 0x15
_BPF_JGE_K = 0x35
_BPF_RET_K = 0x06
_BPF_JA = 0x05
_BPF_AND_K = 0x54
_O_TRUNC_FLAG = 0o1000
_O_ACCMODE = 0o3
_RET_KILL_PROCESS = 0x80000000
_RET_ERRNO = 0x00050000
_RET_ALLOW = 0x7FFF0000
_AF_INET = 2
_AF_INET6 = 10


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
) -> dict:
    """``uid`` (no ``problem``), ``landlock`` (``abi`` >= 1), or ``same_user``.

    ``same_user`` only with the operator's ``fallback`` "same-user";
    otherwise ``unavailable_code`` with the reasons.
    """
    report = {"landlock_abi": abi, "fallback": fallback}
    if problem is None:
        return dict(report, mode="uid")
    if abi >= 1:
        return dict(report, mode="landlock")
    if fallback == "same-user":
        return dict(report, mode="same_user")
    raise Failure(
        unavailable_code,
        f"cannot isolate {what}: the Judge is not root and Landlock is unavailable",
        f"{problem}; Landlock: {reason}",
    )


# -- Landlock -------------------------------------------------------------------


class Ruleset:
    """A Landlock ruleset built by the parent; the child enforces it.

    Every filesystem right of the ABI is handled, so a path without a rule
    grants nothing. From ABI 6, signals and abstract unix sockets stay inside
    the child's domain. The network is not handled.
    """

    def __init__(self, libc, abi: int) -> None:
        self.libc = libc
        self.abi = min(abi, LANDLOCK_MAX_ABI)
        self.handled = sum(
            1 << bit
            for bit, since in enumerate(LANDLOCK_FS_RIGHTS)
            if since <= self.abi
        )
        scoped = 0b11 if self.abi >= 6 else 0
        size = 8 if self.abi < 4 else 16 if self.abi < 6 else 24
        self._attr = ctypes.create_string_buffer(
            struct.pack("<QQQ", self.handled, 0, scoped)[:size], size
        )
        fd = syscall(
            libc, LANDLOCK_CREATE_RULESET, ctypes.addressof(self._attr), size, 0
        )
        if fd < 0:
            raise Failure(
                EXIT_INTERNAL,
                "landlock_create_ruleset failed: " + os.strerror(ctypes.get_errno()),
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
    machine: str | None = None,
    pointer_bytes: int | None = None,
    byteorder: str | None = None,
) -> bytes:
    """A seccomp filter as packed ``struct sock_filter[]``.

    A foreign ``seccomp_data.arch`` is killed; x32 syscall numbers (x86_64)
    and ``io_uring_setup`` get EPERM. With ``no_network``, ``socket`` with
    domain AF_INET or AF_INET6 gets EPERM (the low 32 bits of the domain, as
    the kernel truncates ``int family``). With ``truncate_guard`` (Landlock
    ABI 1 and 2, which cannot control truncation) ``truncate`` and an
    ``open``/``openat`` with O_TRUNC and a read-only access mode get EPERM,
    and ``openat2`` ENOSYS. Everything else is allowed. Any machine other
    than x86_64 or aarch64, a 32-bit Python or a big-endian host raises
    :class:`Failure` (125).
    """
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
    program: list = [
        (_BPF_LD_W_ABS, 0, 0, 4),
        (_BPF_JEQ_K, 1, 0, spec["arch"]),
        (_BPF_RET_K, 0, 0, _RET_KILL_PROCESS),
        (_BPF_LD_W_ABS, 0, 0, 0),
    ]
    if spec["x32_mask"] is not None:
        program.append((_BPF_JGE_K, "deny", 0, spec["x32_mask"]))
    program.append((_BPF_JEQ_K, "deny", 0, spec["io_uring_setup"]))
    if truncate_guard:
        program += [
            (_BPF_JEQ_K, "deny", 0, spec["truncate"]),
            (_BPF_JEQ_K, "enosys", 0, spec["openat2"]),
            (_BPF_JEQ_K, "openat_flags", 0, spec["openat"]),
        ]
        if spec["open"] is not None:
            program.append((_BPF_JEQ_K, "open_flags", 0, spec["open"]))
    if no_network:
        program += [
            (_BPF_JEQ_K, 0, "allow", spec["socket"]),
            (_BPF_LD_W_ABS, 0, 0, 16),
            (_BPF_JEQ_K, "deny", 0, _AF_INET),
            (_BPF_JEQ_K, "deny", 0, _AF_INET6),
        ]
    if truncate_guard:
        # The low 32 bits of the flags argument: deny O_TRUNC with O_RDONLY.
        program += [
            (_BPF_JA, 0, 0, "allow"),
            ("open_flags", None, None, None),
            (_BPF_LD_W_ABS, 0, 0, 16 + 8),
            (_BPF_JA, 0, 0, "flags"),
            ("openat_flags", None, None, None),
            (_BPF_LD_W_ABS, 0, 0, 16 + 16),
            ("flags", None, None, None),
            (_BPF_AND_K, 0, 0, _O_TRUNC_FLAG | _O_ACCMODE),
            (_BPF_JEQ_K, "deny", "allow", _O_TRUNC_FLAG),
            ("enosys", None, None, None),
            (_BPF_RET_K, 0, 0, _RET_ERRNO | errno.ENOSYS),
        ]
    program += [
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
