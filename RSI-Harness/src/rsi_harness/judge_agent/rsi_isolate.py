"""rsi-isolate: run one command isolated, from inside a Judge.

RSI Harness mounts this program read-only at /opt/rsi-isolate, with the
launcher /usr/local/bin/rsi-isolate, in every Judge. A Judge procedure uses
it to run code it must not trust, such as a candidate's, instead of
switching users itself.

    rsi-isolate run [--read PATH]... [--write PATH]... [--no-network]
                    [--timeout SEC] [--cwd DIR] [--env NAME[=VALUE]]...
                    [--pass-fd FD]... -- COMMAND [ARG]...
    rsi-isolate isolation

``run`` runs COMMAND with the caller's stdin, stdout and stderr. It reads
only system paths, its own executable and ``--read`` paths, writes only
``--write`` paths, /dev/shm and a private HOME and TMPDIR (deleted
afterwards), and gets only PATH, HOME, TMPDIR and each ``--env``. The
isolation is chosen automatically: when the Judge is root, the command runs
as its own unprivileged uid (62000 to 62063, one per concurrent call) with
no supplementary groups, and the write grants are handed to that uid for the
call, which caps that uid at 4096 processes and threads; otherwise it keeps
the Judge's uid and is confined by Landlock (Linux 5.13 or later), or else
(``userns``) runs in its own user, mount, PID and IPC namespaces, where every
top-level entry but the system paths and the grants is hidden, everything
but the write grants is read-only, and /proc and /dev/shm are its own.
Landlock also confines a root Judge's command where the kernel has it; where
it does not, the call fails while /tests or /logs/verifier is open to other
users. Without any of them the call fails, unless the operator set
RSI_JUDGE_ISOLATION_FALLBACK=same-user. In every mode the command gets no
kernel keyrings and no netlink socket but NETLINK_ROUTE. ``--no-network``
allows only AF_UNIX and NETLINK_ROUTE sockets; where the Judge shares a
network namespace (RSI_JUDGE_NETNS is not "private") AF_UNIX sockets too are
refused unless Landlock scopes abstract sockets (a stream or seqpacket
socketpair stays). When the command ends, or ``--timeout`` expires,
everything it started is killed. ``rsi-isolate isolation`` prints the mode
as JSON.

Exit status: the command's own (128+N when signal N ended it); 123 the
command cannot be isolated; 124 timeout; 125 a usage error, a refused grant
or a setup failure; 126 the command cannot be executed; 127 the command is
not found; 128+N rsi-isolate itself received signal N.

Standard library only (and judge_isolation.py beside it), Python 3.8 or
later, no RSI Harness imports.
"""

from __future__ import annotations

import argparse
import grp
import json
import math
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

if __package__:
    from . import judge_isolation as isolation
else:
    # Run as a script (/opt/rsi-isolate/rsi_isolate.py): -I adds no directory.
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import judge_isolation as isolation

EXIT_UNAVAILABLE = 123
EXIT_TIMEOUT = isolation.EXIT_TIMEOUT
EXIT_SETUP = isolation.EXIT_INTERNAL
EXIT_CANNOT_EXECUTE = 126
EXIT_NOT_FOUND = 127
# The isolated users of a root Judge (gid = uid): RSI Harness's
# JUDGE_ISOLATE_UID and JUDGE_ISOLATE_SLOTS.
ISOLATE_UID = 62000
ISOLATE_SLOTS = 64
# The processes and threads one isolated user may run at once.
ISOLATE_NPROC = 4096
FALLBACK_ENV = "RSI_JUDGE_ISOLATION_FALLBACK"
# "private" when the Judge has its own network namespace (set by RSI Harness).
NETNS_ENV = "RSI_JUDGE_NETNS"
SLOT_DIRECTORY = "/tmp/.rsi-isolate-slots"
PRIVATE_PARENT = "/tmp"
# Always readable (and executable).
SYSTEM_PATHS = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/etc",
    "/opt",
    "/proc",
    "/sys",
)
# Never granted: the Judge's outputs and rsi-agent's state.
NEVER_GRANTED = ("/logs/verifier", "/run/rsi-agent")
# Never writable.
NEVER_WRITABLE = SYSTEM_PATHS + ("/dev", "/tests", "/logs/agent", "/run/rsi-harness")
PROTECTED = ("/logs/verifier", "/tests")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _say(message: str) -> None:
    sys.stderr.write("rsi-isolate: " + message.rstrip("\n") + "\n")
    sys.stderr.flush()


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise isolation.Failure(EXIT_SETUP, message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="rsi-isolate",
        description="Run one command isolated from the Judge.",
    )
    commands = parser.add_subparsers(dest="command", metavar="{run,isolation}")
    commands.required = True
    run = commands.add_parser(
        "run",
        help="run COMMAND isolated (put it after --)",
        usage="rsi-isolate run [options] -- COMMAND [ARG]...",
        description=(
            "Run COMMAND isolated: it reads only system paths, its executable "
            "and --read paths, and writes only --write paths and a private "
            "HOME and TMPDIR."
        ),
    )
    run.add_argument(
        "--read",
        action="append",
        default=[],
        metavar="PATH",
        help="let the command read and execute PATH and everything below it",
    )
    run.add_argument(
        "--write",
        action="append",
        default=[],
        metavar="PATH",
        help="let the command read and write PATH and everything below it",
    )
    run.add_argument(
        "--no-network",
        action="store_true",
        help="allow only AF_UNIX and NETLINK_ROUTE sockets",
    )
    run.add_argument(
        "--timeout",
        type=float,
        metavar="SEC",
        help="seconds before the command and everything it started are ended",
    )
    run.add_argument(
        "--cwd",
        metavar="DIR",
        help="the working directory, inside a grant (default: the private HOME)",
    )
    run.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="set NAME, or pass the caller's NAME on",
    )
    run.add_argument(
        "--pass-fd",
        action="append",
        default=[],
        type=int,
        metavar="FD",
        help="keep this inherited descriptor open",
    )
    commands.add_parser(
        "isolation",
        help="print how a command would be isolated, as JSON",
        description=(
            "Print {mode, landlock_abi, fallback}, plus landlock_scoped and "
            "landlock_signal_scope when Landlock is available, or namespaces, "
            "proc and shm in userns mode; exit 123 when a command cannot be "
            "isolated."
        ),
    )
    return parser


# -- the mode -------------------------------------------------------------------


def _ids_in_use() -> str | None:
    """A passwd or group entry of the image inside the isolated id range."""
    for number in range(ISOLATE_UID, ISOLATE_UID + ISOLATE_SLOTS):
        for lookup, kind in ((pwd.getpwuid, "uid"), (grp.getgrgid, "gid")):
            try:
                lookup(number)
            except KeyError:
                continue
            return f"the image uses {kind} {number}"
    return None


def _open_protected() -> str | None:
    """A protected path that other users can reach."""
    for item in PROTECTED:
        try:
            mode = stat.S_IMODE(os.stat(item).st_mode)
        except FileNotFoundError:
            continue
        if mode & 0o007:
            return f"{item} (mode {mode:04o})"
    return None


def isolation_mode(libc) -> dict:
    """``uid`` (a root Judge), ``landlock``, ``userns``, or the operator's
    ``same_user``."""
    fallback = "same-user" if os.environ.get(FALLBACK_ENV) == "same-user" else "none"
    problem = isolation.uid_mode_problem(ISOLATE_UID, ISOLATE_SLOTS)
    if problem is None:
        problem = _ids_in_use()
    abi, reason = isolation.landlock_abi(libc)
    mode = isolation.choose_mode(
        problem,
        abi,
        reason,
        fallback,
        EXIT_UNAVAILABLE,
        "the command",
        isolation.userns_problem,
    )
    # Without Landlock only file permissions confine the isolated user.
    opened = _open_protected() if mode["mode"] == "uid" and abi < 1 else None
    if opened is not None:
        raise isolation.Failure(
            EXIT_UNAVAILABLE,
            f"cannot isolate the command: Landlock is unavailable and {opened} is "
            "open to other users",
            f"Landlock: {reason}",
        )
    return mode


# -- grants ---------------------------------------------------------------------


def _overlaps(path: str, other: str) -> bool:
    return (
        path == other
        or path.startswith(other.rstrip("/") + "/")
        or other.startswith(path.rstrip("/") + "/")
    )


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _grant(path: str, flag: str, write: bool) -> str:
    if not path:
        raise isolation.Failure(EXIT_SETUP, f"{flag} needs a path")
    real = os.path.realpath(path)

    def refuse(reason: str) -> isolation.Failure:
        return isolation.Failure(EXIT_SETUP, f"refusing {flag} {path}: {reason}")

    for item in NEVER_GRANTED:
        if _overlaps(real, item):
            raise refuse(f"it overlaps {item}, which is never granted")
    if write:
        if real.count("/") < 2:
            raise refuse("it must lie below a top-level directory")
        for item in NEVER_WRITABLE:
            if _overlaps(real, item):
                raise refuse(f"it overlaps {item}, which is never writable")
    if not os.path.lexists(real):
        raise refuse("it does not exist")
    if write:
        _refuse_hard_links(real, flag)
    return real


def _tree(root: str):
    """``root`` and everything below it, never following a symlink."""
    yield root, os.lstat(root)
    if not stat.S_ISDIR(os.lstat(root).st_mode):
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = os.path.join(current, name)
            try:
                yield path, os.lstat(path)
            except FileNotFoundError:
                continue


def _refuse_hard_links(root: str, flag: str) -> None:
    """A hard link would let the command write a file that also lives
    outside the grant."""
    for path, metadata in _tree(root):
        if not stat.S_ISDIR(metadata.st_mode) and metadata.st_nlink > 1:
            raise isolation.Failure(
                EXIT_SETUP,
                f"refusing {flag} {root}: {path} has other hard links",
            )


def _check_reachable(reads, writes) -> None:
    """In a root Judge the command is another user: every read grant and its
    parent directories must be open to other users."""

    def granted(path: str) -> bool:
        return any(_within(path, root) for root in writes)

    for path in reads:
        if granted(path):
            continue
        chain = []
        parent = os.path.dirname(path)
        while True:
            chain.append((parent, stat.S_IXOTH))
            if parent == "/":
                break
            parent = os.path.dirname(parent)
        metadata = os.stat(path)
        need = stat.S_IROTH | (stat.S_IXOTH if stat.S_ISDIR(metadata.st_mode) else 0)
        for item, bits in reversed(chain + [(path, need)]):
            if granted(item):
                continue
            mode = stat.S_IMODE(os.stat(item).st_mode)
            if mode & bits != bits:
                raise isolation.Failure(
                    EXIT_SETUP,
                    f"refusing --read {path}: {item} (mode {mode:04o}) is not open "
                    "to other users, so the isolated user cannot reach it; stage "
                    "the input in a directory of mode 0755",
                )


def _check_protected() -> None:
    ids = range(ISOLATE_UID, ISOLATE_UID + ISOLATE_SLOTS)
    for item in PROTECTED:
        try:
            metadata = os.lstat(item)
        except FileNotFoundError:
            continue
        if metadata.st_uid in ids or metadata.st_gid in ids:
            raise isolation.Failure(EXIT_SETUP, f"{item} is owned by an isolated user")


def _chown_tree(root: str, owner, only_uid=None) -> None:
    """lchown ``root`` and everything below it (only entries of ``only_uid``
    when given); setuid and setgid bits are cleared."""
    for path, metadata in _tree(root):
        if only_uid is not None and metadata.st_uid != only_uid:
            continue
        os.lchown(path, owner[0], owner[1])
        if stat.S_ISREG(metadata.st_mode) and metadata.st_mode & 0o6000:
            os.chmod(path, stat.S_IMODE(metadata.st_mode) & ~0o6000)


class _Handover:
    """A --write grant owned by the isolated user for the call (uid mode)."""

    def __init__(self, path: str, owner) -> None:
        self.path = path
        self.owner = owner
        self.original = os.lstat(path)

    def take(self) -> None:
        _chown_tree(self.path, self.owner)

    def restore(self) -> None:
        original = (self.original.st_uid, self.original.st_gid)
        _chown_tree(self.path, original, only_uid=self.owner[0])
        if not stat.S_ISLNK(os.lstat(self.path).st_mode):
            os.chmod(self.path, stat.S_IMODE(self.original.st_mode))


def _environment(entries) -> dict:
    environment = {"PATH": os.environ.get("PATH") or isolation.DEFAULT_PATH}
    for entry in entries:
        name, separator, value = entry.partition("=")
        if not ENV_NAME.match(name):
            raise isolation.Failure(EXIT_SETUP, f"--env {entry}: not a variable name")
        if name in (FALLBACK_ENV, isolation.ALLOW_MISSING_SIGNAL_ENV):
            raise isolation.Failure(
                EXIT_SETUP,
                f"--env {name}: operator isolation policy is never passed on",
            )
        if separator:
            environment[name] = value
        elif name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _pass_fds(values) -> tuple:
    for descriptor in values:
        if descriptor < 3:
            raise isolation.Failure(
                EXIT_SETUP,
                f"--pass-fd {descriptor}: stdin, stdout and stderr pass anyway",
            )
        try:
            os.fstat(descriptor)
        except OSError:
            raise isolation.Failure(
                EXIT_SETUP, f"--pass-fd {descriptor}: not an open descriptor"
            ) from None
    return tuple(values)


def _executable(command: str, path: str, cwd: str) -> str | None:
    if "/" in command:
        found = os.path.join(cwd, command)
    else:
        found = shutil.which(command, path=path)
    return os.path.realpath(found) if found and os.path.isfile(found) else None


def _interpreter_paths() -> list:
    """This Python's installation, so a command may run it."""
    paths = {os.path.realpath(sys.executable)} if sys.executable else set()
    for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        if prefix:
            paths.add(os.path.realpath(prefix))
    return sorted(path for path in paths if path != "/")


def _allow_missing_landlock_signal() -> bool:
    """Only the Harness sets this Judge variable from the operator policy."""
    value = os.environ.get(isolation.ALLOW_MISSING_SIGNAL_ENV, "0")
    if value not in ("0", "1"):
        raise isolation.Failure(
            EXIT_SETUP, isolation.ALLOW_MISSING_SIGNAL_ENV + " must be 0 or 1"
        )
    return value == "1"


def _ruleset(libc, abi: int, reads, writes, private: str, executable):
    read = list(SYSTEM_PATHS) + _interpreter_paths() + list(reads)
    read += [executable] if executable is not None else []
    write = list(writes) + [private]
    # A rule on another mount of what is never granted would open it.
    isolation.hidden_aliases(NEVER_GRANTED, read + write, EXIT_SETUP)
    ruleset = isolation.Ruleset(
        libc, abi, allow_missing_landlock_signal=_allow_missing_landlock_signal()
    )
    try:
        for path in read:
            ruleset.allow(path, isolation.FS_READ)
        isolation.allow_devices(ruleset)
        if os.path.isdir("/dev/shm") and not os.path.islink("/dev/shm"):
            ruleset.allow("/dev/shm", ruleset.handled)
        for path in write:
            ruleset.allow(path, ruleset.handled)
    except BaseException:
        ruleset.close()
        raise
    return ruleset


def _slot_directory() -> str:
    """The uid slots' lock files: a root-only directory, replaced when an
    isolated user made it first."""
    for _ in range(3):
        try:
            os.mkdir(SLOT_DIRECTORY, 0o700)
        except FileExistsError:
            pass
        else:
            os.chmod(SLOT_DIRECTORY, 0o700)
        metadata = os.lstat(SLOT_DIRECTORY)
        if (
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and not stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            return SLOT_DIRECTORY
        if stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(SLOT_DIRECTORY, ignore_errors=True)
        else:
            os.unlink(SLOT_DIRECTORY)
    raise isolation.Failure(EXIT_SETUP, f"cannot make {SLOT_DIRECTORY} private")


def _remove_shm(uid: int) -> bool:
    """Remove what ``uid`` left in /dev/shm; True when nothing remains."""
    try:
        entries = list(os.scandir("/dev/shm"))
    except OSError:
        return True
    clean = True
    for entry in entries:
        try:
            if entry.stat(follow_symlinks=False).st_uid != uid:
                continue
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
        except FileNotFoundError:
            continue
        except OSError:
            clean = False
    return clean


def _deny_unix(no_network: bool, ruleset) -> bool:
    """--no-network refuses AF_UNIX sockets too where abstract sockets reach
    beyond the Judge: a shared network namespace without Landlock scoping."""
    if not no_network or os.environ.get(NETNS_ENV) == "private":
        return False
    scope = isolation.LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET
    return ruleset is None or not ruleset.scoped & scope


def _top_level(grants, root: str = "/") -> list:
    """Entries of ``root`` that are no system path, /dev or grant; symlinks
    are skipped (they lead to a hidden path or a shown one)."""
    shown = {os.path.basename(path) for path in SYSTEM_PATHS} | {"dev"}
    hidden = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name not in shown and path not in grants and not os.path.islink(path):
            hidden.append(path)
    return hidden


def _userns_plan(reads, writes, private, cwd, executable, program) -> dict:
    """The box of a userns call (judge_isolation.box_argv): only the system
    paths and the grants show; the write grants and HOME are writable; the
    other mounts of what is never granted are hidden too."""
    write = list(writes) + [private]
    granted = SYSTEM_PATHS + tuple(reads) + tuple(write)
    extra = [
        path
        for path in _interpreter_paths() + ([executable] if executable else [])
        if not any(_within(path, root) for root in granted)
    ]
    read = list(reads) + extra
    aliases = isolation.hidden_aliases(NEVER_GRANTED, read + write, EXIT_SETUP)
    return {
        "hide": _top_level(read + write) + aliases,
        "read": read,
        "write": write,
        "cwd": cwd,
        "shm": True,
        "filter": program.hex(),
    }


# -- run ------------------------------------------------------------------------


def run(options: argparse.Namespace, command: list) -> int:
    """``rsi-isolate run``: the command's exit status (see the module docstring)."""
    started = time.monotonic()
    if options.timeout is not None and not (
        options.timeout > 0 and math.isfinite(options.timeout)
    ):
        raise isolation.Failure(EXIT_SETUP, "--timeout must be a positive number")
    deadline = None if options.timeout is None else started + options.timeout
    environment = _environment(options.env)
    pass_fds = _pass_fds(options.pass_fd)
    reads = [_grant(path, "--read", write=False) for path in options.read]
    writes = [_grant(path, "--write", write=True) for path in options.write]
    cwd = None
    if options.cwd is not None:
        cwd = os.path.realpath(options.cwd)
        if not os.path.isdir(cwd):
            raise isolation.Failure(
                EXIT_SETUP, f"--cwd {options.cwd} is not a directory"
            )
        if not any(_within(cwd, root) for root in reads + writes):
            raise isolation.Failure(
                EXIT_SETUP, f"--cwd {options.cwd} must lie inside a --read or --write"
            )

    libc = isolation.load_libc()
    mode = isolation_mode(libc)
    uid_mode = mode["mode"] == "uid"
    userns = mode["mode"] == "userns"
    abi = mode["landlock_abi"]
    if not uid_mode:
        # The command runs as this user: it may not read or trace this process.
        libc.prctl(isolation.PR_SET_DUMPABLE, 0, 0, 0, 0)
    if mode["mode"] == "same_user":
        _say(
            "no filesystem isolation: the operator's same-user fallback runs the "
            "command as the Judge's user"
        )
    if uid_mode:
        _check_protected()
        _check_reachable(reads, writes)

    # Whatever the command starts stays a descendant, even after a double fork.
    libc.prctl(isolation.PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    owner = None
    slot_fd = None
    private = None
    cwd_fd = None
    handovers: list = []
    ruleset = None
    process = None
    timed_out = False
    try:
        if uid_mode:
            slot, slot_fd = isolation.claim_slot(
                _slot_directory(),
                ISOLATE_SLOTS,
                math.inf if deadline is None else deadline,
                f"all {ISOLATE_SLOTS} isolated users stayed busy with other calls",
                "the command",
            )
            owner = (ISOLATE_UID + slot, ISOLATE_UID + slot)
            # A killed earlier call may have left processes or files.
            if not _end_uid(owner[0]) or not _remove_shm(owner[0]):
                raise isolation.Failure(EXIT_SETUP, "cannot end an earlier call")
        private = tempfile.mkdtemp(prefix="rsi-isolate-", dir=PRIVATE_PARENT)
        if owner is not None:
            os.chown(private, owner[0], owner[1])
        environment = dict(
            {"HOME": private, "TMPDIR": private},
            **environment,
        )
        if cwd is None:
            cwd = private
        cwd_fd = os.open(cwd, os.O_RDONLY | isolation.O_DIRECTORY | isolation.O_CLOEXEC)
        if owner is not None:
            for path in writes:
                handovers.append(_Handover(path, owner))
                handovers[-1].take()
        executable = _executable(command[0], environment["PATH"], cwd)
        if executable is not None and any(
            _overlaps(executable, item) for item in NEVER_GRANTED
        ):
            executable = None
        if abi >= 1:
            ruleset = _ruleset(libc, abi, reads, writes, private, executable)
        # Landlock below ABI 3 cannot control truncation: seccomp does.
        truncate_guard = ruleset is not None and abi < 3
        program = isolation.seccomp_program(
            no_network=options.no_network,
            truncate_guard=truncate_guard,
            trace_guard=mode["mode"] in ("same_user", "userns"),
            namespace_guard=userns,
            deny_unix=_deny_unix(options.no_network, ruleset),
        )
        argv = command
        seccomp = None
        if userns:
            plan = _userns_plan(reads, writes, private, cwd, executable, program)
            argv = isolation.box_argv(plan, command)
        else:
            seccomp = isolation.Seccomp(libc, program)
        if isolation.Stop.signum is not None:
            raise isolation.Failure(
                isolation.EXIT_TERMINATED, "terminated before the command started"
            )
        try:
            process = subprocess.Popen(
                argv,
                env=environment,
                close_fds=True,
                pass_fds=pass_fds,
                start_new_session=True,
                preexec_fn=isolation.preexec(  # noqa: PLW1509
                    cwd_fd,
                    owner,
                    libc,
                    ruleset,
                    seccomp,
                    nproc=None if owner is None else ISOLATE_NPROC,
                ),
            )
        except FileNotFoundError:
            raise isolation.Failure(
                EXIT_NOT_FOUND, f"{command[0]}: command not found"
            ) from None
        except subprocess.SubprocessError as error:
            raise isolation.Failure(
                EXIT_SETUP, f"cannot isolate the command: {error}"
            ) from None
        except OSError as error:
            raise isolation.Failure(
                EXIT_CANNOT_EXECUTE, f"cannot execute {command[0]}: {error.strerror}"
            ) from None
        finally:
            if ruleset is not None:
                ruleset.close()
        while True:
            try:
                process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if isolation.Stop.signum is not None:
                isolation.signal_group(process.pid, signal.SIGKILL)
                process.wait()
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                isolation.end_group(process, isolation.KILL_GRACE_SECONDS)
                break
        # Also end what the command left behind in its group.
        isolation.signal_group(process.pid, signal.SIGKILL)
    finally:
        problems = _clean_up(process, owner, cwd_fd, handovers, private, slot_fd)
    if isolation.Stop.signum is not None:
        raise isolation.Failure(
            128 + isolation.Stop.signum, "terminated; the command was stopped"
        )
    if problems:
        raise isolation.Failure(EXIT_SETUP, "cleanup failed: " + "; ".join(problems))
    if timed_out:
        raise isolation.Failure(
            EXIT_TIMEOUT,
            f"the command did not finish within {options.timeout:g} seconds",
        )
    code = process.returncode
    return 128 - code if code < 0 else code


def _end_uid(uid: int) -> bool:
    """End every process of an isolated ``uid``; True once none remain."""
    isolation.kill_uid(uid)
    return isolation.sweep_uid(uid)


def _clean_up(process, owner, cwd_fd, handovers, private, slot_fd) -> list:
    """End everything the call started and hand its places back."""
    problems = []
    if process is not None and process.returncode is None:
        isolation.signal_group(process.pid, signal.SIGKILL)
        process.wait()
    # The uid is this call's alone: end anything that escaped the group.
    if owner is not None and not _end_uid(owner[0]):
        problems.append(f"processes of uid {owner[0]} survived")
    # In every mode: end what the command left running, as its subreaper.
    if not isolation.end_descendants():
        problems.append("processes the command started survived")
    if cwd_fd is not None:
        os.close(cwd_fd)
    for handover in reversed(handovers):
        try:
            handover.restore()
        except OSError as error:
            problems.append(f"cannot hand {handover.path} back: {error}")
    if private is not None:
        shutil.rmtree(private, ignore_errors=True)
    if owner is not None and not _remove_shm(owner[0]):
        problems.append("cannot remove what the command left in /dev/shm")
    if slot_fd is not None:
        os.close(slot_fd)
    return problems


def report_isolation() -> int:
    """``rsi-isolate isolation``: print the mode a command would use."""
    allow_missing_signal = _allow_missing_landlock_signal()
    libc = isolation.load_libc()
    mode = isolation.probe_ruleset(
        libc,
        isolation_mode(libc),
        allow_missing_landlock_signal=allow_missing_signal,
    )
    sys.stdout.write(json.dumps(mode, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = None
    if "--" in arguments:
        index = arguments.index("--")
        arguments, command = arguments[:index], arguments[index + 1 :]
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, isolation.on_signal)
    try:
        options = _parser().parse_args(arguments)
        if options.command == "isolation":
            if command is not None:
                raise isolation.Failure(EXIT_SETUP, "isolation takes no command")
            return report_isolation()
        if not command:
            raise isolation.Failure(EXIT_SETUP, "give the command after --")
        return run(options, command)
    except isolation.Failure as failure:
        code = failure.code
        if code == isolation.EXIT_TERMINATED and isolation.Stop.signum is not None:
            code = 128 + isolation.Stop.signum
        _say(failure.message)
        if failure.detail:
            sys.stderr.write(failure.detail + "\n")
        return code
    except OSError as error:
        _say(f"internal error: {error}")
        return EXIT_SETUP


if __name__ == "__main__":
    sys.exit(main())
