"""rsi-agent: call the run's own Agent once, from inside a Judge.

RSI Harness mounts this program read-only at /opt/rsi-agent, with the
launcher /usr/local/bin/rsi-agent, in every Judge whose task declares
``[metadata.rsi_harness.verifier] model_access = true``, and writes the run's
launch spec into a private tmpfs. It is the only supported way for a Judge
procedure to call a model: the same Agent, CLI, credentials, provider, model
and reasoning effort as Work, so any Agent the Harness supports works here
without task changes.

    rsi-agent exec --prompt-file FILE --output FILE [--schema FILE]
                   [--timeout SEC] [--workdir DIR] [--log FILE]

The Agent reads the prompt, may read files (evidence placed in ``--workdir``
or anywhere world-readable), and its final message is written to
``--output``. With ``--schema`` the Agent is asked for JSON conforming to the
schema, through its own structured-output option when it has one; callers
validate the result either way.

When the Judge runs as root, each call's Agent runs as its own unprivileged
user, one of a small range of uids that no two concurrent calls share, with a
fresh private HOME holding copies of the Agent's login files. It cannot write
the Judge's outputs, read the launch spec or the credential store, signal the
Judge, or touch another call's files or processes. A timeout ends its whole
process group, and whatever still runs as the call's uid is killed when the
call ends. The Agent owns ``--workdir`` for the call: it may delete, replace
or add entries there, so the output and the log must lie outside it.

Exit status: 0 the output was written; 1 the Agent failed; 2 usage error;
3 model access is unavailable or the Agent cannot run one-shot; 124 timeout;
125 internal setup failure; 143 rsi-agent was terminated.

Standard library only, Python 3.8 or later, no RSI Harness imports.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import os
import resource
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

EXIT_OK = 0
EXIT_AGENT = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_TIMEOUT = 124
EXIT_INTERNAL = 125
EXIT_TERMINATED = 143

STATE_ROOT = "/run/rsi-agent"
# Tests only: point rsi-agent at another state root (the caller's own files).
STATE_ROOT_ENV = "RSI_AGENT_STATE_ROOT"
SPEC_VERSION = 1
DEFAULT_TIMEOUT = 1800.0
KILL_GRACE_SECONDS = 5.0
MAX_SLOTS = 1024
SLOT_POLL_SECONDS = 0.2
MAX_SPEC_BYTES = 8 * 1024 * 1024
MAX_PROMPT_BYTES = 4 * 1024 * 1024
MAX_SCHEMA_BYTES = 1024 * 1024
MAX_CREDENTIAL_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
STDERR_TAIL_BYTES = 2048
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
FINAL_MESSAGE = "final-message"
SCHEMA_REQUIREMENT = (
    "Your final message must be only a JSON value that conforms to this JSON Schema:\n"
)
REDACTED = "[REDACTED]"
PR_SET_PDEATHSIG = 1
PR_SET_NO_NEW_PRIVS = 38
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class Failure(Exception):
    """End the call with ``code`` and one reason line on stderr."""

    def __init__(self, code: int, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


class _Redactor:
    def __init__(self) -> None:
        self.values: list[str] = []

    def add(self, values) -> None:
        found = {str(value) for value in values if value}
        self.values = sorted(found | set(self.values), key=len, reverse=True)

    def text(self, value: str) -> str:
        for secret in self.values:
            value = value.replace(secret, REDACTED)
        return value

    def data(self, value: bytes) -> bytes:
        for secret in self.values:
            value = value.replace(secret.encode(), REDACTED.encode())
        return value


REDACT = _Redactor()


def _say(message: str) -> None:
    sys.stderr.write("rsi-agent: " + REDACT.text(message).rstrip("\n") + "\n")
    sys.stderr.flush()


# -- inputs ---------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsi-agent",
        description="Call the run's own Agent once from inside a Judge.",
    )
    commands = parser.add_subparsers(dest="command", metavar="exec")
    commands.required = True
    run = commands.add_parser(
        "exec",
        help="run the Agent on one prompt and write its final message",
        description="Run the run's Agent on one prompt; write its final message.",
    )
    run.add_argument("--prompt-file", required=True, help="the prompt (UTF-8)")
    run.add_argument("--output", required=True, help="where to write the answer")
    run.add_argument("--schema", help="a JSON Schema the answer must conform to")
    run.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds before the Agent is stopped (default {DEFAULT_TIMEOUT:g})",
    )
    run.add_argument(
        "--workdir",
        help=(
            "the Agent's working directory; the Agent owns it during the call "
            "and may change its entries"
        ),
    )
    run.add_argument("--log", help="also write the Agent's redacted stdout/stderr")
    return parser


def _read_bounded(path: str, limit: int, label: str) -> bytes:
    try:
        with open(path, "rb") as stream:
            data = stream.read(limit + 1)
    except OSError as error:
        raise Failure(
            EXIT_USAGE, f"cannot read the {label} {path}: {error.strerror}"
        ) from None
    if len(data) > limit:
        raise Failure(EXIT_USAGE, f"the {label} exceeds {limit} bytes")
    return data


def _load_spec(state_root: str) -> dict:
    path = os.path.join(state_root, "private", "launch.json")
    try:
        descriptor = os.open(path, os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW)
    except FileNotFoundError:
        raise Failure(
            EXIT_UNAVAILABLE,
            "this Judge has no model access (the task must declare "
            "[metadata.rsi_harness.verifier] model_access = true)",
        ) from None
    except PermissionError:
        raise Failure(
            EXIT_UNAVAILABLE,
            "the launch spec is not readable; run rsi-agent as the Judge user",
        ) from None
    except OSError as error:
        raise Failure(
            EXIT_UNAVAILABLE, f"cannot open the launch spec: {error.strerror}"
        ) from None
    with os.fdopen(descriptor, "rb") as stream:
        data = stream.read(MAX_SPEC_BYTES + 1)
    try:
        spec = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Failure(EXIT_INTERNAL, "the launch spec is not valid JSON") from None
    if not isinstance(spec, dict) or spec.get("version") != SPEC_VERSION:
        raise Failure(EXIT_INTERNAL, "the launch spec has an unsupported version")
    redact = spec.get("redact", [])
    if not isinstance(redact, list):
        raise Failure(EXIT_INTERNAL, "the launch spec redaction list is invalid")
    REDACT.add(redact)
    environment = spec.get("environment")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise Failure(EXIT_INTERNAL, "the launch spec environment is invalid")
    for key in ("uid", "gid"):
        value = spec.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise Failure(EXIT_INTERNAL, f"the launch spec {key} is invalid")
    slots = spec.setdefault("slots", 1)
    if (
        not isinstance(slots, int)
        or isinstance(slots, bool)
        or not 0 < slots <= MAX_SLOTS
    ):
        raise Failure(EXIT_INTERNAL, "the launch spec slots count is invalid")
    command = spec.get("command")
    if command is not None and not isinstance(command, str):
        raise Failure(EXIT_INTERNAL, "the launch spec command is invalid")
    output = spec.get("output")
    if command is not None and (
        not isinstance(output, dict)
        or output.get("mode") not in ("file", "stdout-json")
        or not isinstance(output.get("fields", []), list)
    ):
        raise Failure(EXIT_INTERNAL, "the launch spec output contract is invalid")
    for key in ("credentials", "protected"):
        if not isinstance(spec.get(key, []), list):
            raise Failure(EXIT_INTERNAL, f"the launch spec {key} list is invalid")
    return spec


# -- the job directory ----------------------------------------------------------


def _safe_relative(value: object) -> list[str]:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise Failure(EXIT_INTERNAL, "a credential target is not a relative path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise Failure(EXIT_INTERNAL, "a credential target is not a safe path")
    return parts


def _chown(path: str, owner) -> None:
    if owner is not None:
        os.chown(path, owner[0], owner[1], follow_symlinks=False)


def _write_new(path: str, data: bytes, mode: int, owner) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_CLOEXEC | _O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fchmod(descriptor, mode)
        if owner is not None:
            os.fchown(descriptor, owner[0], owner[1])
    finally:
        os.close(descriptor)


def _copy_credentials(spec: dict, home: str, owner) -> None:
    """Copy the Agent's login files from the root store into the private HOME."""
    for entry in spec.get("credentials", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
            raise Failure(EXIT_INTERNAL, "a credential entry is invalid")
        parts = _safe_relative(entry.get("target"))
        try:
            descriptor = os.open(
                entry["source"], os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW
            )
        except OSError as error:
            raise Failure(
                EXIT_INTERNAL, f"cannot read the Agent's login file: {error.strerror}"
            ) from None
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise Failure(EXIT_INTERNAL, "the Agent's login file is not regular")
            content = stream.read(MAX_CREDENTIAL_BYTES + 1)
        if len(content) > MAX_CREDENTIAL_BYTES:
            raise Failure(EXIT_INTERNAL, "the Agent's login file is too large")
        directory = home
        for part in parts[:-1]:
            directory = os.path.join(directory, part)
            try:
                os.mkdir(directory, 0o700)
            except FileExistsError:
                pass
            else:
                _chown(directory, owner)
        _write_new(os.path.join(directory, parts[-1]), content, 0o600, owner)


def _real_directory(path: str, owner_uid: int) -> None:
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


def _refused_workdir(path: str, protected) -> str | None:
    if path == "/" or path.count("/") < 2:
        return "it must be a dedicated directory below a top-level directory"
    for item in protected:
        item = item.rstrip("/") or "/"
        if path == item or path.startswith(item + "/") or item.startswith(path + "/"):
            return f"it overlaps {item}"
    return None


class Workdir:
    """The Agent's working directory: the caller's (handed over), or a new one."""

    def __init__(self, path: str | None, job: str, owner, protected) -> None:
        self.descriptor: int | None = None
        self.original: os.stat_result | None = None
        self.owner = owner
        if path is None:
            self.path = os.path.join(job, "work")
            os.mkdir(self.path, 0o700)
            _chown(self.path, owner)
            self.descriptor = os.open(
                self.path, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW
            )
            return
        real = os.path.realpath(path)
        try:
            descriptor = os.open(
                real, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW
            )
        except OSError as error:
            raise Failure(
                EXIT_USAGE, f"the workdir {path} is not a directory: {error.strerror}"
            ) from None
        self.descriptor = descriptor
        try:
            actual = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            actual = real
        reason = _refused_workdir(actual, protected)
        if reason is not None:
            raise Failure(EXIT_USAGE, f"refusing workdir {actual}: {reason}")
        self.path = actual
        if owner is not None:
            self.original = os.fstat(descriptor)
            try:
                os.fchown(descriptor, owner[0], owner[1])
                os.fchmod(descriptor, 0o700)
            except OSError as error:
                self.original = None
                raise Failure(
                    EXIT_USAGE,
                    f"cannot hand the workdir {actual} to the Agent: {error.strerror}",
                ) from None

    def restore(self) -> None:
        if self.descriptor is None:
            return
        try:
            if self.original is not None:
                os.fchown(self.descriptor, self.original.st_uid, self.original.st_gid)
                os.fchmod(self.descriptor, stat.S_IMODE(self.original.st_mode))
        except OSError:
            pass
        finally:
            os.close(self.descriptor)
            self.descriptor = None


# -- processes ------------------------------------------------------------------


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _group_alive(pgid: int, proc: str = "/proc") -> bool:
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


def _end_group(process: subprocess.Popen, grace: float) -> None:
    """SIGTERM the Agent's process group, then SIGKILL what remains.

    A signal to rsi-agent itself during the grace ends it at once.
    """
    _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and _Stop.signum is None:
        if process.poll() is not None and not _group_alive(process.pid):
            break
        time.sleep(0.05)
    _signal_group(process.pid, signal.SIGKILL)
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


def _preexec(workdir_fd: int, owner, libc):
    def prepare() -> None:
        os.fchdir(workdir_fd)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if owner is not None:
            os.setgroups([])
            os.setgid(owner[1])
            os.setuid(owner[0])
            # Never exec the Agent with any privilege left.
            if (
                os.getresuid() != (owner[0],) * 3
                or os.getresgid() != (owner[1],) * 3
                or os.getgroups()
            ):
                raise OSError(errno.EPERM, "the Agent's user was not dropped")
            if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "cannot set no_new_privs")
        # After the uid change, which would clear it: die with rsi-agent.
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        os.umask(0o077)

    return prepare


def _claim_slot(private: str, slots: int, deadline: float) -> tuple[int, int]:
    """Lock one free uid slot for the whole call; wait while all are taken.

    Returns (slot, lock descriptor). The lock goes with rsi-agent's process,
    so a killed call frees its slot.
    """
    descriptors = []
    try:
        for slot in range(slots):
            descriptors.append(
                os.open(
                    os.path.join(private, f"slot-{slot}.lock"),
                    os.O_RDWR | os.O_CREAT | _O_CLOEXEC | _O_NOFOLLOW,
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
            if _Stop.signum is not None:
                raise Failure(EXIT_TERMINATED, "terminated before the Agent started")
            if time.monotonic() >= deadline:
                raise Failure(
                    EXIT_TIMEOUT,
                    f"all {slots} Agent users stayed busy with other calls",
                )
            time.sleep(SLOT_POLL_SECONDS)
    finally:
        for descriptor in descriptors:
            if descriptor >= 0:
                os.close(descriptor)


def _check_output_paths(options: argparse.Namespace) -> None:
    """The Agent owns its workdir; nothing rsi-agent writes may lie inside it."""
    if options.workdir is None:
        return
    workdir = os.path.realpath(options.workdir)
    for label, path in (("--output", options.output), ("--log", options.log)):
        if path is None:
            continue
        directory = os.path.realpath(os.path.dirname(os.path.abspath(path)))
        if directory == workdir or directory.startswith(workdir.rstrip("/") + "/"):
            raise Failure(EXIT_USAGE, f"{label} must lie outside the --workdir")


# -- output ---------------------------------------------------------------------


def _read_file_output(job_fd: int) -> bytes:
    flags = os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(FINAL_MESSAGE, flags, dir_fd=job_fd)
    except FileNotFoundError:
        raise Failure(EXIT_AGENT, "the Agent wrote no final message") from None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise Failure(
                EXIT_AGENT, "the Agent's final message is a symlink"
            ) from None
        raise Failure(
            EXIT_AGENT, f"cannot read the Agent's final message: {error.strerror}"
        ) from None
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Failure(EXIT_AGENT, "the Agent's final message is not a plain file")
        data = stream.read(MAX_OUTPUT_BYTES + 1)
    if len(data) > MAX_OUTPUT_BYTES:
        raise Failure(EXIT_AGENT, "the Agent's final message is too large")
    return data


def _stdout_json(stdout: bytes) -> dict | None:
    text = stdout.decode("utf-8", errors="replace").strip()
    candidates = [text] + [line for line in reversed(text.splitlines()) if line]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _json_output(stdout: bytes, contract: dict) -> bytes:
    result = _stdout_json(stdout)
    if result is None:
        raise Failure(EXIT_AGENT, "the Agent printed no JSON result")
    fields = [field for field in contract.get("fields", []) if isinstance(field, str)]
    error_field = contract.get("error_field")
    if isinstance(error_field, str) and result.get(error_field) is True:
        # The error text, not a structured answer that came with it.
        detail = next(
            (
                result[field]
                for field in fields
                if isinstance(result.get(field), str) and result[field]
            ),
            "",
        ) or str(result.get("subtype", ""))
        raise Failure(EXIT_AGENT, "the Agent reported an error: " + detail[:500])
    for field in fields:
        value = result.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            return value.encode("utf-8")
        return json.dumps(value, ensure_ascii=False).encode("utf-8")
    raise Failure(EXIT_AGENT, "the Agent's JSON result has no final message")


def _write_atomically(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".rsi-agent-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            mask = os.umask(0)
            os.umask(mask)
            os.fchmod(stream.fileno(), 0o644 & ~mask)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _tail(path: str, limit: int) -> str:
    try:
        with open(path, "rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - limit))
            data = stream.read(limit)
    except OSError:
        return ""
    return REDACT.text(data.decode("utf-8", errors="replace")).strip()


def _read_private(path: str) -> bytes:
    with open(path, "rb") as stream:
        return stream.read(MAX_OUTPUT_BYTES + 1)[:MAX_OUTPUT_BYTES]


# -- the call -------------------------------------------------------------------


class _Stop:
    signum: int | None = None


def _on_signal(signum, frame) -> None:
    del frame
    _Stop.signum = signum


def run(options: argparse.Namespace) -> int:
    state_root = os.environ.get(STATE_ROOT_ENV) or STATE_ROOT
    spec = _load_spec(state_root)
    if spec.get("command") is None:
        raise Failure(
            EXIT_UNAVAILABLE,
            str(
                spec.get("unsupported")
                or f"the run's Agent {spec.get('agent')!r} cannot run in a Judge"
            ),
        )
    if not options.timeout > 0:
        raise Failure(EXIT_USAGE, "--timeout must be a positive number of seconds")
    prompt = _read_bounded(options.prompt_file, MAX_PROMPT_BYTES, "prompt")
    if not prompt.strip():
        raise Failure(EXIT_USAGE, "the prompt is empty")
    schema_bytes = None
    if options.schema is not None:
        schema_bytes = _read_bounded(options.schema, MAX_SCHEMA_BYTES, "schema")
        try:
            schema = json.loads(schema_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Failure(EXIT_USAGE, "the schema is not valid JSON") from None
        if not isinstance(schema, dict):
            raise Failure(EXIT_USAGE, "the schema must be a JSON object")
    output_dir = os.path.dirname(os.path.abspath(options.output))
    if not os.path.isdir(output_dir):
        raise Failure(EXIT_USAGE, f"the output directory {output_dir} does not exist")
    if options.log is not None and not os.path.isdir(
        os.path.dirname(os.path.abspath(options.log))
    ):
        raise Failure(EXIT_USAGE, "the log directory does not exist")
    _check_output_paths(options)
    started = time.monotonic()

    isolated = os.geteuid() == 0
    protected = [str(item) for item in spec.get("protected", []) if item]
    agent_uids = range(spec["uid"], spec["uid"] + spec["slots"])
    if not isolated:
        _say("no isolation: the Judge is not root, so the Agent runs as its user")
    else:
        for item in protected:
            try:
                if os.lstat(item).st_uid in agent_uids:
                    raise Failure(EXIT_INTERNAL, f"{item} is owned by the Agent's user")
            except FileNotFoundError:
                continue

    private = os.path.join(state_root, "private")
    _real_directory(private, os.geteuid())
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise Failure(EXIT_INTERNAL, f"cannot load the C library: {error}") from None
    owner = None
    slot_fd = None
    job = None
    job_fd = None
    workdir = None
    process = None
    private_files: list[str] = []
    try:
        if isolated:
            # Concurrent calls never share a uid: one call's Agent cannot
            # reach another's final message, files or processes.
            slot, slot_fd = _claim_slot(
                private, spec["slots"], started + options.timeout
            )
            owner = (spec["uid"] + slot, spec["gid"] + slot)
            # A killed earlier call may have left processes as this uid.
            if not sweep_uid(owner[0]):
                raise Failure(EXIT_INTERNAL, "cannot end an earlier call's Agent")
        jobs = os.path.join(state_root, "jobs")
        try:
            os.mkdir(jobs, 0o711)
        except FileExistsError:
            pass
        else:
            os.chmod(jobs, 0o711)
        _real_directory(jobs, os.geteuid())
        job_id = secrets.token_hex(8)
        job = os.path.join(jobs, job_id)
        os.mkdir(job, 0o700)
        job_fd = os.open(job, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW)
        home = os.path.join(job, "home")
        os.mkdir(home, 0o700)
        _chown(home, owner)
        _copy_credentials(spec, home, owner)

        command = spec["command"]
        schema_option = spec.get("schema_option")
        option = ""
        if schema_bytes is not None and schema_option:
            schema_path = os.path.join(job, "schema.json")
            _write_new(schema_path, schema_bytes, 0o644, None)
            option = schema_option.replace("{schema_file}", shlex.quote(schema_path))
        elif schema_bytes is not None:
            prompt = (
                prompt.rstrip(b"\n")
                + b"\n\n"
                + SCHEMA_REQUIREMENT.encode()
                + schema_bytes.strip()
                + b"\n"
            )
        output_file = os.path.join(job, FINAL_MESSAGE)
        command = command.replace("{schema_option}", option)
        command = command.replace("{output_file}", shlex.quote(output_file))
        _chown(job, owner)
        workdir = Workdir(options.workdir, job, owner, protected + [state_root])

        stdin_path = os.path.join(private, job_id + ".stdin")
        stdout_path = os.path.join(private, job_id + ".stdout")
        stderr_path = os.path.join(private, job_id + ".stderr")
        private_files = [stdin_path, stdout_path, stderr_path]
        _write_new(stdin_path, prompt, 0o600, None)
        for path in (stdout_path, stderr_path):
            _write_new(path, b"", 0o600, None)
        environment = dict(spec["environment"])
        environment.update(
            HOME=home,
            USER="rsi-agent",
            LOGNAME="rsi-agent",
            PATH=os.environ.get("PATH") or DEFAULT_PATH,
            TERM="dumb",
        )
        timed_out = False
        # Python 3.8 has no parenthesized multi-item ``with``.
        streams = [
            open(stdin_path, "rb"),
            open(stdout_path, "wb"),
            open(stderr_path, "wb"),
        ]
        stdin, stdout, stderr = streams
        try:
            if _Stop.signum is not None:
                raise Failure(EXIT_TERMINATED, "terminated before the Agent started")
            try:
                process = subprocess.Popen(
                    ["/bin/bash", "-c", command],
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    close_fds=True,
                    pass_fds=(workdir.descriptor,),
                    start_new_session=True,
                    preexec_fn=_preexec(workdir.descriptor, owner, libc),  # noqa: PLW1509
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise Failure(
                    EXIT_INTERNAL, f"cannot start the Agent: {error}"
                ) from None
            # --timeout bounds the whole call, a wait for a free slot included.
            deadline = started + options.timeout
            while True:
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if _Stop.signum is not None:
                    _signal_group(process.pid, signal.SIGKILL)
                    process.wait()
                    raise Failure(EXIT_TERMINATED, "terminated; the Agent was stopped")
                if time.monotonic() >= deadline:
                    timed_out = True
                    _end_group(process, KILL_GRACE_SECONDS)
                    break
            # Also end what the Agent left behind in its group.
            _signal_group(process.pid, signal.SIGKILL)
        finally:
            for stream in streams:
                stream.close()
        stdout_data = _read_private(stdout_path)
        if options.log is not None:
            log = (
                b"--- stdout ---\n"
                + stdout_data
                + b"\n--- stderr ---\n"
                + _read_private(stderr_path)
            )
            _write_atomically(options.log, REDACT.data(log))
        if timed_out:
            raise Failure(
                EXIT_TIMEOUT,
                f"the Agent did not finish within {options.timeout:g} seconds",
            )
        code = process.returncode
        if code != 0:
            tail = _tail(stderr_path, STDERR_TAIL_BYTES)
            if code == 127:
                raise Failure(
                    EXIT_UNAVAILABLE,
                    f"the {spec.get('agent')} CLI is not installed in this Judge",
                    tail,
                )
            how = f"signal {-code}" if code < 0 else f"status {code}"
            reason = f"the Agent exited with {how}"
            contract = spec.get("output") or {}
            if contract.get("mode") == "stdout-json":
                # An Agent that prints its own error result says why.
                try:
                    _json_output(stdout_data, contract)
                except Failure as failure:
                    if "reported an error" in failure.message:
                        reason += "; " + failure.message
            raise Failure(EXIT_AGENT, reason, tail)
        contract = spec.get("output") or {}
        if contract.get("mode") == "stdout-json":
            answer = _json_output(stdout_data, contract)
        else:
            answer = _read_file_output(job_fd)
        if not answer.strip():
            raise Failure(EXIT_AGENT, "the Agent's final message is empty")
        _write_atomically(options.output, REDACT.data(answer))
        return EXIT_OK
    finally:
        if process is not None and process.returncode is None:
            _signal_group(process.pid, signal.SIGKILL)
            process.wait()
        if owner is not None:
            # The uid is this call's alone: end anything that escaped the
            # group (a double fork, a new session) before handing it back.
            sweep_uid(owner[0])
        if workdir is not None:
            workdir.restore()
        if job_fd is not None:
            os.close(job_fd)
        if job is not None:
            shutil.rmtree(job, ignore_errors=True)
        for path in private_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        if slot_fd is not None:
            os.close(slot_fd)


def main(argv=None) -> int:
    options = _parser().parse_args(argv)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _on_signal)
    try:
        return run(options)
    except Failure as failure:
        _say(failure.message)
        if failure.detail:
            sys.stderr.write(REDACT.text(failure.detail) + "\n")
        return failure.code
    except OSError as error:
        _say(f"internal error: {error}")
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
