"""rsi-agent: call a model once, from inside a Judge.

RSI Harness mounts this program read-only at /opt/rsi-agent, with the
launcher /usr/local/bin/rsi-agent, in every Judge whose task declares
``[metadata.rsi_harness.verifier] model_access = true`` or ``judge_model``,
and writes the run's launch spec into a private tmpfs. It is the only
supported way for a Judge procedure to call a model.

    rsi-agent exec --prompt-file FILE --output FILE [--schema FILE]
                   [--timeout SEC] [--workdir DIR] [--log FILE]
    rsi-agent llm  --prompt-file FILE --output FILE [--schema FILE]
                   [--timeout SEC] [--log FILE]
    rsi-agent isolation

``exec`` runs the run's own Agent: the same CLI, credentials, provider, model
and reasoning effort as Work, so any Agent the Harness supports works here
without task changes. ``llm`` is one plain call, prompt in and final text
out, to the model the task declares as ``judge_model``, with the credential
the run operator supplied: an OpenAI API key (the Responses API, directly) or
a Codex login (through ``codex app-server`` with every tool off). Both are
isolated the same way, described below.

The Agent reads the prompt, may read files (evidence placed in ``--workdir``
or anywhere world-readable), and its final message is written to
``--output``. With ``--schema`` the Agent is asked for JSON conforming to the
schema, through its own structured-output option when it has one; callers
validate the result either way.

Each call's Agent is isolated, chosen automatically. When the Judge runs as
root, it runs as its own unprivileged user, one of a small range of uids that
no two concurrent calls share, with a fresh private HOME holding copies of
the Agent's login files. When the Judge is not root, it keeps the Judge's uid
and is confined by Landlock (Linux 5.13 or later): it may write only its own
job directory (HOME, TMPDIR, the final message) and ``--workdir``, and may
not read the launch spec, the credential store, the Judge's outputs, hidden
tests or another call's job. Without root or Landlock the call fails (exit
3), unless the Harness operator opted in to a same-user fallback. Either way
it cannot write the Judge's outputs or read the launch spec, a timeout ends
its whole process group, and whatever it left running is killed when the
call ends. The Agent owns ``--workdir`` for the call: it may delete, replace
or add entries there, so the output and the log must lie outside it. For
``llm`` the isolated child is rsi-agent's own worker, which receives the
credential on stdin and leaves only the answer for the caller.
``rsi-agent isolation`` prints the mode a call would use as JSON.

Exit status: 0 the output was written; 1 the Agent or the model call failed;
2 usage error; 3 model or LLM access is unavailable, the Agent cannot run
one-shot, or the call cannot be isolated; 124 timeout; 125 internal setup
failure; 143 rsi-agent was terminated.

Standard library only (and judge_isolation.py beside it), Python 3.8 or
later, no RSI Harness imports.
"""

from __future__ import annotations

import argparse
import base64
import errno
import json
import math
import os
import secrets
import shlex
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
    # Run as a script (/opt/rsi-agent/rsi_agent.py): -I adds no directory.
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import judge_isolation as isolation

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
# Version 2 adds "hidden" and "isolation_fallback"; version 1 still loads.
# The optional operator policy "allow_missing_landlock_signal" defaults off.
SPEC_VERSIONS = (1, 2)
DEFAULT_TIMEOUT = 1800.0
MAX_SLOTS = 1024
MAX_SPEC_BYTES = 8 * 1024 * 1024
MAX_PROMPT_BYTES = 4 * 1024 * 1024
MAX_SCHEMA_BYTES = 1024 * 1024
MAX_CREDENTIAL_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
STDERR_TAIL_BYTES = 2048
FINAL_MESSAGE = "final-message"
SCHEMA_REQUIREMENT = (
    "Your final message must be only a JSON value that conforms to this JSON Schema:\n"
)
REDACTED = "[REDACTED]"
NO_MODEL_ACCESS = (
    "this Judge has no model access (the task must declare "
    "[metadata.rsi_harness.verifier] model_access = true)"
)
NO_LLM_ACCESS = (
    "this Judge has no LLM access (the task must declare "
    "[metadata.rsi_harness.verifier] judge_model)"
)
NO_RSI_AGENT = (
    "this Judge has no rsi-agent (the task must declare "
    "[metadata.rsi_harness.verifier] model_access = true or judge_model)"
)
LLM_API = "api"
LLM_CODEX = "codex-login"
# The internal command the unprivileged side of ``llm`` runs.
LLM_WORKER = "_llm-worker"
WORKER_PREFIX = "rsi-agent-llm:"
# A short neutral system prompt, never a coding agent's.
LLM_INSTRUCTIONS = "You are a helpful assistant. Answer the user's request."
# ``codex app-server`` with nothing but the model: with the thread's empty
# ``environments`` (no shell, apply_patch or view_image) and the one-model
# catalog below (direct tool mode, no catalog tools) these leave a turn with
# no tools at all, no skills, apps, plugins or project docs, and no
# analytics. Checked against the pinned Codex release.
CODEX_LLM_CONFIG = (
    'model_provider="openai"',
    'web_search="disabled"',
    'cli_auth_credentials_store="file"',
    "features.apps=false",
    "features.plugins=false",
    "features.tool_suggest=false",
    "features.goals=false",
    "features.multi_agent=false",
    "features.image_generation=false",
    "features.memories=false",
    "features.hooks=false",
    "features.shell_tool=false",
    "features.view_image=false",
    "features.shell_snapshot=false",
    "tools.experimental_request_user_input.enabled=false",
    "skills.include_instructions=false",
    "skills.bundled.enabled=false",
    "include_permissions_instructions=false",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "project_doc_max_bytes=0",
    "analytics.enabled=false",
    "check_for_update_on_startup=false",
    # Catalog-driven tools: off here, and the catalog never asks for them.
    "features.sleep_tool=false",
    "features.code_mode=false",
    "features.code_mode_only=false",
)
# Codex's system config layers, which -c cannot undo (MCP servers, another
# model provider): a Judge that has one cannot make a tool-less call.
CODEX_SYSTEM_CONFIG = (
    "/etc/codex/config.toml",
    "/etc/codex/requirements.toml",
    "/etc/codex/managed_config.toml",
)
CODEX_CATALOG = "rsi-agent-models.json"
# codex app-server refuses longer turn input (characters).
CODEX_MAX_INPUT_CHARS = 1 << 20
# Codex refreshes a ChatGPT access token this close to its expiry (Claude Code
# an OAuth one, too).
CODEX_REFRESH_WINDOW_SECONDS = 300
# Every effort the Judge model may get; "ultra" is a Codex alias, sent as
# "max" the way Codex resolves it.
LLM_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
# Shared with rsi-isolate (judge_isolation.py); the old names stay.
KILL_GRACE_SECONDS = isolation.KILL_GRACE_SECONDS
DEFAULT_PATH = isolation.DEFAULT_PATH
PR_SET_NO_NEW_PRIVS = isolation.PR_SET_NO_NEW_PRIVS
FS_READ = isolation.FS_READ
_O_CLOEXEC = isolation.O_CLOEXEC
_O_NOFOLLOW = isolation.O_NOFOLLOW
_O_DIRECTORY = isolation.O_DIRECTORY
Failure = isolation.Failure
Ruleset = isolation.Ruleset
readable_paths = isolation.readable_paths
landlock_abi = isolation.landlock_abi
_load_libc = isolation.load_libc


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
        description="Call a model once from inside a Judge.",
    )
    commands = parser.add_subparsers(dest="command", metavar="{exec,llm,isolation}")
    commands.required = True
    run = commands.add_parser(
        "exec",
        help="run the Agent on one prompt and write its final message",
        description="Run the run's Agent on one prompt; write its final message.",
    )
    llm = commands.add_parser(
        "llm",
        help="call the task's Judge model once and write its answer",
        description=(
            "One plain call to the model the task declares as judge_model; "
            "write its answer."
        ),
    )
    for command in (run, llm):
        command.add_argument("--prompt-file", required=True, help="the prompt (UTF-8)")
        command.add_argument(
            "--output", required=True, help="where to write the answer"
        )
        command.add_argument(
            "--schema", help="a JSON Schema the answer must conform to"
        )
        command.add_argument(
            "--timeout",
            type=float,
            default=DEFAULT_TIMEOUT,
            help=f"seconds before the call is stopped (default {DEFAULT_TIMEOUT:g})",
        )
    run.add_argument(
        "--workdir",
        help=(
            "the Agent's working directory; the Agent owns it during the call "
            "and may change its entries"
        ),
    )
    run.add_argument("--log", help="also write the Agent's redacted stdout/stderr")
    llm.add_argument("--log", help="also write the call's redacted diagnostics")
    commands.add_parser(
        "isolation",
        help="print how each call is isolated, as JSON",
        description=(
            "Print {mode, landlock_abi, fallback}; exit 3 when a call "
            "cannot be isolated."
        ),
    )
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


def _load_spec(state_root: str, absent: str = NO_MODEL_ACCESS) -> dict:
    path = os.path.join(state_root, "private", "launch.json")
    try:
        descriptor = os.open(path, os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW)
    except FileNotFoundError:
        raise Failure(EXIT_UNAVAILABLE, absent) from None
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
    if not isinstance(spec, dict) or spec.get("version") not in SPEC_VERSIONS:
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
    for key in ("credentials", "protected", "hidden", "writable"):
        value = spec.get(key, [])
        if not isinstance(value, list) or (
            key != "credentials" and not all(isinstance(item, str) for item in value)
        ):
            raise Failure(EXIT_INTERNAL, f"the launch spec {key} list is invalid")
    if spec.get("isolation_fallback", "none") not in ("none", "same-user"):
        raise Failure(EXIT_INTERNAL, "the launch spec isolation fallback is invalid")
    if type(spec.get("allow_missing_landlock_signal", False)) is not bool:
        raise Failure(EXIT_INTERNAL, "the launch spec signal compatibility is invalid")
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


def _claim_slot(private: str, slots: int, deadline: float) -> tuple[int, int]:
    """Lock one free Agent uid slot for the whole call (see
    judge_isolation.claim_slot)."""
    return isolation.claim_slot(
        private,
        slots,
        deadline,
        f"all {slots} Agent users stayed busy with other calls",
        "the Agent",
    )


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


# -- isolation ------------------------------------------------------------------


def isolation_mode(spec: dict, libc) -> dict:
    """``uid`` (a root Judge), ``landlock``, or the operator's ``same_user``.

    Raises exit 3 when the Judge is neither root nor able to use Landlock
    and the launch spec carries no operator fallback.
    """
    problem = isolation.uid_mode_problem(spec["uid"], spec["slots"], spec["gid"])
    abi, reason = isolation.landlock_abi(libc)
    return isolation.choose_mode(
        problem,
        abi,
        reason,
        spec.get("isolation_fallback", "none"),
        EXIT_UNAVAILABLE,
        "the Agent",
    )


def call_ruleset(
    libc, abi: int, spec: dict, state_root: str, writable
) -> isolation.Ruleset:
    """One call: read all but the hidden paths, write only its own directories."""
    ruleset = isolation.Ruleset(
        libc,
        abi,
        allow_missing_landlock_signal=spec.get("allow_missing_landlock_signal", False),
    )
    try:
        # The Harness lists rsi-agent's state, each login file's place, the
        # Judge's outputs and hidden tests; /tmp (staged evidence) and the
        # trajectory stay readable.
        hidden = [str(item) for item in spec.get("hidden", ["/logs/verifier"])]
        hidden += [state_root] + [
            str(entry.get("source"))
            for entry in spec.get("credentials", [])
            if isinstance(entry, dict)
        ]
        for path in isolation.readable_paths(hidden):
            ruleset.allow(path, isolation.FS_READ)
        isolation.allow_devices(ruleset)
        for path in list(writable) + [str(item) for item in spec.get("writable", [])]:
            ruleset.allow(path, ruleset.handled)
    except BaseException:
        ruleset.close()
        raise
    return ruleset


# -- the call -------------------------------------------------------------------


def _read_inputs(options: argparse.Namespace):
    """The prompt bytes, and the schema's bytes and value (or None, None)."""
    if not options.timeout > 0:
        raise Failure(EXIT_USAGE, "--timeout must be a positive number of seconds")
    prompt = _read_bounded(options.prompt_file, MAX_PROMPT_BYTES, "prompt")
    if not prompt.strip():
        raise Failure(EXIT_USAGE, "the prompt is empty")
    schema_bytes = None
    schema = None
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
    return prompt, schema_bytes, schema


def run(options: argparse.Namespace) -> int:
    """``rsi-agent exec``: the run's Agent on one prompt."""
    state_root = os.environ.get(STATE_ROOT_ENV) or STATE_ROOT
    spec = _load_spec(state_root, NO_MODEL_ACCESS)
    if spec.get("command") is None:
        raise Failure(
            EXIT_UNAVAILABLE,
            str(
                spec.get("unsupported")
                or f"the run's Agent {spec.get('agent')!r} cannot run in a Judge"
            ),
        )
    prompt, schema_bytes, _ = _read_inputs(options)
    # The login copies carry no usable refresh token (the Harness withheld it,
    # so a Judge never rotates Work's): the access token must outlive the call.
    expiry = spec.get("credentials_expire_at")
    if isinstance(expiry, (int, float)) and not isinstance(expiry, bool):
        _check_expiry(
            float(expiry), _call_lifetime(spec, options.timeout), "the Agent's login"
        )

    def build(job: str, home: str, owner) -> tuple:
        _copy_credentials(spec, home, owner)
        command = spec["command"]
        schema_option = spec.get("schema_option")
        option = ""
        stdin = prompt
        if schema_bytes is not None and schema_option:
            schema_path = os.path.join(job, "schema.json")
            _write_new(schema_path, schema_bytes, 0o644, None)
            option = schema_option.replace("{schema_file}", shlex.quote(schema_path))
        elif schema_bytes is not None:
            stdin = (
                prompt.rstrip(b"\n")
                + b"\n\n"
                + SCHEMA_REQUIREMENT.encode()
                + schema_bytes.strip()
                + b"\n"
            )
        output_file = os.path.join(job, FINAL_MESSAGE)
        command = command.replace("{schema_option}", option)
        command = command.replace("{output_file}", shlex.quote(output_file))
        return ["/bin/bash", "-c", command], dict(spec["environment"]), stdin

    def collect(code: int, stdout_data: bytes, stderr_path: str, job_fd: int) -> bytes:
        contract = spec.get("output") or {}
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
            if contract.get("mode") == "stdout-json":
                # An Agent that prints its own error result says why.
                try:
                    _json_output(stdout_data, contract)
                except Failure as failure:
                    if "reported an error" in failure.message:
                        reason += "; " + failure.message
            raise Failure(EXIT_AGENT, reason, tail)
        if contract.get("mode") == "stdout-json":
            answer = _json_output(stdout_data, contract)
        else:
            answer = _read_file_output(job_fd)
        if not answer.strip():
            raise Failure(EXIT_AGENT, "the Agent's final message is empty")
        return answer

    return _isolated_call(
        spec,
        state_root,
        options,
        build=build,
        collect=collect,
        workdir=options.workdir,
        what="the Agent",
    )


def _isolated_call(
    spec: dict, state_root: str, options, *, build, collect, workdir, what
):
    """Run one isolated child (see the module docstring) and write its answer.

    ``build(job, home, owner)`` returns the child's argv, environment and
    stdin; ``collect(code, stdout, stderr_path, job_fd)`` its answer.
    """
    started = time.monotonic()
    libc = isolation.load_libc()
    mode = isolation_mode(spec, libc)
    isolated = mode["mode"] == "uid"
    confined = mode["mode"] == "landlock"
    protected = [str(item) for item in spec.get("protected", []) if item]
    agent_uids = range(spec["uid"], spec["uid"] + spec["slots"])
    if mode["mode"] == "same_user":
        _say(
            "no filesystem isolation: the operator's same-user fallback runs "
            f"{what} as the Judge's user"
        )
    elif isolated:
        for item in protected:
            try:
                if os.lstat(item).st_uid in agent_uids:
                    raise Failure(EXIT_INTERNAL, f"{item} is owned by the Agent's user")
            except FileNotFoundError:
                continue

    private = os.path.join(state_root, "private")
    isolation.real_directory(private, os.geteuid())
    # Whatever the child starts stays a descendant, even after a double fork.
    libc.prctl(isolation.PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    owner = None
    slot_fd = None
    job = None
    job_fd = None
    work = None
    process = None
    ruleset = None
    private_files: list[str] = []
    try:
        # At most ``slots`` calls at once. With per-call uids, concurrent
        # calls never share one: one call's child cannot reach another's
        # final message, files or processes.
        slot, slot_fd = _claim_slot(private, spec["slots"], started + options.timeout)
        if isolated:
            owner = (spec["uid"] + slot, spec["gid"] + slot)
            # A killed earlier call may have left processes as this uid.
            if not isolation.sweep_uid(owner[0]):
                raise Failure(EXIT_INTERNAL, "cannot end an earlier call's Agent")
        jobs = os.path.join(state_root, "jobs")
        try:
            os.mkdir(jobs, 0o711)
        except FileExistsError:
            pass
        else:
            os.chmod(jobs, 0o711)
        isolation.real_directory(jobs, os.geteuid())
        job_id = secrets.token_hex(8)
        job = os.path.join(jobs, job_id)
        os.mkdir(job, 0o700)
        job_fd = os.open(job, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW)
        home = os.path.join(job, "home")
        os.mkdir(home, 0o700)
        _chown(home, owner)
        job_tmp = os.path.join(job, "tmp")
        os.mkdir(job_tmp, 0o700)
        _chown(job_tmp, owner)
        argv, child_environment, stdin_data = build(job, home, owner)
        _chown(job, owner)
        work = Workdir(workdir, job, owner, protected + [state_root])

        stdin_path = os.path.join(private, job_id + ".stdin")
        stdout_path = os.path.join(private, job_id + ".stdout")
        stderr_path = os.path.join(private, job_id + ".stderr")
        private_files = [stdin_path, stdout_path, stderr_path]
        _write_new(stdin_path, stdin_data, 0o600, None)
        for path in (stdout_path, stderr_path):
            _write_new(path, b"", 0o600, None)
        environment = dict(child_environment)
        environment.update(
            HOME=home,
            TMPDIR=job_tmp,
            USER="rsi-agent",
            LOGNAME="rsi-agent",
            PATH=os.environ.get("PATH") or DEFAULT_PATH,
            TERM="dumb",
        )
        if confined:
            writable = [job] + ([work.path] if workdir is not None else [])
            ruleset = call_ruleset(
                libc, mode["landlock_abi"], spec, state_root, writable
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
            if isolation.Stop.signum is not None:
                raise Failure(EXIT_TERMINATED, f"terminated before {what} started")
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    close_fds=True,
                    pass_fds=(work.descriptor,),
                    start_new_session=True,
                    preexec_fn=isolation.preexec(  # noqa: PLW1509
                        work.descriptor, owner, libc, ruleset, umask=0o077
                    ),
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise Failure(EXIT_INTERNAL, f"cannot start {what}: {error}") from None
            finally:
                if ruleset is not None:
                    ruleset.close()
            # --timeout bounds the whole call, a wait for a free slot included.
            deadline = started + options.timeout
            while True:
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if isolation.Stop.signum is not None:
                    isolation.signal_group(process.pid, signal.SIGKILL)
                    process.wait()
                    raise Failure(EXIT_TERMINATED, f"terminated; {what} was stopped")
                if time.monotonic() >= deadline:
                    timed_out = True
                    isolation.end_group(process, isolation.KILL_GRACE_SECONDS)
                    break
            # Also end what the child left behind in its group.
            isolation.signal_group(process.pid, signal.SIGKILL)
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
                f"{what} did not finish within {options.timeout:g} seconds",
            )
        answer = collect(process.returncode, stdout_data, stderr_path, job_fd)
        _write_atomically(options.output, REDACT.data(answer))
        return EXIT_OK
    finally:
        if process is not None and process.returncode is None:
            isolation.signal_group(process.pid, signal.SIGKILL)
            process.wait()
        if owner is not None:
            # The uid is this call's alone: end anything that escaped the
            # group (a double fork, a new session) before handing it back.
            isolation.sweep_uid(owner[0])
        # In every mode: end what the child left running, as its subreaper.
        isolation.end_descendants()
        if ruleset is not None:
            ruleset.close()
        if work is not None:
            work.restore()
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


# -- llm: one plain model call ------------------------------------------------


def _llm_spec(spec: dict) -> dict:
    llm = spec.get("llm")
    if llm is None:
        raise Failure(EXIT_UNAVAILABLE, NO_LLM_ACCESS)
    if (
        not isinstance(llm, dict)
        or llm.get("kind") not in (LLM_API, LLM_CODEX)
        or not isinstance(llm.get("model"), str)
        or not isinstance(llm.get("effort"), str)
    ):
        raise Failure(EXIT_INTERNAL, "the launch spec llm section is invalid")
    if llm["kind"] == LLM_API and not (
        isinstance(llm.get("url"), str) and isinstance(llm.get("api_key"), str)
    ):
        raise Failure(EXIT_INTERNAL, "the launch spec llm API settings are invalid")
    if llm["kind"] == LLM_CODEX and not (
        isinstance(llm.get("codex"), str) and isinstance(llm.get("auth_json"), str)
    ):
        raise Failure(EXIT_INTERNAL, "the launch spec llm Codex settings are invalid")
    for key in ("config", "environment"):
        value = llm.get(key, [] if key == "config" else {})
        valid = (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
            if key == "config"
            else isinstance(value, dict)
            and all(
                isinstance(name, str) and isinstance(item, str)
                for name, item in value.items()
            )
        )
        if not valid:
            raise Failure(EXIT_INTERNAL, f"the launch spec llm {key} is invalid")
    return llm


def run_llm(options: argparse.Namespace) -> int:
    """``rsi-agent llm``: one plain call to the task's Judge model."""
    state_root = os.environ.get(STATE_ROOT_ENV) or STATE_ROOT
    spec = _load_spec(state_root, NO_LLM_ACCESS)
    llm = _llm_spec(spec)
    options.workdir = None
    prompt, _, schema = _read_inputs(options)
    try:
        text = prompt.decode("utf-8")
    except UnicodeDecodeError:
        raise Failure(EXIT_USAGE, "the prompt is not UTF-8 text") from None
    effort = "max" if llm["effort"] == "ultra" else llm["effort"]
    config = list(llm.get("config", []))
    if llm["kind"] == LLM_CODEX:
        if len(text) > CODEX_MAX_INPUT_CHARS:
            raise Failure(
                EXIT_USAGE,
                f"the prompt exceeds {CODEX_MAX_INPUT_CHARS} characters, the "
                "Codex CLI's limit",
            )
        _check_access_token(llm["auth_json"], _call_lifetime(spec, options.timeout))

    def build(job: str, home: str, owner) -> tuple:
        environment = dict(llm.get("environment", {}))
        if llm["kind"] == LLM_CODEX:
            # A private copy of the login, without a usable refresh token
            # (the Harness withheld it): Codex cannot rotate the operator's.
            codex_home = os.path.join(home, ".codex")
            os.mkdir(codex_home, 0o700)
            _chown(codex_home, owner)
            _write_new(
                os.path.join(codex_home, "auth.json"),
                llm["auth_json"].encode("utf-8"),
                0o600,
                owner,
            )
            # The model's only catalog entry: direct tools, none of its own.
            catalog = os.path.join(codex_home, CODEX_CATALOG)
            _write_new(
                catalog,
                json.dumps(codex_catalog(llm["model"], effort)).encode("utf-8"),
                0o600,
                owner,
            )
            config.append("model_catalog_json=" + json.dumps(catalog))
            environment["CODEX_HOME"] = codex_home
        request = {
            "kind": llm["kind"],
            "model": llm["model"],
            "effort": effort,
            "instructions": LLM_INSTRUCTIONS,
            "prompt": text,
            "schema": schema,
            "output": os.path.join(job, FINAL_MESSAGE),
            "timeout": options.timeout,
            "url": llm.get("url"),
            "api_key": llm.get("api_key"),
            "proxy": llm.get("proxy"),
            "codex": llm.get("codex"),
            "config": config,
        }
        argv = [
            sys.executable or "python3",
            "-I",
            "-S",
            "-B",
            os.path.abspath(__file__),
            LLM_WORKER,
        ]
        return argv, environment, json.dumps(request).encode("utf-8")

    def collect(code: int, stdout_data: bytes, stderr_path: str, job_fd: int) -> bytes:
        del stdout_data
        if code != 0:
            raise worker_failure(code, _tail(stderr_path, STDERR_TAIL_BYTES))
        answer = _read_file_output(job_fd)
        if not answer.strip():
            raise Failure(EXIT_AGENT, "the model's answer is empty")
        return answer

    return _isolated_call(
        spec,
        state_root,
        options,
        build=build,
        collect=collect,
        workdir=None,
        what="the model call",
    )


def codex_catalog(model: str, effort: str) -> dict:
    """A one-model Codex catalog (``model_catalog_json``) for the Judge model.

    It replaces Codex's bundled and remote catalogs, whose entries may ask for
    code mode, a clock, a sleep tool or user messages: direct tool mode, no
    catalog tools, no shell, no apply_patch, no tool search. The other fields
    are Codex's own fallback for a model it does not know.
    """
    return {
        "models": [
            {
                "slug": model,
                "display_name": model,
                "description": None,
                # Required in a catalog; the thread's baseInstructions win.
                "base_instructions": LLM_INSTRUCTIONS,
                "default_reasoning_level": effort,
                "supported_reasoning_levels": [
                    {"effort": item, "description": item} for item in LLM_EFFORTS
                ],
                "shell_type": "disabled",
                "visibility": "none",
                "supported_in_api": True,
                "priority": 99,
                "availability_nux": None,
                "upgrade": None,
                "include_apps_usage_instructions": False,
                "support_verbosity": False,
                "default_verbosity": None,
                "apply_patch_tool_type": None,
                "truncation_policy": {"mode": "bytes", "limit": 10000},
                "context_window": 272000,
                "max_context_window": 272000,
                "effective_context_window_percent": 95,
                "experimental_supported_tools": [],
                "supports_search_tool": False,
                "tool_mode": "direct",
            }
        ]
    }


def access_token_expiry(auth_json: str):
    """The ``exp`` of a Codex login's access token, or None when unknown."""
    try:
        token = json.loads(auth_json)["tokens"]["access_token"]
        payload = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        expiry = claims["exp"]
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)):
        return None
    return float(expiry)


def _check_access_token(auth_json: str, timeout: float) -> None:
    """Refuse a call Codex would have to refresh the login for.

    The login carries no usable refresh token, so a call that outlives the
    access token, or starts inside Codex's refresh window, can only fail.
    """
    expiry = access_token_expiry(auth_json)
    if expiry is not None:
        _check_expiry(expiry, timeout, "the Judge LLM's Codex login")


def _call_lifetime(spec: dict, timeout: float) -> float:
    """How long a call can last: ``--timeout``, or less when the round ends first.

    ``round_ends_at`` (epoch seconds) is when the Harness stops this Judge;
    no call outlives it, so a login need not outlive a longer --timeout.
    """
    ends = spec.get("round_ends_at")
    if (
        isinstance(ends, (int, float))
        and not isinstance(ends, bool)
        and math.isfinite(ends)
    ):
        return max(0.0, min(timeout, float(ends) - time.time()))
    return timeout


def _check_expiry(expiry: float, timeout: float, what: str) -> None:
    """Exit 3 when a login's access token would expire during the call."""
    left = expiry - time.time()
    if left <= CODEX_REFRESH_WINDOW_SECONDS + timeout:
        raise Failure(
            EXIT_UNAVAILABLE,
            what
            + " expires "
            + ("already" if left <= 0 else f"in {int(left)} seconds")
            + "; rsi-agent never refreshes it (a Work Agent's login is renewed "
            "before each round): use a fresh login or a shorter --timeout",
        )


def worker_failure(code: int, tail: str) -> Failure:
    """rsi-agent llm's failure for the worker's non-zero exit ``code``."""
    reason = _worker_reason(tail)
    if code == EXIT_UNAVAILABLE:
        return Failure(EXIT_UNAVAILABLE, reason or "the model is unavailable", tail)
    if code in (EXIT_USAGE, EXIT_INTERNAL):
        # The worker's own setup, not the model: keep the exit contract.
        return Failure(code, reason or "the model call's worker failed", tail)
    how = f"signal {-code}" if code < 0 else f"status {code}"
    return Failure(EXIT_AGENT, reason or f"the model call exited with {how}", tail)


def _worker_reason(tail: str) -> str:
    for line in reversed(tail.splitlines()):
        if line.startswith(WORKER_PREFIX):
            return line[len(WORKER_PREFIX) :].strip()
    return ""


class _WorkerFailure(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _api_error_text(body: bytes) -> str:
    try:
        value = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return body.decode("utf-8", errors="replace").strip()[:500]
    error = value.get("error") if isinstance(value, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:500]
    if isinstance(error, str):
        return error[:500]
    return json.dumps(value)[:500]


def _response_text(response: dict, deltas: list) -> str:
    """The final text of a Responses API ``response`` object."""
    status = response.get("status")
    if status == "failed":
        error = response.get("error") or {}
        raise _WorkerFailure(
            EXIT_AGENT,
            "the model call failed: "
            + str(error.get("message") if isinstance(error, dict) else error),
        )
    if status == "incomplete":
        details = response.get("incomplete_details") or {}
        reason = details.get("reason") if isinstance(details, dict) else details
        raise _WorkerFailure(EXIT_AGENT, f"the model's answer is incomplete: {reason}")
    parts = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise _WorkerFailure(
                    EXIT_AGENT, "the model refused: " + str(content.get("refusal"))
                )
            if content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    text = "".join(parts)
    if not text and deltas:
        text = "".join(deltas)
    return text


def _api_call(request: dict) -> str:
    """POST {base}/v1/responses with the operator's key; the final text."""
    import urllib.error
    import urllib.request

    body: dict = {
        "model": request["model"],
        "instructions": request["instructions"],
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": request["prompt"]}],
            }
        ],
        "reasoning": {"effort": request["effort"]},
        "store": False,
        "stream": True,
    }
    if request.get("schema") is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "rsi_agent_output",
                "schema": request["schema"],
                "strict": True,
            }
        }

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            del req, fp, msg, headers
            raise _WorkerFailure(
                EXIT_AGENT, f"the model API redirected (HTTP {code}); refused"
            )

    proxy = request.get("proxy")
    # An explicit (possibly empty) mapping: never the environment's proxies.
    # No redirects: the key would follow them to another host.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {}),
        NoRedirect(),
    )
    http_request = urllib.request.Request(
        request["url"],
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + request["api_key"],
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        response = opener.open(http_request, timeout=max(1.0, request["timeout"]))
    except urllib.error.HTTPError as error:
        raise _WorkerFailure(
            EXIT_AGENT,
            f"the model API returned HTTP {error.code}: "
            + _api_error_text(error.read(65536)),
        ) from None
    except (urllib.error.URLError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise _WorkerFailure(
            EXIT_AGENT, f"cannot reach the model API: {reason}"
        ) from None
    with response:
        if "text/event-stream" not in (response.headers.get("Content-Type") or ""):
            try:
                value = json.loads(response.read(MAX_OUTPUT_BYTES + 1))
            except ValueError:
                raise _WorkerFailure(
                    EXIT_AGENT, "the model API answered with invalid JSON"
                ) from None
            if not isinstance(value, dict):
                raise _WorkerFailure(
                    EXIT_AGENT, "the model API answer is not an object"
                )
            return _response_text(value, [])
        return _sse_text(response)


def _sse_text(stream) -> str:
    deltas: list = []
    data: list = []
    received = 0
    while True:
        raw = stream.readline()
        if not raw:
            break
        received += len(raw)
        if received > 4 * MAX_OUTPUT_BYTES:
            raise _WorkerFailure(EXIT_AGENT, "the model's answer is too large")
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            continue
        if line or not data:
            continue
        payload = "\n".join(data)
        data = []
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "response.output_text.delta":
            deltas.append(str(event.get("delta") or ""))
        elif kind in ("response.completed", "response.failed", "response.incomplete"):
            response = event.get("response")
            if not isinstance(response, dict):
                raise _WorkerFailure(EXIT_AGENT, f"the model API sent a bad {kind}")
            return _response_text(response, deltas)
        elif kind == "error":
            message = event.get("message") or event.get("error") or event
            raise _WorkerFailure(EXIT_AGENT, f"the model call failed: {message}")
    raise _WorkerFailure(EXIT_AGENT, "the model API stream ended before the answer")


class _AppServer:
    """A JSON-RPC client of ``codex app-server`` over stdio (JSON lines)."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self.next_id = 0
        self.notifications: list = []

    def send(self, message: dict) -> None:
        try:
            self.process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            self.process.stdin.flush()
        except OSError:
            raise self.ended() from None

    def ended(self) -> _WorkerFailure:
        code = self.process.poll()
        how = "is still running" if code is None else f"exited with status {code}"
        return _WorkerFailure(EXIT_AGENT, f"codex app-server {how} before the answer")

    def receive(self) -> dict:
        while True:
            line = self.process.stdout.readline()
            if not line:
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                raise self.ended()
            try:
                message = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if "method" in message and "id" in message:
                # A server request (approval, user input, ...): nothing here
                # may act or ask, so refuse it.
                self.send(
                    {
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": "rsi-agent does not handle "
                            + str(message.get("method")),
                        },
                    }
                )
                continue
            return message

    def request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        identifier = self.next_id
        self.send({"id": identifier, "method": method, "params": params})
        while True:
            message = self.receive()
            if "method" in message:
                self.notifications.append(message)
                continue
            if message.get("id") != identifier:
                continue
            if "error" in message:
                error = message.get("error") or {}
                text = error.get("message") if isinstance(error, dict) else error
                raise _WorkerFailure(
                    EXIT_AGENT, f"codex app-server refused {method}: {text}"
                )
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def notification(self) -> dict:
        if self.notifications:
            return self.notifications.pop(0)
        while True:
            message = self.receive()
            if "method" in message:
                return message


def _codex_call(request: dict, workdir: str) -> str:
    """Through the Codex CLI: an ephemeral app-server thread with no tools."""
    codex = request["codex"]
    if not (os.path.isfile(codex) and os.access(codex, os.X_OK)):
        raise _WorkerFailure(
            EXIT_UNAVAILABLE, f"the Codex CLI is not available in this Judge ({codex})"
        )
    present = [path for path in CODEX_SYSTEM_CONFIG if os.path.lexists(path)]
    if present:
        raise _WorkerFailure(
            EXIT_UNAVAILABLE,
            "the Judge image has Codex system config ("
            + ", ".join(present)
            + "), which could add tools or another provider; refused",
        )
    argv = [codex]
    for item in list(CODEX_LLM_CONFIG) + list(request.get("config") or []):
        argv += ["-c", item]
    argv.append("app-server")
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=workdir,
            env=dict(os.environ, NO_COLOR="1"),
            close_fds=True,
        )
    except OSError as error:
        raise _WorkerFailure(
            EXIT_UNAVAILABLE, f"cannot start the Codex CLI: {error.strerror}"
        ) from None
    server = _AppServer(process)
    try:
        server.request(
            "initialize",
            {
                "clientInfo": {"name": "rsi_agent", "title": None, "version": "1"},
                # Experimental: thread/start's environments.
                "capabilities": {"experimentalApi": True},
            },
        )
        server.send({"method": "initialized"})
        thread = server.request(
            "thread/start",
            {
                "model": request["model"],
                "cwd": workdir,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "baseInstructions": request["instructions"],
                "ephemeral": True,
                # No environment: no shell, apply_patch or view_image.
                "environments": [],
            },
        )
        thread_id = (thread.get("thread") or {}).get("id")
        if not isinstance(thread_id, str):
            raise _WorkerFailure(EXIT_AGENT, "codex app-server started no thread")
        turn_params: dict = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": request["prompt"], "text_elements": []}],
            "effort": request["effort"],
        }
        if request.get("schema") is not None:
            turn_params["outputSchema"] = request["schema"]
        turn = server.request("turn/start", turn_params)
        turn_id = (turn.get("turn") or {}).get("id")
        messages: list = []
        last_error = ""
        while True:
            message = server.notification()
            method = message.get("method")
            params = message.get("params") or {}
            if not isinstance(params, dict):
                continue
            if params.get("turnId") not in (None, turn_id):
                continue
            if method == "item/completed":
                item = params.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    messages.append(item)
            elif method == "error" and not params.get("willRetry"):
                error = params.get("error") or {}
                last_error = str(
                    error.get("message") if isinstance(error, dict) else error
                )
            elif method == "turn/completed":
                finished = params.get("turn") or {}
                if turn_id is not None and finished.get("id") != turn_id:
                    continue
                status = finished.get("status")
                if status != "completed":
                    error = finished.get("error") or {}
                    reason = (
                        (error.get("message") if isinstance(error, dict) else None)
                        or last_error
                        or str(status)
                    )
                    raise _WorkerFailure(EXIT_AGENT, f"the model call failed: {reason}")
                final = [
                    item for item in messages if item.get("phase") == "final_answer"
                ]
                chosen = (final or messages or [{}])[-1]
                return str(chosen.get("text") or "")
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def llm_worker() -> int:
    """The unprivileged side of ``rsi-agent llm``: the call itself.

    Reads its request (with the credential) on stdin, writes the answer to
    the job's final-message file, and states a failure on its last stderr
    line. Not a public command.
    """
    try:
        request = json.loads(sys.stdin.buffer.read(MAX_SPEC_BYTES).decode("utf-8"))
        if request["kind"] == LLM_API:
            text = _api_call(request)
        else:
            text = _codex_call(request, os.getcwd())
        if not text.strip():
            raise _WorkerFailure(EXIT_AGENT, "the model's answer is empty")
        data = text.encode("utf-8")
        if len(data) > MAX_OUTPUT_BYTES:
            raise _WorkerFailure(EXIT_AGENT, "the model's answer is too large")
        _write_new(request["output"], data, 0o600, None)
        return EXIT_OK
    except _WorkerFailure as failure:
        sys.stderr.write(
            WORKER_PREFIX + " " + failure.message.replace("\n", " ") + "\n"
        )
        return failure.code
    except (KeyError, TypeError, ValueError, OSError) as error:
        sys.stderr.write(f"{WORKER_PREFIX} internal error: {error}\n")
        return EXIT_INTERNAL


def report_isolation() -> int:
    """``rsi-agent isolation``: print the mode each call would use."""
    state_root = os.environ.get(STATE_ROOT_ENV) or STATE_ROOT
    spec = _load_spec(state_root, NO_RSI_AGENT)
    libc = isolation.load_libc()
    mode = isolation.probe_ruleset(
        libc,
        isolation_mode(spec, libc),
        allow_missing_landlock_signal=spec.get("allow_missing_landlock_signal", False),
    )
    sys.stdout.write(json.dumps(mode, sort_keys=True) + "\n")
    sys.stdout.flush()
    return EXIT_OK


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments == [LLM_WORKER]:
        return llm_worker()
    options = _parser().parse_args(arguments)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, isolation.on_signal)
    try:
        if options.command == "isolation":
            return report_isolation()
        if options.command == "llm":
            return run_llm(options)
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
