"""Run-owned sandbox sessions, finite accounting and whole-family barriers."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from rsi_harness.errors import InfrastructureError, RetryableSubmissionError
from rsi_harness.integrations import sandbox_client as wire
from rsi_harness.runtime.sandbox_contracts import (
    SandboxChildStopped,
    SandboxDownloadError,
    SandboxEnvGrant,
    SandboxError,
    SandboxLease,
    SandboxOwner,
    absolute_path,
    below,
)
from rsi_harness.runtime.sandbox_transfer import encode_bundle


@dataclass(frozen=True)
class SandboxSessionCredentials:
    owner: SandboxOwner
    credential: str = field(repr=False)


@dataclass
class _Session:
    credentials: SandboxSessionCredentials
    grant: object
    deadline: float | None
    frozen: bool = False
    revoked: bool = False
    usage: dict = field(default_factory=dict)
    requests: dict = field(default_factory=dict)
    # Environment grant (v2) and its per-session usage; None for profiles.
    env_grant: object = None
    env_usage: dict = field(default_factory=dict)
    # request_id -> (fingerprint digest, result JSON, error) of ended objects.
    tombstones: dict = field(default_factory=dict)


@dataclass
class _Child:
    lease: SandboxLease
    session: _Session
    profile: object
    deadline: float
    inflight: str | None = "create"
    expired: bool = False
    lifecycle_lock: object = field(default_factory=threading.RLock)
    last_operation: dict | None = None


@dataclass
class _Request:
    fingerprint: tuple
    result: object = None
    error: tuple[str, str, str] | None = None
    complete: bool = False
    # The env, job or stage a v2 record names; it is dropped with that object.
    handle: str | None = None


_ENV_HANDLE = re.compile(r"e[0-9a-f]{32}")


def _cache_error(error):
    # Exception tracebacks retain request bodies and worker frames. Keep only a
    # bounded public diagnostic; every replay raises a fresh exception.
    if isinstance(error, SandboxError):
        return (error.code, error.field[:4096], error.message[:4096])
    return ("infrastructure", "request", "sandbox operation requires recovery")


def _seconds(value, field_name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise SandboxError("invalid", field_name, "expected finite positive seconds")
    return float(value)


class SandboxBroker:
    """``grant`` is a profile grant (v1) or an environment grant (v2).

    An environment grant needs ``envs`` (``sandbox_envs.EnvRuntime``), the
    Docker-facing parts every v2 operation is delegated to.
    """

    def __init__(self, grant, backend, journal, clock=time.monotonic, *, envs=None):
        self.grant, self.backend, self.journal, self.clock = (
            grant,
            backend,
            journal,
            clock,
        )
        self._lock = threading.RLock()
        self._sessions = {}
        self._children = {}
        self._usage = {}
        self._cancelled = False
        self._cancel_requested = threading.Event()
        self._run_deadline = None
        self.recovery_required = False
        self._stop = threading.Event()
        self._watchdog = None
        self._owner = None
        self.envs = None
        if envs is not None and isinstance(grant, SandboxEnvGrant):
            from rsi_harness.runtime.sandbox_envs import SandboxEnvs

            self.envs = SandboxEnvs(self, envs)

    def _profile_grant(self, phase):
        return (
            None
            if isinstance(self.grant, SandboxEnvGrant)
            else getattr(self.grant, phase)
        )

    def _env_grant(self, phase):
        if not isinstance(self.grant, SandboxEnvGrant):
            return None
        return getattr(self.grant.environments, phase)

    def grants_phase(self, phase):
        """Whether the run grants ``phase`` any sandbox capability."""
        return (
            self._profile_grant(phase) is not None or self._env_grant(phase) is not None
        )

    def supports(self, phase, version):
        """Whether ``phase`` may use protocol ``version`` operations at all."""
        if version == 2:
            return self._env_grant(phase) is not None
        return self._profile_grant(phase) is not None

    @property
    def host_policy(self):
        """The operator's environment host table, or None for profile grants."""
        if isinstance(self.grant, SandboxEnvGrant):
            return self.grant.environments.host
        return None

    def start(self):
        """Deadline enforcement is independent of HTTP/request worker slots."""
        with self._lock:
            if self._watchdog is not None:
                return

            def watch():
                while not self._stop.wait(0.1):
                    try:
                        self.sweep_expired()
                    except Exception:
                        self._fail_closed()

            self._watchdog = threading.Thread(
                target=watch, name="rsi-sandbox-watchdog", daemon=True
            )
            self._watchdog.start()

    def open_session(self, owner, deadline=None):
        owner = SandboxOwner.model_validate(owner.model_dump())
        if deadline is not None:
            deadline = _seconds(deadline, "deadline")
        with self._lock:
            identity = (owner.run_id, owner.task_id)
            if self._owner is not None and self._owner != identity:
                raise InfrastructureError("sandbox session owner differs from run")
            phase = self._profile_grant(owner.phase)
            env_grant = self._env_grant(owner.phase)
            if phase is None and env_grant is None:
                raise SandboxError("permission", "phase", "phase has no sandbox grant")
            if (
                self._cancelled
                or self._cancel_requested.is_set()
                or self.recovery_required
            ):
                raise SandboxError("expired", "session", "run cannot open sessions")
            # The run deadline bounds Work only; Judge follows its own deadline.
            if owner.phase == "work" and self._run_deadline is not None:
                if self.clock() >= self._run_deadline:
                    raise SandboxError("expired", "session", "run deadline expired")
                if deadline is not None:
                    deadline = min(deadline, self._run_deadline)
            previous = self._sessions.get(owner.phase)
            if previous is not None and not previous.revoked:
                raise InfrastructureError("sandbox phase already has a session")
            if previous is not None and (
                any(
                    c.session is previous and c.lease.state != "removed"
                    for c in self._children.values()
                )
                or (self.envs is not None and self.envs.retained(previous))
            ):
                raise InfrastructureError("sandbox previous phase requires recovery")
            self._owner = identity
            credentials = SandboxSessionCredentials(owner, secrets.token_urlsafe(32))
            self._sessions[owner.phase] = _Session(
                credentials, phase, deadline, env_grant=env_grant
            )
            return credentials

    def open_judge(self, owner, deadline=None):
        if owner.phase != "judge":
            raise InfrastructureError("sandbox Judge owner required")
        with self._lock:
            work = self._sessions.get("work")
            if work is not None and not work.frozen:
                raise InfrastructureError("sandbox Work must freeze before Judge")
            return self.open_session(owner, deadline)

    def _activate(self, phase, deadline):
        deadline = _seconds(deadline, "deadline")
        with self._lock:
            session = self._sessions.get(phase)
            if session is None:
                return
            if session.deadline is not None:
                raise InfrastructureError("sandbox deadline cannot be reset")
            if session.revoked or self._cancelled or self._cancel_requested.is_set():
                raise InfrastructureError("sandbox revoked session cannot activate")
            session.deadline = deadline

    def activate_work(self, deadline):
        deadline = _seconds(deadline, "deadline")
        with self._lock:
            if (
                self._cancelled
                or self._cancel_requested.is_set()
                or self.recovery_required
            ):
                raise InfrastructureError("sandbox revoked session cannot activate")
            if self._run_deadline is not None:
                raise InfrastructureError("sandbox deadline cannot be reset")
            session = self._sessions.get("work")
            if session is not None:
                if session.deadline is not None:
                    raise InfrastructureError("sandbox deadline cannot be reset")
                if session.revoked or self._cancelled:
                    raise InfrastructureError("sandbox revoked session cannot activate")
            self._run_deadline = deadline
            if session is not None:
                session.deadline = deadline

    def activate_judge(self, deadline):
        # An accepted round keeps its verifier deadline even past Work's end.
        self._activate("judge", deadline)

    def _authenticate(self, credential, *, mutation=False):
        if (
            type(credential) is not str
            or len(credential) > 128
            or not credential.isascii()
        ):
            raise SandboxError("permission", "credential", "invalid or revoked session")
        session = next(
            (
                s
                for s in self._sessions.values()
                if hmac.compare_digest(s.credentials.credential, credential)
            ),
            None,
        )
        if (
            session is None
            or session.revoked
            or self._cancelled
            or self._cancel_requested.is_set()
        ):
            raise SandboxError("permission", "credential", "invalid or revoked session")
        if (
            session.credentials.owner.phase == "work"
            and self._run_deadline is not None
            and self.clock() >= self._run_deadline
        ):
            raise SandboxError("expired", "session", "run deadline expired")
        if session.deadline is not None and self.clock() >= session.deadline:
            raise SandboxError("expired", "session", "owner deadline expired")
        if mutation:
            if self.recovery_required:
                raise SandboxError(
                    "infrastructure", "session", "sandbox recovery required"
                )
            if session.deadline is None or session.frozen:
                raise SandboxError("busy", "session", "session inactive or frozen")
            if self.clock() >= session.deadline:
                raise SandboxError("expired", "session", "owner deadline expired")
        return session

    def authenticate(self, credential, owner):
        """Socket binding additionally restricts a valid token to its exact phase."""
        with self._lock:
            session = self._authenticate(credential)
            if session.credentials.owner != owner:
                raise SandboxError("permission", "owner", "wrong phase endpoint")

    def capabilities(self, credential):
        """``versions`` lists every version this broker speaks (spec 3.3);
        which one a phase may use is ``grant`` (v1) or ``environments`` (v2)
        being non-null, and the server refuses the other version's ops."""
        with self._lock:
            session = self._authenticate(credential)
            profiles = session.grant
            return {
                "version": 1,
                "versions": [1, 2],
                "owner": session.credentials.owner.model_dump(mode="json"),
                "grant": None if profiles is None else profiles.model_dump(mode="json"),
                "profiles": []
                if profiles is None
                else [
                    p.model_dump(mode="json")
                    for p in self.grant.profiles
                    if p.name in profiles.profiles
                ],
                "environments": None
                if self.envs is None
                else self.envs.capabilities(session),
            }

    def _owned(self, session, child_id):
        child = self._children.get(child_id) if type(child_id) is str else None
        if child is None or child.session is not session:
            raise SandboxError(
                "permission", "handle", "handle is not owned by this session"
            )
        return child

    def _dead(self, child):
        return (
            child.expired
            or self._cancelled
            or self._cancel_requested.is_set()
            or child.session.revoked
            or self.clock() >= child.deadline
            or (
                child.session.credentials.owner.phase == "work"
                and self._run_deadline is not None
                and self.clock() >= self._run_deadline
            )
            or (
                child.session.deadline is not None
                and self.clock() >= child.session.deadline
            )
        )

    def _effective_deadline(self, child):
        deadlines = [child.deadline]
        if child.session.deadline is not None:
            deadlines.append(child.session.deadline)
        if (
            child.session.credentials.owner.phase == "work"
            and self._run_deadline is not None
        ):
            deadlines.append(self._run_deadline)
        return min(deadlines)

    def _charge(self, session, **charges):
        for usage, limits in (
            (session.usage, session.grant.limits),
            (self._usage, self.grant.run_limits),
        ):
            for name, value in charges.items():
                if usage.get(name, 0) + value > getattr(limits, name):
                    raise SandboxError("quota", name, "phase or run budget exhausted")
        for usage in (session.usage, self._usage):
            for name, value in charges.items():
                usage[name] = usage.get(name, 0) + value

    def _remaining(self, session, name, ceiling):
        return min(
            ceiling,
            getattr(session.grant.limits, name) - session.usage.get(name, 0),
            getattr(self.grant.run_limits, name) - self._usage.get(name, 0),
        )

    def _request(self, session, request_id, fingerprint):
        if (
            type(request_id) is not str
            or not 1 <= len(request_id) <= 64
            or not request_id.isascii()
            or not all(c.isalnum() or c in "._-" for c in request_id)
        ):
            raise SandboxError(
                "invalid", "request_id", "expected 1..64 safe ASCII characters"
            )
        record = session.requests.get(request_id)
        if record is not None:
            if record.fingerprint != fingerprint:
                raise SandboxError(
                    "invalid", "request_id", "conflicting idempotency key reuse"
                )
            if not record.complete:
                raise SandboxError(
                    "busy", "request_id", "original request still pending"
                )
            if record.error is not None:
                raise SandboxError(*record.error)
        return record

    def _state(self, child, state, *, pending=False):
        try:
            self.journal.set_state(
                child.lease.child_id, state, pending_mutation=pending
            )
        except Exception as error:
            self._fail_closed()
            raise InfrastructureError(
                "sandbox journal unavailable; recovery required"
            ) from error
        child.lease = child.lease.model_copy(
            update={"state": state, "pending_mutation": pending}
        )

    def _fail_closed(self):
        with self._lock:
            self.recovery_required = True
            for session in self._sessions.values():
                session.frozen = True

    def create(self, credential, profile, lifetime_sec, request_id):
        lifetime = _seconds(lifetime_sec, "lifetime_sec")
        with self._lock:
            session = self._authenticate(credential, mutation=True)
            if session.grant is None or profile not in session.grant.profiles:
                raise SandboxError(
                    "permission", "profile", "profile not granted to phase"
                )
            selected = next(p for p in self.grant.profiles if p.name == profile)
            if lifetime > selected.max_lifetime_sec:
                raise SandboxError(
                    "quota", "lifetime_sec", "exceeds approved profile lifetime"
                )
            fingerprint = ("create", profile, lifetime)
            record = self._request(session, request_id, fingerprint)
            if record is not None:
                return record.result
            charges = dict(
                max_live=1,
                max_created=1,
                max_operations=1,
                max_cpus=selected.cpus,
                max_memory_mb=selected.memory_mb,
                max_lifetime_sec=lifetime,
            )
            self._charge(session, **charges)
            now = self.clock()
            child_id = uuid.uuid4().hex
            lease = SandboxLease(
                owner=session.credentials.owner,
                child_id=child_id,
                planned_name="rsi-sandbox-" + child_id,
                image_id=selected.image,
                cpus=selected.cpus,
                memory_mb=selected.memory_mb,
                reserved_lifetime_sec=lifetime,
                created_at=time.time(),
                expires_at=time.time() + lifetime,
                pending_mutation=True,
            )
            try:
                self.journal.plan(lease)
            except Exception:
                for usage in (session.usage, self._usage):
                    for key, value in charges.items():
                        usage[key] -= value
                raise
            child = _Child(lease, session, selected, now + lifetime)
            self._children[child_id] = child
            record = _Request(fingerprint)
            session.requests[request_id] = record
        try:
            identity = self.backend.create(child.lease, selected)
            with self._lock:
                self.journal.record_container(child_id, identity)
                child.lease = child.lease.model_copy(update={"container_id": identity})
                expired = self._dead(child)
            if not expired:
                self.backend.start(child.lease)
            with self._lock:
                expired = self._dead(child)
                self._state(child, "expired" if expired else "running")
                child.inflight = None
            if expired:
                child.expired = True
                self._remove(child)
                raise SandboxError(
                    "expired", "create", "owner or child expired during creation"
                )
            record.result = child.lease
            return record.result
        except SandboxError as error:
            record.error = _cache_error(error)
            raise
        except Exception as error:
            self._fail_closed()
            failure = InfrastructureError(
                "sandbox creation unresolved; recovery required"
            )
            record.error = _cache_error(failure)
            raise failure from error
        finally:
            with self._lock:
                record.complete = True
                # Unknown create retains authority after the request ends.
                child.inflight = None

    def status(self, credential, child_id):
        with self._lock:
            session = self._authenticate(credential)
            child = self._owned(session, child_id)
            return {
                "child_id": child_id,
                "state": child.lease.state,
                "expired": self._dead(child),
                "remaining_sec": max(0, self._effective_deadline(child) - self.clock()),
                "inflight": child.inflight,
                "last_operation": child.last_operation,
            }

    @staticmethod
    def _root(child, root):
        try:
            absolute_path(root)
        except (ValueError, TypeError) as error:
            raise SandboxError(
                "invalid", "root", "expected normalized scratch path"
            ) from error
        if not any(below(root, path) for path, _ in child.profile.tmpfs_mb):
            raise SandboxError("permission", "root", "outside approved scratch")

    def _begin(self, session, child, operation, timeout_sec, **charges):
        timeout = _seconds(timeout_sec, "timeout_sec")
        if self._dead(child) or child.lease.state in ("removed", "stopped", "expired"):
            raise SandboxError("expired", "handle", "child is terminal")
        if child.inflight is not None or child.lease.state != "running":
            raise SandboxError("busy", "handle", "child operation already in flight")
        self._charge(session, max_operations=1, **charges)
        self._state(child, child.lease.state, pending=True)
        child.inflight = operation
        return min(self.clock() + timeout, self._effective_deadline(child))

    def _finish(self, child, result=None):
        with self._lock:
            child.inflight = None
            if not self.recovery_required:
                self._state(child, child.lease.state)
            child.last_operation = {
                "exit_code": getattr(result, "exit_code", None),
                "duration_sec": getattr(result, "duration_sec", None),
            }
            dead = self._dead(child)
        if dead and not self.recovery_required:
            self._remove(child)

    def execute(self, credential, child_id, argv, cwd, env, timeout_sec):
        if type(child_id) is str and _ENV_HANDLE.fullmatch(child_id):
            # Spec B3: envs exec only through exec_start/exec_wait/exec_kill.
            raise SandboxError(
                "invalid", "child_id", "env handles exec through exec_start"
            )
        with self._lock:
            session = self._authenticate(credential, mutation=True)
            child = self._owned(session, child_id)
            self._root(child, cwd)
            if (
                not isinstance(argv, (tuple, list))
                or not argv
                or any(type(x) is not str or "\0" in x for x in argv)
            ):
                raise SandboxError("invalid", "argv", "expected nonempty string argv")
            if not isinstance(env, dict) or any(
                type(k) is not str
                or not k
                or "=" in k
                or "\0" in k
                or type(v) is not str
                or "\0" in v
                for k, v in env.items()
            ):
                raise SandboxError("invalid", "env", "expected string environment")
            if (
                sum(len(x.encode()) for x in argv)
                + sum(len(k.encode()) + len(v.encode()) for k, v in env.items())
                > wire.MAX_COMMAND_BYTES
            ):
                raise SandboxError("quota", "argv/env", "command exceeds 64 KiB")
            cap = self._remaining(session, "max_log_bytes", wire.MAX_OUTPUT_BYTES)
            if cap <= 0:
                raise SandboxError(
                    "quota", "max_log_bytes", "retained output budget exhausted"
                )
            deadline = self._begin(
                session, child, "execute", timeout_sec, max_log_bytes=cap
            )
        result = None
        try:
            result = self.backend.execute(child.lease, argv, cwd, env, deadline, cap)
            with self._lock:
                # Bound encoded text too: replacing invalid UTF-8 can expand bytes.
                stdout = result.stdout.encode()[:cap].decode("utf-8", "ignore")
                stderr = result.stderr.encode()[: cap - len(stdout.encode())].decode(
                    "utf-8", "ignore"
                )
                result = result.model_copy(
                    update={
                        "stdout": stdout,
                        "stderr": stderr,
                        "truncated": result.truncated
                        or stdout != result.stdout
                        or stderr != result.stderr,
                    }
                )
                used = len(stdout.encode()) + len(stderr.encode())
                for usage in (session.usage, self._usage):
                    usage["max_log_bytes"] -= cap - used
                if self._dead(child):
                    raise SandboxError(
                        "expired", "exec", "owner or child deadline expired"
                    )
                if result.timed_out or result.output_limited or result.oom_killed:
                    self._state(child, "stopped", pending=True)
            return result
        except SandboxError as error:
            if error.code == "unknown-outcome":
                self._fail_closed()
            raise
        except Exception as error:
            self._fail_closed()
            raise InfrastructureError(
                "sandbox exec unresolved; recovery required"
            ) from error
        finally:
            self._finish(child, result)

    def upload(self, credential, child_id, root, entries, request_id, timeout_sec):
        digest = hashlib.sha256(encode_bundle(entries)).digest()
        size = sum(len(entry.data) for entry in entries)
        with self._lock:
            session = self._authenticate(credential, mutation=True)
            child = self._owned(session, child_id)
            self._root(child, root)
            fingerprint = (
                "upload",
                child_id,
                root,
                digest,
                timeout_sec,
            )
            record = self._request(session, request_id, fingerprint)
            if record is not None:
                return record.result
            deadline = self._begin(
                session, child, "upload", timeout_sec, max_upload_bytes=size
            )
            record = _Request(fingerprint)
            session.requests[request_id] = record
        try:
            self.backend.upload(child.lease, root, entries, deadline, byte_limit=size)
            if self._dead(child):
                raise SandboxError("expired", "upload", "owner or child expired")
        except Exception as error:
            if isinstance(error, SandboxChildStopped):
                with self._lock:
                    self._state(child, "stopped", pending=True)
            record.error = _cache_error(error)
            if not isinstance(error, SandboxError) or error.code == "unknown-outcome":
                self._fail_closed()
            raise
        finally:
            record.complete = True
            self._finish(child)

    def download(self, credential, child_id, root, paths, timeout_sec):
        if (
            not isinstance(paths, (list, tuple))
            or not 1 <= len(paths) <= wire.MAX_ENTRIES
        ):
            raise SandboxError(
                "invalid", "paths", "expected bounded nonempty path list"
            )
        try:
            for path in paths:
                if path != ".":
                    wire.validate_name(path)
        except ValueError as error:
            raise SandboxError("invalid", "paths", str(error)) from error
        with self._lock:
            session = self._authenticate(credential, mutation=True)
            child = self._owned(session, child_id)
            self._root(child, root)
            cap = self._remaining(session, "max_download_bytes", wire.MAX_BUNDLE_BYTES)
            deadline = self._begin(
                session, child, "download", timeout_sec, max_download_bytes=cap
            )
        try:
            result = self.backend.download(
                child.lease, root, paths, deadline, byte_limit=cap
            )
            encode_bundle(result, byte_limit=cap)
            with self._lock:
                actual = sum(len(entry.data) for entry in result)
                for usage in (session.usage, self._usage):
                    usage["max_download_bytes"] -= cap - actual
                if self._dead(child):
                    raise SandboxError("expired", "download", "owner or child expired")
            return result
        except Exception as error:
            if isinstance(error, SandboxDownloadError):
                # Only host adapter evidence can settle a failed reservation.
                # Unknown/unfinished operations retain their full byte charge.
                with self._lock:
                    charged = min(cap, max(0, error.download_bytes))
                    for usage in (session.usage, self._usage):
                        usage["max_download_bytes"] -= cap - charged
                        if not error.operation_started:
                            usage["max_operations"] -= 1
            if isinstance(error, SandboxChildStopped):
                with self._lock:
                    self._state(child, "stopped", pending=True)
            if not isinstance(error, SandboxError) or error.code == "unknown-outcome":
                self._fail_closed()
            raise
        finally:
            self._finish(child)

    def _remove(self, child, *, maintenance=False):
        with child.lifecycle_lock:
            with self._lock:
                if child.lease.state == "removed":
                    return True
                if child.lease.container_id is None:
                    return False
                lease = child.lease
                busy = (
                    child.inflight is not None
                    and child.inflight != "destroy"
                    and not maintenance
                )
                unresolved = self.recovery_required and lease.pending_mutation
            try:
                self.backend.terminate(lease)
                if busy or unresolved:
                    return False
                self.backend.remove(lease)
                with self._lock:
                    self.journal.mark_removed(lease.child_id)
                    child.lease = lease.model_copy(
                        update={"state": "removed", "pending_mutation": False}
                    )
                    for usage in (child.session.usage, self._usage):
                        for key, value in (
                            ("max_live", 1),
                            ("max_cpus", lease.cpus),
                            ("max_memory_mb", lease.memory_mb),
                        ):
                            usage[key] -= value
                return True
            except Exception as error:
                self._fail_closed()
                raise InfrastructureError(
                    "sandbox cleanup unresolved; recovery required"
                ) from error

    def destroy(self, credential, child_id):
        with self._lock:
            session = self._authenticate(credential, mutation=True)
            child = self._owned(session, child_id)
            if child.lease.state == "removed":
                return
            if child.inflight is not None:
                raise SandboxError("busy", "handle", "child operation still pending")
            self._charge(session, max_operations=1)
            self._state(child, child.lease.state, pending=True)
            child.inflight = "destroy"
        try:
            self._remove(child)
        finally:
            with self._lock:
                child.inflight = None

    def freeze_work(self):
        with self._lock:
            session = self._sessions.get("work")
            if session is None:
                return
            if (
                self.recovery_required
                or self._cancelled
                or self._cancel_requested.is_set()
            ):
                raise InfrastructureError("sandbox family requires recovery")
            family = [
                c
                for c in self._children.values()
                if c.session is session and c.lease.state != "removed"
            ]
            if any(c.inflight for c in family):
                raise RetryableSubmissionError(
                    "sandbox operation busy; retry submission after it completes"
                )
            if self.envs is not None:
                # In-flight Work pulls, env starts and copies: retry submit.
                self.envs.freeze_check(session)
            session.frozen = True
            for child in family:
                child.inflight = "pause"
        try:
            if self.envs is not None:
                self.envs.freeze(session)
            for child in family:
                if self._dead(child) or child.lease.state == "stopped":
                    if not self._remove(child, maintenance=True):
                        raise InfrastructureError(
                            "sandbox expired cleanup requires recovery"
                        )
                    continue
                with child.lifecycle_lock:
                    with self._lock:
                        if child.lease.state == "removed":
                            continue
                        self._state(child, child.lease.state, pending=True)
                    self.backend.pause(child.lease)
                    with self._lock:
                        self._state(child, "paused")
        except Exception as error:
            self._fail_closed()
            for child in family:
                if child.lease.state in ("removed", "paused"):
                    continue
                try:
                    with child.lifecycle_lock:
                        self.backend.pause(child.lease)
                        with self._lock:
                            self._state(child, "paused")
                except Exception:
                    pass
            if self.envs is not None:
                self.envs.repause(session)
            raise InfrastructureError(
                "sandbox family freeze failed; recovery required"
            ) from error
        finally:
            with self._lock:
                for child in family:
                    child.inflight = None

    @property
    def can_resume(self):
        with self._lock:
            work = self._sessions.get("work")
            return not (
                self._cancelled
                or self._cancel_requested.is_set()
                or self.recovery_required
                or (
                    self._run_deadline is not None
                    and self.clock() >= self._run_deadline
                )
                or (
                    work is not None
                    and (
                        work.revoked
                        or work.deadline is None
                        or self.clock() >= work.deadline
                    )
                )
            )

    @contextmanager
    def resume_admission(self, child=None):
        """Final mutation gate, after journal writes and daemon inspection.

        Revocation and unpause are ordered by the same authority lock. Only the
        bounded daemon unpause RPC belongs inside this gate, never an exec or a
        transfer. A cancellation requested during an admitted RPC is detected on
        return and contained; it cannot admit a subsequent resume.
        """
        with self._lock:

            def check():
                family = (
                    [child]
                    if child is not None
                    else [
                        c
                        for c in self._children.values()
                        if c.session.credentials.owner.phase == "work"
                        and c.lease.state != "removed"
                    ]
                )
                if not self.can_resume or any(self._dead(c) for c in family):
                    raise InfrastructureError(
                        "sandbox resume denied; recovery required"
                    )

            check()
            yield
            check()

    def resume_work(self, *, reopen=True):
        with self._lock:
            session = self._sessions.get("work")
            if session is None:
                return
            if not self.can_resume:
                raise InfrastructureError(
                    "sandbox Work cannot resume; recovery required"
                )
            family = [
                c
                for c in self._children.values()
                if c.session is session and c.lease.state != "removed"
            ]
        try:
            for child in family:
                with child.lifecycle_lock:
                    with self._lock:
                        if self._dead(child) or not self.can_resume:
                            raise InfrastructureError(
                                "sandbox expired child requires recovery"
                            )
                        child.inflight = "resume"
                        self._state(child, child.lease.state, pending=True)
                    self.backend.resume(
                        child.lease, admission=lambda: self.resume_admission(child)
                    )
                    with self._lock:
                        self._state(child, "running")
                        child.inflight = None
            if self.envs is not None:
                # Paused envs resume through the same final admission gate;
                # exec timeouts then move by the frozen time.
                self.envs.thaw(session, self.resume_admission)
            with self.resume_admission():
                if reopen:
                    session.frozen = False
        except Exception as error:
            self._fail_closed()
            for child in family:
                try:
                    with child.lifecycle_lock:
                        self.backend.pause(child.lease)
                        with self._lock:
                            self._state(child, "paused")
                except Exception:
                    pass
            if self.envs is not None:
                self.envs.repause(session)
            raise InfrastructureError(
                "sandbox Work resume failed; recovery required"
            ) from error
        finally:
            with self._lock:
                for child in family:
                    child.inflight = None

    def _retired_paused(self, child):
        """A frozen Work child after Work's normal end waits for close().

        It executes nothing while paused and Work can no longer resume.
        Without a paused killer terminating a paused child is not proven
        safe, so sweeping it would fail the run closed and freeze an accepted
        Judge round with it. Cancellation, recovery and close() still sweep
        and drain it; with the paused killer (production) close() removes it.
        """
        session = child.session
        now = self.clock()
        return (
            session.credentials.owner.phase == "work"
            and session.frozen
            and child.lease.state == "paused"
            and not child.lease.pending_mutation
            and child.inflight is None
            and not (
                self._cancelled
                or self._cancel_requested.is_set()
                or self.recovery_required
            )
            and (
                session.revoked
                or (self._run_deadline is not None and now >= self._run_deadline)
                or (session.deadline is not None and now >= session.deadline)
            )
        )

    def sweep_expired(self):
        with self._lock:
            children = [
                c
                for c in self._children.values()
                if c.lease.state != "removed"
                and self._dead(c)
                and not self._retired_paused(c)
            ]
            for child in children:
                child.expired = True
        for child in children:
            try:
                self._remove(child)
            except InfrastructureError:
                # Retain a paused family rather than implicitly thawing it.
                pass
        if self.envs is not None:
            self.envs.sweep()

    def reopen_work(self):
        try:
            with self.resume_admission():
                session = self._sessions.get("work")
                if session is not None:
                    session.frozen = False
        except InfrastructureError as error:
            raise InfrastructureError(
                "sandbox cannot reopen Work; recovery required"
            ) from error

    def contain_work(self):
        with self._lock:
            session = self._sessions.get("work")
            if session is None:
                return
            session.frozen = True
            children = [
                c
                for c in self._children.values()
                if c.session is session and c.lease.state != "removed"
            ]
        errors = []
        for child in children:
            try:
                with child.lifecycle_lock:
                    if child.lease.container_id is None or child.inflight:
                        raise InfrastructureError(
                            "sandbox pending child cannot be frozen"
                        )
                    self.backend.pause(child.lease)
                    with self._lock:
                        self._state(child, "paused")
            except Exception as error:
                errors.append(error)
        if self.envs is not None:
            errors.extend(self.envs.contain(session))
        if errors:
            self._fail_closed()
            raise InfrastructureError(
                "sandbox family containment requires recovery"
            ) from errors[0]

    def cancel_run(self):
        """Request cancellation before waiting for an admitted unpause RPC."""
        self._cancel_requested.set()
        with self._lock:
            self._cancelled = True
            for session in self._sessions.values():
                session.revoked = True
        self.sweep_expired()

    def cancel_work(self):
        """Retire Work at its normal end while an accepted round finishes.

        Only the Work session is revoked. Nothing is removed here and the run
        is not cancelled, so a Judge session keeps its own deadline. Paused
        Work children stay paused until close() cancels the run and drains
        every child.
        """
        with self._lock:
            session = self._sessions.get("work")
            if session is not None:
                session.revoked = True

    def close_judge(self):
        """End the round before Work may resume: within 6 s (plus 0.1 s per
        live service container) nothing of the round executes (every env,
        exec and job is killed with proof), and within 60 s (plus 1 s per
        env) every object is proven removed; otherwise fail closed."""
        with self._lock:
            session = self._sessions.get("judge")
            if session is None:
                return
            session.revoked = True
            children = [c for c in self._children.values() if c.session is session]
        if self.envs is None:
            self._drain_children(children)
            return
        delete_end = self.envs.kill_session(session)
        self._drain_children(children)
        self.envs.remove_session(session, delete_end)

    def _drain_children(self, children):
        deadline = time.monotonic() + 6.0
        while True:
            pending = []
            for child in children:
                if not self._remove(child):
                    pending.append(child)
            if not pending:
                return
            if time.monotonic() >= deadline:
                self._fail_closed()
                raise InfrastructureError("sandbox pending mutation requires recovery")
            time.sleep(0.01)

    def close(self):
        self.cancel_run()
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=6.0)
            if self._watchdog.is_alive():
                self._fail_closed()
                raise InfrastructureError("sandbox watchdog shutdown requires recovery")
        with self._lock:
            children = list(self._children.values())
        if self.envs is None:
            self._drain_children(children)
            return
        try:
            self.envs.close()
        finally:
            self._drain_children(children)

    # -- environments (v2): every operation is delegated to sandbox_envs --------

    def _environments(self):
        if self.envs is None:
            raise SandboxError("permission", "phase", "phase has no environment grant")
        return self.envs

    def wait_condition(self, credential, operation, metadata):
        """Long-poll support for the server: None means answer now."""
        return self._environments().wait_condition(credential, operation, metadata)

    def env_create(self, credential, spec, request_id):
        return self._environments().env_create(credential, spec, request_id)

    def env_start(self, credential, env_id, wait_timeout_sec, request_id):
        return self._environments().env_start(
            credential, env_id, wait_timeout_sec, request_id
        )

    def env_status(self, credential, env_id, wait_sec=0):
        return self._environments().env_status(credential, env_id, wait_sec)

    def env_stop_service(self, credential, env_id, service, timeout_sec, request_id):
        return self._environments().env_stop_service(
            credential, env_id, service, timeout_sec, request_id
        )

    def env_destroy(self, credential, env_id):
        return self._environments().env_destroy(credential, env_id)

    def env_list(self, credential):
        return self._environments().env_list(credential)

    def exec_start(
        self,
        credential,
        env_id,
        service,
        argv,
        cwd,
        env,
        user,
        timeout_sec,
        merge_stderr,
        request_id,
    ):
        return self._environments().exec_start(
            credential,
            env_id,
            service,
            argv,
            cwd,
            env,
            user,
            timeout_sec,
            merge_stderr,
            request_id,
        )

    def exec_wait(
        self, credential, exec_id, stdout_offset, stderr_offset, wait_sec, max_bytes
    ):
        return self._environments().exec_wait(
            credential, exec_id, stdout_offset, stderr_offset, wait_sec, max_bytes
        )

    def exec_kill(self, credential, exec_id, signal, scope):
        return self._environments().exec_kill(credential, exec_id, signal, scope)

    def stage_put(
        self, credential, stage_id, offset, final, sha256, request_id, payload
    ):
        return self._environments().stage_put(
            credential, stage_id, offset, final, sha256, request_id, payload
        )

    def stage_get(self, credential, stage_id, offset, length):
        return self._environments().stage_get(credential, stage_id, offset, length)

    def copy_in(self, credential, env_id, service, dest_dir, stage_id, request_id):
        return self._environments().copy_in(
            credential, env_id, service, dest_dir, stage_id, request_id
        )

    def copy_out(self, credential, env_id, service, path, max_bytes, exclude):
        return self._environments().copy_out(
            credential, env_id, service, path, max_bytes, exclude
        )

    def path_stat(self, credential, env_id, service, path, follow):
        return self._environments().path_stat(credential, env_id, service, path, follow)

    def tool_install(self, credential, env_id, service, tool):
        return self._environments().tool_install(credential, env_id, service, tool)

    def image_pull(self, credential, ref, policy, request_id):
        return self._environments().image_pull(credential, ref, policy, request_id)

    def image_build(self, credential, **fields):
        return self._environments().image_build(credential, **fields)

    def job_wait(self, credential, job_id, log_offset, wait_sec):
        return self._environments().job_wait(credential, job_id, log_offset, wait_sec)

    def job_cancel(self, credential, job_id):
        return self._environments().job_cancel(credential, job_id)

    def image_list(self, credential):
        return self._environments().image_list(credential)

    def image_release(self, credential, image):
        return self._environments().image_release(credential, image)
