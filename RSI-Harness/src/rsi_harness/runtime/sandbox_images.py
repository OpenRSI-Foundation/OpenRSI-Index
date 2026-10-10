"""Image jobs (pulls and builds) and the image handles they bind (spec 3.3, 4).

A job runs on its own broker thread, never on a request slot: ``image_pull``
and ``image_build`` return a ``j…`` handle at once and ``job_wait`` reads
its bounded log and final state. A job binds at most one ``i…`` image handle
of its session; env services name images only through such a handle, and
the handle resolves only in that session (S2).

Pulled images are a host cache: their journal record is a ledger entry
only, never removed from the host (S9) and never holding the reservation.
A pull the sandbox brought to the host is also recorded in the host's pull
ledger (sandbox_ledger), which only an operator's ``prune-images`` acts on.
Built images are run-owned and removed by their session (sandbox_build).
Reference normalization is lifted from archive/vm-evaluation's
``native_images.py``.
"""

from __future__ import annotations

import logging
import math
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from docker.errors import APIError, DockerException, NotFound

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_env_contracts import SandboxImageLease
from rsi_harness.runtime.sandbox_env_docker import image_preflight

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
# Bounded broker memory per session: live image handles (at most
# EnvLimits.max_image_handles, sandbox_contracts) and ended jobs.
MAX_ENDED_JOBS = 64
MAX_JOB_LOG = MIB
JOB_LOG_READ = 256 * 1024
MAX_REF = 256
JOB_STATES = ("queued", "running", "succeeded", "failed", "canceled", "timed_out")
ENDED_JOB = ("succeeded", "failed", "canceled", "timed_out")
PULL_POLICIES = ("missing", "always")


# -- pull references (lifted from archive/vm-evaluation native_images.py) ----


def pull_reference(value: object) -> tuple[str, str]:
    """(domain/remote, tag or sha256 digest) with docker's name defaults.

    Reference v0.6.0 name and tag grammar; digests must be sha256; a bare
    64-hex ID is refused (it names no registry).
    """
    if (
        type(value) is not str
        or not 1 <= len(value) <= MAX_REF
        or re.fullmatch(r"[a-f0-9]{64}", value)
    ):
        raise SandboxError("invalid", "ref", "unsupported image reference")
    named, separator, digest = value.partition("@")
    if separator and not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise SandboxError("invalid", "ref", "image digests must be sha256")
    first, slash, rest = named.partition("/")
    if not slash:
        domain, remote = "docker.io", "library/" + named
    elif (
        first == "localhost"
        or first == "index.docker.io"
        or "." in first
        or ":" in first
        or first.lower() != first
    ):
        domain, remote = ("docker.io" if first == "index.docker.io" else first), rest
    else:
        domain, remote = "docker.io", named
    if domain == "docker.io" and "/" not in remote:
        remote = "library/" + remote
    remote, colon, tag = remote.partition(":")
    domain_component = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    host = domain_component + r"(?:\." + domain_component + ")*"
    component = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
    if (
        not re.fullmatch(r"(?:" + host + r"|\[[a-fA-F0-9:]+\])(?::[0-9]+)?", domain)
        or not re.fullmatch(component + r"(?:/" + component + r")*", remote)
        or (colon and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag))
        or len(domain + "/" + remote) > 255
    ):
        raise SandboxError("invalid", "ref", "unsupported image reference")
    return domain + "/" + remote, digest if separator else tag if colon else "latest"


def registry_of(repository: str) -> str:
    return repository.split("/", 1)[0]


def reference_text(repository: str, tag: str) -> str:
    return f"{repository}@{tag}" if tag.startswith("sha256:") else f"{repository}:{tag}"


def pulled_from(attrs: Mapping[str, Any], repository: str, tag: str) -> bool:
    """Whether a cached image was pulled from ``repository`` (S3, S9).

    Only a ``RepoDigests`` entry proves registry provenance; a host-built or
    retagged image has none for that repository, so a name alone never binds
    it. A digest reference also needs exactly that digest.
    """
    for entry in attrs.get("RepoDigests") or ():
        try:
            source, digest = pull_reference(entry)
        except SandboxError:
            continue
        if source == repository and (not tag.startswith("sha256:") or digest == tag):
            return True
    return False


class JobFailed(Exception):
    """A job's user-facing failure; ``kind`` is a job_wait error kind."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# Pulls raised this name before builds shared the job runner.
PullFailed = JobFailed


class DockerImagePuller:
    """Anonymous registry pulls through the host daemon; never a build.

    No ``X-Registry-Auth`` is sent: host registry credentials never serve a
    sandbox. ``cancel`` closes the progress stream, which cancels the pull
    in the daemon.
    """

    def __init__(self, api: Any) -> None:
        self._api = api

    def inspect(self, reference: str) -> dict[str, Any] | None:
        try:
            return self._api.inspect_image(reference)
        except NotFound:
            return None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot inspect sandbox image: {error}"
            ) from error

    def image_ids(self) -> list[str]:
        """Every image ID on the host, intermediate ones included."""
        return self._api.images(all=True, quiet=True)

    def pull(
        self,
        repository: str,
        tag: str,
        *,
        progress: Callable[[str], None],
        stream: Callable[[Any], None],
        budget: Callable[[int], None] | None = None,
    ) -> None:
        """Run one pull to its end; ``stream`` receives the closable response.

        ``budget`` sees the summed layer sizes the daemon announced so far and
        may raise ``JobFailed`` to stop the pull (closing the stream cancels
        it in the daemon) before an oversize image is fully downloaded.
        """
        try:
            response = self._api._post(
                self._api._url("/images/create"),
                params={"fromImage": repository, "tag": tag},
                stream=True,
                timeout=None,
            )
        except (DockerException, OSError) as error:
            raise JobFailed("infrastructure", "the daemon did not answer") from error
        stream(response)
        try:
            try:
                self._api._raise_for_status(response)
            except APIError as error:
                status = error.status_code or 0
                LOGGER.info("sandbox image pull refused (%s): %s", status, error)
                kind = "registry" if 400 <= status < 500 else "infrastructure"
                raise JobFailed(kind, "the registry refused the pull") from None
            layers: dict[str, int] = {}
            for item in self._api._stream_helper(response, decode=True):
                if not isinstance(item, dict):
                    continue
                if item.get("error") or item.get("errorDetail"):
                    detail = item.get("errorDetail") or {}
                    message = str(detail.get("message") or item.get("error"))
                    raise JobFailed("registry", message[:1024])
                total = (item.get("progressDetail") or {}).get("total")
                layer = item.get("id")
                if (
                    budget is not None
                    and isinstance(layer, str)
                    and type(total) is int
                    and total > layers.get(layer, 0)
                ):
                    layers[layer] = total
                    budget(sum(layers.values()))
                line = " ".join(
                    str(item[key])
                    for key in ("id", "status", "progress")
                    if isinstance(item.get(key), str) and item.get(key)
                )
                if line:
                    progress(line[:1024] + "\n")
        finally:
            response.close()


# -- records -------------------------------------------------------------------


@dataclass(eq=False)
class Image:
    """One live image handle of a session."""

    handle: str
    session: Any
    kind: str
    image_id: str
    ref: str
    bytes: int
    attrs: Mapping[str, Any]
    lease: SandboxImageLease
    # Built images: the per-session dedupe key (B9) of the build.
    fingerprint: str | None = None


@dataclass(eq=False)
class Job:
    """``detail`` is the pull's (repository, tag, policy) or the build's
    request; ``closers`` stop in-flight work when the job is cancelled."""

    job_id: str
    session: Any
    kind: str
    handle: str
    deadline: float
    detail: Any = None
    state: str = "queued"
    log: bytearray = field(default_factory=bytearray)
    log_truncated: bool = False
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    generation: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    timed_out: bool = False
    closers: list[Callable[[], None]] = field(default_factory=list)
    thread: threading.Thread | None = None
    ended_at: float | None = None


def image_view(attrs: Mapping[str, Any], handle: str) -> dict[str, Any]:
    config = attrs.get("Config") or {}
    check = config.get("Healthcheck") or {}
    return {
        "handle": handle,
        "image_id": attrs.get("Id"),
        "bytes": attrs.get("Size") if type(attrs.get("Size")) is int else 0,
        "os": attrs.get("Os"),
        "arch": attrs.get("Architecture"),
        "workdir": config.get("WorkingDir") or "",
        "user": config.get("User") or "",
        "entrypoint": config.get("Entrypoint"),
        "cmd": config.get("Cmd"),
        "env_count": len(config.get("Env") or []),
        "exposed_ports": sorted(config.get("ExposedPorts") or {}),
        "volumes": sorted(config.get("Volumes") or {}),
        "has_healthcheck": bool(check.get("Test") and check["Test"][0] != "NONE"),
    }


# -- runner --------------------------------------------------------------------


class JobRunner:
    """The broker's image jobs and image handles, for every session.

    It holds no authority of its own: ``envs`` (SandboxEnvs) authenticates,
    charges quotas and writes the journal, and every mutation of the job
    and image tables happens under the broker lock. ``builds`` runs build
    jobs and removes built images (sandbox_build.BuildService); without it
    a runtime offers pulls only. ``ledger`` (sandbox_ledger.PullLedger)
    records successful pulls; without it none are recorded.
    """

    def __init__(
        self, envs: Any, puller: Any, builds: Any = None, ledger: Any = None
    ) -> None:
        self._envs = envs
        self._lock = envs._lock
        self._clock = envs._clock
        self._puller = puller
        self.builds = builds
        self.ledger = ledger
        self.jobs: dict[str, Job] = {}
        self.images: dict[str, Image] = {}

    # -- registry --------------------------------------------------------------

    def owned_image(self, session: Any, handle: object, field_name: str) -> Image:
        image = self.images.get(handle) if type(handle) is str else None
        if image is None or image.session is not session:
            raise SandboxError(
                "permission", field_name, "image handle is not owned by this session"
            )
        return image

    def owned_job(self, session: Any, job_id: object) -> Job:
        job = self.jobs.get(job_id) if type(job_id) is str else None
        if job is None or job.session is not session:
            raise SandboxError(
                "permission", "job_id", "handle is not owned by this session"
            )
        return job

    def session_images(self, session: Any) -> list[Image]:
        return [image for image in self.images.values() if image.session is session]

    def pending(self, session: Any) -> list[Job]:
        return [
            job
            for job in self.jobs.values()
            if job.session is session and job.state not in ENDED_JOB
        ]

    def admit_handle(self, session: Any) -> None:
        """Under the broker lock: room for one more image handle."""
        live = len(self.session_images(session)) + len(self.pending(session))
        if live >= session.env_grant.max_image_handles:
            raise SandboxError("quota", "image", "too many live image handles")

    def find_built(self, session: Any, fingerprint: str) -> Image | None:
        for image in self.session_images(session):
            if image.kind == "built" and image.fingerprint == fingerprint:
                return image
        return None

    def bind(self, image: Image) -> None:
        self.images[image.handle] = image

    def listing(self, session: Any) -> list[dict[str, Any]]:
        return [
            {
                "handle": image.handle,
                "image_id": image.image_id,
                "kind": image.kind,
                "ref": image.ref,
                "bytes": image.bytes,
                "in_use": self._envs.image_in_use(image),
            }
            for image in self.session_images(session)
        ]

    def unbind_pulled(self, image: Image) -> None:
        """Under the broker lock: unbind a pulled handle (the image stays)."""
        self._envs._commit_image(image.lease.model_copy(update={"state": "removed"}))
        del self.images[image.handle]

    # -- jobs ------------------------------------------------------------------

    def submit(
        self,
        session: Any,
        kind: str,
        handle: str,
        deadline: float,
        detail: Any,
        body: Callable[[Job], None],
    ) -> Job:
        """Under the broker lock: register a job and start its thread."""
        job = Job(
            "j" + secrets.token_hex(16),
            session,
            kind,
            handle,
            deadline,
            detail=detail,
        )
        self.jobs[job.job_id] = job
        self._trim(session)
        job.thread = threading.Thread(
            target=body, args=(job,), name=f"rsi-sandbox-image-{kind}", daemon=True
        )
        self._touch(job)
        job.thread.start()
        return job

    def ended(
        self,
        session: Any,
        kind: str,
        handle: str,
        result: dict[str, Any],
    ) -> Job:
        """Under the broker lock: a job already succeeded (a deduplicated
        build); it runs nothing and holds no running slot."""
        job = Job(
            "j" + secrets.token_hex(16),
            session,
            kind,
            handle,
            self._clock(),
            state="succeeded",
            result=result,
            ended_at=self._clock(),
        )
        self.jobs[job.job_id] = job
        self._trim(session)
        self._touch(job)
        return job

    def _touch(self, job: Job) -> None:
        self._envs._touch(job)

    def running(self, job: Job) -> None:
        with self._lock:
            if job.state == "queued":
                job.state = "running"
                self._touch(job)

    def log(self, job: Job, text: str | bytes) -> None:
        data = text.encode("utf-8", "replace") if isinstance(text, str) else text
        with self._lock:
            room = MAX_JOB_LOG - len(job.log)
            keep = data[
                : max(0, min(room, self._envs._remaining(job.session, "log_bytes")))
            ]
            if len(keep) < len(data):
                job.log_truncated = True
            if keep:
                self._envs._charge(job.session, log_bytes=len(keep))
                job.log += keep
                self._touch(job)

    def hold(self, job: Job, closer: Callable[[], None]) -> None:
        """Register how to stop the job's in-flight work; runs it at once if
        the job was already cancelled."""
        with self._lock:
            job.closers.append(closer)
            stop = job.cancel.is_set()
        if stop:
            self.close(job)

    def release(self, job: Job, closer: Callable[[], None]) -> None:
        with self._lock:
            if closer in job.closers:
                job.closers.remove(closer)

    @staticmethod
    def close(job: Job) -> None:
        for closer in list(job.closers):
            try:
                closer()
            except Exception as error:
                LOGGER.debug("stopping sandbox %s job failed: %s", job.kind, error)

    def finish(
        self,
        job: Job,
        state: str,
        error: dict[str, str] | None,
        result: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            job.state, job.error, job.result = state, error, result
            job.closers.clear()
            job.ended_at = self._clock()
            self._envs._refund(job.session, jobs_running=1)
            self._touch(job)

    def outcome(self, job: Job, failure: BaseException) -> tuple[str, dict[str, str]]:
        """The ended state and error of a job that did not succeed."""
        if job.timed_out:
            return "timed_out", {"kind": "timeout", "message": "timed out"}
        if job.cancel.is_set() or (
            isinstance(failure, JobFailed) and failure.kind == "canceled"
        ):
            return "canceled", {
                "kind": "canceled",
                "message": f"the {job.kind} was canceled",
            }
        if isinstance(failure, JobFailed):
            return "failed", {"kind": failure.kind, "message": failure.message[:1024]}
        LOGGER.warning("sandbox image %s %s failed: %s", job.kind, job.job_id, failure)
        return "failed", {"kind": "infrastructure", "message": f"the {job.kind} failed"}

    def _trim(self, session: Any) -> None:
        ended = sorted(
            (
                job
                for job in self.jobs.values()
                if job.session is session and job.state in ENDED_JOB
            ),
            key=lambda job: job.ended_at or 0.0,
        )
        for job in ended[: max(0, len(ended) - MAX_ENDED_JOBS)]:
            del self.jobs[job.job_id]
            self._envs._forget_requests(session, job.job_id)

    def view(self, session: Any, job_id: object, log_offset: object) -> dict:
        """Under the broker lock: one bounded job_wait read."""
        job = self.owned_job(session, job_id)
        if type(log_offset) is not int or not 0 <= log_offset <= len(job.log):
            raise SandboxError("invalid", "log_offset", "offset beyond the log")
        chunk = bytes(job.log[log_offset : log_offset + JOB_LOG_READ])
        return {
            "job_id": job.job_id,
            "kind": job.kind,
            "state": job.state,
            "log": chunk.decode("utf-8", "replace"),
            "next_offset": log_offset + len(chunk),
            "log_truncated": job.log_truncated,
            "result": job.result,
            "error": job.error,
        }

    def cancel(self, job: Job) -> str:
        with self._lock:
            if job.state not in ENDED_JOB:
                job.cancel.set()
            state = job.state
        self.close(job)
        return state

    def expire(self, now: float) -> list[Job]:
        """Under the broker lock: flag jobs past their deadline (or of a
        revoked session); the caller closes them outside the lock."""
        late = [
            job
            for job in self.jobs.values()
            if job.state not in ENDED_JOB
            and (now >= job.deadline or job.session.revoked)
            and not job.cancel.is_set()
        ]
        for job in late:
            job.timed_out = not job.session.revoked
            job.cancel.set()
        return late

    def cancel_session(self, session: Any) -> list[Job]:
        with self._lock:
            jobs = self.pending(session)
            for job in jobs:
                job.cancel.set()
        for job in jobs:
            self.close(job)
        return jobs

    # -- pulls -----------------------------------------------------------------

    def run_pull(self, job: Job) -> None:
        # Another backend's images (sandbox_e2b) resolve pulls themselves.
        custom = getattr(self._puller, "run_pull", None)
        if custom is not None:
            custom(self, job)
            return
        repository, tag, policy, lease = job.detail
        state, error, image = "failed", None, None
        reference = reference_text(repository, tag)
        try:
            self.running(job)
            known = self._host_images()
            before = self._puller.inspect(reference)
            # A cached image serves "missing" only when it was pulled from
            # this repository: a host-built image under an approved name never
            # binds (S3, S9).
            pulled = not (
                policy == "missing"
                and before is not None
                and pulled_from(before, repository, tag)
            )
            if pulled:
                with self._lock:
                    exhausted = not self._envs._remaining(job.session, "pull_bytes")
                if exhausted:
                    # Never a download past the budget; a cached image is free.
                    raise JobFailed("quota", "pull budget exhausted")
                self.log(job, f"pulling {reference}\n")
                self._puller.pull(
                    repository,
                    tag,
                    progress=lambda line: self.log(job, line),
                    stream=lambda response: self.hold(job, response.close),
                    budget=lambda announced: self._pull_budget(job, announced),
                )
            if job.cancel.is_set():
                raise JobFailed("canceled", "the pull was canceled")
            attrs = self._puller.inspect(reference)
            if attrs is None:
                raise JobFailed("registry", "the pulled image is not present")
            if attrs.get("Architecture") != "amd64":
                raise JobFailed("refused", "only linux/amd64 images are supported")
            try:
                image_preflight(attrs, attrs.get("Id"))
            except SandboxError as refusal:
                raise JobFailed("refused", refusal.message) from None
            size = attrs.get("Size") if type(attrs.get("Size")) is int else 0
            with self._lock:
                if job.session.revoked or job.cancel.is_set():
                    raise JobFailed("canceled", "the pull was canceled")
                if pulled:
                    try:
                        self._envs._charge(job.session, pull_bytes=size)
                    except SandboxError:
                        # Its bytes are on the host now: spend the budget so
                        # no further pull fits (S6).
                        self._envs._force(job.session, pull_bytes=size)
                        raise JobFailed(
                            "quota", "the image exceeds the pull budget"
                        ) from None
                lease = SandboxImageLease(
                    owner=lease.owner,
                    handle=job.handle,
                    kind="pulled",
                    image_id=attrs["Id"],
                    pre_existing=before is not None and before.get("Id") == attrs["Id"],
                    state="present",
                    bytes=size,
                )
                self._envs._commit_image(lease)
                image = Image(
                    job.handle,
                    job.session,
                    "pulled",
                    attrs["Id"],
                    reference,
                    size,
                    attrs,
                    lease,
                )
                self.bind(image)
                state = "succeeded"
            self._record_pull(repository, tag, lease, before, known)
        except Exception as failure:
            state, error = self.outcome(job, failure)
        finally:
            if state != "succeeded":
                try:
                    self._envs._commit_image(
                        lease.model_copy(update={"state": "removed"})
                    )
                except InfrastructureError:
                    pass  # already failed closed
            self.finish(
                job,
                state,
                error,
                None
                if image is None
                else {"image": image_view(image.attrs, image.handle)},
            )

    def _host_images(self) -> frozenset[str] | None:
        """The host's image IDs before a pull, for the pull ledger: the
        journal's ``pre_existing`` looks at the pulled reference only, the
        ledger at the image under any name. None: nothing is recorded."""
        if self.ledger is None:
            return None
        try:
            return frozenset(self._puller.image_ids())
        except Exception as error:
            LOGGER.warning(
                "sandbox pull ledger: host images unknown (%s); the pull is "
                "not recorded",
                type(error).__name__,
            )
            return None

    def _record_pull(
        self,
        repository: str,
        tag: str,
        lease: SandboxImageLease,
        before: Mapping[str, Any] | None,
        known: frozenset[str] | None,
    ) -> None:
        """Outside the broker lock: the host pull ledger. Never fails the
        pull; on an error the image simply stays unprunable."""
        if self.ledger is None or known is None:
            return
        try:
            self.ledger.record(
                lease.image_id,
                repository,
                tag,
                pre_existing=lease.pre_existing or lease.image_id in known,
                before=None if before is None else before.get("Id"),
                run_id=lease.owner.run_id,
            )
        except Exception as error:
            LOGGER.warning(
                "sandbox pull ledger not updated (%s); the image stays unprunable",
                type(error).__name__,
            )

    def _pull_budget(self, job: Job, announced: int) -> None:
        """Stop a pull whose announced layers exceed the remaining budget;
        what it announced is spent, so no further pull fits (S6)."""
        with self._lock:
            if announced <= self._envs._remaining(job.session, "pull_bytes"):
                return
            self._envs._force(job.session, pull_bytes=announced)
        raise JobFailed("quota", "the image exceeds the pull budget")


def job_deadline(session: Any) -> float:
    return session.deadline if session.deadline is not None else math.inf


__all__ = [
    "ENDED_JOB",
    "JOB_STATES",
    "PULL_POLICIES",
    "DockerImagePuller",
    "Image",
    "Job",
    "JobFailed",
    "JobRunner",
    "PullFailed",
    "image_view",
    "job_deadline",
    "pull_reference",
    "pulled_from",
    "reference_text",
    "registry_of",
]
