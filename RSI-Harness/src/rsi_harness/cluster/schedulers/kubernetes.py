"""Synchronous Kubernetes submission: one Job of one Pod per allocation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from rsi_harness.cluster.config import (
    KubernetesSchedulerProfile,
    KubernetesVolumeProfile,
)
from rsi_harness.cluster.schedulers.slurm import SlurmJobResult, SlurmJobSpec
from rsi_harness.errors import InfrastructureError, SetupError

_JOB_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]{0,50}[a-z0-9])?$")
_LABEL_VALUE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,61}[A-Za-z0-9])?$")
_NAME_LABEL = "rsi-harness.io/name"
_MANAGED_BY = {"app.kubernetes.io/managed-by": "rsi-harness"}
# Set with the suspension that cancels a Job; a queue (Kueue) suspends too.
_CANCELLED = "rsi-harness.io/cancelled"
_CONTAINER = "rsi-harness"
_ACCOUNT_NAME = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
# A container that cannot start; anything else waiting is still pending.
_FATAL_WAITING = {
    "CreateContainerConfigError",
    "CreateContainerError",
    "ErrImageNeverPull",
    "InvalidImageName",
}
_PULL_BACKOFF = {"ErrImagePull", "ImagePullBackOff"}
_MISSING_CLAIM = re.compile(r'persistentvolumeclaim "[^"]+" not found')
# How long a start failure must persist before the Job is cancelled: an
# admission refusal is final sooner than a mount or sandbox that may clear.
_REFUSED_SECONDS = 60.0
_START_FAILURE_SECONDS = 300.0
# Events of a Pod that cannot be created or started; quota waits are pending.
_START_FAILURE_EVENTS = {"FailedCreate", "FailedMount", "FailedCreatePodSandBox"}
_TRANSIENT_ERRORS = (
    "connection refused",
    # kubectl's own wording: "The connection to the server HOST was refused".
    "was refused",
    "no route to host",
    "connection timed out",
    "http2: client connection lost",
    "net/http: request canceled",
    "service unavailable",
    "serviceunavailable",
    "the server was unable to return a response",
    "connection reset",
    "i/o timeout",
    "tls handshake timeout",
    "unable to connect to the server",
    "the server is currently unable",
    "etcdserver: request timed out",
    "context deadline exceeded",
)
# How long the API may stay unreachable before a query fails, and how long
# one kubectl request may take.
_TRANSIENT_SECONDS = 900.0
# How long cancelling after an interrupt may wait for the API.
_INTERRUPT_CANCEL_SECONDS = 120.0
_REQUEST_TIMEOUT = "60s"
# Above twice the request timeout: a create first fetches the API schema.
_KUBECTL_TIMEOUT = 150.0
# The submit host's environment reaches the Engine (as Slurm's --export=ALL
# would) through a 0600 file, never the Job object; not these, which belong
# to the submit host's session or to the Pod.
_LOCAL_ENVIRONMENT = {
    "COLORTERM",
    "DISPLAY",
    "HOME",
    "HOSTNAME",
    "HOSTTYPE",
    "KUBECONFIG",
    "LOGNAME",
    "LS_COLORS",
    "MACHTYPE",
    "MAIL",
    "OLDPWD",
    "OSTYPE",
    "PATH",
    "PWD",
    "SHELL",
    "SHLVL",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "WAYLAND_DISPLAY",
    "_",
}
_LOCAL_PREFIXES = (
    "APPTAINER",
    "BASH_FUNC",
    "CONDA",
    "CUDA_",
    "DBUS_",
    "KUBERNETES_",
    "LD_",
    "LSB_",
    "NVIDIA_",
    "NVM_",
    "PYTHON",
    "RSI_HARNESS_",
    "SINGULARITY",
    "SLURM_",
    "SSH_",
    "SUDO_",
    "TERM_",
    "UV_",
    "VIRTUAL_ENV",
    "VSCODE_",
    "XDG_",
    "npm_",
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Apptainer and rsi-box need user namespaces; the GPUs are the device
# plugin's allocation (NVIDIA_VISIBLE_DEVICES), not node indices.
_WRAPPER = """\
exec >>"$1" 2>>"$2"
# A Job whose create landed after the Harness cancelled it.
if [[ -e $7 ]]; then
  echo "kubernetes preflight: the Harness cancelled this job before it started" >&2
  exit 1
fi
if [[ -n $6 ]]; then
  if [[ ! -r $6 ]]; then
    echo "kubernetes preflight: the job's environment file is missing or" \\
      "unreadable: $6" >&2
    exit 1
  fi
  set -a
  . "$6"
  set +a
  # Credentials: read once, never left on the shared volume.
  rm -f -- "$6"
fi
if ! command -v unshare >/dev/null; then
  echo "kubernetes preflight: the image has no unshare (util-linux)" >&2
  exit 1
fi
# What Apptainer needs: a user namespace that can mount a fresh /proc.
if ! unshare --user --map-root-user true 2>/dev/null; then
  echo "kubernetes preflight: this Pod cannot create user namespaces," \\
    "which Apptainer needs: see scheduler.pod_security and, on nodes with" \\
    "kernel.apparmor_restrict_unprivileged_userns=1, scheduler.apparmor_profile" >&2
  exit 1
fi
if ! unshare --user --map-root-user --mount --pid --fork --mount-proc true \\
    2>/dev/null; then
  covered=$(awk '$5 ~ "^/proc/" {print $5}' /proc/self/mountinfo)
  if [[ $covered == *"/proc/driver/nvidia"* ]]; then
    advice="an NVIDIA runtime hook or CDI hook does: inject GPUs with CDI"
    advice+=" from a spec without disable-device-node-modification"
  else
    advice="a masked /proc does: see scheduler.pod_security"
    advice+=" (userns needs procMount Unmasked)"
  fi
  echo "kubernetes preflight: this Pod cannot mount a fresh /proc, which" \\
    "Apptainer needs; the container runtime mounts over /proc" \\
    "($(head -n 3 <<<"$covered" | paste -sd ' ')), as $advice" >&2
  exit 1
fi
if (( $4 )); then
  # The device plugin's UUIDs when it passes them; otherwise (CDI, volume
  # mounts, indices) the Pod holds only its own GPUs, numbered 0..N-1.
  devices=${NVIDIA_VISIBLE_DEVICES:-}
  if [[ $devices == GPU-* || $devices == MIG-* ]]; then
    IFS=, read -ra allocated <<<"$devices"
  else
    allocated=()
    for (( index = 0; index < $4; index++ )); do allocated+=("$index"); done
  fi
  shopt -s nullglob
  nodes=("$5"/nvidia[0-9]*)
  # MIG instances share their GPU's device node.
  [[ $devices == MIG-* ]] && expected=${#nodes[@]} || expected=$4
  if (( ${#allocated[@]} != $4 || ${#nodes[@]} != expected )); then
    echo "kubernetes preflight: the Pod has ${#nodes[@]} GPU devices and" \\
      "${#allocated[@]} allocated (NVIDIA_VISIBLE_DEVICES=$devices), not $4:" \\
      "see scheduler.gpu_resource, runtime_class and privileged_devices" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$(IFS=,; printf '%s' "${allocated[*]}")
  export CUDA_VISIBLE_DEVICES
fi
exec /bin/bash "$3"
"""
CommandRunner = Callable[
    [tuple[str, ...], str | None], subprocess.CompletedProcess[str]
]


def _run(argv: tuple[str, ...], stdin: str | None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            input=stdin,
            # --request-timeout bounds the request, not kubectl's discovery.
            timeout=_KUBECTL_TIMEOUT,
            # A terminal's Ctrl-C reaches the Harness, which cancels the Job
            # with kubectl, never kubectl itself.
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        lines = stderr.strip().splitlines()
        detail = f"kubectl gave no answer in {_KUBECTL_TIMEOUT:.0f} s: i/o timeout"
        if lines:
            detail = f"{detail} ({lines[-1]})"
        return subprocess.CompletedProcess(argv, 124, "", detail)


class KubernetesJobSpec(SlurmJobSpec):
    """Share the cluster allocation fields; render a Kubernetes Job."""

    # Run jobs get the submit host's environment (credentials included).
    forward_environment: bool = False
    # Build-only scheduling overrides.
    image: str | None = None
    node_selector: dict[str, str] | None = None
    tolerations: tuple[dict[str, Any], ...] | None = None


class KubernetesJobResult(SlurmJobResult):
    """Terminal Job status and its Pod's exit code."""


def _label_value(value: str) -> str:
    if _LABEL_VALUE.fullmatch(value):
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", value)[:50].strip("._-") or "rsi"
    return f"{stem}-{digest}"


def _short(message: str, limit: int = 10**4) -> str:
    """The scheduler's reason, without its preemption report, cut at a word."""
    text = message.splitlines()[0].split(" preemption:")[0].strip() if message else ""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def _timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _walltime_seconds(walltime: str) -> int:
    hours, minutes = (int(part) for part in walltime.split(":"))
    return (hours * 60 + minutes) * 60


def _volume_at(
    profile: KubernetesSchedulerProfile, path: PurePosixPath
) -> tuple[int, KubernetesVolumeProfile] | None:
    """The innermost volume mounted over ``path``, as the Pod sees it."""
    matches = [
        (index, volume)
        for index, volume in enumerate(profile.volumes)
        if path == volume.path or path.is_relative_to(volume.path)
    ]
    return max(matches, key=lambda item: len(item[1].path.parts), default=None)


def uncovered_paths(
    profile: KubernetesSchedulerProfile, paths: tuple[Path, ...]
) -> tuple[Path, ...]:
    """Paths a Pod would not see at the submit host's path."""
    missing = []
    for path in paths:
        posix = PurePosixPath(path)
        found = _volume_at(profile, posix)
        if found is not None:
            if not found[1].shared:
                missing.append(path)
        elif not any(
            posix == root or posix.is_relative_to(root) for root in profile.image_paths
        ):
            missing.append(path)
    return tuple(missing)


class KubernetesScheduler:
    def __init__(
        self,
        profile: KubernetesSchedulerProfile,
        *,
        owner: str,
        run_as: tuple[int, int, tuple[int, ...]] | None = None,
        runner: CommandRunner = _run,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        token: Callable[[], str] = lambda: secrets.token_hex(3),
    ) -> None:
        self._profile = profile
        self._owner = owner
        self._run_as = run_as
        self._runner = runner
        self._sleeper = sleeper
        self._clock = clock
        self._token = token
        # One name per job spec, so a dry run shows the submitted files.
        self._names: dict[tuple[str, str], str] = {}
        # Each Job's credential file, removed once the Job has ended.
        self._environment_files: dict[str, Path] = {}
        self._job_directories: dict[str, Path] = {}

    # -- rendering -------------------------------------------------------

    def _kubectl(self, *args: str) -> tuple[str, ...]:
        profile = self._profile
        argv = [profile.kubectl]
        if profile.kubeconfig is not None:
            argv += ["--kubeconfig", str(profile.kubeconfig)]
        if profile.context is not None:
            argv += ["--context", profile.context]
        argv += [
            f"--request-timeout={_REQUEST_TIMEOUT}",
            "--namespace",
            profile.namespace,
            *args,
        ]
        return tuple(argv)

    def job_name(self, spec: KubernetesJobSpec) -> str:
        """A fresh DNS-1123 name: finished Jobs keep theirs until removed."""
        key = (spec.name, str(spec.script_path))
        if key not in self._names:
            stem = re.sub(r"[^a-z0-9-]+", "-", spec.name.lower()).strip("-")
            name = f"{stem[:44].rstrip('-')}-{self._token()}"
            if _JOB_NAME.fullmatch(name) is None:
                raise SetupError(
                    f"cannot derive a Kubernetes Job name from {spec.name}"
                )
            self._names[key] = name
        return self._names[key]

    @staticmethod
    def job_directory(spec: KubernetesJobSpec, name: str) -> Path:
        """The Job's manifest and account files, beside its logs and outside
        the run's frozen control directory."""
        return spec.stdout_path.parent / name

    @classmethod
    def environment_path(cls, spec: KubernetesJobSpec, name: str) -> Path:
        return cls.job_directory(spec, name) / "environment"

    def environment_text(self, environ: Mapping[str, str]) -> str:
        # The profile's own settings win over the submit host's.
        return "".join(
            f"{key}={shlex.quote(value)}\n"
            for key, value in sorted(environ.items())
            if _ENV_NAME.fullmatch(key)
            and key not in _LOCAL_ENVIRONMENT
            and key not in self._profile.environment
            and not key.startswith(_LOCAL_PREFIXES)
        )

    def render_submit(self, spec: KubernetesJobSpec) -> tuple[str, ...]:
        path = self.job_directory(spec, self.job_name(spec)) / "job.json"
        return self._kubectl("create", "--output=name", f"--filename={path}")

    def render_manifest(
        self, spec: KubernetesJobSpec, name: str | None = None
    ) -> dict[str, Any]:
        profile = self._profile
        name = name or self.job_name(spec)
        if (
            spec.gpu_count
            and profile.pod_security == "privileged"
            and profile.privileged_devices == "host"
        ):
            raise SetupError(
                "privileged Pods see every GPU on the node, so GPUs cannot be "
                'confined: use pod_security = "userns", or set '
                'privileged_devices = "allocated" where the container runtime '
                "hides host devices (privileged_without_host_devices)"
            )
        labels = {
            **profile.labels,
            **_MANAGED_BY,
            _NAME_LABEL: _label_value(spec.name),
            "rsi-harness.io/owner": _label_value(self._owner),
        }
        walltime = _walltime_seconds(spec.walltime)
        # The Pod's deadline counts from its start, not from queueing. An
        # advance signal is the Pod's termination at the deadline: the
        # grace period that follows is the reserve the Engine drains in.
        grace = spec.signal_seconds or 30
        deadline = walltime - spec.signal_seconds if spec.signal_seconds else walltime
        if deadline <= 0:
            raise SetupError("Kubernetes job walltime leaves no time before its drain")
        requests: dict[str, str] = {
            "cpu": str(spec.cpu_slots),
            "memory": f"{spec.memory_mb}Mi",
        }
        limits = dict(requests)
        if spec.gpu_count:
            requests[profile.gpu_resource] = limits[profile.gpu_resource] = str(
                spec.gpu_count
            )
        if spec.local_tmp_mb and spec.request_tmp:
            # Requested for scheduling only: a limit would evict the Pod.
            requests["ephemeral-storage"] = f"{spec.local_tmp_mb}Mi"
        environment = {
            **profile.environment,
            "USER": self._owner,
            "LOGNAME": self._owner,
            "RSI_HARNESS_KUBERNETES_JOB": name,
        }
        stdout = str(spec.stdout_path).replace("%j", name)
        stderr = str(spec.stderr_path).replace("%j", name)
        container: dict[str, Any] = {
            "name": _CONTAINER,
            "image": spec.image or profile.image,
            "imagePullPolicy": profile.image_pull_policy,
            "command": [
                "/bin/bash",
                "-c",
                _WRAPPER,
                _CONTAINER,
                stdout,
                stderr,
                str(spec.script_path),
                str(spec.gpu_count),
                "/dev",
                str(self.environment_path(spec, name))
                if spec.forward_environment
                else "",
                str(self.job_directory(spec, name) / "cancelled"),
            ],
            "workingDir": str(spec.stdout_path.parent),
            "env": [
                {"name": key, "value": value} for key, value in environment.items()
            ],
            "resources": {"requests": requests, "limits": limits},
            "volumeMounts": [
                self._mount(index, volume)
                for index, volume in enumerate(profile.volumes)
            ],
        }
        pod: dict[str, Any] = {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": deadline,
            "terminationGracePeriodSeconds": grace,
            # No API credential and no other services' addresses reach the
            # Pod; network policy decides what public-network Work can reach.
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "containers": [container],
            "volumes": [
                self._volume(index, volume)
                for index, volume in enumerate(profile.volumes)
            ],
        }
        security: dict[str, Any]
        if profile.pod_security == "privileged":
            security = {"privileged": True, "allowPrivilegeEscalation": True}
        else:
            pod["hostUsers"] = False
            security = {
                "procMount": "Unmasked",
                "seccompProfile": {"type": "Unconfined"},
                "appArmorProfile": {"type": "Unconfined"},
            }
        if profile.apparmor_profile is not None:
            security["appArmorProfile"] = {
                "type": "Localhost",
                "localhostProfile": profile.apparmor_profile,
            }
        container["securityContext"] = security
        if self._run_as is not None:
            uid, gid, groups = self._run_as
            pod["securityContext"] = {"runAsUser": uid, "runAsGroup": gid}
            if groups:
                pod["securityContext"]["supplementalGroups"] = list(groups)
            if uid:
                # Apptainer refuses a uid the image's passwd does not name.
                directory = self.job_directory(spec, name)
                for target in ("passwd", "group"):
                    container["volumeMounts"].append(
                        self._file_mount(
                            directory / target, PurePosixPath("/etc") / target
                        )
                    )
        node_selector = (
            spec.node_selector
            if spec.node_selector is not None
            else profile.node_selector
        )
        if node_selector:
            pod["nodeSelector"] = dict(node_selector)
        tolerations = (
            spec.tolerations if spec.tolerations is not None else profile.tolerations
        )
        if tolerations:
            pod["tolerations"] = [dict(item) for item in tolerations]
        if spec.excluded_hosts:
            pod["affinity"] = {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "kubernetes.io/hostname",
                                        "operator": "NotIn",
                                        "values": list(spec.excluded_hosts),
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        if profile.service_account is not None:
            pod["serviceAccountName"] = profile.service_account
        if profile.priority_class is not None:
            pod["priorityClassName"] = profile.priority_class
        if profile.runtime_class is not None:
            pod["runtimeClassName"] = profile.runtime_class
        if profile.image_pull_secrets:
            pod["imagePullSecrets"] = [
                {"name": secret} for secret in profile.image_pull_secrets
            ]
        job_spec: dict[str, Any] = {
            # A retry would rerun the payload over the same run logs.
            "backoffLimit": 0,
            "template": {"metadata": {"labels": dict(labels)}, "spec": pod},
        }
        if profile.ttl_seconds_after_finished is not None:
            job_spec["ttlSecondsAfterFinished"] = profile.ttl_seconds_after_finished
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": profile.namespace,
                "labels": labels,
            },
            "spec": job_spec,
        }

    def account_text(self) -> dict[str, str]:
        assert self._run_as is not None
        uid, gid, groups = self._run_as
        name = self._owner if _ACCOUNT_NAME.fullmatch(self._owner) else "rsi"
        passwd = (
            "root:x:0:0:root:/root:/bin/bash\n"
            f"{name}:x:{uid}:{gid}:{name}:/tmp:/bin/bash\n"
            "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
        )
        group = "root:x:0:\n" + "".join(
            f"{name if number == gid else f'g{number}'}:x:{number}:{name}\n"
            for number in dict.fromkeys((gid, *groups))
            if number
        )
        return {"passwd": passwd, "group": group}

    def _file_mount(self, source: Path, target: PurePosixPath) -> dict[str, Any]:
        """Mount one file of the shared volume that holds it (subPath)."""
        found = _volume_at(self._profile, PurePosixPath(source))
        if found is None or not found[1].shared:
            raise SetupError(f"no shared Kubernetes volume carries the job's {source}")
        index, volume = found
        relative = PurePosixPath(source).relative_to(volume.path)
        if volume.sub_path is not None:
            relative = PurePosixPath(volume.sub_path) / relative
        return {
            "name": f"v{index}",
            "mountPath": str(target),
            "subPath": str(relative),
            "readOnly": True,
        }

    @staticmethod
    def _mount(index: int, volume: KubernetesVolumeProfile) -> dict[str, Any]:
        mount: dict[str, Any] = {"name": f"v{index}", "mountPath": str(volume.path)}
        if volume.read_only:
            mount["readOnly"] = True
        if volume.sub_path is not None:
            mount["subPath"] = volume.sub_path
        return mount

    @staticmethod
    def _volume(index: int, volume: KubernetesVolumeProfile) -> dict[str, Any]:
        item: dict[str, Any] = {"name": f"v{index}"}
        if volume.host_path is not None:
            item["hostPath"] = {"path": str(volume.host_path), "type": "Directory"}
        elif volume.claim is not None:
            item["persistentVolumeClaim"] = {"claimName": volume.claim}
        elif volume.nfs_server is not None:
            item["nfs"] = {"server": volume.nfs_server, "path": str(volume.nfs_path)}
        else:
            item["emptyDir"] = {}
        return item

    def uncovered(self, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        return uncovered_paths(self._profile, paths)

    # -- scheduler port --------------------------------------------------

    def submit(self, spec: KubernetesJobSpec) -> str:
        if not spec.script_path.is_file():
            raise SetupError(f"Kubernetes payload does not exist: {spec.script_path}")
        name = self.job_name(spec)
        manifest = self.render_manifest(spec, name)
        directory = self.job_directory(spec, name)
        directory.mkdir(mode=0o755, exist_ok=True)
        self._job_directories[name] = directory
        if self._run_as is not None and self._run_as[0]:
            for filename, text in self.account_text().items():
                (directory / filename).write_text(text)
                (directory / filename).chmod(0o644)
        if spec.forward_environment:
            descriptor = os.open(
                self.environment_path(spec, name),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            with os.fdopen(descriptor, "w") as stream:
                stream.write(self.environment_text(os.environ))
            self._environment_files[name] = self.environment_path(spec, name)
        path = directory / "job.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        # The name is ours: a create the server accepted before an error
        # (a timeout) is found by name, and a retry cannot run it twice.
        restore = self._interrupt_on_termination()
        try:
            completed = self._call(*self.render_submit(spec)[len(self._kubectl()) :])
        except KeyboardInterrupt:
            restore()
            # A create the server received just before the interrupt may land
            # after a first look; its Pod finds the cancelled marker.
            if self._cancel_interrupted(name, second_look=5.0):
                raise KeyboardInterrupt(
                    f"interrupted; Kubernetes Job {name} cancelled"
                ) from None
            raise KeyboardInterrupt(
                f"interrupted before Kubernetes Job {name} was created; should it "
                "appear, it stops before the payload"
            ) from None
        except BaseException:
            self._forget_environment(name)
            raise
        finally:
            restore()
        if completed.returncode == 0:
            if completed.stdout.strip() != f"job.batch/{name}":
                self._forget_environment(name)
                raise InfrastructureError("Kubernetes submission returned no exact Job")
            return name
        try:
            job = self._job(name)
        except InfrastructureError:
            job = None
        if job is not None and (job.get("metadata", {}).get("labels") or {}).get(
            _NAME_LABEL
        ) == _label_value(spec.name):
            return name
        self._forget_environment(name)
        raise InfrastructureError(
            f"Kubernetes submission failed: {self._detail(completed)}"
        )

    def require_name_available(self, name: str) -> None:
        completed = self._call(
            "get",
            "jobs",
            f"--selector={_NAME_LABEL}={_label_value(name)}",
            "-o",
            "json",
        )
        if completed.returncode:
            raise InfrastructureError(
                f"cannot inspect Kubernetes Jobs: {self._detail(completed)}"
            )
        for job in self._items(completed):
            job_id = str(job.get("metadata", {}).get("name") or "")
            if self._terminal(job, self._pod(job_id) if job_id else None) is None:
                raise InfrastructureError(
                    f"Kubernetes Job of name {name} is already active: {job_id}"
                )

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float = 5.0,
        on_state: Callable[[str], None] | None = None,
    ) -> KubernetesJobResult:
        self._require_job_id(job_id)
        previous_state: str | None = None
        missing_polls = 0
        failing_since: float | None = None
        restore = self._interrupt_on_termination()
        try:
            while True:
                job = self._job(job_id)
                if job is None:
                    missing_polls += 1
                    if missing_polls >= 12:
                        self._forget_environment(job_id)
                        raise InfrastructureError(
                            f"Kubernetes Job {job_id} no longer exists"
                        )
                    self._sleeper(poll_seconds)
                    continue
                missing_polls = 0
                pod = self._pod(job_id)
                terminal = self._terminal(job, pod)
                if terminal is not None:
                    state, _ended = terminal
                    self._forget_environment(job_id)
                    if state != previous_state and on_state is not None:
                        on_state(state)
                    rejected = self._rejection(pod)
                    if state != "COMPLETED" and rejected:
                        raise InfrastructureError(
                            f"Kubernetes rejected the Pod of Job {job_id}: {rejected}"
                        )
                    return KubernetesJobResult(
                        job_id=job_id,
                        state=state,
                        exit_code=self._exit_code(pod, state),
                    )
                state, reason, message = self._pod_state(pod)
                window: float | None = None
                if reason in _FATAL_WAITING:
                    window = 0.0
                elif reason in _PULL_BACKOFF:
                    window = _START_FAILURE_SECONDS
                elif reason == "Unschedulable" and _MISSING_CLAIM.search(message):
                    # Not capacity: a claim the Pod names does not exist.
                    window = _REFUSED_SECONDS
                elif (
                    state == "PENDING"
                    and reason != "Unschedulable"
                    and (failure := self._start_failure(job_id, pod))
                ):
                    reason, message = failure
                    window = (
                        _REFUSED_SECONDS
                        if reason == "FailedCreate"
                        else _START_FAILURE_SECONDS
                    )
                if window is None:
                    failing_since = None
                else:
                    now = self._clock()
                    failing_since = now if failing_since is None else failing_since
                    if now - failing_since >= window:
                        self.cancel(job_id)
                        raise InfrastructureError(
                            f"Kubernetes Job {job_id} cannot start its container: "
                            f"{reason}: {_short(message)}"
                        )
                label = "ImagePull" if reason in _PULL_BACKOFF else reason
                if reason == "Unschedulable" and message:
                    label = f"Unschedulable: {_short(message, 120)}"
                shown = (
                    f"{state} ({label})"
                    if state == "PENDING" and label and label != "ContainerCreating"
                    else state
                )
                if shown != previous_state and on_state is not None:
                    on_state(shown)
                previous_state = shown
                self._sleeper(poll_seconds)
        except KeyboardInterrupt:
            restore()
            self._cancel_interrupted(job_id)
            raise KeyboardInterrupt(
                f"interrupted; Kubernetes Job {job_id} cancelled"
            ) from None
        finally:
            restore()

    def _cancel_interrupted(self, job_id: str, *, second_look: float = 0.0) -> bool:
        """Cancel after an interrupt; a second interrupt cannot stop it, and
        the API gets a short wait. Whether the Job was there."""
        print(f"cancelling Kubernetes Job {job_id}", file=sys.stderr, flush=True)
        ignored = {}
        if threading.current_thread() is threading.main_thread():
            ignored = {
                number: signal.signal(number, signal.SIG_IGN)
                for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
            }
        try:
            found = self._suspend(job_id, patience=_INTERRUPT_CANCEL_SECONDS)
            if not found and second_look:
                self._sleeper(second_look)
                found = self._suspend(job_id, patience=_INTERRUPT_CANCEL_SECONDS)
            return found
        except InfrastructureError as error:
            raise InfrastructureError(
                f"interrupted; could not cancel Kubernetes Job {job_id}, which "
                f"may still run: {error}"
            ) from None
        finally:
            for number, handler in ignored.items():
                signal.signal(number, handler)

    def _forget_environment(self, job_id: str) -> None:
        # The Engine read it at start, and a cancelled Job never restarts.
        environment = self._environment_files.pop(job_id, None)
        if environment is not None:
            environment.unlink(missing_ok=True)

    @staticmethod
    def _interrupt_on_termination() -> Callable[[], None]:
        """SIGTERM or SIGHUP to the submit command cancels the Job as Ctrl-C
        does, instead of leaving it running untracked."""
        if threading.current_thread() is not threading.main_thread():
            return lambda: None

        def interrupt(_signum: int, _frame: object) -> None:
            raise KeyboardInterrupt

        previous = {
            number: signal.signal(number, interrupt)
            for number in (signal.SIGTERM, signal.SIGHUP)
        }

        def restore() -> None:
            for number, handler in previous.items():
                signal.signal(number, handler)

        return restore

    def finished(self, job_id: str) -> KubernetesJobResult | None:
        """The Job's end, or None while it still runs or waits to start;
        state ``UNKNOWN`` once Kubernetes has removed it."""
        self._require_job_id(job_id)
        job = self._job(job_id)
        if job is None:
            return KubernetesJobResult(job_id=job_id, state="UNKNOWN", exit_code=1)
        pod = self._pod(job_id)
        terminal = self._terminal(job, pod)
        if terminal is None:
            return None
        state, ended_at = terminal
        return KubernetesJobResult(
            job_id=job_id,
            state=state,
            exit_code=self._exit_code(pod, state),
            ended_at=ended_at,
        )

    def cancel(self, job_id: str) -> None:
        self._require_job_id(job_id)
        self._suspend(job_id)

    def _suspend(self, job_id: str, *, patience: float = _TRANSIENT_SECONDS) -> bool:
        # Suspending deletes the Pod gracefully (the Engine gets SIGTERM and
        # its grace period to close the attempt), never starts another, and
        # keeps the Job and its end for ``finished`` and resume. Should a
        # queue later unsuspend it, the Engine refuses to rerun the attempt.
        patch = {
            "metadata": {"annotations": {_CANCELLED: "true"}},
            "spec": {"suspend": True},
        }
        directory = self._job_directories.get(job_id)
        if directory is not None:
            # Should its Pod start anyway, it stops before the payload.
            (directory / "cancelled").touch()
        try:
            completed = self._call(
                "patch",
                "job",
                job_id,
                "--type=merge",
                f"--patch={json.dumps(patch, separators=(',', ':'))}",
                patience=patience,
            )
        finally:
            self._forget_environment(job_id)
        if completed.returncode and not self._not_found(completed):
            raise InfrastructureError(
                f"cannot cancel Kubernetes Job {job_id}: {self._detail(completed)}"
            )
        return not completed.returncode

    # -- queries ---------------------------------------------------------

    def _call(
        self, *args: str, patience: float = _TRANSIENT_SECONDS
    ) -> subprocess.CompletedProcess[str]:
        argv = self._kubectl(*args)
        started = self._clock()
        while True:
            completed = self._runner(argv, None)
            if not completed.returncode or not any(
                marker in self._detail(completed).lower()
                for marker in _TRANSIENT_ERRORS
            ):
                return completed
            if self._clock() - started + self._profile.poll_seconds >= patience:
                raise InfrastructureError(
                    "the Kubernetes API has been unreachable for "
                    f"{patience / 60:.0f} minutes: {self._detail(completed)}"
                )
            self._sleeper(self._profile.poll_seconds)

    def _job(self, job_id: str) -> dict[str, Any] | None:
        completed = self._call("get", "job", job_id, "-o", "json")
        if completed.returncode:
            if self._not_found(completed):
                return None
            raise InfrastructureError(
                f"cannot query Kubernetes Job {job_id}: {self._detail(completed)}"
            )
        try:
            job = json.loads(completed.stdout)
        except ValueError as error:
            raise InfrastructureError(
                f"invalid Kubernetes status for Job {job_id}"
            ) from error
        if not isinstance(job, dict) or job.get("metadata", {}).get("name") != job_id:
            raise InfrastructureError(f"invalid Kubernetes status for Job {job_id}")
        return job

    def _pod(self, job_id: str) -> dict[str, Any] | None:
        completed = self._call(
            "get", "pods", f"--selector=job-name={job_id}", "-o", "json"
        )
        if completed.returncode:
            raise InfrastructureError(
                f"cannot query the Pod of Kubernetes Job {job_id}: "
                f"{self._detail(completed)}"
            )
        pods = self._items(completed)
        if len(pods) > 1:
            raise InfrastructureError(f"Kubernetes Job {job_id} has several Pods")
        return pods[0] if pods else None

    def _start_failure(
        self, job_id: str, pod: Mapping[str, Any] | None
    ) -> tuple[str, str] | None:
        """The newest warning that the Job's Pod cannot be created or
        started, if that is the latest thing that happened to it."""
        names = [job_id]
        pod_name = ((pod or {}).get("metadata") or {}).get("name")
        if pod_name:
            names.append(str(pod_name))
        events: list[dict[str, Any]] = []
        for name in names:
            completed = self._call(
                "get",
                "events",
                f"--field-selector=involvedObject.name={name}",
                "-o",
                "json",
            )
            if completed.returncode:
                return None
            events += self._items(completed)
        events.sort(
            key=lambda event: str(
                event.get("lastTimestamp") or event.get("eventTime") or ""
            )
        )
        if not events:
            return None
        latest = events[-1]
        reason = str(latest.get("reason") or "")
        message = str(latest.get("message") or "")
        if reason not in _START_FAILURE_EVENTS or "exceeded quota" in message:
            return None
        return reason, message

    @staticmethod
    def _items(completed: subprocess.CompletedProcess[str]) -> list[dict[str, Any]]:
        try:
            items = json.loads(completed.stdout).get("items", [])
        except (ValueError, AttributeError) as error:
            raise InfrastructureError("invalid Kubernetes list output") from error
        if not isinstance(items, list):
            raise InfrastructureError("invalid Kubernetes list output")
        return [item for item in items if isinstance(item, dict)]

    @classmethod
    def _not_found(cls, completed: subprocess.CompletedProcess[str]) -> bool:
        return "notfound" in cls._detail(completed).lower().replace(" ", "")

    @classmethod
    def _terminal(
        cls, job: Mapping[str, Any], pod: Mapping[str, Any] | None
    ) -> tuple[str, float | None] | None:
        status = job.get("status") or {}
        annotations = (job.get("metadata") or {}).get("annotations") or {}
        for condition in status.get("conditions") or ():
            if condition.get("status") != "True":
                continue
            kind = condition.get("type")
            # Cancelled by the Harness, and its Pod is gone (a terminating
            # Pod is still listed). Another suspension is a queue's: pending.
            if kind == "Suspended" and _CANCELLED in annotations and pod is None:
                return "CANCELLED", _timestamp(condition.get("lastTransitionTime"))
            if kind == "Complete":
                ended = _timestamp(status.get("completionTime")) or _timestamp(
                    condition.get("lastTransitionTime")
                )
                return "COMPLETED", ended
            if kind == "Failed":
                deadline = (
                    condition.get("reason") == "DeadlineExceeded"
                    or cls._pod_reason(pod) == "DeadlineExceeded"
                )
                return (
                    "TIMEOUT" if deadline else "FAILED",
                    _timestamp(condition.get("lastTransitionTime")),
                )
        return None

    @staticmethod
    def _container_status(pod: Mapping[str, Any] | None) -> Mapping[str, Any]:
        statuses = ((pod or {}).get("status") or {}).get("containerStatuses") or ()
        for status in statuses:
            if status.get("name") == _CONTAINER:
                return status
        return {}

    @classmethod
    def _pod_state(cls, pod: Mapping[str, Any] | None) -> tuple[str, str, str]:
        if pod is None:
            return "PENDING", "", ""
        waiting = (cls._container_status(pod).get("state") or {}).get("waiting") or {}
        reason = str(waiting.get("reason") or "")
        message = str(waiting.get("message") or "")
        phase = (pod.get("status") or {}).get("phase") or "Pending"
        for condition in (pod.get("status") or {}).get("conditions") or ():
            if (
                condition.get("type") == "PodScheduled"
                and condition.get("status") == "False"
                and condition.get("reason") == "Unschedulable"
            ):
                return "PENDING", "Unschedulable", str(condition.get("message") or "")
        # A Pod that ended before its Job did is still the Job's run.
        return ("PENDING" if phase == "Pending" else "RUNNING"), reason, message

    @classmethod
    def _rejection(cls, pod: Mapping[str, Any] | None) -> str:
        """Why the kubelet refused a Pod whose container never ran (an
        AppArmor profile not loaded, for example)."""
        if pod is None or (cls._container_status(pod).get("state") or {}).get(
            "terminated"
        ):
            return ""
        status = pod.get("status") or {}
        reason, message = status.get("reason"), status.get("message")
        return f"{reason}: {message}" if reason and message else ""

    @staticmethod
    def _pod_reason(pod: Mapping[str, Any] | None) -> str:
        return str(((pod or {}).get("status") or {}).get("reason") or "")

    @classmethod
    def _exit_code(cls, pod: Mapping[str, Any] | None, state: str) -> int:
        terminated = (cls._container_status(pod).get("state") or {}).get(
            "terminated"
        ) or {}
        code = terminated.get("exitCode")
        if isinstance(code, int):
            if state != "COMPLETED" and code == 0:
                return 1
            return code
        return 0 if state == "COMPLETED" else 1

    @staticmethod
    def _require_job_id(job_id: str) -> None:
        if _JOB_NAME.fullmatch(job_id) is None:
            raise SetupError(f"invalid exact Kubernetes Job name: {job_id!r}")

    @staticmethod
    def _detail(completed: subprocess.CompletedProcess[str]) -> str:
        return completed.stderr.strip() or completed.stdout.strip()


__all__ = [
    "KubernetesJobResult",
    "KubernetesJobSpec",
    "KubernetesScheduler",
    "uncovered_paths",
]
