"""Pure sandbox fixtures, independent of Docker and host inventory."""

import json
import tomllib
from contextlib import nullcontext

from rsi_harness.runtime.sandbox_contracts import (
    SandboxLimits,
    SandboxPhaseGrant,
    SandboxPolicy,
    SandboxProfile,
    SandboxTask,
)
from rsi_harness.runtime.sandbox_policy import (
    load_sandbox_policy,
    parse_sandbox_task,
    resolve_env_grant,
    resolve_sandbox_grant,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeSandboxBackend:
    """Controllable daemon boundary; broker/journal/accounting stay real.

    ``paused_killer`` models production's paused-container killer (M3):
    a paused child is killed without a thaw. Without it termination of a
    paused child is refused, as for a backend built without one.
    """

    def __init__(self, *, paused_killer=True):
        self.states = {}
        self.events = []
        self.hooks = {}
        self.last_deadline = None
        self.paused_killer = paused_killer

    def _event(self, operation, lease):
        self.events.append((operation, lease.child_id))
        if operation in self.hooks:
            self.hooks[operation](lease)

    def create(self, lease, profile):
        self.states[lease.child_id] = {"Running": False, "Paused": False}
        self._event("create", lease)
        return lease.child_id * 2

    def start(self, lease):
        self._event("start", lease)
        self.states[lease.child_id]["Running"] = True

    def inspect(self, lease):
        return {"State": self.states[lease.child_id].copy()}

    def pause(self, lease):
        self._event("pause", lease)
        self.states[lease.child_id]["Paused"] = True

    def resume(self, lease, *, admission=None):
        with admission() if admission is not None else nullcontext():
            self._event("resume", lease)
            self.states[lease.child_id]["Paused"] = False

    def terminate(self, lease):
        from rsi_harness.errors import InfrastructureError

        self._event("terminate", lease)
        if self.states[lease.child_id]["Paused"]:
            if not self.paused_killer:
                raise InfrastructureError("paused: safe termination unavailable")
            self.states[lease.child_id]["Paused"] = False
        self.states[lease.child_id]["Running"] = False

    def remove(self, lease):
        self._event("remove", lease)
        del self.states[lease.child_id]

    def execute(self, lease, argv, cwd, env, deadline, output_limit):
        from rsi_harness.runtime.sandbox_contracts import SandboxResult

        self.last_deadline = deadline
        self._event("execute", lease)
        return SandboxResult(exit_code=3, stdout="out", stderr="err", duration_sec=0.1)

    def upload(self, lease, root, entries, deadline, *, byte_limit):
        self._event("upload", lease)

    def download(self, lease, root, paths, deadline, *, byte_limit):
        self._event("download", lease)
        return ()


def make_profile():
    return SandboxProfile(
        name="offline",
        image="sha256:" + "a" * 64,
        cpus=1,
        memory_mb=256,
        pids=32,
        max_lifetime_sec=120,
        workdir="/workspace",
        tmpfs_mb=(("/workspace", 32), ("/tmp", 8), ("/dev/shm", 8)),
    )


def make_limits():
    return SandboxLimits(
        max_live=2,
        max_created=8,
        max_operations=100,
        max_cpus=2,
        max_memory_mb=512,
        max_lifetime_sec=1200,
        max_upload_bytes=64 * 1024**2,
        max_download_bytes=64 * 1024**2,
        max_log_bytes=8 * 1024**2,
    )


def make_sandbox_task():
    phase = SandboxPhaseGrant(profiles=("offline",), limits=make_limits())
    return SandboxTask(version=1, profiles=(make_profile(),), work=phase, judge=phase)


def make_sandbox_policy():
    task = make_sandbox_task()
    return SandboxPolicy(
        version=1,
        profiles=task.profiles,
        work=task.work,
        judge=task.judge,
        run_limits=make_limits(),
        pool_cpus=8,
        pool_memory_mb=4096,
    )


def make_sandbox_grant():
    return resolve_sandbox_grant(
        make_sandbox_task(),
        make_sandbox_policy(),
        image_ids={"offline": make_profile().image},
        parent_cpus=1,
        parent_memory_mb=256,
    )


def sandbox_toml():
    return """
[metadata.rsi_harness.sandbox]
version = 1
[[metadata.rsi_harness.sandbox.profiles]]
name = "offline"
image = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
cpus = 1
memory_mb = 256
pids = 32
max_lifetime_sec = 120
workdir = "/workspace"
[metadata.rsi_harness.sandbox.profiles.tmpfs_mb]
"/workspace" = 32
"/tmp" = 8
"/dev/shm" = 8
[metadata.rsi_harness.sandbox.work]
profiles = ["offline"]
[metadata.rsi_harness.sandbox.work.limits]
max_live = 2
max_created = 8
max_operations = 100
max_cpus = 2
max_memory_mb = 512
max_lifetime_sec = 1200
max_upload_bytes = 67108864
max_download_bytes = 67108864
max_log_bytes = 8388608
"""


BUILDER_IMAGE = "moby/buildkit@sha256:" + "f" * 64
BUILDER_IMAGE_ID = "sha256:" + "b" * 64


def builder_inspect(image_id=BUILDER_IMAGE_ID, **config):
    """A ``docker image inspect`` result for the cached BuildKit builder."""
    return {
        "Id": image_id,
        "RepoDigests": [BUILDER_IMAGE],
        "Config": {
            "Entrypoint": ["buildkitd"],
            "Volumes": {"/var/lib/buildkit": {}},
            **config,
        },
    }


def sandbox_policy_toml(*, pool=True):
    """The v1 helper policy exactly as an operator would write it.

    Without ``pool`` only the profile tables remain, to append after another
    policy's top-level keys (a mixed v1 + environments policy file).
    """
    phase = """
profiles = ["offline"]
[{name}.limits]
max_live = 2
max_created = 8
max_operations = 100
max_cpus = 2
max_memory_mb = 512
max_lifetime_sec = 1200
max_upload_bytes = 67108864
max_download_bytes = 67108864
max_log_bytes = 8388608
"""
    return (
        ("version = 1\npool_cpus = 8\npool_memory_mb = 4096\n" if pool else "")
        + """
[[profiles]]
name = "offline"
image = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
cpus = 1
memory_mb = 256
pids = 32
max_lifetime_sec = 120
workdir = "/workspace"
[profiles.tmpfs_mb]
"/workspace" = 32
"/tmp" = 8
"/dev/shm" = 8
[work]"""
        + phase.format(name="work")
        + "[judge]"
        + phase.format(name="judge")
        + """[run_limits]
max_live = 2
max_created = 8
max_operations = 100
max_cpus = 2
max_memory_mb = 512
max_lifetime_sec = 1200
max_upload_bytes = 67108864
max_download_bytes = 67108864
max_log_bytes = 8388608
"""
    )


def env_policy_toml(
    *, pool_cpus=64, pool_memory_mb=131072, build=None, allowlist=None, tmux=None
):
    """The operator's one-time environment grant (spec section 3.5).

    ``build`` overrides fields of both phases' ``[environments.<phase>.build]``
    tables (e.g. ``{"state_fs": "tmpfs", "builder_image": ...}``).
    ``allowlist`` (a dict, maybe empty) adds network allowlist to both phases
    with that ``[environments.<phase>.allowlist]`` table. ``tmux`` (a dict
    of ``path`` and ``sha256``) adds ``[environments.host.tmux]``.
    """
    host_tmux = ""
    if tmux is not None:
        host_tmux = "\n[environments.host.tmux]\n" + "".join(
            f"{key} = {json.dumps(value)}\n" for key, value in tmux.items()
        )
    phase = """
network = ["public", "none"]
pull = true
registries = ["docker.io"]
max_envs_live = 4
max_envs_created = 400
max_services_per_env = 8
max_containers_live = 16
max_cpus_live = 16
max_memory_mb_live = 32768
max_disk_mb_live = 8192
cpus_per_container = 4
memory_mb_per_container = 8192
pids_per_container = 8192
disk_mb_per_container = 10240
max_env_lifetime_sec = 14400
max_wait_timeout_sec = 900
max_execs_running = 64
max_exec_output_bytes = 16777216
max_jobs_running = 4
max_pull_mb = 20480
max_log_bytes = 2147483648
max_upload_bytes = 17179869184
max_download_bytes = 17179869184
max_operations = 1000000

[environments.{name}.build]
builder_image = "{builder}"
network = ["public", "none"]
cpus = 4
memory_mb = 8192
pids = 4096
disk_mb = 6144
state_fs = "loop-ext4"
max_builds = 64
max_concurrent_builds = 1
max_build_sec = 3600
max_context_mb = 1024
max_image_mb = 8192
max_images_total_mb = 6144
syntax_frontends = ["docker.io/docker/dockerfile"]
"""
    if allowlist is not None:
        phase = (
            phase.replace(
                'network = ["public", "none"]\npull',
                'network = ["public", "none", "allowlist"]\npull',
                1,
            )
            + "\n[environments.{name}.allowlist]\n"
            + "".join(
                f"{key} = {json.dumps(value)}\n" for key, value in allowlist.items()
            )
        )
    return (
        f"""
pool_cpus = {pool_cpus}
pool_memory_mb = {pool_memory_mb}

[environments.host]
pool_disk_mb = 40960
disk_floor_mb = 10240
disk_hard_floor_mb = 4096
request_slots_active = 8
request_slots_queued = 32
waiters = 64
no_new_privileges = true
"""
        + host_tmux
        + """
[environments.judge]"""
        + _build_overrides(phase.format(name="judge", builder=BUILDER_IMAGE), build)
        + "\n[environments.work]"
        + _build_overrides(phase.format(name="work", builder=BUILDER_IMAGE), build)
        + """
[environments.run_limits]
max_envs_created = 1000
max_operations = 2000000
max_builds = 256
max_pull_mb = 40960
max_upload_bytes = 34359738368
max_download_bytes = 34359738368
max_log_bytes = 4294967296
"""
    )


def _build_overrides(table, build):
    """Replace ``key = value`` lines of a phase's build table."""
    if not build:
        return table
    head, marker, body = table.partition(".build]")
    lines = body.split("\n")
    for key, value in build.items():
        literal = json.dumps(value)
        index = next(
            (i for i, line in enumerate(lines) if line.startswith(f"{key} = ")), None
        )
        if index is None:
            lines.insert(1, f"{key} = {literal}")
        else:
            lines[index] = f"{key} = {literal}"
    return head + marker + "\n".join(lines)


def env_task_toml():
    return """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.judge]
network = ["public"]
pull = true
build = true
[metadata.rsi_harness.sandbox.environments.judge.limits]
max_envs_live = 2
"""


def load_policy_text(tmp_path, text, name="sandbox-policy.toml"):
    path = tmp_path / name
    path.write_text(text)
    return load_sandbox_policy(path)


def make_env_policy(tmp_path, **options):
    return load_policy_text(tmp_path, env_policy_toml(**options))


def make_env_task(text=None):
    raw = tomllib.loads(text or env_task_toml())
    return parse_sandbox_task(raw["metadata"]["rsi_harness"]["sandbox"])


def make_env_grant(tmp_path, task=None, **options):
    return resolve_env_grant(
        task or make_env_task(),
        make_env_policy(tmp_path, **options),
        builder_images={"work": builder_inspect(), "judge": builder_inspect()},
        parent_cpus=1,
        parent_memory_mb=256,
    )


def make_env_spec(**service_updates):
    """A valid two-service wire EnvSpec (main depends on a healthy sidecar)."""
    main = {
        "image": "i" + "a" * 32,
        "command": ["sleep", "infinity"],
        "env": {"MODE": "test"},
        "working_dir": "/app",
        "cpus": 1,
        "memory_mb": 1024,
        "tmpfs": {"/scratch": 64},
        "mounts": [{"volume": "shared", "target": "/data", "read_only": False}],
        "depends_on": {"kv": {"condition": "healthy", "required": True}},
        "extra_hosts": [["mirror.internal", "203.0.113.7"]],
    }
    main.update(service_updates)
    return {
        "version": 1,
        "network": "public",
        "lifetime_sec": 600,
        "disk_mb": 2048,
        "volumes": {"shared": {"seeded": True}},
        "services": {
            "main": main,
            "kv": {
                "image": "i" + "b" * 32,
                "cpus": 0.5,
                "memory_mb": 256,
                "pids": 256,
                "aliases": ["kvstore"],
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval_sec": 1,
                    "retries": 30,
                },
                "mounts": [{"volume": "shared", "target": "/seed", "read_only": True}],
            },
        },
    }
