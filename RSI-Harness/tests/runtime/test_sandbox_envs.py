"""Brokered envs: ownership, quotas, replay, phase barriers, failure isolation."""

import hashlib
import io
import signal
import tarfile
import threading
import time
from dataclasses import dataclass

import pytest

from rsi_harness.errors import InfrastructureError, RetryableSubmissionError
from rsi_harness.runtime import sandbox_envs
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_archive import STAGE_TTL_SEC, ArchiveSummary
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_contracts import SandboxError, SandboxOwner
from rsi_harness.runtime.sandbox_disk import DiskVerdict
from rsi_harness.runtime.sandbox_env_docker import (
    EnvStartResult,
    ServiceStatus,
    _revise,
    _service,
    _volume,
    plan_env,
)
from rsi_harness.runtime.sandbox_envs import EnvRuntime
from rsi_harness.runtime.sandbox_exec import ExecPump, ExecTarget
from rsi_harness.runtime.sandbox_images import (
    PullFailed,
    pull_reference,
    reference_text,
)
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_exec import (
    CONTAINER,
    INIT_PID,
    TICK_SEC,
    FakeApi,
    FakeHost,
    RecordingKiller,
)
from tests.sandbox_helpers import (
    FakeClock,
    FakeSandboxBackend,
    make_env_grant,
    make_env_task,
)

MIB = 1024**2
BUSYBOX = "busybox:1.37.0"
BUSYBOX_REF = "docker.io/library/busybox:1.37.0"
BUSYBOX_ID = "sha256:" + "a" * 64
TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
[metadata.rsi_harness.sandbox.environments.work.limits]
max_envs_live = 2
max_envs_created = 3
max_execs_running = 2
max_exec_output_bytes = 4096
[metadata.rsi_harness.sandbox.environments.judge]
network = ["none"]
pull = true
[metadata.rsi_harness.sandbox.environments.judge.limits]
max_envs_live = 2
"""


def image_attrs(image_id=BUSYBOX_ID, repo_digests=None, size=4 * MIB, **config):
    return {
        "Id": image_id,
        "Os": "linux",
        "Architecture": "amd64",
        "Size": size,
        # Docker's familiar form: provenance of an image pulled from Hub.
        "RepoDigests": ["busybox@sha256:" + "d" * 64]
        if repo_digests is None
        else repo_digests,
        "Config": {"Env": ["PATH=/bin"], "Cmd": ["sh"], **config},
    }


# -- fakes ---------------------------------------------------------------------


class FakeResponse:
    def __init__(self):
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


class FakePuller:
    """The daemon's image store and a registry; pulls can be held open."""

    def __init__(self):
        self.local = {}
        self.registry = {BUSYBOX_REF: image_attrs()}
        self.gate = None
        self.pulls = []
        # Summed layer sizes the daemon announces during a pull.
        self.announce = None

    def inspect(self, reference):
        return self.local.get(reference)

    def image_ids(self):
        return [attrs["Id"] for attrs in self.local.values()]

    def pull(self, repository, tag, *, progress, stream, budget=None):
        response = FakeResponse()
        stream(response)
        self.pulls.append((repository, tag))
        progress("abc123 Pull complete\n")
        if budget is not None and self.announce is not None:
            budget(self.announce)
        if self.gate is not None:
            while not (self.gate.wait(0.01) or response.closed.is_set()):
                pass
            if response.closed.is_set():
                raise OSError("stream closed")
        reference = reference_text(repository, tag)
        if reference not in self.registry:
            raise PullFailed("registry", "manifest unknown")
        self.local[reference] = self.registry[reference]


class FakeEnvBackend:
    """SandboxEnvDockerBackend's surface over an in-memory daemon.

    Every step commits through the broker exactly like the real backend;
    ``errors`` maps an operation to the exception it raises.
    """

    def __init__(self):
        self.events = []
        self.containers = {}
        self.errors = {}
        self.start_gate = None
        self.destroy_gate = None
        # operation -> Event the operation waits on (at most 5 s) first.
        self.gates = {}
        # Allowlist envs: notes env_create reports, refresh_network calls.
        self.allowlist_notes_result = ()
        self.refreshes = []
        self.plans = []

    def _hook(self, operation, lease):
        self.events.append((operation, lease.env_id))
        gate = self.gates.get(operation)
        if gate is not None:
            gate.wait(5)
        error = self.errors.get(operation)
        if error is not None:
            raise error

    def plan(self, owner, env_id, spec, images, *, default_pids, **options):
        self.plans.append(
            plan_env(owner, env_id, spec, images, default_pids=default_pids, **options)
        )
        return self.plans[-1]

    def allowlist_notes(self, plan):
        return self.allowlist_notes_result

    def refresh_network(self, plan):
        self.refreshes.append(plan.env_id)
        gate = self.gates.get("refresh")
        if gate is not None:
            gate.wait(5)
        error = self.errors.get("refresh")
        if error is not None:
            raise error
        return True

    def create(self, plan, lease, commit):
        self._hook("create", lease)
        if plan.network is not None:
            lease = commit(_revise(lease, network_id=plan.env_id[1:] * 2))
        for volume in plan.volumes:
            lease = commit(_volume(lease, volume.idx, created=True))
        for service in plan.services:
            identity = hashlib.sha256(service.container_name.encode()).hexdigest()
            self.containers[identity] = "created"
            lease = commit(
                _service(lease, service.idx, container_id=identity, state="created")
            )
        return commit(_revise(lease, state="created", pending_mutation=False))

    def start(self, plan, lease, commit, *, wait_timeout_sec, cancelled):
        lease = commit(_revise(lease, state="starting"))
        self._hook("start", lease)
        while self.start_gate is not None and not self.start_gate.wait(0.01):
            if cancelled():
                return EnvStartResult(
                    commit(_revise(lease, state="failed", reason="canceled")),
                    "failed",
                    "canceled",
                )
        for record in lease.services:
            self.containers[record.container_id] = "running"
            lease = _service(lease, record.idx, state="running")
        return EnvStartResult(commit(_revise(lease, state="ready")), "ready")

    def status(self, lease):
        self.events.append(("status", lease.env_id))
        return {
            record.name: ServiceStatus(
                self.containers.get(record.container_id, "removed")
            )
            for record in lease.services
        }

    def diagnostics(self, lease, service):
        return {"health_tail": "", "log_tail": "bounded tail"}

    def pause(self, lease, commit):
        self._hook("pause", lease)
        for record in lease.services:
            if self.containers.get(record.container_id) == "running":
                self.containers[record.container_id] = "paused"
                lease = _service(lease, record.idx, state="paused")
        if lease.state == "ready":
            lease = _revise(lease, state="paused")
        return commit(lease)

    def resume(self, lease, commit, *, admission=None):
        self._hook("resume", lease)
        for record in lease.services:
            if record.state == "paused":
                with admission():
                    self.containers[record.container_id] = "running"
                lease = _service(lease, record.idx, state="running")
        if lease.state == "paused":
            lease = _revise(lease, state="ready")
        return commit(lease)

    def stop_service(self, lease, commit, service, *, timeout_sec):
        self._hook("stop_service", lease)
        record = next(item for item in lease.services if item.name == service)
        self.containers[record.container_id] = "exited"
        return commit(_service(lease, record.idx, state="exited")), 0

    def terminate(self, lease, commit, *, deadline=None):
        self._hook("terminate", lease)
        for record in lease.services:
            if self.containers.get(record.container_id) in ("running", "paused"):
                self.containers[record.container_id] = "exited"
                lease = commit(_service(lease, record.idx, state="exited"))
        return lease

    def fail(self, lease, commit, reason):
        self._hook("fail", lease)
        lease = self.terminate(lease, commit)
        return commit(_revise(lease, state="failed", reason=reason))

    def destroy(self, lease, commit, *, reason=None):
        self._hook("destroy", lease)
        while self.destroy_gate is not None and not self.destroy_gate.wait(0.01):
            pass
        lease = commit(_revise(lease, state="stopping", reason=reason or lease.reason))
        for record in lease.services:
            self.containers.pop(record.container_id, None)
            lease = commit(_service(lease, record.idx, state="removed"))
        for volume in lease.volumes:
            lease = commit(_volume(lease, volume.idx, created=False))
        lease = commit(_revise(lease, network_id=None, pending_mutation=False))
        return commit(_revise(lease, state="removed"))

    def archive_target(self, lease, service):
        self._hook("archive_target", lease)
        return (lease.env_id, service)

    def exec_target(self, lease, service):
        self._hook("exec_target", lease)
        record = next(item for item in lease.services if item.name == service)

        def inspect():
            state = self.containers.get(record.container_id)
            if state is None:
                return None
            return {
                "Id": CONTAINER,
                "State": {
                    "Running": state in ("running", "paused"),
                    "Paused": state == "paused",
                    "Pid": INIT_PID,
                },
            }

        return ExecTarget(CONTAINER, inspect)


class FakeTransfer:
    """ArchiveTransfer's surface over a real stage store."""

    def __init__(self, stages):
        self.stages = stages
        self.calls = []
        self.gate = None
        # Container paths path_stat reports as missing.
        self.absent = set()

    def copy_in(self, target, dest_dir, stage_id):
        if self.gate is not None:
            self.gate.wait(5)
        with self.stages.consume(stage_id) as (stage, summary):
            self.calls.append(("copy_in", target, dest_dir, stage.read()))
        return {"entries": summary.entries, "bytes": summary.bytes}

    def copy_out(self, target, path, *, max_bytes, exclude):
        self.calls.append(("copy_out", target, path, max_bytes))
        with self.stages.create_result() as result:
            result.file.write(tar_of({"out.txt": b"result"}))
            result.summary = ArchiveSummary(1, 6)
        return {"stage_id": result.stage_id, "bytes": 6, "entries": 1, "skipped": 0}

    def path_stat(self, target, path, *, follow):
        self.calls.append(("path_stat", target, path, follow))
        if path in self.absent:
            return {"exists": False, "kind": None, "size": None, "mode": None}
        return {"exists": True, "kind": "dir", "size": 0, "mode": 0o755}


class FakeDisk:
    def __init__(self):
        self.verdict = DiskVerdict(usage={})
        self.budgets = []
        self.forgotten = []
        self.full = False

    def admit(self, requested_mb=0, *, field="disk_mb"):
        if self.full:
            raise SandboxError("quota", field, "host disk free is below the floor")

    def poll(self, budgets):
        self.budgets = list(budgets)
        return self.verdict

    def usage(self, env_id):
        from rsi_harness.runtime.sandbox_disk import EnvDiskUsage

        return EnvDiskUsage(env_id, {"main": 3 * MIB})

    def forget(self, env_id):
        self.forgotten.append(env_id)


def tar_of(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@dataclass
class Kit:
    broker: object
    clock: FakeClock
    backend: FakeEnvBackend
    puller: FakePuller
    api: FakeApi
    host: FakeHost
    disk: FakeDisk
    transfers: list
    killer: RecordingKiller
    # Append anything to make the spool filesystem refuse stage writes.
    spool_full: list

    @property
    def envs(self):
        return self.broker.envs

    @property
    def pump(self):
        return self.broker.envs.pump

    def io(self):
        for _ in range(16):
            self.pump.step_io(0)

    def sweep(self):
        """One watchdog turn, including the disk watchdog's own thread."""
        self.broker.sweep_expired()
        self.envs.join_disk(5)

    def turn(self, seconds=0.0):
        end = round(self.clock.now + seconds, 6)
        while True:
            self.io()
            self.pump.step_control()
            self.io()
            if self.clock.now >= end:
                return
            self.clock.now = round(min(end, self.clock.now + TICK_SEC), 6)


def build_kit(tmp_path, grant):
    """A broker over the fakes with ``grant``; close with ``close_kit``."""
    from rsi_harness.runtime.sandbox import SandboxBroker

    clock = FakeClock()
    backend, puller, disk = FakeEnvBackend(), FakePuller(), FakeDisk()
    api, host, killer = FakeApi(), FakeHost(tmp_path / "host"), RecordingKiller()
    transfers = []

    def transfer(stages):
        transfers.append(FakeTransfer(stages))
        return transfers[-1]

    spool_full = []

    def spool_admit(field_name):
        if spool_full:
            raise SandboxError("quota", field_name, "spool disk is below the floor")

    runtime = EnvRuntime(
        backend=backend,
        images=puller,
        transfer=transfer,
        pump=lambda on_finish: ExecPump(
            api,
            tmp_path / "spool",
            killer=killer,
            table=host.table,
            on_finish=on_finish,
            clock=clock,
            start_threads=False,
        ),
        spool_root=tmp_path / "spool",
        disk=disk,
        table=host.table,
        spool_admit=spool_admit,
    )
    journal = SandboxJournal(authority(LeaseStore(tmp_path / "leases")))
    broker = SandboxBroker(grant, FakeSandboxBackend(), journal, clock, envs=runtime)
    return Kit(
        broker, clock, backend, puller, api, host, disk, transfers, killer, spool_full
    )


def close_kit(kit):
    kit.envs.join_disk(5)
    kit.envs.join_refresh(5)
    kit.pump.close()


@pytest.fixture
def kit(tmp_path):
    made = build_kit(tmp_path, make_env_grant(tmp_path, make_env_task(TASK)))
    yield made
    close_kit(made)


def open_work(kit, deadline=10_000):
    session = kit.broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
    )
    kit.broker.activate_work(deadline)
    return session


def open_round(kit, round_id="r1", deadline=5_000):
    session = kit.broker.open_judge(
        SandboxOwner(run_id="run-1", task_id="task", phase="judge", round_id=round_id),
        None,
    )
    kit.broker.activate_judge(deadline)
    return session


def pull(kit, session, request_id="pull", ref=BUSYBOX):
    job_id = kit.broker.image_pull(session.credential, ref, "missing", request_id)[
        "job_id"
    ]
    kit.envs.images.jobs[job_id].thread.join(5)
    view = kit.broker.job_wait(session.credential, job_id, 0, 0)
    assert view["state"] == "succeeded", view
    return view["result"]["image"]["handle"]


def single(handle, **service):
    return {
        "version": 1,
        "network": "none",
        "lifetime_sec": 600,
        "disk_mb": 64,
        "services": {
            "main": {
                "image": handle,
                "command": ["sleep", "600"],
                "cpus": 0.5,
                "memory_mb": 64,
                **service,
            }
        },
    }


def ready(kit, session, handle, request_id="env", spec=None):
    created = kit.broker.env_create(
        session.credential, spec or single(handle), request_id
    )
    env_id = created["env_id"]
    kit.broker.env_start(session.credential, env_id, 30, request_id + "-start")
    join_starter(kit, env_id)
    assert kit.broker.env_status(session.credential, env_id)["state"] == "ready"
    return env_id


def join_starter(kit, env_id):
    starter = kit.envs._envs[env_id].starter
    if starter is not None:
        starter.join(5)


def join_reaper(kit, env_id):
    reaper = kit.envs._envs[env_id].reaper
    if reaper is not None:
        reaper.join(5)


def start_exec(kit, session, env_id, request_id="x1", argv=("sh", "-c", "work"), **f):
    fields = dict(cwd=None, env={}, user=None, timeout_sec=None, merge_stderr=False)
    fields.update(f)
    exec_id = kit.broker.exec_start(
        session.credential, env_id, "main", list(argv), request_id=request_id, **fields
    )["exec_id"]
    docker = kit.api.last()
    docker.pid = 200 + 10 * len(kit.api.created)
    kit.host.add(docker.pid, start=5000 + docker.pid)
    kit.io()
    return exec_id, docker


def finish(kit, docker, code=0):
    docker.running, docker.exit_code = False, code
    kit.host.remove(docker.pid)
    docker.peer.close()
    kit.turn(1.5)


def stage(kit, session, data, request_id="stage"):
    return kit.broker.stage_put(
        session.credential,
        None,
        0,
        True,
        hashlib.sha256(data).hexdigest(),
        request_id,
        data,
    )["stage_id"]


def usage(kit, session):
    return kit.broker.capabilities(session.credential)["environments"]["usage"]


# -- ownership -------------------------------------------------------------------


def test_every_handle_resolves_only_in_the_session_that_created_it(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    env_id = ready(kit, work, handle)
    exec_id, _ = start_exec(kit, work, env_id)
    job_id = next(iter(kit.envs.images.jobs))
    stage_id = stage(kit, work, tar_of({"a": b"1"}))
    kit.broker.freeze_work()
    judge = open_round(kit)
    token = judge.credential

    for operation, call in (
        ("env_status", lambda: kit.broker.env_status(token, env_id)),
        ("env_destroy", lambda: kit.broker.env_destroy(token, env_id)),
        ("exec_wait", lambda: kit.broker.exec_wait(token, exec_id, 0, 0, 0, 1)),
        ("exec_kill", lambda: kit.broker.exec_kill(token, exec_id, "KILL", "group")),
        ("job_wait", lambda: kit.broker.job_wait(token, job_id, 0, 0)),
        ("image_release", lambda: kit.broker.image_release(token, handle)),
        (
            "copy_in",
            lambda: kit.broker.copy_in(token, env_id, "main", "/", stage_id, "c"),
        ),
        ("env_create", lambda: kit.broker.env_create(token, single(handle), "e")),
    ):
        with pytest.raises(SandboxError) as caught:
            call()
        assert caught.value.code == "permission", operation
    # Stages are per session too: Judge's spool has no Work stage.
    with pytest.raises(SandboxError, match="unknown or expired stage"):
        kit.broker.stage_get(token, stage_id, 0, 10)

    judge_handle = pull(kit, judge)
    judge_env = ready(kit, judge, judge_handle, spec=single(judge_handle))
    # The frozen Work token still authenticates, but never sees a Judge env.
    with pytest.raises(SandboxError, match="permission"):
        kit.broker.env_status(work.credential, judge_env)
    kit.broker.close_judge()

    later = open_round(kit, "r2")
    for call in (
        lambda: kit.broker.env_status(later.credential, judge_env),
        lambda: kit.broker.env_create(later.credential, single(judge_handle), "e"),
    ):
        with pytest.raises(SandboxError, match="permission"):
            call()
    assert [env.env_id for env in kit.broker.journal.envs()] == [env_id]


def test_a_profile_grant_offers_no_environment_operations(tmp_path):
    from rsi_harness.runtime.sandbox import SandboxBroker
    from tests.sandbox_helpers import make_sandbox_grant

    journal = SandboxJournal(authority(LeaseStore(tmp_path)))
    broker = SandboxBroker(
        make_sandbox_grant(), FakeSandboxBackend(), journal, FakeClock()
    )
    work = broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), 1000
    )
    capabilities = broker.capabilities(work.credential)
    assert capabilities["versions"] == [1, 2]
    assert capabilities["version"] == 1 and capabilities["environments"] is None
    assert set(capabilities) == {
        "version",
        "versions",
        "owner",
        "grant",
        "profiles",
        "environments",
    }
    assert broker.supports("work", 1) and not broker.supports("work", 2)
    with pytest.raises(SandboxError, match="permission"):
        broker.env_list(work.credential)
    # Spec B3: the v1 synchronous exec never serves an env handle.
    with pytest.raises(SandboxError) as caught:
        broker.execute(work.credential, "e" + "0" * 32, ["true"], "/workspace", {}, 5)
    assert (caught.value.code, caught.value.field) == ("invalid", "child_id")


def test_an_environment_grant_offers_no_profiles(kit):
    work = open_work(kit)
    capabilities = kit.broker.capabilities(work.credential)
    assert (capabilities["grant"], capabilities["profiles"]) == (None, [])
    environments = capabilities["environments"]
    assert environments["network"] == ["public", "none"]
    assert environments["build"] is None  # the task requested no build
    assert environments["limits"]["max_envs_live"] == 2
    with pytest.raises(SandboxError, match="permission.*profile"):
        kit.broker.create(work.credential, "offline", 10, "one")
    with pytest.raises(SandboxError, match="permission: build"):
        kit.broker.image_build(
            work.credential,
            stage_id="s" + "0" * 32,
            dockerfile=None,
            dockerfile_inline=None,
            target=None,
            build_args={},
            labels={},
            no_cache=False,
            network="public",
            timeout_sec=60,
            request_id="build",
        )


# -- quotas ----------------------------------------------------------------------


def test_live_quotas_are_refunded_and_cumulative_ones_are_not(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    first = ready(kit, work, handle, "one")
    second = ready(kit, work, handle, "two")
    assert usage(kit, work)["envs_live"] == 2
    assert usage(kit, work)["cpus_live"] == 1.0
    with pytest.raises(SandboxError) as caught:
        kit.broker.env_create(work.credential, single(handle), "three")
    assert (caught.value.code, caught.value.field) == ("quota", "max_envs_live")

    kit.broker.env_destroy(work.credential, first)
    kit.broker.env_destroy(work.credential, first)  # idempotent, charged once
    assert usage(kit, work)["envs_live"] == 1
    assert usage(kit, work)["memory_mb_live"] == 64
    assert usage(kit, work)["swap_mb_live"] == 64  # the default swap_ratio 1
    ready(kit, work, handle, "three")
    kit.broker.env_destroy(work.credential, second)
    with pytest.raises(SandboxError) as caught:
        kit.broker.env_create(work.credential, single(handle), "four")
    assert (caught.value.code, caught.value.field) == ("quota", "max_envs_created")
    spent = usage(kit, work)
    assert (spent["envs_live"], spent["envs_created"]) == (1, 3)
    assert spent["operations"] >= 9
    # Removed envs leave the journal: the lease stays bounded.
    assert len(kit.broker.journal.envs()) == 1


def build_big_kit(tmp_path):
    """A Work session at the hard live caps: 128 envs of 512 containers."""
    from rsi_harness.runtime.sandbox_policy import resolve_env_grant
    from tests.sandbox_helpers import builder_inspect, env_policy_toml, load_policy_text

    policy = env_policy_toml(pool_cpus=256)
    for old, new in (
        ("max_envs_live = 4", "max_envs_live = 128"),
        ("max_containers_live = 16", "max_containers_live = 512"),
        ("max_cpus_live = 16", "max_cpus_live = 64"),
    ):
        policy = policy.replace(old, new)
    task = make_env_task(
        "[metadata.rsi_harness.sandbox]\nversion = 2\n"
        "[metadata.rsi_harness.sandbox.environments.work]\n"
        'network = ["none"]\npull = true\n'
    )
    grant = resolve_env_grant(
        task,
        load_policy_text(tmp_path, policy),
        builder_images={"work": builder_inspect(), "judge": builder_inspect()},
        parent_cpus=1,
        parent_memory_mb=256,
    )
    return build_kit(tmp_path, grant)


def test_a_session_holds_128_live_envs_and_refuses_the_129th(tmp_path):
    """Each of the 128 envs runs an image of its own (a SWE-bench set): a
    session holds two image handles per live env; the next is refused."""
    big = build_big_kit(tmp_path)
    try:
        work = open_work(big)
        refs = [f"rsi/image{index}:1" for index in range(257)]
        for index, ref in enumerate(refs):
            big.puller.registry[reference_text(*pull_reference(ref))] = image_attrs(
                f"sha256:{index:064x}", [f"rsi/image{index}@sha256:{index:064x}"], 1
            )
        handles = [
            pull(big, work, f"pull{index}", ref) for index, ref in enumerate(refs[:128])
        ]
        # Created envs hold every live quota; starting them adds nothing.
        envs = [
            big.broker.env_create(work.credential, single(handle), f"env{index}")
            for index, handle in enumerate(handles)
        ]
        spent = usage(big, work)
        assert (spent["envs_live"], spent["cpus_live"]) == (128, 64.0)
        with pytest.raises(SandboxError) as caught:
            big.broker.env_create(work.credential, single(handles[0]), "env128")
        assert (caught.value.code, caught.value.field) == ("quota", "max_envs_live")
        big.broker.env_destroy(work.credential, envs[0]["env_id"])
        ready(big, work, handles[0], "env128")
        assert usage(big, work)["envs_live"] == 128
        assert len(big.broker.journal.envs()) == 128

        handles += [
            pull(big, work, f"pull{index}", ref)
            for index, ref in enumerate(refs[128:256], start=128)
        ]
        assert len(set(handles)) == 256
        with pytest.raises(SandboxError) as caught:
            big.broker.image_pull(work.credential, refs[256], "missing", "pull256")
        assert (caught.value.code, caught.value.field) == ("quota", "image")
        # A handle a live env runs stays bound; an unused one frees a place.
        with pytest.raises(SandboxError) as caught:
            big.broker.image_release(work.credential, handles[1])
        assert caught.value.code == "busy"
        big.broker.image_release(work.credential, handles[255])
        pull(big, work, "pull256", refs[256])
    finally:
        close_kit(big)


def test_swap_follows_the_grant_ratio_and_is_a_refunded_live_quota(tmp_path):
    task = TASK.replace(
        "max_envs_live = 2\nmax_envs_created = 3\n",
        "max_envs_live = 2\nmax_envs_created = 3\nswap_ratio = 0.5\n",
    )
    swap_kit = build_kit(tmp_path, make_env_grant(tmp_path, make_env_task(task)))
    try:
        work = open_work(swap_kit)
        limits = swap_kit.broker.capabilities(work.credential)["environments"]["limits"]
        assert (limits["swap_ratio"], limits["max_swap_mb_live"]) == (0.5, 16384)
        handle = pull(swap_kit, work)
        first = ready(swap_kit, work, handle, "one", spec=single(handle, memory_mb=65))
        host = swap_kit.backend.plans[-1].services[0].host_config
        assert (host["Memory"], host["MemorySwap"]) == (65 * MIB, (65 + 32) * MIB)
        assert usage(swap_kit, work)["swap_mb_live"] == 32
        # Swap is refused like any live limit, never clamped.
        work_session = swap_kit.broker._sessions["work"]
        work_session.env_usage["swap_mb_live"] = 16384 - 31
        with pytest.raises(SandboxError) as caught:
            swap_kit.broker.env_create(work.credential, single(handle), "two")
        assert (caught.value.code, caught.value.field) == ("quota", "max_swap_mb_live")
        work_session.env_usage["swap_mb_live"] = 32
        swap_kit.broker.env_destroy(work.credential, first)
        assert usage(swap_kit, work)["swap_mb_live"] == 0
    finally:
        close_kit(swap_kit)


def test_running_execs_and_unused_output_are_refunded_on_exit(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    logged = usage(kit, work)["log_bytes"]  # the pull's progress log
    first, docker = start_exec(kit, work, env_id, "x1")
    start_exec(kit, work, env_id, "x2")
    assert usage(kit, work)["execs_running"] == 2
    assert usage(kit, work)["log_bytes"] == logged + 2 * 2 * 4096
    with pytest.raises(SandboxError) as caught:
        start_exec(kit, work, env_id, "x3")
    assert (caught.value.code, caught.value.field) == ("quota", "max_execs_running")

    docker.peer.sendall(b"\x01\x00\x00\x00\x00\x00\x00\x05hello")
    finish(kit, docker)
    view = kit.broker.exec_wait(work.credential, first, 0, 0, 0, 1024)
    assert (view["state"], view["exit_code"]) == ("exited", 0)
    spent = usage(kit, work)
    assert spent["execs_running"] == 1
    # Only the five bytes kept count; the rest of the reservation came back.
    assert spent["log_bytes"] == logged + 2 * 4096 + 5
    start_exec(kit, work, env_id, "x3")


# -- idempotency -----------------------------------------------------------------


def test_request_ids_replay_results_and_refuse_conflicting_reuse(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    created = kit.broker.env_create(work.credential, single(handle), "same")
    assert kit.broker.env_create(work.credential, single(handle), "same") == created
    assert [op for op, _ in kit.backend.events].count("create") == 1
    assert len(kit.broker.journal.envs()) == 1
    with pytest.raises(SandboxError) as caught:
        kit.broker.env_create(work.credential, single(handle, cpus=1), "same")
    assert (caught.value.code, caught.value.field) == ("invalid", "request_id")

    env_id = created["env_id"]
    kit.broker.env_start(work.credential, env_id, 30, "go")
    assert kit.broker.env_start(work.credential, env_id, 30, "go") == {
        "state": "starting"
    }
    join_starter(kit, env_id)
    exec_id, _ = start_exec(kit, work, env_id, "run")
    operations = usage(kit, work)["operations"]
    again = kit.broker.exec_start(
        work.credential, env_id, "main", ["sh", "-c", "work"],
        cwd=None, env={}, user=None, timeout_sec=None, merge_stderr=False,
        request_id="run",
    )  # fmt: skip
    assert again == {"exec_id": exec_id}
    assert len(kit.api.created) == 1
    assert usage(kit, work)["operations"] == operations
    with pytest.raises(SandboxError, match="invalid.*request_id"):
        start_exec(kit, work, env_id, "run", argv=("true",))

    data = tar_of({"seed.txt": b"seed"})
    stage_id = stage(kit, work, data, "put")
    assert stage(kit, work, data, "put") == stage_id
    result = kit.broker.copy_in(work.credential, env_id, "main", "/app", stage_id, "in")
    # The consumed stage cannot be copied twice, but the request replays.
    assert kit.broker.copy_in(
        work.credential, env_id, "main", "/app", stage_id, "in"
    ) == (result)
    assert [call[0] for call in kit.transfers[0].calls] == ["copy_in"]


# -- phase barriers --------------------------------------------------------------


def test_submit_waits_for_a_work_pull_or_env_start_in_flight(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    kit.puller.gate = threading.Event()
    job_id = kit.broker.image_pull(work.credential, "busybox:1.36", "always", "slow")[
        "job_id"
    ]
    with pytest.raises(RetryableSubmissionError):
        kit.broker.freeze_work()
    assert not kit.broker._sessions["work"].frozen
    kit.puller.gate.set()
    kit.envs.images.jobs[job_id].thread.join(5)
    assert kit.broker.job_wait(work.credential, job_id, 0, 0)["error"]["kind"] == (
        "registry"
    )

    kit.backend.start_gate = threading.Event()
    env_id = kit.broker.env_create(work.credential, single(handle), "env")["env_id"]
    kit.broker.env_start(work.credential, env_id, 30, "start")
    assert kit.broker.env_status(work.credential, env_id)["state"] == "starting"
    with pytest.raises(RetryableSubmissionError):
        kit.broker.freeze_work()
    assert ("pause", env_id) not in kit.backend.events
    kit.backend.start_gate.set()
    join_starter(kit, env_id)
    kit.broker.freeze_work()
    assert ("pause", env_id) in kit.backend.events
    assert kit.broker.env_status(work.credential, env_id)["state"] == "paused"


def test_frozen_work_exec_deadline_moves_by_the_frozen_time(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    exec_id, docker = start_exec(kit, work, env_id, timeout_sec=10)
    kit.turn(2.0)
    kit.broker.freeze_work()
    kit.clock.now += 500  # a Judge round
    judge = open_round(kit)
    assert judge is not None
    kit.broker.close_judge()
    kit.turn(1.0)
    assert kit.killer.calls == []
    kit.broker.resume_work()
    assert kit.broker.env_status(work.credential, env_id)["state"] == "ready"
    # 8 s of the timeout were left at the freeze.
    kit.turn(8.0 - 2 * TICK_SEC)
    assert kit.killer.calls == []
    kit.turn(3 * TICK_SEC)
    assert kit.killer.calls == [(docker.pid, signal.SIGTERM, True)]


def test_a_judge_activated_just_before_the_work_deadline_gets_its_own_lifetime(kit):
    """Spec A8 (M0): the round's envs live by its verifier timeout, not by
    the 60 s that Work has left; Work ends at its own deadline."""
    work = open_work(kit, deadline=kit.clock.now + 600)
    work_env = ready(kit, work, pull(kit, work))
    kit.clock.now += 540
    kit.broker.freeze_work()
    judge = open_round(kit, deadline=kit.clock.now + 900)
    limits = kit.broker.capabilities(judge.credential)["environments"]["limits"]
    spec = single(pull(kit, judge, "judge-pull"))
    del spec["lifetime_sec"]  # the grant's, as the Harbor plugin asks

    created = kit.broker.env_create(judge.credential, spec, "judge-env")

    assert created["expires_in_sec"] == pytest.approx(900)
    kit.clock.now += 3  # seeding
    stale = min(limits["max_wait_timeout_sec"], created["expires_in_sec"])
    with pytest.raises(SandboxError, match="remaining lifetime"):
        kit.broker.env_start(judge.credential, created["env_id"], stale, "stale")
    kit.broker.env_start(judge.credential, created["env_id"], stale - 4, "start")
    join_starter(kit, created["env_id"])
    kit.clock.now += 120  # past Work's deadline
    kit.sweep()

    view = kit.broker.env_status(judge.credential, created["env_id"])
    assert (view["state"], view["remaining_sec"]) == ("ready", pytest.approx(777))
    exec_id, _ = start_exec(kit, judge, created["env_id"], "judge-exec")
    assert exec_id
    with pytest.raises(SandboxError) as expired:
        kit.broker.env_status(work.credential, work_env)
    assert expired.value.code == "expired"


def test_close_judge_kills_then_removes_every_round_object_before_work(kit):
    work = open_work(kit)
    work_env = ready(kit, work, pull(kit, work))
    kit.broker.freeze_work()
    judge = open_round(kit)
    judge_env = ready(kit, judge, pull(kit, judge, "judge-pull"))
    exec_id, _ = start_exec(kit, judge, judge_env)
    kit.puller.gate = threading.Event()
    job_id = kit.broker.image_pull(judge.credential, "busybox:9", "always", "held")[
        "job_id"
    ]
    kit.backend.events.clear()

    kit.broker.close_judge()
    kit.io()  # the pump drops the round's exec records on its IO turn

    operations = [op for op, env in kit.backend.events if env == judge_env]
    assert operations.index("terminate") < operations.index("destroy")
    assert kit.envs.images.jobs[job_id].state == "canceled"
    assert exec_id not in kit.pump._records
    assert [env.env_id for env in kit.broker.journal.envs()] == [work_env]
    assert [image.owner.phase for image in kit.broker.journal.images()] == ["work"]
    assert not kit.broker.recovery_required
    kit.broker.resume_work()
    assert kit.broker.env_status(work.credential, work_env)["state"] == "ready"


@pytest.mark.parametrize("stage", ["terminate", "destroy"])
def test_unproven_judge_cleanup_fails_closed_and_work_never_resumes(kit, stage):
    work = open_work(kit)
    work_env = ready(kit, work, pull(kit, work))
    kit.broker.freeze_work()
    judge = open_round(kit)
    ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.backend.errors[stage] = InfrastructureError("recovery_required: unproven")

    with pytest.raises(InfrastructureError, match="recovery"):
        kit.broker.close_judge()

    assert kit.broker.recovery_required
    if stage == "terminate":
        assert "destroy" not in [op for op, _ in kit.backend.events]
    with pytest.raises(InfrastructureError):
        kit.broker.resume_work()
    assert kit.backend.containers[
        kit.envs._envs[work_env].lease.services[0].container_id
    ] == ("paused")


def test_close_judge_waits_for_a_removal_the_watchdog_began(kit):
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    env_id = ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.backend.destroy_gate = threading.Event()
    kit.clock.now += 601  # the env's lifetime ends; the round's does not
    kit.broker.sweep_expired()
    assert kit.envs._envs[env_id].reaper is not None
    threading.Timer(0.3, kit.backend.destroy_gate.set).start()

    kit.broker.close_judge()

    assert not kit.broker.recovery_required
    operations = [op for op, env in kit.backend.events if env == env_id]
    # The removal killed first (its proof, then its journaled stop);
    # close_judge relied on it and killed nothing.
    assert operations[-3:] == ["terminate", "terminate", "destroy"]
    assert operations.count("terminate") == 2
    status = kit.envs._envs[env_id].lease
    assert (status.state, status.reason) == ("removed", "expired")


def test_the_kill_stage_does_not_wait_on_a_removal_queued_for_the_journal(
    kit, monkeypatch
):
    """A removal proves its kill before it writes the journal: when its
    journal write queues (many envs ending at once), close_judge's kill
    stage still relies on it instead of failing closed."""
    monkeypatch.setattr(sandbox_envs, "KILL_SEC", 0.3)
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    env_id = ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.backend.gates["terminate"] = killing = threading.Event()
    kit.clock.now += 601
    kit.broker.sweep_expired()  # the reaper waits at its kill
    held = threading.Event()
    commit_env = kit.broker.journal.commit_env

    def queued(lease):
        held.wait(5)  # the journal is busy well past the kill stage
        return commit_env(lease)

    monkeypatch.setattr(kit.broker.journal, "commit_env", queued)
    threading.Timer(0.1, killing.set).start()  # once close_judge waits on it
    threading.Timer(1.0, held.set).start()

    kit.broker.close_judge()

    assert not kit.broker.recovery_required
    status = kit.envs._envs[env_id].lease
    assert (status.state, status.reason) == ("removed", "expired")
    kit.broker.resume_work()


def test_a_removal_slower_than_the_kill_stage_does_not_fail_closed(kit, monkeypatch):
    monkeypatch.setattr(sandbox_envs, "KILL_SEC", 0.2)
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    env_id = ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.backend.destroy_gate = threading.Event()
    kit.clock.now += 601
    kit.broker.sweep_expired()
    threading.Timer(0.8, kit.backend.destroy_gate.set).start()

    began = time.monotonic()
    kit.broker.close_judge()

    assert time.monotonic() - began >= 0.7  # the delete stage waited for it
    assert not kit.broker.recovery_required
    assert kit.envs._envs[env_id].lease.state == "removed"
    kit.broker.resume_work()


def test_a_kill_that_misses_the_kill_stage_fails_closed(kit, monkeypatch):
    monkeypatch.setattr(sandbox_envs, "KILL_SEC", 0.2)
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.backend.gates["terminate"] = held = threading.Event()
    kit.clock.now += 601
    kit.broker.sweep_expired()  # the reaper's kill now hangs
    try:
        with pytest.raises(InfrastructureError, match="recovery"):
            kit.broker.close_judge()
    finally:
        held.set()
    assert kit.broker.recovery_required
    with pytest.raises(InfrastructureError):
        kit.broker.resume_work()


def slow_kills(kit, monkeypatch, seconds):
    """Each terminate takes ``seconds`` and fails past its deadline."""
    terminate = kit.backend.terminate

    def slow(lease, commit, *, deadline=None):
        remaining = None if deadline is None else deadline - kit.clock.now
        time.sleep(seconds)
        if remaining is not None and remaining < seconds:
            raise InfrastructureError("recovery_required: kill proof missed")
        return terminate(lease, commit, deadline=deadline)

    monkeypatch.setattr(kit.backend, "terminate", slow)


def test_a_round_kills_its_envs_at_once_not_one_by_one(kit, monkeypatch):
    # One at a time, two 0.4 s kills would miss the 0.6 s kill stage.
    monkeypatch.setattr(sandbox_envs, "KILL_SEC", 0.6)
    monkeypatch.setattr(sandbox_envs, "KILL_SEC_PER_CONTAINER", 0.0)
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    handle = pull(kit, judge, "judge-pull")
    envs = [ready(kit, judge, handle, f"env{index}") for index in range(2)]
    slow_kills(kit, monkeypatch, 0.4)
    kit.broker.close_judge()
    assert not kit.broker.recovery_required
    assert [kit.envs._envs[env_id].lease.state for env_id in envs] == ["removed"] * 2
    kit.broker.resume_work()


def test_the_kill_stage_grows_with_the_rounds_containers(kit, monkeypatch):
    monkeypatch.setattr(sandbox_envs, "KILL_SEC", 0.2)
    monkeypatch.setattr(sandbox_envs, "KILL_SEC_PER_CONTAINER", 0.5)
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    env_id = ready(kit, judge, pull(kit, judge, "judge-pull"))
    slow_kills(kit, monkeypatch, 0.4)  # over KILL_SEC, within 0.2 + 0.5
    kit.broker.close_judge()
    assert not kit.broker.recovery_required
    assert kit.envs._envs[env_id].lease.state == "removed"


def test_a_removal_that_misses_the_delete_stage_fails_closed(kit, monkeypatch):
    monkeypatch.setattr(sandbox_envs, "DELETE_SEC", 0.3)
    monkeypatch.setattr(sandbox_envs, "DELETE_SEC_PER_ENV", 0.0)
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.backend.destroy_gate = threading.Event()
    threading.Timer(0.8, kit.backend.destroy_gate.set).start()

    with pytest.raises(InfrastructureError, match="recovery"):
        kit.broker.close_judge()

    assert kit.broker.recovery_required
    with pytest.raises(InfrastructureError):
        kit.broker.resume_work()


def test_the_delete_stage_grows_with_the_rounds_envs(kit, monkeypatch):
    monkeypatch.setattr(sandbox_envs, "DELETE_SEC", 0.3)
    monkeypatch.setattr(sandbox_envs, "DELETE_SEC_PER_ENV", 1.0)
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit)
    env_id = ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.backend.destroy_gate = threading.Event()
    threading.Timer(0.8, kit.backend.destroy_gate.set).start()  # within 0.3 + 1

    kit.broker.close_judge()

    assert not kit.broker.recovery_required
    assert kit.envs._envs[env_id].lease.state == "removed"
    kit.broker.resume_work()


def test_frozen_env_expired_meanwhile_is_removed_not_thawed(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    kit.broker.freeze_work()
    kit.clock.now += 700  # past the env's 600 s lifetime, not Work's deadline
    kit.broker.sweep_expired()
    assert ("destroy", env_id) not in kit.backend.events  # frozen: untouched
    kit.broker.resume_work()
    assert ("resume", env_id) not in kit.backend.events
    assert kit.broker.env_status(work.credential, env_id)["state"] == "removed"


# -- expiry and disk ---------------------------------------------------------------


def test_an_expired_env_is_removed_by_the_sweep_and_refunded(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    kit.clock.now += 601
    kit.sweep()
    join_reaper(kit, env_id)
    status = kit.broker.env_status(work.credential, env_id)
    assert (status["state"], status["reason"]) == ("removed", "expired")
    assert usage(kit, work)["envs_live"] == 0
    kit.sweep()  # the disk thread drops the removed env's watchdog state
    assert kit.disk.forgotten == [env_id]


def test_disk_watchdog_fails_an_env_over_quota_then_reclaims_it(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    status = kit.broker.env_status(work.credential, env_id)
    assert status["services"]["main"]["disk_mb_used"] == 3
    kit.disk.verdict = DiskVerdict(usage={}, over_quota=(env_id,))
    kit.sweep()
    join_reaper(kit, env_id)
    assert [budget.env_id for budget in kit.disk.budgets] == [env_id]
    status = kit.broker.env_status(work.credential, env_id)
    assert (status["state"], status["reason"]) == ("failed", "disk_quota")

    kit.disk.verdict = DiskVerdict(usage={}, over_quota=(env_id,), reclaim=(env_id,))
    kit.sweep()
    join_reaper(kit, env_id)
    assert kit.broker.env_status(work.credential, env_id)["state"] == "removed"


def test_a_failed_disk_probe_never_fails_the_run_closed(kit):
    work = open_work(kit)
    ready(kit, work, pull(kit, work))

    def broken(budgets):
        raise InfrastructureError("cannot measure sandbox env container disk")

    kit.disk.poll = broken
    kit.sweep()
    assert not kit.broker.recovery_required


# -- failure isolation --------------------------------------------------------------


def test_one_env_with_an_unknown_outcome_is_quarantined_alone(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    drifted = ready(kit, work, handle, "drifted")
    healthy = ready(kit, work, handle, "healthy")
    kit.backend.errors["exec_target"] = InfrastructureError(
        "recovery_required: service is not owned by its planned identity"
    )
    with pytest.raises(SandboxError, match="infrastructure.*quarantined"):
        start_exec(kit, work, drifted)
    join_reaper(kit, drifted)
    del kit.backend.errors["exec_target"]

    status = kit.broker.env_status(work.credential, drifted)
    assert (status["state"], status["reason"]) == ("failed", "quarantined")
    assert ("fail", drifted) in kit.backend.events
    with pytest.raises(SandboxError, match="busy.*quarantined"):
        start_exec(kit, work, drifted, "again")
    assert not kit.broker.recovery_required
    start_exec(kit, work, healthy, "run-healthy")
    kit.broker.env_destroy(work.credential, drifted)
    assert usage(kit, work)["envs_live"] == 1


def test_an_unresolved_start_quarantines_only_its_env(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    healthy = ready(kit, work, handle, "healthy")
    kit.backend.errors["start"] = InfrastructureError("recovery_required: start lost")
    env_id = kit.broker.env_create(work.credential, single(handle), "lost")["env_id"]
    kit.broker.env_start(work.credential, env_id, 30, "lost-start")
    join_starter(kit, env_id)
    status = kit.broker.env_status(work.credential, env_id)
    assert (status["state"], status["reason"]) == ("failed", "quarantined")
    assert not kit.broker.recovery_required
    assert kit.broker.env_status(work.credential, healthy)["state"] == "ready"


def test_a_rolled_back_create_leaves_nothing_and_no_recovery(kit):
    from rsi_harness.errors import SetupError

    work = open_work(kit)
    handle = pull(kit, work)
    kit.backend.errors["create"] = SetupError("refused; every object is proven absent")
    with pytest.raises(SandboxError, match="rolled back"):
        kit.broker.env_create(work.credential, single(handle), "refused")
    assert usage(kit, work)["envs_live"] == 0
    assert not kit.broker.recovery_required


def test_unprovable_env_removal_fails_the_run_closed(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    kit.backend.errors["destroy"] = InfrastructureError("recovery_required: remains")
    with pytest.raises(InfrastructureError, match="recovery"):
        kit.broker.env_destroy(work.credential, env_id)
    assert kit.broker.recovery_required
    assert usage(kit, work)["envs_live"] == 1  # never refunded without proof
    with pytest.raises(SandboxError, match="infrastructure"):
        kit.broker.env_create(work.credential, single("i" + "0" * 32), "next")


@pytest.mark.parametrize("step", ["plan_env", "commit_env"])
def test_a_journal_failure_fails_the_whole_run_closed(kit, monkeypatch, step):
    work = open_work(kit)
    handle = pull(kit, work)
    other = ready(kit, work, handle, "other")

    def broken(*args):
        raise OSError("journal disk full")

    monkeypatch.setattr(kit.broker.journal, step, broken)
    with pytest.raises(InfrastructureError, match="journal"):
        kit.broker.env_create(work.credential, single(handle), "broken")
    assert kit.broker.recovery_required
    if step == "plan_env":
        # Nothing reached Docker, and the reservation was returned.
        assert [op for op, _ in kit.backend.events].count("create") == 1
        assert usage(kit, work)["envs_live"] == 1
    with pytest.raises(SandboxError, match="infrastructure|busy"):
        start_exec(kit, work, other)


# -- stages, copies, images ---------------------------------------------------------


def test_stages_upload_in_frames_and_copies_charge_their_bytes(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    data = tar_of({"big.bin": b"x" * 5000})
    token = work.credential
    first = kit.broker.stage_put(token, None, 0, False, None, "f1", data[:4096])
    with pytest.raises(SandboxError, match="offset"):
        kit.broker.stage_put(token, first["stage_id"], 1, False, None, "f2", b"")
    final = kit.broker.stage_put(
        token,
        first["stage_id"],
        4096,
        True,
        hashlib.sha256(data).hexdigest(),
        "f3",
        data[4096:],
    )
    assert (final["bytes"], final["entries"]) == (len(data), 1)
    assert usage(kit, work)["upload_bytes"] == len(data)
    kit.broker.copy_in(token, env_id, "main", "/app", final["stage_id"], "copy")

    out = kit.broker.copy_out(token, env_id, "main", "/app/out.txt", 1 << 20, [])
    body = kit.broker.stage_get(token, out["stage_id"], 0, 1 << 20)
    with tarfile.open(fileobj=io.BytesIO(body)) as archive:
        assert archive.extractfile("out.txt").read() == b"result"
    assert usage(kit, work)["download_bytes"] == len(body)
    assert kit.broker.path_stat(token, env_id, "main", "/app", True)["kind"] == "dir"


def test_copies_are_refused_while_the_env_is_paused(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    stage_id = stage(kit, work, tar_of({"a": b"1"}))
    kit.broker.freeze_work()
    kit.broker.reopen_work()  # the session opens, the env stays paused
    with pytest.raises(SandboxError, match="busy"):
        kit.broker.copy_in(work.credential, env_id, "main", "/", stage_id, "c")


def test_pulls_are_registry_scoped_and_images_release_only_when_unused(kit):
    work = open_work(kit)
    with pytest.raises(SandboxError, match="permission.*registry"):
        kit.broker.image_pull(work.credential, "ghcr.io/a/b:1", "missing", "p")
    with pytest.raises(SandboxError, match="invalid"):
        kit.broker.image_pull(work.credential, "a" * 64, "missing", "p")
    kit.puller.local[BUSYBOX_REF] = image_attrs()
    handle = pull(kit, work)
    # Cached and policy "missing": no pull ran and no pull budget was spent.
    assert kit.puller.pulls == [] and usage(kit, work)["image_bytes"] == 0
    [image] = kit.broker.journal.images()
    assert (image.handle, image.state, image.pre_existing) == (handle, "present", True)
    env_id = ready(kit, work, handle)
    listed = kit.broker.image_list(work.credential)["images"]
    assert [(item["handle"], item["in_use"]) for item in listed] == [(handle, True)]
    with pytest.raises(SandboxError, match="busy"):
        kit.broker.image_release(work.credential, handle)
    kit.broker.env_destroy(work.credential, env_id)
    assert kit.broker.image_release(work.credential, handle) == {"ok": True}
    assert kit.broker.journal.images() == ()
    with pytest.raises(SandboxError, match="permission"):
        kit.broker.env_create(work.credential, single(handle), "gone")


def test_a_cancelled_pull_binds_no_handle(kit):
    work = open_work(kit)
    kit.puller.gate = threading.Event()
    job_id = kit.broker.image_pull(work.credential, BUSYBOX, "always", "p")["job_id"]
    assert kit.broker.job_cancel(work.credential, job_id)["state"] in (
        "queued",
        "running",
    )
    kit.envs.images.jobs[job_id].thread.join(5)
    view = kit.broker.job_wait(work.credential, job_id, 0, 0)
    assert (view["state"], view["error"]["kind"]) == ("canceled", "canceled")
    assert "Pull complete" in view["log"]
    assert kit.broker.image_list(work.credential) == {"images": []}
    assert kit.broker.journal.images() == ()
    assert usage(kit, work)["jobs_running"] == 0


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("busybox", ("docker.io/library/busybox", "latest")),
        ("busybox:1.37.0", ("docker.io/library/busybox", "1.37.0")),
        ("user/app:v1", ("docker.io/user/app", "v1")),
        ("ghcr.io/o/r@sha256:" + "b" * 64, ("ghcr.io/o/r", "sha256:" + "b" * 64)),
        ("localhost:5000/x", ("localhost:5000/x", "latest")),
    ],
)
def test_pull_references_take_docker_defaults(ref, expected):
    assert pull_reference(ref) == expected


# -- long-polls ----------------------------------------------------------------------


def test_long_poll_conditions_wait_without_the_broker(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    exec_id, docker = start_exec(kit, work, env_id)
    wait = {"exec_id": exec_id, "stdout_offset": 0, "stderr_offset": 0, "wait_sec": 5}
    changed = kit.broker.wait_condition(work.credential, "exec_wait", wait)
    assert changed is not None and not changed()
    docker.peer.sendall(b"\x01\x00\x00\x00\x00\x00\x00\x02hi")
    kit.io()
    assert changed()
    # Unread output answers at once.
    assert kit.broker.wait_condition(work.credential, "exec_wait", wait) is None
    assert (
        kit.broker.wait_condition(
            work.credential, "env_status", {"env_id": env_id, "wait_sec": 0}
        )
        is None
    )
    # A settled env answers at once; a starting one is waited on.
    assert (
        kit.broker.wait_condition(
            work.credential, "env_status", {"env_id": env_id, "wait_sec": 5}
        )
        is None
    )
    kit.backend.start_gate = threading.Event()
    other = kit.broker.env_create(work.credential, single(pull(kit, work, "p2")), "o")
    kit.broker.env_start(work.credential, other["env_id"], 30, "o-start")
    status = kit.broker.wait_condition(
        work.credential, "env_status", {"env_id": other["env_id"], "wait_sec": 5}
    )
    assert not status()
    kit.backend.start_gate.set()
    join_starter(kit, other["env_id"])
    assert status()
    with pytest.raises(SandboxError, match="invalid.*wait_sec"):
        kit.broker.wait_condition(
            work.credential, "env_status", {"env_id": env_id, "wait_sec": 31}
        )


# -- review regressions --------------------------------------------------------------


def with_run_limits(kit, **limits):
    grant = kit.broker.grant
    environments = grant.environments
    run_limits = environments.run_limits.model_copy(update=limits)
    kit.broker.grant = grant.model_copy(
        update={
            "environments": environments.model_copy(update={"run_limits": run_limits})
        }
    )


def test_run_limits_bound_work_and_every_judge_round_together(kit):
    with_run_limits(kit, max_envs_created=3)
    work = open_work(kit)
    handle = pull(kit, work)
    first = ready(kit, work, handle, "one")
    ready(kit, work, handle, "two")
    assert kit.envs._run_usage["envs_live"] == 2
    kit.broker.env_destroy(work.credential, first)
    # Live usage comes back to the run too; created envs never do.
    assert kit.envs._run_usage["envs_live"] == 1
    assert kit.envs._run_usage["envs_created"] == 2
    kit.broker.freeze_work()

    judge = open_round(kit)
    judge_handle = pull(kit, judge, "judge-pull")
    ready(kit, judge, judge_handle, "judge-one")
    with pytest.raises(SandboxError) as caught:
        kit.broker.env_create(judge.credential, single(judge_handle), "judge-two")
    # Judge's own phase limit (400) is far away: the run's budget refused it.
    assert (caught.value.code, caught.value.field) == ("quota", "max_envs_created")
    assert usage(kit, judge)["envs_created"] == 1


def test_request_ids_replay_and_conflict_after_their_object_ended(kit, monkeypatch):
    work = open_work(kit)
    handle = pull(kit, work)
    created = kit.broker.env_create(work.credential, single(handle), "same")
    kit.broker.env_destroy(work.credential, created["env_id"])

    assert kit.broker.env_create(work.credential, single(handle), "same") == created
    with pytest.raises(SandboxError) as caught:
        kit.broker.env_create(work.credential, single(handle, cpus=1), "same")
    assert (caught.value.code, caught.value.field) == ("invalid", "request_id")
    assert [op for op, _ in kit.backend.events].count("create") == 1
    assert usage(kit, work)["envs_created"] == 1

    # A consumed stage's upload request replays too.
    env_id = ready(kit, work, handle, "env")
    data = tar_of({"a": b"1"})
    stage_id = stage(kit, work, data, "put")
    kit.broker.copy_in(work.credential, env_id, "main", "/", stage_id, "in")
    assert stage(kit, work, data, "put") == stage_id

    # Tombstones are bounded: the oldest request_id then counts as new
    # (here refused by the created-env budget, not as a conflicting reuse).
    monkeypatch.setattr(sandbox_envs, "MAX_TOMBSTONES", 1)
    kit.broker.env_destroy(work.credential, env_id)
    assert "same" not in kit.broker._sessions["work"].tombstones
    ready(kit, work, handle, "third")
    with pytest.raises(SandboxError) as caught:
        kit.broker.env_create(work.credential, single(handle, cpus=1), "same")
    assert (caught.value.code, caught.value.field) == ("quota", "max_envs_created")


def test_submit_waits_for_a_work_copy_in_flight(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    stage_id = stage(kit, work, tar_of({"a": b"1"}))
    transfer = kit.transfers[0]
    transfer.gate = threading.Event()
    copy = threading.Thread(
        target=kit.broker.copy_in,
        args=(work.credential, env_id, "main", "/", stage_id, "c"),
    )
    copy.start()
    deadline = time.monotonic() + 5
    while not kit.envs._envs[env_id].uses:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    try:
        with pytest.raises(RetryableSubmissionError):
            kit.broker.freeze_work()
        assert not kit.broker._sessions["work"].frozen
        assert ("pause", env_id) not in kit.backend.events
    finally:
        transfer.gate.set()
        copy.join(5)
    kit.broker.freeze_work()
    assert ("pause", env_id) in kit.backend.events


def test_submit_waits_for_a_work_exec_start_in_flight(kit):
    # runc refuses to start an exec in a paused container: freezing before
    # the start lands would fail a command the caller was told is running.
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    gate, entered = threading.Event(), threading.Event()
    exec_target = kit.backend.exec_target

    def gated(lease, service):
        entered.set()
        gate.wait(5)
        return exec_target(lease, service)

    kit.backend.exec_target = gated
    start = threading.Thread(target=start_exec, args=(kit, work, env_id))
    start.start()
    assert entered.wait(5)
    try:
        with pytest.raises(RetryableSubmissionError):
            kit.broker.freeze_work()
        assert not kit.broker._sessions["work"].frozen
        assert ("pause", env_id) not in kit.backend.events
    finally:
        gate.set()
        start.join(5)
    assert not kit.envs._envs[env_id].uses
    kit.broker.freeze_work()
    assert ("pause", env_id) in kit.backend.events


def test_an_env_that_cannot_pause_at_freeze_is_killed_not_the_run(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    stuck = ready(kit, work, handle, "stuck")
    kit.backend.errors["pause"] = InfrastructureError("cannot pause")
    kit.broker.freeze_work()
    assert not kit.broker.recovery_required
    assert kit.envs._envs[stuck].quarantined
    lease = kit.envs._envs[stuck].lease
    assert (lease.state, lease.reason) == ("failed", "quarantined")
    assert ("fail", stuck) in kit.backend.events


def test_an_env_that_can_be_neither_paused_nor_killed_fails_closed(kit):
    work = open_work(kit)
    stuck = ready(kit, work, pull(kit, work))
    kit.backend.errors["pause"] = InfrastructureError("cannot pause")
    kit.backend.errors["fail"] = InfrastructureError("recovery_required: kill")
    with pytest.raises(InfrastructureError, match="recovery"):
        kit.broker.freeze_work()
    assert kit.broker.recovery_required
    # The failed freeze re-pauses (best effort) what it can.
    assert [op for op, env in kit.backend.events if env == stuck].count("pause") == 2


def test_a_pull_past_its_deadline_times_out_and_binds_nothing(kit):
    open_work(kit)
    kit.broker.freeze_work()
    judge = open_round(kit, deadline=200)
    kit.puller.gate = threading.Event()
    job_id = kit.broker.image_pull(judge.credential, BUSYBOX, "always", "slow")[
        "job_id"
    ]
    kit.clock.now = 201  # the round's deadline, which bounds its pulls
    kit.sweep()
    job = kit.envs.images.jobs[job_id]
    job.thread.join(5)
    assert (job.state, job.error["kind"]) == ("timed_out", "timeout")
    assert kit.envs.images.images == {} and kit.broker.journal.images() == ()


def test_unconsumed_stages_expire_after_their_ttl(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    stage_id = stage(kit, work, tar_of({"a": b"1"}))
    out = kit.broker.copy_out(work.credential, env_id, "main", "/out.txt", 1 << 20, [])
    kit.clock.now += STAGE_TTL_SEC
    kit.sweep()
    for expired in (stage_id, out["stage_id"]):
        with pytest.raises(SandboxError, match="unknown or expired stage"):
            kit.broker.stage_get(work.credential, expired, 0, 10)
    assert kit.envs._spools[work].prepaid == {}


def test_the_hard_floor_removes_an_env_with_disk_quota(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    kit.disk.verdict = DiskVerdict(usage={}, hard_floor=(env_id,))
    kit.sweep()
    join_reaper(kit, env_id)
    status = kit.broker.env_status(work.credential, env_id)
    assert (status["state"], status["reason"]) == ("removed", "disk_quota")
    assert not kit.broker.recovery_required


def test_host_floors_refuse_before_any_reservation_or_journal_write(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    env_id = ready(kit, work, handle)
    spent = usage(kit, work)
    kit.disk.full = True
    for call in (
        lambda: kit.broker.env_create(work.credential, single(handle), "full"),
        lambda: kit.broker.image_pull(work.credential, BUSYBOX, "always", "full"),
    ):
        with pytest.raises(SandboxError) as caught:
            call()
        assert caught.value.code == "quota"
    kit.disk.full = False
    kit.spool_full.append(True)
    for call in (
        lambda: stage(kit, work, tar_of({"a": b"1"}), "full-stage"),
        lambda: kit.broker.copy_out(work.credential, env_id, "main", "/o", 10, []),
    ):
        with pytest.raises(SandboxError) as caught:
            call()
        assert caught.value.code == "quota"
    assert usage(kit, work) == spent
    assert [env.env_id for env in kit.broker.journal.envs()] == [env_id]
    assert len(kit.broker.journal.images()) == 1


def test_an_unresolved_create_removes_its_env_without_failing_the_run(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    kit.backend.errors["create"] = InfrastructureError("recovery_required: lost")
    with pytest.raises(SandboxError) as caught:
        kit.broker.env_create(work.credential, single(handle), "lost")
    assert (caught.value.code, caught.value.field) == ("infrastructure", "spec")
    [env_id] = kit.envs._envs
    assert ("destroy", env_id) in kit.backend.events
    assert usage(kit, work)["envs_live"] == 0
    assert kit.broker.journal.envs() == ()
    assert not kit.broker.recovery_required


def test_an_unresolved_create_whose_removal_fails_fails_closed(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    kit.backend.errors["create"] = InfrastructureError("recovery_required: lost")
    kit.backend.errors["destroy"] = InfrastructureError("recovery_required: kept")
    with pytest.raises(InfrastructureError, match="recovery"):
        kit.broker.env_create(work.credential, single(handle), "lost")
    assert kit.broker.recovery_required


def test_work_ending_with_a_retained_env_blocks_a_new_work_session(kit):
    work = open_work(kit)
    ready(kit, work, pull(kit, work))
    kit.broker.cancel_work()
    with pytest.raises(InfrastructureError, match="requires recovery"):
        kit.broker.open_session(work.owner, None)


def test_containment_pauses_work_envs_and_fails_closed_when_it_cannot(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    kit.broker.contain_work()
    assert kit.envs._envs[env_id].lease.state == "paused"
    assert not kit.broker.recovery_required

    kit.backend.errors["pause"] = InfrastructureError("cannot pause")
    with pytest.raises(InfrastructureError, match="containment"):
        kit.broker.contain_work()
    assert kit.broker.recovery_required


def test_broker_close_removes_every_live_env(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    first = ready(kit, work, handle, "one")
    kit.broker.freeze_work()
    judge = open_round(kit)
    second = ready(kit, judge, pull(kit, judge, "judge-pull"))
    kit.broker.close()
    assert {env for op, env in kit.backend.events if op == "destroy"} == {
        first,
        second,
    }
    assert kit.broker.journal.envs() == () and kit.broker.journal.images() == ()
    assert not kit.broker.recovery_required


def test_cancellation_removes_live_envs_with_reason_canceled(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    kit.broker.cancel_run()  # its own sweep begins the removal
    join_reaper(kit, env_id)
    lease = kit.envs._envs[env_id].lease
    assert (lease.state, lease.reason) == ("removed", "canceled")


def test_a_failed_closed_run_still_expires_envs_that_are_not_paused(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    kit.broker._fail_closed()  # an unrelated journal hiccup
    kit.clock.now += 700
    kit.sweep()
    join_reaper(kit, env_id)
    lease = kit.envs._envs[env_id].lease
    assert (lease.state, lease.reason) == ("removed", "expired")


def test_housekeeping_errors_never_fail_the_run_closed(kit, monkeypatch):
    work = open_work(kit)
    ready(kit, work, pull(kit, work))
    stage(kit, work, tar_of({"a": b"1"}))
    spool = kit.envs._spools[work]

    def broken():
        raise PermissionError("stage unlink refused")

    monkeypatch.setattr(spool.stages, "sweep", broken)
    kit.sweep()
    assert not kit.broker.recovery_required


@pytest.mark.parametrize("operation", ["exec", "stop"])
def test_a_transient_inspect_failure_leaves_the_env_alone(kit, operation):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    transient = InfrastructureError(
        "cannot inspect sandbox env service main: timed out"
    )
    if operation == "exec":
        kit.backend.errors["exec_target"] = transient
        with pytest.raises(SandboxError) as caught:
            start_exec(kit, work, env_id)
    else:
        kit.backend.errors["stop_service"] = transient
        with pytest.raises(SandboxError) as caught:
            kit.broker.env_stop_service(work.credential, env_id, "main", 1, "stop")
    assert caught.value.code == "infrastructure"
    assert "quarantined" not in caught.value.message
    join_reaper(kit, env_id)
    assert kit.broker.env_status(work.credential, env_id)["state"] == "ready"
    assert ("fail", env_id) not in kit.backend.events
    assert not kit.envs._envs[env_id].quarantined


@pytest.mark.parametrize(
    "repo_digests",
    [[], ["registry.example/other/busybox@sha256:" + "d" * 64]],
)
def test_a_cached_image_without_registry_provenance_is_never_bound(kit, repo_digests):
    work = open_work(kit)
    kit.puller.local["docker.io/library/private-local:dev"] = image_attrs(
        "sha256:" + "f" * 64, repo_digests=repo_digests
    )
    job_id = kit.broker.image_pull(
        work.credential, "private-local:dev", "missing", "p"
    )["job_id"]
    kit.envs.images.jobs[job_id].thread.join(5)
    view = kit.broker.job_wait(work.credential, job_id, 0, 0)
    # A real anonymous pull ran instead, and the registry has no such image.
    assert kit.puller.pulls == [("docker.io/library/private-local", "dev")]
    assert (view["state"], view["error"]["kind"]) == ("failed", "registry")
    assert kit.broker.image_list(work.credential) == {"images": []}


def test_a_cached_digest_reference_needs_exactly_that_digest(kit):
    work = open_work(kit)
    reference = "docker.io/library/busybox@sha256:" + "e" * 64
    kit.puller.local[reference] = image_attrs()  # provenance for digest d...d
    job_id = kit.broker.image_pull(work.credential, reference, "missing", "p")["job_id"]
    kit.envs.images.jobs[job_id].thread.join(5)
    assert kit.puller.pulls == [("docker.io/library/busybox", "sha256:" + "e" * 64)]


def test_an_oversize_pull_spends_the_budget_so_no_further_pull_fits(kit):
    work = open_work(kit)
    kit.puller.registry[BUSYBOX_REF] = image_attrs(size=10**13)
    job_id = kit.broker.image_pull(work.credential, BUSYBOX, "missing", "big")["job_id"]
    kit.envs.images.jobs[job_id].thread.join(5)
    view = kit.broker.job_wait(work.credential, job_id, 0, 0)
    assert (view["state"], view["error"]["kind"]) == ("failed", "quota")
    assert usage(kit, work)["image_bytes"] == 10**13
    with pytest.raises(SandboxError) as caught:
        kit.broker.image_pull(work.credential, "busybox:1.36", "always", "next")
    assert (caught.value.code, caught.value.field) == ("quota", "max_pull_mb")
    # A "missing" pull is admitted, but one that would download never starts.
    kit.puller.registry["docker.io/library/busybox:1.36"] = image_attrs()
    job_id = kit.broker.image_pull(work.credential, "busybox:1.36", "missing", "m")[
        "job_id"
    ]
    kit.envs.images.jobs[job_id].thread.join(5)
    view = kit.broker.job_wait(work.credential, job_id, 0, 0)
    assert (view["state"], view["error"]["kind"]) == ("failed", "quota")
    assert kit.puller.pulls == [("docker.io/library/busybox", "1.37.0")]
    assert kit.broker.image_list(work.credential) == {"images": []}
    assert usage(kit, work)["jobs_running"] == 0


@pytest.mark.parametrize("ref", [BUSYBOX, "busybox@sha256:" + "d" * 64])
def test_a_cached_image_is_bound_after_the_pull_budget_is_spent(kit, ref):
    """By tag or by digest, a cached image with registry provenance needs no
    download, so policy "missing" binds it whatever budget is left."""
    work = open_work(kit)
    kit.puller.local[BUSYBOX_REF] = image_attrs()
    kit.puller.local["docker.io/library/busybox@sha256:" + "d" * 64] = image_attrs()
    kit.envs._run_usage["pull_bytes"] = 40960 * MIB  # the run's max_pull_mb
    handle = pull(kit, work, ref=ref)
    assert kit.puller.pulls == []
    assert usage(kit, work)["image_bytes"] == 0
    [image] = kit.broker.journal.images()
    assert (image.handle, image.state, image.pre_existing) == (handle, "present", True)
    ready(kit, work, handle)
    # The same image by policy "always" would contact the registry: refused.
    with pytest.raises(SandboxError) as caught:
        kit.broker.image_pull(work.credential, ref, "always", "again")
    assert (caught.value.code, caught.value.field) == ("quota", "max_pull_mb")


def test_a_pull_announcing_more_than_the_budget_stops_before_it_lands(kit):
    work = open_work(kit)
    kit.puller.announce = 10**13
    job_id = kit.broker.image_pull(work.credential, BUSYBOX, "missing", "big")["job_id"]
    kit.envs.images.jobs[job_id].thread.join(5)
    view = kit.broker.job_wait(work.credential, job_id, 0, 0)
    assert (view["state"], view["error"]["kind"]) == ("failed", "quota")
    assert BUSYBOX_REF not in kit.puller.local
    assert usage(kit, work)["image_bytes"] == 10**13


def test_copy_out_stages_are_charged_when_made_not_when_read(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    token = work.credential
    stages = [
        kit.broker.copy_out(token, env_id, "main", "/out.txt", 10**12, [])
        for _ in range(3)
    ]
    size = kit.envs._spools[work].prepaid[stages[0]["stage_id"]]
    assert usage(kit, work)["download_bytes"] == 3 * size
    body = kit.broker.stage_get(token, stages[0]["stage_id"], 0, 1 << 20)
    kit.broker.stage_get(token, stages[0]["stage_id"], 0, 1 << 20)  # re-read
    assert len(body) == size and usage(kit, work)["download_bytes"] == 3 * size

    # The spool never holds more unread copy_out bytes than the budget left.
    kit.envs._run_usage["download_bytes"] = (
        kit.broker.grant.environments.run_limits.max_download_bytes - size + 1
    )
    with pytest.raises(SandboxError) as caught:
        kit.broker.copy_out(token, env_id, "main", "/out.txt", 10**12, [])
    assert (caught.value.code, caught.value.field) == ("quota", "max_download_bytes")
    assert len(kit.envs._spools[work].prepaid) == 3


# -- allowlist envs -------------------------------------------------------------------

ALLOW_TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["allowlist", "none"]
pull = true
[metadata.rsi_harness.sandbox.environments.judge]
network = ["none"]
pull = true
"""
ALLOW_BOUNDS = {
    "max_entries": 3,
    "patterns": ["pypi.org", "*.pypi.org", "8.8.0.0/16", "10.0.0.0/8"],
    "private_cidrs": ["10.1.0.0/16"],
    "refresh_sec": 30.0,
}


@pytest.fixture
def allow_kit(tmp_path):
    grant = make_env_grant(tmp_path, make_env_task(ALLOW_TASK), allowlist=ALLOW_BOUNDS)
    made = build_kit(tmp_path, grant)
    yield made
    close_kit(made)


def allowlisted(handle, *entries):
    return {**single(handle), "network": "allowlist", "allowlist": list(entries)}


def refresh_turn(kit, seconds):
    kit.clock.now += seconds
    kit.sweep()
    kit.envs.join_refresh(5)


def test_the_grant_reports_the_operator_allowlist_bounds(allow_kit):
    work = open_work(allow_kit)

    granted = allow_kit.broker.capabilities(work.credential)["environments"]
    assert granted["network"] == ["allowlist", "none"]
    assert granted["allowlist"] == ALLOW_BOUNDS
    assert "allowlist" not in granted["limits"]


@pytest.mark.parametrize(
    ("entries", "code", "field"),
    (
        (
            ("pypi.org", "files.pypi.org", "8.8.8.8", "8.8.4.4"),
            "quota",
            "spec.allowlist",
        ),
        (("pypi.org", "example.com"), "permission", "spec.allowlist.1"),
        (("1.1.1.1",), "permission", "spec.allowlist.0"),
        # Matches a pattern, but only 10.1/16 is an approved private range.
        (("10.2.0.1",), "permission", "spec.allowlist.0"),
    ),
    ids=("max-entries", "pattern-hostname", "pattern-ip", "private-range"),
)
def test_allowlist_entries_are_bounded_by_the_operator(allow_kit, entries, code, field):
    work = open_work(allow_kit)
    handle = pull(allow_kit, work)

    with pytest.raises(SandboxError) as caught:
        allow_kit.broker.env_create(
            work.credential, allowlisted(handle, *entries), "refused"
        )

    assert (caught.value.code, caught.value.field) == (code, field)
    assert allow_kit.envs._envs == {}
    assert not any(event[0] == "create" for event in allow_kit.backend.events)


def test_allowlist_needs_the_phase_grant(allow_kit):
    judge = open_round(allow_kit)
    handle = pull(allow_kit, judge)
    # The judge did not request it: no bounds, whatever the operator offers.
    granted = allow_kit.broker.capabilities(judge.credential)["environments"]
    assert (granted["network"], granted["allowlist"]) == (["none"], None)

    with pytest.raises(SandboxError) as caught:
        allow_kit.broker.env_create(
            judge.credential, allowlisted(handle, "pypi.org"), "refused"
        )

    assert (caught.value.code, caught.value.field) == ("permission", "spec.network")


def test_an_allowlist_env_is_planned_with_the_private_cidrs_and_reports_notes(
    allow_kit,
):
    work = open_work(allow_kit)
    handle = pull(allow_kit, work)
    allow_kit.backend.allowlist_notes_result = ("allowlist entry x allows nothing",)

    created = allow_kit.broker.env_create(
        work.credential, allowlisted(handle, "PyPI.org:443", "10.1.2.3"), "env"
    )

    plan = allow_kit.envs._envs[created["env_id"]].plan
    assert plan.network.allowlist == ("pypi.org:443", "10.1.2.3")
    assert plan.network.private_cidrs == ("10.1.0.0/16",)
    assert created["notes"] == ["allowlist entry x allows nothing"]


def test_allowlist_envs_are_refreshed_every_refresh_sec_off_the_watchdog(allow_kit):
    work = open_work(allow_kit)
    handle = pull(allow_kit, work)
    allowed = ready(allow_kit, work, handle, "allowed", allowlisted(handle, "pypi.org"))
    ready(allow_kit, work, handle, "none")

    refresh_turn(allow_kit, 29)
    assert allow_kit.backend.refreshes == []
    refresh_turn(allow_kit, 1)
    assert allow_kit.backend.refreshes == [allowed]
    refresh_turn(allow_kit, 10)
    assert allow_kit.backend.refreshes == [allowed]

    # A slow resolution holds no broker lock and blocks no watchdog turn.
    gate = allow_kit.backend.gates["refresh"] = threading.Event()
    refresh_turn(allow_kit, 0)  # starts nothing: not due yet
    allow_kit.clock.now += 20
    allow_kit.sweep()
    assert allow_kit.broker.env_status(work.credential, allowed)["state"] == "ready"
    gate.set()
    allow_kit.envs.join_refresh(5)
    assert allow_kit.backend.refreshes == [allowed, allowed]


def test_a_failed_allow_chain_update_quarantines_only_that_env(allow_kit):
    work = open_work(allow_kit)
    handle = pull(allow_kit, work)
    allowed = ready(allow_kit, work, handle, "allowed", allowlisted(handle, "pypi.org"))
    other = ready(allow_kit, work, handle, "other", allowlisted(handle, "8.8.8.8"))
    allow_kit.backend.errors["refresh"] = InfrastructureError("iptables denied")

    refresh_turn(allow_kit, 30)
    for env_id in (allowed, other):
        join_reaper(allow_kit, env_id)

    for env_id in (allowed, other):
        status = allow_kit.broker.env_status(work.credential, env_id)
        assert (status["state"], status["reason"]) == ("failed", "quarantined")
    assert not allow_kit.broker.recovery_required
    del allow_kit.backend.errors["refresh"]
    # A quarantined env is never refreshed again.
    refresh_turn(allow_kit, 30)
    assert allow_kit.backend.refreshes == [allowed, other]


# -- operator tools ------------------------------------------------------------------

TMUX_BINARY = b"\x7fELF static tmux"
TMUX_PATH = "/usr/local/bin/tmux"


@pytest.fixture
def tmux_kit(tmp_path):
    binary = tmp_path / "tmux"
    binary.write_bytes(TMUX_BINARY)
    tmux = {"path": str(binary), "sha256": hashlib.sha256(TMUX_BINARY).hexdigest()}
    # Its own root: a test may also use the default kit.
    (tmp_path / "tmux-kit").mkdir()
    grant = make_env_grant(tmp_path, make_env_task(TASK), tmux=tmux)
    made = build_kit(tmp_path / "tmux-kit", grant)
    yield made
    close_kit(made)


def transfer_of(kit, session):
    with kit.broker._lock:
        return kit.envs._spool(kit.envs._session(session.credential))


def test_the_operator_tmux_is_copied_only_where_none_is(tmux_kit):
    kit = tmux_kit
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    assert kit.broker.capabilities(work.credential)["environments"]["tools"] == ["tmux"]
    spool = transfer_of(kit, work)
    spool.transfer.absent.add(TMUX_PATH)
    before = usage(kit, work)

    result = kit.broker.tool_install(work.credential, env_id, "main", "tmux")

    assert result == {"tool": "tmux", "path": TMUX_PATH, "installed": True}
    stat, copy = spool.transfer.calls
    assert stat == ("path_stat", (env_id, "main"), TMUX_PATH, False)
    assert copy[:3] == ("copy_in", (env_id, "main"), "/usr/local/bin")
    with tarfile.open(fileobj=io.BytesIO(copy[3])) as archive:
        [member] = archive.getmembers()
        assert (member.name, member.mode, member.uid) == ("tmux", 0o755, 0)
        assert archive.extractfile(member).read() == TMUX_BINARY
    after = usage(kit, work)
    # One operation; the operator's bytes are no upload of the caller's.
    assert after["operations"] == before["operations"] + 1
    assert after["upload_bytes"] == before["upload_bytes"] == 0
    assert list(spool.stages._root.iterdir()) == []

    # Present now (or shipped by the image): never replaced.
    spool.transfer.absent.clear()
    result = kit.broker.tool_install(work.credential, env_id, "main", "tmux")
    assert result == {"tool": "tmux", "path": TMUX_PATH, "installed": False}
    assert [call[0] for call in spool.transfer.calls] == [
        "path_stat",
        "copy_in",
        "path_stat",
    ]


def test_a_changed_operator_tmux_fails_closed_and_copies_nothing(tmux_kit, tmp_path):
    kit = tmux_kit
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    spool = transfer_of(kit, work)
    spool.transfer.absent.add(TMUX_PATH)
    (tmp_path / "tmux").write_bytes(TMUX_BINARY + b" swapped")

    with pytest.raises(SandboxError) as caught:
        kit.broker.tool_install(work.credential, env_id, "main", "tmux")

    assert (caught.value.code, caught.value.field) == ("infrastructure", "tool")
    assert "sha256" in caught.value.message
    assert [call[0] for call in spool.transfer.calls] == ["path_stat"]
    assert list(spool.stages._root.iterdir()) == []
    assert not kit.broker.recovery_required
    assert kit.broker.env_status(work.credential, env_id)["state"] == "ready"


def test_tool_install_needs_the_operator_key_and_an_owned_env(kit, tmux_kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    assert kit.broker.capabilities(work.credential)["environments"]["tools"] == []
    with pytest.raises(SandboxError) as caught:
        kit.broker.tool_install(work.credential, env_id, "main", "tmux")
    assert (caught.value.code, caught.value.field) == ("permission", "tool")
    assert transfer_of(kit, work).transfer.calls == []

    work = open_work(tmux_kit)
    env_id = ready(tmux_kit, work, pull(tmux_kit, work))
    with pytest.raises(SandboxError, match="invalid: tool"):
        tmux_kit.broker.tool_install(work.credential, env_id, "main", "vim")
    tmux_kit.broker.freeze_work()
    judge = open_round(tmux_kit)
    with pytest.raises(SandboxError) as caught:
        tmux_kit.broker.tool_install(judge.credential, env_id, "main", "tmux")
    assert (caught.value.code, caught.value.field) == ("permission", "env_id")
