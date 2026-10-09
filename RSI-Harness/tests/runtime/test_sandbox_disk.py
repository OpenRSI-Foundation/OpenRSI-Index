"""Env disk: soft limits measured by the Engine, reclaimed after grace, host floors."""

from types import SimpleNamespace

import pytest

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_disk import (
    DiskWatchdog,
    DockerDiskProbe,
    EnvDiskBudget,
    EnvDiskUsage,
)
from tests.sandbox_helpers import FakeClock

MIB = 1024**2
ENV_A = "e" + "a" * 32
ENV_B = "e" + "b" * 32


class Probe:
    def __init__(self):
        self.containers = {}
        self.volumes = {}
        self.free = 100_000 * MIB
        self.calls = []

    def container_sizes(self, run_id):
        self.calls.append(("containers", run_id))
        return {env: dict(sizes) for env, sizes in self.containers.items()}

    def volume_sizes(self, run_id):
        self.calls.append(("volumes", run_id))
        return {env: dict(sizes) for env, sizes in self.volumes.items()}

    def host_free_bytes(self):
        self.calls.append(("host", None))
        return self.free


def watchdog(probe, clock, **options):
    values = dict(run_id="run-1", floor_mb=10_240, hard_floor_mb=4096, clock=clock)
    values.update(options)
    return DiskWatchdog(probe, **values)


def kinds(probe):
    return [kind for kind, _ in probe.calls]


def test_each_measurement_runs_on_its_own_cadence():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock)
    budgets = [EnvDiskBudget(ENV_A, 64)]

    dog.poll(budgets)
    assert kinds(probe) == ["containers", "volumes", "host"]
    probe.calls.clear()
    for _ in range(30):
        clock.now += 1
        dog.poll(budgets)
    # 30 s: containers every 10 s, host every 2 s, volumes not yet (60 s).
    assert kinds(probe).count("containers") == 3
    assert kinds(probe).count("host") == 15
    assert "volumes" not in kinds(probe)
    clock.now += 30
    dog.poll(budgets)
    assert kinds(probe).count("volumes") == 1
    assert {run for kind, run in probe.calls if kind != "host"} == {"run-1"}


def test_env_over_its_soft_limit_fails_then_is_reclaimed_after_grace():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock, container_interval=1.0)
    budgets = [EnvDiskBudget(ENV_A, 64), EnvDiskBudget(ENV_B, 64)]
    probe.containers = {ENV_A: {"main": 40 * MIB}, ENV_B: {"main": 10 * MIB}}
    probe.volumes = {ENV_A: {"rsi-sbvol-a-0": 20 * MIB}}

    verdict = dog.poll(budgets)
    assert verdict.over_quota == ()
    assert verdict.usage[ENV_A].total == 60 * MIB
    assert verdict.usage[ENV_A].service_mb("main") == 40

    # Writable layer plus env volumes; the next container poll sees it.
    probe.containers[ENV_A]["main"] = 256 * MIB
    clock.now += 0.5
    assert dog.poll(budgets).over_quota == ()
    clock.now += 0.5
    verdict = dog.poll(budgets)
    assert (verdict.over_quota, verdict.reclaim) == ((ENV_A,), ())
    clock.now += 59
    assert dog.poll(budgets).reclaim == ()
    clock.now += 1
    verdict = dog.poll(budgets)
    assert (verdict.over_quota, verdict.reclaim) == ((ENV_A,), (ENV_A,))
    # A removed env is forgotten, and its state with it.
    verdict = dog.poll([EnvDiskBudget(ENV_B, 64)])
    assert verdict.over_quota == () and set(verdict.usage) == {ENV_B}


@pytest.mark.parametrize(("extra", "over"), ((0, False), (1, True)))
def test_soft_limit_is_exactly_disk_mb(extra, over):
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock)
    probe.containers = {ENV_A: {"main": 40 * MIB}}
    probe.volumes = {ENV_A: {"rsi-sbvol-a-0": 24 * MIB + extra}}
    verdict = dog.poll([EnvDiskBudget(ENV_A, 64)])
    assert verdict.over_quota == ((ENV_A,) if over else ())


def test_an_unsized_object_keeps_its_last_sample_and_is_reported():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock, container_interval=1.0, volume_interval=1.0)
    budgets = [EnvDiskBudget(ENV_A, 64), EnvDiskBudget(ENV_B, 64)]
    probe.containers = {ENV_A: {"main": 10 * MIB}, ENV_B: {"main": None}}
    probe.volumes = {ENV_A: {"rsi-sbvol-a-0": 70 * MIB}}
    verdict = dog.poll(budgets)
    assert verdict.over_quota == (ENV_A,)
    # Never measured: 0 for now, but the broker is told.
    assert verdict.unmeasured == (ENV_B,)

    # The Engine cannot size the volume (-1): its last sample still counts.
    probe.volumes = {ENV_A: {"rsi-sbvol-a-0": None}}
    probe.containers = {ENV_A: {"main": 10 * MIB}, ENV_B: {"main": MIB}}
    clock.now += 1
    verdict = dog.poll(budgets)
    assert verdict.usage[ENV_A].total == 80 * MIB
    assert verdict.unmeasured == (ENV_A,)


def test_a_failed_probe_is_retried_on_the_next_poll():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock)
    budgets = [EnvDiskBudget(ENV_A, 64)]
    dog.poll(budgets)
    clock.now += 10
    probe.containers = {ENV_A: {"main": 100 * MIB}}
    measure = probe.container_sizes

    def busy(run_id):
        raise InfrastructureError("engine busy")

    probe.container_sizes = busy
    with pytest.raises(InfrastructureError):
        dog.poll(budgets)
    probe.container_sizes = measure
    clock.now += 0.1
    # Not a full interval later: the failure left the sample due.
    assert dog.poll(budgets).over_quota == (ENV_A,)


def test_per_container_cap_is_enforced_even_under_the_env_total():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock)
    probe.containers = {ENV_A: {"main": 30 * MIB, "db": 30 * MIB}}
    assert dog.poll([EnvDiskBudget(ENV_A, 100, container_mb=32)]).over_quota == ()
    dog.forget(ENV_A)
    probe.containers[ENV_A]["db"] = 33 * MIB
    clock.now += 10
    assert dog.poll([EnvDiskBudget(ENV_A, 100, container_mb=32)]).over_quota == (ENV_A,)


def test_below_the_hard_floor_the_largest_env_goes_first_one_per_sample():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock)
    budgets = [EnvDiskBudget(ENV_A, 10_000), EnvDiskBudget(ENV_B, 10_000)]
    probe.containers = {ENV_A: {"main": 100 * MIB}, ENV_B: {"main": 900 * MIB}}
    probe.free = 3000 * MIB

    verdict = dog.poll(budgets)
    assert verdict.hard_floor == (ENV_B,)
    assert verdict.refuse_new and verdict.host_free_mb == 3000
    clock.now += 1
    # Same host sample: its removal has not been observed yet.
    assert dog.poll(budgets).hard_floor == ()
    clock.now += 1
    assert dog.poll(budgets).hard_floor == (ENV_A,)
    probe.free = 5000 * MIB
    clock.now += 2
    verdict = dog.poll(budgets)
    assert verdict.hard_floor == () and verdict.refuse_new


def test_admission_refuses_below_the_floor():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock)
    probe.free = 10_240 * MIB
    dog.admit()
    probe.free -= 1
    clock.now += 2
    with pytest.raises(SandboxError) as caught:
        dog.admit()
    assert caught.value.code == "quota"


def test_admission_may_also_keep_the_request_above_the_floor():
    probe, clock = Probe(), FakeClock()
    dog = watchdog(probe, clock)
    probe.free = 12_000 * MIB
    dog.admit(1000)
    with pytest.raises(SandboxError) as caught:
        dog.admit(2000, field="spec.disk_mb")
    assert (caught.value.code, caught.value.field) == ("quota", "spec.disk_mb")
    # A stale sample is refreshed before deciding.
    probe.free = 20_000 * MIB
    clock.now += 2
    dog.admit(2000)
    with pytest.raises(ValueError):
        watchdog(probe, clock, floor_mb=10, hard_floor_mb=11)


class DockerAPI:
    def __init__(self):
        self.listed = []

    def containers(self, *, all, size, filters):
        assert all is True and size is True
        self.listed.append(filters)
        return [
            {
                "Labels": {
                    "rsi-harness.sandbox-env": ENV_A,
                    "rsi-harness.sandbox-service": "main",
                },
                "SizeRw": 5 * MIB,
            },
            {
                "Labels": {
                    "rsi-harness.sandbox-env": ENV_A,
                    "rsi-harness.sandbox-service": "db",
                },
            },
            {
                "Labels": {
                    "rsi-harness.sandbox-env": ENV_A,
                    "rsi-harness.sandbox-service": "odd",
                },
                "SizeRw": -1,
            },
            {"Labels": {"rsi-harness.sandbox-env": ENV_B}, "SizeRw": 1},
        ]

    def _url(self, path):
        return path

    def _get(self, url, params):
        assert (url, params) == ("/system/df", {"type": "volume"})
        return "response"

    def _result(self, response, json):
        run = {"rsi-harness.run-id": "run-1", "rsi-harness.role": "sandbox-env-vol"}
        return {
            "Volumes": [
                {
                    "Name": "rsi-sbvol-a-0",
                    "Labels": {**run, "rsi-harness.sandbox-env": ENV_A},
                    "UsageData": {"Size": 7 * MIB, "RefCount": 1},
                },
                {
                    "Name": "rsi-sbvol-a-1",
                    "Labels": {**run, "rsi-harness.sandbox-env": ENV_A},
                    "UsageData": {"Size": -1, "RefCount": 0},
                },
                {
                    "Name": "other-run",
                    "Labels": {
                        **run,
                        "rsi-harness.run-id": "run-2",
                        "rsi-harness.sandbox-env": ENV_A,
                    },
                    "UsageData": {"Size": 99},
                },
                {"Name": "unlabelled", "Labels": None, "UsageData": {"Size": 5}},
            ]
        }


def test_docker_probe_reads_engine_measurements_by_exact_labels():
    api = DockerAPI()
    probe = DockerDiskProbe(
        SimpleNamespace(api=api),
        docker_root="/var/lib/docker",
        statvfs=lambda path: SimpleNamespace(f_bavail=10, f_frsize=4096),
    )
    # A zero SizeRw is omitted; a negative or non-integer one, like
    # UsageData.Size -1, is unknown and never free.
    assert probe.container_sizes("run-1") == {
        ENV_A: {"main": 5 * MIB, "db": 0, "odd": None}
    }
    assert api.listed == [
        {"label": ["rsi-harness.run-id=run-1", "rsi-harness.role=sandbox-env"]}
    ]
    assert probe.volume_sizes("run-1") == {
        ENV_A: {"rsi-sbvol-a-0": 7 * MIB, "rsi-sbvol-a-1": None}
    }
    assert probe.host_free_bytes() == 40960
    usage = EnvDiskUsage(ENV_A, {"main": 1}, {"v": 2})
    assert usage.total == 3


def test_unmeasurable_disk_is_an_infrastructure_error():
    def denied(path):
        raise PermissionError("denied")

    probe = DockerDiskProbe(
        SimpleNamespace(api=SimpleNamespace()), docker_root="/x", statvfs=denied
    )
    for measure in (
        lambda: probe.container_sizes("run-1"),
        lambda: probe.volume_sizes("run-1"),
        probe.host_free_bytes,
    ):
        with pytest.raises(InfrastructureError):
            measure()
