"""Builder and built-image recovery: a crash at any step converges to nothing.

The builder is made (or removed) by the real BuilderBackend over the fake
Engine of test_sandbox_build, with tmpfs or loop-ext4 state (a fake command
runner), and the process "dies" just before any one event: a journal
commit, a rule install, a bridge, volume, mkfs, losetup or container call,
a start or a removal. Recovery then runs on the last durable journal. Real
Docker and kill -9 are in tests/integration/test_sandbox_build_docker.py.
"""

import tempfile
from pathlib import Path

import pytest

from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager, ResourceLease
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_build import BuilderBackend, built_image_labels
from rsi_harness.runtime.sandbox_buildfs import (
    LoopExt4StateFs,
    TmpfsStateFs,
    remove_loop_file,
)
from rsi_harness.runtime.sandbox_contracts import SandboxReservation
from rsi_harness.runtime.sandbox_env_contracts import (
    SandboxImageLease,
    builder_loop_file,
    built_image_tag,
)
from tests.runtime.test_sandbox_build import (
    BUILDER_ID,
    BuildWorld,
    EngineAPI,
    grant,
    owner,
)
from tests.runtime.test_sandbox_buildfs import Runner
from tests.runtime.test_sandbox_env_recovery import (
    Crash,
    CrashingEvents,
    EnvRecoveryBackend,
)

RESERVATION = SandboxReservation(
    cpus=4,
    memory_mb=4096,
    pool_cpus=8,
    pool_memory_mb=8192,
    disk_mb=2048,
    pool_disk_mb=40960,
)
IMAGE_ID = "sha256:" + "7" * 64
HANDLE = "i" + "5" * 32


class World(BuildWorld):
    """A BuildWorld whose builders use loop-ext4 or tmpfs state."""

    def __init__(self, tmp_path, state_fs):
        super().__init__()
        self.state_fs = state_fs
        self.data = tmp_path / "managed"
        self.runner = Runner(self.events)
        api = EngineAPI(self.engine)
        if state_fs == "loop-ext4":
            statefs = LoopExt4StateFs(api, self.data, runner=self.runner)
        else:
            statefs = TmpfsStateFs(api)
        self.backend = BuilderBackend(
            self.network.client,
            self.network.backend,
            lambda kind: statefs,
            cgroup_root=None,
            sleep=lambda seconds: None,
            kill_proof_sec=1.0,
        )

    def arm(self, after):
        """Die just before the ``after``-th event from now (None: never)."""
        at = None if after is None else len(self.events) + after
        events = CrashingEvents(self.events, at)
        self.events = self.network.events = events
        self.network.firewall.events = events
        self.runner.events = events
        return events


class BuildRecoveryBackend(EnvRecoveryBackend):
    """EnvRecoveryBackend plus loop files and the image store."""

    def remove_builder_loop(self, path):
        self.world.events.append(("remove-builder-loop", str(path)))
        remove_loop_file(path, self.world.runner)

    def inspect_image(self, reference):
        attrs = self.engine.images.get(reference)
        return None if attrs is None else self._image(attrs)

    def image_in_use(self, image_id):
        return any(
            attrs["Image"] == image_id for attrs in self.engine.containers.values()
        )

    def remove_image(self, reference):
        try:
            self.engine.remove_image(reference)
        except Exception as error:
            if type(error).__name__ != "NotFound":
                raise


def resource_lease(**updates):
    values = dict(
        run_id="run-1",
        task_id="task",
        coordinator_pid=4321,
        coordinator_started_at=50.0,
        phase="agent_running",
        sandbox_reservation=RESERVATION,
        sandbox_env_authority=True,
    )
    values.update(updates)
    return ResourceLease(**values)


def journaled(tmp_path, lease=None):
    store = LeaseStore(tmp_path / "leases")
    store.write(lease or resource_lease())

    def mutate(transform):
        current = store.read("run-1")
        updated = transform(current)
        if updated is not current:
            store.write(updated)
        return store.read("run-1")

    return store, SandboxJournal(mutate)


def committing(world, journal):
    def commit(lease):
        world.events.append(("commit", lease.state))
        journal.commit_builder(lease)
        return lease

    return commit


def recover(tmp_path, world, store, **options):
    backend = BuildRecoveryBackend(world, **options)
    manager = RecoveryManager(store=store, backend=backend, managed_root=world.data)
    return manager, backend


def assert_nothing_left(world):
    assert world.engine.containers == {}
    assert world.engine.volumes == {}
    assert world.network.store == {}
    assert world.network.firewall.installed == {}
    assert world.runner.attached == {}
    build = builder_loop_file(world.data, "run-1", BUILDER_ID).parent
    assert not build.exists()


def create(world, journal):
    build = grant(state_fs=world.state_fs).environments.work.build
    plan = world.backend.plan(owner(), BUILDER_ID, build)
    journal.plan_builder(plan.lease())
    commit = committing(world, journal)
    return lambda: world.backend.create(plan, plan.lease(), commit)


def destroy(world, journal):
    step = create(world, journal)
    lease = step()
    commit = committing(world, journal)
    return lambda: world.backend.destroy(lease, commit)


SCENARIOS = {"create": create, "destroy": destroy}


def _events(name, state_fs):
    """The number of events of one uninterrupted scenario step."""
    with tempfile.TemporaryDirectory() as root:
        world = World(Path(root), state_fs)
        _, journal = journaled(Path(root))
        step = SCENARIOS[name](world, journal)
        before = len(world.events)
        step()
        return len(world.events) - before


# Every event of each scenario, plus the finished step.
CRASHES = [
    (name, state_fs, after)
    for name in SCENARIOS
    for state_fs in ("tmpfs", "loop-ext4")
    for after in range(_events(name, state_fs) + 1)
]


@pytest.mark.parametrize(("scenario", "state_fs", "after"), CRASHES)
def test_a_crash_at_any_builder_step_converges_to_all_removed(
    tmp_path, scenario, state_fs, after
):
    world = World(tmp_path, state_fs)
    store, journal = journaled(tmp_path)
    step = SCENARIOS[scenario](world, journal)
    world.arm(after)
    try:
        step()
    except Crash:
        pass
    world.arm(None)
    crashed = store.read("run-1")
    manager, backend = recover(tmp_path, world, store)
    start = len(world.events)
    manager.recover("run-1")
    assert_nothing_left(world)
    events = world.events[start:]
    if state_fs == "loop-ext4" and crashed.sandbox_builders:
        # Per builder (spec order): container, volume, loop device and
        # file, then bridge and rule; the directory sweep is only a backstop.
        path = str(builder_loop_file(world.data, "run-1", BUILDER_ID))
        loop = events.index(("remove-builder-loop", path))
        kinds = [event[0] for event in events]
        assert all(
            index < loop
            for index, kind in enumerate(kinds)
            if kind in ("container-remove", "volume-remove")
        )
        assert all(
            index > loop
            for index, kind in enumerate(kinds)
            if kind in ("network-remove", "remove")
        )
    recovered = store.read("run-1")
    assert recovered.sandbox_builders == ()
    assert not recovered.recovery_required
    assert recovered.sandbox_reservation is None
    # Only a builder still planned (its creates in flight) is settled first.
    pending = [b for b in crashed.sandbox_builders if b.state == "planned"]
    assert backend.settlements == (
        [tuple(b.rule_id for b in pending if b.network_id is None)] if pending else []
    )


def test_a_pending_builder_with_a_live_coordinator_fails_closed(tmp_path):
    world = World(tmp_path, "tmpfs")
    store, journal = journaled(tmp_path)
    step = create(world, journal)
    world.arm(2)  # after the probe and the rule, before the bridge
    with pytest.raises(Crash):
        step()
    world.arm(None)
    manager, _ = recover(tmp_path, world, store, alive=True)
    with pytest.raises(RuntimeError, match="coordinator may still run"):
        manager.recover("run-1")
    retained = store.read("run-1")
    assert retained.recovery_required and retained.sandbox_builders
    assert world.network.firewall.installed  # the rule is kept


def test_a_foreign_container_holding_the_builder_name_is_never_removed(tmp_path):
    world = World(tmp_path, "tmpfs")
    store, journal = journaled(tmp_path)
    create(world, journal)()
    [lease] = journal.builders()
    world.engine.containers[lease.container_id]["Config"]["Labels"] = {
        "rsi-harness.run-id": "someone-else"
    }
    manager, _ = recover(tmp_path, world, store)
    with pytest.raises(RuntimeError, match="not owned by its planned identity"):
        manager.recover("run-1")
    assert lease.container_id in world.engine.containers


def test_a_builder_volume_with_other_options_is_not_ours(tmp_path):
    world = World(tmp_path, "tmpfs")
    store, journal = journaled(tmp_path)
    create(world, journal)()
    [lease] = journal.builders()
    world.engine.volumes[lease.volume_name]["Options"] = {
        "type": "none",
        "o": "bind",
        "device": "/",
    }
    manager, _ = recover(tmp_path, world, store)
    with pytest.raises(RuntimeError, match="volume .* not owned"):
        manager.recover("run-1")
    assert lease.volume_name in world.engine.volumes


def test_unjournaled_builder_objects_of_the_run_are_swept(tmp_path):
    world = World(tmp_path, "loop-ext4")
    store, journal = journaled(tmp_path)
    create(world, journal)()
    # The record left the journal (a proven removal a late create outran).
    store.write(store.read("run-1").model_copy(update={"sandbox_builders": ()}))
    manager, _ = recover(tmp_path, world, store)
    manager.recover("run-1")
    assert_nothing_left(world)


def built(state, *, image_id=IMAGE_ID, handle=HANDLE):
    return SandboxImageLease(
        owner=owner(),
        handle=handle,
        kind="built",
        image_id=image_id,
        tag=built_image_tag("run-1", handle),
        state=state,
        bytes=4096,
    )


def add_image(world, image_id=IMAGE_ID, *, tags=(), handle=HANDLE, labels=None):
    attrs = {
        "Id": image_id,
        "RepoTags": list(tags),
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {
            "Labels": built_image_labels(owner(), handle) if labels is None else labels
        },
    }
    world.engine.images[image_id] = attrs
    for tag in tags:
        world.engine.images[tag] = attrs


@pytest.mark.parametrize(
    ("state", "tagged", "present"),
    [
        # Journaled before the load stream ended: never registered (B8).
        ("planned", False, False),
        ("loading", False, False),
        # The load finished before the crash: found by its config digest.
        ("loading", False, True),
        ("loading", True, True),
        ("present", True, True),
        ("leaked", True, True),
    ],
)
def test_built_images_converge_by_journaled_digest_and_tag(
    tmp_path, state, tagged, present
):
    world = World(tmp_path, "tmpfs")
    lease = built(state, image_id=None if state == "planned" else IMAGE_ID)
    if present:
        add_image(world, tags=(lease.tag,) if tagged else ())
    store, _ = journaled(tmp_path, resource_lease(sandbox_images=(lease,)))
    manager, _ = recover(tmp_path, world, store)
    manager.recover("run-1")
    assert IMAGE_ID not in world.engine.images and lease.tag not in world.engine.images
    recovered = store.read("run-1")
    assert recovered.sandbox_images == () and recovered.sandbox_reservation is None


@pytest.mark.parametrize("state", ["loading", "present", "leaked"])
@pytest.mark.parametrize("labels", [{}, {"rsi-harness.run-id": "run-2"}])
def test_a_journaled_digest_is_removed_without_the_label_sweep(tmp_path, state, labels):
    """Steps 7 and 8: the journaled ID alone names the image (a dangling
    one included), whatever labels it carries."""
    world = World(tmp_path, "tmpfs")
    lease = built(state)
    tags = () if state == "loading" else (lease.tag,)
    add_image(world, tags=tags, labels=labels)
    store, _ = journaled(tmp_path, resource_lease(sandbox_images=(lease,)))
    manager, _ = recover(tmp_path, world, store)
    manager.recover("run-1")
    assert IMAGE_ID not in world.engine.images and lease.tag not in world.engine.images
    assert store.read("run-1").sandbox_images == ()


def test_an_image_with_the_runs_tag_prefix_is_swept_without_labels(tmp_path):
    """Step 9: the run's ``rsi-sbx-img`` tag prefix, labels or not."""
    world = World(tmp_path, "tmpfs")
    orphan = "sha256:" + "6" * 64
    add_image(
        world, orphan, tags=(built_image_tag("run-1", "i" + "6" * 32),), labels={}
    )
    other = "sha256:" + "4" * 64
    add_image(world, other, tags=(built_image_tag("run-2", "i" + "6" * 32),), labels={})
    # No builder or built-image record: every recovery with env authority
    # sweeps (a load may have registered after its record was settled).
    store, _ = journaled(tmp_path)
    manager, _ = recover(tmp_path, world, store)
    manager.recover("run-1")
    assert orphan not in world.engine.images
    assert other in world.engine.images  # another run's tag prefix


def test_an_unjournaled_image_with_the_runs_build_labels_is_swept(tmp_path):
    world = World(tmp_path, "tmpfs")
    orphan = "sha256:" + "6" * 64
    add_image(world, orphan, tags=(built_image_tag("run-1", "i" + "6" * 32),))
    foreign = "sha256:" + "5" * 64
    add_image(world, foreign, labels={"rsi-harness.run-id": "run-2"})
    store, _ = journaled(tmp_path, resource_lease(sandbox_images=(built("removed"),)))
    manager, _ = recover(tmp_path, world, store)
    manager.recover("run-1")
    assert orphan not in world.engine.images
    assert foreign in world.engine.images  # another run's image is left alone


def test_a_built_image_still_used_by_a_container_fails_closed(tmp_path):
    world = World(tmp_path, "tmpfs")
    lease = built("leaked")
    add_image(world, tags=(lease.tag,))
    world.engine.containers["c" * 64] = {
        "Id": "c" * 64,
        "Name": "/foreign",
        "Image": IMAGE_ID,
        "Config": {"Labels": {}},
        "State": {"Running": True, "Paused": False},
        "Mounts": [],
    }
    store, _ = journaled(tmp_path, resource_lease(sandbox_images=(lease,)))
    manager, _ = recover(tmp_path, world, store)
    with pytest.raises(RuntimeError, match="still used"):
        manager.recover("run-1")
    retained = store.read("run-1")
    assert retained.recovery_required
    assert retained.sandbox_reservation is not None
    assert IMAGE_ID in world.engine.images


def test_a_labelled_image_with_a_foreign_tag_fails_closed(tmp_path):
    world = World(tmp_path, "tmpfs")
    add_image(world, tags=("ubuntu:24.04",))
    store, _ = journaled(tmp_path, resource_lease(sandbox_images=(built("removed"),)))
    manager, _ = recover(tmp_path, world, store)
    with pytest.raises(RuntimeError, match="foreign tag"):
        manager.recover("run-1")
    assert IMAGE_ID in world.engine.images


def test_a_loop_file_left_in_the_build_directory_is_detached_and_removed(tmp_path):
    world = World(tmp_path, "loop-ext4")
    path = builder_loop_file(world.data, "run-1", "b" + "9" * 32)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    world.runner.attached["/dev/loop5"] = str(path)
    (path.parent / "notes.txt").write_text("x")
    store, _ = journaled(tmp_path)
    manager, _ = recover(tmp_path, world, store)
    with pytest.raises(RuntimeError, match="foreign entry notes.txt"):
        manager.recover("run-1")
    (path.parent / "notes.txt").unlink()
    manager.recover("run-1")
    assert world.runner.attached == {} and not path.parent.exists()
