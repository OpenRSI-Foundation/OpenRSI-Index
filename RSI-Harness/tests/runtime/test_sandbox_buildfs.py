"""Builder state filesystems: exact command order and rollback at every step.

The loop-ext4 path needs root (mkfs, losetup, a volume on a loop device), so
it runs here against a fake command runner and a fake Engine; the real
ENOSPC behaviour is the operator check's (spec 8, item 4).
"""

import copy

import pytest
from docker.errors import NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.sandbox_buildfs import (
    CommandFailed,
    LoopExt4StateFs,
    TmpfsStateFs,
    builder_volume_labels,
    loop_devices,
    remove_loop_file,
)
from rsi_harness.runtime.sandbox_env_contracts import builder_loop_file
from tests.runtime.test_sandbox_env_docker import daemon_error
from tests.runtime.test_sandbox_policy_v2 import make_builder_lease

MIB = 1024**2
BUILDER = "b" + "3" * 32


def planned(**updates):
    values = dict(
        state="planned",
        loop_device=None,
        network_id=None,
        container_id=None,
        pending_mutation=True,
        disk_mb=64,
    )
    values.update(updates)
    return make_builder_lease(BUILDER, **values)


class Runner:
    """losetup and mkfs.ext4 over an in-memory loop table."""

    def __init__(self, events):
        self.events = events
        self.attached = {}  # device -> backing file
        self.fail = {}  # command name -> exception
        self.show = None

    def __call__(self, argv):
        argv = list(argv)
        self.events.append(("run", *argv))
        key = argv[0] if argv[0] != "losetup" else " ".join(argv[:2])
        if key in self.fail:
            raise self.fail[key]
        if argv[0] == "mkfs.ext4":
            return ""
        if argv[1] == "--find":
            if self.show is not None:
                return self.show + "\n"  # malformed output, nothing attached
            device = f"/dev/loop{len(self.attached) + 7}"
            self.attached[device] = argv[3]
            return device + "\n"
        if argv[1] == "-j":
            return "".join(
                f"{device}: []: ({path})\n"
                for device, path in self.attached.items()
                if path == argv[2]
            )
        if argv[1] == "-d":
            self.attached.pop(argv[2], None)
            return ""
        raise AssertionError(argv)


class Volumes:
    """The Engine's volume half, recording every call."""

    def __init__(self, events):
        self.events = events
        self.volumes = {}
        self.create_error = None
        self.drift = lambda attrs: None
        self.remove_error = None

    def inspect_volume(self, name):
        if name not in self.volumes:
            raise NotFound(name)
        return copy.deepcopy(self.volumes[name])

    def create_volume(self, name, driver=None, driver_opts=None, labels=None):
        self.events.append(("volume-create", name, dict(driver_opts)))
        if self.create_error is not None:
            raise self.create_error
        attrs = {
            "Name": name,
            "Driver": driver,
            "Labels": dict(labels),
            "Options": dict(driver_opts),
            "Scope": "local",
        }
        self.drift(attrs)
        self.volumes[name] = attrs
        return attrs

    def remove_volume(self, name, force=False):
        assert force is False
        self.events.append(("volume-remove", name))
        if self.remove_error is not None:
            raise self.remove_error
        self.volumes.pop(name, None)


@pytest.fixture
def world(tmp_path):
    events = []
    runner, api = Runner(events), Volumes(events)
    journal = []

    def commit(lease):
        events.append(("commit", lease.loop_device))
        journal.append(lease)
        return lease

    fs = LoopExt4StateFs(api, tmp_path / "data", runner=runner)
    return fs, runner, api, commit, events, journal, tmp_path / "data"


def test_loop_ext4_is_file_mkfs_losetup_journal_then_volume(world):
    fs, runner, api, commit, events, journal, data = world
    lease = fs.create(planned(), commit)
    path = builder_loop_file(data, "run-1", BUILDER)
    assert path == data / "run-1" / "sb" / "build" / f"{BUILDER[1:17]}.img"
    # Sparse: the size is the bound, not an allocation.
    info = path.stat()
    assert info.st_size == 64 * MIB and info.st_blocks == 0
    assert oct(info.st_mode & 0o777) == "0o600"
    assert (path.parent.stat().st_mode & 0o777) == 0o700
    assert events == [
        (
            "run",
            "mkfs.ext4",
            "-q",
            "-F",
            "-m",
            "0",
            "-E",
            "lazy_itable_init=1,lazy_journal_init=1",
            str(path),
        ),
        ("run", "losetup", "--find", "--show", str(path)),
        # The device is journaled before any volume can name it.
        ("commit", "/dev/loop7"),
        (
            "volume-create",
            lease.volume_name,
            {"type": "ext4", "device": "/dev/loop7"},
        ),
    ]
    assert lease.loop_device == "/dev/loop7"
    assert api.volumes[lease.volume_name]["Labels"] == builder_volume_labels(lease)
    fs.attest(lease)


@pytest.mark.parametrize(
    ("step", "failure"),
    [
        ("mkfs.ext4", CommandFailed("mkfs.ext4 exited 1")),
        ("losetup --find", CommandFailed("losetup exited 1")),
        ("show", None),
        ("volume", daemon_error("volume create denied", 403)),
        ("drift", None),
    ],
)
def test_every_failed_step_rolls_back_in_reverse(world, step, failure):
    fs, runner, api, commit, events, journal, data = world
    if step in ("mkfs.ext4", "losetup --find"):
        runner.fail[step] = failure
    elif step == "show":
        runner.show = "/tmp/evil"  # not a loop device
    elif step == "volume":
        api.create_error = failure
    else:
        api.drift = lambda attrs: attrs["Options"].update(o="bind")
    with pytest.raises(SetupError, match="proven absent"):
        fs.create(planned(), commit)
    path = builder_loop_file(data, "run-1", BUILDER)
    assert not path.exists()
    assert runner.attached == {}
    assert api.volumes == {}
    if journal:
        # A journaled device is cleared again once it is detached.
        assert journal[-1].loop_device is None


def test_a_file_already_at_the_planned_path_is_never_reused(world):
    """O_EXCL: a leftover file is neither truncated nor formatted."""
    fs, runner, api, commit, events, journal, data = world
    path = builder_loop_file(data, "run-1", BUILDER)
    path.parent.mkdir(mode=0o700, parents=True)
    path.write_bytes(b"keep")
    with pytest.raises(SetupError, match="proven absent"):
        fs.create(planned(), commit)
    assert not any(event[:2] == ("run", "mkfs.ext4") for event in events)
    assert not any(event[:3] == ("run", "losetup", "--find") for event in events)
    assert not path.exists() or path.read_bytes() == b"keep"
    assert runner.attached == {} and api.volumes == {}


def test_an_unanswered_volume_create_retains_everything(world):
    fs, runner, api, commit, events, journal, data = world
    api.create_error = OSError("connection reset")
    with pytest.raises(InfrastructureError, match="recovery_required"):
        fs.create(planned(), commit)
    # The daemon may still make the volume: nothing is rolled back.
    assert builder_loop_file(data, "run-1", BUILDER).exists()
    assert list(runner.attached) == ["/dev/loop7"]


def test_removal_is_volume_then_every_loop_device_then_the_file(world):
    fs, runner, api, commit, events, journal, data = world
    lease = fs.create(planned(), commit)
    path = builder_loop_file(data, "run-1", BUILDER)
    # A second attachment the journal never saw (a crash before a commit).
    runner.attached["/dev/loop9"] = str(path)
    del events[:]
    removed = fs.remove(lease, commit)
    assert [event[:2] for event in events] == [
        ("volume-remove", lease.volume_name),
        ("run", "losetup"),
        ("run", "losetup"),
        ("run", "losetup"),
        ("run", "losetup"),
        ("commit", None),
    ]
    assert runner.attached == {} and not path.exists()
    assert removed.loop_device is None
    # Idempotent.
    fs.remove(removed, commit)


def test_a_device_that_stays_attached_is_unproven(world):
    fs, runner, api, commit, events, journal, data = world
    lease = fs.create(planned(), commit)
    runner.fail["losetup -d"] = CommandFailed("busy")
    with pytest.raises(InfrastructureError, match="stays attached"):
        fs.remove(lease, commit)
    assert builder_loop_file(data, "run-1", BUILDER).exists()


def test_a_foreign_volume_holding_the_planned_name_is_never_removed(world):
    fs, runner, api, commit, events, journal, data = world
    lease = planned()
    api.volumes[lease.volume_name] = {
        "Name": lease.volume_name,
        "Driver": "local",
        "Labels": {"owner": "someone-else"},
        "Options": {},
        "Scope": "local",
    }
    with pytest.raises(InfrastructureError, match="not owned"):
        fs.remove(lease, commit)
    assert lease.volume_name in api.volumes


def test_recovery_detaches_a_file_found_by_losetup_j(tmp_path):
    events = []
    runner = Runner(events)
    path = tmp_path / "sb" / "build" / "0123456789abcdef.img"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    runner.attached = {"/dev/loop3": str(path), "/dev/loop4": "/elsewhere.img"}
    assert loop_devices(path, runner) == ("/dev/loop3",)
    remove_loop_file(path, runner)
    assert runner.attached == {"/dev/loop4": "/elsewhere.img"}
    assert not path.exists()
    remove_loop_file(path, runner)  # idempotent
    link = tmp_path / "sb" / "build" / "fedcba9876543210.img"
    link.symlink_to(tmp_path / "outside")
    with pytest.raises(InfrastructureError, match="symlink"):
        remove_loop_file(link, runner)


def test_tmpfs_is_the_same_volume_path_without_a_device(tmp_path):
    events = []
    api = Volumes(events)
    fs = TmpfsStateFs(api)
    lease = planned(state_fs="tmpfs", loop_device=None)
    assert fs.create(lease, lambda lease: lease) == lease
    assert events == [
        (
            "volume-create",
            lease.volume_name,
            {"type": "tmpfs", "device": "tmpfs", "o": "size=64m"},
        )
    ]
    fs.attest(lease)
    fs.remove(lease, lambda lease: lease)
    assert api.volumes == {}
    with pytest.raises(InfrastructureError, match="not tmpfs"):
        fs.create(planned(), lambda lease: lease)
