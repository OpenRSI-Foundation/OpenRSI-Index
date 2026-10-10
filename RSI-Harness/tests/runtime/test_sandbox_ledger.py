"""The host pull ledger and prune-images: only images the sandbox first
brought to the host are ever removed, and never while anything uses them."""

import json
import logging
import multiprocessing
import stat
import threading
import time
from datetime import UTC, datetime

import pytest
from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime import sandbox_ledger
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_ledger import (
    DROPPED,
    KEPT,
    REMOVED,
    WOULD_DROP,
    WOULD_REMOVE,
    ImagePruner,
    PullLedger,
    pull_ledger_root,
)
from tests.runtime.test_sandbox_env_recovery import resource_lease
from tests.runtime.test_sandbox_envs import (
    BUSYBOX,
    BUSYBOX_ID,
    BUSYBOX_REF,
    image_attrs,
    kit,  # noqa: F401 - the broker fixture
    open_work,
    pull,
)
from tests.runtime.test_sandbox_policy_v2 import make_image_lease, pulled_image_lease

ID_A = "sha256:" + "a" * 64
ID_B = "sha256:" + "b" * 64
REPO = "public.ecr.aws/docker/library/busybox"
REF_A = REPO + ":1.36.1"
DAY = 86400.0


class Clock:
    def __init__(self, now=1_900_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def ledger_at(tmp_path, clock=None, **options):
    return PullLedger(
        pull_ledger_root(tmp_path / "data"), clock=clock or Clock(), **options
    )


def record(
    ledger,
    image_id=ID_A,
    ref=REF_A,
    *,
    pre_existing=False,
    before=None,
    run_id="run-1",
):
    repository, _, tag = ref.rpartition(":")
    ledger.record(
        image_id,
        repository,
        tag,
        pre_existing=pre_existing,
        before=before,
        run_id=run_id,
    )


# -- the ledger ----------------------------------------------------------------


def test_a_first_pull_becomes_an_entry_with_lease_store_modes(tmp_path):
    clock = Clock()
    ledger = ledger_at(tmp_path, clock)
    record(ledger)

    [entry] = ledger.entries()
    assert entry.model_dump(mode="json") == {
        "image_id": ID_A,
        "references": [REF_A],
        "registry": "public.ecr.aws",
        "pre_existing": False,
        "run_id": "run-1",
        "last_run_id": "run-1",
        "first_pulled_at": "2030-03-17T17:46:40Z",
        "last_used_at": "2030-03-17T17:46:40Z",
    }
    assert ledger.root == (tmp_path / "data" / "sandbox-images").resolve()
    assert stat.S_IMODE(ledger.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600
    assert json.loads(ledger.path.read_text())["schema_version"] == 1


def test_an_image_already_on_the_host_never_gets_an_entry(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, pre_existing=True)
    assert ledger.entries() == ()
    assert not ledger.path.exists()


def test_later_pulls_refresh_the_entry_and_add_only_references_they_placed(
    tmp_path,
):
    clock = Clock()
    ledger = ledger_at(tmp_path, clock)
    record(ledger)
    first = ledger.entries()[0].first_pulled_at

    clock.now += DAY
    # A cached reference (pre-existing now) refreshes last_used_at only.
    record(ledger, pre_existing=True, before=ID_A, run_id="run-2")
    # An operator's own tag of the same image is never adopted.
    record(ledger, ref="docker.io/library/mine:v1", pre_existing=True, before=ID_A)
    clock.now += DAY
    # A tag this pull put on the host is added.
    record(ledger, ref=REPO + ":latest", pre_existing=True, run_id="run-3")

    [entry] = ledger.entries()
    assert entry.references == (REF_A, REPO + ":latest")
    assert (entry.run_id, entry.last_run_id) == ("run-1", "run-3")
    assert entry.first_pulled_at == first
    assert entry.last_used_at == datetime.fromtimestamp(clock.now, UTC)


def test_a_pull_that_moves_a_name_records_it_only_when_the_name_is_ours(
    tmp_path,
):
    ledger = ledger_at(tmp_path)
    python = "docker.io/library/python:3.13-slim-bookworm"
    # Policy "always" after the registry moved a tag an operator pulled by
    # hand: the new image gets an entry, the operator's name does not.
    record(ledger, ID_B, python, before="sha256:" + "0" * 64)
    # The sandbox's own tag moved: the new image takes it over.
    record(ledger)
    id_c = "sha256:" + "c" * 64
    record(ledger, id_c, before=ID_A, run_id="run-2")

    entries = {entry.image_id: entry.references for entry in ledger.entries()}
    assert entries == {ID_B: (), ID_A: (REF_A,), id_c: (REF_A,)}


def test_the_last_use_never_moves_back(tmp_path):
    clock = Clock()
    ledger = ledger_at(tmp_path, clock)
    record(ledger)
    clock.now -= 60
    record(ledger)
    [entry] = ledger.entries()
    assert entry.last_used_at == entry.first_pulled_at


def test_references_and_entries_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_ledger, "MAX_REFERENCES", 2)
    monkeypatch.setattr(sandbox_ledger, "MAX_ENTRIES", 1)
    ledger = ledger_at(tmp_path)
    for tag in ("1", "2", "3"):
        record(ledger, ref=f"{REPO}:{tag}")
    assert ledger.entries()[0].references == (REPO + ":1", REPO + ":2")
    with pytest.raises(ValueError, match="full"):
        record(ledger, image_id=ID_B)


def _record_many(root, worker, count):
    ledger = PullLedger(root, lock_wait=60)
    for index in range(count):
        image_id = "sha256:" + f"{worker:02x}{index:04x}".ljust(64, "0")
        ledger.record(
            image_id,
            REPO,
            f"w{worker}-{index}",
            pre_existing=False,
            before=None,
            run_id="run-1",
        )


def test_concurrent_processes_and_threads_never_lose_an_update(tmp_path):
    root = pull_ledger_root(tmp_path / "data")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_record_many, args=(root, worker, 15))
        for worker in range(3)
    ]
    threads = [
        threading.Thread(target=_record_many, args=(root, worker, 15))
        for worker in range(3, 6)
    ]
    for item in (*processes, *threads):
        item.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    for thread in threads:
        thread.join(60)
    assert len(PullLedger(root).entries()) == 6 * 15


def test_a_record_waits_for_the_ledger_lock(tmp_path):
    ledger = ledger_at(tmp_path, lock_wait=10)
    done = threading.Event()

    def pull():
        record(ledger)
        done.set()

    with ledger._locked():
        thread = threading.Thread(target=pull)
        thread.start()
        assert not done.wait(0.3)
    thread.join(5)
    assert done.is_set() and len(ledger.entries()) == 1


def test_a_record_gives_up_on_a_lock_held_past_its_bound(tmp_path):
    ledger = ledger_at(tmp_path, lock_wait=0.2)
    with ledger._locked():
        started = time.monotonic()
        with pytest.raises(InfrastructureError, match="busy"):
            record(ledger)
        assert time.monotonic() - started < 2
    assert ledger.entries() == ()


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"schema_version": 2, "images": []}),
        # A reference not in the ledger's normalized form.
        json.dumps(
            {
                "schema_version": 1,
                "images": [
                    {
                        "image_id": ID_A,
                        "references": ["busybox:1.36.1"],
                        "registry": "docker.io",
                        "pre_existing": False,
                        "run_id": "run-1",
                        "last_run_id": "run-1",
                        "first_pulled_at": "2030-03-17T17:46:40Z",
                        "last_used_at": "2030-03-17T17:46:40Z",
                    }
                ],
            }
        ),
    ],
)
def test_an_unreadable_ledger_is_never_overwritten(tmp_path, content):
    ledger = ledger_at(tmp_path)
    ledger.root.mkdir(parents=True)
    ledger.path.write_text(content)
    with pytest.raises(InfrastructureError, match="unreadable"):
        record(ledger, image_id=ID_B)
    with pytest.raises(InfrastructureError, match="unreadable"):
        ledger.entries()
    assert ledger.path.read_text() == content


def test_forget_keeps_an_entry_a_pull_used_again_meanwhile(tmp_path):
    clock = Clock()
    ledger = ledger_at(tmp_path, clock)
    record(ledger)
    record(ledger, image_id=ID_B, ref=REPO + ":b")
    seen = {entry.image_id: entry.last_used_at for entry in ledger.entries()}
    clock.now += 60
    record(ledger, image_id=ID_B, ref=REPO + ":b", pre_existing=True, before=ID_B)
    ledger.forget(seen)
    assert [entry.image_id for entry in ledger.entries()] == [ID_B]


# -- brokered pulls ------------------------------------------------------------


def test_only_a_pull_that_brings_the_image_records_it(kit, tmp_path):  # noqa: F811
    ledger = ledger_at(tmp_path)
    kit.envs.images.ledger = ledger
    work = open_work(kit)
    pull(kit, work)
    [entry] = ledger.entries()
    assert (entry.image_id, entry.references, entry.run_id) == (
        BUSYBOX_ID,
        (BUSYBOX_REF,),
        "run-1",
    )
    # Cached now: the next pull only refreshes it.
    pull(kit, work, request_id="again")
    assert ledger.entries()[0].references == (BUSYBOX_REF,)


def test_a_pull_of_a_cached_image_records_nothing(kit, tmp_path):  # noqa: F811
    ledger = ledger_at(tmp_path)
    kit.envs.images.ledger = ledger
    kit.puller.local[BUSYBOX_REF] = image_attrs()
    work = open_work(kit)
    pull(kit, work)
    assert kit.broker.journal.images()[0].pre_existing
    assert ledger.entries() == ()


@pytest.mark.parametrize(
    "name",
    [
        # docker pull busybox@sha256:...: the operator's digest reference only.
        "docker.io/library/busybox@sha256:" + "d" * 64,
        # A dangling image: no reference at all.
        BUSYBOX_ID,
    ],
)
def test_an_image_already_on_the_host_under_another_name_records_nothing(
    kit,  # noqa: F811
    tmp_path,
    name,
):
    ledger = ledger_at(tmp_path)
    kit.envs.images.ledger = ledger
    kit.puller.local[name] = image_attrs()
    work = open_work(kit)
    pull(kit, work)
    # Pulled (the reference was absent), yet the image ID was on the host.
    assert kit.puller.pulls and not kit.broker.journal.images()[0].pre_existing
    assert ledger.entries() == ()


def test_a_pull_that_moves_an_operators_name_never_records_that_name(
    kit,  # noqa: F811
    tmp_path,
):
    ledger = ledger_at(tmp_path)
    kit.envs.images.ledger = ledger
    kit.puller.local[BUSYBOX_REF] = image_attrs(ID_B)
    work = open_work(kit)
    job_id = kit.broker.image_pull(work.credential, BUSYBOX, "always", "p")["job_id"]
    kit.envs.images.jobs[job_id].thread.join(5)
    assert kit.broker.job_wait(work.credential, job_id, 0, 0)["state"] == "succeeded"
    [entry] = ledger.entries()
    assert (entry.image_id, entry.references) == (BUSYBOX_ID, ())


def test_a_failed_pull_records_nothing(kit, tmp_path):  # noqa: F811
    ledger = ledger_at(tmp_path)
    kit.envs.images.ledger = ledger
    work = open_work(kit)
    job_id = kit.broker.image_pull(work.credential, "busybox:0.0", "missing", "p")[
        "job_id"
    ]
    kit.envs.images.jobs[job_id].thread.join(5)
    assert kit.broker.job_wait(work.credential, job_id, 0, 0)["state"] == "failed"
    assert ledger.entries() == ()


class BrokenLedger:
    def record(self, *args, **kwargs):
        raise PermissionError("/root/secret/path is not writable")


@pytest.mark.parametrize("broken", ["raises", "unreadable"])
def test_a_ledger_error_never_fails_a_pull(kit, tmp_path, caplog, broken):  # noqa: F811
    if broken == "raises":
        kit.envs.images.ledger = BrokenLedger()
    else:
        ledger = ledger_at(tmp_path)
        ledger.root.mkdir(parents=True)
        ledger.path.write_text("{")
        kit.envs.images.ledger = ledger
    work = open_work(kit)
    with caplog.at_level(logging.WARNING):
        handle = pull(kit, work)
    assert [
        item["handle"] for item in kit.broker.image_list(work.credential)["images"]
    ] == [handle]
    [message] = [r.getMessage() for r in caplog.records if "ledger" in r.getMessage()]
    assert message.startswith("sandbox pull ledger not updated (")
    assert "secret" not in message


def test_a_busy_ledger_never_stalls_a_pull(kit, tmp_path, caplog):  # noqa: F811
    ledger = ledger_at(tmp_path, lock_wait=0.2)
    kit.envs.images.ledger = ledger
    work = open_work(kit)
    with ledger._locked(), caplog.at_level(logging.WARNING):
        pull(kit, work)
    assert [r.getMessage() for r in caplog.records if "ledger" in r.getMessage()] == [
        "sandbox pull ledger not updated (InfrastructureError); the image stays "
        "unprunable"
    ]
    assert ledger.entries() == ()


def test_unknown_host_images_record_nothing_and_never_fail_a_pull(
    kit,  # noqa: F811
    tmp_path,
    caplog,
    monkeypatch,
):
    ledger = ledger_at(tmp_path)
    kit.envs.images.ledger = ledger

    def image_ids():
        raise OSError("/var/run/docker.sock: secret")

    monkeypatch.setattr(kit.puller, "image_ids", image_ids)
    work = open_work(kit)
    with caplog.at_level(logging.WARNING):
        pull(kit, work)
    [message] = [r.getMessage() for r in caplog.records if "ledger" in r.getMessage()]
    assert message.startswith("sandbox pull ledger: host images unknown (OSError)")
    assert ledger.entries() == ()


# -- prune-images --------------------------------------------------------------


def conflict():
    return APIError("conflict", response=type("R", (), {"status_code": 409})())


def repository_of(reference):
    return reference.split("@")[0] if "@" in reference else reference.rpartition(":")[0]


def single_reference(references):
    """Docker's isSingleReference: at most one tag, and every digest in its
    repository; removing such an image by ID removes its references too."""
    if len(references) <= 1:
        return len(references) == 1
    tags = [item for item in references if "@" not in item]
    digests = {repository_of(item) for item in references if "@" in item}
    if len(tags) > 1:
        return False
    return digests == {repository_of(tags[0] if tags else references[0])}


class FakeDocker:
    """The daemon's image store as prune sees it (docker's rmi semantics):
    removing a tag untags and drops its repository's digests with its last
    tag; the last reference removes the image, unless a container uses it;
    removing by ID removes a single reference with the image and conflicts
    across repositories."""

    def __init__(self):
        self.images = {}
        self.containers_list = []
        self.removed = []
        self.fail = None
        # Call name -> callable(reference) run first (tests' race points).
        self.hooks = {}

    def add(self, image_id, tags=(), digests=None, size=4096):
        repositories = {tag.rpartition(":")[0] for tag in tags}
        self.images[image_id] = {
            "Id": image_id,
            "RepoTags": list(tags),
            "RepoDigests": [f"{repo}@sha256:{image_id[7:]}" for repo in repositories]
            if digests is None
            else list(digests),
            "Size": size,
        }

    def use(self, image_id, state="exited"):
        self.containers_list.append(
            {"Id": "c" * 64, "ImageID": image_id, "State": state}
        )

    def inspect_image(self, reference):
        if "inspect_image" in self.hooks:
            self.hooks["inspect_image"](reference)
        if self.fail == "inspect":
            raise APIError("daemon down")
        for image_id, attrs in self.images.items():
            if reference in (image_id, *attrs["RepoTags"], *attrs["RepoDigests"]):
                return json.loads(json.dumps(attrs))
        raise NotFound("no such image")

    def containers(self, all=False):
        assert all is True
        if self.fail == "containers":
            raise APIError("daemon down")
        return list(self.containers_list)

    def remove_image(self, image, force=False, noprune=False):
        assert (force, noprune) == (False, False)
        self.removed.append(image)
        if hook := self.hooks.pop("remove_image", None):
            hook(image)
        if self.fail == "conflict":
            raise conflict()
        if self.fail == "error":
            status = type("R", (), {"status_code": 500})()
            raise APIError("boom /secret", response=status)
        used = {item["ImageID"] for item in self.containers_list}
        if image in self.images:
            attrs = self.images[image]
            references = attrs["RepoTags"] + attrs["RepoDigests"]
            if (references and not single_reference(references)) or image in used:
                raise conflict()
            del self.images[image]
            return
        for image_id, attrs in self.images.items():
            tags, digests = list(attrs["RepoTags"]), list(attrs["RepoDigests"])
            if image in tags:
                tags.remove(image)
                repository = repository_of(image)
                if not any(repository_of(tag) == repository for tag in tags):
                    digests = [d for d in digests if repository_of(d) != repository]
            elif image in digests:
                digests.remove(image)
            else:
                continue
            if not tags and not digests:
                if image_id in used:
                    raise conflict()
                del self.images[image_id]
            else:
                attrs["RepoTags"], attrs["RepoDigests"] = tags, digests
            return
        raise NotFound("no such image")


class World:
    def __init__(self, tmp_path):
        self.clock = Clock()
        self.ledger = ledger_at(tmp_path, self.clock)
        self.leases = LeaseStore(tmp_path / "data" / "leases")
        self.docker = FakeDocker()
        self.pruner = ImagePruner(
            self.ledger, self.leases, self.docker, clock=self.clock
        )

    def pulled(self, image_id=ID_A, ref=REF_A, *, tags=None, **options):
        self.docker.add(image_id, [ref] if tags is None else tags, **options)
        record(self.ledger, image_id, ref)

    def run(self, older_than=None, *, dry_run=False, confirm=lambda rows: True):
        return self.pruner.run(older_than, dry_run=dry_run, confirm=confirm)


def actions(rows):
    return [(row.image.image_id, row.action, row.reason) for row in rows]


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_prune_removes_a_first_pulled_image_and_never_touches_the_rest(world):
    world.pulled(size=5000)
    # Pre-existing on the host: no entry, so never inspected or removed.
    world.docker.add(ID_B, ["busybox:1.37.0"])
    record(world.ledger, ID_B, "docker.io/library/busybox:1.37.0", pre_existing=True)

    rows = world.run()

    assert actions(rows) == [(ID_A, REMOVED, "")]
    assert rows[0].bytes == 5000
    assert world.docker.removed == ["public.ecr.aws/docker/library/busybox:1.36.1"]
    assert set(world.docker.images) == {ID_B}
    assert world.ledger.entries() == ()


def test_dry_run_lists_and_changes_nothing(world):
    world.pulled()
    world.pulled(ID_B, REPO + ":b")
    del world.docker.images[ID_B]
    before = world.ledger.path.read_text()

    rows = world.run(dry_run=True)

    assert actions(rows) == [
        (ID_A, WOULD_REMOVE, ""),
        (ID_B, WOULD_DROP, "no longer on the host"),
    ]
    assert world.docker.removed == []
    assert world.ledger.path.read_text() == before


def test_an_image_that_no_longer_exists_is_dropped_from_the_ledger(world):
    world.pulled()
    del world.docker.images[ID_A]
    assert actions(world.run()) == [(ID_A, DROPPED, "no longer on the host")]
    assert world.ledger.entries() == ()


@pytest.mark.parametrize("state", ["running", "exited", "created", "paused"])
def test_an_image_any_container_uses_is_kept(world, state):
    world.pulled()
    world.docker.use(ID_A, state)
    assert actions(world.run()) == [(ID_A, KEPT, "used by container cccccccccccc")]
    assert world.docker.removed == [] and ID_A in world.docker.images
    assert len(world.ledger.entries()) == 1


def test_an_image_a_run_lease_holds_is_kept(world):
    world.pulled()
    world.leases.write(
        resource_lease(
            sandbox_images=(
                pulled_image_lease(image_id=ID_A, state="present", pre_existing=False),
            ),
        )
    )
    assert actions(world.run()) == [(ID_A, KEPT, "held by run run-1")]
    assert world.docker.removed == []


@pytest.mark.parametrize(
    "image",
    [
        pulled_image_lease(image_id=ID_A, state="planned"),
        pulled_image_lease(image_id=ID_A, state="removed"),
        make_image_lease(image_id=ID_A, state="present"),
        make_image_lease(image_id=ID_A, state="leaked"),
    ],
    ids=["pulled-planned", "pulled-removed", "built-present", "built-leaked"],
)
def test_only_a_present_pulled_handle_holds_an_image(world, image):
    world.pulled()
    world.leases.write(resource_lease(sandbox_images=(image,)))
    assert actions(world.run()) == [(ID_A, REMOVED, "")]


def test_an_unreadable_lease_keeps_every_image(world):
    world.pulled()
    world.leases.root.mkdir(parents=True)
    (world.leases.root / "run-9.json").write_text("{")
    assert actions(world.run()) == [(ID_A, KEPT, "run lease run-9 is unreadable")]
    assert world.docker.removed == []


@pytest.mark.parametrize("late", ["container", "lease"])
def test_use_is_checked_again_right_before_removing(world, late):
    world.pulled()

    def confirm(rows):
        assert actions(rows) == [(ID_A, WOULD_REMOVE, "")]
        if late == "container":
            world.docker.use(ID_A, "running")
        else:
            world.leases.write(
                resource_lease(
                    sandbox_images=(pulled_image_lease(image_id=ID_A, state="present"),)
                )
            )
        return True

    [row] = world.run(confirm=confirm)
    assert row.action == KEPT and world.docker.removed == []
    assert len(world.ledger.entries()) == 1


def test_a_tag_the_ledger_did_not_record_keeps_the_image_untouched(world):
    world.pulled(tags=[REF_A, "mine:v1"])
    # The dry run says what the real run does, and neither untags anything.
    kept = [(ID_A, KEPT, "other tags remain: mine:v1")]
    assert actions(world.run(dry_run=True)) == kept
    assert actions(world.run()) == kept
    assert world.docker.removed == []
    assert world.docker.images[ID_A]["RepoTags"] == [REF_A, "mine:v1"]
    assert len(world.ledger.entries()) == 1


def test_reasons_are_bounded(world):
    tags = [f"docker.io/library/operator-image-{index}:v1" for index in range(8)]
    world.pulled(tags=[REF_A, *tags])
    [row] = world.run()
    assert row.action == KEPT and len(row.reason) == sandbox_ledger.MAX_REASON
    assert row.reason.startswith("other tags remain: ") and row.reason.endswith("...")


def test_a_tag_that_moved_to_another_image_is_never_removed(world):
    world.pulled()
    # The operator re-pulled the tag: it now names another image, and the
    # old one keeps only its digest.
    world.docker.images[ID_A]["RepoTags"] = []
    world.docker.add(ID_B, [REF_A])

    assert actions(world.run()) == [(ID_A, REMOVED, "")]
    assert world.docker.removed == [REPO + "@sha256:" + "a" * 64]
    assert world.docker.images[ID_B]["RepoTags"] == [REF_A]


def test_a_tag_that_moves_right_before_its_removal_is_left_alone(world):
    world.pulled()

    def move(reference):
        if reference == REF_A and ID_B not in world.docker.images:
            world.docker.images[ID_A]["RepoTags"] = []
            world.docker.add(ID_B, [REF_A])

    world.docker.hooks["inspect_image"] = move
    assert actions(world.run()) == [(ID_A, REMOVED, "")]
    assert REF_A not in world.docker.removed
    assert world.docker.images[ID_B]["RepoTags"] == [REF_A]


def test_a_digest_pull_is_removed_by_its_reference(world):
    reference = REPO + "@sha256:" + "d" * 64
    world.docker.add(ID_A, [], digests=[reference])
    world.ledger.record(
        ID_A, REPO, "sha256:" + "d" * 64, pre_existing=False, before=None, run_id="r"
    )
    assert actions(world.run()) == [(ID_A, REMOVED, "")]
    assert world.docker.removed == [reference]


def test_a_tag_added_after_the_checks_is_never_deleted(world):
    reference = REPO + "@sha256:" + "d" * 64
    world.docker.add(ID_A, [], digests=[reference])
    world.ledger.record(
        ID_A, REPO, "sha256:" + "d" * 64, pre_existing=False, before=None, run_id="r"
    )

    def tag(_):
        world.docker.images[ID_A]["RepoTags"].append("mine:v1")

    world.docker.hooks["remove_image"] = tag
    assert actions(world.run()) == [
        (ID_A, KEPT, "untagged 1; references remain: mine:v1")
    ]
    assert world.docker.removed == [reference]
    assert world.docker.images[ID_A]["RepoTags"] == ["mine:v1"]
    assert len(world.ledger.entries()) == 1


def test_an_entry_without_references_waits_for_the_other_name_to_go(world):
    python = "python:3.13-slim-bookworm"
    world.docker.add(ID_A, [python], digests=[])
    record(world.ledger, ID_A, "docker.io/library/" + python, before=ID_B)
    assert world.ledger.entries()[0].references == ()
    assert actions(world.run()) == [(ID_A, KEPT, "other tags remain: " + python)]
    assert world.docker.removed == []
    # Once the operator's name is gone, only removal by ID deletes it.
    world.docker.images[ID_A]["RepoTags"] = []
    assert actions(world.run()) == [(ID_A, REMOVED, "")]
    assert world.docker.removed == [ID_A]


def test_references_from_other_repositories_keep_the_image(world):
    world.pulled(digests=[REPO + "@sha256:" + "d" * 64, "busybox@sha256:" + "d" * 64])
    world.docker.images[ID_A]["RepoTags"] = []
    assert actions(world.run()) == [
        (ID_A, KEPT, "references from other repositories remain")
    ]
    assert world.docker.removed == []


def test_a_docker_conflict_keeps_the_image_and_its_entry(world):
    world.pulled()
    world.docker.fail = "conflict"
    assert actions(world.run()) == [(ID_A, KEPT, "Docker refused: conflict")]
    assert ID_A in world.docker.images and len(world.ledger.entries()) == 1


def test_a_docker_error_keeps_the_image_and_logs_its_class_only(world, caplog):
    world.pulled()
    world.docker.fail = "error"
    with caplog.at_level(logging.WARNING):
        assert actions(world.run()) == [(ID_A, KEPT, "Docker error")]
    assert ID_A in world.docker.images and len(world.ledger.entries()) == 1
    assert [r.getMessage() for r in caplog.records] == [
        "sandbox image prune failed: APIError"
    ]


def test_an_image_back_on_the_host_is_not_dropped(world):
    world.pulled()
    world.pulled(ID_B, REPO + ":b")
    gone = world.docker.images.pop(ID_B)

    def confirm(rows):
        assert actions(rows)[1] == (ID_B, WOULD_DROP, "no longer on the host")
        world.docker.images[ID_B] = gone
        return True

    assert actions(world.run(confirm=confirm)) == [
        (ID_A, REMOVED, ""),
        (ID_B, KEPT, "back on the host"),
    ]
    assert [entry.image_id for entry in world.ledger.entries()] == [ID_B]


def test_a_ledger_error_after_removing_still_reports_the_removals(
    world, caplog, monkeypatch
):
    world.pulled()

    def forget(removed):
        raise PermissionError("/root/secret/pulled.json")

    monkeypatch.setattr(world.ledger, "forget", forget)
    with caplog.at_level(logging.WARNING):
        assert actions(world.run()) == [(ID_A, REMOVED, "")]
    assert ID_A not in world.docker.images
    assert [r.getMessage() for r in caplog.records] == [
        "sandbox pull ledger not updated: PermissionError"
    ]


def test_older_than_selects_by_last_use(world):
    world.pulled()
    world.clock.now += 3 * DAY
    world.pulled(ID_B, REPO + ":b")
    world.clock.now += 3600
    assert actions(world.run(2 * DAY)) == [
        (ID_A, REMOVED, ""),
        (ID_B, KEPT, "last used 1h ago"),
    ]
    assert [entry.image_id for entry in world.ledger.entries()] == [ID_B]


def test_nothing_is_removed_without_confirmation(world):
    world.pulled()
    assert world.run(confirm=lambda rows: False) is None
    assert world.docker.removed == [] and len(world.ledger.entries()) == 1


def test_an_empty_ledger_needs_no_docker(world):
    world.docker.fail = "containers"
    assert world.run() == ()
    assert not world.ledger.root.exists()


@pytest.mark.parametrize("fail", ["containers", "inspect"])
def test_a_daemon_error_while_planning_fails_the_command(world, fail):
    world.pulled()
    world.docker.fail = fail
    with pytest.raises(InfrastructureError):
        world.run()
    assert world.docker.removed == []


def test_a_lease_store_it_cannot_list_holds_everything(world, monkeypatch):
    world.pulled()
    world.leases.root.mkdir(parents=True)
    real = sandbox_ledger.os.listdir

    def listdir(path):
        if str(path) == str(world.leases.root):
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(sandbox_ledger.os, "listdir", listdir)
    assert actions(world.run()) == [(ID_A, KEPT, "the run leases are unreadable")]
