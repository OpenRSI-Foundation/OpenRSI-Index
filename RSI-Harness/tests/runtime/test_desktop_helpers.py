"""Exact helper intents survive controller loss without age-based cleanup."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from docker.errors import NotFound

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.desktop_helpers import HELPER_LABEL, DesktopHelperIntents


class Helpers:
    def __init__(self):
        self.items = {}

    def get(self, name):
        if name not in self.items:
            raise NotFound("missing")
        return self.items[name]

    def create(self, intent):
        events = []

        def remove(**kwargs):
            events.append(("remove", kwargs))
            del self.items[intent["name"]]

        helper = SimpleNamespace(
            attrs={
                "Name": "/" + intent["name"],
                "Image": intent["image"],
                "Config": {"Labels": {HELPER_LABEL: intent["nonce"]}},
            },
            reload=lambda: events.append("reload"),
            remove=remove,
            events=events,
        )
        self.items[intent["name"]] = helper
        return helper


def client_for(daemon="daemon-1"):
    return SimpleNamespace(containers=Helpers(), info=lambda: {"ID": daemon})


@pytest.mark.parametrize("started", [False, True])
def test_helper_cleanup_covers_created_but_not_started_and_started(tmp_path, started):
    client = client_for()
    intents = DesktopHelperIntents(client, tmp_path / "intents")
    with pytest.raises(RuntimeError, match="controller exception"):
        with intents.operation("sha256:image") as intent:
            helper = client.containers.create(intent)
            if started:
                helper.events.append("start")
            raise RuntimeError("controller exception")
    assert helper.events[-1] == ("remove", {"force": True, "v": True})
    assert not list(intents.root.glob("*.json"))


def test_live_locked_operation_is_not_reclaimed(tmp_path):
    client = client_for()
    intents = DesktopHelperIntents(client, tmp_path / "intents")
    with intents.operation("sha256:image") as first:
        a = client.containers.create(first)
        with intents.operation("sha256:image") as second:
            b = client.containers.create(second)
            intents.recover()
            assert not a.events and not b.events
        assert first["name"] in client.containers.items
    assert not client.containers.items


@pytest.mark.parametrize("created", [False, True])
def test_controller_death_reclaims_only_its_exact_helper(tmp_path, created):
    root = tmp_path / "intents"
    # os._exit bypasses contextmanager/finally and releases the kernel lock.
    script = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from rsi_harness.runtime.desktop_helpers import DesktopHelperIntents
client = SimpleNamespace(info=lambda: {"ID": "daemon-1"})
with DesktopHelperIntents(client, Path(sys.argv[1])).operation("sha256:image"):
    os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(root)], check=True, env=os.environ
    )
    intent = json.loads(next(root.glob("*.json")).read_text())
    client = client_for()
    intents = DesktopHelperIntents(client, root)
    if created:
        helper = client.containers.create(intent)
    intents.recover()
    if created:
        assert helper.events[-1] == ("remove", {"force": True, "v": True})
        assert not list(root.glob("*.json"))
    else:
        # Missing does not prove a killed request cannot materialize later.
        assert list(root.glob("*.json"))
        helper = client.containers.create(intent)
        intents.recover()
        assert helper.events[-1][0] == "remove"
        assert not list(root.glob("*.json"))


@pytest.mark.parametrize("field", ["image", "nonce", "name"])
def test_recovery_does_not_remove_unattested_helper(tmp_path, field):
    client = client_for()
    intents = DesktopHelperIntents(client, tmp_path / "intents")
    with pytest.raises(InfrastructureError, match="ownership"):
        with intents.operation("sha256:image") as intent:
            helper = client.containers.create(intent)
            if field == "nonce":
                helper.attrs["Config"]["Labels"][HELPER_LABEL] = "other"
            elif field == "image":
                helper.attrs["Image"] = "different"
            else:
                helper.attrs["Name"] = "/different"
    with pytest.raises(InfrastructureError, match="ownership"):
        intents.recover()
    assert not any(isinstance(event, tuple) for event in helper.events)
    assert list(intents.root.glob("*.json"))


def test_different_daemon_and_unregistered_containers_are_untouched(tmp_path):
    client = client_for()
    intents = DesktopHelperIntents(client, tmp_path / "intents")
    with intents.operation("sha256:image") as intent:
        pass  # No observed container: preserve an unresolved create intent.
    other = client_for("other-daemon")
    helper = other.containers.create(intent)
    DesktopHelperIntents(other, intents.root).recover()
    assert not helper.events
    assert list(intents.root.glob("*.json"))


def test_unpublished_interrupted_intent_is_discarded_without_daemon_lookup(tmp_path):
    intents = DesktopHelperIntents(client_for(), tmp_path / "intents")
    pending = intents.root / ".pending-interrupted"
    pending.write_text('{"partial":')
    intents.recover()
    assert not pending.exists()


def test_failed_intent_persistence_never_calls_daemon_cleanup(tmp_path, monkeypatch):
    from rsi_harness.runtime import desktop_helpers

    client = client_for()
    intents = DesktopHelperIntents(client, tmp_path / "intents")
    client.containers.get = lambda name: pytest.fail("unissued create queried")

    def fail(path):
        raise OSError("fsync failed")

    monkeypatch.setattr(desktop_helpers, "fsync_directory", fail)
    with pytest.raises(OSError, match="fsync failed"):
        with intents.operation("sha256:image"):
            pytest.fail("operation reached before durable intent")
    assert not list(intents.root.glob("*.json"))
    assert not list(intents.root.glob(".pending-*"))


@pytest.mark.parametrize("failure", [None, "create", "start", "wait", "logs", "remove"])
def test_host_helper_fault_boundaries_use_owned_intent(tmp_path, monkeypatch, failure):
    from rsi_harness.runtime import desktop, desktop_helpers

    client = client_for()
    intents = DesktopHelperIntents(client, tmp_path / "intents")
    monkeypatch.setattr(desktop_helpers, "DesktopHelperIntents", lambda client: intents)
    images = []

    def image_get(image):
        images.append(image)
        return SimpleNamespace(id="sha256:fixed-image")

    client.images = SimpleNamespace(get=image_get)
    normal_create = client.containers.create

    def step(name, value=None):
        if failure == name:
            raise RuntimeError(f"{name} failed")
        return value

    def create(image, command, **options):
        intent = {
            "name": options["name"],
            "image": image,
            "nonce": options["labels"][HELPER_LABEL],
        }
        helper = normal_create(intent)
        helper.start = lambda: step("start")
        helper.wait = lambda **kwargs: step("wait", {"StatusCode": 0})
        helper.logs = lambda **kwargs: step("logs", b"output")
        if failure == "remove":
            helper.remove = lambda **kwargs: step("remove")
        step("create")  # Simulate response loss after the daemon created it.
        return helper

    client.containers.create = create
    host = desktop.DesktopHost(client)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            host._execute(["getent", "ahostsv4", "host.docker.internal"], host=False)
    else:
        assert host._execute(["getent"], host=False).returncode == 0
    assert images == [desktop.HELPER_IMAGE]
    assert "@sha256:" in images[0]
    assert bool(list(intents.root.glob("*.json"))) is (failure == "remove")
    assert bool(client.containers.items) is (failure == "remove")
