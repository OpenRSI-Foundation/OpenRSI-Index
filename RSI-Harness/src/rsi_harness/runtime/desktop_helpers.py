"""Controller-owned intents for short-lived Docker Desktop host helpers.

Helpers can be needed before a run lease exists. A locked, durable intent covers
create-before-start and controller death; recovery uses exact ownership, never
container age, PID reuse, or a prefix search over arbitrary daemon containers.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from docker.errors import NotFound

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.durable import durable_mkdir, fsync_directory

HELPER_LABEL = "org.openrsi.desktop-helper-intent"


class DesktopHelperIntents:
    def __init__(self, client: Any, root: Path | None = None) -> None:
        self.client = client
        self.root = root or Path.home() / ".cache/rsi-harness/desktop-helpers"
        durable_mkdir(self.root)
        metadata = self.root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise InfrastructureError("Desktop helper intent directory is not private")
        self.daemon = self.client.info()["ID"]
        if not isinstance(self.daemon, str) or not self.daemon:
            raise InfrastructureError("Docker daemon identity is missing")

    @contextmanager
    def _guard(self):
        descriptor = os.open(
            self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _remove(self, intent: dict[str, str]) -> bool:
        try:
            helper = self.client.containers.get(intent["name"])
        except NotFound:
            # A killed create request may still be in flight at the daemon.
            # Keep its intent, so a later recovery can reclaim a late container.
            return False
        helper.reload()
        if (
            helper.attrs.get("Config", {}).get("Labels", {}).get(HELPER_LABEL)
            != intent["nonce"]
            or helper.attrs.get("Image") != intent["image"]
            or helper.attrs.get("Name") != "/" + intent["name"]
        ):
            raise InfrastructureError("Desktop helper ownership cannot be proven")
        helper.remove(force=True, v=True)
        return True

    def recover(self) -> None:
        with self._guard():
            # Writers hold this same guard until an intent is durable/published.
            # Unpublished files cannot have an associated Docker create request.
            for pending in self.root.glob(".pending-*"):
                pending.unlink()
            for path in self.root.glob("*.json"):
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue  # A live controller still owns this operation.
                    with os.fdopen(os.dup(descriptor)) as source:
                        intent = json.load(source)
                    if intent["daemon"] != self.daemon:
                        continue
                    if self._remove(intent):
                        path.unlink()
                        fsync_directory(self.root)
                finally:
                    os.close(descriptor)

    @contextmanager
    def operation(self, image: str):
        self.recover()
        nonce = uuid.uuid4().hex
        intent = {
            "nonce": nonce,
            "name": "rsi-desktop-helper-" + nonce,
            "image": image,
            "daemon": self.daemon,
        }
        path = self.root / (nonce + ".json")
        descriptor = None
        pending = None
        published = False
        try:
            # The guard prevents recovery from seeing an unlocked partial intent.
            with self._guard():
                descriptor, pending = tempfile.mkstemp(
                    prefix=".pending-", dir=self.root
                )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                with os.fdopen(os.dup(descriptor), "w") as output:
                    json.dump(intent, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(pending, path)
                fsync_directory(self.root)
                published = True
            yield intent
        finally:
            if descriptor is not None:
                try:
                    if not published:
                        # The caller never received the intent, so create could
                        # not have been issued; no daemon lookup is necessary.
                        with self._guard():
                            path.unlink(missing_ok=True)
                            if pending is not None:
                                Path(pending).unlink(missing_ok=True)
                    # A failed remove leaves the intent for the next recovery.
                    elif self._remove(intent):
                        with self._guard():
                            path.unlink()
                            fsync_directory(self.root)
                finally:
                    os.close(descriptor)
