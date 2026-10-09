"""One optional lifecycle port for parent composition and round isolation."""

from __future__ import annotations

import errno
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerMount
from rsi_harness.runtime.sandbox_contracts import SandboxOwner

# The endpoint's Python modules (spec 6): the stdlib client, the compose
# front-end and the Harbor plugin, importable in Work and Judge through
# PYTHONPATH=$RSI_SANDBOX_PYTHONPATH. Keys are the injected names, values
# the package sources beside sandbox_client.py; nothing here imports them.
ENDPOINT_MODULES = {
    "rsi_sandbox_client.py": "sandbox_client.py",
    "rsi_sandbox_compose.py": "sandbox_compose.py",
    "rsi_sandbox_harbor.py": "sandbox_harbor_env.py",
}


def _write_new(path, data, mode):
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode
    )
    with os.fdopen(descriptor, "wb") as output:
        # The umask must not narrow what the parent UID has to read.
        os.fchmod(output.fileno(), mode)
        output.write(data)


class NullSandboxLifecycle:
    enabled = False
    can_resume = True
    work_ended_normally = False

    def prepare_work(self):
        return None

    def prepare_judge(self, round_id):
        return None

    def activate_work(self, deadline):
        pass

    def activate_judge(self, deadline):
        pass

    def freeze_work(self):
        pass

    def resume_work(self):
        pass

    def resume_parent(self, runtime, container):
        runtime.unpause(container)

    def reopen_work(self):
        pass

    def contain_work(self):
        pass

    def close_judge(self):
        pass

    def cancel_work(self):
        pass

    def cancel_run(self):
        pass

    def close(self):
        pass

    def release_resources(self):
        pass


@dataclass(frozen=True)
class SandboxEndpoint:
    directory: Path
    mount: ContainerMount
    environment: dict[str, str] = field(repr=False)
    owner: SandboxOwner
    server: object = field(repr=False)
    identity: tuple[int, int]


class SandboxLifecycle(NullSandboxLifecycle):
    enabled = True

    def __init__(self):
        self.broker = None
        self.root = None
        self._run_id = self._task_id = None
        self._endpoints = {}
        self._slots = None
        self._cancelled = False
        self._work_cancelled = False
        self._work_deadline = None
        self._reservation = None
        self._transport = None
        self._other_transports = ()

    def own_transport(self, transport, *others):
        # ``transport`` is the 5 s control client; ``others`` (the env
        # runtime's own client) are only closed with it.
        self._transport = transport
        self._other_transports = others

    def reserve(self, pool, grant, run_id, mutate):
        pool.reserve_run(run_id, grant, mutate)
        self._reservation = (pool, run_id, mutate)

    def release_resources(self):
        if self._reservation is not None:
            pool, run_id, mutate = self._reservation
            pool.release_run(run_id, mutate)
            self._reservation = None
        if self._transport is not None:
            try:
                self._transport.close()
            finally:
                self._transport = None
                others, self._other_transports = self._other_transports, ()
                for transport in others:
                    transport.close()

    @staticmethod
    def validate_root(root):
        # Reserve room for one fixed-size random directory and the socket basename.
        if len(os.fsencode(Path(root) / ("x" * 8) / "s")) > 107:
            raise SetupError(
                "sandbox socket path exceeds 107 bytes; use a shorter data root"
            )

    def configure(self, broker, root, run_id, task_id):
        from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

        if self.broker is not None or self._cancelled:
            raise InfrastructureError("sandbox lifecycle cannot be rebound")
        self.validate_root(root)
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.broker = broker
        self._run_id, self._task_id = run_id, task_id
        # Work and Judge share one gate, sized by the operator's policy.
        self._slots = SandboxRequestSlots.from_policy(
            getattr(broker, "host_policy", None)
        )
        broker.start()

    def _prepare(self, phase, round_id=None):
        from rsi_harness.integrations import sandbox_client as wire
        from rsi_harness.runtime.sandbox_server import SANDBOX_TARGET, SandboxServer

        if self.broker is None or self._cancelled:
            raise InfrastructureError("sandbox lifecycle is unavailable")
        if not self.broker.grants_phase(phase):
            return None
        previous = self._endpoints.get(phase)
        if previous is not None:
            if previous.owner.round_id != round_id:
                raise InfrastructureError(
                    "sandbox previous round endpoint is still active"
                )
            return previous
        owner = SandboxOwner(
            run_id=self._run_id, task_id=self._task_id, phase=phase, round_id=round_id
        )
        credentials = (
            self.broker.open_judge(owner)
            if phase == "judge"
            else self.broker.open_session(owner)
        )
        directory = self.root / uuid.uuid4().hex[:8]
        # The run-owned ancestor stays 0700 on the host. The exact read-only
        # endpoint bind must be traversable by the task's existing parent UID;
        # authentication remains the ephemeral exec-only bearer credential.
        directory.mkdir(mode=0o755)
        directory.chmod(0o755)
        metadata = directory.stat()
        server = SandboxServer(
            self.broker, directory / "s", owner, slots=self._slots, parent_users=True
        )
        environment = {
            "RSI_SANDBOX_SOCKET": SANDBOX_TARGET + "/s",
            "RSI_SANDBOX_TOKEN": credentials.credential,
            # Work and Judge both merge this endpoint environment.
            "RSI_SANDBOX_PYTHONPATH": SANDBOX_TARGET + "/py",
        }
        endpoint = SandboxEndpoint(
            directory,
            ContainerMount(
                source=directory, target=PurePosixPath(SANDBOX_TARGET), read_only=True
            ),
            environment,
            owner,
            server,
            (metadata.st_dev, metadata.st_ino),
        )
        # Register before I/O so failed startup still has exact cleanup authority.
        self._endpoints[phase] = endpoint
        try:
            source = Path(wire.__file__)
            _write_new(directory / "rsi-sandbox", source.read_bytes(), 0o755)
            (directory / "py").mkdir(mode=0o755)
            (directory / "py").chmod(0o755)
            for name, module in ENDPOINT_MODULES.items():
                data = source.with_name(module).read_bytes()
                _write_new(directory / "py" / name, data, 0o644)
            server.start()
        except BaseException:
            self.cancel_run()
            raise
        return endpoint

    def prepare_work(self):
        return self._prepare("work")

    def prepare_judge(self, round_id):
        return self._prepare("judge", round_id)

    def activate_work(self, deadline):
        if self._work_deadline is not None:
            raise InfrastructureError("sandbox Work deadline cannot be reset")
        self._work_deadline = deadline
        self.broker.activate_work(deadline)

    def activate_judge(self, deadline):
        self.broker.activate_judge(deadline)

    @property
    def can_resume(self):
        clock = time.monotonic if self.broker is None else self.broker.clock
        return (
            not self._cancelled
            and not self._work_cancelled
            and (self._work_deadline is None or clock() < self._work_deadline)
            and (self.broker is None or self.broker.can_resume)
        )

    @property
    def work_ended_normally(self):
        """Work reached its deadline or was retired, not cancelled or failed.

        An accepted Judge round that outlives such an end keeps its result;
        Work then stays paused until the coordinator removes it.
        """
        clock = time.monotonic if self.broker is None else self.broker.clock
        ended = self._work_cancelled or (
            self._work_deadline is not None and clock() >= self._work_deadline
        )
        return (
            ended
            and not self._cancelled
            and (self.broker is None or not self.broker.recovery_required)
        )

    def freeze_work(self):
        self.broker.freeze_work()
        endpoint = self._endpoints.get("work")
        if endpoint is not None:
            endpoint.server.suspend()

    def resume_work(self):
        self.broker.resume_work(reopen=False)

    def resume_parent(self, runtime, container):
        if self.broker is None or not self.can_resume:
            raise InfrastructureError("sandbox parent resume denied; recovery required")
        # The ordinary parent client permits long build calls. Never hold the
        # resume gate using that timeout; production owns this same-daemon 5s
        # control transport for child lifecycle operations as well.
        control = self._transport or getattr(self.broker.backend, "client", None)
        runtime.unpause(
            container, admission=self.broker.resume_admission, control_client=control
        )

    def reopen_work(self):
        self.broker.reopen_work()
        endpoint = self._endpoints.get("work")
        if endpoint is not None:
            endpoint.server.resume()

    def contain_work(self):
        if self.broker is not None:
            self.broker.contain_work()

    def _drop_endpoint(self, phase):
        endpoint = self._endpoints.get(phase)
        if endpoint is None:
            return
        endpoint.server.stop()
        info = endpoint.directory.lstat()
        if (info.st_dev, info.st_ino) != endpoint.identity:
            raise InfrastructureError(
                "sandbox endpoint identity changed; recovery required"
            )
        (endpoint.directory / "rsi-sandbox").unlink(missing_ok=True)
        modules = endpoint.directory / "py"
        for name in ENDPOINT_MODULES:
            (modules / name).unlink(missing_ok=True)
        if modules.is_dir() and not modules.is_symlink():
            modules.rmdir()
        endpoint.directory.rmdir()
        del self._endpoints[phase]

    def close_judge(self):
        try:
            if self.broker is not None:
                self.broker.close_judge()
        finally:
            self._drop_endpoint("judge")

    def cancel_work(self):
        # Mark first: a round that sees the broker revocation must also see
        # the normal end, never a transient "not normal" end.
        self._work_cancelled = True
        if self.broker is not None:
            self.broker.cancel_work()

    def cancel_run(self):
        # Mark first so no round mistakes a cancellation for a normal end.
        self._cancelled = True
        if self.broker is not None:
            self.broker.cancel_run()

    def close(self):
        try:
            self.cancel_run()
            if self.broker is not None:
                self.broker.close()
        finally:
            errors = []
            for phase in list(self._endpoints):
                try:
                    self._drop_endpoint(phase)
                except Exception as error:
                    errors.append(error)
            if errors:
                raise InfrastructureError(
                    "sandbox endpoint cleanup requires recovery"
                ) from errors[0]
        self._remove_root()

    def _remove_root(self):
        """A cleanly closed run leaves no ``sb`` (spec 8 A5).

        The broker's spool (``sb/spool``, sandbox_spool_root) holds only
        closed stage and exec files once broker.close() returned; a broker
        that needs recovery keeps it for ``rsi-harness recover``. Every
        builder is removed by then, so ``sb/build`` (loop-ext4 state files)
        is empty and goes too; a file left in it stays for recovery, which
        detaches its loop device first. ``sb`` itself goes only when nothing
        else is left in it.
        """
        if self.root is None or self.broker is None:
            return
        if getattr(self.broker, "recovery_required", False):
            return
        spool = self.root / "spool"
        if spool.is_symlink():
            raise InfrastructureError(
                "sandbox spool is not the run's spool; recovery required"
            )
        if spool.is_dir():
            # rmtree refuses a symlinked root and never follows links below.
            shutil.rmtree(spool)
        build = self.root / "build"
        if build.is_symlink():
            raise InfrastructureError(
                "sandbox build directory is not the run's; recovery required"
            )
        try:
            build.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            if error.errno != errno.ENOTEMPTY:
                raise
        try:
            self.root.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            if error.errno != errno.ENOTEMPTY:
                raise
