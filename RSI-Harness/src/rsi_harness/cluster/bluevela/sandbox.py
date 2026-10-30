"""Brokered remote environments of one cluster run
(docs/sandbox-operator-guide.md).

The local runtime's broker (``SandboxBroker`` over ``remote_env_runtime``)
runs inside the compute-node Engine process as the job user: no Docker
daemon and no root, because every env is a remote backend's sandbox. Its
per-phase endpoint (a Unix socket directory) lives under node-local scratch
and is bind-mounted read-only into Work and the single-node Judge where the
local runtime mounts it (``/run/rsi-harness/sandbox``). The compute node
must reach the backend's API, directly or through its table's ``proxy``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rsi_harness.errors import SetupError
from rsi_harness.models import RunPlan
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import LeaseMutator, SandboxJournal
from rsi_harness.runtime.sandbox_contracts import SandboxEnvGrant
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from rsi_harness.runtime.sandbox_tools import check_tools


class ClusterSandbox:
    """The run's sandbox lifecycle, bound when the coordinator's lease exists.

    ``client_factory`` maps (credentials, the backend's table) to its client
    (default: the backend's own, sandbox_remote.BACKENDS); tests pass a fake.
    """

    def __init__(
        self,
        plan: RunPlan,
        run_id: str,
        root: Path,
        *,
        client_factory: Callable[[tuple[str, ...], Any], Any] | None = None,
    ) -> None:
        if not isinstance(plan.sandbox, SandboxEnvGrant):
            raise SetupError("cluster sandbox requires an environment grant")
        self.grant = plan.sandbox
        self.task_id = plan.task.task_id
        self.run_id = run_id
        self.root = Path(root)
        self.lifecycle = SandboxLifecycle()
        self.secrets: set[str] = set()
        self._client_factory = client_factory

    def bind(self, mutate: LeaseMutator) -> Path | None:
        """``on_lease_ready``: read the credentials, journal where the
        sandboxes are (never the credentials), start the broker and open
        Work's endpoint. Returns the Work endpoint directory, or None without
        a Work grant."""
        from rsi_harness.runtime.sandbox_remote import (
            remote_backend,
            remote_env_runtime,
        )

        host = self.grant.environments.host
        settings = host.remote
        assert settings is not None
        backend = remote_backend(host.backend)
        check_tools(host)
        self.lifecycle.validate_root(self.root)
        credentials = backend.credentials(settings)
        self.secrets.update(credentials)
        # Durable first: a crash after this still lets recovery find them.
        field = f"sandbox_{host.backend}"
        mutate(lambda lease: lease.model_copy(update={field: settings}))
        client = (self._client_factory or backend.client)(credentials, settings)
        self.lifecycle.own_transport(client)
        self.root.mkdir(mode=0o700)
        runtime = remote_env_runtime(client, spool_root=self.root / "spool", host=host)
        broker = SandboxBroker(
            self.grant, None, SandboxJournal(mutate), time.monotonic, envs=runtime
        )
        self.lifecycle.configure(broker, self.root, self.run_id, self.task_id)
        endpoint = self.lifecycle.prepare_work()
        if endpoint is None:
            return None
        self.secrets.add(endpoint.environment["RSI_SANDBOX_TOKEN"])
        return endpoint.directory

    def work_environment(self) -> dict[str, str]:
        endpoint = self.lifecycle.prepare_work()
        return {} if endpoint is None else dict(endpoint.environment)


def recover_cluster_sandboxes(
    run_root: Path,
    run_id: str | None = None,
    *,
    remote_client: Callable[[Any], Any] | None = None,
) -> tuple[str, ...]:
    """``rsi-harness recover|cleanup --cluster``: kill the remote sandboxes
    of one cluster run (or of every run under ``run_root``) by their
    metadata.

    Work and Judge were processes of the scheduler job, which ended them;
    only the sandboxes outlive it. Run this after the job has ended.
    """
    run_root = Path(run_root)
    if run_id is not None:
        candidates = [run_root / run_id]
    elif run_root.is_dir():
        candidates = sorted(run_root.iterdir())
    else:
        candidates = []
    recovered: list[str] = []
    for run_dir in candidates:
        if not (run_dir / "leases").is_dir():
            continue
        store = LeaseStore(run_dir / "leases")
        manager = RecoveryManager(
            store=store,
            backend=None,  # type: ignore[arg-type]  # remote only: no Docker
            managed_root=run_dir,
            remote_client=remote_client,
        )
        for selected in store.list_run_ids():
            if manager.recover_remote(selected):
                recovered.append(selected)
    return tuple(recovered)


__all__ = ["ClusterSandbox", "recover_cluster_sandboxes"]
