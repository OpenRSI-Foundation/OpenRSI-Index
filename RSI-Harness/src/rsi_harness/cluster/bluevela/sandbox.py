"""Brokered E2B environments of one cluster run (docs/sandbox-operator-guide.md).

The local runtime's broker (``SandboxBroker`` over ``e2b_env_runtime``) runs
inside the compute-node Engine process as the job user: no Docker daemon and
no root, because every env is an E2B sandbox. Its per-phase endpoint (a Unix
socket directory) lives under node-local scratch and is bind-mounted
read-only into Work and the single-node Judge where the local runtime mounts
it (``/run/rsi-harness/sandbox``). The compute node must reach the E2B API,
directly or through ``[environments.host.e2b] proxy``.
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
from rsi_harness.runtime.sandbox_contracts import EnvE2BHost, SandboxEnvGrant
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from rsi_harness.runtime.sandbox_tools import check_tools


def _default_client(api_key: str, settings: EnvE2BHost) -> Any:
    from rsi_harness.runtime.sandbox_e2b import e2b_client

    return e2b_client(settings, api_key)


class ClusterSandbox:
    """The run's sandbox lifecycle, bound when the coordinator's lease exists.

    ``client_factory`` maps (API key, ``[environments.host.e2b]``) to the E2B
    client; tests pass a fake.
    """

    def __init__(
        self,
        plan: RunPlan,
        run_id: str,
        root: Path,
        *,
        client_factory: Callable[[str, EnvE2BHost], Any] | None = None,
    ) -> None:
        if not isinstance(plan.sandbox, SandboxEnvGrant):
            raise SetupError("cluster sandbox requires an environment grant")
        self.grant = plan.sandbox
        self.task_id = plan.task.task_id
        self.run_id = run_id
        self.root = Path(root)
        self.lifecycle = SandboxLifecycle()
        self.secrets: set[str] = set()
        self._client_factory = client_factory or _default_client

    def bind(self, mutate: LeaseMutator) -> Path | None:
        """``on_lease_ready``: read the key, journal where the sandboxes are
        (never the key), start the broker and open Work's endpoint. Returns
        the Work endpoint directory, or None without a Work grant."""
        from rsi_harness.runtime.sandbox_e2b import e2b_env_runtime, read_api_key

        host = self.grant.environments.host
        settings = host.e2b
        assert settings is not None
        check_tools(host)
        self.lifecycle.validate_root(self.root)
        api_key = read_api_key(settings)
        self.secrets.add(api_key)
        # Durable first: a crash after this still lets recovery find them.
        mutate(lambda lease: lease.model_copy(update={"sandbox_e2b": settings}))
        client = self._client_factory(api_key, settings)
        self.lifecycle.own_transport(client)
        self.root.mkdir(mode=0o700)
        runtime = e2b_env_runtime(client, spool_root=self.root / "spool", host=host)
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
    e2b_client: Callable[[EnvE2BHost], Any] | None = None,
) -> tuple[str, ...]:
    """``rsi-harness recover|cleanup --cluster``: kill the E2B sandboxes of
    one cluster run (or of every run under ``run_root``) by their metadata.

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
            backend=None,  # type: ignore[arg-type]  # E2B only: no Docker
            managed_root=run_dir,
            e2b_client=e2b_client,
        )
        for selected in store.list_run_ids():
            if manager.recover_e2b(selected):
                recovered.append(selected)
    return tuple(recovered)


__all__ = ["ClusterSandbox", "recover_cluster_sandboxes"]
