"""Single-writer child journal and shared-data-root admission reservations."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.recovery import (
    LeaseStore,
    ResourceLease,
    retained_sandbox_resources,
)
from rsi_harness.runtime.sandbox_contracts import (
    SandboxEnvGrant,
    SandboxGrant,
    SandboxLease,
    SandboxReservation,
    SandboxState,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    BuilderLease,
    SandboxEnvLease,
    SandboxImageLease,
)

LeaseMutator = Callable[[Callable[[ResourceLease], ResourceLease]], ResourceLease]


class SandboxJournal:
    """All changes run inside the coordinator's authority, never its run flock."""

    def __init__(self, mutate: LeaseMutator) -> None:
        self._mutate = mutate

    def snapshot(self) -> tuple[SandboxLease, ...]:
        return self._mutate(lambda lease: lease).sandboxes

    def plan(self, child: SandboxLease) -> None:
        def transform(lease: ResourceLease) -> ResourceLease:
            if any(item.child_id == child.child_id for item in lease.sandboxes):
                raise InfrastructureError("duplicate sandbox planned identity")
            return lease.model_copy(update={"sandboxes": lease.sandboxes + (child,)})

        self._mutate(transform)

    def _update(
        self, child_id: str, transform: Callable[[SandboxLease], SandboxLease]
    ) -> None:
        def update(lease: ResourceLease) -> ResourceLease:
            if not any(child.child_id == child_id for child in lease.sandboxes):
                raise InfrastructureError("unknown sandbox durable identity")
            children = tuple(
                transform(child) if child.child_id == child_id else child
                for child in lease.sandboxes
            )
            return lease.model_copy(update={"sandboxes": children})

        self._mutate(update)

    def record_container(self, child_id: str, container_id: str) -> None:
        def transform(child: SandboxLease) -> SandboxLease:
            if child.container_id is not None and child.container_id != container_id:
                raise InfrastructureError(
                    "sandbox container identity cannot be rebound"
                )
            return child.model_copy(update={"container_id": container_id})

        self._update(child_id, transform)

    def set_state(
        self, child_id: str, state: SandboxState, *, pending_mutation: bool = False
    ) -> None:
        self._update(
            child_id,
            lambda child: child.model_copy(
                update={"state": state, "pending_mutation": pending_mutation}
            ),
        )

    def mark_removed(self, child_id: str) -> None:
        self.set_state(child_id, "removed")

    # -- environments and images (schema 6) --------------------------------

    def envs(self) -> tuple[SandboxEnvLease, ...]:
        return self._mutate(lambda lease: lease).sandbox_envs

    def images(self) -> tuple[SandboxImageLease, ...]:
        return self._mutate(lambda lease: lease).sandbox_images

    def builders(self) -> tuple[BuilderLease, ...]:
        return self._mutate(lambda lease: lease).sandbox_builders

    def plan_env(self, env: SandboxEnvLease) -> None:
        """The whole cleanup plan of an env, before any Docker call."""

        def transform(lease: ResourceLease) -> ResourceLease:
            if any(item.env_id == env.env_id for item in lease.sandbox_envs):
                raise InfrastructureError("duplicate sandbox env planned identity")
            return lease.model_copy(
                update={"sandbox_envs": lease.sandbox_envs + (env,)}
            )

        self._mutate(transform)

    def commit_env(self, env: SandboxEnvLease) -> SandboxEnvLease:
        """Replace an env record; a proven-removed env leaves the journal.

        Removal is proof of absence, so recovery needs nothing from it and
        the lease stays bounded however many envs a run creates.
        """

        def transform(lease: ResourceLease) -> ResourceLease:
            if not any(item.env_id == env.env_id for item in lease.sandbox_envs):
                raise InfrastructureError("unknown sandbox env durable identity")
            removed = env.state == "removed" and not env.pending_mutation
            envs = tuple(
                item if item.env_id != env.env_id else env
                for item in lease.sandbox_envs
                if not (removed and item.env_id == env.env_id)
            )
            return lease.model_copy(update={"sandbox_envs": envs})

        self._mutate(transform)
        return env

    def plan_image(self, image: SandboxImageLease) -> None:
        def transform(lease: ResourceLease) -> ResourceLease:
            if any(item.handle == image.handle for item in lease.sandbox_images):
                raise InfrastructureError("duplicate sandbox image planned identity")
            return lease.model_copy(
                update={"sandbox_images": lease.sandbox_images + (image,)}
            )

        self._mutate(transform)

    def commit_image(self, image: SandboxImageLease) -> None:
        """Replace an image record; a removed record leaves the journal.

        A pulled record is a ledger entry only (the image stays cached), so
        unbinding its handle leaves nothing to recover; a built image is
        ``removed`` only once its absence is proven.
        """

        def transform(lease: ResourceLease) -> ResourceLease:
            if not any(item.handle == image.handle for item in lease.sandbox_images):
                raise InfrastructureError("unknown sandbox image durable identity")
            released = image.state == "removed"
            images = tuple(
                item if item.handle != image.handle else image
                for item in lease.sandbox_images
                if not (released and item.handle == image.handle)
            )
            return lease.model_copy(update={"sandbox_images": images})

        self._mutate(transform)

    def plan_builder(self, builder: BuilderLease) -> None:
        """The whole cleanup plan of a builder, before any Docker call."""

        def transform(lease: ResourceLease) -> ResourceLease:
            if any(
                item.builder_id == builder.builder_id for item in lease.sandbox_builders
            ):
                raise InfrastructureError("duplicate sandbox builder planned identity")
            return lease.model_copy(
                update={"sandbox_builders": lease.sandbox_builders + (builder,)}
            )

        self._mutate(transform)

    def commit_builder(self, builder: BuilderLease) -> BuilderLease:
        """Replace a builder record; a proven-removed builder leaves the
        journal, as an env does."""

        def transform(lease: ResourceLease) -> ResourceLease:
            if not any(
                item.builder_id == builder.builder_id for item in lease.sandbox_builders
            ):
                raise InfrastructureError("unknown sandbox builder durable identity")
            removed = builder.state == "removed" and not builder.pending_mutation
            builders = tuple(
                item if item.builder_id != builder.builder_id else builder
                for item in lease.sandbox_builders
                if not (removed and item.builder_id == builder.builder_id)
            )
            return lease.model_copy(update={"sandbox_builders": builders})

        self._mutate(transform)
        return builder


class SandboxAdmissionPool:
    """Reservations cover this explicit data-root partition, not arbitrary jobs."""

    def __init__(self, lease_store: LeaseStore) -> None:
        self._store = lease_store

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self._store.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self._store.root / ".sandbox-admission.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def reserve_run(
        self,
        run_id: str,
        grant: SandboxGrant | SandboxEnvGrant,
        mutate: LeaseMutator,
    ) -> None:
        requested = SandboxReservation(
            cpus=grant.reserved_cpus,
            memory_mb=grant.reserved_memory_mb,
            pool_cpus=grant.pool_cpus,
            pool_memory_mb=grant.pool_memory_mb,
            **(
                {"disk_mb": grant.reserved_disk_mb, "pool_disk_mb": grant.pool_disk_mb}
                if isinstance(grant, SandboxEnvGrant)
                else {}
            ),
        )
        with self._lock():
            cpus, memory, disk = 0, 0, 0
            existing = None
            for identity in self._store.list_run_ids():
                try:
                    lease = self._store.read(identity)
                    if lease is None:
                        raise ValueError("lease disappeared during admission")
                except (OSError, ValueError, AttributeError) as error:
                    raise InfrastructureError(
                        f"sandbox pool cannot validate lease {identity}: {error}"
                    ) from error
                reservation = lease.sandbox_reservation
                if reservation is None:
                    continue
                if identity == run_id:
                    existing = reservation
                    continue
                # Profile-only reservations hold no disk and name no disk pool.
                disk_pools = {reservation.pool_disk_mb, requested.pool_disk_mb}
                if (reservation.pool_cpus, reservation.pool_memory_mb) != (
                    grant.pool_cpus,
                    grant.pool_memory_mb,
                ) or len(disk_pools - {None}) > 1:
                    raise SetupError(
                        "sandbox pool capacity differs from outstanding reservations; "
                        "recover/drain before changing partition"
                    )
                cpus += reservation.cpus
                memory += reservation.memory_mb
                disk += reservation.disk_mb
            if existing is not None:
                if existing != requested:
                    raise InfrastructureError(
                        "sandbox reservation envelope cannot change"
                    )
                return
            disk_pool = requested.pool_disk_mb
            if (
                cpus + requested.cpus > grant.pool_cpus
                or memory + requested.memory_mb > grant.pool_memory_mb
                or (disk_pool is not None and disk + requested.disk_mb > disk_pool)
            ):
                wanted = f"{requested.cpus} CPUs/{requested.memory_mb} MiB"
                available = (
                    f"{grant.pool_cpus - cpus} CPUs/{grant.pool_memory_mb - memory} MiB"
                )
                if disk_pool is not None:
                    wanted += f"/{requested.disk_mb} MiB disk"
                    available += f"/{disk_pool - disk} MiB disk"
                raise SetupError(
                    f"sandbox pool exhausted: requested {wanted}; available {available}"
                )

            host = getattr(getattr(grant, "environments", None), "host", None)
            e2b = host.e2b if host is not None and host.backend == "e2b" else None

            def reserve(lease: ResourceLease) -> ResourceLease:
                if lease.run_id != run_id or lease.sandbox_reservation is not None:
                    raise InfrastructureError("sandbox reservation owner changed")
                return lease.model_copy(
                    update={
                        "sandbox_reservation": requested,
                        # Outlives the reservation: recovery keeps sweeping.
                        "sandbox_env_authority": lease.sandbox_env_authority
                        or requested.pool_disk_mb is not None,
                        # Where the run's E2B sandboxes are (never the key).
                        "sandbox_e2b": lease.sandbox_e2b or e2b,
                    }
                )

            mutate(reserve)

    def release_run(self, run_id: str, mutate: LeaseMutator) -> None:
        with self._lock():

            def release(lease: ResourceLease) -> ResourceLease:
                if lease.run_id != run_id:
                    raise InfrastructureError("sandbox reservation owner mismatch")
                if any(
                    child.state != "removed" or child.pending_mutation
                    for child in lease.sandboxes
                ):
                    raise InfrastructureError(
                        "sandbox reservation retained until child cleanup is proven"
                    )
                if retained_sandbox_resources(lease):
                    raise InfrastructureError(
                        "sandbox reservation retained until environment, image "
                        "and builder cleanup is proven"
                    )
                if (
                    lease.work.container_id
                    or lease.work.planned_container
                    or lease.judge.container_id
                    or lease.judge.planned_container
                ):
                    raise InfrastructureError(
                        "sandbox envelope retained until parent cleanup is proven"
                    )
                if lease.sandbox_reservation is None:
                    return lease
                return lease.model_copy(update={"sandbox_reservation": None})

            mutate(release)
