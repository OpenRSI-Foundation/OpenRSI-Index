"""PBS Pro scheduling for the same image, resource, and Work/Judge lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsi_harness.cluster.batch_adapter import OneHostBatchAdapter
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.config import PBSSchedulerProfile
from rsi_harness.cluster.schedulers.pbs import PBSJobSpec, PBSScheduler


class PBSClusterAdapter(OneHostBatchAdapter):
    scheduler_log_pattern = "pbs.%j"
    supports_cpu_work = True
    scheduler_label = "PBS"

    @property
    def _pbs(self) -> PBSSchedulerProfile:
        scheduler = self.profile.scheduler
        assert isinstance(scheduler, PBSSchedulerProfile)
        return scheduler

    def _make_scheduler(self) -> PBSScheduler:
        scheduler = self._pbs
        return PBSScheduler(
            submit_binary=scheduler.submit_binary,
            status_binary=scheduler.status_binary,
            cancel_binary=scheduler.cancel_binary,
            select_binary=scheduler.select_binary,
        )

    def _job_spec(self, **values: Any) -> PBSJobSpec:
        scheduler = self._pbs
        return PBSJobSpec(
            **values,
            project=scheduler.project,
            chunk_resources=scheduler.chunk_resources,
            chunk_extra=scheduler.chunk_extra,
        )

    def _build_spec(
        self, run_dir: Path, name: str, resources: Any, script: Path
    ) -> PBSJobSpec:
        spec = BlueVelaClusterAdapter._build_spec(
            self, run_dir, name, resources, script
        )
        assert isinstance(spec, PBSJobSpec)
        overrides = self._pbs.build
        if overrides is None:
            return spec
        values = spec.model_dump()
        for source, target in (
            ("queue", "queue"),
            ("account", "group"),
            ("project", "project"),
            ("chunk_resources", "chunk_resources"),
        ):
            value = getattr(overrides, source)
            if value is not None:
                values[target] = value
        values.update(gpu_count=overrides.gpus, exclusive=overrides.exclusive)
        spec = PBSJobSpec.model_validate(values)
        return spec
