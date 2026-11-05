"""Fujitsu TCS (PJM) scheduling for the same image, resource, and Work/Judge
lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsi_harness.cluster.batch_adapter import OneHostBatchAdapter
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.config import PJMSchedulerProfile
from rsi_harness.cluster.schedulers.pjm import PJMJobSpec, PJMScheduler


class PJMClusterAdapter(OneHostBatchAdapter):
    scheduler_log_pattern = "pjm.%j"
    supports_cpu_work = True
    scheduler_label = "PJM"

    @property
    def _pjm(self) -> PJMSchedulerProfile:
        scheduler = self.profile.scheduler
        assert isinstance(scheduler, PJMSchedulerProfile)
        return scheduler

    def _make_scheduler(self) -> PJMScheduler:
        scheduler = self._pjm
        return PJMScheduler(
            submit_binary=scheduler.submit_binary,
            status_binary=scheduler.status_binary,
            cancel_binary=scheduler.cancel_binary,
        )

    def _job_spec(self, **values: Any) -> PJMJobSpec:
        scheduler = self._pjm
        return PJMJobSpec(
            **values,
            gpu_resource=scheduler.gpu_resource,
            cpu_resource=scheduler.cpu_resource,
            memory_resource=scheduler.memory_resource,
            resources=scheduler.resources,
        )

    def _build_spec(
        self, run_dir: Path, name: str, resources: Any, script: Path
    ) -> PJMJobSpec:
        spec = BlueVelaClusterAdapter._build_spec(
            self, run_dir, name, resources, script
        )
        assert isinstance(spec, PJMJobSpec)
        overrides = self._pjm.build
        if overrides is None:
            return spec
        values = spec.model_dump()
        if overrides.resource_group is not None:
            values["queue"] = overrides.resource_group
        if overrides.group is not None:
            values["group"] = overrides.group
        values.update(gpu_count=overrides.gpus, exclusive=overrides.exclusive)
        spec = PJMJobSpec.model_validate(values)
        return spec
