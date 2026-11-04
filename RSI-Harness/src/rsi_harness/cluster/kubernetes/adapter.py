"""Kubernetes scheduling for the same image, resource, and Work/Judge lifecycle.

Each build or run job is a Kubernetes Job of one Pod that plays the part of
a single-node allocation: the Engine inside it runs Work and Judge with
Apptainer, as on LSF and Slurm.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.bluevela.resources import (
    ClusterResources,
    MultiNodeResources,
    derive_resource_plan,
)
from rsi_harness.cluster.config import KubernetesSchedulerProfile
from rsi_harness.cluster.schedulers.kubernetes import (
    KubernetesJobSpec,
    KubernetesScheduler,
    uncovered_paths,
)
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import SetupError
from rsi_harness.models import TaskDefinition

# Kubernetes maps ids 0-65535 into a user-namespace (hostUsers: false) Pod.
_USERNS_MAX_ID = 65535


class KubernetesClusterAdapter(SlurmClusterAdapter):
    """Reuse the Slurm adapter's single-node and segment semantics; submit
    Kubernetes Jobs."""

    scheduler_log_pattern = "kubernetes.%j"
    supports_cpu_work = True
    scheduler_label = "Kubernetes"

    @property
    def _kubernetes(self) -> KubernetesSchedulerProfile:
        scheduler = self.profile.scheduler
        assert isinstance(scheduler, KubernetesSchedulerProfile)
        return scheduler

    def _compile(self, request: ClusterRunRequest) -> TaskDefinition:
        task = self.compiler.compile(request.task_dir, request.options)
        if (
            max(task.gpu_requirement.count, task.verifier.gpu_count)
            > self.profile.resources.gpus_per_node
            or task.require_disjoint_phase_nodes
        ):
            raise SetupError(
                "Kubernetes runs use one Pod on one node; this task needs "
                "several nodes (use Slurm or LSF)"
            )
        definition = super()._compile(request)
        resources = derive_resource_plan(definition, self.profile)
        if resources.multi_node is not None:
            raise SetupError(
                "Kubernetes runs use one Pod on one node; this task needs "
                "several nodes (use Slurm or LSF)"
            )
        scheduler = self._kubernetes
        single = resources.single_node
        if (
            single is not None
            and single.total_gpus
            and scheduler.pod_security == "privileged"
            and scheduler.privileged_devices == "host"
        ):
            raise SetupError(
                "privileged Pods see every GPU on the node, so GPUs cannot be "
                'confined: use pod_security = "userns", or set '
                'privileged_devices = "allocated" where the container runtime '
                "hides host devices (privileged_without_host_devices)"
            )
        return definition

    def _make_scheduler(self) -> KubernetesScheduler:
        scheduler = self._kubernetes
        uid, gid = os.getuid(), os.getgid()
        groups = tuple(sorted(set(os.getgroups())))
        if uid == 0:
            # Privileged root is the node's root; user-namespace root still
            # writes as host root to the shared volumes (idmapped mounts).
            raise SetupError(
                "Kubernetes Pods run as the submitting user, never root: "
                "submit as an unprivileged user"
            )
        if scheduler.pod_security == "userns":
            if max(uid, gid) > _USERNS_MAX_ID:
                which = f"uid {uid}" if uid > _USERNS_MAX_ID else f"gid {gid}"
                raise SetupError(
                    f"{which} is beyond 65535, which a user-namespace Pod "
                    'cannot map: use pod_security = "privileged"'
                )
            # Groups beyond 65535 cannot be mapped into the Pod.
            groups = tuple(group for group in groups if group <= _USERNS_MAX_ID)
        return KubernetesScheduler(
            scheduler, owner=self.profile.owner, run_as=(uid, gid, groups)
        )

    def _validate_runtime_inputs(self, request: ClusterRunRequest) -> None:
        if request.agent_auth is not None and request.agent_auth.value == "local":
            raise SetupError(
                "--agent-auth local reads a login from your home directory, which "
                "a Kubernetes Pod does not have: export the Agent's API key "
                "(it reaches the Engine through the run's environment file)"
            )
        super()._validate_runtime_inputs(request)

    def _run_spec(self, *args: Any, **kwargs: Any) -> KubernetesJobSpec:
        spec = super()._run_spec(*args, **kwargs)
        assert isinstance(spec, KubernetesJobSpec)
        # The Engine reads the Agent's and Judge's credentials from its
        # environment; the build job gets none.
        return spec.model_copy(update={"forward_environment": True})

    def _job_spec(self, **values: Any) -> KubernetesJobSpec:
        return KubernetesJobSpec(**values, request_tmp=self._kubernetes.request_tmp)

    def _build_spec(
        self,
        run_dir: Path,
        name: str,
        resources: ClusterResources | MultiNodeResources,
        script: Path,
    ) -> KubernetesJobSpec:
        spec = BlueVelaClusterAdapter._build_spec(
            self, run_dir, name, resources, script
        )
        assert isinstance(spec, KubernetesJobSpec)
        overrides = self._kubernetes.build
        if overrides is None:
            return spec
        return KubernetesJobSpec.model_validate(
            {
                **spec.model_dump(),
                "image": overrides.image,
                "node_selector": overrides.node_selector,
                "tolerations": overrides.tolerations,
                "gpu_count": overrides.gpus,
            }
        )

    def _check_job_paths(self, agent_paths: tuple[Path, ...]) -> None:
        """The Pod sees only its volumes and image: every path the jobs use
        at the submit host's path must be in one of them."""
        profile = self.profile
        paths: list[Path] = [
            profile.storage.run_root,
            profile.storage.image_cache,
            profile.storage.logs_root,
            profile.storage.hf_home,
            profile.storage.hf_datasets_cache,
            # The Engine runs with this interpreter, its venv and stdlib.
            Path(sys.executable),
            Path(sys.prefix),
            Path(sys.base_prefix),
            *agent_paths,
        ]
        # A Podman or Apptainer in the image is checked by the job drivers.
        if profile.builder.podman_bundle is not None:
            paths.append(profile.builder.podman_bundle)
        apptainer = profile.apptainer
        if apptainer.judge_authority_root is not None:
            paths.append(apptainer.judge_authority_root)
        paths += [
            bind.source for bind in (*apptainer.work_binds, *apptainer.judge_binds)
        ]
        paths += list(apptainer.extra_binds)
        missing = uncovered_paths(self._kubernetes, tuple(dict.fromkeys(paths)))
        if missing:
            raise SetupError(
                "the Kubernetes Pod would not see these paths at the submit "
                f"host's paths: {', '.join(str(path) for path in missing)}; "
                "mount them with [[scheduler.volumes]] or list them in "
                "scheduler.image_paths"
            )
