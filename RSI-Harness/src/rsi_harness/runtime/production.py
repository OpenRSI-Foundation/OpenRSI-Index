"""Concrete production composition for CLI run, recovery, and cleanup."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import docker
from docker.errors import APIError, NotFound

from rsi_harness.config import EngineConfig
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopAgentAdapter,
    rsi_loop_agent_environment,
    rsi_loop_runtime_secret_values,
    validate_agent_reasoning_effort,
)
from rsi_harness.models import (
    WORK_FEEDBACK_ROOT,
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerVolumeMount,
    EvaluationRequest,
    ManagedNetwork,
    ManagedWorkdirVolume,
    RootfsSnapshotLease,
    RootfsSnapshotMode,
    RunRequest,
    RunResult,
    SubmissionReport,
    WorkQuiescence,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.coordinator import (
    EmbeddedSubmissionServerFactory,
    ProductionCoordinatorBackend,
    RunCoordinator,
    RunPreparation,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.environment import (
    MissingRuntimeEnvironmentError,
    resolve_runtime_environment,
    runtime_template_name,
)
from rsi_harness.runtime.gpu import (
    NVIDIA_VISIBLE_DEVICES_ENV,
    NvidiaSmiInventory,
    assert_work_gpu_quiescent,
    nvidia_visible_devices_value,
    resolve_gpu_plan,
)
from rsi_harness.runtime.images import DockerImageBuilder
from rsi_harness.runtime.judge import DockerJudgeRuntimeFactory, JudgeRunner
from rsi_harness.runtime.local_auth import (
    AgentAuthMaterial,
    resolve_agent_auth,
)
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    FirewallRuleNotFound,
    NetworkPolicyEnforcer,
    NetworkPolicyLease,
    PinnedEndpoint,
)
from rsi_harness.runtime.recovery import (
    LeaseStore,
    RecoveryManager,
    SnapshotRecoveryAuthority,
)
from rsi_harness.runtime.redaction import redact_exact_values, redact_text
from rsi_harness.runtime.rootfs_snapshot import DockerRootfsSnapshotBackend
from rsi_harness.runtime.snapshot import OverlaySnapshotBackend
from rsi_harness.runtime.workdir_volume import (
    DockerWorkdirVolumeBackend,
    normalized_workdir_volume_container_references,
)
from rsi_harness.runtime.workspace import delete_managed_workspace
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.config import RSILoopConfig, load_config


class _Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def _safe_endpoint_label(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "configured endpoint"
    if parsed.hostname is None:
        return "configured endpoint"
    return parsed.hostname


def _parse_agent_endpoint(value: str) -> Any:
    try:
        return urlsplit(value)
    except ValueError as error:
        raise SetupError(
            "invalid Agent provider/proxy endpoint for configured endpoint"
        ) from error


def _agent_endpoints(
    definition: Any,
    config: RSILoopConfig,
    agent_auth: AgentAuthMaterial | None = None,
) -> tuple[PinnedEndpoint, ...]:
    """Resolve provider/proxy authority before any runtime mutation."""

    selected = create_agent(definition.agent.name, config)
    model = definition.agent.model or config.agent_model or selected.default_model
    environment = rsi_loop_agent_environment(config, selected, model)
    default_provider = (
        environment.get(selected.api_base_env)
        if selected.api_base_env
        else None
    ) or getattr(selected, "default_api_base_url", None)
    providers = (
        agent_auth.provider_endpoints
        if agent_auth is not None and agent_auth.provider_endpoints
        else ((default_provider,) if default_provider else ())
    )
    provider_schemes = {
        _parse_agent_endpoint(provider).scheme for provider in providers
    }
    provider_scheme = (
        next(iter(provider_schemes)) if len(provider_schemes) == 1 else None
    )
    proxy_keys = {
        "http": ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"),
        "https": ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"),
    }.get(
        provider_scheme,
        (
            "http_proxy",
            "HTTP_PROXY",
            "https_proxy",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
        ),
    )
    proxies = tuple(
        dict.fromkeys(
            environment[key]
            for key in proxy_keys
            if environment.get(key)
        )
    )
    if definition.agent.network.mode == "no-network" and not (
        providers or proxies
    ):
        raise SetupError(
            "no-network Agent requires a registered default API base URL, "
            "an explicit API base URL, or an HTTP/HTTPS proxy so provider "
            "access can be pinned"
        )
    endpoints = list(proxies)
    for provider in providers:
        if not proxies or _provider_bypasses_proxy(provider, environment):
            endpoints.append(provider)
    endpoints = list(dict.fromkeys(endpoints))
    for endpoint in endpoints:
        parsed = _parse_agent_endpoint(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise SetupError(
                "invalid Agent provider/proxy endpoint for "
                f"{_safe_endpoint_label(endpoint)}"
            )
    pinned = NetworkPolicyEnforcer(run_id="provider-preflight").pin_endpoints(
        tuple(endpoints)
    )
    ipv4_pinned: list[PinnedEndpoint] = []
    for endpoint in pinned:
        addresses = tuple(
            address
            for address in endpoint.addresses
            if isinstance(address, ipaddress.IPv4Address)
        )
        if not addresses:
            raise SetupError(
                "Agent provider/proxy endpoint has no IPv4-safe address: "
                f"{endpoint.hostname}"
            )
        ipv4_pinned.append(
            PinnedEndpoint(
                hostname=endpoint.hostname,
                port=endpoint.port,
                addresses=addresses,
            )
        )
    result = tuple(ipv4_pinned)
    if definition.agent.network.mode == "allowlist":
        for endpoint in result:
            if not _allowlist_contains_endpoint(
                definition.agent.network.allowlist, endpoint
            ):
                raise SetupError(
                    "Agent allowlist does not explicitly permit provider/proxy "
                    f"endpoint {endpoint.hostname}"
                )
    return result


def _provider_bypasses_proxy(
    provider: str, environment: Mapping[str, str]
) -> bool:
    hostname = urlsplit(provider).hostname
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    try:
        provider_address = ipaddress.ip_address(normalized)
    except ValueError:
        provider_address = None
    entries = tuple(
        value
        for key in ("no_proxy", "NO_PROXY")
        if (value := environment.get(key))
    )
    for entry in entries:
        for raw_token in entry.split(","):
            token = raw_token.strip().lower()
            if not token:
                continue
            if token == "*":
                return True
            token = token.rsplit(":", 1)[0] if token.count(":") == 1 else token
            try:
                network = ipaddress.ip_network(token, strict=False)
            except ValueError:
                network = None
            if provider_address is not None and network is not None:
                if provider_address in network:
                    return True
                continue
            token = token.removeprefix("*.").lstrip(".")
            if normalized == token or normalized.endswith(f".{token}"):
                return True
    return False


def _allowlist_contains_endpoint(
    allowlist: Sequence[str], endpoint: PinnedEndpoint
) -> bool:
    networks = []
    for entry in allowlist:
        normalized = entry.rstrip(".").lower()
        if normalized == endpoint.hostname:
            return True
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            continue
    return all(
        any(address in network for network in networks)
        for address in endpoint.addresses
    )


_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")


def _decode_mountinfo_field(value: str) -> str:
    return _MOUNTINFO_ESCAPE.sub(
        lambda match: chr(int(match.group(1), 8)), value
    )


def _mountinfo_references_workspace(text: str, workspace: Path) -> bool:
    """Fail closed if any mount target/source/layer references a workspace."""

    root = workspace.resolve(strict=False)

    def references(raw_value: str) -> bool:
        decoded = _decode_mountinfo_field(raw_value)
        candidate = Path(decoded)
        if not candidate.is_absolute():
            return False
        resolved = candidate.resolve(strict=False)
        return resolved == root or root in resolved.parents

    lines = text.splitlines()
    if not lines:
        raise SetupError("cannot inspect workspace mounts: mountinfo is empty")
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as error:
            raise SetupError(
                "cannot inspect workspace mounts: malformed mountinfo"
            ) from error
        if len(fields) < 7 or separator + 3 >= len(fields):
            raise SetupError(
                "cannot inspect workspace mounts: malformed mountinfo"
            )
        # mount root, mount target, and filesystem source may each reference a
        # managed tree.  Overlay/FUSE paths live in keyed mount/super options.
        if any(
            references(value)
            for value in (fields[3], fields[4], fields[separator + 2])
        ):
            return True
        for options in (fields[5], fields[separator + 3]):
            for option in options.split(","):
                key, present, value = option.partition("=")
                if not present or key not in {"lowerdir", "upperdir", "workdir"}:
                    continue
                values = value.split(":") if key == "lowerdir" else (value,)
                if any(references(candidate) for candidate in values):
                    return True
    return False


class ProductionRecoveryBackend:
    """Bind recovery authority to Docker labels, overlay manifests and iptables."""

    def __init__(self, client: Any, snapshot: Any, firewall: Any) -> None:
        self._client = client
        self._snapshot = snapshot
        self._firewall = firewall

    @staticmethod
    def _filter(labels: Mapping[str, str]) -> dict[str, list[str]]:
        return {"label": [f"{key}={value}" for key, value in labels.items()]}

    def list_containers(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        found = self._client.containers.list(all=True, filters=self._filter(labels))
        return tuple((item.id, self._container_state(item)) for item in found)

    @staticmethod
    def _container_state(container: Any) -> Mapping[str, Any]:
        container.reload()
        state = container.attrs.get("State") or {}
        config = container.attrs.get("Config") or {}
        labels = config.get("Labels") or {}
        return {
            "running": bool(state.get("Running", False)),
            "paused": bool(state.get("Paused", False)),
            "labels": dict(labels),
        }

    def inspect_container(self, container_id: str) -> Mapping[str, Any] | None:
        try:
            return self._container_state(self._client.containers.get(container_id))
        except NotFound:
            return None

    def stop_container(self, container_id: str) -> None:
        try:
            self._client.containers.get(container_id).stop(timeout=10)
        except NotFound:
            return

    def remove_container(self, container_id: str) -> None:
        try:
            self._client.containers.get(container_id).remove(force=False)
        except NotFound:
            return

    def list_images(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        found = self._client.images.list(filters=self._filter(labels))
        return tuple((item.id, self._image_state(item)) for item in found)

    @staticmethod
    def _image_state(image: Any) -> Mapping[str, Any]:
        image.reload()
        attrs = image.attrs
        config = attrs.get("Config") or {}
        labels = config.get("Labels") or {}
        repo_tags = attrs.get("RepoTags") or ()
        return {
            "id": attrs.get("Id"),
            "labels": dict(labels),
            "repo_tags": tuple(repo_tags),
        }

    def inspect_image(self, image_id: str) -> Mapping[str, Any] | None:
        try:
            return self._image_state(self._client.images.get(image_id))
        except NotFound:
            return None

    def image_in_use(self, image_id: str) -> bool:
        found = self._client.containers.list(
            all=True, filters={"ancestor": image_id}
        )
        if not isinstance(found, list):
            raise TypeError("Docker returned malformed image references")
        return bool(found)

    def remove_image(self, image_id: str) -> None:
        try:
            self._client.images.remove(image_id, force=False)
        except NotFound:
            return

    def list_volumes(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        found = self._client.volumes.list(filters=self._filter(labels))
        if not isinstance(found, list):
            raise TypeError("Docker returned malformed volume results")
        return tuple(
            (state["name"], state)
            for state in (self._volume_state(volume) for volume in found)
        )

    def inspect_volume(self, name: str) -> Mapping[str, Any] | None:
        try:
            return self._volume_state(self._client.volumes.get(name))
        except NotFound:
            return None

    def volume_in_use(self, name: str) -> bool:
        return bool(self._volume_container_references(name))

    def remove_volume(self, name: str) -> None:
        try:
            self._client.volumes.get(name).remove(force=False)
        except NotFound:
            return

    def _volume_state(self, volume: Any) -> Mapping[str, Any]:
        volume.reload()
        attrs = volume.attrs
        name = attrs.get("Name")
        if not isinstance(name, str):
            raise TypeError("Docker returned malformed volume name")
        labels = attrs.get("Labels")
        options = attrs.get("Options")
        scope = attrs.get("Scope")
        if labels is None:
            labels = {}
        if options is None:
            options = {}
        if scope is None:
            scope = ""
        if not isinstance(labels, Mapping) or not isinstance(options, Mapping):
            raise TypeError("Docker returned malformed volume attributes")
        if not isinstance(scope, str):
            raise TypeError("Docker returned malformed volume scope")
        return {
            "name": name,
            "driver": attrs.get("Driver"),
            "labels": dict(labels),
            "options": dict(options),
            "scope": scope,
            "container_references": normalized_workdir_volume_container_references(
                self._client, name
            ),
        }

    def _volume_container_references(
        self, name: str
    ) -> tuple[Mapping[str, Any], ...]:
        return normalized_workdir_volume_container_references(self._client, name)

    def list_networks(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        found = self._client.networks.list(filters=self._filter(labels))
        result = []
        for network in found:
            network.reload()
            state = dict(network.attrs)
            state["labels"] = dict(network.attrs.get("Labels") or {})
            result.append((network.id, state))
        return tuple(result)

    def network_in_use(self, network_id: str) -> bool:
        try:
            network = self._client.networks.get(network_id)
        except NotFound:
            return False
        network.reload()
        return bool(network.attrs.get("Containers"))

    def remove_network(self, network_id: str) -> None:
        try:
            self._client.networks.get(network_id).remove()
        except NotFound:
            return

    @staticmethod
    def is_mounted(path: Path) -> bool:
        return path.is_mount()

    def discover_snapshot_leases(
        self,
        *,
        run_id: str,
        round_id: str | None,
        lease_id: str | None,
    ) -> tuple[SnapshotRecoveryAuthority, ...]:
        return self._snapshot.discover_snapshot_leases(
            run_id=run_id, round_id=round_id, lease_id=lease_id
        )

    def release_snapshot(self, authority: SnapshotRecoveryAuthority) -> None:
        self._snapshot.release_snapshot(authority)

    def unpause_container(self, container_id: str) -> None:
        try:
            self._client.containers.get(container_id).unpause()
        except NotFound:
            return

    def remove_policy(self, rule_id: str) -> None:
        try:
            self._firewall.remove(rule_id)
        except FirewallRuleNotFound:
            return

    def policy_exists(self, rule_id: str) -> bool:
        return bool(self._firewall.exists(rule_id))

    @staticmethod
    def workspace_is_mounted(workspace: Path) -> bool:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text()
        except OSError as error:
            raise SetupError(f"cannot inspect workspace mounts: {error}") from error
        return _mountinfo_references_workspace(mountinfo, workspace)

    def delete_workspace(
        self,
        workspace: Path,
        *,
        image_ref: str,
        run_id: str,
        task_id: str,
    ) -> None:
        delete_managed_workspace(
            self._client,
            workspace=workspace,
            image_ref=image_ref,
            run_id=run_id,
            task_id=task_id,
        )


class _ProductionRunComposition:
    def __init__(
        self,
        *,
        client: Any,
        data_root: Path,
        logs_root: Path,
        inventory: Any,
        rsi_loop_config: RSILoopConfig,
        snapshot: Any,
        firewall: Any,
        bind_host: str,
        bridge_gateway: str,
        omit_gpu_device_requests_for_tests: bool,
        agent_adapter_factory: Callable[[RSILoopConfig, Any], Any],
        quiescence_checker: Callable[[Any, ContainerRef], None] | None,
        api_endpoints: tuple[PinnedEndpoint, ...],
        agent_secret_env: Mapping[str, str],
        verifier_secret_env: Mapping[str, str],
        agent_auth: AgentAuthMaterial | None = None,
        event_callback: Callable[[str, object], None] | None = None,
        preparation: RunPreparation | None = None,
        compiler: HarborTaskCompiler | None = None,
    ) -> None:
        del inventory
        self.client = client
        self.data_root = data_root
        self.logs_root = logs_root
        self.rsi_loop_config = rsi_loop_config
        self.snapshot = snapshot
        self.firewall = firewall
        self.bind_host = bind_host
        self.bridge_gateway = bridge_gateway
        self.omit_gpu = omit_gpu_device_requests_for_tests
        self.agent_adapter_factory = agent_adapter_factory
        self.quiescence_checker = quiescence_checker or (
            lambda allocation, ref: assert_work_gpu_quiescent(
                allocation, self.client.containers.get(ref.container_id)
            )
        )
        self.api_endpoints = api_endpoints
        self.agent_secret_env = dict(agent_secret_env)
        self.verifier_secret_env = dict(verifier_secret_env)
        self.agent_auth = agent_auth
        self.event_callback = event_callback or (lambda _name, _value: None)
        self.preparation = preparation
        self.agent_output_secrets = rsi_loop_runtime_secret_values(rsi_loop_config)
        if agent_auth is not None:
            self.agent_output_secrets.update(agent_auth.secret_values)
        self.compiler = compiler or HarborTaskCompiler()
        self.rootfs_snapshots = DockerRootfsSnapshotBackend(client)
        self.workdir_volumes = DockerWorkdirVolumeBackend(client)
        self.definition: Any = (
            None if preparation is None else preparation.definition
        )
        self.plan: Any = None
        self.run_id = ""
        self.artifacts: RunArtifactWriter | None = None
        self.enforcer: NetworkPolicyEnforcer | None = None
        self.work_runtime: DockerContainerRuntime | None = None
        self.work_provisioner: DockerContainerRuntime | None = None
        self.agent: Any = None
        self.started_server: Any = None
        self.planned_work_network: str | None = None
        self.work_network: ManagedNetwork | None = None
        self.planned_workdir_volume: ManagedWorkdirVolume | None = None
        self.workdir_volume: ManagedWorkdirVolume | None = None
        self.planned_work_policy: NetworkPolicyLease | None = None

    def backend(self) -> ProductionCoordinatorBackend:
        return ProductionCoordinatorBackend(
            compiler=self.compiler,
            allocator=self.allocate,
            image_preparer=self.prepare_images,
            plan_preparer=self.prepare_plan,
            artifact_starter=self.start_artifacts,
            server_starter=self.start_server,
            network_planner=self.plan_network,
            network_creator=self.create_network,
            network_remover=self.remove_network,
            workdir_volume_planner=self.plan_workdir_volume,
            workdir_volume_creator=self.create_workdir_volume,
            workdir_volume_attester=self.attest_workdir_volume,
            workdir_volume_remover=self.remove_workdir_volume,
            work_name_planner=self.plan_work_container,
            work_creator=self.create_work,
            work_feedback_attester=self.attest_work_feedback_mount,
            work_policy_planner=self.plan_work_policy,
            work_policy_installer=self.install_work_policy,
            work_policy_remover=self.remove_work_policy,
            work_starter=self.start_work,
            work_remover=self.remove_work,
            hook_installer=self.install_hooks,
            agent_preparer=self.prepare_agent,
            agent_runner=self.run_agent,
            agent_stopper=self.stop_agent,
            work_quiescer=self.quiesce_work,
            retained_work_planner=self.plan_retained_work,
            work_retainer=self.retain_work,
            retained_work_releaser=self.release_retained_work,
            evaluator=self.evaluate,
            event_recorder=self.event_callback,
            preparation=self.preparation,
        )

    def allocate(self, definition: Any, selectors: Sequence[str]) -> Any:
        preparation = self.preparation
        if preparation is None:
            raise SetupError("production GPU allocation requires prevalidated run")
        if definition is not preparation.definition:
            raise SetupError("compiled task changed after production prevalidation")
        if tuple(selectors) != tuple(preparation.request.gpu_selectors):
            raise SetupError("GPU selectors changed after production prevalidation")
        return preparation.gpu_plan

    def prepare_images(self, definition: Any) -> Any:
        selected = create_agent(definition.agent.name, self.rsi_loop_config)
        images = DockerImageBuilder(
            self.client,
            rsi_loop_config=self.rsi_loop_config,
        ).prepare(definition, agent=selected)
        self.event_callback("images_ready", None)
        return images

    def prepare_plan(
        self,
        definition: Any,
        images: Any,
        gpu_plan: Any,
        request: RunRequest,
        run_id: str,
    ) -> Any:
        self.run_id = run_id
        self.event_callback("run_started", run_id)
        if self.preparation is not None:
            if definition is not self.preparation.definition:
                raise SetupError("RunPlan definition differs from prevalidated run")
            if gpu_plan is not self.preparation.gpu_plan:
                raise SetupError("RunPlan GPU plan differs from prevalidated run")
        else:
            agent_updates = {}
            if request.model is not None:
                agent_updates["model"] = request.model
            if request.reasoning_effort is not None:
                agent_updates["reasoning_effort"] = request.reasoning_effort
            if agent_updates:
                definition = definition.model_copy(
                    update={
                        "agent": definition.agent.model_copy(
                            update=agent_updates
                        )
                    }
                )
        run_root = (self.data_root / run_id).resolve()
        run_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        run_root.chmod(0o700)
        paths = (
            request.paths.model_copy(
                update={
                    "root": run_root,
                    "workspace": run_root / "workspace",
                    "logs": self.logs_root,
                }
            )
            if request.paths is not None
            else None
        )
        if paths is None:
            from rsi_harness.models import RunPaths

            paths = RunPaths(
                root=run_root,
                workspace=run_root / "workspace",
                logs=self.logs_root,
            )
        self.plan = self.compiler.finalize(
            definition, images, gpu_plan, paths, "docker-rootfs"
        )
        self.definition = self.plan.task
        metadata_staging = run_root / "rsi-loop-task"
        metadata = self.compiler.write_rsi_loop_metadata(
            self.plan.task, metadata_staging
        )
        generated = self.data_root / "generated-tasks"
        generated.mkdir(parents=True, exist_ok=True)
        target = generated / f"{self.plan.task.task_id}.json"
        temporary = generated / f".{self.plan.task.task_id}.{run_id}.tmp"
        temporary.write_bytes(metadata.read_bytes())
        os.replace(temporary, target)
        self.enforcer = NetworkPolicyEnforcer(
            run_id=run_id,
            firewall=self.firewall,
            engine_destinations=(self.bridge_gateway,),
        )
        return self.plan

    def start_artifacts(self, plan: Any, run_id: str) -> RunArtifactWriter:
        self.artifacts = RunArtifactWriter(plan, run_id=run_id)
        self.artifacts.start()
        return self.artifacts

    def start_server(self, evaluator: Any, artifacts: Any, clock: Any) -> Any:
        self.started_server = EmbeddedSubmissionServerFactory(
            bind_host=self.bind_host,
            port=0,
            bridge_gateway=self.bridge_gateway,
        )(evaluator, artifacts, clock)
        return self.started_server

    def _runtime_common(self, role: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.definition.task_id,
            "role": role,
            "task_source_dir": self.definition.source_dir,
            "allowed_mount_roots": (self.data_root, self.logs_root),
            "omit_gpu_device_requests_for_tests": self.omit_gpu,
        }

    def plan_network(self, plan: Any, run_id: str) -> str:
        del plan, run_id
        self.work_provisioner = DockerContainerRuntime(
            self.client, **self._runtime_common("work")
        )
        self.planned_work_network = self.work_provisioner.planned_network_name(
            "phase"
        )
        return self.planned_work_network

    def create_network(self, plan: Any, run_id: str, planned: str) -> ManagedNetwork:
        del plan, run_id
        if self.work_provisioner is None:
            raise RuntimeError("Work network was not planned")
        if planned != self.planned_work_network:
            raise SetupError("Work network durable plan identity mismatch")
        network = self.work_provisioner.create_network("phase", internal=False)
        # Return immediately after the external mutation. The coordinator must
        # record the actual identity before any fallible post-create setup.
        self.work_network = network
        return network

    def remove_network(self, network: ManagedNetwork) -> None:
        if self.work_provisioner is None:
            raise RuntimeError("Work network provisioner is unavailable")
        self.work_provisioner.remove_network(network)
        if self.work_network == network:
            self.work_network = None

    def plan_workdir_volume(
        self, plan: Any, run_id: str
    ) -> ManagedWorkdirVolume | None:
        if plan.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS:
            return None
        if plan.rootfs_snapshot_mode is not RootfsSnapshotMode.SPLIT_WORKDIR:
            raise SetupError("unsupported rootfs snapshot mode")
        planned = self.workdir_volumes.plan(
            run_id=run_id,
            task_id=plan.task.task_id,
            target=plan.workdir,
        )
        self.planned_workdir_volume = planned
        return planned

    def create_workdir_volume(
        self,
        plan: Any,
        run_id: str,
        planned: ManagedWorkdirVolume,
    ) -> ManagedWorkdirVolume:
        if plan.rootfs_snapshot_mode is not RootfsSnapshotMode.SPLIT_WORKDIR:
            raise SetupError(
                "managed WORKDIR volume creation requires split-workdir mode"
            )
        if planned != self.planned_workdir_volume:
            raise SetupError("WORKDIR volume durable plan identity mismatch")
        if (
            planned.run_id != run_id
            or planned.task_id != plan.task.task_id
            or planned.target != plan.workdir
            or planned.snapshot_mode is not plan.rootfs_snapshot_mode
        ):
            raise SetupError("WORKDIR volume durable plan provenance mismatch")
        actual = self.workdir_volumes.create(planned)
        self.workdir_volume = actual
        return actual

    def remove_workdir_volume(self, volume: ManagedWorkdirVolume) -> None:
        if self.workdir_volume != volume:
            raise SetupError("WORKDIR volume cleanup identity mismatch")
        self.workdir_volumes.remove(volume)
        self.workdir_volume = None
        self.planned_workdir_volume = None

    def attest_workdir_volume(
        self, work: ContainerRef, volume: ManagedWorkdirVolume
    ) -> None:
        if self.workdir_volume != volume:
            raise InfrastructureError(
                "recovery_required: Work WORKDIR mount attestation authority "
                "differs from the created volume"
            )
        if self.work_runtime is None:
            raise InfrastructureError(
                "recovery_required: Work runtime is unavailable for mount attestation"
            )
        self.work_runtime.attest_workdir_volume_mount(
            work,
            ContainerVolumeMount(
                volume=volume,
                target=volume.target,
                read_only=False,
            ),
        )

    def plan_work_container(self, plan: Any, run_id: str) -> str:
        del plan, run_id
        if self.work_network is None or self.planned_work_network is None:
            raise RuntimeError("Work network actual authority is unavailable")
        if self.work_network.name != self.planned_work_network:
            raise SetupError("created Work network differs from durable plan")
        if self.enforcer is None:
            raise RuntimeError("network enforcer was not initialized")
        if self.artifacts is None:
            raise RuntimeError("Work feedback artifacts were not initialized")
        self.work_runtime = DockerContainerRuntime(
            self.client,
            **self._runtime_common("work"),
            network=self.work_network,
            network_policy_enforcer=self.enforcer,
            staging_dir=self.data_root / self.run_id / "staging",
            work_feedback_dir=self.artifacts.feedback_root,
        )
        try:
            self.agent = self.agent_adapter_factory(
                self.rsi_loop_config, self.work_runtime
            )
        except Exception as error:
            raise self._agent_failure(error) from None
        return self.work_runtime.planned_container_name("main")

    def create_work(
        self,
        plan: Any,
        run_id: str,
        network: ManagedNetwork,
        planned_name: str,
        workdir_volume: ManagedWorkdirVolume | None,
    ) -> ContainerRef:
        del run_id, network
        try:
            environment = self._resolved_agent_environment(plan)
        except MissingRuntimeEnvironmentError as error:
            raise self._agent_failure(
                InfrastructureError(
                    "missing Agent runtime environment variable "
                    f"'{error.name}'"
                )
            ) from None
        container_environment = {
            key: environment[key]
            for key, template in plan.task.agent.environment
            if key != NVIDIA_VISIBLE_DEVICES_ENV
            and runtime_template_name(template) is None
        }
        container_environment.setdefault("HOME", "/home/agent")
        try:
            if self.work_runtime is None or self.artifacts is None:
                raise RuntimeError("Work runtime feedback authority is unavailable")
            if plan.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR:
                if (
                    workdir_volume is None
                    or workdir_volume != self.workdir_volume
                    or workdir_volume.target != plan.workdir
                ):
                    raise SetupError(
                        "split-workdir Work requires exact managed volume authority"
                    )
            elif (
                plan.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS
                and workdir_volume is not None
            ):
                raise SetupError(
                    "full-rootfs Work cannot carry WORKDIR volume authority"
                )
            volume_mounts = (
                (
                    ContainerVolumeMount(
                        volume=workdir_volume,
                        target=plan.workdir,
                        read_only=False,
                    ),
                )
                if plan.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR
                else ()
            )
            return self.work_runtime.create(
                ContainerSpec(
                    image=plan.images.work_ref,
                    command=("sleep", "infinity"),
                    workdir=plan.workdir,
                    user="0",
                    environment=tuple(sorted(container_environment.items())),
                    mounts=(
                        ContainerMount(
                            source=self.artifacts.feedback_root,
                            target=WORK_FEEDBACK_ROOT,
                            read_only=True,
                        ),
                    ),
                    volume_mounts=volume_mounts,
                    tmpfs=(
                        tuple(mount.tmpfs for mount in self.agent_auth.mounts)
                        if self.agent_auth is not None
                        else ()
                    ),
                    gpu_allocation=plan.gpu_plan.work,
                    shm_size=plan.task.service.shm_size,
                    cpus=plan.task.service.cpus,
                    memory_mb=plan.task.service.memory_mb,
                    storage_mb=plan.task.service.storage_mb,
                ),
                planned_name=planned_name,
            )
        except Exception as error:
            raise self._agent_failure(error) from None

    def attest_work_feedback_mount(self, work: ContainerRef) -> None:
        if self.work_runtime is None:
            raise InfrastructureError(
                "recovery_required: Work runtime is unavailable for feedback "
                "mount attestation"
            )
        self.work_runtime.attest_work_feedback_mount(work)

    def _control_endpoint(self) -> tuple[str, ...]:
        if self.started_server is None:
            raise RuntimeError("submission server was not started")
        return (self.started_server.endpoint.url,)

    def _api_endpoints(self) -> tuple[PinnedEndpoint, ...]:
        return self.api_endpoints

    def plan_work_policy(
        self, plan: Any, work: ContainerRef, network: ManagedNetwork
    ) -> str:
        if self.enforcer is None:
            raise RuntimeError("network enforcer is unavailable")
        self.planned_work_policy = self.enforcer.plan(
            work,
            plan.task.agent.network,
            network=network,
            api_endpoints=self._api_endpoints(),
            control_endpoints=self._control_endpoint(),
        )
        return self.planned_work_policy.rule_id

    def install_work_policy(
        self, plan: Any, work: ContainerRef, network: ManagedNetwork
    ) -> str:
        if self.enforcer is None or self.work_runtime is None:
            raise RuntimeError("Work policy runtime is unavailable")
        if self.planned_work_policy is None:
            raise RuntimeError("Work policy was not planned")
        lease = self.enforcer.apply(
            work,
            plan.task.agent.network,
            network=network,
            api_endpoints=self._api_endpoints(),
            control_endpoints=self._control_endpoint(),
            planned=self.planned_work_policy,
        )
        try:
            self.work_runtime.install_network_policy(work, lease)
        except BaseException as primary:
            try:
                self.enforcer.cleanup(lease)
            except BaseException as rollback_error:
                raise InfrastructureError(
                    "recovery_required: installed Work network policy rollback "
                    f"is unproven for {lease.rule_id}: "
                    f"{redact_text(str(rollback_error))}"
                ) from primary
            raise
        return lease.rule_id

    def remove_work_policy(self, rule_id: str) -> None:
        if self.enforcer is None or self.planned_work_policy is None:
            raise RuntimeError("Work policy is unavailable")
        if self.planned_work_policy.rule_id != rule_id:
            raise SetupError("Work policy cleanup identity mismatch")
        self.enforcer.cleanup(self.planned_work_policy)

    def start_work(self, work: ContainerRef) -> None:
        if self.work_runtime is None:
            raise RuntimeError("Work runtime is unavailable")
        self.work_runtime.start(work)
        self.event_callback("work_started", work.container_id)

    def remove_work(self, work: ContainerRef) -> None:
        if self.work_runtime is None:
            raise RuntimeError("Work runtime is unavailable")
        self.work_runtime.remove(work)

    def install_hooks(
        self, plan: Any, work: ContainerRef, submit_url: str, token: str
    ) -> None:
        self.agent_output_secrets.update((submit_url, token))
        try:
            if self.agent is None or self.work_runtime is None:
                raise RuntimeError("Agent adapter is unavailable")
            self.agent.install_hooks(
                AgentHookRequest(
                    run_plan=plan,
                    container=work,
                    submit_url=submit_url,
                    token=token,
                )
            )
            if self.agent_auth is not None:
                self.work_runtime.inject_agent_auth(work, self.agent_auth)
        except Exception as error:
            raise self._agent_failure(error) from None

    def prepare_agent(
        self, plan: Any, max_submissions: int | None = None
    ) -> Any:
        try:
            if self.agent is None or self.artifacts is None:
                raise RuntimeError("Agent artifacts are unavailable")
            prepared = self.agent.prepare(
                AgentPrepareRequest(
                    run_plan=plan,
                    prompt_path=self.artifacts.root / "agent_prompt.md",
                    max_submissions=max_submissions,
                )
            )
            resolved = self._resolved_agent_environment(plan)
            exec_environment = {
                key: resolved[key]
                for key, template in plan.task.agent.environment
                if key != NVIDIA_VISIBLE_DEVICES_ENV
                and runtime_template_name(template) is not None
            }
            exec_environment.update(prepared.environment)
            exec_environment[NVIDIA_VISIBLE_DEVICES_ENV] = (
                nvidia_visible_devices_value(plan.gpu_plan.work)
            )
            return replace(
                prepared,
                environment=tuple(sorted(exec_environment.items())),
            )
        except Exception as error:
            raise self._agent_failure(error) from None

    def _resolved_agent_environment(self, plan: Any) -> dict[str, str]:
        environment = resolve_runtime_environment(
            tuple(
                (key, value)
                for key, value in plan.task.agent.environment
                if key != NVIDIA_VISIBLE_DEVICES_ENV
            ),
            self.agent_secret_env,
        )
        secret_names = set(plan.task.agent.secret_env_names)
        self.agent_output_secrets.update(
            environment[key]
            for key, template in plan.task.agent.environment
            if key != NVIDIA_VISIBLE_DEVICES_ENV
            and runtime_template_name(template) in secret_names
            and environment[key]
        )
        return environment

    def run_agent(
        self, prepared: Any, work: ContainerRef, timeout: float | None
    ) -> AgentRunResult:
        try:
            if self.agent is None or self.artifacts is None:
                raise RuntimeError("Agent runtime is unavailable")
            agent_name = getattr(prepared, "agent_name", None)
            if agent_name is None and self.definition is not None:
                agent_name = self.definition.agent.name
            self.event_callback(
                "agent_started",
                {
                    "name": agent_name or "Agent",
                    "timeout_seconds": timeout,
                    "output_path": self.artifacts.root / "agent_output.txt",
                },
            )
            result = self.agent.run(
                AgentRunRequest(
                    prepared=prepared,
                    container=work,
                    timeout_seconds=timeout,
                    output_path=self.artifacts.root / "agent_output.txt",
                    output_redact_values=tuple(sorted(self.agent_output_secrets)),
                    output_callback=lambda value: self.event_callback(
                        "agent_output", value
                    ),
                )
            )
            safe_output = redact_text(
                redact_exact_values(result.output, self.agent_output_secrets)
            )
            safe_result = AgentRunResult(
                exit_code=result.exit_code,
                output=safe_output,
                timed_out=result.timed_out,
                output_truncated=result.output_truncated,
                full_output_captured=result.full_output_captured,
                cancelled=result.cancelled,
            )
            if not result.full_output_captured:
                self.artifacts.root.joinpath("agent_output.txt").write_text(
                    safe_output
                )
            self.artifacts.root.joinpath("run_agent.log").write_text(safe_output)
            self.event_callback(
                "agent_finished",
                {
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "cancelled": result.cancelled,
                },
            )
        except Exception as error:
            raise self._agent_failure(error) from None
        cleanup_error = self._clear_agent_bindings()
        if cleanup_error is not None:
            raise cleanup_error from None
        return safe_result

    def stop_agent(self, work: ContainerRef) -> None:
        try:
            if self.work_runtime is None:
                raise RuntimeError("Work runtime is unavailable")
            self.work_runtime.stop(work)
        except Exception as error:
            raise self._agent_failure(error) from None
        cleanup_error = self._clear_agent_bindings()
        if cleanup_error is not None:
            raise cleanup_error from None

    def quiesce_work(self, work: ContainerRef) -> WorkQuiescence:
        """Pause a running Work container or prove it is already stopped."""
        if self.work_runtime is None:
            raise RuntimeError("Work runtime is unavailable")
        try:
            container = self.client.containers.get(work.container_id)
            container.reload()
            state = container.attrs.get("State")
        except NotFound as error:
            raise InfrastructureError(
                "Work container disappeared before final retention"
            ) from error
        except (AttributeError, KeyError, TypeError) as error:
            raise InfrastructureError(
                "ambiguous Work container state before final retention"
            ) from error
        if not isinstance(state, dict):
            raise InfrastructureError(
                "ambiguous Work container state before final retention"
            )
        running = state.get("Running")
        paused = state.get("Paused")
        if not isinstance(running, bool) or not isinstance(paused, bool):
            raise InfrastructureError(
                "ambiguous Work container state before final retention"
            )
        if running is False:
            if paused:
                raise InfrastructureError(
                    "ambiguous stopped and paused Work container state"
                )
            return WorkQuiescence.STOPPED
        if paused:
            return WorkQuiescence.PAUSED
        self.work_runtime.pause(work)
        try:
            container.reload()
            state = container.attrs.get("State")
        except (APIError, AttributeError, KeyError, TypeError) as error:
            raise InfrastructureError(
                "ambiguous Work container state after final pause"
            ) from error
        if not isinstance(state, dict):
            raise InfrastructureError(
                "ambiguous Work container state after final pause"
            )
        running = state.get("Running")
        paused = state.get("Paused")
        if running is False and paused is False:
            return WorkQuiescence.STOPPED
        if running is True and paused is True:
            return WorkQuiescence.PAUSED
        raise InfrastructureError(
            "ambiguous Work container state after final pause"
        )

    def plan_retained_work(self, plan: Any, work: ContainerRef) -> str:
        del work
        return self.rootfs_snapshots.planned_ref(
            run_id=self.run_id,
            task_id=plan.task.task_id,
            round_id="final",
            purpose="retained-work",
        )

    def retain_work(
        self, plan: Any, work: ContainerRef, planned_ref: str
    ) -> RootfsSnapshotLease:
        return self.rootfs_snapshots.acquire(
            work,
            run_id=self.run_id,
            task_id=plan.task.task_id,
            round_id="final",
            purpose="retained-work",
            planned_ref=planned_ref,
        )

    def release_retained_work(self, retained: RootfsSnapshotLease) -> None:
        self.rootfs_snapshots.release(retained)

    def _agent_secret_snapshot(self) -> set[str]:
        return {
            *rsi_loop_runtime_secret_values(self.rsi_loop_config),
            *self.agent_output_secrets,
        }

    def _safe_agent_error(
        self,
        error: BaseException,
        *,
        secret_values: set[str] | None = None,
    ) -> InfrastructureError:
        values = (
            self._agent_secret_snapshot()
            if secret_values is None
            else secret_values
        )
        safe_error = redact_text(redact_exact_values(str(error), values))
        return InfrastructureError(safe_error or type(error).__name__)

    def _agent_failure(self, error: BaseException) -> InfrastructureError:
        secret_values = self._agent_secret_snapshot()
        primary = self._safe_agent_error(error, secret_values=secret_values)
        cleanup = self._clear_agent_bindings(secret_values=secret_values)
        if cleanup is None:
            return primary
        return InfrastructureError(
            f"{primary}; Agent transient binding cleanup failed: {cleanup}"
        )

    def _clear_agent_bindings(
        self, *, secret_values: set[str] | None = None
    ) -> InfrastructureError | None:
        values = (
            self._agent_secret_snapshot()
            if secret_values is None
            else secret_values
        )
        try:
            if self.agent is not None:
                clear = getattr(self.agent, "clear_transient_bindings", None)
                if callable(clear):
                    clear()
        except Exception as error:
            # A custom adapter that cannot prove it dropped control authority
            # is removed from the composition so no later stage can reuse it.
            self.agent = None
            return self._safe_agent_error(error, secret_values=values)
        finally:
            self.agent_output_secrets.clear()
        return None

    def evaluate(self, request: EvaluationRequest, observer: Any) -> SubmissionReport:
        if self.work_runtime is None or self.enforcer is None or self.artifacts is None:
            raise RuntimeError("Judge production composition is incomplete")
        factory = DockerJudgeRuntimeFactory(
            self.client,
            run_id=self.run_id,
            task_id=self.definition.task_id,
            task_source_dir=self.definition.source_dir,
            allowed_mount_roots=(self.data_root, self.logs_root),
            network_policy_enforcer=self.enforcer,
            lifecycle_observer=observer,
            work_container=request.work_container,
            omit_gpu_device_requests_for_tests=self.omit_gpu,
        )
        self.event_callback("judge_started", {"round_id": request.round_id})
        runner = JudgeRunner(
            run_id=self.run_id,
            workdir_volume=self.workdir_volume,
            work_runtime=self.work_runtime,
            judge_runtime_factory=factory,
            snapshot_backend=self.rootfs_snapshots,
            artifact_writer=self.artifacts,
            quiescence_checker=self.quiescence_checker,
            lifecycle_observer=observer,
            verifier_secret_env=self.verifier_secret_env,
            event_callback=self.event_callback,
        )
        report = runner.evaluate(request)
        self.event_callback(
            "judge_finished",
            {
                "round_id": report.round_id,
                "status": report.status.value,
                "score": report.score,
                "runtime_seconds": report.duration_seconds,
            },
        )
        return report


class ProductionRuntimeServices:
    """Application service used by the CLI; all defaults are production ports."""

    def __init__(
        self,
        *,
        data_root: Path,
        logs_root: Path,
        docker_client: Any | None = None,
        inventory: Any | None = None,
        rsi_loop_config: RSILoopConfig | None = None,
        snapshot_backend: Any | None = None,
        firewall_backend: Any | None = None,
        coordinator_factory: Callable[..., Any] = RunCoordinator,
        bind_host: str = "0.0.0.0",
        bridge_gateway: str | None = None,
        omit_gpu_device_requests_for_tests: bool = False,
        agent_adapter_factory: Callable[[RSILoopConfig, Any], Any] | None = None,
        quiescence_checker: Callable[[Any, ContainerRef], None] | None = None,
        engine_config: EngineConfig | None = None,
        event_callback: Callable[[str, object], None] | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.logs_root = Path(logs_root).expanduser().resolve()
        self.client = docker_client
        self.inventory = inventory or NvidiaSmiInventory()
        self.rsi_loop_config = rsi_loop_config or load_config()
        self.snapshot = snapshot_backend
        self.firewall = firewall_backend
        self.coordinator_factory = coordinator_factory
        self.bind_host = bind_host
        self.bridge_gateway = bridge_gateway
        self.omit_gpu = omit_gpu_device_requests_for_tests
        self.agent_adapter_factory = agent_adapter_factory or (
            lambda config, runtime: RSILoopAgentAdapter(config, runtime=runtime)
        )
        self.quiescence_checker = quiescence_checker
        self.event_callback = event_callback
        configured_agent_env = {} if engine_config is None else engine_config.secret_env
        configured_verifier_env = (
            {} if engine_config is None else engine_config.verifier_secret_env
        )
        self.agent_secret_env = {**os.environ, **configured_agent_env}
        self.verifier_secret_env = {**os.environ, **configured_verifier_env}

    def _client(self) -> Any:
        if self.client is None:
            self.client = docker.from_env()
            self.client.ping()
        return self.client

    def _gateway(self, client: Any) -> str:
        if self.bridge_gateway is not None:
            return self.bridge_gateway
        try:
            configs = client.networks.get("bridge").attrs["IPAM"]["Config"]
            gateway = next(
                str(item["Gateway"]) for item in configs if item.get("Gateway")
            )
        except Exception as error:
            raise SetupError(
                f"cannot resolve Docker host bridge gateway: {error}"
            ) from error
        return gateway

    def available_agents(self) -> tuple[str, ...]:
        return RSILoopAgentAdapter(self.rsi_loop_config).available_agents()

    def run(self, request: RunRequest) -> RunResult:
        # Static Harbor validation is deliberately outside the coordinator so
        # an unsupported local task cannot create the data root, contact
        # Docker, or be collapsed into a generic terminal result.
        compiler = HarborTaskCompiler()
        definition = compiler.compile(request.task_dir, request.options)
        validate_agent_reasoning_effort(
            definition.agent.name, request.reasoning_effort
        )
        agent_updates = {}
        if request.model is not None:
            agent_updates["model"] = request.model
        if request.reasoning_effort is not None:
            agent_updates["reasoning_effort"] = request.reasoning_effort
        if agent_updates:
            definition = definition.model_copy(
                update={
                    "agent": definition.agent.model_copy(update=agent_updates)
                }
            )
        needs_gpus = (
            definition.gpu_requirement.count != 0 or definition.verifier.gpu_count != 0
        )
        gpu_plan = resolve_gpu_plan(
            definition.gpu_requirement,
            judge_count=definition.verifier.gpu_count,
            requested=request.gpu_selectors,
            inventory=self.inventory.list_devices() if needs_gpus else (),
        )
        preparation = RunPreparation(
            request=request,
            definition=definition,
            gpu_plan=gpu_plan,
        )
        agent_auth = resolve_agent_auth(
            source=request.agent_auth,
            agent_name=definition.agent.name,
            agent_api_key=self.rsi_loop_config.agent_api_key,
        )
        api_endpoints = _agent_endpoints(
            definition, self.rsi_loop_config, agent_auth
        )
        client = self._client()
        snapshot = self.snapshot or OverlaySnapshotBackend(self.data_root)
        firewall = self.firewall or DockerIptablesFirewallBackend(client)
        ports = _ProductionRunComposition(
            client=client,
            data_root=self.data_root,
            logs_root=self.logs_root,
            inventory=self.inventory,
            rsi_loop_config=self.rsi_loop_config,
            snapshot=snapshot,
            firewall=firewall,
            bind_host=self.bind_host,
            bridge_gateway=self._gateway(client),
            omit_gpu_device_requests_for_tests=self.omit_gpu,
            agent_adapter_factory=self.agent_adapter_factory,
            quiescence_checker=self.quiescence_checker,
            api_endpoints=api_endpoints,
            agent_secret_env=self.agent_secret_env,
            verifier_secret_env=self.verifier_secret_env,
            agent_auth=agent_auth,
            event_callback=self.event_callback,
            preparation=preparation,
            compiler=compiler,
        )
        coordinator = self.coordinator_factory(
            backend=ports.backend(),
            lease_store=LeaseStore(self.data_root / "leases"),
            clock=_Clock(),
        )
        return coordinator.run(request)

    def _recovery(self) -> RecoveryManager:
        client = self._client()
        snapshot = self.snapshot or OverlaySnapshotBackend(self.data_root)
        firewall = self.firewall or DockerIptablesFirewallBackend(client)
        return RecoveryManager(
            store=LeaseStore(self.data_root / "leases"),
            backend=ProductionRecoveryBackend(client, snapshot, firewall),
            managed_root=self.data_root,
        )

    def recover(self, run_id: str | None) -> tuple[str, ...]:
        return self._recovery().recover(run_id)

    def cleanup(self, run_id: str, *, delete_workspace: bool) -> None:
        self._recovery().cleanup(run_id, delete_workspace=delete_workspace)


__all__ = ["ProductionRecoveryBackend", "ProductionRuntimeServices"]
