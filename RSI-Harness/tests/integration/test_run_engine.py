from __future__ import annotations

import ipaddress
import json
import shutil
import traceback
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import docker
import pytest

from rsi_harness.config import EngineConfig
from rsi_harness.errors import InfrastructureError, SetupError, UnsupportedTaskError
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    AgentAuthSource,
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunResult,
    CompileOptions,
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerTmpfs,
    ContainerVolumeMount,
    EvaluationRequest,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    ManagedNetwork,
    ManagedWorkdirVolume,
    PreparedAgent,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunRequest,
    RunResult,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
    WorkQuiescence,
)
from rsi_harness.runtime.coordinator import ProductionCoordinatorBackend
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
)
from rsi_harness.runtime.network import DockerIptablesFirewallBackend
from rsi_harness.runtime.recovery import (
    JudgeResourceLease,
    LeaseStore,
    RecoveryManager,
    ResourceLease,
    WorkResourceLease,
)
from rsi_harness.runtime.workdir_volume import (
    DockerWorkdirVolumeBackend,
    managed_workdir_volume_labels,
)
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tasks" / "minimal-gpu"


def _managed_workdir_volume(plan, *, run_id: str = "run-1"):
    return ManagedWorkdirVolume(
        name=f"rsi-harness-workdir-{'1' * 64}",
        run_id=run_id,
        task_id=plan.task.task_id,
        target=plan.workdir,
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce="f" * 64,
    )


def test_production_installs_hooks_before_securing_generic_agent_auth(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import _ProductionRunComposition

    secret = "LOCAL-CODEX-AUTH-TOKEN"
    document = json.dumps({"tokens": {"access_token": secret}}).encode()
    auth = AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options="rw,nosuid,nodev,noexec,mode=0700",
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=document,
                        mode=0o600,
                    ),
                ),
            ),
        ),
        secret_values=frozenset({document.decode(), secret}),
    )
    events: list[str] = []

    class WorkRuntime:
        spec = None

        def create(self, spec, *, planned_name):
            del planned_name
            self.spec = spec
            return ContainerRef(container_id="work-1", role="work")

        def inject_agent_auth(self, work, material):
            assert work == ContainerRef(container_id="work-1", role="work")
            assert material is auth
            events.append("auth")

    class Agent:
        def install_hooks(self, request):
            assert request.container.container_id == "work-1"
            events.append("hooks")

    devices = (
        GPUDevice(index=0, uuid="GPU-a", name="H100"),
        GPUDevice(index=1, uuid="GPU-b", name="H100"),
        GPUDevice(index=2, uuid="GPU-c", name="H100"),
        GPUDevice(index=3, uuid="GPU-d", name="H100"),
    )
    plan = make_run_plan(tmp_path / "plan").model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=devices),
                work=GPUAllocation(devices=devices[:2]),
                judge=GPUAllocation(devices=devices[2:]),
                judge_mode=JudgeGPUMode.DISJOINT,
            )
        }
    )
    volume = _managed_workdir_volume(plan)
    composition = _ProductionRunComposition(
        client=object(),
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=object(),
        rsi_loop_config=RSILoopConfig(),
        snapshot=object(),
        firewall=object(),
        bind_host="127.0.0.1",
        bridge_gateway="127.0.0.1",
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda _config, _runtime: Agent(),
        quiescence_checker=None,
        api_endpoints=(),
        agent_secret_env={},
        verifier_secret_env={},
        agent_auth=auth,
    )
    runtime = WorkRuntime()
    composition.work_runtime = runtime
    composition.start_artifacts(plan, "run-1")
    composition.workdir_volume = volume
    work = composition.create_work(
        plan,
        "run-1",
        ManagedNetwork(
            network_id="network-1",
            name="network-1",
            run_id="run-1",
            task_id=plan.task.task_id,
            role="work",
            internal=False,
        ),
        "planned-work",
        volume,
    )
    composition.agent = Agent()

    assert runtime.spec.tmpfs == tuple(mount.tmpfs for mount in auth.mounts)
    assert runtime.spec.gpu_allocation.uuids == ("GPU-a", "GPU-b")
    composition.install_hooks(
        plan, work, "http://172.30.0.1:9020", "control-token"
    )

    assert events == ["hooks", "auth"]
    assert secret in composition.agent_output_secrets


def test_production_services_resolve_explicit_agent_auth_before_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rsi_harness.runtime.production as production

    auth = AgentAuthMaterial(
        agent_name="codex",
        mounts=(),
        secret_values=frozenset({"synthetic"}),
    )
    calls = []
    captured = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return auth

    monkeypatch.setattr(production, "resolve_agent_auth", resolve)

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend.work_creator.__self__)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="local-auth", status=RunStatus.COMPLETED)

    services = production.ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(agent_api_base_url="https://127.0.0.1/v1"),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )
    services.run(
        RunRequest(
            task_dir=FIXTURE.resolve(), agent_auth=AgentAuthSource.LOCAL
        )
    )

    assert len(calls) == 1
    assert calls[0]["source"] is AgentAuthSource.LOCAL
    assert calls[0]["agent_name"] == "codex"
    assert calls[0]["agent_api_key"] is None
    assert captured[0].agent_auth is auth


def _no_network_task(tmp_path: Path) -> Path:
    task = tmp_path / "no-network-task"
    shutil.copytree(FIXTURE, task)
    config = task.joinpath("task.toml")
    config.write_text(
        config.read_text().replace(
            'network_mode = "public"', 'network_mode = "no-network"'
        )
    )
    return task


def _allowlist_task(tmp_path: Path, *hosts: str) -> Path:
    task = tmp_path / "allowlist-task"
    shutil.copytree(FIXTURE, task)
    config = task.joinpath("task.toml")
    quoted = ", ".join(json.dumps(host) for host in hosts)
    config.write_text(
        config.read_text().replace(
            'network_mode = "public"',
            f'network_mode = "allowlist"\nallowed_hosts = [{quoted}]',
        )
    )
    return task


class _OneDeviceInventory:
    def list_devices(self):
        return (GPUDevice(index=0, uuid="GPU-test", name="Test GPU"),)


@pytest.mark.parametrize(
    ("agent_name", "default_url", "hostname"),
    (
        ("codex", "https://api.openai.com", "api.openai.com"),
        ("claude-code", "https://api.anthropic.com", "api.anthropic.com"),
    ),
)
def test_no_network_stock_agent_pins_registered_default_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_name: str,
    default_url: str,
    hostname: str,
) -> None:
    from rsi_harness.runtime.network import NetworkPolicyEnforcer, PinnedEndpoint
    from rsi_harness.runtime.production import ProductionRuntimeServices

    requested: list[tuple[str, ...]] = []

    def pin_default(self, endpoints):
        del self
        requested.append(tuple(endpoints))
        return (
            PinnedEndpoint(
                hostname=hostname,
                port=443,
                addresses=(ipaddress.ip_address("203.0.113.80"),),
            ),
        )

    monkeypatch.setattr(NetworkPolicyEnforcer, "pin_endpoints", pin_default)

    class Coordinator:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    result = services.run(
        RunRequest(
            task_dir=_no_network_task(tmp_path).resolve(),
            options=CompileOptions(agent_name=agent_name),
        )
    )

    assert result.status is RunStatus.COMPLETED
    assert requested == [(default_url,)]


def test_no_network_codex_local_login_pins_chatgpt_provider_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness.runtime import production
    from rsi_harness.runtime.network import NetworkPolicyEnforcer, PinnedEndpoint
    from rsi_harness.runtime.production import ProductionRuntimeServices

    auth = AgentAuthMaterial(
        agent_name="codex",
        mounts=(),
        secret_values=frozenset({"local-login-secret"}),
        provider_endpoints=(
            "https://chatgpt.com/backend-api/codex",
            "https://auth.openai.com",
        ),
    )
    monkeypatch.setattr(production, "resolve_agent_auth", lambda **_kwargs: auth)
    requested: list[tuple[str, ...]] = []

    def pin_chatgpt(self, endpoints):
        del self
        requested.append(tuple(endpoints))
        return tuple(
            PinnedEndpoint(
                hostname=urlsplit(endpoint).hostname or "",
                port=443,
                addresses=(ipaddress.ip_address(f"203.0.113.{80 + index}"),),
            )
            for index, endpoint in enumerate(endpoints)
        )

    monkeypatch.setattr(NetworkPolicyEnforcer, "pin_endpoints", pin_chatgpt)

    class Coordinator:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="chatgpt-local", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    result = services.run(
        RunRequest(
            task_dir=_no_network_task(tmp_path).resolve(),
            agent_auth=AgentAuthSource.LOCAL,
        )
    )

    assert result.status is RunStatus.COMPLETED
    assert requested == [
        (
            "https://chatgpt.com/backend-api/codex",
            "https://auth.openai.com",
        )
    ]


def test_no_network_agent_without_registered_default_fails_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness.runtime import production
    from rsi_harness.runtime.production import ProductionRuntimeServices

    class AgentWithoutDefault:
        api_base_env = "CUSTOM_BASE_URL"
        api_key_env = "CUSTOM_API_KEY"
        model_env = "CUSTOM_MODEL"
        default_model = "custom-model"
        default_api_base_url = None

        @staticmethod
        def augment_env(environment, model) -> None:
            del environment, model

    monkeypatch.setattr(
        production,
        "create_agent",
        lambda _name, _config: AgentWithoutDefault(),
    )
    data = tmp_path / "data"

    class Coordinator:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("coordinator construction is a runtime mutation")

    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError, match="no-network.*API base URL.*proxy"):
        services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert not data.exists()


def test_no_network_proxy_is_resolved_as_the_only_provider_endpoint(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[tuple[str, ...]] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            composition = backend.network_planner.__self__
            captured.append(composition._api_endpoints())

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    proxy = "http://proxy-user:proxy-password@203.0.113.10:8080"
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(https_proxy=proxy),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert len(captured) == 1 and len(captured[0]) == 1
    endpoint = captured[0][0]
    assert endpoint.hostname == "203.0.113.10"
    assert endpoint.port == 8080
    assert tuple(map(str, endpoint.addresses)) == ("203.0.113.10",)
    assert "proxy-password" not in repr(endpoint)


def test_provider_dns_is_pinned_before_docker_or_data_root_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    data = tmp_path / "data"

    class Coordinator:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("coordinator must not be constructed")

    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="https://unresolvable-provider.invalid/v1"
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError, match="cannot resolve.*unresolvable-provider"):
        services.run(
            RunRequest(task_dir=_no_network_task(tmp_path).resolve())
        )

    assert not data.exists()


def test_no_network_explicit_base_is_pinned_without_credential_persistence(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.extend(backend.network_planner.__self__._api_endpoints())

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="https://provider-user:provider-password@203.0.113.11:8443/v1"
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert len(captured) == 1
    assert captured[0].hostname == "203.0.113.11"
    assert captured[0].port == 8443
    assert "provider-password" not in repr(captured[0])


def test_invalid_provider_endpoint_error_excludes_credentials_before_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url=(
                "https://provider-user:opaque-password@203.0.113.11:notaport/v1"
            )
        ),
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError) as caught:
        services.run(
            RunRequest(task_dir=_no_network_task(tmp_path).resolve())
        )

    assert "opaque-password" not in str(caught.value)
    assert not data.exists()


def test_malformed_ipv6_provider_is_typed_sanitized_before_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="http://[opaque-endpoint-secret"
        ),
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError, match="invalid Agent.*endpoint") as caught:
        services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert "opaque-endpoint-secret" not in str(caught.value)
    assert not data.exists()


def test_effective_codex_extra_env_overrides_are_all_pinned_before_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.extend(backend.network_planner.__self__._api_endpoints())

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="https://198.51.100.1/v1",
            https_proxy="http://198.51.100.2:8000",
            agent_extra_env={
                "OPENAI_BASE_URL": "https://203.0.113.21:9443/v1",
                "HTTPS_PROXY": "http://203.0.113.22:8080",
                "https_proxy": "http://203.0.113.23:8081",
                "NO_PROXY": "*",
            },
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert {(item.hostname, item.port) for item in captured} == {
        ("203.0.113.21", 9443),
        ("203.0.113.22", 8080),
        ("203.0.113.23", 8081),
    }


def test_effective_claude_extra_env_base_satisfies_no_network_preflight(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.extend(backend.network_planner.__self__._api_endpoints())

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_extra_env={
                "ANTHROPIC_BASE_URL": "https://203.0.113.31:9443"
            }
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(
        RunRequest(
            task_dir=_no_network_task(tmp_path).resolve(),
            options=CompileOptions(agent_name="claude-code"),
        )
    )

    assert [(item.hostname, item.port) for item in captured] == [
        ("203.0.113.31", 9443)
    ]


@pytest.mark.parametrize(
    ("base_url", "irrelevant_proxy", "expected_host"),
    (
        (
            "https://203.0.113.51:9443/v1",
            {"HTTP_PROXY": "http://198.51.100.51:8080"},
            "203.0.113.51",
        ),
        (
            "http://203.0.113.52:9080/v1",
            {"HTTPS_PROXY": "http://198.51.100.52:8080"},
            "203.0.113.52",
        ),
    ),
)
def test_provider_preflight_uses_only_scheme_applicable_proxy(
    tmp_path: Path,
    base_url: str,
    irrelevant_proxy: dict[str, str],
    expected_host: str,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.extend(backend.network_planner.__self__._api_endpoints())

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url=base_url,
            agent_extra_env=irrelevant_proxy,
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert [(item.hostname, item.port) for item in captured] == [
        (expected_host, int(urlsplit(base_url).port or 80))
    ]


def test_no_proxy_cidr_pins_direct_provider_alongside_configured_proxy(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.extend(backend.network_planner.__self__._api_endpoints())

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="https://203.0.113.75:9443/v1",
            https_proxy="http://198.51.100.75:8080",
            no_proxy="203.0.113.0/24",
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert {(item.hostname, item.port) for item in captured} == {
        ("198.51.100.75", 8080),
        ("203.0.113.75", 9443),
    }


def test_allowlist_cidr_union_covers_all_pinned_provider_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness.runtime.network import NetworkPolicyEnforcer, PinnedEndpoint
    from rsi_harness.runtime.production import ProductionRuntimeServices

    def pin_two_addresses(self, endpoints):
        del self, endpoints
        return (
            PinnedEndpoint(
                hostname="provider.example",
                port=443,
                addresses=(
                    ipaddress.ip_address("203.0.113.61"),
                    ipaddress.ip_address("203.0.113.62"),
                ),
            ),
        )

    monkeypatch.setattr(NetworkPolicyEnforcer, "pin_endpoints", pin_two_addresses)

    class Coordinator:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="https://provider.example/v1"
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(
        RunRequest(
            task_dir=_allowlist_task(
                tmp_path, "203.0.113.61/32", "203.0.113.62/32"
            ).resolve()
        )
    )


def test_allowlist_rejects_implicit_provider_endpoint_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="https://203.0.113.41/v1"
        ),
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError, match="allowlist.*203.0.113.41"):
        services.run(
            RunRequest(
                task_dir=_allowlist_task(tmp_path, "packages.example").resolve()
            )
        )

    assert not data.exists()


def test_ipv6_only_provider_is_rejected_before_docker_or_data_root_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url="https://[2001:db8::1]:8443/v1"
        ),
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError, match="IPv4-safe.*2001:db8::1"):
        services.run(RunRequest(task_dir=_no_network_task(tmp_path).resolve()))

    assert not data.exists()


def test_requested_model_selects_provider_endpoint_before_runtime_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness.runtime import production
    from rsi_harness.runtime.production import ProductionRuntimeServices

    class ModelEndpointAgent:
        api_base_env = "MODEL_PROVIDER_BASE_URL"
        api_key_env = "MODEL_PROVIDER_API_KEY"
        model_env = "MODEL_NAME"
        default_model = "default-model"
        default_api_base_url = None

        @staticmethod
        def augment_env(environment, model) -> None:
            suffix = "81" if model == "requested-model" else "80"
            environment["MODEL_PROVIDER_BASE_URL"] = (
                f"https://203.0.113.{suffix}:9443/v1"
            )

    monkeypatch.setattr(
        production, "create_agent", lambda _name, _config: ModelEndpointAgent()
    )
    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.extend(backend.network_planner.__self__._api_endpoints())

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="preflight", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    services.run(
        RunRequest(
            task_dir=_no_network_task(tmp_path).resolve(),
            model="requested-model",
        )
    )

    assert [(item.hostname, item.port) for item in captured] == [
        ("203.0.113.81", 9443)
    ]


class _DockerObjectCollection:
    def __init__(self, objects=()) -> None:
        self.objects = {item.id: item for item in objects}
        self.filters: list[object] = []

    def list(self, **kwargs):
        self.filters.append(kwargs.get("filters"))
        return list(self.objects.values())

    def get(self, identity):
        if identity not in self.objects:
            from docker.errors import NotFound

            raise NotFound("missing")
        return self.objects[identity]


class _DockerContainer:
    def __init__(self, identity, labels, *, collection=None, removable=True):
        self.id = identity
        self.attrs = {
            "Config": {"Labels": dict(labels)},
            "State": {"Running": True, "Paused": identity.startswith("work")},
        }
        self.collection = collection
        self.removable = removable

    def reload(self):
        pass

    def stop(self, timeout=10):
        del timeout
        self.attrs["State"]["Running"] = False

    def unpause(self):
        self.attrs["State"]["Paused"] = False

    def remove(self, force=False):
        del force
        if self.removable:
            self.collection.objects.pop(self.id, None)


class _DockerNetwork:
    def __init__(self, identity, labels, *, collection=None):
        self.id = identity
        self.attrs = {"Labels": dict(labels), "Containers": {}}
        self.collection = collection

    def reload(self):
        pass

    def remove(self):
        self.collection.objects.pop(self.id, None)


class _DockerImage:
    def __init__(self, identity, labels, image_ref):
        self.id = identity
        self.attrs = {
            "Id": identity,
            "RepoTags": [image_ref],
            "Config": {"Labels": dict(labels)},
        }

    def reload(self):
        pass


class _DockerImageCollection(_DockerObjectCollection):
    def remove(self, identity, force=False):
        del force
        self.objects.pop(identity, None)


class _EmptySnapshotRecovery:
    def discover_snapshot_leases(self, **_kwargs):
        return ()

    def release_snapshot(self, _authority):
        raise AssertionError("no snapshot authority expected")


class _EmptyFirewallRecovery:
    def remove(self, _rule_id):
        raise AssertionError("no policy authority expected")


def _production_recovery_client(*, removable_work=True):
    def labels(role):
        return {
            "rsi-harness.run-id": "run-1",
            "rsi-harness.task-id": "minimal-gpu",
            "rsi-harness.role": role,
        }

    containers = _DockerObjectCollection()
    work = _DockerContainer(
        "work-real",
        labels("work"),
        collection=containers,
        removable=removable_work,
    )
    judge = _DockerContainer(
        "judge-real", labels("judge"), collection=containers
    )
    containers.objects = {work.id: work, judge.id: judge}
    networks = _DockerObjectCollection()
    work_network = _DockerNetwork(
        "network-work", labels("work"), collection=networks
    )
    judge_network = _DockerNetwork(
        "network-judge", labels("judge"), collection=networks
    )
    networks.objects = {
        work_network.id: work_network,
        judge_network.id: judge_network,
    }
    return SimpleNamespace(
        containers=containers,
        networks=networks,
        images=_DockerImageCollection(),
    )


def test_checked_in_task_is_generic_one_gpu_harbor_task() -> None:
    definition = HarborTaskCompiler().compile(FIXTURE, CompileOptions())

    assert definition.task_id == "minimal-gpu"
    assert definition.gpu_requirement.count == 1
    assert str(definition.workdir) == "/workspace"
    assert definition.verifier.command == ("/bin/bash", "/tests/test.sh")
    fixture_text = "\n".join(
        path.read_text() for path in sorted(FIXTURE.rglob("*")) if path.is_file()
    )
    assert "judge-only.txt" in fixture_text
    assert '"answer_length"' in fixture_text


def test_production_service_composes_real_coordinator_backend_without_host_gpu_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[object] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    monkeypatch.setenv("PATH", "")
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )
    result = services.run(
        RunRequest(task_dir=FIXTURE.resolve(), gpu_selectors=("GPU-test",))
    )

    assert result.run_id == "production-shaped"
    assert len(captured) == 1
    backend = captured[0]
    assert isinstance(backend, ProductionCoordinatorBackend)
    assert isinstance(backend.compiler, HarborTaskCompiler)


def test_production_service_preserves_explicit_test_quiescence_checker(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[object] = []

    def quiescence_checker(_allocation, _container) -> None:
        pass

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend.evaluator.__self__.quiescence_checker)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        quiescence_checker=quiescence_checker,
    )

    services.run(RunRequest(task_dir=FIXTURE.resolve()))

    assert captured == [quiescence_checker]


def test_production_agent_artifacts_exclude_runtime_control_values(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[ProductionCoordinatorBackend] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    live_events: list[tuple[str, object]] = []
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        event_callback=lambda name, value: live_events.append((name, value)),
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    backend = captured[0]
    composition = backend.agent_runner.__self__

    class Runtime:
        def __init__(self) -> None:
            self.result = AgentRunResult(exit_code=0)
            self.exec_kwargs = {}

        def copy_to(self, _container, _source, _target) -> None:
            pass

        def exec(self, _container, _command, **kwargs) -> AgentRunResult:
            self.exec_kwargs = kwargs
            output_path = kwargs.get("output_path")
            if output_path is not None:
                output_path.write_text(
                    "prefix [REDACTED]::[REDACTED] full trajectory suffix"
                )
            return self.result

    submit_url = "http://control.internal:8123"
    token = "runtime-only-control-secret"
    runtime = Runtime()
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    plan = make_run_plan(tmp_path / "plan")
    work = ContainerRef(container_id="work-1", role="work")
    prepared = adapter.prepare(
        AgentPrepareRequest(
            run_plan=plan,
            prompt_path=(tmp_path / "agent-prompt.md").resolve(),
        )
    )
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=work,
            submit_url=submit_url,
            token=token,
        )
    )
    runtime.result = AgentRunResult(
        exit_code=0,
        output=f"prefix {submit_url}::{token}::{submit_url}{token} suffix",
        output_truncated=True,
        full_output_captured=True,
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    composition.agent = adapter
    composition.artifacts = SimpleNamespace(root=artifact_root)

    result = backend.run_agent(prepared, work, None)

    expected = "prefix [REDACTED]::[REDACTED]::[REDACTED][REDACTED] suffix"
    assert result.output == expected
    assert (artifact_root / "agent_output.txt").read_text() == (
        "prefix [REDACTED]::[REDACTED] full trajectory suffix"
    )
    assert (artifact_root / "run_agent.log").read_text() == expected
    assert runtime.exec_kwargs["output_path"] == (
        artifact_root / "agent_output.txt"
    )
    assert submit_url in runtime.exec_kwargs["output_redact_values"]
    assert token in runtime.exec_kwargs["output_redact_values"]
    assert callable(runtime.exec_kwargs["output_callback"])
    runtime.exec_kwargs["output_callback"]("safe live output\n")
    assert ("agent_output", "safe live output\n") in live_events
    assert live_events[0][0] == "agent_started"
    assert live_events[-2][0] == "agent_finished"
    for path in (artifact_root / "agent_output.txt", artifact_root / "run_agent.log"):
        persisted = path.read_text()
        assert submit_url not in persisted
        assert token not in persisted


def test_production_runtime_config_resolves_task_secrets_and_work_authority(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[ProductionCoordinatorBackend] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    secret = "bare-task-agent-secret"
    engine_config = EngineConfig(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        secret_env={"TASK_AGENT_TOKEN": secret},
        verifier_secret_env={"VERIFIER_TOKEN": "judge-only-secret"},
    )
    services = ProductionRuntimeServices(
        data_root=engine_config.data_root,
        logs_root=engine_config.logs_root,
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        engine_config=engine_config,
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    composition = captured[0].work_creator.__self__
    assert not hasattr(captured[0], "initialize_workspace")
    assert not hasattr(captured[0], "retain_workspace")
    assert not hasattr(composition, "workspace_manager")
    assert not hasattr(composition, "attest_workspace_for_judge")
    plan = make_run_plan(tmp_path / "plan")
    work_device = GPUDevice(index=0, uuid="GPU-test", name="Test GPU")
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "service": plan.task.service.model_copy(
                        update={
                            "shm_size": "3g",
                            "user": "2001:2002",
                            "cpus": 2,
                            "memory_mb": 8192,
                            "storage_mb": 15360,
                        }
                    ),
                    "agent": plan.task.agent.model_copy(
                        update={
                            "environment": (
                                ("ORDINARY_SERVICE_ENV", "ordinary-value"),
                                ("TASK_TOKEN", "${TASK_AGENT_TOKEN}"),
                                ("NVIDIA_VISIBLE_DEVICES", "${MISSING_TASK_VALUE}"),
                            ),
                            "secret_env_names": ("TASK_AGENT_TOKEN",),
                        }
                    ),
                }
            ),
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=(work_device,)),
                work=GPUAllocation(devices=(work_device,)),
                judge=GPUAllocation(),
                judge_mode=JudgeGPUMode.FREEZE_ONLY,
            ),
        }
    )

    class WorkRuntime:
        def __init__(self) -> None:
            self.spec = None

        def create(self, spec, *, planned_name):
            del planned_name
            self.spec = spec
            return ContainerRef(container_id="work-1", role="work")

    work_runtime = WorkRuntime()
    composition.work_runtime = work_runtime
    artifacts = composition.start_artifacts(plan, "run-1")
    volume = _managed_workdir_volume(plan)
    composition.workdir_volume = volume
    work = composition.create_work(
        plan,
        "run-1",
        ManagedNetwork(
            network_id="network-1",
            name="network-1",
            run_id="run-1",
            task_id=plan.task.task_id,
            role="work",
            internal=False,
        ),
        "planned-work",
        volume,
    )

    assert work_runtime.spec is not None
    work_environment = dict(work_runtime.spec.environment)
    assert work_environment["ORDINARY_SERVICE_ENV"] == "ordinary-value"
    assert work_environment["HOME"] == "/home/agent"
    assert "TASK_TOKEN" not in work_environment
    assert "NVIDIA_VISIBLE_DEVICES" not in work_environment
    assert secret not in work_environment.values()
    assert work_runtime.spec.shm_size == "3g"
    assert work_runtime.spec.mounts == (
        ContainerMount(
            source=artifacts.root / "feedback",
            target=PurePosixPath("/run/rsi-harness/feedback"),
            read_only=True,
        ),
    )
    assert (artifacts.root / "feedback").stat().st_mode & 0o777 == 0o755
    assert work_runtime.spec.workdir == plan.workdir
    assert len(work_runtime.spec.volume_mounts) == 1
    assert work_runtime.spec.volume_mounts[0].volume == volume
    assert work_runtime.spec.volume_mounts[0].target == plan.workdir
    assert work_runtime.spec.volume_mounts[0].read_only is False
    assert work_runtime.spec.user == "0"
    assert work_runtime.spec.cpus == 2
    assert work_runtime.spec.memory_mb == 8192
    assert work_runtime.spec.storage_mb == 15360
    assert composition.verifier_secret_env["VERIFIER_TOKEN"] == "judge-only-secret"

    class PreparingAgent:
        def prepare(self, request):
            return PreparedAgent(
                agent_name="codex",
                command=("agent",),
                prompt_path=request.prompt_path,
                environment=(
                    ("PROVIDER_ENV", "provider-value"),
                    ("NVIDIA_VISIBLE_DEVICES", "all"),
                ),
            )

    composition.agent = PreparingAgent()
    composition.artifacts = SimpleNamespace(root=tmp_path / "prepared-artifacts")
    prepared = composition.prepare_agent(plan)
    assert dict(prepared.environment) == {
        "NVIDIA_VISIBLE_DEVICES": "GPU-test",
        "PROVIDER_ENV": "provider-value",
        "TASK_TOKEN": secret,
    }

    class StoppedContainer:
        attrs = {"State": {"Running": False, "Paused": False}}

        @staticmethod
        def reload() -> None:
            pass

    composition.client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _container_id: StoppedContainer())
    )
    assert composition.quiesce_work(work) is WorkQuiescence.STOPPED

    class UnchangedRunningContainer:
        attrs = {"State": {"Running": True, "Paused": False}}

        @staticmethod
        def reload() -> None:
            pass

    composition.client = SimpleNamespace(
        containers=SimpleNamespace(
            get=lambda _container_id: UnchangedRunningContainer()
        )
    )
    composition.work_runtime = SimpleNamespace(pause=lambda _work: None)
    with pytest.raises(InfrastructureError, match="after final pause"):
        composition.quiesce_work(work)

    class RawAgent:
        def run(self, _request):
            return AgentRunResult(exit_code=0, output=f"raw {secret} harmless")

    artifact_root = tmp_path / "artifacts-runtime-config"
    artifact_root.mkdir()
    composition.agent = RawAgent()
    composition.artifacts = SimpleNamespace(root=artifact_root)
    result = composition.run_agent(object(), work, None)

    assert result.output == "raw [REDACTED] harmless"
    assert secret not in artifact_root.joinpath("agent_output.txt").read_text()


@pytest.mark.parametrize(
    "mounts",
    (
        (),
        (
            {
                "Type": "bind",
                "Name": f"rsi-harness-workdir-{'1' * 64}",
                "Destination": "/workspace",
                "RW": True,
            },
        ),
        (
            {
                "Type": "volume",
                "Name": f"rsi-harness-workdir-{'2' * 64}",
                "Destination": "/workspace",
                "RW": True,
            },
        ),
        (
            {
                "Type": "volume",
                "Name": f"rsi-harness-workdir-{'1' * 64}",
                "Destination": "/other",
                "RW": True,
            },
        ),
        (
            {
                "Type": "volume",
                "Name": f"rsi-harness-workdir-{'1' * 64}",
                "Destination": "/workspace",
                "RW": False,
            },
        ),
    ),
)
def test_production_workdir_volume_attestation_rejects_nonexact_mounts(
    tmp_path: Path,
    mounts,
) -> None:
    from rsi_harness.runtime.production import _ProductionRunComposition

    class WorkContainer:
        attrs = {"Mounts": list(mounts)}

        @staticmethod
        def reload() -> None:
            pass

    client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _container_id: WorkContainer())
    )
    composition = _ProductionRunComposition(
        client=client,
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=object(),
        rsi_loop_config=RSILoopConfig(),
        snapshot=object(),
        firewall=object(),
        bind_host="127.0.0.1",
        bridge_gateway="127.0.0.1",
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda _config, _runtime: object(),
        quiescence_checker=None,
        api_endpoints=(),
        agent_secret_env={},
        verifier_secret_env={},
    )
    plan = make_run_plan(tmp_path / "plan")
    volume = _managed_workdir_volume(plan)
    composition.workdir_volume = volume

    with pytest.raises(InfrastructureError, match="mount attestation"):
        composition.attest_workdir_volume(
            ContainerRef(container_id="work-1", role="work"), volume
        )


def test_production_workdir_volume_attestation_accepts_exact_rw_mount(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import _ProductionRunComposition

    plan = make_run_plan(tmp_path / "plan")
    volume = _managed_workdir_volume(plan)

    from tests.fakes import FakeDockerClient

    client = FakeDockerClient()
    volume = volume.model_copy(
        update={
            "name": DockerWorkdirVolumeBackend(client).planned_name(
                run_id="run-1", task_id=plan.task.task_id
            )
        }
    )
    client.volumes.create(
        volume.name,
        driver="local",
        labels=managed_workdir_volume_labels(volume),
    )
    engine_root = tmp_path / "engine"
    engine_root.mkdir()
    plan.task.source_dir.mkdir(parents=True)
    work_runtime = DockerContainerRuntime(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        role="work",
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(engine_root,),
    )
    work = work_runtime.create(
        ContainerSpec(
            image="work-image",
            workdir=plan.workdir,
            volume_mounts=(
                ContainerVolumeMount(
                    volume=volume,
                    target=plan.workdir,
                    read_only=False,
                ),
            ),
        )
    )
    composition = _ProductionRunComposition(
        client=client,
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=object(),
        rsi_loop_config=RSILoopConfig(),
        snapshot=object(),
        firewall=object(),
        bind_host="127.0.0.1",
        bridge_gateway="127.0.0.1",
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda _config, _runtime: object(),
        quiescence_checker=None,
        api_endpoints=(),
        agent_secret_env={},
        verifier_secret_env={},
    )
    composition.workdir_volume = volume
    composition.work_runtime = work_runtime

    composition.attest_workdir_volume(work, volume)


@pytest.mark.integration
def test_prepare_plan_uses_image_workdir_without_overlay_probe(tmp_path: Path) -> None:
    """Probing OverlayFS would make Docker rootfs Judge runs host-dependent."""
    from rsi_harness.runtime.production import _ProductionRunComposition

    class ProbeForbidden:
        def probe(self, _root):
            raise AssertionError("Docker rootfs runs must not probe OverlayFS")

    unresolved = make_run_plan(tmp_path / "definition")
    definition = unresolved.task.model_copy(
        update={
            "workdir": PurePosixPath("/declared-workdir"),
            "service": unresolved.task.service.model_copy(
                update={"workdir": PurePosixPath("/declared-workdir")}
            ),
        }
    )
    images = unresolved.images.model_copy(
        update={
            "workdir": PurePosixPath("/image-workdir"),
            "rootfs_snapshot_mode": RootfsSnapshotMode.SPLIT_WORKDIR,
        }
    )
    composition = _ProductionRunComposition(
        client=object(),
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        snapshot=ProbeForbidden(),
        firewall=object(),
        bind_host="127.0.0.1",
        bridge_gateway="127.0.0.1",
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda _config, _runtime: object(),
        quiescence_checker=None,
        api_endpoints=(),
        agent_secret_env={},
        verifier_secret_env={},
    )

    plan = composition.prepare_plan(
        definition,
        images,
        unresolved.gpu_plan,
        RunRequest(task_dir=definition.source_dir),
        "run-1",
    )

    assert composition.definition is plan.task
    assert plan.snapshot_kind == "docker-rootfs"
    assert plan.workdir == PurePosixPath("/image-workdir")
    metadata = json.loads(
        (tmp_path / "data" / "generated-tasks" / "minimal-gpu.json").read_text()
    )
    assert metadata["cwd"] == "/image-workdir"


def test_real_work_snapshot_and_judge_configs_exclude_exec_only_task_secret(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    try:
        client = docker.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local cached Ubuntu Docker authority unavailable: {error}")

    captured: list[ProductionCoordinatorBackend] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="env-inspect", status=RunStatus.COMPLETED)

    secret = "bare-secret-never-in-docker-config"
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=client,
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        omit_gpu_device_requests_for_tests=True,
        engine_config=EngineConfig(
            data_root=tmp_path / "data",
            logs_root=tmp_path / "logs",
            secret_env={"TASK_SECRET_SOURCE": secret},
        ),
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    composition = captured[0].work_creator.__self__
    work_device = GPUDevice(index=0, uuid="GPU-test", name="Test GPU")
    plan = make_run_plan(tmp_path / "plan")
    plan = plan.model_copy(
        update={
            "images": plan.images.model_copy(
                update={"work_ref": "ubuntu:24.04"}
            ),
            "task": plan.task.model_copy(
                update={
                    "source_dir": FIXTURE.resolve(),
                    "agent": plan.task.agent.model_copy(
                        update={
                            "environment": (
                                ("ORDINARY_SERVICE_ENV", "ordinary-value"),
                                ("TASK_SECRET", "${TASK_SECRET_SOURCE}"),
                                ("NVIDIA_VISIBLE_DEVICES", "all"),
                            ),
                            "secret_env_names": ("TASK_SECRET_SOURCE",),
                        }
                    ),
                }
            ),
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=(work_device,)),
                work=GPUAllocation(devices=(work_device,)),
                judge=GPUAllocation(),
                judge_mode=JudgeGPUMode.FREEZE_ONLY,
            ),
        }
    )
    artifacts = composition.start_artifacts(plan, "env-inspect")
    work_runtime = DockerContainerRuntime(
        client,
        run_id="env-inspect",
        task_id=plan.task.task_id,
        role="work",
        task_source_dir=FIXTURE.resolve(),
        allowed_mount_roots=(tmp_path.resolve(),),
        work_feedback_dir=artifacts.feedback_root,
    )
    composition.work_runtime = work_runtime
    work = None
    workdir_volume = None
    judge = None
    lease = None
    try:
        planned_volume = composition.plan_workdir_volume(plan, "env-inspect")
        assert planned_volume is not None
        workdir_volume = composition.create_workdir_volume(
            plan, "env-inspect", planned_volume
        )
        work = composition.create_work(
            plan,
            "env-inspect",
            ManagedNetwork(
                network_id="unused",
                name="unused",
                run_id="env-inspect",
                task_id=plan.task.task_id,
                role="work",
                internal=False,
            ),
            work_runtime.planned_container_name("main"),
            workdir_volume,
        )
        composition.attest_workdir_volume(work, workdir_volume)
        work_container = client.containers.get(work.container_id)
        work_config = work_container.attrs["Config"]
        feedback_mount = next(
            mount
            for mount in work_container.attrs["Mounts"]
            if mount["Destination"] == "/run/rsi-harness/feedback"
        )
        assert feedback_mount["Source"] == str(artifacts.feedback_root)
        assert feedback_mount["RW"] is False
        assert "ORDINARY_SERVICE_ENV=ordinary-value" in work_config["Env"]
        assert "NVIDIA_VISIBLE_DEVICES=GPU-test" in work_config["Env"]
        assert "NVIDIA_VISIBLE_DEVICES=all" not in work_config["Env"]
        assert secret not in json.dumps(work_config, sort_keys=True)

        planned_ref = composition.rootfs_snapshots.planned_ref(
            run_id="env-inspect",
            task_id=plan.task.task_id,
            round_id="agent-1",
            purpose="judge-round",
        )
        lease = composition.rootfs_snapshots.acquire(
            work,
            run_id="env-inspect",
            task_id=plan.task.task_id,
            round_id="agent-1",
            purpose="judge-round",
            planned_ref=planned_ref,
        )
        image_config = client.images.get(lease.image_id).attrs["Config"]
        assert "ORDINARY_SERVICE_ENV=ordinary-value" in image_config["Env"]
        assert "NVIDIA_VISIBLE_DEVICES=GPU-test" in image_config["Env"]
        assert secret not in json.dumps(image_config, sort_keys=True)

        judge_runtime = DockerContainerRuntime(
            client,
            run_id="env-inspect",
            task_id=plan.task.task_id,
            role="judge",
            task_source_dir=FIXTURE.resolve(),
            allowed_mount_roots=(tmp_path.resolve(),),
        )
        judge = judge_runtime.create(
            ContainerSpec(
                image=lease.image_id,
                command=("sleep", "infinity"),
                environment=(),
            )
        )
        judge_config = client.containers.get(judge.container_id).attrs["Config"]
        assert "ORDINARY_SERVICE_ENV=ordinary-value" in judge_config["Env"]
        assert "NVIDIA_VISIBLE_DEVICES=void" in judge_config["Env"]
        assert "NVIDIA_VISIBLE_DEVICES=GPU-test" not in judge_config["Env"]
        assert secret not in json.dumps(judge_config, sort_keys=True)
    finally:
        if judge is not None:
            judge_runtime.remove(judge)
        if lease is not None:
            composition.rootfs_snapshots.release(lease)
        if work is not None:
            work_runtime.remove(work)
        if workdir_volume is not None:
            composition.remove_workdir_volume(workdir_volume)


def test_production_clears_transient_agent_secrets_when_agent_run_raises(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[ProductionCoordinatorBackend] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(agent_api_key="transient-provider-secret"),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    composition = captured[0].agent_runner.__self__

    task_secret = "transient-task-agent-secret"

    class FailingAgent:
        def run(self, _request):
            raise RuntimeError(
                "agent process failed with transient-provider-secret and "
                f"{task_secret}"
            )

    artifact_root = tmp_path / "failure-artifacts"
    artifact_root.mkdir()
    composition.agent = FailingAgent()
    composition.artifacts = SimpleNamespace(root=artifact_root)
    composition.agent_output_secrets.add(task_secret)

    with pytest.raises(InfrastructureError, match="agent process failed") as caught:
        composition.run_agent(
            object(), ContainerRef(container_id="work-1", role="work"), None
        )

    assert "transient-provider-secret" not in str(caught.value)
    assert task_secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert composition.agent_output_secrets == set()


def test_production_stop_clears_transient_bindings_after_prepare_failure(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[ProductionCoordinatorBackend] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    composition = captured[0].agent_stopper.__self__

    class Agent:
        def __init__(self) -> None:
            self.cleared = False

        def prepare(self, _request):
            raise RuntimeError("prepare failed after hooks")

        def clear_transient_bindings(self) -> None:
            self.cleared = True

    class WorkRuntime:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self, _work) -> None:
            self.stopped = True

    agent = Agent()
    runtime = WorkRuntime()
    composition.agent = agent
    composition.work_runtime = runtime
    composition.artifacts = SimpleNamespace(root=tmp_path)
    composition.agent_output_secrets = {"transient-secret"}
    work = ContainerRef(container_id="work-1", role="work")
    with pytest.raises(InfrastructureError, match="prepare failed"):
        composition.prepare_agent(make_run_plan(tmp_path / "plan"))

    assert agent.cleared is True
    assert composition.agent_output_secrets == set()

    composition.stop_agent(work)

    assert runtime.stopped is True
    assert agent.cleared is True
    assert composition.agent_output_secrets == set()


@pytest.mark.parametrize(
    "failure_stage",
    ("create_work", "install_hooks", "prepare_agent", "run_agent", "stop_agent"),
)
@pytest.mark.parametrize("clear_fails", (False, True))
def test_production_agent_boundaries_redact_complete_runtime_secret_set(
    tmp_path: Path, failure_stage: str, clear_fails: bool
) -> None:
    from rsi_harness.runtime.artifacts import RunArtifactWriter
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[ProductionCoordinatorBackend] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    provider_secret = "bare-provider-secret"
    custom_secret = "bare-custom-extra-secret"
    task_secret = "bare-task-runtime-secret"
    proxy_secret = "bare-proxy-password"
    control_secret = "bare-control-token"
    runtime_secrets = {
        provider_secret,
        custom_secret,
        task_secret,
        proxy_secret,
    }
    all_secrets = runtime_secrets | (
        set() if failure_stage == "create_work" else {control_secret}
    )
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=provider_secret,
            https_proxy=(
                f"http://proxy-user:{proxy_secret}@203.0.113.91:8080"
            ),
            agent_extra_env={"CUSTOM_RUNTIME_VALUE": custom_secret},
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        engine_config=EngineConfig(
            data_root=tmp_path / "data",
            logs_root=tmp_path / "logs",
            secret_env={"TASK_RUNTIME_SECRET": task_secret},
        ),
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    composition = captured[0].work_creator.__self__
    plan = make_run_plan(tmp_path / "plan")
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={
                            "environment": (
                                ("TASK_TOKEN", "${TASK_RUNTIME_SECRET}"),
                            ),
                            "secret_env_names": ("TASK_RUNTIME_SECRET",),
                        }
                    )
                }
            )
        }
    )
    composition.plan = plan
    artifact_root = tmp_path / "boundary-artifacts"
    artifact_root.mkdir()
    feedback_root = artifact_root / "feedback"
    feedback_root.mkdir()
    composition.artifacts = SimpleNamespace(
        root=artifact_root, feedback_root=feedback_root
    )

    def stage_error() -> RuntimeError:
        return RuntimeError("boundary failed: " + "::".join(sorted(all_secrets)))

    class WorkRuntime:
        def create(self, _spec, *, planned_name):
            del planned_name
            if failure_stage == "create_work":
                raise stage_error()
            return ContainerRef(container_id="work-1", role="work")

        def stop(self, _work) -> None:
            if failure_stage == "stop_agent":
                raise stage_error()

    class Agent:
        def __init__(self) -> None:
            self.clear_count = 0

        def install_hooks(self, _request) -> None:
            if failure_stage == "install_hooks":
                raise stage_error()

        def prepare(self, _request):
            if failure_stage == "prepare_agent":
                raise stage_error()
            return PreparedAgent(
                agent_name="codex",
                command=("agent",),
                prompt_path=artifact_root / "agent_prompt.md",
            )

        def run(self, _request):
            if failure_stage == "run_agent":
                raise stage_error()
            return AgentRunResult(exit_code=0)

        def clear_transient_bindings(self) -> None:
            self.clear_count += 1
            if clear_fails:
                raise stage_error()

    runtime = WorkRuntime()
    agent = Agent()
    composition.work_runtime = runtime
    composition.agent = agent
    volume = _managed_workdir_volume(plan)
    composition.workdir_volume = volume
    network = ManagedNetwork(
        network_id="network-1",
        name="network-1",
        run_id="run-1",
        task_id=plan.task.task_id,
        role="work",
        internal=False,
    )

    with pytest.raises(InfrastructureError, match="boundary failed") as caught:
        work = composition.create_work(
            plan, "run-1", network, "planned-work", volume
        )
        composition.install_hooks(
            plan, work, "http://172.30.0.1:9020", control_secret
        )
        prepared = composition.prepare_agent(plan)
        if failure_stage == "run_agent":
            composition.run_agent(prepared, work, None)
        else:
            composition.stop_agent(work)

    formatted_error = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    for secret in all_secrets:
        assert secret not in str(caught.value)
        assert secret not in formatted_error
    assert "[REDACTED]" in str(caught.value)
    assert composition.agent_output_secrets == set()
    assert agent.clear_count >= 1
    if clear_fails:
        assert composition.agent is None

    lease_store = LeaseStore(tmp_path / "boundary-leases")
    lease_store.write(
        ResourceLease(
            run_id="run-1",
            task_id=plan.task.task_id,
            coordinator_pid=1,
            coordinator_started_at=1,
            phase=RunStatus.FAILED.value,
            status=RunStatus.FAILED,
            error=str(caught.value),
        )
    )
    writer = RunArtifactWriter(plan, run_id="run-1")
    writer.start()
    writer.record_engine_error(str(caught.value))
    persisted = lease_store.path_for("run-1").read_text()
    persisted += "".join(
        path.read_text(errors="replace")
        for path in writer.root.rglob("*")
        if path.is_file()
    )
    persisted += "".join(
        path.read_text(errors="replace")
        for path in artifact_root.rglob("*")
        if path.is_file()
    )
    for secret in all_secrets:
        assert secret not in persisted


@pytest.mark.parametrize("rollback_fails", (False, True))
def test_post_install_work_policy_attestation_failure_rolls_back_authoritatively(
    tmp_path: Path, rollback_fails: bool
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[ProductionCoordinatorBackend] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    composition = captured[0].work_policy_installer.__self__
    planned = SimpleNamespace(rule_id="policy-work-1")

    class Enforcer:
        def __init__(self) -> None:
            self.cleaned = []

        def apply(self, *_args, **_kwargs):
            return planned

        def cleanup(self, lease) -> None:
            self.cleaned.append(lease)
            if rollback_fails:
                raise InfrastructureError("firewall rollback failed")

    class WorkRuntime:
        def install_network_policy(self, work, lease) -> None:
            del work, lease
            raise SetupError("post-install exact attestation failed")

    enforcer = Enforcer()
    composition.enforcer = enforcer
    composition.work_runtime = WorkRuntime()
    composition.planned_work_policy = planned
    composition.started_server = SimpleNamespace(
        endpoint=SimpleNamespace(url="http://127.0.0.1:9020")
    )
    plan = make_run_plan(tmp_path / "plan")
    work = ContainerRef(container_id="work-1", role="work")
    network = ManagedNetwork(
        network_id="network-1",
        name="network-1",
        run_id="run-1",
        task_id=plan.task.task_id,
        role="work",
        internal=False,
    )

    expected = InfrastructureError if rollback_fails else SetupError
    match = "recovery_required" if rollback_fails else "exact attestation"
    with pytest.raises(expected, match=match):
        composition.install_work_policy(plan, work, network)

    assert enforcer.cleaned == [planned]


def test_production_wires_verifier_runtime_source_to_judge_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rsi_harness.runtime.production as production
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured: list[ProductionCoordinatorBackend] = []
    verifier_source = {"VERIFIER_TOKEN": "runtime-judge-secret"}

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="production-shaped", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        engine_config=EngineConfig(
            data_root=tmp_path / "data",
            logs_root=tmp_path / "logs",
            verifier_secret_env=verifier_source,
        ),
    )
    services.run(RunRequest(task_dir=FIXTURE.resolve()))
    composition = captured[0].evaluator.__self__
    plan = make_run_plan(tmp_path / "plan")
    composition.run_id = "run-1"
    composition.definition = plan.task
    composition.work_runtime = object()
    composition.enforcer = object()
    composition.artifacts = object()
    runner_kwargs: list[dict[str, object]] = []
    factory_work_containers: list[ContainerRef] = []

    class Factory:
        def __init__(
            self,
            client,
            *,
            run_id,
            task_id,
            task_source_dir,
            allowed_mount_roots,
            network_policy_enforcer,
            lifecycle_observer,
            work_container,
            omit_gpu_device_requests_for_tests,
        ) -> None:
            factory_work_containers.append(work_container)
            del (
                client,
                run_id,
                task_id,
                task_source_dir,
                allowed_mount_roots,
                network_policy_enforcer,
                lifecycle_observer,
                omit_gpu_device_requests_for_tests,
            )

    class Runner:
        def __init__(self, **kwargs) -> None:
            runner_kwargs.append(kwargs)

        def evaluate(self, request):
            return SubmissionReport(
                round_id=request.round_id,
                status=SubmissionStatus.COMPLETED,
                rewards={"reward": 1},
                score=1,
            )

    monkeypatch.setattr(production, "DockerJudgeRuntimeFactory", Factory)
    monkeypatch.setattr(production, "JudgeRunner", Runner)
    request = EvaluationRequest(
        run_plan=plan,
        work_container=ContainerRef(container_id="work-1", role="work"),
        round_id="agent-1",
        verifier_logs=(plan.paths.logs / "verifier" / "agent-1").resolve(),
        verifier_output=(plan.paths.logs / "feedback" / "agent-1.log").resolve(),
    )

    report = composition.evaluate(request, object())

    assert report.status == SubmissionStatus.COMPLETED
    assert runner_kwargs[0]["verifier_secret_env"]["VERIFIER_TOKEN"] == (
        "runtime-judge-secret"
    )
    assert runner_kwargs[0]["snapshot_backend"] is composition.rootfs_snapshots
    assert "workspace_attestor" not in runner_kwargs[0]
    assert factory_work_containers == [request.work_container]


def test_production_service_rejects_unsupported_task_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    invalid = tmp_path / "invalid-task"
    invalid.mkdir()
    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(UnsupportedTaskError):
        services.run(RunRequest(task_dir=invalid.resolve()))

    assert not data.exists()


def test_production_service_rejects_unknown_gpu_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    class Inventory:
        def list_devices(self):
            return (GPUDevice(index=0, uuid="GPU-real", name="Test GPU"),)

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=Inventory(),
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError, match="unknown GPU selector"):
        services.run(
            RunRequest(task_dir=FIXTURE.resolve(), gpu_selectors=("9",))
        )

    assert not data.exists()


def _gpu_pool_task(tmp_path: Path, *, judge_gpus: int = 0) -> Path:
    task = tmp_path / "gpu-pool-task"
    shutil.copytree(FIXTURE, task)
    config = task / "task.toml"
    metadata = ""
    if judge_gpus:
        metadata = (
            "\n[metadata.rsi_harness.verifier]\n"
            f"gpus = {judge_gpus}\n"
        )
    config.write_text(config.read_text().replace("gpus = 1", "gpus = 2") + metadata)
    return task


def test_gpu_pool_prevalidation_accepts_extra_selectors_and_plans_work_slice(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    captured = []

    class Inventory:
        def list_devices(self):
            return tuple(
                GPUDevice(index=index, uuid=f"GPU-{index}", name="Test GPU")
                for index in range(4)
            )

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            self.backend = backend

        def run(self, request: RunRequest) -> RunResult:
            definition = self.backend.compile(request)
            captured.append(
                self.backend.allocate(definition, request.gpu_selectors)
            )
            return RunResult(run_id="gpu-pool", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=Inventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )

    result = services.run(
        RunRequest(
            task_dir=_gpu_pool_task(tmp_path).resolve(),
            gpu_selectors=("GPU-3", "GPU-1", "GPU-2", "GPU-0"),
        )
    )

    assert result.status is RunStatus.COMPLETED
    assert captured[0].authorized_pool.uuids == (
        "GPU-3",
        "GPU-1",
        "GPU-2",
        "GPU-0",
    )
    assert captured[0].work.uuids == ("GPU-3", "GPU-1")


def test_production_reuses_one_compilation_and_inventory_plan_for_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second compile or inventory read could authorize different UUIDs."""
    import rsi_harness.runtime.production as production

    compiled = []
    original_compile = production.HarborTaskCompiler.compile

    def recording_compile(compiler, task_dir, options):
        definition = original_compile(compiler, task_dir, options)
        compiled.append(definition)
        return definition

    monkeypatch.setattr(production.HarborTaskCompiler, "compile", recording_compile)

    class ChangingInventory:
        def __init__(self) -> None:
            self.calls = 0

        def list_devices(self):
            self.calls += 1
            prefix = "authorized" if self.calls == 1 else "changed"
            return tuple(
                GPUDevice(index=index, uuid=f"GPU-{prefix}-{index}", name="Test GPU")
                for index in range(4)
            )

    inventory = ChangingInventory()
    captured: list[tuple[object, RunGPUPlan]] = []
    events: list[tuple[str, object]] = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del clock
            self.backend = backend
            self.lease_store = lease_store

        def run(self, request: RunRequest) -> RunResult:
            definition = self.backend.compile(request)
            gpu_plan = self.backend.allocate(definition, request.gpu_selectors)
            composition = self.backend.work_creator.__self__
            assert composition.allocate(definition, request.gpu_selectors) is gpu_plan
            captured.append((definition, gpu_plan))
            self.lease_store.write(
                ResourceLease(
                    run_id="one-shot",
                    task_id=definition.task_id,
                    coordinator_pid=1,
                    coordinator_started_at=0.0,
                    phase=RunStatus.PREPARING.value,
                    gpu_plan=gpu_plan,
                )
            )
            self.backend.record_event("gpu_plan", gpu_plan)
            return RunResult(run_id="one-shot", status=RunStatus.COMPLETED)

    data = tmp_path / "data"
    services = production.ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=inventory,
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        event_callback=lambda name, value: events.append((name, value)),
    )
    request = RunRequest(
        task_dir=_gpu_pool_task(tmp_path, judge_gpus=2).resolve(),
        gpu_selectors=("0", "1", "2", "3"),
    )

    result = services.run(request)

    assert result.status is RunStatus.COMPLETED
    assert len(compiled) == 1
    assert inventory.calls == 1
    definition, gpu_plan = captured[0]
    assert definition is compiled[0]
    assert gpu_plan.authorized_pool.uuids == (
        "GPU-authorized-0",
        "GPU-authorized-1",
        "GPU-authorized-2",
        "GPU-authorized-3",
    )
    assert gpu_plan.work.uuids == ("GPU-authorized-0", "GPU-authorized-1")
    assert gpu_plan.judge.uuids == ("GPU-authorized-2", "GPU-authorized-3")
    assert [name for name, _value in events] == ["gpu_plan"]
    assert events[0][1] is gpu_plan
    durable = LeaseStore(data / "leases").read("one-shot")
    assert durable is not None
    assert durable.gpu_plan == gpu_plan


def test_gpu_pool_prevalidation_rejects_judge_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    class Inventory:
        def list_devices(self):
            return (
                GPUDevice(index=0, uuid="GPU-0", name="Test GPU"),
                GPUDevice(index=1, uuid="GPU-1", name="Test GPU"),
            )

    class Coordinator:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("coordinator must not be constructed")

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        inventory=Inventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )
    services._client = lambda: (_ for _ in ()).throw(
        AssertionError("Docker must not be contacted")
    )

    with pytest.raises(SetupError, match="Judge requires 3 GPUs"):
        services.run(
            RunRequest(
                task_dir=_gpu_pool_task(tmp_path, judge_gpus=3).resolve(),
                gpu_selectors=("GPU-0", "GPU-1"),
            )
        )

    assert not data.exists()


def test_production_service_rejects_reasoning_for_unsupported_agent_before_mutation(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        bridge_gateway="127.0.0.1",
    )

    with pytest.raises(SetupError, match="unsupported-agent.*reasoning effort"):
        services.run(
            RunRequest(
                task_dir=FIXTURE.resolve(),
                agent_name="unsupported-agent",
                reasoning_effort="high",
                options=CompileOptions(agent_name="unsupported-agent"),
            )
        )

    assert not data.exists()


def test_production_recovery_normalizes_docker_labels_and_finds_all_resources(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRecoveryBackend

    client = _production_recovery_client()
    backend = ProductionRecoveryBackend(
        client, _EmptySnapshotRecovery(), _EmptyFirewallRecovery()
    )
    required = {
        "rsi-harness.run-id": "run-1",
        "rsi-harness.task-id": "minimal-gpu",
        "rsi-harness.role": "work",
    }
    listed = backend.list_containers(labels=required)
    inspected = backend.inspect_container("work-real")
    networks = backend.list_networks(labels=required)
    image_id = f"sha256:{'a' * 64}"
    image_ref = f"rsi-harness-rootfs:retained-work-{'b' * 64}"
    image_labels = {
        "rsi-harness.run-id": "run-1",
        "rsi-harness.task-id": "minimal-gpu",
        "rsi-harness.round-id": "final",
        "rsi-harness.source-container-id": "work-real",
        "rsi-harness.role": "retained-work-rootfs",
    }
    client.images.objects[image_id] = _DockerImage(
        image_id, image_labels, image_ref
    )
    images = backend.list_images(labels=image_labels)
    inspected_image = backend.inspect_image(image_id)

    assert listed[0][1]["labels"]["rsi-harness.run-id"] == "run-1"
    assert inspected is not None and inspected["labels"] == required
    assert networks[0][1]["labels"] == required
    assert images == (
        (
            image_id,
            {
                "id": image_id,
                "labels": image_labels,
                "repo_tags": (image_ref,),
            },
        ),
    )
    assert inspected_image == images[0][1]

    store = LeaseStore(tmp_path / "leases")
    store.write(
        ResourceLease(
            run_id="run-1",
            task_id="minimal-gpu",
            coordinator_pid=1,
            coordinator_started_at=1,
            phase="judging",
            work=WorkResourceLease(container_id="stale-work"),
        )
    )
    manager = RecoveryManager(
        store=store,
        backend=backend,
        managed_root=tmp_path / "managed",
    )

    assert manager.recover("run-1") == ("run-1",)
    final = store.read("run-1")
    assert final is not None
    assert final.work.container_id is None
    assert client.containers.objects == {}
    assert client.networks.objects == {}


def test_production_cleanup_retains_live_work_authority_and_workspace(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRecoveryBackend

    client = _production_recovery_client(removable_work=False)
    client.containers.objects.pop("judge-real")
    workspace = tmp_path / "managed" / "run-1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "answer.txt").write_text("42")
    store = LeaseStore(tmp_path / "leases")
    store.write(
        ResourceLease(
            run_id="run-1",
            task_id="minimal-gpu",
            coordinator_pid=1,
            coordinator_started_at=1,
            phase="completed",
            workspace_path=workspace,
            work=WorkResourceLease(container_id="work-real", paused=True),
        )
    )
    manager = RecoveryManager(
        store=store,
        backend=ProductionRecoveryBackend(
            client, _EmptySnapshotRecovery(), _EmptyFirewallRecovery()
        ),
        managed_root=tmp_path / "managed",
    )

    with pytest.raises(RuntimeError, match="Work removal cannot be proven"):
        manager.cleanup("run-1", delete_workspace=True)

    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.container_id == "work-real"
    assert retained.recovery_required is True
    assert workspace.joinpath("answer.txt").read_text() == "42"


def test_mountinfo_workspace_reference_detects_overlay_lowerdir_not_mount_target(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import _mountinfo_references_workspace

    workspace = tmp_path / "run space" / "workspace"
    escaped = str(workspace).replace(" ", r"\040")
    mountinfo = (
        "41 32 0:52 / /snapshots/merged rw,relatime - overlay overlay "
        f"rw,lowerdir={escaped}:/immutable/base,upperdir=/snapshots/upper,"
        "workdir=/snapshots/work\n"
    )

    assert _mountinfo_references_workspace(mountinfo, workspace) is True
    assert (
        _mountinfo_references_workspace(
            mountinfo, tmp_path / "different" / "workspace"
        )
        is False
    )
    with pytest.raises(SetupError, match="malformed mountinfo"):
        _mountinfo_references_workspace("malformed line\n", workspace)


@pytest.mark.integration
def test_production_recovery_helper_deletes_root_owned_private_workspace(
    tmp_path: Path,
) -> None:
    from rsi_harness.runtime.production import ProductionRecoveryBackend

    try:
        client = docker.from_env()
        client.ping()
        image = client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local Ubuntu Docker cleanup capability unavailable: {error}")
    run_id = "root-owned-cleanup"
    managed = tmp_path / "managed"
    workspace = managed / run_id / "workspace"
    workspace.mkdir(parents=True)
    store = LeaseStore(tmp_path / "leases")
    store.write(
        ResourceLease(
            run_id=run_id,
            task_id="minimal-gpu",
            coordinator_pid=1,
            coordinator_started_at=1,
            phase=RunStatus.COMPLETED.value,
            status=RunStatus.COMPLETED,
            workspace_path=workspace,
            cleanup_image_ref=image.id,
        )
    )
    client.containers.run(
        "ubuntu:24.04",
        [
            "/bin/sh",
            "-c",
            "mkdir -p /target/root-private && "
            "printf payload > /target/root-private/value && "
            "chmod 0700 /target/root-private",
        ],
        volumes={str(workspace): {"bind": "/target", "mode": "rw"}},
        network_mode="none",
        remove=True,
    )
    try:
        with pytest.raises(PermissionError):
            shutil.rmtree(workspace)
        manager = RecoveryManager(
            store=store,
            backend=ProductionRecoveryBackend(
                client, _EmptySnapshotRecovery(), _EmptyFirewallRecovery()
            ),
            managed_root=managed,
        )

        manager.cleanup(run_id, delete_workspace=True)

        assert not workspace.exists()
        labels = {"label": f"rsi-harness.run-id={run_id}"}
        assert client.containers.list(all=True, filters=labels) == []
    finally:
        if workspace.exists():
            client.containers.run(
                "ubuntu:24.04",
                ["/bin/sh", "-c", "chmod -R u+rwX /target"],
                volumes={str(workspace): {"bind": "/target", "mode": "rw"}},
                network_mode="none",
                remove=True,
            )
            shutil.rmtree(workspace)


@pytest.mark.integration
def test_real_docker_recovery_owns_round_and_retained_rootfs_images(
    tmp_path: Path,
) -> None:
    from docker.errors import DockerException, ImageNotFound, NotFound

    from rsi_harness.runtime.production import ProductionRecoveryBackend
    from rsi_harness.runtime.rootfs_snapshot import DockerRootfsSnapshotBackend

    try:
        client = docker.from_env()
        client.ping()
        base = client.images.get("ubuntu:24.04")
    except DockerException as error:
        pytest.skip(f"real Docker recovery capability unavailable: {error}")

    suffix = uuid4().hex
    task_id = "minimal-gpu"
    run_ids = {
        "round": f"recovery-round-{suffix}",
        "retained": f"recovery-retained-{suffix}",
        "referenced": f"recovery-referenced-{suffix}",
    }
    disposable_containers: set[str] = set()
    snapshot = DockerRootfsSnapshotBackend(client)
    backend = ProductionRecoveryBackend(
        client, _EmptySnapshotRecovery(), _EmptyFirewallRecovery()
    )
    store = LeaseStore(tmp_path / "leases")

    def acquire(run_id: str, *, purpose: str, round_id: str):
        work = client.containers.create(
            base.id,
            ["/bin/sh", "-c", "true"],
            network_mode="none",
            labels={
                "rsi-harness.run-id": run_id,
                "rsi-harness.task-id": task_id,
                "rsi-harness.role": "work",
            },
        )
        disposable_containers.add(work.id)
        planned = snapshot.planned_ref(
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
        )
        leased = snapshot.acquire(
            ContainerRef(container_id=work.id, role="work"),
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
            planned_ref=planned,
        )
        return work, leased

    try:
        round_work, round_image = acquire(
            run_ids["round"], purpose="judge-round", round_id="agent-1"
        )
        store.write(
            ResourceLease(
                run_id=run_ids["round"],
                task_id=task_id,
                coordinator_pid=1,
                coordinator_started_at=1,
                phase="judging",
                work=WorkResourceLease(container_id=round_work.id),
                judge=JudgeResourceLease(
                    round_id="agent-1",
                    planned_snapshot=f"snapshot:{run_ids['round']}:agent-1",
                    planned_snapshot_ref=round_image.image_ref,
                    snapshot_lease_id=round_image.lease_id,
                    snapshot_image_id=round_image.image_id,
                    snapshot_image_ref=round_image.image_ref,
                    snapshot_source_container_id=round_work.id,
                ),
            )
        )
        RecoveryManager(
            store=store, backend=backend, managed_root=tmp_path / "managed"
        ).recover(run_ids["round"])
        disposable_containers.discard(round_work.id)
        with pytest.raises(ImageNotFound):
            client.images.get(round_image.image_id)

        retained_work, retained_image = acquire(
            run_ids["retained"], purpose="retained-work", round_id="final"
        )
        retained_work.remove(force=True)
        disposable_containers.discard(retained_work.id)
        store.write(
            ResourceLease(
                run_id=run_ids["retained"],
                task_id=task_id,
                coordinator_pid=1,
                coordinator_started_at=1,
                phase=RunStatus.COMPLETED.value,
                status=RunStatus.COMPLETED,
                work=WorkResourceLease(
                    planned_retained_image_ref=retained_image.image_ref,
                    retained_image_id=retained_image.image_id,
                    retained_image_ref=retained_image.image_ref,
                ),
            )
        )
        retained_manager = RecoveryManager(
            store=store, backend=backend, managed_root=tmp_path / "managed"
        )
        retained_manager.recover(run_ids["retained"])
        assert client.images.get(retained_image.image_id).id == retained_image.image_id
        retained_manager.cleanup(run_ids["retained"], delete_workspace=True)
        with pytest.raises(ImageNotFound):
            client.images.get(retained_image.image_id)

        referenced_work, referenced_image = acquire(
            run_ids["referenced"], purpose="retained-work", round_id="final"
        )
        referenced_work.remove(force=True)
        disposable_containers.discard(referenced_work.id)
        dependent = client.containers.create(
            referenced_image.image_id,
            ["/bin/sh", "-c", "true"],
            network_mode="none",
        )
        disposable_containers.add(dependent.id)
        store.write(
            ResourceLease(
                run_id=run_ids["referenced"],
                task_id=task_id,
                coordinator_pid=1,
                coordinator_started_at=1,
                phase=RunStatus.COMPLETED.value,
                status=RunStatus.COMPLETED,
                work=WorkResourceLease(
                    planned_retained_image_ref=referenced_image.image_ref,
                    retained_image_id=referenced_image.image_id,
                    retained_image_ref=referenced_image.image_ref,
                ),
            )
        )
        referenced_manager = RecoveryManager(
            store=store, backend=backend, managed_root=tmp_path / "managed"
        )
        with pytest.raises(RuntimeError, match="referenced"):
            referenced_manager.cleanup(
                run_ids["referenced"], delete_workspace=True
            )
        assert client.images.get(referenced_image.image_id).id == (
            referenced_image.image_id
        )
        dependent.remove(force=True)
        disposable_containers.discard(dependent.id)
        referenced_manager.cleanup(
            run_ids["referenced"], delete_workspace=True
        )
        with pytest.raises(ImageNotFound):
            client.images.get(referenced_image.image_id)

        for run_id in run_ids.values():
            assert client.images.list(
                filters={"label": f"rsi-harness.run-id={run_id}"}
            ) == []
            assert client.containers.list(
                all=True,
                filters={"label": f"rsi-harness.run-id={run_id}"},
            ) == []
    finally:
        for container_id in tuple(disposable_containers):
            try:
                client.containers.get(container_id).remove(force=True)
            except NotFound:
                pass
        for run_id in run_ids.values():
            for image in client.images.list(
                filters={"label": f"rsi-harness.run-id={run_id}"}
            ):
                try:
                    client.images.remove(image.id, force=True)
                except ImageNotFound:
                    pass


@pytest.mark.integration
def test_real_docker_split_workdir_fake_agent_submits_two_fresh_judges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Run only when the host grants every fail-closed production authority."""
    from rsi_harness.runtime.docker import DockerContainerRuntime
    from rsi_harness.runtime.production import ProductionRuntimeServices

    try:
        client = docker.from_env()
        client.ping()
    except Exception as error:
        pytest.skip(f"real E2E Docker authority unavailable: {error}")
    firewall = DockerIptablesFirewallBackend(client)
    if not firewall.probe():
        pytest.skip(
            "real E2E fail-closed gate: host DOCKER-USER/INPUT firewall "
            "authority is unavailable"
        )
    data = tmp_path / "data"

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            return replace(
                prepared,
                command=(
                    "/bin/bash",
                    "-lc",
                    "set -eu\n"
                    "printf %s wrong > answer.txt\n"
                    "rsi-submit\n"
                    "printf %s 42 > answer.txt\n"
                    "rsi-submit\n"
                    "printf '\\nscripted fake Agent fixed after feedback\\n'\n",
                ),
            )

    created_containers: list[dict[str, object]] = []
    original_create = DockerContainerRuntime.create

    def recording_create(self, spec, *, planned_name=None):
        ref = original_create(self, spec, planned_name=planned_name)
        inspected = client.containers.get(ref.container_id)
        inspected.reload()
        image = client.images.get(inspected.attrs["Image"])
        created_containers.append(
            {
                "run_id": self.recovery_labels["rsi-harness.run-id"],
                "role": ref.role,
                "container_id": ref.container_id,
                "image_id": inspected.attrs["Image"],
                "image_labels": dict(image.attrs["Config"].get("Labels") or {}),
                "mounts": tuple(
                    (
                        mount.get("Type"),
                        mount.get("Name"),
                        mount.get("Destination"),
                        mount.get("RW"),
                    )
                    for mount in inspected.attrs.get("Mounts", [])
                ),
            }
        )
        return ref

    monkeypatch.setattr(DockerContainerRuntime, "create", recording_create)
    before_digest = hash_tree(FIXTURE)
    before_mtimes = {
        path.relative_to(FIXTURE): path.stat().st_mtime_ns
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=firewall,
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: ScriptedAgent(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
    )

    def cleanup_test_runs() -> None:
        lease_root = data / "leases"
        for lease_path in sorted(lease_root.glob("*.json")):
            run_id = lease_path.stem
            services.cleanup(run_id, delete_workspace=True)
            labels = {"label": f"rsi-harness.run-id={run_id}"}
            assert client.containers.list(all=True, filters=labels) == []
            assert client.networks.list(filters=labels) == []
            assert client.images.list(filters=labels) == []
            assert client.volumes.list(filters=labels) == []

    request.addfinalizer(cleanup_test_runs)

    result = services.run(
        RunRequest(
            task_dir=FIXTURE.resolve(),
            agent_name="codex",
            gpu_selectors=("0",),
            options=CompileOptions(agent_name="codex", primary_reward="reward"),
        )
    )

    assert result.status == RunStatus.COMPLETED
    assert result.total_rounds == 2
    assert result.best_round == "agent-2"
    assert result.best_score == 1
    agent_output = (
        tmp_path
        / "logs"
        / "runs"
        / result.run_id
        / "minimal-gpu"
        / "agent_output.txt"
    ).read_text()
    assert "reward: {\"answer_length\": 5.0, \"reward\": 0.0}" in agent_output
    assert "expected the two-character answer 42" in agent_output
    assert "reward: {\"answer_length\": 2.0, \"reward\": 1.0}" in agent_output
    assert "scripted fake Agent fixed after feedback" in agent_output
    run_created = [
        entry
        for entry in created_containers
        if entry["run_id"] == result.run_id
    ]
    work_entries = [entry for entry in run_created if entry["role"] == "work"]
    judge_entries = [entry for entry in run_created if entry["role"] == "judge"]
    assert len(work_entries) == 1
    assert len(judge_entries) == 2
    assert len({entry["container_id"] for entry in judge_entries}) == 2
    assert not (data / result.run_id / "workspace").exists()
    durable = LeaseStore(data / "leases").read(result.run_id)
    assert durable is not None
    assert durable.work.workdir_volume is not None
    assert durable.work.workdir_volume.actual is not None
    workdir_volume = durable.work.workdir_volume.actual
    inspected_volume = client.volumes.get(workdir_volume.name)
    assert inspected_volume.attrs["Labels"] == managed_workdir_volume_labels(
        workdir_volume
    )
    for entry in (*work_entries, *judge_entries):
        workdir_mounts = [
            mount
            for mount in entry["mounts"]
            if mount[2] == str(workdir_volume.target)
        ]
        assert workdir_mounts == [
            (
                "volume",
                workdir_volume.name,
                str(workdir_volume.target),
                entry["role"] == "work",
            )
        ]
    work_container_id = work_entries[0]["container_id"]
    judge_image_ids = []
    for round_id, entry in zip(
        ("agent-1", "agent-2"), judge_entries, strict=True
    ):
        labels = entry["image_labels"]
        assert labels["rsi-harness.run-id"] == result.run_id
        assert labels["rsi-harness.task-id"] == "minimal-gpu"
        assert labels["rsi-harness.round-id"] == round_id
        assert labels["rsi-harness.role"] == "rootfs-snapshot"
        assert labels["rsi-harness.source-container-id"] == work_container_id
        judge_image_ids.append(entry["image_id"])
    assert len(set(judge_image_ids)) == 2
    assert durable.work.retained_image_id is not None
    retained = client.images.get(durable.work.retained_image_id)
    assert retained.attrs["Config"]["Labels"]["rsi-harness.role"] == (
        "retained-work-rootfs"
    )
    assert retained.attrs["Config"]["Labels"]["rsi-harness.run-id"] == result.run_id
    assert hash_tree(FIXTURE) == before_digest
    assert {
        path.relative_to(FIXTURE): path.stat().st_mtime_ns
        for path in FIXTURE.rglob("*")
        if path.is_file()
    } == before_mtimes
    assert not list(tmp_path.rglob("*.tar*"))
    labels = {"label": f"rsi-harness.run-id={result.run_id}"}
    assert client.containers.list(all=True, filters=labels) == []
    assert client.networks.list(filters=labels) == []
    assert client.images.list(
        filters={
            "label": [
                f"rsi-harness.run-id={result.run_id}",
                "rsi-harness.role=rootfs-snapshot",
            ]
        }
    ) == []
