"""Event-recording in-memory doubles for contract and coordinator tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docker.errors import NotFound

from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    ContainerSpec,
    EvaluationRequest,
    PreparedAgent,
    RootfsSnapshotLease,
    SnapshotCapabilities,
    SnapshotLease,
    SubmissionReport,
    WorkQuiescence,
)
from rsi_harness.runtime.rootfs_snapshot import RootfsSnapshotNotCreatedError


class FakeContainerRuntime:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self._next_id = 1

    def create(self, spec: ContainerSpec) -> ContainerRef:
        container = ContainerRef(
            container_id=f"{spec.image}-{self._next_id}", role="work"
        )
        self._next_id += 1
        self.events.append(("create", container.container_id))
        return container

    def start(self, container: ContainerRef) -> None:
        self.events.append(("start", container.container_id))

    def pause(self, container: ContainerRef) -> None:
        self.events.append(("pause", container.container_id))

    def unpause(self, container: ContainerRef) -> None:
        self.events.append(("unpause", container.container_id))

    def remove(self, container: ContainerRef) -> None:
        self.events.append(("remove", container.container_id))

    def count_events(self, name: str) -> int:
        return sum(event_name == name for event_name, _ in self.events)


class FakeSnapshotBackend:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.events: list[tuple[str, object]] = []
        self.lower_paths: list[Path] = []

    def probe(self, workspace_root: Path) -> SnapshotCapabilities:
        self.events.append(("probe", workspace_root))
        return SnapshotCapabilities(
            backend="fake",
            atomic=True,
            immutable=True,
            copy_on_write=True,
            cleanup=True,
        )

    def acquire(self, workspace: Path, *, run_id: str, round_id: str) -> SnapshotLease:
        self.events.append(("acquire", round_id))
        self.lower_paths.append(workspace.resolve())
        root = (self.root / run_id / round_id).resolve()
        return SnapshotLease(
            lease_id=f"{run_id}-{round_id}",
            lower_dir=workspace.resolve(),
            upper_dir=root / "upper",
            work_dir=root / "work",
            merged_dir=root / "merged",
            backend="fake",
        )

    def release(self, lease: SnapshotLease) -> None:
        self.events.append(("release", lease.lease_id))


class FakeJudgeRuntime:
    """One shared event recorder implementing Work and one-shot Judge ports."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.events: list[tuple[str, object]] = []
        self.fail_at: str | set[str] | None = None
        self.exec_result = AgentRunResult(
            exit_code=0,
            output="verifier output",
            full_output_captured=True,
        )
        self.complete_exec_output: bytes | None = None
        self.exec_output_paths: list[Path | None] = []
        self.reward_payload: object | None = {"reward": 1}
        self.created_specs: list[ContainerSpec] = []
        self.exec_environments: list[dict[str, str]] = []
        self.round_ids: list[str] = []
        self.closed_rounds = 0
        self.work_paused = False
        self._safe_to_release_isolation = True
        self._round_id = ""
        self._log_dir: Path | None = None
        self._private_tests: Path | None = None
        self.verifier_test_write: tuple[Path, str] | None = None
        self.verifier_modified_tests = False
        self.work_rootfs: dict[str, str] = {"/etc/agent-state": "agent-state"}
        self.snapshot_images: dict[str, dict[str, str]] = {}
        self.judge_rootfs: dict[str, str] = {}
        self.judge_rootfs_writes: dict[str, str] = {}

    def for_round(self, run_plan: object, round_id: str) -> FakeJudgeRuntime:
        del run_plan
        self._round_id = round_id
        self.round_ids.append(round_id)
        return self

    def assert_gpu_quiet(self, allocation: object, container: ContainerRef) -> None:
        del allocation
        self.events.append(("assert_gpu_quiet", container.container_id))
        self._raise("quiet")

    def pause(self, container: ContainerRef) -> None:
        self.events.append(("pause", container.container_id))
        self.work_paused = True
        self._raise("pause")

    def attest_workspace(self) -> None:
        self.events.append(("workspace_attest", self._round_id or "work"))
        self._raise("attest")

    def unpause(self, container: ContainerRef) -> None:
        self.events.append(("unpause", container.container_id))
        self.work_paused = False
        self._raise("unpause")

    def inspect_quiescence(
        self, container: ContainerRef
    ) -> WorkQuiescence | None:
        self.events.append(("inspect_quiescence", container.container_id))
        self._raise("inspect_quiescence")
        return WorkQuiescence.PAUSED if self.work_paused else None

    def create(self, spec: ContainerSpec) -> ContainerRef:
        self.events.append(("judge_create", self._round_id))
        self.created_specs.append(spec)
        logs_mount = next(
            mount for mount in spec.mounts if str(mount.target) == "/logs/verifier"
        )
        self._log_dir = logs_mount.source
        self.judge_rootfs = dict(self.snapshot_images[spec.image])
        self._raise("create")
        self._safe_to_release_isolation = False
        return ContainerRef(container_id=f"judge-{self._round_id}", role="judge")

    def start(self, container: ContainerRef) -> None:
        del container
        self.events.append(("judge_start", self._round_id))
        self._raise("start")

    def inject_tests(self, container: ContainerRef, source: Path) -> None:
        del container
        self.events.append(("tests_inject", self._round_id))
        private_tests = self.root / "private-tests" / self._round_id
        if private_tests.exists():
            shutil.rmtree(private_tests)
        shutil.copytree(source, private_tests, symlinks=True)
        self._private_tests = private_tests
        self._raise("inject")

    def exec(self, container: ContainerRef, command: tuple[str, ...], **kwargs: Any):
        del container
        self.exec_environments.append(dict(kwargs.get("environment") or {}))
        output_path = kwargs.get("output_path")
        self.exec_output_paths.append(output_path)
        if output_path is not None and self.exec_result.full_output_captured:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(
                self.complete_exec_output
                if self.complete_exec_output is not None
                else self.exec_result.output.encode()
            )
        self.events.append(("judge_exec", command))
        self.judge_rootfs.update(self.judge_rootfs_writes)
        if self.verifier_test_write is not None and self._private_tests is not None:
            relative, payload = self.verifier_test_write
            target = self._private_tests / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload)
            self.verifier_modified_tests = target.read_text() == payload
        if self.reward_payload is not None and self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            (self._log_dir / "reward.json").write_text(
                json.dumps(self.reward_payload)
            )
        self._raise("exec")
        return self.exec_result

    def remove(self, container: ContainerRef) -> None:
        del container
        self.events.append(("judge_remove", self._round_id))
        self._raise("remove")
        if self._private_tests is not None:
            shutil.rmtree(self._private_tests)
            self._private_tests = None
        self._safe_to_release_isolation = True

    def contain_after_remove_failure(self, container: ContainerRef) -> None:
        del container
        self.events.append(("judge_contain", self._round_id))
        try:
            self._raise("contain")
        except Exception:
            self._safe_to_release_isolation = False
            raise
        self._safe_to_release_isolation = False

    @property
    def safe_to_release_isolation(self) -> bool:
        return self._safe_to_release_isolation

    def recovery_context(self, container: ContainerRef) -> str:
        return (
            f"judge_container_id={container.container_id},"
            f"network_id=network-{self._round_id},"
            f"policy_rule_id=policy-{self._round_id}"
        )

    def close(self) -> None:
        self.events.append(("judge_close", self._round_id))
        self.closed_rounds += 1
        self._raise("close")

    def _raise(self, stage: str) -> None:
        if self.fail_at == stage or (
            isinstance(self.fail_at, set) and stage in self.fail_at
        ):
            raise RuntimeError(f"{stage} failed")


class FakeJudgeSnapshotBackend:
    def __init__(self, root: Path, runtime: FakeJudgeRuntime) -> None:
        self.root = root.resolve()
        self.runtime = runtime
        self.fail_at: str | None = None
        self.acquired: list[RootfsSnapshotLease] = []
        self.states: list[dict[str, str]] = []

    def planned_ref(
        self, *, run_id: str, task_id: str, round_id: str, purpose: str
    ) -> str:
        digest = hashlib.sha256(
            f"{run_id}:{task_id}:{round_id}:{purpose}".encode()
        ).hexdigest()
        return f"rsi-harness-rootfs:{purpose}-{digest}"

    def acquire(
        self,
        work: ContainerRef,
        *,
        run_id: str,
        task_id: str,
        round_id: str,
        purpose: str = "judge-round",
        planned_ref: str | None = None,
    ) -> RootfsSnapshotLease:
        self.runtime.events.append(("snapshot_acquire", round_id))
        if self.fail_at == "acquire":
            raise RootfsSnapshotNotCreatedError("acquire failed before commit")
        if self.fail_at == "acquire_ambiguous":
            raise RuntimeError(
                "recovery_required: rootfs commit response is ambiguous"
            )
        expected_ref = self.planned_ref(
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
        )
        assert planned_ref == expected_ref
        digest = hashlib.sha256(f"image:{round_id}".encode()).hexdigest()
        lease = RootfsSnapshotLease(
            lease_id=hashlib.sha256(f"lease:{round_id}".encode()).hexdigest(),
            purpose="judge-round",
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            source_container_id=work.container_id,
            image_id=f"sha256:{digest}",
            image_ref=expected_ref,
        )
        state = dict(self.runtime.work_rootfs)
        self.states.append(state)
        self.runtime.snapshot_images[lease.image_id] = state
        self.acquired.append(lease)
        return lease

    def release(self, lease: RootfsSnapshotLease) -> None:
        self.runtime.events.append(("snapshot_release", lease.round_id))
        if self.fail_at == "release":
            raise RuntimeError("release failed")
        self.runtime.snapshot_images.pop(lease.image_id)


class FakeArtifactWriter:
    def __init__(self) -> None:
        self.reports: list[SubmissionReport] = []

    def record_submission(self, report: SubmissionReport) -> None:
        for index, current in enumerate(self.reports):
            if current.round_id == report.round_id:
                self.reports[index] = report
                return
        self.reports.append(report)


class FakeAgentAdapter:
    def __init__(self, agents: tuple[str, ...] = ("codex",)) -> None:
        self.agents = agents
        self.events: list[tuple[str, object]] = []

    def available_agents(self) -> tuple[str, ...]:
        return self.agents

    def prepare(self, request: AgentPrepareRequest) -> PreparedAgent:
        self.events.append(("prepare", request.run_plan.task.task_id))
        return PreparedAgent(
            agent_name=request.run_plan.task.agent.name,
            command=("agent",),
            prompt_path=request.prompt_path,
        )

    def install_hooks(self, request: AgentHookRequest) -> None:
        self.events.append(("install_hooks", request.container.container_id))

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.events.append(("run", request.container.container_id))
        return AgentRunResult(exit_code=0)


class FakeEvaluator:
    def __init__(
        self,
        report: SubmissionReport,
        *,
        artifact_writer: Any | None = None,
        delay_seconds: float = 0.0,
        release: threading.Event | None = None,
    ) -> None:
        self.report = report
        self.artifact_writer = artifact_writer
        self.events: list[tuple[str, object]] = []
        self.round_ids: list[str] = []
        self.requests: list[EvaluationRequest] = []
        self.delay_seconds = delay_seconds
        self.release = release
        self.started = threading.Event()
        self._state_lock = threading.Lock()
        self._concurrency = 0
        self.max_concurrency = 0

    def evaluate(self, request: EvaluationRequest) -> SubmissionReport:
        self.events.append(("evaluate", request.round_id))
        self.round_ids.append(request.round_id)
        self.requests.append(request)
        with self._state_lock:
            self._concurrency += 1
            self.max_concurrency = max(self.max_concurrency, self._concurrency)
        self.started.set()
        try:
            if self.release is not None:
                self.release.wait(timeout=5)
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            report = self.report.model_copy(update={"round_id": request.round_id})
            if self.artifact_writer is not None:
                self.artifact_writer.record_submission(report)
            return report
        finally:
            with self._state_lock:
                self._concurrency -= 1


class FakeClock:
    def __init__(self, current: datetime | None = None) -> None:
        self.current = current or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.current


class FakeDockerContainer:
    """Specific fake for the pinned Docker SDK container surface."""

    def __init__(
        self,
        container_id: str,
        *,
        pause_states: tuple[bool, ...] = (True,),
        logs_chunks: tuple[bytes, ...] = (b"container logs",),
    ) -> None:
        self.id = container_id
        self.attrs: dict[str, Any] = {
            "State": {"Paused": False, "Running": True},
            "Config": {"Labels": {}, "Image": None},
            "Mounts": [],
        }
        self.pause_states = list(pause_states)
        self.logs_chunks = logs_chunks
        self.logs_stream_closed = False
        self.logs_response_closed = False
        self.events: list[str] = []
        self.removed_kwargs: dict[str, Any] | None = None

    def start(self) -> None:
        self.events.append("start")

    def pause(self) -> None:
        self.events.append("pause")

    def unpause(self) -> None:
        self.events.append("unpause")
        self.attrs["State"]["Paused"] = False

    def reload(self) -> None:
        self.events.append("reload")
        if self.pause_states:
            self.attrs["State"]["Paused"] = self.pause_states.pop(0)

    def stop(self, **kwargs: Any) -> None:
        self.events.append("stop")
        self.attrs["State"]["Running"] = False

    def remove(self, **kwargs: Any) -> None:
        self.events.append("remove")
        self.removed_kwargs = kwargs

    def logs(self, **kwargs: Any):
        self.events.append("logs")
        if kwargs.get("stream"):
            container = self

            class Response:
                def close(self):
                    container.logs_response_closed = True

            class Stream:
                def __init__(self):
                    self._chunks = iter(container.logs_chunks)
                    self._response = Response()

                def __iter__(self):
                    return self

                def __next__(self):
                    return next(self._chunks)

                def close(self):
                    container.logs_stream_closed = True

            return Stream()
        return b"".join(self.logs_chunks)

    def top(self, **kwargs: Any) -> dict[str, object]:
        self.events.append("top")
        return {"Titles": ["PID"], "Processes": [["101"]]}


class FakeDockerContainers:
    def __init__(
        self,
        *,
        run_result: bytes = b"",
        run_hook: Callable[..., bytes] | None = None,
    ) -> None:
        self.created: list[dict[str, Any]] = []
        self.by_id: dict[str, FakeDockerContainer] = {}
        self.runs: list[dict[str, Any]] = []
        self.image_environments: dict[str, dict[str, str]] = {}
        self.run_result = run_result
        self.run_hook = run_hook

    def create(self, image: str, command: list[str], **kwargs: Any):
        record = {"image": image, "command": command, **kwargs}
        self.created.append(record)
        container = FakeDockerContainer(f"container-{len(self.created)}")
        environment = dict(self.image_environments.get(image, {}))
        environment.update(kwargs.get("environment") or {})
        container.attrs["Config"] = {
            "Labels": dict(kwargs.get("labels") or {}),
            "Image": image,
            "Env": [f"{key}={value}" for key, value in environment.items()],
        }
        container.attrs["Mounts"] = [
            {
                "Type": mount.get("Type"),
                "Name": mount.get("Source"),
                "Source": mount.get("Source"),
                "Destination": mount.get("Target"),
                "RW": not bool(mount.get("ReadOnly", False)),
            }
            for mount in kwargs.get("mounts", ())
        ]
        self.by_id[container.id] = container
        return container

    def list(self, *, all: bool, filters: dict[str, Any]):
        assert all is True
        found = [
            container
            for container in self.by_id.values()
            if container.removed_kwargs is None
        ]
        if "label" in filters:
            required = tuple(
                label.split("=", 1) for label in filters["label"]
            )
            found = [
                container
                for container in found
                if all(
                    container.attrs["Config"]["Labels"].get(key) == value
                    for key, value in required
                )
            ]
        if "ancestor" in filters:
            found = [
                container
                for container in found
                if container.attrs["Config"].get("Image")
                == filters["ancestor"]
            ]
        if "volume" in filters:
            found = [
                container
                for container in found
                if any(
                    mount.get("Name") == filters["volume"]
                    or mount.get("Source") == filters["volume"]
                    for mount in container.attrs.get("Mounts", ())
                )
            ]
        return found

    def get(self, container_id: str) -> FakeDockerContainer:
        return self.by_id[container_id]

    def run(self, image: str, command: list[str], **kwargs: Any) -> bytes:
        self.runs.append({"image": image, "command": command, **kwargs})
        if self.run_hook is not None:
            return self.run_hook(image, command, **kwargs)
        return self.run_result


class FakeDockerAPI:
    def __init__(
        self,
        chunks: tuple[bytes, ...] = (),
        exit_code: int = 0,
        chunk_delay_seconds: float = 0,
        stream_close_error: BaseException | None = None,
    ) -> None:
        self.chunks = chunks
        self.exit_code = exit_code
        self.chunk_delay_seconds = chunk_delay_seconds
        self.stream_close_error = stream_close_error
        self.exec_create_calls: list[dict[str, Any]] = []
        self.drained_chunks = 0
        self.stream_closed = False
        self.response_closed = False

    def exec_create(self, container: str, command: object, **kwargs: Any):
        self.exec_create_calls.append(
            {"container": container, "command": command, **kwargs}
        )
        return {"Id": "exec-1"}

    def exec_start(self, exec_id: str, **kwargs: Any):
        assert exec_id == "exec-1"
        assert kwargs == {"stream": True, "demux": False}

        api = self

        class Response:
            def close(self):
                api.response_closed = True

        class Stream:
            def __init__(self):
                self._response = Response()

            def __iter__(self):
                for chunk in api.chunks:
                    deadline = time.monotonic() + api.chunk_delay_seconds
                    while time.monotonic() < deadline:
                        if api.stream_closed or api.response_closed:
                            return
                        time.sleep(0.001)
                    if api.stream_closed or api.response_closed:
                        return
                    api.drained_chunks += 1
                    yield chunk

            def close(self):
                if api.stream_close_error is not None:
                    raise api.stream_close_error
                api.stream_closed = True

        return Stream()

    def exec_inspect(self, exec_id: str) -> dict[str, int]:
        assert exec_id == "exec-1"
        return {"ExitCode": self.exit_code}


class FakeDockerClient:
    def __init__(
        self,
        *,
        chunks: tuple[bytes, ...] = (),
        exit_code: int = 0,
        chunk_delay_seconds: float = 0,
        stream_close_error: BaseException | None = None,
        run_result: bytes = b"",
        run_hook: Callable[..., bytes] | None = None,
    ) -> None:
        self.containers = FakeDockerContainers(
            run_result=run_result,
            run_hook=run_hook,
        )
        self.api = FakeDockerAPI(
            chunks=chunks,
            exit_code=exit_code,
            chunk_delay_seconds=chunk_delay_seconds,
            stream_close_error=stream_close_error,
        )
        self.images = FakeDockerImages()
        self.networks = FakeDockerNetworks()
        self.volumes = FakeDockerVolumes()


class FakeDockerVolume:
    """Specific fake for Docker SDK local volume inspection."""

    def __init__(self, name: str, *, driver: str, labels: dict[str, str]) -> None:
        self.name = name
        self.removed = False
        self.attrs: dict[str, Any] = {
            "Name": name,
            "Driver": driver,
            "Labels": dict(labels),
            "Options": {},
            "Scope": "local",
        }

    def reload(self) -> None:
        return None

    def remove(self, *, force: bool = False) -> None:
        assert force is False
        self.removed = True


class FakeDockerVolumes:
    def __init__(self) -> None:
        self.by_name: dict[str, FakeDockerVolume] = {}
        self.list_result: object | None = None

    def create(
        self, name: str, *, driver: str, labels: dict[str, str]
    ) -> FakeDockerVolume:
        volume = FakeDockerVolume(name, driver=driver, labels=labels)
        self.by_name[name] = volume
        return volume

    def get(self, name: str) -> FakeDockerVolume:
        try:
            volume = self.by_name[name]
        except KeyError as error:
            raise NotFound(name) from error
        if volume.removed:
            raise NotFound(name)
        return volume

    def list(self, *, filters: dict[str, list[str]]) -> object:
        if self.list_result is not None:
            return self.list_result
        labels = filters["label"]
        return [
            volume
            for volume in self.by_name.values()
            if not volume.removed
            and all(
                volume.attrs["Labels"].get(key) == value
                for key, value in (label.split("=", 1) for label in labels)
            )
        ]


class FakeDockerImage:
    def __init__(self, image_id: str, attrs: dict[str, Any] | None = None) -> None:
        self.id = image_id
        self.attrs = attrs or {
            "Id": image_id,
            "RepoDigests": [],
            "Config": {"WorkingDir": "/workspace", "User": ""},
        }

    def reload(self) -> None:
        return None


class FakeDockerImages:
    def __init__(self) -> None:
        self.by_ref: dict[str, FakeDockerImage] = {}
        self.pulled: list[str] = []
        self.built: list[dict[str, Any]] = []
        self.removed: list[str] = []

    def get(self, ref: str) -> FakeDockerImage:
        from docker.errors import ImageNotFound

        if ref not in self.by_ref:
            raise ImageNotFound(ref)
        return self.by_ref[ref]

    def pull(self, ref: str) -> FakeDockerImage:
        self.pulled.append(ref)
        image = FakeDockerImage(f"sha256:pulled-{len(self.pulled)}")
        self.by_ref[ref] = image
        return image

    def list(self, *, filters: dict[str, list[str]]) -> list[FakeDockerImage]:
        required = tuple(
            label.split("=", 1) for label in filters.get("label", ())
        )
        return list(
            dict.fromkeys(
                image
                for image in self.by_ref.values()
                if all(
                    (image.attrs.get("Config") or {})
                    .get("Labels", {})
                    .get(key)
                    == value
                    for key, value in required
                )
            )
        )

    def remove(self, image_id: str, *, force: bool = False) -> None:
        assert force is False
        self.removed.append(image_id)
        self.by_ref = {
            ref: image
            for ref, image in self.by_ref.items()
            if image.id != image_id
            and image.attrs.get("Id") != image_id
        }

    def build(self, **kwargs: Any):
        record = dict(kwargs)
        path = Path(kwargs["path"])
        record["context_files"] = {
            item.relative_to(path).as_posix(): item.read_text()
            for item in path.rglob("*")
            if item.is_file()
        }
        self.built.append(record)
        image = FakeDockerImage(f"sha256:built-{len(self.built)}")
        if "tag" in kwargs:
            self.by_ref[kwargs["tag"]] = image
        return image, iter(({"stream": "ok"},))


class FakeDockerNetwork:
    def __init__(self, network_id: str, attrs: dict[str, Any]) -> None:
        self.id = network_id
        self.attrs = attrs
        self.removed = False
        self.connected: list[str] = []

    def remove(self) -> None:
        self.removed = True

    def reload(self) -> None:
        pass

    def connect(self, container: str) -> None:
        self.connected.append(container)


class FakeDockerNetworks:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.by_id: dict[str, FakeDockerNetwork] = {}

    def create(self, name: str, **kwargs: Any) -> FakeDockerNetwork:
        self.created.append({"name": name, **kwargs})
        network = FakeDockerNetwork(
            f"network-{len(self.created)}",
            {
                "Name": name,
                "Driver": kwargs.get("driver"),
                "Internal": kwargs.get("internal", False),
                "Labels": kwargs.get("labels", {}),
                "Options": kwargs.get("options", {}),
            },
        )
        self.by_id[network.id] = network
        return network

    def get(self, network_id: str) -> FakeDockerNetwork:
        from docker.errors import NotFound

        network = self.by_id.get(network_id)
        if network is None or network.removed:
            raise NotFound(network_id)
        return network

    def list(self, *, filters: dict[str, object] | None = None):
        active = [network for network in self.by_id.values() if not network.removed]
        if not filters:
            return active
        names = filters.get("name")
        if names:
            expected_names = {names} if isinstance(names, str) else set(names)
            active = [
                network
                for network in active
                if network.attrs.get("Name") in expected_names
            ]
        labels = filters.get("label") or ()
        if isinstance(labels, str):
            labels = (labels,)
        expected_labels = dict(label.split("=", 1) for label in labels)
        return [
            network
            for network in active
            if all(
                network.attrs.get("Labels", {}).get(key) == value
                for key, value in expected_labels.items()
            )
        ]


class FakeFirewallBackend:
    def __init__(self, *, probe_result: bool = True) -> None:
        self.probe_result = probe_result
        self.events: list[tuple[str, object]] = []
        self.installed: dict[str, object] = {}

    def probe(self) -> bool:
        self.events.append(("probe", None))
        return self.probe_result

    def install(self, rule_id: str, rules: object) -> None:
        self.events.append(("install", rule_id))
        self.installed[rule_id] = rules

    def is_installed(self, rule_id: str, rules: object) -> bool:
        self.events.append(("is_installed", rule_id))
        return self.installed.get(rule_id) == rules

    def exists(self, rule_id: str) -> bool:
        self.events.append(("exists", rule_id))
        return rule_id in self.installed

    def remove(self, rule_id: str) -> None:
        self.events.append(("remove", rule_id))
        self.installed.pop(rule_id, None)
