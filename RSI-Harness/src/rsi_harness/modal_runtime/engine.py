"""Modal ports for the native RunCoordinator and SubmissionService.

No second scoring loop: numbering, limits, feedback, leases, best selection,
trajectory and final_result.json remain owned by RSI-Harness.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from rsi_harness.cluster.bluevela.network import AgentNetworkBroker
from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopAgentAdapter,
    rsi_loop_runtime_secret_values,
)
from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentResume,
    AgentRunRequest,
    ManagedNetwork,
    ManagedWorkdirVolume,
    RootfsSnapshotLease,
    RootfsSnapshotMode,
    RunRequest,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.coordinator import (
    EmbeddedSubmissionServerFactory,
    ProductionCoordinatorBackend,
    RunPreparation,
)
from rsi_harness.runtime.environment import resolve_runtime_environment
from rsi_harness.runtime.network import NetworkPolicyEnforcer
from rsi_harness.runtime.recovery import retained_round_id
from rsi_harness.runtime.redaction import redact_exact_values, redact_text
from rsi_harness.runtime.reward import read_reward
from rsi_harness.runtime.trajectory import (
    capture_round_trajectory,
    release_round_trajectory,
)
from rsi_loop.harness.config import RSILoopConfig

from .artifacts import ModalArtifactWriter
from .phases import WORK_UID, NamespaceRuntime, copy_regular_tree


class Clock:
    def now(self):
        return datetime.now(UTC)


class ModalAgentAdapter(RSILoopAgentAdapter):
    @staticmethod
    def _stage_solution(runtime, request):
        destination = runtime.root / "solution"
        if destination.exists():
            shutil.rmtree(destination)
        copy_regular_tree(request.run_plan.task.source_dir / "solution", destination)
        for parent, directories, files in os.walk(destination):
            os.chown(parent, WORK_UID, WORK_UID)
            for name in directories + files:
                os.chown(os.path.join(parent, name), WORK_UID, WORK_UID)


class ModalEngine:
    def __init__(self, root, plan, config, run_id, options, *, resume=None):
        self.plan, self.config, self.run_id, self.options = (
            plan,
            config,
            run_id,
            options,
        )
        self.resume = resume
        self.attempt = 1 if resume is None else resume.attempt
        self.clock = Clock()
        self.runtime = NamespaceRuntime(root, plan, config, run_id)
        # Only explicitly supplied API authentication enters Work. Never inherit
        # the Modal worker's environment or its control-plane credentials.
        self.agent_config = RSILoopConfig(
            agent_api_key=os.environ.get("RSI_AGENT_API_KEY"),
            agent_api_base_url=os.environ.get("RSI_AGENT_API_BASE_URL"),
            agent_model=plan.task.agent.model,
        )
        self.agent = ModalAgentAdapter(self.agent_config, runtime=self.runtime)
        self.secrets = rsi_loop_runtime_secret_values(self.agent_config)
        self.artifacts = self.volume = None
        self.deadline = None

    def request(self):
        return RunRequest(
            task_dir=self.plan.task.source_dir,
            agent_name=self.plan.task.agent.name,
            options=self.options,
            paths=self.plan.paths,
            model=self.plan.task.agent.model,
            reasoning_effort=self.plan.task.agent.reasoning_effort,
        )

    def backend(self):
        return ProductionCoordinatorBackend(
            compiler=None,
            allocator=lambda *_: self.plan.gpu_plan,
            image_preparer=lambda *_: self.plan.images,
            plan_preparer=lambda *_: self.plan,
            artifact_starter=self.start_artifacts,
            server_starter=EmbeddedSubmissionServerFactory(
                bind_host="127.0.0.1", port=0, bridge_gateway="127.0.0.1"
            ),
            network_planner=lambda *_: f"modal-{self.run_id}",
            network_creator=lambda plan, run, name: ManagedNetwork(
                network_id=name,
                name=name,
                run_id=run,
                task_id=plan.task.task_id,
                role="work",
                internal=True,
            ),
            network_remover=lambda _: None,
            workdir_volume_planner=self.plan_volume,
            workdir_volume_creator=self.adopt_volume,
            workdir_volume_adopter=self.adopt_volume,
            workdir_volume_attester=self.attest_volume,
            workdir_volume_remover=lambda _: None,
            work_name_planner=lambda *_: self.runtime.work_ref.container_id,
            work_creator=self.create_work,
            work_feedback_attester=self.runtime._require_work,
            work_policy_planner=lambda *_: f"modal-namespaces-{self.run_id}",
            work_policy_installer=lambda *_: f"modal-namespaces-{self.run_id}",
            work_policy_remover=lambda _: None,
            work_starter=lambda _: None,
            work_remover=self.runtime.stop,
            hook_installer=self.install_hooks,
            agent_preparer=self.prepare_agent,
            agent_runner=self.run_agent,
            agent_stopper=self.stop_agent,
            work_quiescer=self.runtime.quiescence,
            retained_work_planner=self.plan_retained,
            work_retainer=self.retain,
            retained_work_releaser=lambda _: None,
            evaluator=self.evaluate,
            event_recorder=self.event,
            preparation=RunPreparation(
                request=self.request(),
                definition=self.plan.task,
                gpu_plan=self.plan.gpu_plan,
            ),
        )

    def start_artifacts(self, plan, run_id):
        self.artifacts = ModalArtifactWriter(plan, run_id=run_id)
        if self.resume:
            self.artifacts.reopen(self.resume.attempt)
        else:
            self.artifacts.start()
        self.runtime.feedback = self.artifacts.feedback_root
        return self.artifacts

    def plan_volume(self, plan, run_id):
        return ManagedWorkdirVolume(
            name=f"modal-{run_id}",
            run_id=run_id,
            task_id=plan.task.task_id,
            target=plan.workdir,
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce=hashlib.sha256(
                f"{run_id}:{plan.workdir}".encode()
            ).hexdigest(),
        )

    def adopt_volume(self, plan, run_id, volume):
        if volume != self.plan_volume(plan, run_id):
            raise SetupError("retained workspace authority differs from its plan")
        self.volume = volume
        return volume

    def attest_volume(self, work, volume):
        self.runtime._require_work(work)
        if volume != self.volume:
            raise SetupError("workspace authority differs from its plan")

    def create_work(self, *_):
        self.runtime.initialize(resume=self.resume is not None)
        return self.runtime.work_ref

    def install_hooks(self, plan, work, submit_url, token):
        self.secrets.update((submit_url, token))
        endpoints = (
            NetworkPolicyEnforcer(run_id=self.run_id).pin_endpoints(
                self.config.provider_urls
            )
            if self.config.provider_urls
            else ()
        )
        network = AgentNetworkBroker(
            Path(f"/tmp/rsi-net-{self.run_id[:16]}-{self.attempt}"),
            endpoints=endpoints,
            submit_url=submit_url,
        )
        self.runtime.network = network
        network.start()
        for path in (network.root, network.submit_socket, network.provider_socket):
            if path:
                os.chown(path, WORK_UID, WORK_UID)
        self.agent.install_hooks(AgentHookRequest(plan, work, submit_url, token))

    def prepare_agent(self, plan, maximum):
        prepared = self.agent.prepare(
            AgentPrepareRequest(
                plan,
                self.artifacts.root / "agent_prompt.md",
                maximum,
                resume=None
                if self.resume is None
                else AgentResume(
                    self.resume.attempt,
                    self.resume.message,
                    self.artifacts.root / "agent_output.txt",
                ),
            )
        )
        env = dict(prepared.environment)
        env.update(resolve_runtime_environment(plan.task.agent.environment, os.environ))
        env["CODEX_HOME"] = "/home/agent/.codex"
        return replace(prepared, environment=tuple(sorted(env.items())))

    def run_agent(self, prepared, work, timeout):
        if self.deadline is not None:
            timeout = min(timeout, max(0.001, self.deadline - time.time()))
        return self.agent.run(
            AgentRunRequest(
                prepared,
                work,
                timeout,
                output_path=self.artifacts.root / "agent_output.txt",
                output_redact_values=tuple(sorted(self.secrets)),
            )
        )

    def stop_agent(self, work):
        try:
            self.runtime.stop(work)
        finally:
            if self.runtime.network:
                self.runtime.network.stop()

    def evaluate(self, request, observer):
        try:
            return self._evaluate(request, observer)
        except Exception as error:
            # The native HTTP boundary reserves PermissionError for auth.
            # Backend failures are unscored and close further submissions.
            detail = redact_text(redact_exact_values(str(error), self.secrets))
            report = SubmissionReport(
                round_id=request.round_id,
                status=SubmissionStatus.INFRASTRUCTURE_ERROR,
                error=f"recovery_required: {type(error).__name__}: {detail}",
                output="Modal phase failed; inspect the native run artifacts.",
            )
            self.artifacts.record_submission(report)
            return report

    def _evaluate(self, request, observer):
        started = time.monotonic()
        runtime = self.runtime
        observer.resource_event(
            "work_pause_planned", work_container_id=request.work_container.container_id
        )
        runtime.pause(request.work_container)
        trajectory = None
        try:
            request.verifier_logs.mkdir(parents=True, mode=0o700, exist_ok=False)
            observer.resource_event("snapshot_planned", round_id=request.round_id)
            round_root = runtime.rounds / request.round_id
            round_root.mkdir()
            snapshot = round_root / "workspace"
            digest = copy_regular_tree(runtime.workspace, snapshot)
            (round_root / "candidate.json").write_text(json.dumps({"sha256": digest}))
            lease_id = f"workspace-{request.round_id}"
            observer.resource_event(
                "snapshot_acquired",
                snapshot_lease_id=lease_id,
                snapshot_merged_path=str(snapshot),
            )
            judge_id = f"modal-{self.run_id}-{request.round_id}"
            observer.resource_event("judge_planned", round_id=request.round_id)
            observer.resource_event("judge_created", judge_container_id=judge_id)
            trajectory = capture_round_trajectory(self.artifacts.root, request.round_id)
            result = runtime.run(
                self.plan.task.verifier.command,
                judge=True,
                snapshot=snapshot,
                logs=request.verifier_logs,
                trajectory=trajectory,
                timeout=min(
                    self.plan.task.verifier.timeout_seconds,
                    max(0.001, self.deadline - time.time()),
                )
                if self.deadline
                else self.plan.task.verifier.timeout_seconds,
                output_path=request.verifier_output,
                environment=resolve_runtime_environment(
                    self.plan.task.verifier.environment, os.environ
                ),
                secrets=tuple(self.secrets),
            )
            reward = read_reward(
                request.verifier_logs, self.plan.task.verifier.primary_reward
            )
            error = reward.error
            status = SubmissionStatus.COMPLETED
            if result.timed_out:
                status, error = SubmissionStatus.VERIFIER_TIMEOUT, "verifier timed out"
            elif result.exit_code != 0 or error:
                status = SubmissionStatus.VERIFIER_ERROR
                error = error or f"verifier exited with {result.exit_code}"
            scored = status is SubmissionStatus.COMPLETED
            report = SubmissionReport(
                round_id=request.round_id,
                status=status,
                rewards=reward.rewards if scored else {},
                score=reward.score if scored else None,
                output=result.output,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                verifier_output_required=True,
                full_output_captured=result.full_output_captured,
                duration_seconds=time.monotonic() - started,
                error=error,
            )
            self.artifacts.record_submission(report)
            observer.resource_event("judge_removed", judge_container_id=judge_id)
            observer.resource_event("snapshot_released", snapshot_lease_id=lease_id)
            return report
        finally:
            if trajectory:
                release_round_trajectory(trajectory)
            runtime.unpause(request.work_container)
            observer.resource_event(
                "work_unpaused", work_container_id=request.work_container.container_id
            )

    def plan_retained(self, *_):
        round_id = retained_round_id("final", self.attempt)
        digest = hashlib.sha256(f"{self.run_id}:{round_id}".encode()).hexdigest()
        return f"rsi-harness-rootfs:retained-work-{digest}"

    def retain(self, plan, work, planned):
        # Native lease identity for filesystem retention; this is not an OCI image.
        digest = hashlib.sha256(str(self.runtime.workspace).encode()).hexdigest()
        return RootfsSnapshotLease(
            lease_id=f"retained-{digest[:24]}",
            purpose="retained-work",
            run_id=self.run_id,
            task_id=plan.task.task_id,
            round_id=retained_round_id("final", self.attempt),
            source_container_id=work.container_id,
            image_id=f"sha256:{digest}",
            image_ref=planned,
        )

    @staticmethod
    def event(name, value):
        # Do not serialize plans, task environments or tokens into provider logs.
        print(f"[rsi-modal] {name}", flush=True)
