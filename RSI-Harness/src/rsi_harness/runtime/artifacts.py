"""Atomically durable, archive-free run artifacts compatible with RSI Loop."""

from __future__ import annotations

import json
import math
import os
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from rsi_harness.models import (
    RunPlan,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.protocols import Clock
from rsi_harness.runtime.redaction import redact_text


class _SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def recorded_submissions(root: Path) -> tuple[tuple[SubmissionReport, float], ...]:
    """The rounds ``evolve_state.json`` records, with their submission times.

    The state file is written last, so a ``report.json`` of a round it does
    not list never counts. A listed round without one keeps its state entry.
    """
    state = root / "evolve_state.json"
    if not state.exists():
        return ()
    recorded = []
    for entry in json.loads(state.read_text())["submissions"]:
        report = root / "submissions" / str(entry["round"]) / "report.json"
        payload = json.loads(report.read_text()) if report.exists() else {}
        recorded.append(
            (
                SubmissionReport(
                    round_id=entry["round"],
                    status=entry["status"],
                    rewards=entry["rewards"],
                    score=entry["score"],
                    exit_code=payload.get("exit_code"),
                    timed_out=payload.get("timed_out", False),
                    verifier_output_required=payload.get(
                        "verifier_output_required", False
                    ),
                    full_output_captured=payload.get("full_output_captured", False),
                    duration_seconds=payload.get("runtime_seconds"),
                    error=payload.get("error"),
                ),
                float(entry["at"]),
            )
        )
    return tuple(recorded)


def _sudo_invoking_owner() -> tuple[int, int] | None:
    if os.geteuid() != 0:
        return None
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid is None and sudo_gid is None:
        return None
    if (
        sudo_uid is None
        or sudo_gid is None
        or not sudo_uid.isdecimal()
        or not sudo_gid.isdecimal()
    ):
        raise ValueError("sudo artifact owner identity is invalid")
    return int(sudo_uid), int(sudo_gid)


class RunArtifactWriter:
    """Write one run's visualizer surface without submission archives."""

    def __init__(
        self,
        run_plan: RunPlan,
        *,
        run_id: str,
        clock: Clock | None = None,
    ) -> None:
        self._run_plan = run_plan
        self._clock = clock or _SystemClock()
        self._reports: dict[str, SubmissionReport] = {}
        self._submitted_at: dict[str, float] = {}
        # Set by reopen when an earlier attempt recorded a round.
        self._submitted_earlier = False
        self._earlier_first_submission: float | None = None
        self._owner = _sudo_invoking_owner()
        self._ownership_root = run_plan.paths.logs / "runs"
        self.root = run_plan.paths.logs / "runs" / run_id / run_plan.task.task_id
        self.feedback_root = self.root / "feedback"

    def start(self) -> None:
        self._ensure_directory(self.root)
        self._ensure_directory(self.feedback_root)
        self.feedback_root.chmod(0o755)
        started_at = self._clock.now()
        self._atomic_write_text(
            self.root / "started_at",
            f"{started_at.isoformat()}\n{started_at.timestamp()}\n",
        )
        self._atomic_write_json(
            self.root / "run-plan.json",
            self._run_plan.model_dump(mode="json"),
        )
        self._atomic_write_text(self.root / "run_agent.log", "")
        self._atomic_write_text(self.root / "agent_output.txt", "")
        self._write_state()

    def reopen(self, attempt: int) -> None:
        """Continue the run's artifacts for a resumed ``attempt``.

        Nothing is truncated: earlier rounds are reloaded, the last attempt's
        result moves to ``attempts/<attempt - 1>/``, and the Agent output
        gains a banner line before the new attempt appends to it.
        """
        if not (self.root / "evolve_state.json").exists():
            raise FileNotFoundError(self.root / "evolve_state.json")
        recorded = recorded_submissions(self.root)
        for report, submitted_at in recorded:
            self._reports[report.round_id] = report
            self._submitted_at[report.round_id] = submitted_at
        if recorded:
            # The newest result kept it; an attempt that crashed wrote none.
            results = (
                self.root / "final_result.json",
                *(
                    self.root / "attempts" / str(number) / "final_result.json"
                    for number in range(attempt - 2, 0, -1)
                ),
            )
            result = next((path for path in results if path.exists()), None)
            self._submitted_earlier = True
            if result is not None:
                self._earlier_first_submission = json.loads(result.read_text()).get(
                    "first_submission_seconds"
                )
        previous = self.root / "attempts" / str(attempt - 1)
        for name in ("final_result.json", "engine_error.json"):
            source = self.root / name
            if not source.exists():
                continue
            self._ensure_directory(previous)
            os.replace(source, previous / name)
            self._fsync_directory(previous)
            self._fsync_directory(self.root)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root / "agent_output.txt", flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"\n--- rsi-harness: attempt {attempt} ---\n")
            stream.flush()
            os.fsync(stream.fileno())

    def record_submission(self, report: SubmissionReport) -> None:
        self._validate_report(report)
        submitted_at = self._clock.now().timestamp()
        self._require_finite(submitted_at, "submission timestamp")
        submission_dir = self._submission_dir(report.round_id)
        self._ensure_directory(submission_dir)
        payload = {
            "round": report.round_id,
            "status": report.status.value,
            "submitted_at": submitted_at,
            "score": report.score,
            "metrics": dict(report.rewards),
            "exit_code": report.exit_code,
            "timed_out": report.timed_out,
            "verifier_output_required": report.verifier_output_required,
            "full_output_captured": report.full_output_captured,
            "runtime_seconds": report.duration_seconds,
            "error": report.error,
        }
        self._atomic_write_json(submission_dir / "report.json", payload)
        feedback = self.feedback_root / f"{report.round_id}.log"
        if report.verifier_output_required and not report.full_output_captured:
            feedback.unlink(missing_ok=True)
            self._fsync_directory(self.feedback_root)
            raise OSError("complete verifier output capture was not durable")
        if report.verifier_output_required and not feedback.exists():
            raise OSError("complete verifier output artifact is missing")
        if not report.verifier_output_required and not feedback.exists():
            self._atomic_write_bytes(feedback, self._bounded_output(report.output))
            feedback.chmod(0o644)
        self._hardlink_output(feedback, submission_dir / "test_output.txt")
        self._reports[report.round_id] = report
        self._submitted_at[report.round_id] = submitted_at
        self.record_state(report)

    def record_state(self, report: SubmissionReport) -> None:
        """Upsert a report's RSI Loop submission-state entry."""
        self._validate_report(report)
        self._reports[report.round_id] = report
        if report.round_id not in self._submitted_at:
            submitted_at = self._clock.now().timestamp()
            self._require_finite(submitted_at, "submission timestamp")
            self._submitted_at[report.round_id] = submitted_at
        self._write_state()

    def finalize(
        self,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        runtime_seconds: float = 0.0,
        timed_out: bool = False,
        agent_exit_code: int | None = None,
        agent_runtime_seconds: float | None = None,
        first_submission_seconds: float | None = None,
        attempts: int | None = None,
    ) -> None:
        """Write ``final_result.json``.

        ``agent_runtime_seconds`` is the Agent's exec, and
        ``first_submission_seconds`` the time from its start to the start of
        the first Judge round that returned a report; both are None when
        there was none. A resumed run gives its number of ``attempts``, and
        its Agent seconds summed over them.
        """
        if self._submitted_earlier:
            # Measured in the earlier attempt that first submitted.
            first_submission_seconds = self._earlier_first_submission
        best = self._best_report()
        payload = {
            "best_score": None if best is None else best.score,
            "best_rewards": self._best_rewards(),
            "best_round": None if best is None else best.round_id,
            "total_rounds": len(self._reports),
            "agent": self._run_plan.task.agent.name,
            "model": self._run_plan.task.agent.model,
            "timeout_seconds": self._run_plan.task.agent.timeout_seconds,
            "timed_out": timed_out,
            "runtime_seconds": runtime_seconds,
            "status": status.value,
            "agent_exit_code": agent_exit_code,
            "agent_runtime_seconds": agent_runtime_seconds,
            "first_submission_seconds": first_submission_seconds,
        }
        if attempts is not None:
            payload["attempts"] = attempts
        self._atomic_write_json(self.root / "final_result.json", payload)

    def record_engine_error(self, error: str) -> None:
        """Persist a redacted Engine diagnostic outside task-authored raw output."""
        redacted = redact_text(error)
        self._atomic_write_json(self.root / "engine_error.json", {"error": redacted})

    def _write_state(self) -> None:
        best = self._best_report()
        submissions = [
            {
                "kind": "agent",
                "round": report.round_id,
                "at": self._submitted_at[report.round_id],
                "status": report.status.value,
                "score": report.score,
                "rewards": dict(report.rewards),
            }
            for report in self._reports.values()
        ]
        self._atomic_write_json(
            self.root / "evolve_state.json",
            {
                "best_score": None if best is None else best.score,
                "best_round": None if best is None else best.round_id,
                "submissions": submissions,
            },
        )

    def _best_report(self) -> SubmissionReport | None:
        scored = [
            report
            for report in self._reports.values()
            if report.status == SubmissionStatus.COMPLETED
            and report.score is not None
            and math.isfinite(report.score)
        ]
        if not scored:
            return None

        def score(report: SubmissionReport) -> float:
            assert report.score is not None
            return report.score

        if self._run_plan.task.score_direction == "minimize":
            return min(scored, key=score)
        return max(scored, key=score)

    def _best_rewards(self) -> dict[str, float]:
        valid = tuple(
            report
            for report in self._reports.values()
            if report.status == SubmissionStatus.COMPLETED
        )
        keys = sorted({key for report in valid for key in report.rewards})
        choose = min if self._run_plan.task.score_direction == "minimize" else max
        return {
            key: choose(
                report.rewards[key] for report in valid if key in report.rewards
            )
            for key in keys
        }

    def _bounded_output(self, output: str) -> bytes:
        limit = self._run_plan.task.verifier.output_limit_bytes
        encoded = output.encode()
        if len(encoded) <= limit:
            return encoded
        return encoded[:limit].decode(errors="ignore").encode()

    def _submission_dir(self, round_id: str) -> Path:
        separators = "/" in round_id or "\\" in round_id
        if not round_id or round_id in {".", ".."} or separators:
            raise ValueError(
                "submission round must be a single non-empty path component"
            )
        return self.root / "submissions" / round_id

    @classmethod
    def _validate_report(cls, report: SubmissionReport) -> None:
        if report.score is not None:
            cls._require_finite(report.score, "submission score")
        for reward_name, reward_value in report.rewards.items():
            cls._require_finite(reward_value, f"reward {reward_name!r}")
        if report.duration_seconds is not None:
            cls._require_finite(report.duration_seconds, "submission duration")

    @staticmethod
    def _require_finite(value: float, field: str) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")

    def _atomic_write_json(self, path: Path, payload: object) -> None:
        self._atomic_write_text(
            path,
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
        )

    def _atomic_write_text(self, path: Path, content: str) -> None:
        self._atomic_write_bytes(path, content.encode())

    def _atomic_write_bytes(self, path: Path, content: bytes) -> None:
        self._ensure_directory(path.parent)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
                if self._owner is not None:
                    os.fchown(temporary.fileno(), *self._owner)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
            self._fsync_directory(path.parent)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _hardlink_output(self, source: Path, target: Path) -> None:
        metadata = source.lstat()
        if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
            raise ValueError("complete verifier output must be a regular file")
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}"
        try:
            os.link(source, temporary, follow_symlinks=False)
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_directory(self, path: Path) -> None:
        missing: list[Path] = []
        current = path
        if current.is_symlink():
            raise NotADirectoryError(current)
        while not current.exists():
            missing.append(current)
            current = current.parent
            if current.is_symlink():
                raise NotADirectoryError(current)
        for directory in reversed(missing):
            directory.mkdir(exist_ok=True)
            self._fsync_directory(directory.parent)
        if not path.is_dir() or path.is_symlink():
            raise NotADirectoryError(path)
        if path == self._ownership_root or self._ownership_root in path.parents:
            candidate = self._ownership_root
            for component in path.relative_to(self._ownership_root).parts:
                metadata = candidate.lstat()
                if not stat.S_ISDIR(metadata.st_mode):
                    raise NotADirectoryError(candidate)
                candidate /= component
            metadata = candidate.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise NotADirectoryError(candidate)
        if self._owner is not None:
            current = path
            while (
                current == self._ownership_root
                or self._ownership_root in current.parents
            ):
                os.chown(current, *self._owner)
                current = current.parent

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


__all__ = ["RunArtifactWriter", "recorded_submissions"]
