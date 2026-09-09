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
        self._hardlink_output(
            feedback, submission_dir / "test_output.txt"
        )
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
    ) -> None:
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
        }
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


__all__ = ["RunArtifactWriter"]
