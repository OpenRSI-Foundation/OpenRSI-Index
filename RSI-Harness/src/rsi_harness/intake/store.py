"""Durable intake records (SQLite), so a restart resumes the queue."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rsi_harness.intake.protocol import (
    RECEIPT_SCHEMA,
    RUNTIME_NAME,
    STATUS_SCHEMA,
    ExecutionMode,
    IntakeDispatch,
    IntakeStatus,
)


class IntakeRecord(BaseModel):
    id: str
    dispatch_id: str
    idempotency_key: str
    task_id: str
    status: IntakeStatus
    execution_mode: ExecutionMode
    artifact_sha256: str
    request: IntakeDispatch
    bundle_path: str | None = None
    webhook_url: str | None = None
    attempt_count: int = 0
    detail: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    received_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    webhook_delivered_at: str | None = None
    webhook_last_error: str = ""

    @property
    def status_url(self) -> str:
        return f"/v1/intake/tasks/{self.id}"

    def receipt(self, *, created: bool) -> dict[str, Any]:
        return {
            "schema_version": RECEIPT_SCHEMA,
            "id": self.id,
            "ale_task_id": self.task_id,
            "task_id": self.task_id,
            "runtime": RUNTIME_NAME,
            "dispatch_id": self.dispatch_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "created": created,
            "execution_mode": self.execution_mode.value,
            "status_url": self.status_url,
            "received_at": self.received_at,
        }

    def view(self) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA,
            "id": self.id,
            "ale_task_id": self.task_id,
            "task_id": self.task_id,
            "runtime": RUNTIME_NAME,
            "dispatch_id": self.dispatch_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "execution_mode": self.execution_mode.value,
            "attempt_count": self.attempt_count,
            "artifact_sha256": self.artifact_sha256,
            "detail": self.detail,
            "result": self.result,
            "received_at": self.received_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "webhook_delivered_at": self.webhook_delivered_at,
            "webhook_last_error": self.webhook_last_error,
        }


class IntakeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS intake_tasks (
                id TEXT PRIMARY KEY,
                dispatch_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                received_at TEXT NOT NULL,
                body TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS intake_dispatch ON intake_tasks (dispatch_id)"
        )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _one(self, sql: str, args: tuple[Any, ...]) -> IntakeRecord | None:
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return IntakeRecord.model_validate_json(row[0]) if row else None

    def get(self, ref: str) -> IntakeRecord | None:
        """By intake id, then dispatch id, then idempotency key."""
        for column in ("id", "dispatch_id", "idempotency_key"):
            record = self._one(
                f"SELECT body FROM intake_tasks WHERE {column} = ? "
                "ORDER BY received_at DESC LIMIT 1",
                (ref,),
            )
            if record is not None:
                return record
        return None

    def get_by_idempotency_key(self, key: str) -> IntakeRecord | None:
        return self._one(
            "SELECT body FROM intake_tasks WHERE idempotency_key = ?", (key,)
        )

    def insert(self, record: IntakeRecord) -> bool:
        """Insert a new record; False if its idempotency key already exists."""
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO intake_tasks VALUES (?, ?, ?, ?, ?, ?)",
                    self._row(record),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def put(self, record: IntakeRecord) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO intake_tasks VALUES (?, ?, ?, ?, ?, ?)",
                self._row(record),
            )

    def update_webhook(
        self, intake_id: str, delivered_at: str | None, last_error: str
    ) -> None:
        """Set only the webhook fields, atomically against other writers."""
        with self._lock:
            row = self._db.execute(
                "SELECT body FROM intake_tasks WHERE id = ?", (intake_id,)
            ).fetchone()
            if row is None:
                return
            record = IntakeRecord.model_validate_json(row[0])
            record.webhook_delivered_at = delivered_at
            record.webhook_last_error = last_error
            self._db.execute(
                "INSERT OR REPLACE INTO intake_tasks VALUES (?, ?, ?, ?, ?, ?)",
                self._row(record),
            )

    @staticmethod
    def _row(record: IntakeRecord) -> tuple[str, ...]:
        return (
            record.id,
            record.dispatch_id,
            record.idempotency_key,
            record.status.value,
            record.received_at,
            record.model_dump_json(),
        )

    def list(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[IntakeRecord]:
        sql = "SELECT body FROM intake_tasks"
        args: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            args = (status,)
        sql += " ORDER BY received_at DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, (*args, limit)).fetchall()
        return [IntakeRecord.model_validate_json(body) for (body,) in rows]

    def unfinished(self) -> list[IntakeRecord]:
        with self._lock:
            rows = self._db.execute(
                "SELECT body FROM intake_tasks WHERE status IN (?, ?) "
                "ORDER BY received_at",
                (IntakeStatus.ACKNOWLEDGED.value, IntakeStatus.RUNNING.value),
            ).fetchall()
        return [IntakeRecord.model_validate_json(body) for (body,) in rows]
