"""Durable record of the request that started a run, read back to rebuild it."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from rsi_harness.models import RunRequest, RunRequestRecord, SandboxPolicySource
from rsi_harness.runtime.durable import fsync_directory
from rsi_harness.runtime.redaction import redact_structure

RUN_REQUEST_FILE = "request.json"


def _without_userinfo(url: str | None) -> str | None:
    """A URL's username and password are credentials; keep everything else."""
    if url is None:
        return None
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit(parts._replace(netloc=f"[REDACTED]@{host}"))


def write_run_request(
    run_root: Path,
    *,
    run_id: str,
    request: RunRequest,
    sandbox_policy_path: Path | None = None,
) -> RunRequestRecord:
    """Atomically write ``run_root/request.json`` (0600) and fsync it."""
    if request.judge_llm is not None:
        request = request.model_copy(
            update={
                "judge_llm": request.judge_llm.model_copy(
                    update={
                        "base_url": _without_userinfo(request.judge_llm.base_url),
                        "proxy": _without_userinfo(request.judge_llm.proxy),
                    }
                )
            }
        )
    record = RunRequestRecord(
        run_id=run_id,
        request=request,
        sandbox_policy=(
            None
            if sandbox_policy_path is None
            else SandboxPolicySource(
                path=sandbox_policy_path,
                sha256=hashlib.sha256(sandbox_policy_path.read_bytes()).hexdigest(),
            )
        ),
    )
    payload = redact_structure(record.model_dump(mode="json"))
    # Refuse before writing a record that cannot be read back.
    record = RunRequestRecord.model_validate_json(json.dumps(payload))
    path = run_root / RUN_REQUEST_FILE
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=run_root
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(run_root)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def read_run_request(run_root: Path) -> RunRequestRecord:
    """Load ``run_root/request.json``."""
    return RunRequestRecord.model_validate_json(
        (run_root / RUN_REQUEST_FILE).read_text()
    )


__all__ = ["RUN_REQUEST_FILE", "read_run_request", "write_run_request"]
