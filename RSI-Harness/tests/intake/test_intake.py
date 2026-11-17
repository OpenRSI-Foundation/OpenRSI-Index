from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rsi_harness.intake.bundle import BundleError, seal_task_tree, unpack_bundle
from rsi_harness.intake.executor import ExecutionOutcome, HarnessExecutor, RunProfile
from rsi_harness.intake.protocol import (
    ExecutionMode,
    sign_dispatch,
    sign_webhook,
    verify_dispatch_signature,
)
from rsi_harness.intake.server import create_app
from rsi_harness.intake.service import IntakeService, IntakeSettings
from rsi_harness.intake.staging import (
    RunOptions,
    StagedTask,
    StagingError,
    canonical_task_id,
    dump_toml,
    stage_task,
)

SECRET = "dispatch-secret"
TOKEN = "bearer-token"
WEBHOOK_SECRET = "webhook-secret"
BREWER = "https://brewer.example.org"

BREWER_TASK_TOML = """\
schema_version = "1.4+ale+openrsi"

[task]
name = "OpenRSI-X/Ising_Monte_Carlo"
version = "1.0.0"
description = "Estimate Tc."

[metadata]
author_name = "Dr. Researcher"

[environment]
workdir = "/app"

[verifier]
timeout_sec = 600.0

[rsi.domain]
category = "Science"
subcategory = "Statistical physics"

[rsi.evaluation]
mode = "benchmark"
max_submissions = 3
score_direction = "minimize"
cooldown_sec = 5

[rsi.runtime]
clean_image = true

[comment]
source = "ale-brew-science"
"""


# --- fixtures and helpers ---


def _write_task(root: Path, toml: str = BREWER_TASK_TOML) -> Path:
    files = {
        "task.toml": (toml, 0o644),
        "instruction.md": ("Estimate the critical temperature.\n", 0o644),
        "environment/Dockerfile": ("FROM python:3.12-slim\nWORKDIR /app\n", 0o644),
        "tests/test.sh": ("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n", 0o755),
        "solution/solve.sh": ("#!/bin/sh\necho solved\n", 0o755),
    }
    for rel, (text, mode) in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(mode)
    for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
        directory.chmod(0o755)
    return root


def _seal(root: Path, task_dir: Path, *, failed_check: bool = False) -> None:
    files, digest = seal_task_tree(task_dir)
    report = {
        "schema_version": 1,
        "task_digest": digest,
        "checks": [{"id": "toml", "status": "failed" if failed_check else "passed"}],
    }
    report_bytes = json.dumps(report, indent=2).encode()
    (root / "offline-report.json").write_bytes(report_bytes)
    artifact = {
        "task_digest": digest,
        "offline_report_digest": "sha256:" + hashlib.sha256(report_bytes).hexdigest(),
        "files": files,
    }
    (root / "artifact.json").write_text(json.dumps(artifact))


def _tar_gz(source: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for path in sorted(source.rglob("*")):
            tf.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
    return buffer.getvalue()


def sealed_bundle(
    tmp_path: Path,
    *,
    tamper: bool = False,
    failed_check: bool = False,
    toml: str = BREWER_TASK_TOML,
) -> bytes:
    root = tmp_path / f"sealed-{time.monotonic_ns()}"
    task_dir = _write_task(root / "task", toml)
    _seal(root, task_dir, failed_check=failed_check)
    if tamper:
        (task_dir / "tests/test.sh").write_text("#!/bin/sh\necho 0\n")
    return _tar_gz(root)


def dispatch_payload(bundle: bytes, **overrides: Any) -> dict[str, Any]:
    dispatch_id = overrides.pop("dispatch_id", "2c8d3b72-1234-4a5b-9f01-aabbccddeeff")
    payload: dict[str, Any] = {
        "schema_version": "ale.task.intake.v1",
        "dispatch_id": dispatch_id,
        "idempotency_key": f"ale-brewer-job-{dispatch_id[:8]}",
        "submitted_at": "2026-11-16T23:55:00+00:00",
        "task": {
            "task_name": "ising_monte_carlo",
            "task_repo": "OpenRSI-Foundation/OpenRSI-X",
            "domain": "Science/Statistical physics",
        },
        "author": {"name": "Dr. Researcher"},
        "review": {"decision": "approve"},
        "artifact": {
            "name": "task-bundle.tar.gz",
            "sha256": "sha256:" + hashlib.sha256(bundle).hexdigest(),
            "size_bytes": len(bundle),
            "download_url": (
                f"{BREWER}/api/v1/outbox/bundles/{dispatch_id}?expires=1&signature=x"
            ),
        },
    }
    payload.update(overrides)
    return payload


def signed_headers(
    body: bytes,
    key: str,
    *,
    secret: str = SECRET,
    token: str = TOKEN,
    ts: int | None = None,
) -> dict[str, str]:
    stamp = str(int(time.time()) if ts is None else ts)
    return {
        "X-ALE-Idempotency-Key": key,
        "X-ALE-Timestamp": stamp,
        "X-ALE-Signature": sign_dispatch(secret, stamp, key, body),
        "Authorization": f"Bearer {token}",
    }


class _Response:
    def __init__(
        self,
        status_code: int,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.raw = io.BytesIO(content)
        self.text = content.decode(errors="replace")
        self.headers = headers or {}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeHttp:
    def __init__(self) -> None:
        self.bundles: dict[str, bytes] = {}
        self.redirects: dict[str, str] = {}
        self.downloads: list[str] = []
        self.webhooks: list[tuple[str, bytes, dict[str, str]]] = []
        self.on_webhook: Any = None

    def get(self, url: str, **_: Any) -> _Response:
        self.downloads.append(url)
        if url in self.redirects:
            return _Response(302, headers={"Location": self.redirects[url]})
        for prefix, content in self.bundles.items():
            if url.startswith(prefix):
                return _Response(200, content)
        return _Response(404, b"missing")

    def post(
        self, url: str, data: bytes, headers: dict[str, str], **_: Any
    ) -> _Response:
        self.webhooks.append((url, data, headers))
        if self.on_webhook is not None:
            self.on_webhook(json.loads(data))
        return _Response(200)


class FakeExecutor:
    def __init__(self, outcomes: list[ExecutionOutcome] | None = None) -> None:
        self.calls: list[tuple[ExecutionMode, StagedTask]] = []
        self.outcomes = outcomes or []

    def execute(
        self, mode: ExecutionMode, staged: StagedTask, work_dir: Path
    ) -> ExecutionOutcome:
        self.calls.append((mode, staged))
        if self.outcomes:
            return self.outcomes.pop(0)
        return ExecutionOutcome(
            True,
            "completed",
            {
                "run_id": "run-1",
                "exit_code": 0,
                "final_result": {"status": "completed", "best_score": 1.0},
            },
        )


def make_service(
    tmp_path: Path,
    executor: FakeExecutor | None = None,
    http: FakeHttp | None = None,
    **settings: Any,
) -> tuple[IntakeService, FakeExecutor, FakeHttp]:
    executor = executor or FakeExecutor()
    http = http or FakeHttp()
    base: dict[str, Any] = {
        "secret": SECRET,
        "token": TOKEN,
        "webhook_url": "auto",
        "webhook_secret": WEBHOOK_SECRET,
        "state_dir": tmp_path / "state",
        "default_mode": ExecutionMode.RUN,
        "webhook_backoff_sec": 0.0,
    }
    base.update(settings)
    service = IntakeService(IntakeSettings(**base), executor, http=http)  # type: ignore[arg-type]
    return service, executor, http


def post_json(client: TestClient, payload: dict[str, Any], **header_overrides: str):
    body = json.dumps(payload).encode()
    headers = signed_headers(body, payload["idempotency_key"])
    headers.update(header_overrides)
    headers["Content-Type"] = "application/json"
    return client.post("/v1/intake/tasks", content=body, headers=headers)


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


# --- protocol ---


def test_signature_matches_the_spec_formula() -> None:
    body = b'{"a": 1}'
    ok, reason = verify_dispatch_signature(
        secret=SECRET,
        signature_header=sign_dispatch(SECRET, "1794960000", "key", body),
        timestamp_header="1794960000",
        idempotency_key="key",
        metadata_bytes=body,
        now=1794960100,
    )
    assert ok, reason
    import hmac

    expected = hmac.new(
        SECRET.encode(), b"1794960000.key." + body, hashlib.sha256
    ).hexdigest()
    assert sign_dispatch(SECRET, "1794960000", "key", body) == f"sha256={expected}"


@pytest.mark.parametrize(
    ("ts", "sig", "reason"),
    [
        ("1794960000", "sha256=" + "0" * 64, "invalid X-ALE-Signature"),
        ("1794950000", None, "stale X-ALE-Timestamp"),
        ("soon", None, "invalid X-ALE-Timestamp"),
    ],
)
def test_signature_rejections(ts: str, sig: str | None, reason: str) -> None:
    body = b"{}"
    header = sig or sign_dispatch(SECRET, ts, "key", body)
    ok, got = verify_dispatch_signature(
        secret=SECRET,
        signature_header=header,
        timestamp_header=ts,
        idempotency_key="key",
        metadata_bytes=body,
        now=1794960000,
    )
    assert not ok and got == reason


# --- HTTP API ---


def test_health_is_public(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        for path in ("/healthz", "/v1/health"):
            body = client.get(path).json()
            assert body["status"] == "ok"
            assert body["service"] == "rsi-intake"
            assert body["protocol"] == "ale.task.intake.v1"


def test_pull_url_dispatch_runs_end_to_end(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, executor, http = make_service(tmp_path)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    with TestClient(create_app(service)) as client:
        response = post_json(
            client, payload, **{"X-ALE-Dispatch-Id": payload["dispatch_id"]}
        )
        assert response.status_code == 202, response.text
        receipt = response.json()
        assert receipt["schema_version"] == "ale.task.intake.receipt.v1"
        assert receipt["runtime"] == "rsi"
        assert receipt["created"] is True
        assert receipt["status"] == "acknowledged"
        assert receipt["execution_mode"] == "run"
        assert (
            receipt["ale_task_id"]
            == receipt["task_id"]
            == "openrsi-x/ising_monte_carlo"
        )
        service.wait_idle()

        view = client.get(receipt["status_url"], headers=auth()).json()
    assert view["status"] == "completed", view["detail"]
    assert view["attempt_count"] == 1
    assert view["task_id"] == "openrsi-x/ising_monte_carlo"
    result = view["result"]
    assert result["sealed"] is True and result["task_digest"].startswith("sha256:")
    assert result["adapted"] is True
    assert result["final_result"] == {"status": "completed", "best_score": 1.0}

    mode, staged = executor.calls[0]
    assert mode is ExecutionMode.RUN
    assert staged.run_options == RunOptions(
        score_direction="minimize", cooldown_seconds=5.0
    )
    config = tomllib.loads((staged.task_dir / "task.toml").read_text())
    assert config["schema_version"] == "1.4"
    assert "rsi" not in config and "comment" not in config
    assert config["metadata"]["openrsi"]["comment"] == {"source": "ale-brew-science"}
    assert config["metadata"]["rsi_harness"]["verifier"] == {
        "max_submissions": 3,
        "clean_image": True,
    }
    assert config["metadata"]["category"] == "Science"

    statuses = [json.loads(body)["status"] for _, body, _ in http.webhooks]
    assert statuses == ["running", "completed"]
    url, body, headers = http.webhooks[-1]
    assert url == f"{BREWER}/api/v1/webhooks/ale-status"
    assert headers["X-ALE-Signature"] == sign_webhook(WEBHOOK_SECRET, body)
    sent = json.loads(body)
    assert sent["dispatch_id"] == payload["dispatch_id"]
    assert sent["ale_task_id"] == "openrsi-x/ising_monte_carlo"
    assert sent["response_body"]["schema_version"] == "ale.task.intake.status.v1"


def test_idempotent_replay_and_conflict(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, _, http = make_service(tmp_path)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    with TestClient(create_app(service)) as client:
        first = post_json(client, payload)
        replay = post_json(client, payload)
        assert first.status_code == 202 and replay.status_code == 200
        assert replay.json()["created"] is False
        assert replay.json()["id"] == first.json()["id"]

        conflicting = dict(payload, artifact=dict(payload["artifact"], sha256="a" * 64))
        conflict = post_json(client, conflicting)
        assert conflict.status_code == 409
        assert "detail" in conflict.json()
        service.wait_idle()


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (lambda h, p: h.update({"Authorization": "Bearer wrong"}), 401),
        (lambda h, p: h.update({"X-ALE-Signature": "sha256=" + "0" * 64}), 401),
        (lambda h, p: h.update({"X-ALE-Timestamp": "1000"}), 401),
        (lambda h, p: h.update({"X-ALE-Dispatch-Id": "other"}), 422),
    ],
)
def test_dispatch_rejections(tmp_path: Path, mutate, status: int) -> None:
    service, _, _ = make_service(tmp_path)
    payload = dispatch_payload(b"bundle")
    body = json.dumps(payload).encode()
    headers = signed_headers(body, payload["idempotency_key"])
    headers["Content-Type"] = "application/json"
    mutate(headers, payload)
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/intake/tasks", content=body, headers=headers)
    assert response.status_code == status
    assert response.json()["detail"]


def test_schema_errors_are_422(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)
    payload = dispatch_payload(b"bundle", schema_version="ale.task.intake.v0")
    with TestClient(create_app(service)) as client:
        response = post_json(client, payload)
    assert response.status_code == 422
    assert "schema_version" in response.json()["detail"]


def test_token_only_server_ignores_unknown_hmac(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, _, http = make_service(tmp_path, secret="")
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    body = json.dumps(payload).encode()
    # The Brewer signs with its own session secret, unknown to this runtime.
    headers = signed_headers(body, payload["idempotency_key"], secret="brewer-session")
    headers["Content-Type"] = "application/json"
    with TestClient(create_app(service)) as client:
        assert (
            client.post("/v1/intake/tasks", content=body, headers=headers).status_code
            == 202
        )
        service.wait_idle()


def test_multipart_bundle(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, executor, http = make_service(tmp_path)
    payload = dispatch_payload(bundle, execution_mode="validate")
    payload["artifact"]["download_url"] = ""
    metadata = json.dumps(payload)
    headers = signed_headers(metadata.encode(), payload["idempotency_key"])
    with TestClient(create_app(service)) as client:
        bad = dict(
            payload,
            artifact=dict(payload["artifact"], sha256="b" * 64),
            idempotency_key="other-key",
        )
        bad_meta = json.dumps(bad)
        bad_headers = signed_headers(bad_meta.encode(), "other-key")
        rejected = client.post(
            "/v1/intake/tasks",
            headers=bad_headers,
            data={"metadata": bad_meta},
            files={"bundle": ("b.tar.gz", bundle)},
        )
        assert rejected.status_code == 422
        assert "sha256" in rejected.json()["detail"]

        response = client.post(
            "/v1/intake/tasks",
            headers=headers,
            data={"metadata": metadata},
            files={"bundle": ("task-bundle.tar.gz", bundle)},
        )
        assert response.status_code == 202, response.text
        assert response.json()["execution_mode"] == "validate"
        service.wait_idle()
        view = client.get(
            f"/v1/intake/tasks/{payload['dispatch_id']}", headers=auth()
        ).json()
    assert view["status"] == "completed", view["detail"]
    assert http.downloads == []
    assert executor.calls[0][0] is ExecutionMode.VALIDATE


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"tamper": True}, "do not match artifact.json"),
        ({"failed_check": True}, "failed checks"),
        (
            {"toml": BREWER_TASK_TOML + "\n[rsi.resources]\nworld = 4\n"},
            "rsi.resources.world",
        ),
    ],
)
def test_bad_bundles_fail_with_reason(
    tmp_path: Path, kwargs: dict[str, Any], detail: str
) -> None:
    bundle = sealed_bundle(tmp_path, **kwargs)
    service, executor, http = make_service(tmp_path)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    with TestClient(create_app(service)) as client:
        assert post_json(client, payload).status_code == 202
        service.wait_idle()
        view = client.get(
            f"/v1/intake/tasks/{payload['idempotency_key']}", headers=auth()
        ).json()
    assert view["status"] == "failed"
    assert detail in view["detail"]
    assert executor.calls == []
    assert json.loads(http.webhooks[-1][1])["status"] == "failed"


def test_download_sha_mismatch_fails(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, _, http = make_service(tmp_path)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle + b"x"
    with TestClient(create_app(service)) as client:
        post_json(client, payload)
        service.wait_idle()
        view = client.get(
            f"/v1/intake/tasks/{payload['dispatch_id']}", headers=auth()
        ).json()
    assert view["status"] == "failed"
    assert "does not match artifact.sha256" in view["detail"]


def test_retry_reuses_the_downloaded_bundle(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    executor = FakeExecutor(
        [ExecutionOutcome(False, "run failed: boom", {"exit_code": 1})]
    )
    service, _, http = make_service(tmp_path, executor=executor)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    with TestClient(create_app(service)) as client:
        intake_id = post_json(client, payload).json()["id"]
        service.wait_idle()
        failed = client.get(f"/v1/intake/tasks/{intake_id}", headers=auth()).json()
        assert failed["status"] == "failed" and "boom" in failed["detail"]

        assert (
            client.post(
                f"/v1/intake/tasks/{intake_id}/retry", headers=auth()
            ).status_code
            == 202
        )
        service.wait_idle()
        done = client.get(f"/v1/intake/tasks/{intake_id}", headers=auth()).json()
        assert done["status"] == "completed" and done["attempt_count"] == 2
        assert (
            client.post(
                f"/v1/intake/tasks/{intake_id}/retry", headers=auth()
            ).status_code
            == 409
        )
    assert len(http.downloads) == 1


def test_query_routes_need_bearer(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)
    with TestClient(create_app(service)) as client:
        assert client.get("/v1/intake/tasks").status_code == 401
        assert client.get("/v1/intake/tasks", headers=auth()).json() == {"tasks": []}
        assert (
            client.get("/v1/intake/tasks?status=bogus", headers=auth()).status_code
            == 422
        )
        assert client.get("/v1/intake/tasks/nope", headers=auth()).status_code == 404
        assert (
            client.post("/v1/intake/tasks/nope/retry", headers=auth()).status_code
            == 404
        )


def test_restart_requeues_unfinished_tasks(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, _, http = make_service(tmp_path)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    from rsi_harness.intake.protocol import IntakeDispatch

    record, created = service.submit(
        IntakeDispatch.model_validate(payload)
    )  # never started
    assert created
    service.store.close()

    restarted, executor, _ = make_service(tmp_path, http=http)
    restarted.start()
    restarted.wait_idle()
    assert restarted.get(record.id).status.value == "completed"
    assert len(executor.calls) == 1
    restarted.stop()


# --- staging, archives ---


def test_native_task_is_staged_untouched(tmp_path: Path) -> None:
    native = '[task]\nname = "rsi/science-ising3d"\n\n[environment]\ngpus = 0\n'
    source = _write_task(tmp_path / "src", 'schema_version = "1.4"\n' + native)
    staged = stage_task(source, tmp_path / "dst", fallback_name="x")
    assert not staged.adapted and staged.notices == ()
    assert (tmp_path / "dst/task.toml").read_text() == (
        source / "task.toml"
    ).read_text()
    assert staged.task_id == "rsi/science-ising3d"


def test_unsupported_rsi_runtime_sandbox_is_refused(tmp_path: Path) -> None:
    toml = BREWER_TASK_TOML.replace("clean_image = true", 'sandbox = "docker"')
    with pytest.raises(StagingError, match="rsi.runtime.sandbox"):
        stage_task(
            _write_task(tmp_path / "src", toml), tmp_path / "dst", fallback_name="x"
        )


def test_dump_toml_round_trips() -> None:
    data = {
        "schema_version": "1.4",
        "task": {"name": "a/b", "keywords": ["x", "y"]},
        "environment": {"env": {"A B": 'c"d\n'}, "gpus": 0, "cpus": 1.5},
        "artifacts": [{"source": "/a"}, {"source": "/b"}],
        "verifier": {"timeout_sec": 600.0, "inline": [{"k": 1}, 2]},
    }
    assert tomllib.loads(dump_toml(data)) == data


def test_task_ids_match_ale() -> None:
    assert (
        canonical_task_id("ising", task_repo="OpenRSI-Foundation/OpenRSI-X")
        == "openrsi-x/ising"
    )
    assert (
        canonical_task_id("Ising Model", domain="Science/Physics")
        == "science/ising_model"
    )
    assert canonical_task_id("rsi/hubbard2d") == "rsi/hubbard2d"


def _zip_with(name: str, mode: int | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        info = zipfile.ZipInfo(name)
        if mode is not None:
            info.external_attr = mode << 16
        zf.writestr(info, "x")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("archive", "message"),
    [
        (_zip_with("../escape.txt"), "escapes"),
        (_zip_with("/abs.txt"), "absolute"),
        (_zip_with("a\\b.txt"), "backslash"),
        (_zip_with("link", 0o120777), "link"),
    ],
)
def test_unsafe_archives_are_refused(
    tmp_path: Path, archive: bytes, message: str
) -> None:
    path = tmp_path / "bundle.zip"
    path.write_bytes(archive)
    with pytest.raises(BundleError, match=message):
        unpack_bundle(path, tmp_path / "out")


def test_tar_symlinks_are_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tf:
        info = tarfile.TarInfo("task.toml")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    path = tmp_path / "bundle.tar"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(BundleError, match="link"):
        unpack_bundle(path, tmp_path / "out")


# --- executor ---


class RecordingRunner:
    def __init__(self, exit_codes: dict[str, int]) -> None:
        self.exit_codes = exit_codes
        self.commands: list[list[str]] = []

    def __call__(
        self, command: list[str], cwd: Path, log: Path, env: dict[str, str]
    ) -> int:
        self.commands.append(command)
        assert not any(key.startswith("RSI_INTAKE_") for key in env)
        phase = cwd.name
        logs_root = Path(command[command.index("--logs-root") + 1])
        run_dir = logs_root / "runs" / f"{phase}-run" / "task"
        run_dir.mkdir(parents=True)
        (run_dir / "final_result.json").write_text(
            json.dumps({"status": "completed", "best_score": 0.5, "agent": phase})
        )
        log.write_text(f"{phase} log\nOPENAI_API_KEY=sk-secretsecretsecret\n")
        return self.exit_codes.get(phase, 0)


def _staged(tmp_path: Path) -> StagedTask:
    return StagedTask(
        task_dir=tmp_path / "task",
        task_id="x/y",
        adapted=True,
        notices=(),
        run_options=RunOptions(score_direction="minimize", cooldown_seconds=5.0),
    )


def test_run_mode_validates_then_runs_the_agent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_INTAKE_SECRET", "leak")
    runner = RecordingRunner({})
    executor = HarnessExecutor(
        RunProfile(agent="codex", model="m", timeout=60, judge_llm_auth="api"),
        runner=runner,
    )
    monkeypatch.setattr(
        executor, "_lint", lambda staged: ExecutionOutcome(True, "", {"lint_ok": True})
    )
    outcome = executor.execute(ExecutionMode.RUN, _staged(tmp_path), tmp_path / "work")
    assert outcome.ok, outcome.detail
    oracle, agent = runner.commands
    assert oracle[oracle.index("--agent") + 1] == "oracle"
    assert agent[agent.index("--agent") + 1] == "codex"
    assert agent[agent.index("--model") + 1] == "m"
    for command in (oracle, agent):
        assert command[command.index("--score-direction") + 1] == "minimize"
        assert command[command.index("--judge-llm-auth") + 1] == "api"
    assert outcome.result["validation"]["oracle_passed"] is True
    assert outcome.result["run_id"] == "run-run"
    assert outcome.result["final_result"]["agent"] == "run"


def test_failed_oracle_stops_before_the_agent(tmp_path: Path, monkeypatch) -> None:
    runner = RecordingRunner({"validate": 1})
    executor = HarnessExecutor(RunProfile(), runner=runner)
    monkeypatch.setattr(
        executor, "_lint", lambda staged: ExecutionOutcome(True, "", {"lint_ok": True})
    )
    outcome = executor.execute(ExecutionMode.RUN, _staged(tmp_path), tmp_path / "work")
    assert not outcome.ok
    assert outcome.detail.startswith("validation (oracle) failed")
    assert "sk-secretsecretsecret" not in outcome.detail
    assert len(runner.commands) == 1


def test_lint_compiles_a_staged_brewer_task(tmp_path: Path) -> None:
    source = _write_task(tmp_path / "src")
    staged = stage_task(source, tmp_path / "dst", fallback_name="x")
    outcome = HarnessExecutor(RunProfile()).execute(
        ExecutionMode.LINT, staged, tmp_path / "w"
    )
    assert outcome.ok, outcome.detail


@pytest.mark.parametrize(
    "extra",
    [
        ('[environment]\nworkdir = "/app"', 'network_mode = "no-network"'),
        (
            '[environment]\nworkdir = "/app"',
            'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]',
        ),
        ("[verifier]\ntimeout_sec = 600.0", 'network_mode = "no-network"'),
    ],
)
def test_lint_keeps_brewer_network_settings(
    tmp_path: Path, extra: tuple[str, str]
) -> None:
    anchor, addition = extra
    toml = BREWER_TASK_TOML.replace(anchor, f"{anchor}\n{addition}")
    staged = stage_task(
        _write_task(tmp_path / "src", toml), tmp_path / "dst", fallback_name="x"
    )
    outcome = HarnessExecutor(RunProfile()).execute(
        ExecutionMode.LINT, staged, tmp_path / "w"
    )
    assert outcome.ok, outcome.detail


# --- spec details: .tar.zst, digest spellings, webhook override ---


def _tar_zst(source: Path) -> bytes:
    import zstandard

    plain = io.BytesIO()
    with tarfile.open(fileobj=plain, mode="w") as tf:
        for path in sorted(source.rglob("*")):
            tf.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
    return zstandard.ZstdCompressor().compress(plain.getvalue())


def test_tar_zst_bundle_with_bare_hex_digests(tmp_path: Path) -> None:
    from rsi_harness.intake.bundle import tree_digest

    root = tmp_path / "sealed"
    task_dir = _write_task(root / "task")
    _seal(root, task_dir)
    artifact = json.loads((root / "artifact.json").read_text())
    for entry in artifact["files"]:
        entry["digest"] = entry["digest"].removeprefix("sha256:").upper()
    # The Brewer's task digest over the files list as recorded, written bare.
    bare_digest = tree_digest(artifact["files"]).removeprefix("sha256:")
    artifact["task_digest"] = bare_digest
    report = json.loads((root / "offline-report.json").read_text())
    report["task_digest"] = "sha256:" + bare_digest
    report_bytes = json.dumps(report).encode()
    (root / "offline-report.json").write_bytes(report_bytes)
    artifact["offline_report_digest"] = hashlib.sha256(report_bytes).hexdigest()
    (root / "artifact.json").write_text(json.dumps(artifact))
    bundle = _tar_zst(root)

    service, executor, http = make_service(tmp_path)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    with TestClient(create_app(service)) as client:
        assert post_json(client, payload).status_code == 202
        service.wait_idle()
        view = client.get(
            f"/v1/intake/tasks/{payload['dispatch_id']}", headers=auth()
        ).json()
    assert view["status"] == "completed", view["detail"]
    assert view["result"]["sealed"] is True
    assert len(executor.calls) == 1


def test_dispatch_webhook_url_overrides_the_configured_one(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, _, http = make_service(
        tmp_path, webhook_url="https://configured.example.org/hook"
    )
    payload = dispatch_payload(
        bundle, status_webhook_url=f"{BREWER}/api/v1/webhooks/ale-status"
    )
    http.bundles[payload["artifact"]["download_url"]] = bundle
    with TestClient(create_app(service)) as client:
        post_json(client, payload)
        service.wait_idle()
    assert {url for url, _, _ in http.webhooks} == {
        f"{BREWER}/api/v1/webhooks/ale-status"
    }


def test_dispatch_webhook_url_must_pass_allowed_origins(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path, allowed_origins=(BREWER,))
    payload = dispatch_payload(
        b"bundle", status_webhook_url="https://elsewhere.example.org/hook"
    )
    with TestClient(create_app(service)) as client:
        response = post_json(client, payload)
    assert response.status_code == 422
    assert "status_webhook_url" in response.json()["detail"]


# --- review fixes ---


def test_retry_during_webhook_delivery_is_not_lost(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    executor = FakeExecutor([ExecutionOutcome(False, "run failed: boom", {})])
    service, _, http = make_service(tmp_path, executor=executor)
    payload = dispatch_payload(bundle)
    http.bundles[payload["artifact"]["download_url"]] = bundle
    retried: list[str] = []

    def retry_while_delivering(sent: dict[str, Any]) -> None:
        # The operator retries while the "failed" webhook is in flight.
        if sent["status"] == "failed" and not retried:
            retried.append(service.retry(sent["dispatch_id"]).id)

    http.on_webhook = retry_while_delivering
    with TestClient(create_app(service)) as client:
        post_json(client, payload)
        service.wait_idle()
        view = client.get(
            f"/v1/intake/tasks/{payload['dispatch_id']}", headers=auth()
        ).json()
    assert retried
    assert view["status"] == "completed", view["detail"]
    assert view["attempt_count"] == 2
    assert view["webhook_delivered_at"]


def test_download_redirects_are_held_to_allowed_origins(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, executor, http = make_service(tmp_path, allowed_origins=(BREWER,))
    payload = dispatch_payload(bundle)
    url = payload["artifact"]["download_url"]
    http.redirects[url] = "http://169.254.169.254/latest/meta-data"
    with TestClient(create_app(service)) as client:
        assert post_json(client, payload).status_code == 202
        service.wait_idle()
        view = client.get(
            f"/v1/intake/tasks/{payload['dispatch_id']}", headers=auth()
        ).json()
    assert view["status"] == "failed"
    assert "redirect" in view["detail"]
    assert http.downloads == [url]
    assert executor.calls == []


def test_download_follows_an_allowed_redirect(tmp_path: Path) -> None:
    bundle = sealed_bundle(tmp_path)
    service, _, http = make_service(tmp_path, allowed_origins=(BREWER,))
    payload = dispatch_payload(bundle)
    url = payload["artifact"]["download_url"]
    http.redirects[url] = "/storage/bundle.tar.gz"
    http.bundles[f"{BREWER}/storage/"] = bundle
    with TestClient(create_app(service)) as client:
        post_json(client, payload)
        service.wait_idle()
        view = client.get(
            f"/v1/intake/tasks/{payload['dispatch_id']}", headers=auth()
        ).json()
    assert view["status"] == "completed", view["detail"]


def _tar_of(files: dict[str, bytes], modes: dict[str, int] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = (modes or {}).get(name, 0o644)
            tf.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_a_seal_with_extra_root_entries_is_refused_not_run_unsealed(
    tmp_path: Path,
) -> None:
    from rsi_harness.intake.bundle import verify_unpacked

    root = tmp_path / "sealed"
    _seal(root, _write_task(root / "task"))
    (root / "task.toml").write_text('schema_version = "1.4"\n')
    with pytest.raises(BundleError, match="extra: task.toml"):
        verify_unpacked(root)


def test_archive_litter_next_to_a_seal_is_ignored(tmp_path: Path) -> None:
    from rsi_harness.intake.bundle import verify_unpacked

    root = tmp_path / "unpacked"
    sealed = root / "bundle"
    _seal(sealed, _write_task(sealed / "task"))
    (root / "__MACOSX").mkdir()
    (sealed / ".DS_Store").write_bytes(b"\0")
    verified = verify_unpacked(root)
    assert verified.sealed and verified.task_dir == sealed / "task"


def test_tar_entries_without_permissions_stay_readable(tmp_path: Path) -> None:
    from rsi_harness.intake.bundle import verify_unpacked

    toml = b'schema_version = "1.4"\n\n[environment]\ngpus = 0\n'
    archive = tmp_path / "bundle.tar"
    archive.write_bytes(_tar_of({"task/task.toml": toml}, modes={"task/task.toml": 0}))
    unpack_bundle(archive, tmp_path / "out")
    verified = verify_unpacked(tmp_path / "out")
    assert (verified.task_dir / "task.toml").read_bytes() == toml


def test_read_only_task_toml_still_stages(tmp_path: Path) -> None:
    source = _write_task(tmp_path / "src")
    (source / "task.toml").chmod(0o444)
    staged = stage_task(source, tmp_path / "dst", fallback_name="x")
    assert staged.adapted
    assert "1.4" in (staged.task_dir / "task.toml").read_text()


def test_oversized_bodies_are_refused_before_reading(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path, max_bundle_bytes=1024)
    payload = dispatch_payload(b"x" * 10_000)
    payload["artifact"]["download_url"] = ""
    metadata = json.dumps(payload)
    headers = signed_headers(metadata.encode(), payload["idempotency_key"])
    with TestClient(create_app(service)) as client:
        too_big = client.post(
            "/v1/intake/tasks",
            headers=headers,
            data={"metadata": metadata},
            files={"bundle": ("b.tar.gz", b"x" * (5 * 1024 * 1024))},
        )
        assert too_big.status_code == 413

        def chunks():
            yield b'{"schema_version": "'
            for _ in range(5):
                yield b"x" * (1024 * 1024)

        chunked = client.post(
            "/v1/intake/tasks",
            content=chunks(),
            headers={**auth(), "Content-Type": "application/json"},
        )
        assert chunked.status_code == 413


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--cluster", "slurm", "--gpus", "0"], "--gpus is local-only"),
        (["--judge-llm-base-url", "http://x"], "need --judge-llm-auth"),
    ],
)
def test_serve_refuses_a_run_profile_run_would_reject(
    tmp_path: Path, args: list[str], message: str
) -> None:
    from typer.testing import CliRunner

    from rsi_harness.cli import app

    result = CliRunner().invoke(
        app,
        ["serve", "--state-dir", str(tmp_path), *args],
        env={"RSI_INTAKE_TOKEN": TOKEN},
    )
    assert result.exit_code == 2
    assert message in result.output
