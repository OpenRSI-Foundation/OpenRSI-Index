from __future__ import annotations

import hashlib
import json
import stat

import pytest
from pydantic import ValidationError

from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    JudgeLLMAuth,
    JudgeLLMRequest,
    RunRequest,
)
from rsi_harness.runtime.run_request import (
    RUN_REQUEST_FILE,
    read_run_request,
    write_run_request,
)


def test_run_request_round_trips_privately_without_secret_values(tmp_path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text("[sandbox]\n")
    request = RunRequest(
        task_dir=tmp_path / "task",
        agent_name="claude",
        gpu_selectors=("0", "1"),
        options=CompileOptions(
            agent_name="claude",
            primary_reward="reward",
            score_direction="minimize",
            agent_timeout_seconds=600.0,
            max_submissions=3,
            cooldown_seconds=5.0,
            disable_stop_hook=True,
        ),
        model="model-x",
        reasoning_effort="high",
        agent_auth=AgentAuthSource.LOCAL,
        judge_llm=JudgeLLMRequest(
            auth=JudgeLLMAuth.API,
            api_key_env="JUDGE_KEY",
            base_url="https://judge-user:pw-secret@judge.example/v1",
            proxy="https://ghp_SeCrEtToKeN123456@proxy.example:3128",
        ),
    )

    written = write_run_request(
        tmp_path, run_id="run-1", request=request, sandbox_policy_path=policy
    )

    path = tmp_path / RUN_REQUEST_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text()
    for secret in ("judge-user", "pw-secret", "ghp_SeCrEtToKeN123456"):
        assert secret not in text
    loaded = read_run_request(tmp_path)
    assert loaded == written
    assert loaded.run_id == "run-1"
    assert loaded.request.options == request.options
    assert loaded.request.gpu_selectors == ("0", "1")
    assert loaded.request.agent_auth is AgentAuthSource.LOCAL
    assert loaded.request.judge_llm is not None
    assert loaded.request.judge_llm.api_key_env == "JUDGE_KEY"
    assert loaded.request.judge_llm.base_url == "https://[REDACTED]@judge.example/v1"
    assert loaded.request.judge_llm.proxy == "https://[REDACTED]@proxy.example:3128"
    assert loaded.sandbox_policy is not None
    assert loaded.sandbox_policy.path == policy
    assert (
        loaded.sandbox_policy.sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_run_request_rejects_an_unknown_schema(tmp_path) -> None:
    write_run_request(tmp_path, run_id="run-1", request=RunRequest(task_dir=tmp_path))
    path = tmp_path / RUN_REQUEST_FILE
    payload = json.loads(path.read_text())
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="schema version"):
        read_run_request(tmp_path)
