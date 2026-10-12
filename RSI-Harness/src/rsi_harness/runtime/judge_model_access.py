"""The Agent model access a task may grant its Judge, and nothing more.

A task opts in with ``[metadata.rsi_harness.verifier] model_access = true``.
The Judge then calls the run's own Agent through ``rsi-agent``, the only
supported way for a Judge to call a model: the same Agent CLI, environment
(API key, base URL, proxy, model, reasoning effort) and, for a local login,
the same in-memory credential files on a private tmpfs. Those stay out of the
Judge procedure's environment; only ``rsi-agent`` reads them, from a launch
spec on a private tmpfs. The Judge's network reaches only the provider/proxy
endpoints pinned for Work; every other destination stays rejected, so model
access never opens general egress.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    rsi_loop_oneshot,
    rsi_loop_runtime_secret_values,
)
from rsi_harness.judge_agent import (
    JUDGE_AGENT_PRIVATE,
    JUDGE_AGENT_SLOTS,
    JUDGE_AGENT_STATE_ROOT,
    JUDGE_AGENT_UID,
)
from rsi_harness.models import JUDGE_TRAJECTORY_ROOT
from rsi_harness.runtime.local_auth import AgentAuthMaterial
from rsi_harness.runtime.network import PinnedEndpoint
from rsi_harness.runtime.redaction import is_credential_name
from rsi_loop.harness.config import RSILoopConfig

# The Agent's HOME in Work; a local login's credential files sit below it.
JUDGE_AGENT_HOME = PurePosixPath("/home/agent")
JUDGE_AGENT_NAME_ENV = "RSI_AGENT_NAME"
JUDGE_AGENT_MODEL_ENV = "RSI_AGENT_MODEL"
JUDGE_AGENT_REASONING_EFFORT_ENV = "RSI_AGENT_REASONING_EFFORT"
LAUNCH_SPEC_VERSION = 1
# Never a working directory for the Agent: Judge outputs and private state.
_PROTECTED = (
    PurePosixPath("/logs/verifier"),
    JUDGE_TRAJECTORY_ROOT,
    PurePosixPath("/tests"),
    JUDGE_AGENT_PRIVATE,
)


@dataclass(frozen=True, slots=True, repr=False)
class JudgeModelAccess:
    """Pinned endpoints, the procedure's environment, and rsi-agent's spec.

    ``environment`` is what the Judge procedure sees: only the Agent's name,
    model and reasoning effort. ``launch`` is the JSON launch spec, with the
    Agent's credentials, that only ``rsi-agent`` reads.
    """

    endpoints: tuple[PinnedEndpoint, ...]
    environment: tuple[tuple[str, str], ...]
    auth: AgentAuthMaterial | None
    secret_values: frozenset[str]
    launch: bytes

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise SetupError(
                "Judge model access requires a pinned Agent provider or proxy endpoint"
            )
        if not isinstance(self.launch, bytes) or not self.launch:
            raise SetupError("Judge model access requires an rsi-agent launch spec")

    def __repr__(self) -> str:
        hosts = ", ".join(endpoint.hostname for endpoint in self.endpoints)
        return f"JudgeModelAccess(endpoints=[{hosts}], <redacted>)"

    @property
    def extra_hosts(self) -> tuple[tuple[str, str], ...]:
        """Name the pinned addresses; a no-network Judge resolves nothing else."""
        return tuple(
            dict.fromkeys(
                (endpoint.hostname, str(address))
                for endpoint in self.endpoints
                for address in endpoint.addresses
            )
        )


def resolve_judge_model_access(
    definition: Any,
    config: RSILoopConfig,
    agent_auth: AgentAuthMaterial | None,
    api_endpoints: tuple[PinnedEndpoint, ...],
) -> JudgeModelAccess | None:
    """Grant the Judge the Work Agent's model access only on task opt-in."""

    if not definition.verifier.model_access:
        return None
    if not api_endpoints:
        raise SetupError(
            "Judge model access requires a pinned Agent provider or proxy "
            "endpoint; configure the Agent API base URL or a proxy"
        )
    oneshot = rsi_loop_oneshot(config, definition.agent)
    agent_environment: dict[str, str] = oneshot["environment"]
    secret_values = _secret_values(agent_environment, config, agent_auth)
    launch = build_launch_spec(oneshot, agent_auth, secret_values)
    environment = {JUDGE_AGENT_NAME_ENV: oneshot["agent"]}
    if oneshot["model"]:
        environment[JUDGE_AGENT_MODEL_ENV] = oneshot["model"]
    if oneshot["reasoning_effort"]:
        environment[JUDGE_AGENT_REASONING_EFFORT_ENV] = oneshot["reasoning_effort"]
    return JudgeModelAccess(
        endpoints=tuple(api_endpoints),
        environment=tuple(sorted(environment.items())),
        auth=agent_auth,
        secret_values=frozenset(secret_values),
        launch=launch,
    )


def build_launch_spec(
    oneshot: Mapping[str, Any],
    agent_auth: AgentAuthMaterial | None,
    secret_values: set[str] | frozenset[str],
) -> bytes:
    """The launch spec ``rsi-agent`` reads from the Judge's private tmpfs."""
    credentials = []
    protected = [str(path) for path in _PROTECTED]
    for mount in () if agent_auth is None else agent_auth.mounts:
        target = mount.tmpfs.target
        try:
            relative = target.relative_to(JUDGE_AGENT_HOME)
        except ValueError:
            raise SetupError(
                "Agent login files must sit below the Agent's home to reach rsi-agent"
            ) from None
        protected.append(str(target))
        for auth_file in mount.files:
            credentials.append(
                {
                    "source": str(target / auth_file.path),
                    "target": str(relative / auth_file.path),
                }
            )
    spec = {
        "version": LAUNCH_SPEC_VERSION,
        "agent": oneshot["agent"],
        "model": oneshot["model"],
        "reasoning_effort": oneshot["reasoning_effort"],
        "uid": JUDGE_AGENT_UID,
        "gid": JUDGE_AGENT_UID,
        "slots": JUDGE_AGENT_SLOTS,
        "state_root": str(JUDGE_AGENT_STATE_ROOT),
        "command": oneshot["command"],
        "schema_option": oneshot["schema_option"],
        "output": oneshot["output"],
        "unsupported": oneshot["unsupported"],
        "environment": dict(oneshot["environment"]),
        "credentials": credentials,
        "protected": protected,
        "redact": sorted(secret_values),
    }
    return json.dumps(spec, sort_keys=True, allow_nan=False).encode()


def _secret_values(
    environment: Mapping[str, str],
    config: RSILoopConfig,
    agent_auth: AgentAuthMaterial | None,
) -> set[str]:
    values = set(rsi_loop_runtime_secret_values(config))
    values.update(
        value for key, value in environment.items() if value and is_credential_name(key)
    )
    if agent_auth is not None:
        values.update(agent_auth.secret_values)
    return {value for value in values if value}


__all__ = [
    "JUDGE_AGENT_HOME",
    "JUDGE_AGENT_MODEL_ENV",
    "JUDGE_AGENT_NAME_ENV",
    "JUDGE_AGENT_REASONING_EFFORT_ENV",
    "JudgeModelAccess",
    "build_launch_spec",
    "resolve_judge_model_access",
]
