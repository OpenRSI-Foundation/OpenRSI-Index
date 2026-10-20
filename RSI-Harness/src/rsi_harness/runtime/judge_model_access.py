"""The model access a task may grant its Judge, and nothing more.

``[metadata.rsi_harness.verifier] model_access = true``: the Judge calls the
run's own Agent through ``rsi-agent exec``, the same Agent CLI, environment
(API key, base URL, proxy, model, reasoning effort) and, for a local login,
the same credential files on a private tmpfs.

A local login is the one Work's Agent keeps refreshing: before each round the
Judge gets Work's current files (``JudgeModelAccess.with_work_login``), with
every refresh token replaced by a placeholder, so a Judge never refreshes, and
so never rotates away, the token Work depends on. rsi-agent refuses a call
the access token would not outlive.

``judge_model = "<id>"``: the Judge calls that model once through
``rsi-agent llm`` with the operator's Judge LLM credential (judge_llm.py).

``rsi-agent`` is the only supported way for a Judge to call a model. The
credentials stay out of the Judge procedure's environment; only ``rsi-agent``
reads them, from a launch spec on a private tmpfs. With ``pinned`` egress
(Docker, and a cluster whose Work Agent has no network) the Judge's network
reaches only the endpoints pinned for these calls; every other destination
stays rejected, so model access never opens general egress. With ``site``
egress (a cluster whose Work Agent reaches its model over the site network)
the Judge uses that same network, governed by the site rather than by the
Harness.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

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
from rsi_harness.runtime.judge_llm import JudgeLLMAccess, judge_codex_login
from rsi_harness.runtime.judge_login_refresh import OperatorCodexLogin
from rsi_harness.runtime.local_auth import (
    WITHHELD_REFRESH_TOKEN,
    AgentAuthMaterial,
    judge_login_copy,
    login_expiry,
    login_paths,
    login_secret_values,
    login_withheld_values,
    withheld_values,
    withhold_refresh_token,
)
from rsi_harness.runtime.network import PinnedEndpoint
from rsi_harness.runtime.redaction import is_credential_name
from rsi_loop.harness.config import RSILoopConfig

# The Agent's HOME in Work; a local login's credential files sit below it.
JUDGE_AGENT_HOME = PurePosixPath("/home/agent")
JUDGE_AGENT_NAME_ENV = "RSI_AGENT_NAME"
JUDGE_AGENT_MODEL_ENV = "RSI_AGENT_MODEL"
JUDGE_AGENT_REASONING_EFFORT_ENV = "RSI_AGENT_REASONING_EFFORT"
LAUNCH_SPEC_VERSION = 2
JudgeEgress = Literal["pinned", "site"]
IsolationFallback = Literal["none", "same-user"]
# Never a working directory for the Agent: Judge outputs and private state.
_PROTECTED = (
    PurePosixPath("/logs/verifier"),
    JUDGE_TRAJECTORY_ROOT,
    PurePosixPath("/tests"),
    JUDGE_AGENT_PRIVATE,
)


# Renewed logins whose credentials stay redacted: a Judge only ever holds the
# latest, and Work's older tokens are expired or rotated away.
KEPT_LOGIN_GENERATIONS = 4


@dataclass(frozen=True, slots=True, repr=False)
class JudgeModelAccess:
    """Pinned endpoints, the procedure's environment, and rsi-agent's spec.

    ``environment`` is what the Judge procedure sees: only the Agent's name,
    model and reasoning effort, and the Judge LLM's model and effort.
    ``launch`` is the JSON launch spec, with the credentials, that only
    ``rsi-agent`` reads. ``codex_cli`` is the host directory holding the
    pinned Codex CLI for a Codex-login Judge LLM, once it is prepared.
    ``operator_llm_model`` names the Judge model when its credential is the
    operator's rather than the Work Agent's: such a Judge needs the clean
    image.

    ``auth`` is the Judge's copy of the Work Agent's local login (refresh
    tokens withheld). ``work_login`` is that login as last read from Work,
    the reference the next read is checked against; it never reaches a
    Judge. It is set when a Judge uses the login: ``rsi-agent exec``, or
    ``rsi-agent llm`` when the Work Agent lends its ChatGPT login
    (``llm_reuses_login``).

    ``operator_login`` is the operator's Codex login for ``rsi-agent llm``
    (``--judge-llm-auth codex-login``), which the Harness keeps fresh on the
    host; each round's copy reaches the Judge through ``with_operator_login``.

    ``secret_values`` is ``base_secret_values`` (everything known before the
    first renewal, the run-start logins included) plus the credentials of the
    last ``KEPT_LOGIN_GENERATIONS`` logins read since (``login_generations``):
    bounded, however often the logins change. ``withheld_values`` are the
    refresh tokens among them: redacted on the host, but never in the launch
    spec, since no Judge ever holds one.
    """

    endpoints: tuple[PinnedEndpoint, ...]
    environment: tuple[tuple[str, str], ...]
    auth: AgentAuthMaterial | None
    secret_values: frozenset[str]
    launch: bytes
    egress: JudgeEgress = "pinned"
    needs_codex_cli: bool = False
    codex_cli: Path | None = None
    operator_llm_model: str | None = None
    work_login: AgentAuthMaterial | None = None
    llm_reuses_login: bool = False
    base_secret_values: frozenset[str] | None = None
    login_generations: tuple[frozenset[str], ...] = ()
    operator_login: OperatorCodexLogin | None = None
    withheld_values: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.egress not in ("pinned", "site"):
            raise SetupError("Judge model egress must be pinned or site")
        if self.egress == "pinned" and not self.endpoints:
            raise SetupError(
                "Judge model access requires a pinned Agent provider or proxy endpoint"
            )
        if not isinstance(self.launch, bytes) or not self.launch:
            raise SetupError("Judge model access requires an rsi-agent launch spec")

    def __repr__(self) -> str:
        hosts = ", ".join(endpoint.hostname for endpoint in self.endpoints)
        return (
            f"JudgeModelAccess(egress={self.egress}, endpoints=[{hosts}], <redacted>)"
        )

    def with_work_login(self, login: AgentAuthMaterial) -> JudgeModelAccess:
        """This access with ``login``, the Work Agent's current login.

        ``login`` was checked against ``work_login`` (local_auth.current_login).
        Its credentials join the redaction set, which keeps the run-start
        values and the last ``KEPT_LOGIN_GENERATIONS`` logins' credentials.
        """
        if self.work_login is None:
            raise SetupError("this Judge does not use the Work Agent's login")
        if login.agent_name != self.work_login.agent_name or login_paths(
            login
        ) != login_paths(self.work_login):
            raise SetupError("the Work Agent's login changed shape")
        base, generations, secret_values = self._secrets_with(login.secret_values)
        withheld = self._withheld_with(login_withheld_values(login), secret_values)
        spec = json.loads(self.launch)
        spec["redact"] = launch_redaction(secret_values, withheld)
        auth = None
        if self.auth is not None:
            auth = judge_login_copy(login)
            spec["credentials_expire_at"] = login_expiry(login)
        if self.llm_reuses_login:
            [content] = [
                entry.content for mount in login.mounts for entry in mount.files
            ]
            spec["llm"]["auth_json"] = withhold_refresh_token(
                login.agent_name, content
            ).decode("utf-8")
        return replace(
            self,
            auth=auth,
            secret_values=secret_values,
            launch=json.dumps(spec, sort_keys=True, allow_nan=False).encode(),
            work_login=login,
            base_secret_values=base,
            login_generations=generations,
            withheld_values=withheld,
        )

    def with_operator_login(self, content: bytes) -> JudgeModelAccess:
        """This access with ``content``, the operator's current Codex login.

        ``rsi-agent llm`` gets it with the refresh token withheld; its
        credentials join the redaction set as a Work login's do.
        """
        if self.operator_login is None:
            raise SetupError("this Judge does not use the operator's Codex login")
        login = json.loads(content.decode("utf-8"))
        base, generations, secret_values = self._secrets_with(
            login_secret_values(login)
        )
        withheld = self._withheld_with(withheld_values("codex", content), secret_values)
        spec = json.loads(self.launch)
        spec["redact"] = launch_redaction(secret_values, withheld)
        spec["llm"]["auth_json"] = judge_codex_login(content)
        return replace(
            self,
            secret_values=secret_values,
            launch=json.dumps(spec, sort_keys=True, allow_nan=False).encode(),
            base_secret_values=base,
            login_generations=generations,
            withheld_values=withheld,
        )

    def round_launch(self, ends_at: float) -> bytes:
        """The launch spec for one round, which the Harness stops at ``ends_at``.

        rsi-agent checks a login against a call's ``--timeout`` only up to
        then: no call outlives the round.
        """
        spec = json.loads(self.launch)
        spec["round_ends_at"] = ends_at
        return json.dumps(spec, sort_keys=True, allow_nan=False).encode()

    def _withheld_with(
        self, values: frozenset[str], secret_values: frozenset[str]
    ) -> frozenset[str]:
        """The refresh tokens among ``secret_values``, ``values`` included."""
        return (self.withheld_values | values) & secret_values

    def _secrets_with(
        self, values: frozenset[str]
    ) -> tuple[frozenset[str], tuple[frozenset[str], ...], frozenset[str]]:
        """The redaction set once a renewed login's ``values`` join it."""
        base = (
            self.secret_values
            if self.base_secret_values is None
            else self.base_secret_values
        )
        generations = (*self.login_generations, values)[-KEPT_LOGIN_GENERATIONS:]
        return base, generations, base.union(*generations)

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


def judge_needs_rsi_agent(verifier: Any) -> bool:
    """Whether a Judge gets rsi-agent: model access, a Judge LLM, or both."""
    return bool(verifier.model_access) or verifier.judge_model is not None


def resolve_judge_model_access(
    definition: Any,
    config: RSILoopConfig,
    agent_auth: AgentAuthMaterial | None,
    api_endpoints: tuple[PinnedEndpoint, ...],
    judge_llm: JudgeLLMAccess | None = None,
    *,
    egress: JudgeEgress = "pinned",
    isolation_fallback: IsolationFallback = "none",
    allow_missing_landlock_signal: bool = False,
    extra_agent_environment: Mapping[str, str] | None = None,
) -> JudgeModelAccess | None:
    """Grant the Judge model access only on task opt-in.

    ``isolation_fallback`` and ``allow_missing_landlock_signal`` come from
    the operator's cluster profile, never from a task.
    ``extra_agent_environment`` adds variables to the Agent's own
    environment (a cluster's pinned egress relay, for example).

    ``model_access``: the Work Agent's own access, through ``rsi-agent exec``.
    ``judge_model``: ``judge_llm``, through ``rsi-agent llm``.
    """

    model_access = bool(definition.verifier.model_access)
    if (definition.verifier.judge_model is None) != (judge_llm is None):
        raise SetupError("the Judge LLM differs from the task's judge_model")
    if not model_access and judge_llm is None:
        return None
    endpoints: list[PinnedEndpoint] = []
    environment: dict[str, str] = {}
    secret_values: set[str] = set()
    oneshot: Mapping[str, Any] | None = None
    auth = None
    if model_access:
        if egress == "pinned" and not api_endpoints:
            raise SetupError(
                "Judge model access requires a pinned Agent provider or proxy "
                "endpoint; configure the Agent API base URL or a proxy"
            )
        oneshot = rsi_loop_oneshot(config, definition.agent)
        agent_environment: dict[str, str] = oneshot["environment"]
        secret_values |= _secret_values(agent_environment, config, agent_auth)
        endpoints.extend(api_endpoints)
        # Never Work's refresh token: a Judge's refresh would rotate it away.
        auth = None if agent_auth is None else judge_login_copy(agent_auth)
        environment[JUDGE_AGENT_NAME_ENV] = oneshot["agent"]
        if oneshot["model"]:
            environment[JUDGE_AGENT_MODEL_ENV] = oneshot["model"]
        if oneshot["reasoning_effort"]:
            environment[JUDGE_AGENT_REASONING_EFFORT_ENV] = oneshot["reasoning_effort"]
    if judge_llm is not None:
        secret_values |= judge_llm.secret_values
        endpoints.extend(
            endpoint for endpoint in judge_llm.endpoints if endpoint not in endpoints
        )
        environment.update(judge_llm.environment)
    # Only a lent ChatGPT login is Work's own file; an API key never changes.
    llm_reuses_login = (
        judge_llm is not None
        and judge_llm.reused
        and judge_llm.needs_codex_cli
        and agent_auth is not None
    )
    # Refresh tokens: redacted on the host, never sent to a Judge.
    withheld = frozenset() if agent_auth is None else login_withheld_values(agent_auth)
    if judge_llm is not None:
        withheld |= judge_llm.withheld_values
    withheld &= frozenset(secret_values)
    launch = build_launch_spec(
        oneshot,
        auth,
        secret_values,
        visibility=definition.verifier.visibility,
        isolation_fallback=isolation_fallback,
        allow_missing_landlock_signal=allow_missing_landlock_signal,
        extra_environment=extra_agent_environment,
        llm=None if judge_llm is None else judge_llm.spec,
        credentials_expire_at=None if auth is None else login_expiry(auth),
        withheld_values=withheld,
    )
    return JudgeModelAccess(
        endpoints=tuple(endpoints),
        environment=tuple(sorted(environment.items())),
        auth=auth,
        secret_values=frozenset(secret_values),
        launch=launch,
        egress=egress,
        needs_codex_cli=judge_llm is not None and judge_llm.needs_codex_cli,
        operator_llm_model=(
            None if judge_llm is None or judge_llm.reused else judge_llm.model
        ),
        work_login=agent_auth if auth is not None or llm_reuses_login else None,
        llm_reuses_login=llm_reuses_login,
        operator_login=None if judge_llm is None else judge_llm.operator_login,
        withheld_values=withheld,
    )


# rsi-agent exec's answer in a Judge whose task only declared judge_model.
_NO_MODEL_ACCESS = (
    "this Judge has no model access (the task must declare "
    "[metadata.rsi_harness.verifier] model_access = true)"
)


def build_launch_spec(
    oneshot: Mapping[str, Any] | None,
    agent_auth: AgentAuthMaterial | None,
    secret_values: set[str] | frozenset[str],
    *,
    visibility: str = "hidden",
    isolation_fallback: IsolationFallback = "none",
    allow_missing_landlock_signal: bool = False,
    extra_environment: Mapping[str, str] | None = None,
    llm: Mapping[str, Any] | None = None,
    credentials_expire_at: float | None = None,
    withheld_values: frozenset[str] = frozenset(),
) -> bytes:
    """The launch spec ``rsi-agent`` reads from the Judge's private tmpfs.

    ``oneshot`` is the run's Agent for ``exec`` (None: no model access);
    ``llm`` the Judge LLM for ``llm`` (None: no judge_model).

    ``hidden`` lists what a call confined by Landlock (a Judge that is not
    root) may never read: rsi-agent's state, each login file's place, the
    Judge's outputs and, for hidden visibility, the tests. The trajectory and
    /tmp, where the Judge stages evidence, stay readable.

    ``credentials_expire_at`` is when the login files' access token expires
    (epoch seconds): ``rsi-agent exec`` refuses a call that would outlive it.

    ``withheld_values`` (refresh tokens) stay out of ``redact``: no Judge holds
    one, and the launch spec is readable inside the Judge.
    """
    if isolation_fallback not in ("none", "same-user"):
        raise SetupError("the Judge isolation fallback must be none or same-user")
    if type(allow_missing_landlock_signal) is not bool:
        raise SetupError("allow_missing_landlock_signal must be a boolean")
    if oneshot is None:
        oneshot = {
            "agent": None,
            "model": None,
            "reasoning_effort": None,
            "command": None,
            "schema_option": None,
            "output": None,
            "unsupported": _NO_MODEL_ACCESS,
            "environment": {},
        }
    # The command becomes the Agent's argv, which /proc shows every process
    # of the Judge's uid (a Landlock route below ABI 6 included): credentials
    # travel only in the environment or the login files.
    for template in (oneshot["command"], oneshot["schema_option"]):
        if any(value in str(template or "") for value in secret_values):
            raise SetupError("the Agent's one-shot command must not carry a credential")
    credentials = []
    protected = [str(path) for path in _PROTECTED]
    hidden = [str(JUDGE_AGENT_STATE_ROOT), "/logs/verifier"]
    if visibility != "public":
        hidden.append("/tests")
    for mount in () if agent_auth is None else agent_auth.mounts:
        target = mount.tmpfs.target
        try:
            relative = target.relative_to(JUDGE_AGENT_HOME)
        except ValueError:
            raise SetupError(
                "Agent login files must sit below the Agent's home to reach rsi-agent"
            ) from None
        protected.append(str(target))
        hidden.append(str(target))
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
        "environment": {**oneshot["environment"], **(extra_environment or {})},
        "credentials": credentials,
        "protected": protected,
        "hidden": hidden,
        "isolation_fallback": isolation_fallback,
        "allow_missing_landlock_signal": allow_missing_landlock_signal,
        "redact": launch_redaction(secret_values, withheld_values),
    }
    if llm is not None:
        spec["llm"] = dict(llm)
    if credentials_expire_at is not None:
        spec["credentials_expire_at"] = credentials_expire_at
    return json.dumps(spec, sort_keys=True, allow_nan=False).encode()


def launch_redaction(
    secret_values: frozenset[str] | set[str], withheld: frozenset[str]
) -> list[str]:
    """What rsi-agent redacts: every secret a Judge can hold, no refresh token."""
    return sorted(
        value
        for value in secret_values
        if value not in withheld and value != WITHHELD_REFRESH_TOKEN
    )


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
    "LAUNCH_SPEC_VERSION",
    "IsolationFallback",
    "JudgeEgress",
    "JUDGE_AGENT_HOME",
    "JUDGE_AGENT_MODEL_ENV",
    "JUDGE_AGENT_NAME_ENV",
    "JUDGE_AGENT_REASONING_EFFORT_ENV",
    "JudgeModelAccess",
    "build_launch_spec",
    "judge_needs_rsi_agent",
    "launch_redaction",
    "resolve_judge_model_access",
]
