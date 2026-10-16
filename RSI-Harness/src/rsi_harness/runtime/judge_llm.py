"""The Judge LLM: one plain model call a task may grant its Judge.

A task opts in with ``[metadata.rsi_harness.verifier] judge_model``. Each
Judge then calls that model through ``rsi-agent llm``, independent of the
Work Agent, with a credential the run operator supplies:

- ``api``: an OpenAI API key; ``rsi-agent`` posts to ``{base}/v1/responses``
  itself (through a proxy when one is given);
- ``codex-login``: a Codex login made with a ChatGPT plan; ``rsi-agent`` goes
  through the pinned Codex CLI (``codex app-server`` with every tool off).
  The Judge gets the access token only: its copy's refresh token is a
  placeholder, so nothing in a Judge can rotate (and so spend) the
  operator's refresh token. The Harness keeps the login fresh on the host
  instead: before each round Codex itself refreshes it in place when its
  access token would not outlive the round (judge_login_refresh.py).

With no credential given, a Codex Work Agent that reaches OpenAI itself (a
ChatGPT login, or an API key on OpenAI's own API) lends its credential, whatever
model Work uses. A lent ChatGPT login is the one Work's Agent keeps
refreshed: each Judge round gets Work's current copy (judge_model_access.py),
as the operator's is. A task that declares ``judge_model``
without any usable credential fails here, before the Agent starts, and so
does an operator credential for a Judge that starts from a snapshot of Work
(programs there are the Agent's), and an operator's Codex login that is the
Work Agent's own (its refresh would log Work out). The credential reaches only
``rsi-agent``'s launch spec; the Judge reaches only the endpoints this module
names, pinned like the Agent's.
"""

from __future__ import annotations

import functools
import json
import os
import pwd
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from rsi_harness.errors import SetupError
from rsi_harness.judge_agent import JUDGE_CODEX_BINARY, rsi_agent
from rsi_harness.models import JudgeLLMAuth, JudgeLLMRequest
from rsi_harness.runtime.judge_login_refresh import OperatorCodexLogin
from rsi_harness.runtime.local_auth import (
    WITHHELD_REFRESH_TOKEN,
    AgentAuthMaterial,
    codex_login_owner,
    login_secret_values,
    read_codex_login,
    withheld_values,
    withhold_refresh_token,
)
from rsi_harness.runtime.network import PinnedEndpoint, provider_bypasses_proxy
from rsi_loop.harness.config import RSILoopConfig

DEFAULT_API_BASE_URL = "https://api.openai.com"
DEFAULT_API_KEY_ENV = "RSI_JUDGE_LLM_API_KEY"
# What the Codex CLI reaches with a ChatGPT login: the Codex backend and its
# account routes on chatgpt.com. Not auth.openai.com: the Judge never
# refreshes the login.
CODEX_LOGIN_ENDPOINTS = ("https://chatgpt.com",)
_MAX_KEY_BYTES = 64 * 1024
JUDGE_MODEL_ENV = "RSI_JUDGE_MODEL"
JUDGE_MODEL_EFFORT_ENV = "RSI_JUDGE_MODEL_EFFORT"


@dataclass(frozen=True, slots=True, repr=False)
class JudgeLLMAccess:
    """The resolved Judge LLM: pinned endpoints and ``rsi-agent``'s settings.

    ``spec`` is the ``llm`` section of the launch spec, credential included;
    only ``rsi-agent`` reads it. ``reused`` says the Work Agent lent it.
    ``expires_at`` is when an operator's Codex login's access token expires
    (epoch seconds) at the run's start, when known; ``operator_login`` is
    that login, which the Harness refreshes before a round would outlive it.
    A login Work lends is renewed from Work each round, so it has neither.
    ``withheld_values`` are the login's refresh tokens, in ``secret_values``
    for the host's redaction but never sent to a Judge.
    """

    kind: JudgeLLMAuth
    model: str
    effort: str
    endpoints: tuple[PinnedEndpoint, ...]
    spec: Mapping[str, Any] = field(default_factory=dict)
    secret_values: frozenset[str] = frozenset()
    reused: bool = False
    expires_at: float | None = None
    operator_login: OperatorCodexLogin | None = None
    withheld_values: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise SetupError("the Judge LLM requires a pinned endpoint")

    def __repr__(self) -> str:
        hosts = ", ".join(endpoint.hostname for endpoint in self.endpoints)
        return (
            f"JudgeLLMAccess(kind={self.kind.value!r}, model={self.model!r}, "
            f"endpoints=[{hosts}], <redacted>)"
        )

    @property
    def needs_codex_cli(self) -> bool:
        return self.kind is JudgeLLMAuth.CODEX_LOGIN

    @property
    def environment(self) -> tuple[tuple[str, str], ...]:
        """What the Judge procedure sees: the model and effort, for records."""
        return ((JUDGE_MODEL_ENV, self.model), (JUDGE_MODEL_EFFORT_ENV, self.effort))


def resolve_judge_llm(
    definition: Any,
    request: JudgeLLMRequest | None,
    config: RSILoopConfig,
    agent_auth: AgentAuthMaterial | None,
    *,
    pin: Callable[[tuple[str, ...]], tuple[PinnedEndpoint, ...]],
    environ: Mapping[str, str] | None = None,
    process_uid: int | None = None,
    effective_uid: int | None = None,
    user_lookup: Callable[[int], Any] = pwd.getpwuid,
) -> JudgeLLMAccess | None:
    """Resolve the task's Judge LLM credential, or fail before the run starts."""

    model = definition.verifier.judge_model
    if model is None:
        return None
    effort = definition.verifier.judge_model_effort
    environment = os.environ if environ is None else environ
    reused = request is None
    if request is not None:
        credential = _operator_credential(
            request,
            model,
            config,
            environ=environment,
            process_uid=process_uid,
            effective_uid=effective_uid,
            user_lookup=user_lookup,
        )
    else:
        credential = _work_credential(definition, config, agent_auth)
    if credential is None:
        raise SetupError(
            f"the task's Judge LLM is {model!r}, but no credential for it was "
            "given: pass --judge-llm-auth api (API key in "
            f"${DEFAULT_API_KEY_ENV} or --judge-llm-api-key-file) or "
            "--judge-llm-auth codex-login (a codex Work Agent lends its own "
            "only with a ChatGPT login or an API key on OpenAI's own API)"
        )
    kind, settings, secrets = credential
    verifier = definition.verifier
    if not reused and verifier.visibility != "public" and not verifier.clean_image:
        raise SetupError(snapshot_judge_refusal(model))
    expires_at = None
    withheld = frozenset()
    if kind is JudgeLLMAuth.API:
        url = _responses_url(settings["base_url"])
        proxy = settings.get("proxy")
        targets = (proxy,) if proxy else (settings["base_url"],)
        spec: dict[str, Any] = {
            "kind": kind.value,
            "model": model,
            "effort": effort,
            "url": url,
            "api_key": settings["api_key"],
            "proxy": proxy,
        }
    else:
        proxy = settings.get("proxy")
        targets = (proxy,) if proxy else CODEX_LOGIN_ENDPOINTS
        auth_json = _without_refresh_token(settings["auth_json"])
        withheld = withheld_values("codex", settings["auth_json"].encode("utf-8"))
        shared = agent_auth is not None and withheld & agent_auth.secret_values
        if not reused and shared:
            raise SetupError(shared_login_refusal(model))
        expires_at = None if reused else rsi_agent.access_token_expiry(auth_json)
        spec = {
            "kind": kind.value,
            "model": model,
            "effort": effort,
            "codex": str(JUDGE_CODEX_BINARY),
            "auth_json": auth_json,
            "environment": (
                {"https_proxy": proxy, "HTTPS_PROXY": proxy} if proxy else {}
            ),
            "config": [],
        }
    for url_value in (settings.get("base_url"), proxy):
        secrets |= _url_secrets(url_value)
    return JudgeLLMAccess(
        kind=kind,
        model=model,
        effort=effort,
        endpoints=pin(tuple(targets)),
        spec=spec,
        secret_values=frozenset(value for value in secrets if value),
        reused=reused,
        expires_at=expires_at,
        operator_login=settings.get("operator_login"),
        withheld_values=withheld,
    )


def snapshot_judge_refusal(model: str) -> str:
    """Why an operator's Judge LLM credential needs a clean Judge image."""
    return (
        f"the Judge LLM {model!r} would get the operator's credential in a Judge "
        "that starts from a snapshot of Work (clean_image = false, or a WORKDIR "
        "of /), where the Agent can change the programs that read it; use the "
        "clean Judge image, or a codex Work Agent on this model to lend its own"
    )


def shared_login_refusal(model: str) -> str:
    """Why the operator's Codex login may not be the Work Agent's own."""
    return (
        f"the Judge LLM {model!r} would use the Work Agent's own Codex login: "
        "the Harness refreshes an operator's login on the host, which spends "
        "the refresh token Work's private copy holds and logs Work out; leave "
        "out --judge-llm-auth so Work lends its login, or use a separate login "
        "(--judge-llm-codex-home)"
    )


def _without_refresh_token(content: str) -> str:
    return judge_codex_login(content.encode("utf-8"))


def judge_codex_login(content: bytes) -> str:
    """The Judge's copy of a ChatGPT login: the refresh token withheld."""
    copy = withhold_refresh_token("codex", content).decode("utf-8")
    tokens = json.loads(copy).get("tokens")
    if not isinstance(tokens, dict) or tokens.get("refresh_token") != (
        WITHHELD_REFRESH_TOKEN
    ):
        raise SetupError("the Codex login's refresh token could not be withheld")
    return copy


def _operator_credential(
    request: JudgeLLMRequest,
    model: str,
    config: RSILoopConfig,
    *,
    environ: Mapping[str, str],
    process_uid: int | None,
    effective_uid: int | None,
    user_lookup: Callable[[int], Any],
) -> tuple[JudgeLLMAuth, dict[str, Any], frozenset[str]]:
    # The run's proxy settings, as for the Agent, unless one is given.
    proxies = _config_proxies(config)
    proxy = _checked_url(request.proxy, "proxy") if request.proxy else None
    if request.auth is JudgeLLMAuth.API:
        if request.api_key_file is not None:
            key = _read_key_file(request.api_key_file)
        else:
            name = request.api_key_env or DEFAULT_API_KEY_ENV
            key = (environ.get(name) or "").strip()
            if not key:
                raise SetupError(
                    f"the Judge LLM {model!r} needs an API key: ${name} is not set"
                )
        base_url = _checked_url(request.base_url or DEFAULT_API_BASE_URL, "base URL")
        if proxy is None:
            proxy = _work_proxy(base_url, proxies)
        return (
            JudgeLLMAuth.API,
            {"api_key": key, "base_url": base_url, "proxy": proxy},
            frozenset({key}),
        )
    try:
        content = read_codex_login(
            request.codex_home,
            environ=environ,
            process_uid=process_uid,
            effective_uid=effective_uid,
            user_lookup=user_lookup,
        )
    except SetupError as error:
        where = (
            request.codex_home
            if request.codex_home is not None
            else environ.get("CODEX_HOME") or "~/.codex"
        )
        raise SetupError(
            f"the Judge LLM {model!r} needs a Codex login: auth.json in {where} "
            f"(--judge-llm-codex-home, $CODEX_HOME or ~/.codex): {error}"
        ) from None
    login = _parse_login(content)
    if not _is_chatgpt_login(login):
        raise SetupError(
            "the Codex login is not a ChatGPT login; for an API key use "
            "--judge-llm-auth api"
        )
    if proxy is None:
        proxy = _work_proxy(CODEX_LOGIN_ENDPOINTS[0], proxies)
    # The same login, re-read before each round with the same checks.
    lookup = {
        name: environ[name] for name in ("CODEX_HOME", "SUDO_UID") if name in environ
    }
    owner = dict(
        environ=lookup,
        process_uid=process_uid,
        effective_uid=effective_uid,
        user_lookup=user_lookup,
    )
    directory, uid, gid = codex_login_owner(request.codex_home, **owner)
    operator_login = OperatorCodexLogin(
        codex_home=directory,
        uid=uid,
        gid=gid,
        read=functools.partial(read_codex_login, request.codex_home, **owner),
        proxy=proxy,
    )
    return (
        JudgeLLMAuth.CODEX_LOGIN,
        {
            "auth_json": content.decode("utf-8"),
            "proxy": proxy,
            "operator_login": operator_login,
        },
        login_secret_values(login),
    )


def _config_proxies(config: RSILoopConfig) -> dict[str, str]:
    environment = {}
    for value, names in (
        (config.http_proxy, ("http_proxy", "HTTP_PROXY")),
        (config.https_proxy, ("https_proxy", "HTTPS_PROXY")),
        (config.no_proxy, ("no_proxy", "NO_PROXY")),
    ):
        if value:
            environment.update(dict.fromkeys(names, value))
    return environment


def _work_credential(
    definition: Any,
    config: RSILoopConfig,
    agent_auth: AgentAuthMaterial | None,
) -> tuple[JudgeLLMAuth, dict[str, Any], frozenset[str]] | None:
    """The Codex Work Agent's own credential, when it reaches OpenAI itself.

    Any model on that account will do, so Work's model does not matter. A
    Work Agent pointed at another base URL (a proxy or another provider) may
    not serve the Judge model, so it lends nothing.
    """
    from rsi_harness.integrations.rsi_loop import rsi_loop_agent_environment
    from rsi_loop.harness.agent.codex import CodexAgent

    if definition.agent.name != CodexAgent.name:
        return None
    agent = CodexAgent(config)
    work_model = definition.agent.model or config.agent_model or agent.default_model
    environment = rsi_loop_agent_environment(config, agent, work_model)
    if agent_auth is not None:
        content = b"".join(
            auth_file.content
            for mount in agent_auth.mounts
            for auth_file in mount.files
        )
        login = _parse_login(content)
        if _is_chatgpt_login(login):
            return (
                JudgeLLMAuth.CODEX_LOGIN,
                {
                    "auth_json": content.decode("utf-8"),
                    "proxy": _work_proxy(CODEX_LOGIN_ENDPOINTS[0], environment),
                },
                login_secret_values(login),
            )
        key = login.get("OPENAI_API_KEY")
        if isinstance(key, str) and key:
            return (
                JudgeLLMAuth.API,
                {
                    "api_key": key,
                    "base_url": DEFAULT_API_BASE_URL,
                    "proxy": _work_proxy(DEFAULT_API_BASE_URL, environment),
                },
                login_secret_values(login) | {key},
            )
        return None
    key = environment.get(agent.api_key_env)
    if not key:
        return None
    configured = environment.get(agent.api_base_env or "")
    if (
        configured
        and urlsplit(configured).hostname != urlsplit(DEFAULT_API_BASE_URL).hostname
    ):
        return None
    return (
        JudgeLLMAuth.API,
        {
            "api_key": key,
            "base_url": DEFAULT_API_BASE_URL,
            "proxy": _work_proxy(DEFAULT_API_BASE_URL, environment),
        },
        frozenset({key}),
    )


def _work_proxy(provider: str, environment: Mapping[str, str]) -> str | None:
    """The Work Agent's proxy for ``provider``, as Work reaches it."""
    scheme = urlsplit(provider).scheme
    keys = (
        ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
        if scheme == "https"
        else ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY")
    )
    proxy = next((environment[key] for key in keys if environment.get(key)), None)
    if proxy is None or provider_bypasses_proxy(provider, environment):
        return None
    return _checked_url(proxy, "proxy")


def _parse_login(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SetupError("the Codex login must contain valid JSON") from None
    if not isinstance(value, dict):
        raise SetupError("the Codex login must contain a JSON object")
    return value


def _is_chatgpt_login(login: Mapping[str, Any]) -> bool:
    tokens = login.get("tokens")
    return (
        login.get("auth_mode") in (None, "chatgpt")
        and isinstance(tokens, dict)
        and isinstance(tokens.get("access_token"), str)
        and isinstance(tokens.get("refresh_token"), str)
    )


def _read_key_file(path: Path) -> str:
    try:
        metadata = os.stat(path)
        if not stat.S_ISREG(metadata.st_mode):
            raise SetupError(f"the Judge LLM API key file {path} is not a file")
        with open(path, "rb") as stream:
            data = stream.read(_MAX_KEY_BYTES + 1)
    except OSError as error:
        raise SetupError(
            f"cannot read the Judge LLM API key file {path}: {error.strerror}"
        ) from None
    if len(data) > _MAX_KEY_BYTES:
        raise SetupError("the Judge LLM API key file is too large")
    try:
        key = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise SetupError("the Judge LLM API key file is not text") from None
    if not key or any(character.isspace() for character in key):
        raise SetupError("the Judge LLM API key file must hold one key")
    return key


def _checked_url(value: str, label: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise SetupError(f"the Judge LLM {label} is not a valid URL") from None
    del port
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
    ):
        raise SetupError(f"the Judge LLM {label} must be an http(s) URL")
    return value.rstrip("/")


def _responses_url(base_url: str) -> str:
    """``{base}/v1/responses``; a base that already ends in /v1 keeps it once."""
    base = base_url.rstrip("/")
    if urlsplit(base).path.endswith("/v1"):
        return base + "/responses"
    return base + "/v1/responses"


def _url_secrets(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    parsed = urlsplit(value)
    found = set()
    for credential in (parsed.username, parsed.password):
        if credential:
            found.update({value, credential, unquote(credential)})
    return frozenset(found)


__all__ = [
    "CODEX_LOGIN_ENDPOINTS",
    "WITHHELD_REFRESH_TOKEN",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_API_KEY_ENV",
    "JUDGE_MODEL_EFFORT_ENV",
    "JUDGE_MODEL_ENV",
    "JudgeLLMAccess",
    "judge_codex_login",
    "resolve_judge_llm",
    "shared_login_refusal",
    "snapshot_judge_refusal",
]
