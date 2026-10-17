"""The Judge LLM: credentials, the reuse rule, preflight, pins and the Judge."""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.judge_agent import (
    JUDGE_AGENT_TMPFS,
    JUDGE_CODEX_BINARY,
    JUDGE_CODEX_CACHE,
    JUDGE_CODEX_TARGET,
)
from rsi_harness.models import (
    ContainerMount,
    ContainerSpec,
    ContainerTmpfs,
    JudgeLLMAuth,
    JudgeLLMRequest,
    RunRequest,
    RunResult,
    RunStatus,
)
from rsi_harness.runtime.judge import require_model_access
from rsi_harness.runtime.judge_llm import (
    CODEX_LOGIN_ENDPOINTS,
    DEFAULT_API_KEY_ENV,
    WITHHELD_REFRESH_TOKEN,
    JudgeLLMAccess,
    resolve_judge_llm,
)
from rsi_harness.runtime.judge_model_access import (
    JudgeModelAccess,
    resolve_judge_model_access,
)
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
    login_secret_values,
)
from rsi_harness.runtime.network import PinnedEndpoint
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan
from tests.fakes import FakeDockerClient
from tests.runtime.test_docker import make_runtime

API_KEY = "sk-synthetic-judge-llm-8a1f"
WORK_KEY = "sk-synthetic-work-agent-31b0"
ACCESS_TOKEN = "SYNTHETIC-ACCESS-9f0d"
REFRESH_TOKEN = "SYNTHETIC-REFRESH-2c77"
CHATGPT_LOGIN = {
    "auth_mode": "chatgpt",
    "OPENAI_API_KEY": None,
    "tokens": {
        "id_token": "h.e30.s",
        "access_token": ACCESS_TOKEN,
        "refresh_token": REFRESH_TOKEN,
        "account_id": "acc-1",
    },
}


# The Judge's copy of a login: the refresh token withheld.
JUDGE_LOGIN = {
    **CHATGPT_LOGIN,
    "tokens": {**CHATGPT_LOGIN["tokens"], "refresh_token": WITHHELD_REFRESH_TOKEN},
}


def jwt(**claims) -> str:
    import base64

    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return "h." + payload.decode() + ".s"


def pin(endpoints: tuple[str, ...]) -> tuple[PinnedEndpoint, ...]:
    """Pin without DNS: every name to a documentation address."""
    pinned = []
    for url in endpoints:
        parsed = urlsplit(url)
        pinned.append(
            PinnedEndpoint(
                hostname=parsed.hostname,
                port=parsed.port or (443 if parsed.scheme == "https" else 80),
                addresses=(ipaddress.ip_address("198.51.100.20"),),
            )
        )
    return tuple(pinned)


def definition(
    *,
    judge_model: str | None = "gpt-judge",
    effort: str = "high",
    agent: str = "codex",
    model: str | None = "gpt-judge",
    model_access: bool = False,
    clean_image: bool = True,
):
    return SimpleNamespace(
        verifier=SimpleNamespace(
            judge_model=judge_model,
            judge_model_effort=effort if judge_model else None,
            model_access=model_access,
            visibility="hidden",
            clean_image=clean_image,
            network=SimpleNamespace(mode="no-network"),
        ),
        agent=SimpleNamespace(
            name=agent,
            model=model,
            reasoning_effort=None,
            network=SimpleNamespace(mode="no-network"),
        ),
    )


def resolve(request=None, config=None, agent_auth=None, task=None, **options):
    return resolve_judge_llm(
        task or definition(),
        request,
        config or RSILoopConfig(),
        agent_auth,
        pin=pin,
        **{"environ": {}, **options},
    )


def codex_home(tmp_path: Path, login: dict, mode: int = 0o600) -> Path:
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps(login))
    (home / "auth.json").chmod(mode)
    return home


def login_material(login: dict) -> AgentAuthMaterial:
    return AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options="rw,nosuid,nodev,noexec,mode=0700",
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=json.dumps(login).encode(),
                    ),
                ),
            ),
        ),
        secret_values=frozenset({ACCESS_TOKEN, REFRESH_TOKEN}),
    )


# -- operator credentials ----------------------------------------------------------


def test_no_judge_model_means_no_judge_llm_whatever_the_operator_gives() -> None:
    request = JudgeLLMRequest(auth=JudgeLLMAuth.API)
    assert resolve(request, task=definition(judge_model=None)) is None


def test_api_key_from_the_default_variable() -> None:
    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.API),
        environ={DEFAULT_API_KEY_ENV: API_KEY},
    )

    assert access.kind is JudgeLLMAuth.API and not access.reused
    assert access.spec == {
        "kind": "api",
        "model": "gpt-judge",
        "effort": "high",
        "url": "https://api.openai.com/v1/responses",
        "api_key": API_KEY,
        "proxy": None,
    }
    assert [(e.hostname, e.port) for e in access.endpoints] == [("api.openai.com", 443)]
    assert API_KEY in access.secret_values
    assert API_KEY not in repr(access)
    assert dict(access.environment) == {
        "RSI_JUDGE_MODEL": "gpt-judge",
        "RSI_JUDGE_MODEL_EFFORT": "high",
    }


def test_api_key_from_a_named_variable_or_a_file(tmp_path: Path) -> None:
    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.API, api_key_env="MY_KEY"),
        environ={"MY_KEY": API_KEY, DEFAULT_API_KEY_ENV: "other"},
    )
    assert access.spec["api_key"] == API_KEY

    key_file = tmp_path / "key"
    key_file.write_text(API_KEY + "\n")
    access = resolve(JudgeLLMRequest(auth=JudgeLLMAuth.API, api_key_file=key_file))
    assert access.spec["api_key"] == API_KEY


@pytest.mark.parametrize(
    ("request_options", "environ", "message"),
    [
        ({}, {}, rf"needs an API key: \${DEFAULT_API_KEY_ENV} is not set"),
        ({"api_key_env": "MY_KEY"}, {"MY_KEY": " "}, r"\$MY_KEY is not set"),
        ({"base_url": "ftp://models.example"}, None, "base URL must be an http"),
        ({"base_url": "https://models.example/?q=1"}, None, "base URL must be"),
        ({"proxy": "socks5://proxy.example:1080"}, None, "proxy must be an http"),
    ],
)
def test_api_credential_errors_name_the_problem(
    request_options, environ, message
) -> None:
    with pytest.raises(SetupError, match=message):
        resolve(
            JudgeLLMRequest(auth=JudgeLLMAuth.API, **request_options),
            environ={DEFAULT_API_KEY_ENV: API_KEY} if environ is None else environ,
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "cannot read"), ("", "must hold one key"), ("a b", "must hold one key")],
)
def test_api_key_file_must_hold_one_key(tmp_path: Path, content, message) -> None:
    key_file = tmp_path / "key"
    if content is not None:
        key_file.write_text(content)
    with pytest.raises(SetupError, match=message):
        resolve(JudgeLLMRequest(auth=JudgeLLMAuth.API, api_key_file=key_file))


@pytest.mark.parametrize(
    ("base_url", "url", "endpoint"),
    [
        ("https://models.example", "https://models.example/v1/responses", 443),
        ("https://models.example/v1/", "https://models.example/v1/responses", 443),
        (
            "http://10.0.0.5:8080/openai",
            "http://10.0.0.5:8080/openai/v1/responses",
            8080,
        ),
    ],
)
def test_api_base_url_takes_one_v1(base_url, url, endpoint) -> None:
    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.API, base_url=base_url),
        environ={DEFAULT_API_KEY_ENV: API_KEY},
    )

    assert access.spec["url"] == url
    assert access.endpoints[0].port == endpoint


def test_api_proxy_is_the_only_endpoint_and_its_credentials_are_secret() -> None:
    proxy = "http://user:proxy-pass-55@proxy.example:3128"
    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.API, proxy=proxy),
        environ={DEFAULT_API_KEY_ENV: API_KEY},
    )

    assert access.spec["proxy"] == proxy
    assert [(e.hostname, e.port) for e in access.endpoints] == [("proxy.example", 3128)]
    assert {"proxy-pass-55", proxy} <= access.secret_values


def test_api_defaults_to_the_runs_proxy_unless_no_proxy_bypasses_it() -> None:
    config = RSILoopConfig(https_proxy="http://corp-proxy.example:8080")
    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.API),
        config=config,
        environ={DEFAULT_API_KEY_ENV: API_KEY},
    )
    assert access.spec["proxy"] == "http://corp-proxy.example:8080"
    assert access.endpoints[0].hostname == "corp-proxy.example"

    bypass = RSILoopConfig(
        https_proxy="http://corp-proxy.example:8080", no_proxy=".openai.com"
    )
    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.API),
        config=bypass,
        environ={DEFAULT_API_KEY_ENV: API_KEY},
    )
    assert access.spec["proxy"] is None
    assert access.endpoints[0].hostname == "api.openai.com"


def test_codex_login_from_a_codex_home(tmp_path: Path) -> None:
    home = codex_home(tmp_path, CHATGPT_LOGIN)

    access = resolve(JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home))

    assert access.kind is JudgeLLMAuth.CODEX_LOGIN and access.needs_codex_cli
    # The Judge never holds the refresh token, so it cannot spend it.
    assert json.loads(access.spec["auth_json"]) == JUDGE_LOGIN
    assert REFRESH_TOKEN not in json.dumps(dict(access.spec))
    assert access.spec["codex"] == str(JUDGE_CODEX_BINARY)
    assert access.spec["environment"] == {}
    # Not auth.openai.com: nothing in the Judge refreshes the login.
    assert [e.hostname for e in access.endpoints] == ["chatgpt.com"]
    assert tuple(f"https://{e.hostname}" for e in access.endpoints) == (
        CODEX_LOGIN_ENDPOINTS
    )
    assert {ACCESS_TOKEN, REFRESH_TOKEN} <= access.secret_values
    assert REFRESH_TOKEN not in repr(access)


def test_codex_login_defaults_to_codex_home_and_takes_a_proxy(tmp_path: Path) -> None:
    home = codex_home(tmp_path, CHATGPT_LOGIN)

    access = resolve(
        JudgeLLMRequest(
            auth=JudgeLLMAuth.CODEX_LOGIN, proxy="http://proxy.example:3128"
        ),
        environ={"CODEX_HOME": str(home)},
    )

    assert access.spec["environment"] == {
        "https_proxy": "http://proxy.example:3128",
        "HTTPS_PROXY": "http://proxy.example:3128",
    }
    assert [e.hostname for e in access.endpoints] == ["proxy.example"]


def test_codex_login_defaults_to_the_users_codex_directory(tmp_path: Path) -> None:
    user = SimpleNamespace(pw_uid=os.getuid(), pw_gid=4242, pw_dir=str(tmp_path))
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "auth.json").write_text(json.dumps(CHATGPT_LOGIN))
    (tmp_path / ".codex" / "auth.json").chmod(0o600)

    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN),
        user_lookup=lambda uid: user,
    )

    assert json.loads(access.spec["auth_json"]) == JUDGE_LOGIN
    # Refreshed later as that user, in that directory.
    operator = access.operator_login
    assert operator.codex_home == tmp_path / ".codex"
    assert (operator.uid, operator.gid) == (os.getuid(), 4242)


def test_a_codex_login_close_to_expiry_is_accepted_and_kept_fresh(
    tmp_path: Path,
) -> None:
    # No run-start expiry check: the Harness refreshes the login before a
    # Judge round would outlive it.
    soon = time.time() + 60
    login = {
        **CHATGPT_LOGIN,
        "tokens": {**CHATGPT_LOGIN["tokens"], "access_token": jwt(exp=soon)},
    }
    home = codex_home(tmp_path, login)

    access = resolve(JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home))

    assert access.expires_at == pytest.approx(soon)
    operator = access.operator_login
    assert operator is not None
    assert operator.codex_home == home
    assert (operator.uid, operator.gid) == (os.getuid(), os.getgid())
    assert json.loads(operator.read()) == login
    assert REFRESH_TOKEN not in repr(operator)


@pytest.mark.parametrize("shared", [True, False])
def test_the_operators_login_may_not_be_the_work_agents_own(
    tmp_path: Path, shared: bool
) -> None:
    """Refreshing it on the host would spend the refresh token Work's private
    copy holds: Work's next refresh would fail and log it out."""
    home = codex_home(tmp_path, CHATGPT_LOGIN)
    work = (
        CHATGPT_LOGIN
        if shared
        else {
            **CHATGPT_LOGIN,
            "tokens": {
                **CHATGPT_LOGIN["tokens"],
                "refresh_token": "SYNTHETIC-OTHER-REFRESH-5d1e",
            },
        }
    )
    request = JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home)
    work_auth = dataclasses.replace(
        login_material(work), secret_values=login_secret_values(work)
    )

    if shared:
        with pytest.raises(SetupError) as error:
            resolve(request, agent_auth=work_auth)
        assert "Work Agent's own Codex login" in str(error.value)
        assert "leave out --judge-llm-auth" in str(error.value)
        assert REFRESH_TOKEN not in str(error.value)
    else:
        access = resolve(request, agent_auth=work_auth)
        assert access.operator_login is not None
        assert access.withheld_values == {REFRESH_TOKEN}


def test_a_missing_codex_login_names_the_judge_llm_and_where(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(SetupError) as error:
        resolve(JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=empty))
    assert "Judge LLM 'gpt-judge'" in str(error.value)
    assert str(empty) in str(error.value)
    assert "--judge-llm-codex-home" in str(error.value)


@pytest.mark.parametrize(
    ("visibility", "reuse", "refused"),
    [("hidden", False, True), ("hidden", True, False), ("public", False, False)],
)
def test_a_snapshot_judge_gets_no_operator_credential(
    visibility: str, reuse: bool, refused: bool, monkeypatch
) -> None:
    task = definition(clean_image=False)
    task.verifier.visibility = visibility
    request = None if reuse else JudgeLLMRequest(auth=JudgeLLMAuth.API)
    config = RSILoopConfig(agent_api_key=WORK_KEY)

    if refused:
        with pytest.raises(SetupError, match="snapshot of Work"):
            resolve(
                request,
                config=config,
                task=task,
                environ={DEFAULT_API_KEY_ENV: API_KEY},
            )
    else:
        access = resolve(
            request, config=config, task=task, environ={DEFAULT_API_KEY_ENV: API_KEY}
        )
        assert access.reused is reuse


@pytest.mark.parametrize(
    ("login", "mode", "message"),
    [
        ({"OPENAI_API_KEY": "sk-x"}, 0o600, "not a ChatGPT login"),
        ({**CHATGPT_LOGIN, "auth_mode": "apikey"}, 0o600, "not a ChatGPT login"),
        (CHATGPT_LOGIN, 0o640, "deny group and other"),
    ],
)
def test_codex_login_must_be_a_private_chatgpt_login(
    tmp_path: Path, login, mode, message
) -> None:
    home = codex_home(tmp_path, login, mode)
    with pytest.raises(SetupError, match=message):
        resolve(JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"auth": "api", "codex_home": "/tmp/x"}, "only to a codex-login"),
        ({"auth": "codex-login", "api_key_env": "K"}, "only to an api"),
        ({"auth": "codex-login", "base_url": "https://x"}, "only to an api"),
        (
            {"auth": "api", "api_key_env": "K", "api_key_file": "/tmp/k"},
            "not both",
        ),
    ],
)
def test_request_rejects_options_of_the_other_kind(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        JudgeLLMRequest(**options)


# -- the reuse rule -------------------------------------------------------------------


def test_codex_work_agent_lends_its_api_key_whatever_its_model() -> None:
    config = RSILoopConfig(agent_api_key=WORK_KEY, agent_model="gpt-work")

    access = resolve(None, config=config, task=definition(model="gpt-other"))

    assert access.reused and access.kind is JudgeLLMAuth.API
    assert access.spec["api_key"] == WORK_KEY
    assert access.spec["url"] == "https://api.openai.com/v1/responses"
    assert WORK_KEY in access.secret_values


def test_codex_work_agent_on_openais_own_base_url_lends_its_api_key() -> None:
    config = RSILoopConfig(
        agent_api_key=WORK_KEY, agent_api_base_url="https://api.openai.com/v1"
    )

    access = resolve(None, config=config)

    assert access.reused
    assert access.spec["url"] == "https://api.openai.com/v1/responses"


def test_codex_work_agent_on_another_base_url_lends_nothing() -> None:
    config = RSILoopConfig(
        agent_api_key=WORK_KEY,
        agent_api_base_url="http://198.51.100.9:8443/v1",
        http_proxy="http://work-proxy.example:3128",
        no_proxy="198.51.100.9",
    )

    with pytest.raises(SetupError) as error:
        resolve(None, config=config)
    assert "--judge-llm-auth" in str(error.value)
    assert WORK_KEY not in str(error.value)


def test_codex_work_agent_lends_its_proxy_and_default_url() -> None:
    config = RSILoopConfig(
        agent_api_key=WORK_KEY, https_proxy="http://work-proxy.example:3128"
    )

    access = resolve(None, config=config)

    assert access.spec["url"] == "https://api.openai.com/v1/responses"
    assert access.spec["proxy"] == "http://work-proxy.example:3128"


def test_codex_work_agent_lends_its_chatgpt_login() -> None:
    access = resolve(None, agent_auth=login_material(CHATGPT_LOGIN))

    assert access.reused and access.kind is JudgeLLMAuth.CODEX_LOGIN
    assert json.loads(access.spec["auth_json"]) == JUDGE_LOGIN
    assert {ACCESS_TOKEN, REFRESH_TOKEN} <= access.secret_values


def test_codex_work_agent_lends_an_api_key_login() -> None:
    access = resolve(None, agent_auth=login_material({"OPENAI_API_KEY": WORK_KEY}))

    assert access.kind is JudgeLLMAuth.API
    assert access.spec["api_key"] == WORK_KEY


@pytest.mark.parametrize(
    "task",
    [
        definition(agent="claude-code"),
    ],
)
def test_no_credential_fails_before_the_run_naming_the_model(task) -> None:
    config = RSILoopConfig(agent_api_key=WORK_KEY)
    with pytest.raises(SetupError) as error:
        resolve(None, config=config, task=task)
    assert "'gpt-judge'" in str(error.value)
    assert "--judge-llm-auth" in str(error.value)
    assert WORK_KEY not in str(error.value)


def test_codex_work_agent_lends_its_login_whatever_its_model() -> None:
    access = resolve(
        None, agent_auth=login_material(CHATGPT_LOGIN), task=definition(model=None)
    )
    assert access.reused and access.kind is JudgeLLMAuth.CODEX_LOGIN


def test_operator_credential_wins_over_reuse() -> None:
    access = resolve(
        JudgeLLMRequest(auth=JudgeLLMAuth.API),
        config=RSILoopConfig(agent_api_key=WORK_KEY),
        environ={DEFAULT_API_KEY_ENV: API_KEY},
    )
    assert not access.reused and access.spec["api_key"] == API_KEY


# -- with model access ----------------------------------------------------------------


def llm_access(kind: JudgeLLMAuth = JudgeLLMAuth.API) -> JudgeLLMAccess:
    spec = {"kind": kind.value, "model": "gpt-judge", "effort": "high"}
    return JudgeLLMAccess(
        kind=kind,
        model="gpt-judge",
        effort="high",
        endpoints=pin(("https://api.openai.com",)),
        spec={**spec, "url": "https://api.openai.com/v1/responses", "api_key": API_KEY},
        secret_values=frozenset({API_KEY}),
    )


def test_judge_model_alone_gives_rsi_agent_llm_but_not_exec() -> None:
    access = resolve_judge_model_access(
        definition(), RSILoopConfig(), None, (), llm_access()
    )

    launch = json.loads(access.launch)
    assert launch["command"] is None
    assert "model_access = true" in launch["unsupported"]
    assert launch["llm"]["api_key"] == API_KEY
    assert API_KEY in launch["redact"]
    assert access.auth is None and not access.needs_codex_cli
    assert dict(access.environment) == {
        "RSI_JUDGE_MODEL": "gpt-judge",
        "RSI_JUDGE_MODEL_EFFORT": "high",
    }
    assert access.endpoints == pin(("https://api.openai.com",))


def test_model_access_and_judge_model_share_one_spec_and_pin_set() -> None:
    work_endpoint = pin(("http://198.51.100.9:8443",))
    task = definition(model_access=True, model="gpt-test")
    config = RSILoopConfig(
        agent_api_key=WORK_KEY, agent_api_base_url="http://198.51.100.9:8443/v1"
    )
    llm = llm_access(JudgeLLMAuth.CODEX_LOGIN)

    access = resolve_judge_model_access(task, config, None, work_endpoint, llm)

    launch = json.loads(access.launch)
    assert launch["command"] and launch["llm"]["kind"] == "codex-login"
    assert {WORK_KEY, API_KEY} <= set(launch["redact"])
    assert access.endpoints == work_endpoint + llm.endpoints
    assert access.needs_codex_cli and access.codex_cli is None
    environment = dict(access.environment)
    assert environment["RSI_AGENT_NAME"] == "codex"
    assert environment["RSI_JUDGE_MODEL"] == "gpt-judge"
    # The same endpoint pinned twice is allowed once.
    again = resolve_judge_model_access(
        task,
        config,
        None,
        llm.endpoints,
        llm,
    )
    assert again.endpoints == llm.endpoints


def test_judge_llm_must_match_the_task() -> None:
    with pytest.raises(SetupError, match="differs from the task"):
        resolve_judge_model_access(definition(), RSILoopConfig(), None, (), None)
    with pytest.raises(SetupError, match="differs from the task"):
        resolve_judge_model_access(
            definition(judge_model=None), RSILoopConfig(), None, (), llm_access()
        )


def declare_judge_model(plan, model: str | None = "gpt-judge"):
    verifier = plan.task.verifier.model_copy(
        update={"judge_model": model, "judge_model_effort": "medium" if model else None}
    )
    return plan.model_copy(
        update={"task": plan.task.model_copy(update={"verifier": verifier})}
    )


def test_judge_rounds_require_the_resolved_access(tmp_path: Path) -> None:
    plan = declare_judge_model(make_run_plan(tmp_path))
    access = resolve_judge_model_access(
        definition(), RSILoopConfig(), None, (), llm_access()
    )

    assert require_model_access(plan, access) is access
    with pytest.raises(SetupError, match="none was resolved"):
        require_model_access(plan, None)
    with pytest.raises(SetupError, match="task without it"):
        require_model_access(declare_judge_model(make_run_plan(tmp_path), None), access)
    codex = JudgeModelAccess(
        endpoints=access.endpoints,
        environment=access.environment,
        auth=None,
        secret_values=access.secret_values,
        launch=access.launch,
        needs_codex_cli=True,
    )
    with pytest.raises(SetupError, match="Codex CLI"):
        require_model_access(plan, codex)


# -- the Codex CLI in the Judge ------------------------------------------------------


def codex_cache(tmp_path: Path) -> Path:
    directory = tmp_path / "engine-root" / JUDGE_CODEX_CACHE / "codex-0.0.0-x86_64"
    directory.mkdir(parents=True)
    (directory / "codex").write_text("#!/bin/sh\n")
    (directory / "codex").chmod(0o755)
    return directory


def test_docker_mounts_the_cached_codex_cli_read_only(tmp_path: Path) -> None:
    directory = codex_cache(tmp_path)
    judge = make_runtime(
        FakeDockerClient(), tmp_path, role="judge", judge_model_access=True
    )

    ref = judge.create(
        ContainerSpec(
            image="judge-image",
            mounts=(
                ContainerMount(
                    source=directory, target=JUDGE_CODEX_TARGET, read_only=True
                ),
            ),
            tmpfs=JUDGE_AGENT_TMPFS,
        )
    )
    assert ref.role == "judge"


@pytest.mark.parametrize(
    "change",
    ["writable", "no-access", "work", "elsewhere", "missing-binary"],
)
def test_docker_refuses_an_inexact_codex_cli_mount(tmp_path: Path, change) -> None:
    directory = codex_cache(tmp_path)
    options = {"role": "judge", "judge_model_access": True}
    read_only = True
    if change == "writable":
        read_only = False
    elif change == "no-access":
        options = {"role": "judge"}
    elif change == "work":
        options = {"role": "work"}
    elif change == "elsewhere":
        directory = tmp_path / "engine-root" / "other" / "codex-0.0.0-x86_64"
        directory.mkdir(parents=True)
        (directory / "codex").write_text("")
    else:
        (directory / "codex").unlink()
    runtime = make_runtime(FakeDockerClient(), tmp_path, **options)

    with pytest.raises(SetupError, match="Codex CLI mount"):
        runtime.create(
            ContainerSpec(
                image="image",
                mounts=(
                    ContainerMount(
                        source=directory, target=JUDGE_CODEX_TARGET, read_only=read_only
                    ),
                ),
            )
        )


# -- production preflight -------------------------------------------------------


def _production(tmp_path: Path, metadata: str, request_options: dict, **config):
    from rsi_harness.runtime.production import ProductionRuntimeServices
    from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    task_config = task / "task.toml"
    task_config.write_text(
        task_config.read_text() + "\n[metadata.rsi_harness.verifier]\n" + metadata
    )
    captured = []
    events = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request):
            del request
            return RunResult(run_id="judge-llm", status=RunStatus.COMPLETED)

    class NoDocker:
        def __getattr__(self, name):
            raise AssertionError(f"Docker was contacted before preflight: {name}")

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=NoDocker(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=WORK_KEY,
            # Address literals: pinned without DNS.
            agent_api_base_url="http://198.51.100.8:8443/v1",
            **config,
        ),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        event_callback=lambda name, value: events.append((name, value)),
    )
    request = RunRequest(
        task_dir=task, agent_name="codex", model="gpt-test", **request_options
    )
    return services, request, captured, events


def test_production_fails_before_the_agent_without_a_judge_llm_credential(
    tmp_path: Path,
) -> None:
    services, request, captured, _ = _production(
        tmp_path, 'judge_model = "gpt-judge"\n', {}
    )

    with pytest.raises(SetupError, match="'gpt-judge'"):
        services.run(request)
    assert captured == []
    assert not (tmp_path / "data").exists()


def test_production_hands_the_judge_llm_to_the_judges(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, API_KEY)
    services, request, captured, events = _production(
        tmp_path,
        'judge_model = "gpt-judge"\njudge_model_effort = "low"\n',
        {
            "judge_llm": JudgeLLMRequest(
                auth=JudgeLLMAuth.API, base_url="http://198.51.100.30:9000"
            )
        },
    )
    services.client = object()

    services.run(request)

    composition = captured[0].work_creator.__self__
    access = composition.judge_model_access
    launch = json.loads(access.launch)
    assert launch["command"] is None
    assert launch["llm"]["url"] == "http://198.51.100.30:9000/v1/responses"
    assert launch["llm"]["effort"] == "low"
    assert [(e.hostname, e.port) for e in access.endpoints] == [("198.51.100.30", 9000)]
    assert (
        "judge_llm",
        {
            "model": "gpt-judge",
            "effort": "low",
            "credential": "api",
            "reused": False,
            "expires_at": None,
        },
    ) in events
    assert API_KEY not in json.dumps([value for _, value in events])


def test_production_shares_one_pinned_endpoint_for_exec_and_llm(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, API_KEY)
    services, request, captured, events = _production(
        tmp_path,
        'judge_model = "gpt-test"\nmodel_access = true\n',
        {
            "judge_llm": JudgeLLMRequest(
                auth=JudgeLLMAuth.API, base_url="http://198.51.100.8:8443/v1"
            )
        },
    )
    services.client = object()

    services.run(request)

    access = captured[0].work_creator.__self__.judge_model_access
    launch = json.loads(access.launch)
    assert launch["llm"]["api_key"] == API_KEY
    assert launch["command"] is not None
    # One pinned endpoint serves both exec and llm.
    assert [(e.hostname, e.port) for e in access.endpoints] == [("198.51.100.8", 8443)]
    assert dict(events)["judge_llm"]["reused"] is False


def test_production_codex_agent_on_another_base_url_lends_nothing(
    tmp_path: Path,
) -> None:
    services, request, _, _ = _production(tmp_path, 'judge_model = "gpt-test"\n', {})

    with pytest.raises(SetupError, match="--judge-llm-auth"):
        services.run(request)


def test_prepare_images_adds_the_codex_cli_for_a_codex_login(
    tmp_path: Path, monkeypatch
) -> None:
    from rsi_harness.runtime import production

    home = codex_home(tmp_path, CHATGPT_LOGIN)
    services, request, captured, _ = _production(
        tmp_path,
        'judge_model = "gpt-judge"\n',
        {
            "judge_llm": JudgeLLMRequest(
                auth=JudgeLLMAuth.CODEX_LOGIN,
                codex_home=home,
                proxy="http://198.51.100.40:3128",
            )
        },
    )
    services.client = object()
    services.run(request)
    composition = captured[0].work_creator.__self__
    assert composition.judge_model_access.needs_codex_cli
    calls = {}

    class Builder:
        def __init__(self, client, **options):
            del client, options

        def prepare(self, definition, *, agent):
            calls["work_agent"] = agent.name
            return images

        def prepare_judge_codex(self, definition, **options):
            calls["codex"] = options
            return tmp_path / "data" / JUDGE_CODEX_CACHE / "codex-1-x86_64"

    images = make_run_plan(tmp_path).images
    monkeypatch.setattr(production, "DockerImageBuilder", Builder)
    composition.prepare_images(composition.definition)

    assert (
        calls["codex"]["cache_root"] == tmp_path.resolve() / "data" / JUDGE_CODEX_CACHE
    )
    assert calls["codex"]["work_agent"].name == "codex"
    assert composition.judge_model_access.codex_cli == (
        tmp_path / "data" / JUDGE_CODEX_CACHE / "codex-1-x86_64"
    )


def test_judge_llm_options_without_a_judge_model_are_reported_unused(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, API_KEY)
    services, request, captured, events = _production(
        tmp_path, "", {"judge_llm": JudgeLLMRequest(auth=JudgeLLMAuth.API)}
    )
    services.client = object()

    services.run(request)

    assert ("judge_llm_ignored", None) in events
    assert captured[0].work_creator.__self__.judge_model_access is None


def test_prepare_images_refuses_an_operator_credential_without_the_clean_image(
    tmp_path: Path, monkeypatch
) -> None:
    from rsi_harness.models import RootfsSnapshotMode
    from rsi_harness.runtime import production

    monkeypatch.setenv(DEFAULT_API_KEY_ENV, API_KEY)
    services, request, captured, _ = _production(
        tmp_path,
        'judge_model = "gpt-judge"\n',
        {
            "judge_llm": JudgeLLMRequest(
                auth=JudgeLLMAuth.API, base_url="http://198.51.100.30:9000"
            )
        },
    )
    services.client = object()
    services.run(request)
    composition = captured[0].work_creator.__self__
    # A WORKDIR of /: the Judge would start from a snapshot of Work.
    images = make_run_plan(tmp_path).images.model_copy(
        update={"rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS}
    )

    class Builder:
        def __init__(self, client, **options):
            del client, options

        def prepare(self, definition, *, agent):
            del definition, agent
            return images

        def prepare_judge_codex(self, definition, **options):
            raise AssertionError("refused before the Codex CLI")

    monkeypatch.setattr(production, "DockerImageBuilder", Builder)
    with pytest.raises(SetupError, match="snapshot of Work.*WORKDIR is /"):
        composition.prepare_images(composition.definition)
