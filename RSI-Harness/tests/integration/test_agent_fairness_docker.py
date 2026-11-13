"""Real Docker: subagents run on the run's model, and no account-level tool.

Each Work case is a full production run of the Agent's real pinned CLI on the
fixture task, against tests/fakes/fake_model_server.py, as in
test_driven_agents_docker.py. The task image plants what would otherwise
reach another model or bring more tools: agent definitions naming a
stronger model, settings turning the CLI's subagent pin off or naming
another main, alias or vision model, system-wide managed config, MCP
servers, extensions. The fake model then asks the CLI to start subagents
(and, for Kimi Code, to edit its config first; for Qwen Code, to read an
image). Every request the fake receives,
subagents' included, must name the run's model, offer none of the Agent's
account-level tools, and, where the CLI has the parameter, offer no subagent
``model``. A subagent the run's model may use answers; one naming another
model fails before any request.

Two Judge cases run Claude Code and Codex one-shot through ``rsi-agent`` in
a model-access Judge, the same way.
"""

from __future__ import annotations

import base64
import json
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rsi_harness.models import CompileOptions, RunRequest, RunStatus
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_agent_one_shot_docker import _docker as _python_docker
from tests.integration.test_agent_one_shot_docker import _SubmitOnly
from tests.integration.test_driven_agents_docker import CASES as AGENT_CASES
from tests.integration.test_driven_agents_docker import MODEL, DrivenRun, _run
from tests.integration.test_judge_model_access_docker import PYTHON_IMAGE
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory
from tests.integrations.test_agent_driver import _load_fake_model_server
from tests.integrations.test_agent_fairness import (
    CLAUDE_ACCOUNT_TOOLS,
    QWEN_GUARD_TOOLS,
)

pytestmark = pytest.mark.integration

# Each subagent's prompt carries its number, so its own requests show.
SUBAGENT = "fairness-subagent-{}"
# The fake model's text reply, which a subagent returns to its caller.
DONE = "I think I am finished"
SUBMIT = "printf 42 > /workspace/answer.txt && rsi-submit"
# At most this many messages in a subagent's first request.
SUBAGENT_MESSAGES = 4
# A Codex model whose catalog entry turns on multi-agent by itself.
CODEX_MODEL = "gpt-6-sol"
AGENT_TIMEOUT_SECONDS = 120
API_KEY = "sk-synthetic-fairness-key-2e8a"
MAX_OUTPUT_TOKENS = 8192
# Every tool a Claude Code request may offer: local tools only. A release
# that adds one fails here until it is denied or added.
CLAUDE_LOCAL_TOOLS = (
    "Agent",
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "NotebookEdit",
    "Read",
    "ReportFindings",
    "Skill",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "TodoWrite",
    "Write",
)
# A 1x1 PNG, for Qwen Code's vision route.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)


def _spawn(tool: str, number: int, **arguments: Any) -> dict[str, Any]:
    return {
        "tool": tool,
        "arguments": {
            "description": f"helper {number}",
            "prompt": SUBAGENT.format(number),
            **arguments,
        },
    }


@dataclass(frozen=True)
class FairnessCase:
    agent: str
    # Files the task image carries, by absolute path.
    plant: Mapping[str, str] = field(default_factory=dict)
    # Tool calls after rsi-submit, as fake-model ``commands`` entries, each
    # with a text one of the tool results before the next call must hold.
    calls: tuple[tuple[Any, str], ...] = ()
    # Numbers of the subagents (SUBAGENT) that must have reached the model.
    subagents: tuple[int, ...] = ()
    # Tools no request may offer.
    absent_tools: tuple[str, ...] = ()
    # Tools that, where offered, must have no ``model`` parameter.
    no_model_param: tuple[str, ...] = ()
    # Strings no request may contain anywhere.
    absent_text: tuple[str, ...] = ()
    # Tools the main session may not offer, though its subagents do.
    main_absent_tools: tuple[str, ...] = ()
    # When set, every request's tools are among these.
    allowed_tools: tuple[str, ...] = ()
    model: str = MODEL
    # The model every request names, when not ``model`` (an alias).
    sent_model: str = ""
    # Added to RSI_AGENT_EXTRA_ENV.
    extra_env: Mapping[str, str] = field(default_factory=dict)
    # Call rsi-submit first, through the CLI's shell tool.
    submit: bool = True
    name: str = ""

    def __str__(self) -> str:
        return self.name or self.agent


_STRONG_SETTINGS = {
    "env": {
        "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "0",
        "CLAUDE_CODE_SUBAGENT_MODEL": "claude-opus-x",
    }
}
# Managed settings, which outrank the launch's --settings.
_CLAUDE_MANAGED = {
    "/etc/claude-code/managed-settings.json": json.dumps(_STRONG_SETTINGS)
}
_CLAUDE_STRONG_AGENT = {
    "/workspace/.claude/agents/strong.md": (
        "---\nname: strong\ndescription: Strong helper\nmodel: opus\n"
        "tools: Bash, CronCreate, SendMessage\n---\nHelp.\n"
    ),
}
# Managed config, which outranks -c flags: multi-agent back on.
_CODEX_MANAGED = {
    "/etc/codex/managed_config.toml": (
        "[agents]\nenabled = true\n\n"
        "[features]\nmulti_agent = true\nmulti_agent_v2 = true\n"
    ),
}
# What makes Kimi bind subagents to another model, as the Agent could write
# it into its own config during Work: a second model, chosen unforced.
_KIMI_POOL = (
    'cfg="$KIMI_CODE_HOME/config.toml"'
    " && sed -i '/^\\[secondary_model\\]$/,/^$/d' \"$cfg\""
    ' && printf \'\\n[secondary_model]\\nmodel = "big"\\n\\n[models.big]\\n'
    'provider = "rsi"\\nmodel = "big-model"\\nmax_context_size = 200000\\n'
    'max_output_size = 8192\\ncapabilities = ["tool_use"]\\n\' >> "$cfg"'
    ' && sleep 2 && grep -c big-model "$cfg" && echo config-edited'
)

CASES = (
    FairnessCase(
        "claude-code",
        plant={
            **_CLAUDE_STRONG_AGENT,
            "/workspace/.claude/settings.json": json.dumps(_STRONG_SETTINGS),
            "/workspace/.mcp.json": json.dumps(
                {"mcpServers": {"planted": {"command": "true"}}}
            ),
        },
        calls=(
            (
                _spawn(
                    "Agent",
                    1,
                    subagent_type="strong",
                    model="opus",
                    run_in_background=False,
                ),
                DONE,
            ),
            (
                _spawn(
                    "Agent",
                    2,
                    subagent_type="general-purpose",
                    model="opus",
                    run_in_background=False,
                ),
                DONE,
            ),
        ),
        subagents=(1, 2),
        absent_tools=(*CLAUDE_ACCOUNT_TOOLS, "WebSearch", "WebFetch"),
        no_model_param=("Agent",),
        allowed_tools=CLAUDE_LOCAL_TOOLS,
    ),
    # An alias the operator maps to the endpoint's model: the working
    # directory's settings remap it and name another main model, and the
    # image's managed settings turn the subagent pin off.
    FairnessCase(
        "claude-code",
        plant={
            **_CLAUDE_STRONG_AGENT,
            **_CLAUDE_MANAGED,
            "/workspace/.claude/settings.json": json.dumps(
                {
                    "model": "claude-opus-x",
                    "env": {"ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-x"},
                }
            ),
        },
        calls=(
            (
                _spawn(
                    "Agent",
                    1,
                    subagent_type="strong",
                    model="opus",
                    run_in_background=False,
                ),
                DONE,
            ),
        ),
        subagents=(1,),
        absent_tools=(*CLAUDE_ACCOUNT_TOOLS, "WebSearch", "WebFetch"),
        no_model_param=("Agent",),
        allowed_tools=CLAUDE_LOCAL_TOOLS,
        model="sonnet",
        sent_model=MODEL,
        extra_env={"ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL},
        name="claude-code-alias-managed",
    ),
    FairnessCase(
        "codex",
        plant=_CODEX_MANAGED,
        absent_text=("spawn_agent", "Multi-agent tools"),
        name="codex-default-model",
    ),
    # Its catalog entry turns multi-agent on by itself. Its tools are not
    # function tools the fake can call, so it only answers.
    FairnessCase(
        "codex",
        plant=_CODEX_MANAGED,
        absent_text=("spawn_agent", "Multi-agent tools"),
        model=CODEX_MODEL,
        submit=False,
    ),
    FairnessCase(
        "kimi-code",
        calls=(
            (_KIMI_POOL, "config-edited"),
            (_spawn("Agent", 1, subagent_type="coder"), DONE),
        ),
        subagents=(1,),
        no_model_param=("Agent", "AgentSwarm"),
    ),
    FairnessCase(
        "opencode",
        plant={
            "/workspace/.opencode/agent/strong.md": (
                "---\ndescription: Strong helper\nmode: subagent\n"
                "model: rsi/strong-model\n---\nHelp.\n"
            ),
            # A second model, and the run's renamed to another id.
            "/workspace/opencode.json": json.dumps(
                {
                    "provider": {
                        "rsi": {
                            "models": {
                                "strong-model": {
                                    "name": "strong-model",
                                    "tool_call": True,
                                },
                                MODEL: {"id": "strong-model"},
                            }
                        }
                    }
                }
            ),
            # Managed config, merged after the run's: the whitelist widened.
            "/etc/opencode/opencode.json": json.dumps(
                {"provider": {"rsi": {"whitelist": [MODEL, "strong-model"]}}}
            ),
        },
        calls=(
            (_spawn("task", 1, subagent_type="general"), DONE),
            (_spawn("task", 2, subagent_type="strong"), "Model not found"),
        ),
        subagents=(1,),
    ),
    FairnessCase(
        "qwen-code",
        plant={
            "/workspace/.qwen/agents/strong.md": (
                "---\nname: strong\ndescription: Strong helper\n"
                "model: big-model\n---\nHelp.\n"
            ),
            # Another main model, and a vision model on the run's endpoint
            # that would take a whole image turn.
            "/workspace/.qwen/settings.json": json.dumps(
                {
                    "model": {"name": "big-model"},
                    "fastModel": "big-model",
                    "compactionModel": "big-model",
                    "experimental": {"agentTeam": True},
                    "visionModel": "strong-vl",
                    "modelProviders": {
                        "openai": [
                            {
                                "id": "strong-vl",
                                "envKey": "OPENAI_API_KEY",
                                "baseUrl": "@API_BASE_URL@",
                                "capabilities": {"vision": True, "agent": True},
                            }
                        ]
                    },
                }
            ),
        },
        calls=(
            (_spawn("agent", 1, subagent_type="strong"), "not found in registry"),
            (
                f"printf %s {PNG} | base64 -d > /workspace/pic.png && echo png-made",
                "png-made",
            ),
            (
                {"tool": "read_file", "arguments": {"file_path": "/workspace/pic.png"}},
                "pic.png",
            ),
        ),
        # Agent teams, which the planted settings would turn on, bring these.
        absent_tools=(*QWEN_GUARD_TOOLS, "team_delete", "task_create"),
    ),
    FairnessCase(
        "pi",
        plant={
            "/home/agent/.pi/agent/extensions/probe/index.ts": (
                "export default function (pi) {\n"
                "  pi.registerTool({name: 'probe_tool', label: 'probe',"
                " description: 'probe', parameters: {type: 'object',"
                " properties: {}}, execute: async () => ({content: []})});\n"
                "}\n"
            ),
            "/workspace/.pi/settings.json": json.dumps(
                {"extensions": ["+builtin:mcp"]}
            ),
        },
        absent_tools=("probe_tool",),
    ),
    FairnessCase(
        "zcode",
        plant={
            "/workspace/.zcode/agents/strong.md": (
                "---\nname: strong\ndescription: Strong helper\n"
                "model: rsi/other-model\n---\nHelp.\n"
            ),
            "/workspace/.zcode/agents/same.md": (
                "---\nname: same\ndescription: Same helper\n"
                "model: inherit\n---\nHelp.\n"
            ),
        },
        calls=(
            (_spawn("Agent", 1, subagent_type="strong"), "model-not-found"),
            (_spawn("Agent", 2, subagent_type="same"), DONE),
        ),
        subagents=(2,),
        absent_tools=("OffPeakList", "OffPeakCreate", "CronCreate", "CronDelete"),
        # ZCode gives a subagent its own default tools; the deny list the
        # driver sends reaches the main session only. A subagent's
        # ReplyToChannel has no bot conversation to reply to.
        main_absent_tools=("ReplyToChannel",),
    ),
)

_DOCKERFILE = """\
FROM ubuntu:24.04
RUN mkdir -p /workspace
COPY plant/ /
WORKDIR /workspace
"""


def _assert_requests(case: Any, requests: list[dict[str, Any]]) -> None:
    assert requests
    for entry in requests:
        sent = getattr(case, "sent_model", "") or case.model
        assert entry["model"] == sent, entry
        offered = set(entry["tool_names"])
        assert not offered & set(case.absent_tools), entry
        allowed = getattr(case, "allowed_tools", ())
        assert not allowed or offered <= set(allowed), offered - set(allowed)
        assert not [name for name in offered if name.startswith("mcp__")], entry
        assert not set(entry["watched"]) & set(case.absent_text), entry
        if entry["n"] == 1 or "Agent" in offered:
            assert not offered & set(getattr(case, "main_absent_tools", ())), entry
        for name, params in entry["tool_params"].items():
            assert "model" not in params, (name, params)
    for tool in case.no_model_param:
        # The check above ran on requests that offer the tool.
        assert any(tool in r["tool_params"] for r in requests), tool
    for number in case.subagents:
        # The subagent's own conversation: its prompt in a request that
        # starts afresh, not the caller's, which holds the whole run.
        prompt = SUBAGENT.format(number)
        assert any(
            prompt in r["watched"] and r["n_messages"] <= SUBAGENT_MESSAGES
            for r in requests
        ), prompt


def _assert_results(
    calls: list[tuple[Any, str]], requests: list[dict[str, Any]], *, submit: bool
) -> None:
    """Each call's result, up to the next call, holds its text."""
    made = [r["n"] for r in requests if r["call"]][1 if submit else 0 :]
    assert len(made) == len(calls), [r["call"] for r in requests]
    ends = [*made[1:], max(r["n"] for r in requests) + 1][: len(made)]
    for (_, expected), start, end in zip(calls, made, ends, strict=True):
        results = [r["last_tool"] or "" for r in requests if start < r["n"] <= end]
        assert any(expected in text for text in results), (expected, results)


def _scenario(case: Any, commands: list[Any]) -> dict[str, Any]:
    return {
        "cycle": ["text"],
        # The Agent's first request makes the first call, and every tool
        # result the next; any other request (a subagent's) answers.
        "at": {"1": "tool"} if commands else {},
        "after_tool": "next",
        "commands": commands,
        "schemas": list(case.no_model_param),
        "watch": [
            *case.absent_text,
            *(SUBAGENT.format(number) for number in case.subagents),
        ],
    }


@pytest.mark.parametrize("case", CASES, ids=str)
def test_subagents_use_the_run_model_and_account_tools_stay_off(
    tmp_path: Path, request: pytest.FixtureRequest, case: FairnessCase
) -> None:
    [base] = [c for c in AGENT_CASES if c.agent == case.agent]
    commands = [*([SUBMIT] if case.submit else []), *(c for c, _ in case.calls)]
    scenario = _scenario(case, commands)
    files = {f"plant{path}": text for path, text in case.plant.items()} or {
        "plant/.keep": ""
    }

    run: DrivenRun = _run(
        tmp_path,
        request,
        base,
        scenario,
        stop_hook=False,
        timeout_seconds=AGENT_TIMEOUT_SECONDS,
        dockerfile=_DOCKERFILE,
        model=case.model,
        task_files=files,
        extra_env=case.extra_env,
    )

    output = run.agent_output
    status = RunStatus.COMPLETED if case.submit else RunStatus.NO_VALID_SUBMISSION
    assert run.result.status == status, (run.result, output[-4000:])
    assert run.result.total_rounds == int(case.submit)
    _assert_requests(case, run.requests)
    _assert_results(list(case.calls), run.requests, submit=case.submit)


# -- A Judge's one-shot call --------------------------------------------------


@dataclass(frozen=True)
class OneShotFairnessCase:
    agent: str
    plant: Mapping[str, str] = field(default_factory=dict)
    calls: tuple[tuple[Any, str], ...] = ()
    subagents: tuple[int, ...] = ()
    absent_tools: tuple[str, ...] = ()
    no_model_param: tuple[str, ...] = ()
    absent_text: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    model: str = MODEL
    base_path: str = "/v1"

    def __str__(self) -> str:
        return self.agent


ONESHOT_CASES = (
    OneShotFairnessCase(
        "claude-code",
        plant={".claude/settings.json": json.dumps(_STRONG_SETTINGS)},
        calls=(
            (
                _spawn(
                    "Agent",
                    1,
                    subagent_type="general-purpose",
                    model="opus",
                    run_in_background=False,
                ),
                DONE,
            ),
        ),
        subagents=(1,),
        absent_tools=(*CLAUDE_ACCOUNT_TOOLS, "WebSearch", "WebFetch"),
        no_model_param=("Agent",),
        allowed_tools=CLAUDE_LOCAL_TOOLS,
        base_path="",
    ),
    OneShotFairnessCase(
        "codex",
        absent_text=("spawn_agent", "Multi-agent tools"),
        model=CODEX_MODEL,
    ),
)

JUDGE_SCRIPT = r"""#!/bin/bash
set -u
mkdir -p /tmp/oneshot/work
python3 - @FILES@ <<'PY'
import base64, json, os, sys
for name, text in json.loads(base64.b64decode(sys.argv[1])).items():
    path = os.path.join("/tmp/oneshot/work", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as stream:
        stream.write(text)
PY
printf '%s\n' 'Answer the question.' > /tmp/oneshot/prompt.md
rsi-agent exec --prompt-file /tmp/oneshot/prompt.md --workdir /tmp/oneshot/work \
    --timeout 180 --log /tmp/oneshot/call.log --output /tmp/oneshot/call.out
status=$?
echo "call exit $status"
test "$status" = 0 || cat /tmp/oneshot/call.log
python3 -c 'import json, sys
json.dump({"reward": 1 if sys.argv[1] == "0" else 0},
          open("/logs/verifier/reward.json", "w"))' "$status"
"""


def _oneshot_task(root: Path, case: OneShotFairnessCase) -> Path:
    task = root / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PYTHON_IMAGE}\nRUN mkdir -p /workspace\nWORKDIR /workspace\n"
    )
    config = task / "task.toml"
    text = config.read_text()
    for old, new in (
        ('network_mode = "public"', 'network_mode = "no-network"'),
        ("[verifier]\ntimeout_sec = 30", "[verifier]\ntimeout_sec = 300"),
    ):
        assert old in text
        text = text.replace(old, new)
    config.write_text(text + "\n[metadata.rsi_harness.verifier]\nmodel_access = true\n")
    files = base64.b64encode(json.dumps(dict(case.plant)).encode()).decode()
    (task / "tests" / "test.sh").write_text(JUDGE_SCRIPT.replace("@FILES@", files))
    return task


@pytest.mark.parametrize("case", ONESHOT_CASES, ids=str)
def test_judge_one_shot_subagents_use_the_run_model(
    tmp_path: Path, request: pytest.FixtureRequest, case: OneShotFairnessCase
) -> None:
    client = _python_docker()
    gateway = next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )
    commands = [call for call, _ in case.calls]
    server = _load_fake_model_server().FakeModelServer(
        tmp_path / "model-requests.jsonl",
        _scenario(case, commands),
        host="0.0.0.0",
    )
    threading.Thread(target=server.httpd.serve_forever, daemon=True).start()
    request.addfinalizer(server.stop)
    port = server.httpd.server_address[1]
    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=API_KEY,
            agent_api_base_url=f"http://{gateway}:{port}{case.base_path}",
            agent_context_window="200000",
            agent_max_output_tokens=str(MAX_OUTPUT_TOKENS),
        ),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: _SubmitOnly(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
    )

    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)

    result = services.run(
        RunRequest(
            task_dir=_oneshot_task(tmp_path, case),
            agent_name=case.agent,
            model=case.model,
            gpu_selectors=("0",),
            options=CompileOptions(agent_name=case.agent, primary_reward="reward"),
        )
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    feedback_path = run_root / "feedback" / "agent-1.log"
    feedback = feedback_path.read_text() if feedback_path.exists() else ""
    assert result.status == RunStatus.COMPLETED, (result, feedback)
    assert "call exit 0" in feedback, feedback
    assert result.best_score == 1, feedback
    requests = server.requests()
    _assert_requests(case, requests)
    _assert_results(list(case.calls), requests, submit=False)
