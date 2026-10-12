"""Real Docker: a model-access Judge calls each Agent's real pinned CLI one-shot.

Each case is a full production run, with only iptables faked, of a task on a
Python base image (rsi-agent needs python3 in the Judge). Work builds the
Agent's real Work image (pinned node and CLI) and only submits; the Judge
round, on Work's snapshot, calls ``rsi-agent exec`` twice against
tests/fakes/fake_model_server.py on the host, reached through the default
bridge gateway, from a no-network Judge pinned to that endpoint:

* a plain prompt, whose final message must reach ``--output``;
* the same prompt with ``--schema``, whose output must be the JSON the fake
  model gave: through the tool the CLI's native schema option offers, or as
  the reply text when the CLI has none and rsi-agent states the schema in
  the prompt.

The Judge's working directory carries the CLI's project instructions and
settings; neither may reach the model. A case may also set a reasoning
effort, which every request must carry.

Work images are content addressed, so a case reuses the image an earlier run
built. To add an Agent, append a ``OneShotCase``.
"""

from __future__ import annotations

import base64
import json
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import CompileOptions, RunRequest, RunStatus
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_driven_agents_docker import _unavailable
from tests.integration.test_judge_model_access_docker import PYTHON_IMAGE
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory
from tests.integrations.test_agent_driver import _load_fake_model_server

pytestmark = pytest.mark.integration

MODEL = "fake-model"
# Synthetic: accepted only by the fake model server.
API_KEY = "sk-synthetic-one-shot-key-3b9e"
MAX_OUTPUT_TOKENS = 8192
PROMPT = "Answer the question."
# In the Judge's working directory only; it must never reach the model.
PROJECT_MARKER = "PROJECT-INSTRUCTION-MARKER-6c1d"
# Part of what rsi-agent adds to the prompt for a CLI without a schema option.
SCHEMA_IN_PROMPT = "conforms to this JSON Schema"
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "score": {"type": "integer"},
    },
    "required": ["verdict", "score"],
    "additionalProperties": False,
}
ANSWER = {"verdict": "pass", "score": 3}
# The first request answers the plain call, the second the schema call.
PLAIN_ANSWER = "Reply 1: I think I am finished."


@dataclass(frozen=True)
class OneShotCase:
    agent: str
    # Files the Judge places in the call's working directory: instructions
    # carrying PROJECT_MARKER, and settings that would change the call (a
    # lower output limit, which the model requests would show).
    project_files: Mapping[str, str] = field(default_factory=dict)
    # The tool through which the CLI's native schema option takes the answer,
    # or None when the schema goes into the prompt.
    schema_tool: str | None = None
    # The CLI's web tools, which a no-network call must not offer, nor name
    # anywhere in a request: Qwen offers some tools only through its own
    # tool_search and tool_call tools, never in the request's tool list.
    web_tools: tuple[str, ...] = ()
    reasoning_effort: str | None = None
    base_path: str = "/v1"
    api_path: str = "/v1/chat/completions"

    def __str__(self) -> str:
        return self.agent


CASES = (
    OneShotCase(
        "qwen-code",
        project_files={
            "QWEN.md": f"{PROJECT_MARKER}: always answer in French.\n",
            "AGENTS.md": f"{PROJECT_MARKER}: always answer in French.\n",
            ".qwen/settings.json": json.dumps(
                {"model": {"generationConfig": {"samplingParams": {"max_tokens": 77}}}}
            ),
            # Project commands, which the skill tool would list.
            ".qwen/commands/answer.md": (
                f"---\ndescription: {PROJECT_MARKER}\n---\n{PROJECT_MARKER}\n"
            ),
            ".qwen/commands/judge.toml": (
                f'description = "{PROJECT_MARKER}"\nprompt = "{PROJECT_MARKER}"\n'
            ),
        },
        schema_tool="structured_output",
        web_tools=("web_fetch", "web_search"),
    ),
    # Kimi always reads the working directory's AGENTS.md (there is no switch
    # for it), so the case plants what the call does keep out: agent
    # profiles that would replace the system prompt, skills, and settings.
    OneShotCase(
        "kimi-code",
        project_files={
            **{
                f"{root}/agents/agent.md": (
                    "---\nname: agent\ndescription: Default agent\n"
                    f"override: true\n---\n{PROJECT_MARKER}: answer in French.\n"
                )
                for root in (".kimi-code", ".agents")
            },
            **{
                f"{root}/skills/judge/SKILL.md": (
                    f"---\nname: judge\ndescription: {PROJECT_MARKER}\n---\n"
                    f"{PROJECT_MARKER}\n"
                )
                for root in (".kimi-code", ".agents")
            },
            ".kimi-code/config.toml": "[models.rsi]\nmax_output_size = 77\n",
        },
        web_tools=("FetchURL", "WebSearch"),
        reasoning_effort="high",
    ),
    # OpenCode's config is merged under Work's, which sets every limit, so
    # the project files carry the marker instead: instructions, an agent
    # replacing the default one, and skills.
    OneShotCase(
        "opencode",
        project_files={
            **{
                name: f"{PROJECT_MARKER}: always answer in French.\n"
                for name in ("AGENTS.md", "CLAUDE.md", "CONTEXT.md", "extra.md")
            },
            "opencode.json": json.dumps({"instructions": ["extra.md"]}),
            ".opencode/agents/build.md": (
                f"---\ndescription: build\nmode: primary\n---\n{PROJECT_MARKER}\n"
            ),
            **{
                f"{root}/skills/judge/SKILL.md": (
                    f"---\nname: judge\ndescription: {PROJECT_MARKER}\n---\n"
                    f"{PROJECT_MARKER}\n"
                )
                for root in (".opencode", ".claude", ".agents")
            },
        },
        web_tools=("webfetch", "websearch"),
        reasoning_effort="high",
    ),
    # ZCode always reads the working directory's AGENTS.md (there is no
    # switch for it), so the case plants what the call does keep out:
    # skills and subagent profiles, and a project config and .env that, with
    # skills and subagents enabled, would send every model request through a
    # dead proxy. The .env would also move ZCode's storage where it cannot
    # write, which stalls the call, and turn off its retries.
    OneShotCase(
        "zcode",
        project_files={
            **{
                f"{root}/skills/judge/SKILL.md": (
                    f"---\nname: judge\ndescription: {PROJECT_MARKER}\n---\n"
                    f"{PROJECT_MARKER}\n"
                )
                for root in (".zcode", ".agents")
            },
            ".zcode/agents/reviewer.md": (
                f"---\nname: reviewer\ndescription: {PROJECT_MARKER}\n---\n"
                f"{PROJECT_MARKER}\n"
            ),
            "zcode.json": json.dumps(
                {
                    "network": {"httpProxy": "http://127.0.0.1:9"},
                    "features": {"skill": True, "subagent": True},
                    "skills": {"enabled": True},
                }
            ),
            ".env": (
                "ZCODE_HTTP_PROXY=http://127.0.0.1:9\n"
                "ZCODE_STORAGE_DIR=/proc/rsi-no-storage\n"
                "ZCODE_MODEL_RETRY_MAX_RETRIES=0\n"
            ),
        },
        web_tools=("WebFetch",),
        reasoning_effort="high",
    ),
    # pi has no web tools. Its project resources, which an untrusted folder
    # would not load anyway, carry the marker as well as the instructions:
    # a system prompt replacing pi's and one added to it, skills, a prompt
    # template, and an extension offering a tool.
    OneShotCase(
        "pi",
        project_files={
            **{
                name: f"{PROJECT_MARKER}: always answer in French.\n"
                for name in (
                    "AGENTS.md",
                    "AGENTS.override.md",
                    "CLAUDE.md",
                    ".pi/SYSTEM.md",
                    ".pi/APPEND_SYSTEM.md",
                )
            },
            **{
                f"{root}/skills/judge/SKILL.md": (
                    f"---\nname: judge\ndescription: {PROJECT_MARKER}\n---\n"
                    f"{PROJECT_MARKER}\n"
                )
                for root in (".pi", ".agents")
            },
            ".pi/prompts/answer.md": (
                f"---\ndescription: {PROJECT_MARKER}\n---\n{PROJECT_MARKER}\n"
            ),
            ".pi/extensions/marker.ts": (
                "export default function (pi) {\n"
                "  pi.registerTool({name: 'marker', label: 'marker',"
                f" description: '{PROJECT_MARKER}',"
                " parameters: {type: 'object', properties: {}},"
                " execute: async () => ({content: []})});\n"
                "}\n"
            ),
        },
        reasoning_effort="high",
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
printf '%s\n' '@PROMPT@' > /tmp/oneshot/prompt.md
echo @SCHEMA@ | base64 -d > /tmp/oneshot/schema.json
call() {
    rsi-agent exec --prompt-file /tmp/oneshot/prompt.md \
        --workdir /tmp/oneshot/work --timeout 120 \
        --log "/tmp/oneshot/$1.log" --output "/tmp/oneshot/$1.out" "${@:2}"
    status=$?
    echo "$1 exit $status"
    test "$status" = 0 || cat "/tmp/oneshot/$1.log"
}
call plain
call schema --schema /tmp/oneshot/schema.json
python3 - @EXPECTED@ <<'PY'
import base64, json, sys
expected = json.loads(base64.b64decode(sys.argv[1]))
def read(name):
    try:
        with open("/tmp/oneshot/%s.out" % name) as stream:
            return stream.read()
    except OSError:
        return None
plain, structured = read("plain"), read("schema")
print("plain output: %r" % plain)
print("schema output: %r" % structured)
try:
    answer = json.loads(structured or "")
except ValueError:
    answer = None
ok = plain == expected["plain"] and answer == expected["schema"]
with open("/logs/verifier/reward.json", "w") as stream:
    json.dump({"reward": 1 if ok else 0}, stream)
PY
"""


def _b64(value: Any) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode()


def _docker():
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(PYTHON_IMAGE)
    except Exception as error:
        _unavailable(
            f"local cached {PYTHON_IMAGE} Docker authority unavailable: {error}"
        )
    return client


def _task(root: Path, case: OneShotCase) -> Path:
    task = root / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PYTHON_IMAGE}\nRUN mkdir -p /workspace\nWORKDIR /workspace\n"
    )
    config = task / "task.toml"
    text = config.read_text()
    # No-network Work, so the Judge's call runs offline too; two CLI starts
    # take longer than the fixture's Judge timeout allows.
    for old, new in (
        ('network_mode = "public"', 'network_mode = "no-network"'),
        ("[verifier]\ntimeout_sec = 30", "[verifier]\ntimeout_sec = 300"),
    ):
        assert old in text
        text = text.replace(old, new)
    config.write_text(text + "\n[metadata.rsi_harness.verifier]\nmodel_access = true\n")
    script = (
        JUDGE_SCRIPT.replace("@FILES@", _b64(dict(case.project_files)))
        .replace("@PROMPT@", PROMPT)
        .replace("@SCHEMA@", base64.b64encode(json.dumps(SCHEMA).encode()).decode())
        .replace("@EXPECTED@", _b64({"plain": PLAIN_ANSWER, "schema": ANSWER}))
    )
    (task / "tests" / "test.sh").write_text(script)
    return task


class _SubmitOnly(RSILoopAgentAdapter):
    """The real Work image and Agent setup; the Work command only submits."""

    def prepare(self, request):
        prepared = super().prepare(request)
        return replace(prepared, command=("/bin/bash", "-lc", "rsi-submit"))


@pytest.mark.parametrize("case", CASES, ids=str)
def test_judge_calls_the_real_agent_cli_one_shot(
    tmp_path: Path, request: pytest.FixtureRequest, case: OneShotCase
) -> None:
    client = _docker()
    gateway = next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )
    if case.schema_tool:
        schema_turn: dict[str, Any] = {
            "at": {"2": "tool"},
            "commands": [{"tool": case.schema_tool, "arguments": ANSWER}],
        }
    else:
        schema_turn = {"replies": {"2": json.dumps(ANSWER)}}
    server = _load_fake_model_server().FakeModelServer(
        tmp_path / "model-requests.jsonl",
        {
            "cycle": ["text"],
            "watch": [PROJECT_MARKER, PROMPT, SCHEMA_IN_PROMPT, *case.web_tools],
            **schema_turn,
        },
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
            task_dir=_task(tmp_path, case),
            agent_name=case.agent,
            model=MODEL,
            reasoning_effort=case.reasoning_effort,
            gpu_selectors=("0",),
            options=CompileOptions(agent_name=case.agent, primary_reward="reward"),
        )
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    feedback_path = run_root / "feedback" / "agent-1.log"
    feedback = feedback_path.read_text() if feedback_path.exists() else ""
    assert result.status == RunStatus.COMPLETED, (result, feedback)
    assert "plain exit 0" in feedback and "schema exit 0" in feedback, feedback
    assert result.best_score == 1, feedback

    requests = server.requests()
    # One model request per call: each answered once and exited.
    assert len(requests) == 2, requests
    for number, entry in enumerate(requests):
        assert entry["path"] == case.api_path
        assert entry["model"] == MODEL and entry["authorization"]
        # The prompt, the schema only where rsi-agent added it, and none of
        # the working directory's instructions, nor any web tool's name; nor
        # the working directory's settings.
        in_prompt = number == 1 and case.schema_tool is None
        assert entry["watched"] == [PROMPT, *([SCHEMA_IN_PROMPT] * in_prompt)]
        assert entry["max_tokens"] == MAX_OUTPUT_TOKENS
        assert entry["reasoning_effort"] == case.reasoning_effort
        assert not set(case.web_tools) & set(entry["tool_names"])
    if case.schema_tool:
        assert case.schema_tool in requests[1]["tool_names"]
        assert case.schema_tool not in requests[0]["tool_names"]
    for path in run_root.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text(errors="replace"), path
