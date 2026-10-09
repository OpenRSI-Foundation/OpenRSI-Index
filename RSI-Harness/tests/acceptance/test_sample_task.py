"""The harbor-in-judge sample (spec 7 M9): it compiles, the sample operator
policy grants it once, and its fixed procedure runs exactly Harbor 0.21's
``harbor run --env rsi_sandbox_harbor:ManagedSandboxEnvironment`` over the
suites shipped in /tests and scores the jobs itself."""

from __future__ import annotations

import filecmp
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions
from rsi_harness.runtime.sandbox_contracts import SandboxEnvTask
from rsi_harness.runtime.sandbox_policy import (
    load_sandbox_policy,
    resolve_env_grant,
    validate_sandbox_policy,
)
from rsi_harness.task.compiler import HarborTaskCompiler

REPO = Path(__file__).parents[2]
SAMPLE = REPO / "sample_tasks" / "harbor-in-judge"
POLICY = REPO / "sample_tasks" / "harbor-in-judge" / "operator-policy.toml"
TB2 = Path(os.environ.get("RSI_TB2_DIR", "/mnt/y1/temp/terminal_bench_2"))
TB2_TASKS = (
    "fix-git",
    "regex-log",
    "adaptive-rejection-sampler",
    "kv-store-grpc",
    "nginx-request-logging",
    "git-multibranch",
)
PLUGIN = "rsi_sandbox_harbor:ManagedSandboxEnvironment"
BUILDKIT_DIGEST = (
    "moby/buildkit@sha256:"
    "1e110c71d389d6d24f67b9438e2f7b8da749a6ff407b22a1631e025c95599368"
)
BUILDER = {
    "Id": "sha256:" + "b" * 64,
    "RepoDigests": [BUILDKIT_DIGEST],
    "Config": {"Entrypoint": ["buildkitd"], "Volumes": {"/var/lib/buildkit": {}}},
}
# A stand-in for Harbor's CLI: records each call and writes the trial
# results a real job would (oracle 1.0; nop $STUB_NOP_REWARD, default 0.0).
STUB_HARBOR = """#!{python}
import json, os, sys, tomllib
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["STUB_CALLS"], "a") as calls:
    calls.write(json.dumps({{
        "argv": args,
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "HARBOR_TELEMETRY": os.environ.get("HARBOR_TELEMETRY"),
    }}) + "\\n")

def values(flag):
    return [args[i + 1] for i, item in enumerate(args) if item == flag]

def name(task):  # Harbor's: [task] name, else the directory
    config = tomllib.loads((task / "task.toml").read_text())
    return (config.get("task") or {{}}).get("name") or task.name

path = Path(values("--path")[0])
agent = values("--agent")[0]
tasks = values("--include-task-name") or (
    [name(path)] if (path / "task.toml").exists()
    else sorted(name(p) for p in path.iterdir() if (p / "task.toml").exists())
)
job = Path(values("--jobs-dir")[0]) / values("--job-name")[0]
job.mkdir(parents=True)
(job / "result.json").write_text(json.dumps({{"id": "job"}}))
reward = 1.0 if agent == "oracle" else float(os.environ.get("STUB_NOP_REWARD", "0"))
for task in tasks:
    trial = job / (task.replace("/", "-") + "__abc")
    trial.mkdir()
    (trial / "result.json").write_text(json.dumps({{
        "task_name": task, "trial_name": trial.name, "exception_info": None,
        "verifier_result": {{"rewards": {{"reward": reward}}}},
    }}))
"""


def harbor_reward():
    """The sample's scorer, loaded without leaving bytecode in its /tests
    (the directory the Judge receives)."""
    spec = importlib.util.spec_from_file_location(
        "harbor_reward", SAMPLE / "tests" / "harbor_reward.py"
    )
    module = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    return module


def test_the_sample_compiles_as_a_cpu_task_with_a_v2_request():
    definition = HarborTaskCompiler().compile(SAMPLE, CompileOptions())

    assert definition.gpu_requirement.count == definition.verifier.gpu_count == 0
    assert str(definition.workdir) == "/workspace"
    sandbox = definition.sandbox
    assert isinstance(sandbox, SandboxEnvTask) and sandbox.version == 2
    for phase in (sandbox.environments.work, sandbox.environments.judge):
        assert phase.network == ("public", "none")
        assert phase.pull and phase.build
        assert phase.build_network == ("public",)
    assert dict(definition.verifier.environment) == {
        "RSI_HARBOR_SUITES": "${RSI_HARBOR_SUITES:-tb2}",
        "RSI_HARBOR_TB2_TASKS": "${RSI_HARBOR_TB2_TASKS:-}",
        "RSI_HARBOR_CONCURRENCY": "${RSI_HARBOR_CONCURRENCY:-2}",
    }


def test_the_operator_policy_grants_the_sample_once():
    sandbox = HarborTaskCompiler().compile(SAMPLE, CompileOptions()).sandbox
    policy = load_sandbox_policy(POLICY)

    assert validate_sandbox_policy(sandbox, policy, "docker") == sandbox
    grant = resolve_env_grant(
        sandbox, policy, {"work": BUILDER, "judge": BUILDER}, 2, 4096
    )
    environments = grant.environments
    for phase in (environments.work, environments.judge):
        assert phase.registries == ("docker.io", "public.ecr.aws")
        assert phase.build.builder_image == BUILDER["Id"]
        assert phase.build.state_fs == "loop-ext4"
        assert phase.build.network == ("public",)
    assert (grant.reserved_cpus, grant.reserved_memory_mb) == (28, 51456)
    assert grant.reserved_disk_mb == environments.host.pool_disk_mb == 65536
    policy_text = POLICY.read_text()
    assert policy_text.count(f'builder_image = "{BUILDKIT_DIGEST}"') == 2


def test_the_policy_refuses_what_it_does_not_approve():
    sandbox = HarborTaskCompiler().compile(SAMPLE, CompileOptions()).sandbox
    policy = load_sandbox_policy(POLICY)
    environments = policy.environments
    no_build = policy.model_copy(
        update={
            "environments": environments.model_copy(
                update={"judge": environments.judge.model_copy(update={"build": None})}
            )
        }
    )
    offline = policy.model_copy(
        update={
            "environments": environments.model_copy(
                update={
                    "work": environments.work.model_copy(update={"network": ("none",)})
                }
            )
        }
    )

    with pytest.raises(SetupError, match="judge.build: image build not approved"):
        validate_sandbox_policy(sandbox, no_build, "docker")
    with pytest.raises(SetupError, match="work.network: public not approved"):
        validate_sandbox_policy(sandbox, offline, "docker")


def test_the_work_image_installs_the_harbor_the_harness_pins():
    dockerfile = (SAMPLE / "environment" / "Dockerfile").read_text()
    pinned = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert "harbor==0.21.0" in pinned
    # Harbor and nothing else: no Docker client, socket or DOCKER_HOST.
    assert [
        line for line in dockerfile.splitlines() if line and not line.startswith("#")
    ] == [
        "FROM python:3.13-slim-bookworm",
        "RUN pip install --no-cache-dir harbor==0.21.0",
        "WORKDIR /workspace",
    ]


@pytest.mark.skipif(not TB2.is_dir(), reason="the TB2 source copy is not here")
def test_the_suites_are_the_published_tasks():
    tests = SAMPLE / "tests"
    assert sorted(path.name for path in (tests / "tb2-subset").iterdir()) == sorted(
        TB2_TASKS
    )
    for name in TB2_TASKS:
        assert_same_tree(TB2 / name, tests / "tb2-subset" / name)
    assert_same_tree(
        REPO / "tests" / "fixtures" / "tasks" / "harbor-compose-sidecar",
        tests / "compose-sidecar",
    )
    # A4's variant: no docker_image, and the one Dockerfile adjustment of
    # the M8 build test (the history of the 404ing repo, from the prebuilt
    # image, behind a git URL rewrite); everything else is fix-git.
    build = tests / "fix-git-build"
    original = (TB2 / "fix-git" / "task.toml").read_text().splitlines()
    assert (build / "task.toml").read_text().splitlines() == [
        line for line in original if not line.startswith("docker_image")
    ]
    dockerfile = (build / "environment" / "Dockerfile").read_text().splitlines()
    published = [line for line in original_dockerfile() if line]
    assert [line for line in dockerfile if line in set(published)] == published
    assert "FROM alexgshaw/fix-git:20251031 AS upstream" in dockerfile
    assert (
        'RUN git config --system url."file:///app/resources/personal-site.git"'
        ".insteadOf https://github.com/TheMikeMerrill/personal-site.git"
    ) in dockerfile
    for name in ("environment/resources", "tests", "solution"):
        assert_same_tree(TB2 / "fix-git" / name, build / name)
    for name in ("environment/setup.sh", "instruction.md"):
        assert filecmp.cmp(TB2 / "fix-git" / name, build / name, shallow=False)


def original_dockerfile() -> list[str]:
    """TB2 fix-git's own Dockerfile."""
    return (TB2 / "fix-git" / "environment" / "Dockerfile").read_text().splitlines()


def assert_same_tree(left: Path, right: Path) -> None:
    compared = filecmp.dircmp(left, right)
    assert (compared.left_only, compared.right_only, compared.diff_files) == (
        [],
        [],
        [],
    ), right
    for child in compared.common_dirs:
        assert_same_tree(left / child, right / child)


# -- the fixed procedure ---------------------------------------------------------


@pytest.fixture
def stub_harbor(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    harbor = bin_dir / "harbor"
    harbor.write_text(STUB_HARBOR.format(python=sys.executable))
    harbor.chmod(0o755)
    calls = tmp_path / "calls.jsonl"

    def run(**environment):
        env = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STUB_CALLS": str(calls),
            "RSI_SANDBOX_PYTHONPATH": "/run/rsi-harness/sandbox/py",
            **environment,
        }
        env = {key: value for key, value in env.items() if value is not None}
        result = subprocess.run(
            [
                "bash",
                str(SAMPLE / "tests" / "run_suites.sh"),
                str(tmp_path / "jobs"),
                str(tmp_path / "reward.json"),
                str(tmp_path / "summary.json"),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        recorded = (
            [json.loads(line) for line in calls.read_text().splitlines()]
            if calls.exists()
            else []
        )
        return result, recorded

    return run


def test_run_suites_runs_harbor_through_the_plugin_and_scores(stub_harbor, tmp_path):
    result, calls = stub_harbor(
        RSI_HARBOR_SUITES="tb2,compose,build", RSI_HARBOR_TB2_TASKS="fix-git,regex-log"
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": 1.0}
    tests = SAMPLE / "tests"
    expected = [
        (suite, path, agent)
        for suite, path in (
            ("tb2", tests / "tb2-subset"),
            ("compose", tests / "compose-sidecar"),
            ("build", tests / "fix-git-build"),
        )
        for agent in ("oracle", "nop")
    ]
    assert len(calls) == len(expected)
    for call, (suite, path, agent) in zip(calls, expected, strict=True):
        argv = call["argv"]
        assert argv[:7] == [
            "run",
            "--env",
            PLUGIN,
            "--agent",
            agent,
            "--path",
            str(path),
        ]
        assert argv[argv.index("--job-name") + 1] == f"{suite}-{agent}"
        assert call["PYTHONPATH"] == "/run/rsi-harness/sandbox/py"
        assert call["HARBOR_TELEMETRY"] == "0"
        includes = [
            argv[i + 1] for i, a in enumerate(argv) if a == "--include-task-name"
        ]
        assert includes == (["fix-git", "regex-log"] if suite == "tb2" else [])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["suites"]["tb2"]["tasks"] == ["fix-git", "regex-log"]
    assert summary["suites"]["compose"]["tasks"] == ["rsi/harbor-compose-sidecar"]

    # Harbor 0.21.0's own CLI resolves this argv to the plugin import path.
    printed = subprocess.run(
        [str(Path(sys.executable).parent / "harbor"), *calls[1]["argv"]]
        + ["--print-config"],
        capture_output=True,
        text=True,
        env={**os.environ, "HARBOR_TELEMETRY": "0"},
        check=True,
    )
    config = json.loads(printed.stdout)
    assert config["environment"] == {"import_path": PLUGIN}
    assert config["agents"] == [{"name": "nop"}]
    assert config["datasets"] == [
        {"path": str(tests / "tb2-subset"), "task_names": ["fix-git", "regex-log"]}
    ]


def test_a_nop_trial_that_scores_makes_the_reward_zero(stub_harbor, tmp_path):
    result, _ = stub_harbor(RSI_HARBOR_SUITES="compose", STUB_NOP_REWARD="1")

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": 0.0}


def test_run_suites_needs_the_sandbox_endpoint(stub_harbor):
    result, calls = stub_harbor(RSI_SANDBOX_PYTHONPATH=None)

    assert result.returncode != 0 and calls == []
    assert "no sandbox endpoint" in result.stderr


def test_the_judge_procedure_writes_its_reward_to_the_verifier_logs():
    text = (SAMPLE / "tests" / "test.sh").read_text()
    assert (
        'bash "$here/run_suites.sh" /logs/verifier/harbor-jobs \\\n'
        "    /logs/verifier/reward.json /logs/verifier/harbor-summary.json"
    ) in text
    runner = (SAMPLE / "tests" / "run_suites.sh").read_text()
    assert (
        "PYTHONPATH=$RSI_SANDBOX_PYTHONPATH harbor run \\\n"
        "            --env rsi_sandbox_harbor:ManagedSandboxEnvironment"
    ) in runner


# -- harbor_reward ----------------------------------------------------------------


def write_trial(job: Path, task: str, reward=None, error=None, trial="x") -> None:
    trial = job / f"{task}__{trial}"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "trial_name": trial.name,
                "exception_info": error,
                "verifier_result": None if reward is None else {"rewards": reward},
            }
        )
    )


def test_suite_tasks_are_one_task_or_a_dataset(tmp_path):
    module = harbor_reward()
    tests = SAMPLE / "tests"

    # Harbor's names: [task] name when the task has one, else the directory.
    assert module.suite_tasks(tests / "compose-sidecar") == [
        "rsi/harbor-compose-sidecar"
    ]
    assert module.suite_tasks(tests / "fix-git-build") == ["fix-git-build"]
    assert module.suite_tasks(tests / "tb2-subset") == sorted(TB2_TASKS)
    assert module.suite_tasks(tests / "tb2-subset", ["regex-log"]) == ["regex-log"]
    with pytest.raises(ValueError, match="no task named nope"):
        module.suite_tasks(tests / "tb2-subset", ["nope"])


@pytest.mark.parametrize(
    "break_it",
    [
        "missing nop job",
        "oracle zero",
        "nop error",
        "extra trial",
        "unreadable",
        "wrong task",
        "duplicate task",
        "reward not a number",
    ],
)
def test_anything_but_oracle_one_and_nop_zero_scores_zero(tmp_path, break_it):
    module = harbor_reward()
    oracle, nop = tmp_path / "s-oracle", tmp_path / "s-nop"
    # As many trials as expected tasks, but not the expected tasks.
    ran = {"wrong task": ("a", "c"), "duplicate task": ("a", "a")}.get(
        break_it, ("a", "b")
    )
    for index, task in enumerate(ran):
        reward = {"oracle zero": 0.0, "reward not a number": "yes"}.get(break_it, 1)
        write_trial(oracle, task, {"reward": reward}, trial=str(index))
        if break_it != "missing nop job":
            error = {"exception_type": "RuntimeError", "exception_message": "x"}
            write_trial(
                nop,
                task,
                {"reward": 0},
                error if break_it == "nop error" and task == "b" else None,
                trial=str(index),
            )
    (oracle / "result.json").write_text("{}")  # the job's own result
    if break_it == "extra trial":
        write_trial(oracle, "c", {"reward": 1})
    if break_it == "unreadable":
        (oracle / "a__0" / "result.json").write_text("{")

    summary = module.score(tmp_path, {"s": ["a", "b"]})

    assert summary["reward"] == 0.0
    whole = module.score(tmp_path, {})
    assert whole["reward"] == 0.0  # no suite is no pass either


def test_every_expected_trial_right_scores_one(tmp_path):
    module = harbor_reward()
    for task in ("a", "b"):
        write_trial(tmp_path / "s-oracle", task, {"reward": 1})
        write_trial(tmp_path / "s-nop", task, {"reward": 0.0})

    summary = module.score(tmp_path, {"s": ["a", "b"]})

    assert summary["reward"] == 1.0
    assert summary["suites"]["s"]["oracle"]["trials"] == 2
