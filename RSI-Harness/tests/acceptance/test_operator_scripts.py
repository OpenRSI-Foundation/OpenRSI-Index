"""The operator scripts, as far as a non-root user can run them: syntax,
--dry-run plans (every check and command, nothing run or created), the
root refusal, and every test node the root check names exists."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.acceptance import audit

REPO = Path(__file__).parents[2]
OPERATOR = REPO / "scripts" / "operator"
ROOT_CHECK = OPERATOR / "sandbox_root_check.sh"
ACCEPTANCE = OPERATOR / "sandbox_acceptance.sh"
SCENARIOS = (
    "a1",
    "a2",
    "a4",
    "a7-env-create",
    "a7-build",
    "a7-load",
    "a7-paused",
    "a8",
)
# The scripts default to $REPO/.venv; point them at this interpreter and
# its entry point so the tests work in any checkout.
_SIBLING = Path(sys.executable).with_name("rsi-harness")
HARNESS = str(_SIBLING) if _SIBLING.is_file() else shutil.which("rsi-harness")


def script_env() -> dict[str, str]:
    if HARNESS is None:
        pytest.skip("no rsi-harness entry point next to the interpreter or on PATH")
    return {**os.environ, "RSI_PYTHON": sys.executable, "RSI_HARNESS": HARNESS}


def run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
        env=script_env(),
    )


@pytest.mark.parametrize("script", [ROOT_CHECK, ACCEPTANCE])
def test_the_scripts_parse_and_are_executable(script):
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert os.access(script, os.X_OK)


def test_the_root_check_dry_run_prints_every_check_and_runs_none(tmp_path):
    scratch = tmp_path / "scratch"
    result = run(ROOT_CHECK, "--dry-run", "--scratch", str(scratch))

    assert result.returncode == 0, result.stderr
    ids = re.findall(r"^== (R\d+) ", result.stdout, re.MULTILINE)
    # Spec 8 step 5: the acceptance (R10) is part of it, then the audit;
    # R11 (allowlist envs) runs before them.
    assert ids == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R11",
        "R10",
        "R9",
    ]
    rows = re.findall(r"^(R\d+)\s+DRY-RUN\s", result.stdout, re.MULTILINE)
    assert rows == ids
    assert "dry run: nothing was run" in result.stdout
    assert not scratch.exists()


def test_the_root_check_names_only_existing_tests():
    result = run(ROOT_CHECK, "--dry-run", "--scratch", "/nonexistent")
    nodes = set(re.findall(r"(tests/[\w/]+\.py)(?:::(\w+))?", result.stdout))

    assert nodes
    for path, name in nodes:
        module = REPO / path
        assert module.is_file(), path
        if name:
            tree = ast.parse(module.read_text())
            defined = {
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
            assert name in defined, f"{path}::{name}"


def test_the_root_check_filters_and_runs_the_acceptance_unless_skipped():
    result = run(ROOT_CHECK, "--dry-run", "--only", "R3,R10")

    assert re.findall(r"^== (R\d+) ", result.stdout, re.MULTILINE) == ["R3", "R10"]
    assert "sandbox_acceptance.sh --scratch" in result.stdout

    skipped = run(ROOT_CHECK, "--dry-run", "--only", "R10", "--skip-acceptance")
    assert "sandbox_acceptance.sh" not in skipped.stdout
    assert re.search(r"^R10\s+SKIPPED\s.*incomplete", skipped.stdout, re.MULTILINE)


def test_the_root_check_names_the_pidfd_path_of_its_interpreter():
    result = run(ROOT_CHECK, "--dry-run", "--only", "R5")

    assert re.search(
        r"^== interpreter \S+python\S* 3\.\d+\.\d+ (os\.pidfd_open|libc/syscall)",
        result.stdout,
        re.MULTILINE,
    )


@pytest.mark.parametrize(
    ("script", "only"), [(ROOT_CHECK, "R3,R12"), (ACCEPTANCE, "a1,a3")]
)
def test_an_unknown_only_name_is_refused(script, only):
    """--only naming nothing must not end in an empty, passing table."""
    result = run(script, "--dry-run", "--only", only)

    assert result.returncode == 2
    assert "unknown" in result.stderr


def _audit_commands(stdout: str) -> list[list[str]]:
    """Every ``python -m tests.acceptance.audit`` command a dry run prints,
    as the arguments after the module."""
    commands = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("$ ") or "tests.acceptance.audit" not in line:
            continue
        words = shlex.split(line.removeprefix("$ "))
        if "tests.acceptance.audit" in words:
            commands.append(words[words.index("tests.acceptance.audit") + 1 :])
    return commands


@pytest.mark.parametrize("script", [ROOT_CHECK, ACCEPTANCE])
def test_every_audit_command_the_scripts_run_parses(script):
    """The exact audit command lines (the placeholders PID and RUN_ID
    stand for what only the run tells) are ones the audit CLI accepts."""
    result = run(script, "--dry-run")
    commands = _audit_commands(result.stdout)

    assert commands
    kinds = set()
    for words in commands:
        words = ["1234" if word == "PID" else word for word in words]
        parsed = audit.parser().parse_args(words)
        kinds.add(parsed.command)
        if parsed.command == "leftovers":
            assert parsed.run_id == "RUN_ID"
    expected = (
        {"host-diff"} if script == ROOT_CHECK else {"watch", "leftovers", "verdict"}
    )
    assert kinds == expected
    if script == ACCEPTANCE:
        strict = [words for words in commands if "--strict" in words]
        assert len(strict) == len(SCENARIOS)


def test_the_acceptance_dry_run_prints_every_scenario_and_runs_none(tmp_path):
    scratch = tmp_path / "scratch"
    result = run(ACCEPTANCE, "--dry-run", "--scratch", str(scratch))

    assert result.returncode == 0, result.stderr
    announced = re.findall(r"^== \[([\w-]+)\] Judge", result.stdout, re.MULTILINE)
    assert tuple(announced) == SCENARIOS
    runs = [line for line in result.stdout.splitlines() if "scripted_cli" in line]
    assert len(runs) == len(SCENARIOS)
    for line in runs:
        assert " -- run " in line and " --agent codex --sandbox-policy " in line
    assert result.stdout.count(f"{HARNESS} recover RUN_ID") == 4
    for moment in ("env_create", "build", "load", "paused"):
        assert f"--kill-at {moment}" in result.stdout
    assert "--timeout 300" in result.stdout
    assert "dry run: nothing was run" in result.stdout
    assert not scratch.exists()


def test_the_acceptance_runs_only_the_chosen_scenarios():
    result = run(ACCEPTANCE, "--dry-run", "--only", "a4,a7-load")

    announced = re.findall(r"^== \[([\w-]+)\] Judge", result.stdout, re.MULTILINE)
    assert announced == ["a4", "a7-load"]


def test_the_swebench_scenario_runs_only_when_named_with_its_own_sample():
    result = run(ACCEPTANCE, "--dry-run", "--only", "swebench")

    assert result.returncode == 0, result.stderr
    announced = re.findall(r"^== \[([\w-]+)\] Judge", result.stdout, re.MULTILINE)
    assert announced == ["swebench"]
    [line] = [line for line in result.stdout.splitlines() if "scripted_cli" in line]
    words = shlex.split(line.strip().removeprefix("$ "))
    sample = REPO / "sample_tasks" / "swebench-in-judge"
    assert words[words.index("run") + 1] == str(sample)
    assert words[words.index("--sandbox-policy") + 1] == str(
        REPO / "sample_tasks" / "swebench-in-judge" / "operator-policy.toml"
    )
    assert words[words.index("--work-files") + 1] == str(sample / "tests")
    assert "RSI_ACCEPTANCE_PROBE=0" in words
    for words in _audit_commands(result.stdout):
        audit.parser().parse_args(["1234" if w == "PID" else w for w in words])
    assert "verdict swebench --dir" in result.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="checks the non-root refusal")
@pytest.mark.parametrize("script", [ROOT_CHECK, ACCEPTANCE])
def test_the_scripts_refuse_to_run_without_root(script, tmp_path):
    scratch = tmp_path / "scratch"
    result = run(script, "--scratch", str(scratch))

    assert result.returncode == 2
    assert "run as root" in result.stderr
    assert not scratch.exists()


# -- the vLLM-in-Judge demo -------------------------------------------------------

VLLM_DEMO = OPERATOR / "vllm_demo.sh"


def test_the_vllm_demo_parses_and_is_executable():
    assert subprocess.run(["bash", "-n", str(VLLM_DEMO)]).returncode == 0
    assert os.access(VLLM_DEMO, os.X_OK)


def test_the_vllm_demo_dry_run_prints_the_run_and_runs_none(tmp_path):
    scratch = tmp_path / "scratch"
    result = run(
        VLLM_DEMO,
        "--dry-run",
        "--scratch",
        str(scratch),
        "--gpus",
        "5",
        "--tasks",
        "rsi/hello-file",
        "--model",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--revision",
        "abc",
    )

    assert result.returncode == 0, result.stderr
    [line] = [line for line in result.stdout.splitlines() if "scripted_cli" in line]
    words = shlex.split(line.strip().removeprefix("$ "))
    assert words[:2] == ["env", "RSI_VLLM_TASKS=rsi/hello-file"]
    assert words[words.index("--work-script") + 1].endswith(
        "sample_tasks/vllm-in-judge/work/agent.sh"
    )
    assert "RSI_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct" in words
    assert "RSI_VLLM_REVISION=abc" in words
    command = words[words.index("--") + 1 :]
    assert command[:4] == [
        "run",
        str(REPO / "sample_tasks" / "vllm-in-judge"),
        "--agent",
        "codex",
    ]
    assert command[command.index("--sandbox-policy") + 1] == str(
        REPO / "sample_tasks" / "vllm-in-judge" / "operator-policy.toml"
    )
    assert command[command.index("--gpus") + 1] == "5"
    assert command[command.index("--data-root") + 1] == f"{scratch}/vllm/data"
    assert f"{HARNESS} cleanup RUN_ID --delete-workspace" in result.stdout
    assert "dry run: nothing was run" in result.stdout
    assert not scratch.exists()


def test_every_audit_command_of_the_vllm_demo_parses():
    commands = _audit_commands(run(VLLM_DEMO, "--dry-run").stdout)

    parsed = [
        audit.parser().parse_args(["1234" if word == "PID" else word for word in words])
        for words in commands
    ]
    assert [item.command for item in parsed] == [
        "watch",
        "leftovers",
        "leftovers",
        "verdict",
    ]
    assert [item.strict for item in parsed[1:3]] == [False, True]
    assert parsed[3].scenario == "vllm"


def test_the_vllm_demo_keeps_the_run_with_keep():
    result = run(VLLM_DEMO, "--dry-run", "--keep")

    assert "rsi-harness cleanup" not in result.stdout
    assert "--strict" not in result.stdout


def test_the_vllm_demo_refuses_an_unknown_argument():
    result = run(VLLM_DEMO, "--dry-run", "--only", "a1")

    assert result.returncode == 2
    assert "unknown argument" in result.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="checks the non-root refusal")
def test_the_vllm_demo_refuses_to_run_without_root(tmp_path):
    scratch = tmp_path / "scratch"
    result = run(VLLM_DEMO, "--scratch", str(scratch))

    assert result.returncode == 2
    assert "run as root" in result.stderr
    assert not scratch.exists()


def fake_host(tmp_path, *, gpu_mib, free_gib, work_image):
    """nvidia-smi, docker and df stand-ins on PATH: one GPU holding
    ``gpu_mib``, ``free_gib`` free under Docker's root, the sample's Work
    image cached or not (the task images always are)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "nvidia-smi": f"""#!/bin/sh
case "$*" in
    *nounits*) echo "{gpu_mib}" ;;
    *) echo "4, GPU-x, {gpu_mib} MiB" ;;
esac
""",
        "docker": f"""#!/bin/sh
[ "$1" = info ] && {{ echo /srv/docker; exit 0; }}
case "$*" in
    *rsi-sample-vllm-in-judge*) exit {0 if work_image else 1} ;;
esac
exit 0
""",
        "df": f"""#!/bin/sh
echo " Avail"
echo " {free_gib}G"
""",
    }
    for name, text in scripts.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    return {**script_env(), "PATH": f"{bin_dir}:{os.environ['PATH']}"}


@pytest.mark.parametrize(
    ("gpu_mib", "free_gib", "work_image", "refusals"),
    [
        (0, 47, True, []),
        (30000, 47, True, ["GPU 4 is busy (30000 MiB used, over 1024 MiB)"]),
        (0, 18, False, ["18 GiB free", "need 20 GiB", "not enough disk"]),
        (0, 18, True, []),
        (0, 12, True, ["need 15 GiB", "not enough disk"]),
    ],
)
def test_the_vllm_demo_preflight_refuses_a_busy_gpu_or_a_full_disk(
    tmp_path, gpu_mib, free_gib, work_image, refusals
):
    env = fake_host(tmp_path, gpu_mib=gpu_mib, free_gib=free_gib, work_image=work_image)
    result = subprocess.run(
        ["bash", str(VLLM_DEMO), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
        env=env,
    )

    # A dry run prints what preflight would refuse and runs nothing.
    assert result.returncode == 0, result.stderr
    assert "under the Docker root /srv/docker" in result.stdout
    for refusal in refusals:
        assert refusal in result.stdout
    if not refusals:
        assert "busy" not in result.stdout and "not enough" not in result.stdout


def _readline(process: subprocess.Popen, *, timeout: float) -> str:
    """One line of stdout, or "" when none comes within ``timeout``."""
    lines: list[str] = []
    reader = threading.Thread(
        target=lambda: lines.append(process.stdout.readline()), daemon=True
    )
    reader.start()
    reader.join(timeout)
    return lines[0] if lines else ""


def _kill_group(process: subprocess.Popen) -> None:
    """Kill the session ``process`` leads, background jobs included."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)


def test_an_interrupted_vllm_demo_stops_its_run_and_prints_the_recovery(tmp_path):
    """The trap of the real script around a stand-in run: a background job
    that, like `rsi-harness run`, never sees the Ctrl-C itself."""
    text = VLLM_DEMO.read_text()
    functions = "\n".join(
        re.search(pattern, text, re.MULTILINE | re.DOTALL)[0]
        for pattern in (r"^show\(\) \{.*?^\}", r"^RUN_PID=$.*?^\}")
    )
    run_id = "0123456789abcdef0123456789abcdef"
    leases = tmp_path / "vllm" / "data" / "leases"
    leases.mkdir(parents=True)
    (leases / f"{run_id}.json").write_text("{}")
    marker = tmp_path / "run-ended"
    harness = f"""
set -uo pipefail
REPO={shlex.quote(str(REPO))}
PY={shlex.quote(sys.executable)}
SCRATCH={shlex.quote(str(tmp_path))}
{functions}
RECOVER=(rsi-harness recover RUN_ID --data-root "$SCRATCH/vllm/data")
CLEANUP=(rsi-harness cleanup RUN_ID --delete-workspace --yes)
trap interrupted INT TERM
"$PY" -c 'import signal, sys, time
signal.signal(signal.SIGTERM, lambda *_: (open(sys.argv[1], "w").close(), sys.exit(0)))
open(sys.argv[1] + ".started", "w").close()
time.sleep(60)' {shlex.quote(str(marker))} &
RUN_PID=$!
until [[ -e {shlex.quote(str(marker))}.started ]]; do
    kill -0 "$RUN_PID" 2> /dev/null || {{ echo "run died before starting"; exit 3; }}
    sleep 0.05
done
echo ready
wait "$RUN_PID"
echo "never interrupted"
"""
    process = subprocess.Popen(
        ["bash", "-c", harness],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO,
        start_new_session=True,
    )
    try:
        line = _readline(process, timeout=60)
        assert line.strip() == "ready", (line, process.poll())
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=60)
    finally:
        _kill_group(process)

    assert process.returncode == 130, stderr
    assert "never interrupted" not in stdout
    # The run got TERM (the Ctrl-C alone never reaches a background job).
    assert marker.exists()
    assert "interrupted: stopping rsi-harness run (pid " in stderr
    assert f"rsi-harness recover {run_id} --data-root" in stderr
    assert f"rsi-harness cleanup {run_id} --delete-workspace --yes" in stderr


def test_the_vllm_demo_usage_names_the_recovery():
    result = run(VLLM_DEMO, "--help")

    assert result.returncode == 0
    assert "rsi-harness recover RUN --data-root" in result.stdout


PREPULL = OPERATOR / "prepull_images.sh"
PREPULL_MANIFEST = REPO / "sample_tasks" / "harbor-in-judge" / "images.manifest"


def fake_registry(tmp_path, *, present=(), flaky=(), broken=()):
    """A docker stand-in on PATH: ``present`` refs are on the host, a
    ``flaky`` ref's first pull fails, a ``broken`` ref's every pull does."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    for ref in present:
        (state / ref.replace("/", "_")).touch()
    root = tmp_path / "docker-root"
    root.mkdir()
    (bin_dir / "docker").write_text(
        f"""#!/bin/bash
state={state}
ref=${{@: -1}}
key=${{ref//\\//_}}
tried=$state/$key.tried
echo "$*" >> {tmp_path}/calls
case "$1" in
    info) echo {root} ;;
    image) [ -e "$state/$key" ] ;;
    pull)
        case " {" ".join(broken)} " in *" $ref "*) exit 1 ;; esac
        case " {" ".join(flaky)} " in
            *" $ref "*) [ -e "$tried" ] || {{ touch "$tried"; exit 1; }} ;;
        esac
        touch "$state/$key" ;;
esac
"""
    )
    (bin_dir / "docker").chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RSI_PREPULL_PAUSE_SEC": "0",
    }


def prepull(env, *args):
    return subprocess.run(
        ["bash", str(PREPULL), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
        env=env,
    )


def test_the_prepull_example_manifest_is_digests_the_broker_accepts():
    from rsi_harness.runtime.sandbox_images import pull_reference

    assert subprocess.run(["bash", "-n", str(PREPULL)]).returncode == 0
    assert os.access(PREPULL, os.X_OK)
    refs = [
        line
        for line in PREPULL_MANIFEST.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert len(refs) == 9
    for ref in refs:
        repository, digest = pull_reference(ref)
        assert repository.startswith("docker.io/") and digest.startswith("sha256:")


def test_the_prepull_skips_present_images_retries_and_lists_failures(tmp_path):
    a, b, c, d = (
        f"docker.io/rsi/{name}@sha256:" + digit * 64
        for name, digit in (("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"))
    )
    manifest = tmp_path / "images.manifest"
    manifest.write_text(f"# set\n{a}\n\n{b}  # flaky\n{c}\n{d}\n")
    env = fake_registry(tmp_path, present=[a], flaky=[b], broken=[c])

    dry = prepull(env, "--dry-run", str(manifest))
    assert dry.returncode == 0, dry.stderr
    assert dry.stdout.count("would pull") == 4
    assert not (tmp_path / "calls").exists()

    result = prepull(env, "--retries", "2", str(manifest))
    assert result.returncode == 1
    assert "pulled 2, already present 1, failed 1" in result.stdout
    assert f"retry 1/2 in 0 s: {b}" in result.stderr
    assert f"failed {c}" in result.stderr
    pulls = [
        line
        for line in (tmp_path / "calls").read_text().splitlines()
        if line.startswith("pull")
    ]
    assert [line.split()[-1] for line in pulls] == [b, b, c, c, c, d]

    again = prepull(env, str(manifest))  # c still fails, the rest are there
    assert "pulled 0, already present 3, failed 1" in again.stdout


@pytest.mark.parametrize(
    "line", ["busybox:1.37.0", "busybox@sha256:abc", "Busybox@sha256:" + "a" * 64]
)
def test_the_prepull_refuses_a_line_that_is_not_a_digest(tmp_path, line):
    manifest = tmp_path / "images.manifest"
    manifest.write_text(line + "\n")
    result = prepull(fake_registry(tmp_path), str(manifest))
    assert result.returncode == 2
    assert "not name@sha256" in result.stderr


def test_the_prepull_dry_run_of_an_empty_manifest_lists_nothing(tmp_path):
    manifest = tmp_path / "images.manifest"
    manifest.write_text("# nothing yet\n\n")
    result = prepull(fake_registry(tmp_path), "--dry-run", str(manifest))
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"0 images in {manifest}\n"


BUILD_TMUX = OPERATOR / "build_static_tmux.sh"


def fake_builder(tmp_path, *, status=0):
    """A docker stand-in on PATH whose ``run`` prints a runnable 'tmux'."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        f"""#!/bin/bash
printf '%s\\n' "$@" > {tmp_path}/args
printf '#!/bin/sh\\necho tmux 3.5a\\n'
exit {status}
"""
    )
    (bin_dir / "docker").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def build_tmux(env, *args):
    return subprocess.run(
        ["bash", str(BUILD_TMUX), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
        env=env,
    )


def test_the_tmux_build_runs_a_pinned_throwaway_container_and_prints_the_hash(
    tmp_path,
):
    assert subprocess.run(["bash", "-n", str(BUILD_TMUX)]).returncode == 0
    assert os.access(BUILD_TMUX, os.X_OK)
    output = tmp_path / "tmux"
    result = build_tmux(fake_builder(tmp_path), str(output))
    assert result.returncode == 0, result.stderr
    data = output.read_bytes()
    assert result.stdout == f"{hashlib.sha256(data).hexdigest()}  {output}\n"
    assert output.stat().st_mode & 0o777 == 0o755
    assert "tmux 3.5a" in result.stderr  # the binary ran on the host
    args = (tmp_path / "args").read_text().splitlines()
    assert args[:2] == ["run", "--rm"]
    image = re.compile(r"alpine:3\.21@sha256:[0-9a-f]{64}")
    assert any(image.fullmatch(arg) for arg in args)
    for pin in ("TMUX_SHA256=", "NCURSES_SHA256=", "FALLBACKS=xterm-256color"):
        assert any(arg.startswith(pin) for arg in args), pin
    # The build script, the last (multi-line) argument of `sh -c`.
    script = (tmp_path / "args").read_text().split("\nsh\n-c\n", 1)[1]
    assert "--enable-static" in script and "sha256sum -c" in script


def test_a_failed_tmux_build_leaves_no_binary(tmp_path):
    output = tmp_path / "tmux"
    result = build_tmux(fake_builder(tmp_path, status=1), str(output))
    assert result.returncode == 1
    assert list(tmp_path.glob("tmux*")) == []
    usage = build_tmux(dict(os.environ))
    assert usage.returncode == 2 and "usage" in usage.stderr
