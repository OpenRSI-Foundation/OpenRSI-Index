"""Real Docker standing in for a cluster Judge: ``rsi-agent llm`` in a Judge
that is not root, confined by Landlock, with the launch spec the cluster
Engine builds and the Codex CLI copy it verifies on its node.

A cluster Judge is an Apptainer container running as the invoking user;
here a container running as the current uid has the same shape: no root, so
rsi-agent confines each call with Landlock. The Codex CLI is the stand-in
app-server, installed as the Engine installs the pinned one (copied and
checked against the expected SHA-256) and bound read-only at
/opt/rsi-agent-codex.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.cluster.bluevela.judge_llm import (
    ClusterJudgeLLM,
    OperatorLoginFile,
    filesystem_type,
    install_pinned_codex,
    resolve_cluster_judge_llm,
    unresolved_endpoints,
)
from rsi_harness.judge_agent import (
    JUDGE_AGENT_TMPFS,
    JUDGE_CODEX_CACHE,
    JUDGE_CODEX_TARGET,
    install_judge_agent,
)
from rsi_harness.models import (
    ContainerMount,
    ContainerSpec,
    ContainerTmpfs,
    JudgeLLMAuth,
    JudgeLLMRequest,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.judge import PUBLIC_JUDGE_TMP
from rsi_harness.runtime.judge_model_access import resolve_judge_model_access
from rsi_loop.harness.config import RSILoopConfig
from tests.integration.test_judge_llm_docker import (
    FAKE_APP_SERVER,
    REFRESH_TOKEN,
    _login,
)
from tests.integration.test_judge_model_access_docker import (
    NON_ROOT,
    PYTHON_IMAGE,
    _docker,
)
from tests.runtime.test_judge_llm import definition

pytestmark = pytest.mark.integration


def _generation_login() -> dict:
    """The synthetic login, its access token carrying a generation claim the
    stand-in answers ``LOGIN`` with."""
    login = _login()
    header, payload, signature = login["tokens"]["access_token"].split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["generation"] = 7
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    login["tokens"]["access_token"] = f"{header}.{encoded.decode()}.{signature}"
    return login


def test_a_landlock_judge_calls_the_judge_llm_through_the_nodes_codex_copy(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    client = _docker(PYTHON_IMAGE)
    task_source = tmp_path / "task-source"
    task_source.mkdir()
    engine_root = tmp_path / "engine-root"
    engine_root.mkdir()
    verifier = engine_root / "verifier"
    verifier.mkdir(mode=0o777)
    verifier.chmod(0o777)
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps(_generation_login()))
    (home / "auth.json").chmod(0o600)
    # What the submit host recorded, and what the job resolves again.
    prepared = ClusterJudgeLLM(
        request=JudgeLLMRequest(auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home),
        codex={"x86_64": FAKE_APP_SERVER},
        login=OperatorLoginFile(
            inode=(home / "auth.json").stat().st_ino,
            uid=(home / "auth.json").stat().st_uid,
            filesystem=filesystem_type(home / "auth.json"),
        ),
    )
    task = definition(judge_model="gpt-judge")
    task.verifier.network.mode = "public"
    judge_llm = resolve_cluster_judge_llm(
        task,
        prepared,
        agent_auth=None,
        config=RSILoopConfig(),
        egress="site",
        pin=unresolved_endpoints,
        environ={},
    )
    access = resolve_judge_model_access(
        task, RSILoopConfig(), None, (), judge_llm, egress="site"
    )
    # Docker admits only a cached copy's shape; on a cluster it is node tmp.
    (engine_root / JUDGE_CODEX_CACHE).mkdir()
    codex_cli = install_pinned_codex(
        FAKE_APP_SERVER,
        engine_root / JUDGE_CODEX_CACHE / "codex-cluster",
        expected=hashlib.sha256(FAKE_APP_SERVER.read_bytes()).hexdigest(),
    )
    agent_dir = install_judge_agent(engine_root)
    runtime = DockerContainerRuntime(
        client,
        run_id=f"cluster-llm-{uuid.uuid4().hex[:8]}",
        task_id="cluster-llm",
        role="judge",
        task_source_dir=task_source,
        allowed_mount_roots=(engine_root,),
        omit_gpu_device_requests_for_tests=True,
        judge_model_access=True,
    )
    judge = runtime.create(
        ContainerSpec(
            image=PYTHON_IMAGE,
            command=("sleep", "infinity"),
            user=NON_ROOT,
            read_only_rootfs=True,
            tmpfs=(
                ContainerTmpfs(
                    target=PurePosixPath("/tests"),
                    options="rw,exec,nosuid,nodev,mode=0700",
                ),
                PUBLIC_JUDGE_TMP,
                *JUDGE_AGENT_TMPFS,
            ),
            mounts=(
                ContainerMount(source=verifier, target=PurePosixPath("/logs/verifier")),
                ContainerMount(
                    source=agent_dir,
                    target=PurePosixPath("/opt/rsi-agent"),
                    read_only=True,
                ),
                ContainerMount(
                    source=agent_dir / "rsi-agent",
                    target=PurePosixPath("/usr/local/bin/rsi-agent"),
                    read_only=True,
                ),
                ContainerMount(
                    source=codex_cli, target=JUDGE_CODEX_TARGET, read_only=True
                ),
            ),
        )
    )
    request.addfinalizer(lambda: runtime.remove(judge))
    container = client.containers.get(judge.container_id)
    container.start()
    runtime.inject_judge_agent(
        judge, access.round_launch(time.time() + 600), owner=NON_ROOT
    )

    code, output = container.exec_run(("rsi-agent", "isolation"), user=NON_ROOT)
    assert code == 0, output
    assert json.loads(output)["mode"] == "landlock"

    script = (
        "set -u; mkdir -p /tmp/q && printf 'IDENTITY\\n' > /tmp/q/p.md && "
        "rsi-agent llm --prompt-file /tmp/q/p.md --output /tmp/q/a.json "
        '--timeout 60; echo "exit $?"; cat /tmp/q/a.json; echo; '
        "printf 'LOGIN\\n' > /tmp/q/l.md && "
        "rsi-agent llm --prompt-file /tmp/q/l.md --output /tmp/q/l.json "
        "--timeout 60 && cat /tmp/q/l.json; echo; "
        "if touch /opt/rsi-agent-codex/x 2>/dev/null; then echo CODEX-WRITABLE; fi; "
        "cat /run/rsi-agent/private/launch.json >/dev/null 2>&1 && echo SPEC-READ"
    )
    code, output = container.exec_run(("sh", "-c", script), user=NON_ROOT)
    text = output.decode()
    assert code == 0, text
    assert "exit 0" in text, text
    identity = json.loads(text.split("exit 0\n", 1)[1].splitlines()[0])
    # The Judge's own uid, confined to its own call's job directory.
    assert identity["uid"] == int(NON_ROOT.split(":")[0])
    assert identity["home"].startswith("/run/rsi-agent/jobs/")
    assert identity["codex_home"] == identity["home"] + "/.codex"
    assert identity["login_mode"] == "0o600"
    # The Judge's copy of the login never holds the refresh token.
    assert '{"generation": 7, "withheld": true}' in text, text
    assert REFRESH_TOKEN not in text
    assert "CODEX-WRITABLE" not in text
    # The Judge procedure (the Verifier user) reads rsi-agent's spec, as for
    # model access; the call it runs does not (checked by Landlock above).
    assert "SPEC-READ" in text
