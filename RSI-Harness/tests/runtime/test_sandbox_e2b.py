"""The E2B env backend over an in-process fake of the SDK (no account).

The broker, journal, quotas, exec pump and stage spool are the real ones;
only the E2B SDK surface (tests/e2b_fake.py) is faked. Its sandboxes are
directories and its commands real local processes, so execs, signals and
tar copies run for real.
"""

import base64
import hashlib
import io
import json
import logging
import tarfile
import threading
import time
from dataclasses import dataclass

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime import sandbox_e2b
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_contracts import (
    EnvE2BHost,
    EnvHostPolicy,
    SandboxError,
    SandboxOwner,
)
from rsi_harness.runtime.sandbox_e2b import (
    E2BTemplates,
    e2b_env_runtime,
    parse_image_env,
    read_api_key,
    refuse_unsupported,
    sandbox_metadata,
    template_name,
    template_shape,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    SandboxEnvLease,
    SandboxEnvServiceLease,
    env_container_name,
    parse_env_spec,
)
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from tests.e2b_fake import FakeE2B
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_production import composed  # noqa: F401
from tests.sandbox_helpers import (
    FakeSandboxBackend,
    env_policy_toml,
    load_policy_text,
    make_env_task,
)

MIB = 1024**2
IMAGE = "busybox:1.37.0"
REFERENCE = "docker.io/library/busybox:1.37.0"
KEY_ENV = "RSI_TEST_E2B_KEY"
E2B_KEY = f'api_key_env = "{KEY_ENV}"'
TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
[metadata.rsi_harness.sandbox.environments.work.limits]
max_envs_live = 3
max_exec_output_bytes = 65536
[metadata.rsi_harness.sandbox.environments.judge]
network = ["public", "none"]
pull = true
"""


def e2b_policy_text(*, e2b='api_key_env = "RSI_TEST_E2B_KEY"', **options):
    text = env_policy_toml(**options)
    text = text.replace(
        "no_new_privileges = true\n", 'no_new_privileges = true\nbackend = "e2b"\n', 1
    )
    return text.replace(
        "\n[environments.judge]",
        f"\n[environments.host.e2b]\n{e2b}\n\n[environments.judge]",
        1,
    )


def e2b_grant(tmp_path, task=TASK, **options):
    return resolve_env_grant(
        make_env_task(task),
        load_policy_text(tmp_path, e2b_policy_text(**options)),
        builder_images={},
        parent_cpus=1,
        parent_memory_mb=256,
    )


@dataclass
class Kit:
    broker: SandboxBroker
    fake: FakeE2B
    store: LeaseStore
    grant: object

    @property
    def envs(self):
        return self.broker.envs


def build_kit(tmp_path, **options):
    fake = FakeE2B(tmp_path / "e2b")
    grant = e2b_grant(tmp_path, **options)
    runtime = e2b_env_runtime(
        fake, spool_root=tmp_path / "spool", host=grant.environments.host
    )
    store = LeaseStore(tmp_path / "leases")
    journal = SandboxJournal(authority(store))
    broker = SandboxBroker(
        grant, FakeSandboxBackend(), journal, time.monotonic, envs=runtime
    )
    return Kit(broker, fake, store, grant)


@pytest.fixture
def kit(tmp_path):
    made = build_kit(tmp_path)
    try:
        yield made
    finally:
        try:
            made.broker.close()
        finally:
            made.fake.shutdown()


def open_work(kit):
    session = kit.broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
    )
    kit.broker.activate_work(time.monotonic() + 600)
    return session


def open_round(kit, round_id="r1"):
    session = kit.broker.open_judge(
        SandboxOwner(run_id="run-1", task_id="task", phase="judge", round_id=round_id),
        None,
    )
    kit.broker.activate_judge(time.monotonic() + 600)
    return session


def pull_view(kit, session, request_id="pull", ref=IMAGE):
    job_id = kit.broker.image_pull(session.credential, ref, "missing", request_id)[
        "job_id"
    ]
    kit.envs.images.jobs[job_id].thread.join(10)
    return kit.broker.job_wait(session.credential, job_id, 0, 0)


def pull(kit, session, request_id="pull", ref=IMAGE):
    view = pull_view(kit, session, request_id, ref)
    assert view["state"] == "succeeded", view
    return view["result"]["image"]["handle"]


def single(handle, network="public", **service):
    return {
        "version": 1,
        "network": network,
        "lifetime_sec": 600,
        "disk_mb": 64,
        "services": {
            "main": {
                "image": handle,
                "command": ["sleep", "600"],
                "cpus": 0.5,
                "memory_mb": 64,
                **service,
            }
        },
    }


def ready(kit, session, handle, request_id="env", spec=None):
    created = kit.broker.env_create(
        session.credential, spec or single(handle), request_id
    )
    env_id = created["env_id"]
    kit.broker.env_start(session.credential, env_id, 30, request_id + "-start")
    starter = kit.envs._envs[env_id].starter
    if starter is not None:
        starter.join(10)
    assert kit.broker.env_status(session.credential, env_id)["state"] == "ready"
    return env_id


def sandbox_of(kit, env_id):
    return kit.fake.sandboxes[kit.envs._envs[env_id].lease.services[0].sandbox_id]


def start_exec(kit, session, env_id, argv, request_id="x1", **fields):
    values = dict(cwd=None, env={}, user=None, timeout_sec=None, merge_stderr=False)
    values.update(fields)
    return kit.broker.exec_start(
        session.credential, env_id, "main", list(argv), request_id=request_id, **values
    )["exec_id"]


def wait_exec(kit, session, exec_id, until=None, timeout=20.0):
    """Read the exec to ``until`` (default: its end); output accumulated."""
    until = until or (lambda view: view["state"] != "running")
    out, err = b"", b""
    end = time.monotonic() + timeout
    while True:
        view = kit.broker.exec_wait(
            session.credential, exec_id, len(out), len(err), 0, MIB
        )
        out += base64.b64decode(view["stdout_b64"])
        err += base64.b64decode(view["stderr_b64"])
        view = {**view, "stdout": out, "stderr": err}
        if until(view):
            return view
        assert time.monotonic() < end, view
        time.sleep(0.05)


def run(kit, session, env_id, argv, request_id="x1", **fields):
    view = wait_exec(
        kit, session, start_exec(kit, session, env_id, argv, request_id, **fields)
    )
    return view


def stage(kit, session, data, request_id="stage"):
    return kit.broker.stage_put(
        session.credential,
        None,
        0,
        True,
        hashlib.sha256(data).hexdigest(),
        request_id,
        data,
    )["stage_id"]


def tar_of(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o640
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def until_true(check, timeout=10.0):
    end = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < end
        time.sleep(0.02)


# -- templates ---------------------------------------------------------------------


def test_template_names_follow_the_digest_the_rounded_shape_and_the_recipe(
    monkeypatch,
):
    digest = "sha256:" + "d" * 64
    assert template_shape(0.5, 1000) == (1, 1024)
    assert template_shape(2, 2048) == (2, 2048)
    assert template_shape(2.01, 2049) == (4, 2560)
    assert template_shape(3, 512) == (4, 512)
    assert template_shape(4.5, 512) == (6, 512)
    name = template_name("rsi", REFERENCE, 1, 1024)
    assert name == template_name("rsi", REFERENCE, 1, 1024)
    assert name.startswith("rsi-") and len(name) == 24
    # By digest: the same image under two names is one template.
    assert template_name("rsi", f"docker.io/a/x@{digest}", 1, 1024) == template_name(
        "rsi", f"ghcr.io/b/y@{digest}", 1, 1024
    )
    others = {
        template_name("rsi", "docker.io/library/busybox:1.36", 1, 1024),
        template_name("rsi", REFERENCE, 2, 1024),
        template_name("rsi", REFERENCE, 1, 1536),
        template_name("team", REFERENCE, 1, 1024),
    }
    assert name not in others and len(others) == 4
    monkeypatch.setattr(sandbox_e2b, "RECIPE", "r2")
    assert template_name("rsi", REFERENCE, 1, 1024) != name


def test_a_pull_builds_the_template_once_and_reuses_it(kit):
    work = open_work(kit)
    view = pull_view(kit, work)
    assert view["state"] == "succeeded"
    # The phase's per-container shape (4 cpus, 8192 MiB), the image as asked.
    [(name, image, cpus, memory)] = kit.fake.builds
    assert (image, cpus, memory) == (REFERENCE, 4, 8192)
    assert name == template_name("rsi", REFERENCE, 4, 8192)
    assert f"building template {name}" in view["log"]
    assert "Step 1/2: FROM" in view["log"]
    image_view = view["result"]["image"]
    assert image_view["os"] == "linux" and image_view["bytes"] == 0

    pull(kit, work, "again")
    assert len(kit.fake.builds) == 1
    # Known to this broker now: not even asked again.
    assert [call for call in kit.fake.calls if call[0] == "template_exists"] == [
        ("template_exists", name)
    ]


def test_a_template_e2b_already_has_is_used_without_a_build(kit):
    name = template_name("rsi", REFERENCE, 4, 8192)
    kit.fake.templates[name] = {"image": REFERENCE}
    work = open_work(kit)
    pull(kit, work)
    assert kit.fake.builds == []


def test_concurrent_ensures_of_one_name_build_once(tmp_path):
    fake = FakeE2B(tmp_path / "e2b")
    fake.build_gate = threading.Event()
    templates = E2BTemplates(fake, "rsi")
    results = []

    def ensure():
        results.append(templates.ensure("rsi-x", REFERENCE, 1, 512))

    threads = [threading.Thread(target=ensure) for _ in range(4)]
    for thread in threads:
        thread.start()
    until_true(lambda: len(fake.builds) == 1)
    time.sleep(0.1)
    fake.build_gate.set()
    for thread in threads:
        thread.join(10)
    assert len(fake.builds) == 1
    assert sorted(results) == [False, False, False, True]


def test_a_failed_build_fails_the_pull_and_is_retried_by_the_next(kit):
    kit.fake.build_error = RuntimeError("unsupported distribution")
    work = open_work(kit)
    view = pull_view(kit, work)
    assert view["state"] == "failed"
    assert view["error"]["kind"] == "image"
    assert "unsupported distribution" in view["error"]["message"]
    assert kit.broker.image_list(work.credential)["images"] == []
    # No failure cache: the next pull builds again.
    kit.fake.build_error = None
    pull(kit, work, "retry")
    assert len(kit.fake.builds) == 2


def test_the_recorded_image_env_drops_what_envd_and_the_shell_add():
    raw = (
        b"PATH=/opt/bin:/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
        b":/sbin:/bin\0HOME=/root\0USER=root\0LOGNAME=root\0PWD=/\0SHLVL=1\0"
        b"A=1=2\0EMPTY=\0E2B_SANDBOX=true\0E2B_SANDBOX_ID=ibuild\0"
        b"E2B_TEMPLATE_ID=tbuild\0E2B_EVENTS_ADDRESS=http://192.0.2.1\0"
    )
    assert parse_image_env(raw) == {
        "PATH": "/opt/bin:/usr/bin",
        "A": "1=2",
        "EMPTY": "",
    }


def test_envs_get_the_images_workdir_user_and_entrypoint_from_its_registry(kit):
    # E2B's build keeps only the image's ENV (TB2's fix-git has
    # WORKDIR /app/personal-site, which its test.sh requires).
    kit.fake.image_configs[REFERENCE] = {
        "WorkingDir": "/app/site/",
        "User": "agent",
        "Entrypoint": ["env", "FROM_ENTRYPOINT=1"],
        "Cmd": ["sleep", "1"],
    }
    work = open_work(kit)
    image = pull_view(kit, work)["result"]["image"]
    assert (image["workdir"], image["user"], image["entrypoint"], image["cmd"]) == (
        "/app/site/",
        "agent",
        ["env", "FROM_ENTRYPOINT=1"],
        ["sleep", "1"],
    )
    handle = image["handle"]
    env_id = ready(kit, work, handle)
    service = next(item for item in kit.fake.started if item[1][0] == "setsid")
    # As the Engine merges: the image's entrypoint before the spec's command,
    # the cleaned WORKDIR, the image's USER.
    assert service[1] == ("setsid", "env", "FROM_ENTRYPOINT=1", "sleep", "600")
    assert service[3:] == ("/app/site", "agent")
    view = run(kit, work, env_id, ["pwd"])
    assert view["stdout"].endswith(b"/app/site\n"), view
    assert kit.fake.started[-1][3:] == ("/app/site", "agent")
    # The spec's fields win; an entrypoint of its own clears the image's Cmd.
    spec = single(handle, working_dir="/srv", user="root", entrypoint=["sleep"])
    spec["services"]["main"]["command"] = ["300"]
    ready(kit, work, handle, "own", spec)
    service = [item for item in kit.fake.started if item[1][0] == "setsid"][-1]
    assert service[1] == ("setsid", "sleep", "300")
    assert service[3:] == ("/srv", "root")
    # Neither command nor entrypoint: the image's entrypoint and Cmd.
    spec = single(handle)
    del spec["services"]["main"]["command"]
    ready(kit, work, handle, "image-cmd", spec)
    service = [item for item in kit.fake.started if item[1][0] == "setsid"][-1]
    assert service[1] == ("setsid", "env", "FROM_ENTRYPOINT=1", "sleep", "1")
    # One registry read per image and broker.
    assert [call[0] for call in kit.fake.calls].count("image_config") == 1


def test_a_pull_fails_without_the_image_config_and_builds_nothing(kit):
    kit.fake.fail["image_config"] = RuntimeError("registry unreachable")
    work = open_work(kit)
    view = pull_view(kit, work)
    assert view["state"] == "failed" and view["error"]["kind"] == "image", view
    assert "registry unreachable" in view["error"]["message"]
    assert kit.fake.builds == []
    pull(kit, work, "retry")


def test_registry_image_config_reads_the_amd64_config_anonymously(monkeypatch):
    httpx = pytest.importorskip("httpx")
    config = {
        "config": {
            "Env": ["PATH=/usr/bin"],
            "WorkingDir": "/app/personal-site",
            "Cmd": ["bash"],
            "Labels": {"a": "b"},
            "User": "",
        }
    }
    blob = json.dumps(config).encode()
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    amd64 = "sha256:" + "a" * 64
    base = "https://registry-1.docker.io/v2/alexgshaw/fix-git"
    seen = []

    def answer(request):
        url = str(request.url)
        seen.append((url, request.headers.get("authorization")))
        if url.startswith("https://auth.docker.io/token"):
            assert request.url.params["scope"] == "repository:alexgshaw/fix-git:pull"
            return httpx.Response(200, json={"token": "t0k"})
        if url.startswith("https://cdn.example/"):
            return httpx.Response(200, content=served)
        if request.headers.get("authorization") != "Bearer t0k":
            challenge = (
                'Bearer realm="https://auth.docker.io/token",'
                'service="registry.docker.io",'
                'scope="repository:alexgshaw/fix-git:pull"'
            )
            return httpx.Response(401, headers={"www-authenticate": challenge})
        if url == f"{base}/manifests/20251031":
            platforms = [
                ("linux", "arm64", "sha256:" + "b" * 64),
                ("linux", "amd64", amd64),
            ]
            return httpx.Response(
                200,
                json={
                    "manifests": [
                        {"digest": d, "platform": {"os": o, "architecture": a}}
                        for o, a, d in platforms
                    ]
                },
            )
        if url == f"{base}/manifests/{amd64}":
            return httpx.Response(200, json={"config": {"digest": digest}})
        if url == f"{base}/blobs/{digest}":
            return httpx.Response(307, headers={"location": "https://cdn.example/b"})
        return httpx.Response(404)

    real = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **options: real(transport=httpx.MockTransport(answer), **options),
    )
    served = blob
    found = sandbox_e2b.registry_image_config("docker.io/alexgshaw/fix-git", "20251031")
    assert found == {
        "Env": ["PATH=/usr/bin"],
        "WorkingDir": "/app/personal-site",
        "Cmd": ["bash"],
    }
    # The token never goes to the blob's CDN.
    assert ("https://cdn.example/b", None) in seen
    served = blob + b" "
    with pytest.raises(InfrastructureError, match="does not match its digest"):
        sandbox_e2b.registry_image_config("docker.io/alexgshaw/fix-git", "20251031")


# -- envs ---------------------------------------------------------------------------


def test_env_lifecycle_runs_one_sandbox_with_metadata_network_and_timeout(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    public = ready(kit, work, handle)
    private = ready(kit, work, handle, "private", single(handle, network="none"))

    box = sandbox_of(kit, public)
    assert box.allow_internet and not sandbox_of(kit, private).allow_internet
    assert box.template == template_name("rsi", REFERENCE, 4, 8192)
    assert box.metadata == {
        "rsi_run_id": "run-1",
        "rsi_task_id": "task",
        "rsi_phase": "work",
        "rsi_role": "sandbox-env",
        "rsi_env_id": public,
        "rsi_service": "main",
    }
    # The dead-man switch: E2B kills it a minute after the env's deadline.
    assert 600 <= box.timeout <= 661
    # The service command runs in its own group, as root, in /.
    service = next(item for item in kit.fake.started if item[1][0] == "setsid")
    assert service[1] == ("setsid", "sleep", "600") and service[3:] == ("/", "root")

    lease = kit.envs._envs[public].lease
    assert lease.backend == "e2b" and lease.network_name is None
    assert lease.services[0].sandbox_id == box.sandbox_id
    status = kit.broker.env_status(work.credential, public)
    assert status["services"]["main"]["state"] == "running"
    listing = kit.broker.env_list(work.credential)["envs"]
    assert {item["env_id"] for item in listing} == {public, private}
    capabilities = kit.broker.capabilities(work.credential)["environments"]
    assert capabilities["backend"] == "e2b"

    kit.broker.env_destroy(work.credential, public)
    assert box.state is None
    assert kit.broker.env_status(work.credential, public)["state"] == "removed"
    assert [env.env_id for env in kit.broker.journal.envs()] == [private]


def test_stop_service_ends_the_service_and_refuses_execs(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    answer = kit.broker.env_stop_service(work.credential, env_id, "main", 2, "stop")
    assert answer["state"] == "exited" and answer["exit_code"] == 143
    status = kit.broker.env_status(work.credential, env_id)
    assert status["services"]["main"]["state"] == "exited"
    with pytest.raises(SandboxError, match="not running"):
        start_exec(kit, work, env_id, ["true"])


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (
            lambda spec: spec.update(network="allowlist", allowlist=["example.com"]),
            "spec.network",
        ),
        (
            lambda spec: spec["services"].update(
                db={
                    "image": spec["services"]["main"]["image"],
                    "cpus": 1,
                    "memory_mb": 64,
                }
            ),
            "spec.services",
        ),
        (lambda spec: spec.update(volumes={"data": {}}), "spec.volumes"),
        (
            lambda spec: spec["services"]["main"].update(
                tmpfs={"/scratch": 8}, memory_mb=128
            ),
            "spec.services.main.tmpfs",
        ),
        (
            lambda spec: spec["services"]["main"].update(read_only=True),
            "spec.services.main.read_only",
        ),
        (
            lambda spec: spec["services"]["main"].update(
                healthcheck={"test": ["CMD", "true"]}
            ),
            "spec.services.main.healthcheck",
        ),
    ],
)
def test_what_one_sandbox_cannot_express_is_refused(change, field):
    spec = single("i" + "a" * 32)
    change(spec)
    with pytest.raises(SandboxError) as refused:
        refuse_unsupported(parse_env_spec(spec))
    assert refused.value.code == "unsupported"
    assert refused.value.field == field
    assert "unsupported on the e2b backend" in str(refused.value)


def test_a_compose_env_is_refused_at_create_and_charges_nothing(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    spec = single(handle)
    spec["services"]["db"] = {"image": handle, "cpus": 0.5, "memory_mb": 64}
    with pytest.raises(SandboxError, match="unsupported on the e2b backend"):
        kit.broker.env_create(work.credential, spec, "compose")
    assert kit.fake.live() == []
    usage = kit.broker.capabilities(work.credential)["environments"]["usage"]
    assert usage["envs_live"] == 0


def test_image_builds_are_refused_on_the_e2b_backend(kit, tmp_path):
    work = open_work(kit)
    with pytest.raises(SandboxError) as refused:
        kit.broker.image_build(
            work.credential,
            stage_id="s" + "0" * 32,
            dockerfile=None,
            dockerfile_inline="FROM busybox\n",
            target=None,
            build_args={},
            labels={},
            no_cache=False,
            network="none",
            timeout_sec=60,
            request_id="build",
        )
    assert refused.value.code == "unsupported"
    assert "unsupported on the e2b backend" in str(refused.value)
    # A task asking for builds is refused when the run is set up.
    (tmp_path / "policy").mkdir()
    with pytest.raises(SetupError, match="unsupported on the e2b backend"):
        e2b_grant(
            tmp_path / "policy",
            task=TASK.replace("pull = true", "pull = true\nbuild = true", 1),
        )


# -- execs --------------------------------------------------------------------------


def test_exec_output_streams_before_the_command_ends(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    exec_id = start_exec(
        kit, work, env_id, ["sh", "-c", "echo one; echo err >&2; sleep 1; echo two"]
    )
    early = wait_exec(kit, work, exec_id, until=lambda view: view["stdout"])
    assert early["state"] == "running" and early["stdout"] == b"one\n"
    final = wait_exec(kit, work, exec_id)
    assert (final["state"], final["exit_code"]) == ("exited", 0)
    assert final["stdout"] == b"one\ntwo\n" and final["stderr"] == b"err\n"


def test_execs_get_the_image_env_spec_env_cwd_and_user(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    env_id = ready(
        kit, work, handle, spec=single(handle, env={"SPEC": "spec"}, working_dir="/srv")
    )
    kit.broker.copy_in(
        work.credential,
        env_id,
        "main",
        "/srv/work",
        stage(kit, work, tar_of({"seed": b""})),
        "seed",
    )
    script = 'echo "$IMAGE_ONLY $SPEC $EXEC ${HOME:-nohome}"; echo "$PATH"; pwd'
    view = run(
        kit, work, env_id, ["sh", "-c", script], cwd="/srv/work", env={"EXEC": "x"}
    )
    lines = view["stdout"].decode().splitlines()
    assert lines[0] == "from-image spec x nohome"
    # The image's PATH, without the build's fallback suffix.
    assert lines[1] == "/usr/local/bin:/usr/bin:/bin"
    assert lines[2].endswith("/srv/work")
    started = kit.fake.started[-1]
    assert started[1] == ("setsid", "sh", "-c", script) and started[4] == "root"
    # Numeric users map through the image's passwd; 0 is root.
    run(kit, work, env_id, ["true"], "as-agent", user="1000")
    assert kit.fake.started[-1][4] == "agent"
    run(kit, work, env_id, ["true"], "as-root", user="0")
    assert kit.fake.started[-1][4] == "root"
    with pytest.raises(SandboxError) as refused:
        start_exec(kit, work, env_id, ["true"], "nobody", user="4242")
    assert (refused.value.code, refused.value.field) == ("invalid", "user")


def test_an_exec_past_its_timeout_is_terminated_with_its_group(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    begin = time.monotonic()
    view = run(
        kit, work, env_id, ["sh", "-c", "sleep 30 & sleep 30; wait"], timeout_sec=1
    )
    assert (view["state"], view["reason"]) == ("timed_out", "timeout")
    assert view["signal"] == "TERM"
    assert time.monotonic() - begin < 10
    group = [item for item in kit.fake.started if item[1][:3] == ("kill", "-s", "TERM")]
    assert group and group[0][1][-1].startswith("-")


@pytest.mark.parametrize(
    ("signal_name", "scope"), [("TERM", "group"), ("INT", "process")]
)
def test_exec_kill_signals_the_command(kit, signal_name, scope):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    exec_id = start_exec(kit, work, env_id, ["sleep", "30"])
    answer = kit.broker.exec_kill(work.credential, exec_id, signal_name, scope)
    assert answer["delivered"]
    view = wait_exec(kit, work, exec_id)
    assert (view["state"], view["reason"], view["signal"]) == (
        "killed",
        "interrupt",
        signal_name,
    )
    target = kit.fake.started[-1][1][-1]
    assert target.startswith("-") == (scope == "group")


def test_execs_see_envds_values_for_the_live_sandbox_not_the_builds(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    view = run(kit, work, env_id, ["sh", "-c", 'echo "$E2B_SANDBOX_ID"'])
    assert view["stdout"].decode().strip() == sandbox_of(kit, env_id).sandbox_id
    assert not any(key.startswith("E2B_") for key in kit.fake.started[-1][2]), (
        "the recorded build sandbox's E2B_* values are not re-applied"
    )


def test_an_exec_cut_by_a_pause_is_followed_after_resume(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    exec_id = start_exec(
        kit, work, env_id, ["sh", "-c", "echo one; sleep 1; echo two; exit 5"]
    )
    wait_exec(kit, work, exec_id, until=lambda view: view["stdout"])
    kit.broker.freeze_work()
    kit.broker.resume_work()
    view = wait_exec(kit, work, exec_id)
    assert (view["state"], view["exit_code"]) == ("exited", 5)
    assert view["stdout"] == b"one\ntwo\n"
    assert any(call[0] == "connect" for call in kit.fake.calls)


def test_a_pause_cut_service_stream_is_followed_after_resume(kit):
    """E2B ends every process stream on a pause: the service's exit, and
    stop_service's TERM to it, must still work after resume."""
    work = open_work(kit)
    handle = pull(kit, work)
    exiting = ready(
        kit,
        work,
        handle,
        "exiting",
        spec=single(handle, command=["sh", "-c", "sleep 1; exit 3"]),
    )
    running = ready(kit, work, handle, "running")
    kit.broker.freeze_work()
    kit.broker.resume_work()

    def service(env_id):
        return kit.broker.env_status(work.credential, env_id)["services"]["main"]

    until_true(lambda: service(exiting)["exit_code"] == 3)
    assert service(exiting)["state"] == "exited"
    assert service(running)["state"] == "running"
    answer = kit.broker.env_stop_service(work.credential, running, "main", 2, "stop")
    assert answer == {"state": "exited", "exit_code": 143}


# -- copies -------------------------------------------------------------------------


def test_copy_in_then_copy_out_round_trips_through_tar(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    payload = bytes(range(256)) * 64
    stage_id = stage(kit, work, tar_of({"dir/a.txt": b"hello\n", "blob": payload}))
    copied = kit.broker.copy_in(
        work.credential, env_id, "main", "/data/in", stage_id, "in"
    )
    assert copied == {"entries": 2, "bytes": 6 + len(payload)}

    found = kit.broker.path_stat(
        work.credential, env_id, "main", "/data/in/dir/a.txt", False
    )
    assert (found["exists"], found["kind"], found["size"], found["mode"]) == (
        True,
        "file",
        6,
        0o640,
    )
    assert found["mtime"]
    assert (
        kit.broker.path_stat(work.credential, env_id, "main", "/data/in/dir", False)[
            "kind"
        ]
        == "dir"
    )
    missing = kit.broker.path_stat(work.credential, env_id, "main", "/nope", True)
    assert missing["exists"] is False

    out = kit.broker.copy_out(
        work.credential, env_id, "main", "/data/in", 10 * MIB, ["*.txt"]
    )
    assert (out["entries"], out["skipped"]) == (3, 1)
    data = kit.broker.stage_get(work.credential, out["stage_id"], 0, 10 * MIB)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = {member.name: member for member in archive.getmembers()}
        assert sorted(members) == ["in", "in/blob", "in/dir"]
        assert archive.extractfile(members["in/blob"]).read() == payload
    with pytest.raises(SandboxError, match="does not exist"):
        kit.broker.copy_out(work.credential, env_id, "main", "/data/none", MIB, [])
    with pytest.raises(SandboxError) as special:
        kit.broker.copy_in(
            work.credential,
            env_id,
            "main",
            "/proc/x",
            stage(kit, work, tar_of({"a": b""}), "s2"),
            "p",
        )
    assert special.value.code == "unsupported"
    # No scratch tar is left behind.
    assert list((sandbox_of(kit, env_id).root / "tmp").iterdir()) == []


def test_tool_install_uploads_the_operators_tmux_once(tmp_path):
    tool = tmp_path / "tmux"
    tool.write_bytes(b"echo fake-tmux\n")
    sha = hashlib.sha256(tool.read_bytes()).hexdigest()
    made = build_kit(tmp_path, tmux={"path": str(tool), "sha256": sha})
    try:
        work = open_work(made)
        env_id = ready(made, work, pull(made, work))
        assert made.broker.capabilities(work.credential)["environments"]["tools"] == [
            "tmux"
        ]
        first = made.broker.tool_install(work.credential, env_id, "main", "tmux")
        assert first == {
            "tool": "tmux",
            "path": "/usr/local/bin/tmux",
            "installed": True,
        }
        found = made.broker.path_stat(
            work.credential, env_id, "main", "/usr/local/bin/tmux", False
        )
        assert (found["kind"], found["mode"]) == ("file", 0o755)
        view = run(made, work, env_id, ["sh", "/usr/local/bin/tmux"])
        assert view["stdout"] == b"fake-tmux\n"
        again = made.broker.tool_install(work.credential, env_id, "main", "tmux")
        assert again["installed"] is False
    finally:
        made.broker.close()
        made.fake.shutdown()


# -- phases -------------------------------------------------------------------------


def test_freeze_pauses_work_sandboxes_and_resume_resumes_them(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    box = sandbox_of(kit, env_id)
    kit.broker.freeze_work()
    assert box.state == "paused"
    assert kit.broker.env_status(work.credential, env_id)["state"] == "paused"
    judge = open_round(kit)
    assert judge is not None
    kit.broker.close_judge()
    box.timeout = 0
    kit.broker.resume_work()
    assert box.state == "running"
    # E2B restarts its timer on a pause: the timeout is set again.
    assert 500 < box.timeout <= 661
    assert kit.broker.env_status(work.credential, env_id)["state"] == "ready"
    assert run(kit, work, env_id, ["echo", "back"])["stdout"] == b"back\n"


@pytest.mark.parametrize("hours", [None, 24])
def test_a_sandbox_timeout_stays_within_the_teams_maximum_length(tmp_path, hours):
    # E2B refuses a timeout above the team's limit (1 h on Hobby): an env
    # that outlives it is still created and resumed, and E2B ends it first.
    options = {} if hours is None else {"e2b": f"{E2B_KEY}\nmax_sandbox_hours = 24"}
    kit = build_kit(tmp_path, **options)
    kit.fake.max_hours = hours or 1
    try:
        work = kit.broker.open_session(
            SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
        )
        kit.broker.activate_work(time.monotonic() + 3 * 3600)
        handle = pull(kit, work)
        env_id = ready(kit, work, handle, spec={**single(handle), "lifetime_sec": 7200})
        box = sandbox_of(kit, env_id)
        wanted = (3600,) if hours is None else range(7100, 7261)
        assert box.timeout in wanted
        kit.broker.freeze_work()
        kit.broker.resume_work()
        assert box.state == "running" and box.timeout in wanted
    finally:
        try:
            kit.broker.close()
        finally:
            kit.fake.shutdown()


def test_close_judge_kills_the_rounds_sandboxes_and_keeps_work(kit):
    work = open_work(kit)
    work_env = ready(kit, work, pull(kit, work))
    kit.broker.freeze_work()
    judge = open_round(kit)
    judge_env = ready(kit, judge, pull(kit, judge, "judge-pull"))
    judge_box = sandbox_of(kit, judge_env)
    assert judge_box.metadata["rsi_phase"] == "judge"
    assert judge_box.metadata["rsi_round_id"] == "r1"
    exec_id = start_exec(kit, judge, judge_env, ["sleep", "30"], "judge-exec")
    assert exec_id

    kit.broker.close_judge()

    assert judge_box.state is None
    assert sandbox_of(kit, work_env).state == "paused"
    assert [env.env_id for env in kit.broker.journal.envs()] == [work_env]
    assert not kit.broker.recovery_required
    kit.broker.resume_work()
    assert sandbox_of(kit, work_env).state == "running"


def test_run_close_kills_every_sandbox(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    ready(kit, work, handle, "one")
    ready(kit, work, handle, "two")
    kit.broker.freeze_work()
    judge = open_round(kit)
    ready(kit, judge, pull(kit, judge, "judge-pull"))
    assert len(kit.fake.live()) == 3
    kit.broker.close()
    assert kit.fake.live() == []
    assert kit.broker.journal.envs() == ()
    assert not kit.broker.recovery_required


# -- recovery -------------------------------------------------------------------------


def _recovery(tmp_path, lease, fake):
    from tests.runtime.test_sandbox_env_docker import EnvWorld
    from tests.runtime.test_sandbox_env_recovery import EnvRecoveryBackend

    store = LeaseStore(tmp_path / "recovery-leases")
    store.write(lease)
    clients = []

    def client(settings):
        clients.append(settings)
        return fake

    manager = RecoveryManager(
        store=store,
        backend=EnvRecoveryBackend(EnvWorld()),
        managed_root=tmp_path / "managed",
        e2b_client=client,
    )
    return manager, store, clients


def _crashed_run(tmp_path):
    """A broker that died with a Work env paused and a Judge env running;
    another run's sandbox lives beside them."""
    from tests.runtime.test_sandbox_env_recovery import resource_lease

    made = build_kit(tmp_path)
    try:
        return _crash(made, resource_lease)
    except BaseException:
        made.fake.shutdown()
        raise


def _crash(made, resource_lease):
    work = open_work(made)
    work_env = ready(made, work, pull(made, work))
    made.broker.freeze_work()
    judge = open_round(made)
    ready(made, judge, pull(made, judge, "judge-pull"))
    other = made.fake.create(
        next(iter(made.fake.templates)),
        timeout=60,
        metadata=sandbox_metadata(
            SandboxOwner(run_id="run-2", task_id="task", phase="work"),
            "e" + "2" * 32,
            "main",
        ),
        allow_internet=False,
    )
    assert sandbox_of(made, work_env).state == "paused"
    durable = made.store.read("run-1")
    lease = resource_lease(
        *durable.sandbox_envs,
        sandbox_e2b=made.grant.environments.host.e2b,
    )
    return made, lease, other


@pytest.mark.parametrize("action", ["recover", "cleanup"])
def test_recovery_kills_the_runs_sandboxes_by_metadata(tmp_path, action):
    made, lease, other = _crashed_run(tmp_path)
    try:
        assert len(lease.sandbox_envs) == 2
        manager, store, clients = _recovery(tmp_path, lease, made.fake)

        getattr(manager, action)("run-1")

        assert [box.sandbox_id for box in made.fake.live()] == [other]
        assert clients == [made.grant.environments.host.e2b]
        assert made.fake.closed
        recovered = store.read("run-1")
        assert recovered.sandbox_envs == ()
        assert not recovered.recovery_required
        assert ("list", {"rsi_run_id": "run-1"}) in made.fake.calls
    finally:
        made.fake.shutdown()


def test_unreachable_e2b_keeps_the_lease_for_another_recovery(tmp_path):
    made, lease, _other = _crashed_run(tmp_path)
    try:
        made.fake.fail["list"] = RuntimeError("e2b unreachable")
        manager, store, _ = _recovery(tmp_path, lease, made.fake)
        with pytest.raises(RuntimeError, match="e2b sandbox cleanup failed"):
            manager.recover("run-1")
        retained = store.read("run-1")
        assert retained.recovery_required
        assert len(retained.sandbox_envs) == 2
        assert len(made.fake.live()) == 3
    finally:
        made.fake.shutdown()


# -- the key and the Docker backend ----------------------------------------------------


KEY = "e2b_" + "k3y" * 12


def test_the_api_key_is_read_from_a_file_or_a_variable_never_echoed(tmp_path):
    path = tmp_path / "e2b.key"
    path.write_text(KEY + "\n")
    assert read_api_key(EnvE2BHost(api_key_file=str(path))) == KEY
    assert read_api_key(EnvE2BHost(api_key_env="K"), {"K": KEY}) == KEY
    for settings, environ in (
        (EnvE2BHost(api_key_env="K"), {}),
        (EnvE2BHost(api_key_file=str(tmp_path / "missing")), {}),
    ):
        with pytest.raises(SetupError, match="e2b") as missing:
            read_api_key(settings, environ)
        assert KEY not in str(missing.value)
    with pytest.raises(ValueError, match="exactly one"):
        EnvE2BHost(api_key_env="K", api_key_file=str(path))


def test_sdk_errors_and_build_logs_never_carry_the_key():
    pytest.importorskip("e2b")
    client = sandbox_e2b.SdkE2BClient(KEY)
    cleaned = client._clean(RuntimeError(f"401 for key {KEY}"))
    assert KEY not in str(cleaned) and "***" in str(cleaned)


def test_the_policy_keeps_the_docker_backend_by_default():
    host = dict(pool_disk_mb=1024, disk_floor_mb=512, disk_hard_floor_mb=256)
    assert EnvHostPolicy(**host).backend == "docker"
    with pytest.raises(ValueError, match="go together"):
        EnvHostPolicy(**host, backend="e2b")
    with pytest.raises(ValueError, match="go together"):
        EnvHostPolicy(**host, e2b=EnvE2BHost(api_key_env="K"))


def test_docker_env_leases_keep_their_bridge_rules():
    owner = SandboxOwner(run_id="run-1", task_id="task", phase="work")
    env_id = "e" + "1" * 32
    service = SandboxEnvServiceLease(
        idx=0,
        name="main",
        planned_name=env_container_name(env_id, 0),
        image="i" + "a" * 32,
        image_id="sha256:" + "b" * 64,
    )
    values = dict(
        owner=owner,
        env_id=env_id,
        spec_sha256="c" * 64,
        created_at=1.0,
        expires_at=2.0,
        network_mode="public",
        cpus_milli=500,
        memory_mb=64,
        disk_mb=64,
        services=(service,),
    )
    with pytest.raises(ValueError, match="private firewalled bridge"):
        SandboxEnvLease(**values)
    lease = SandboxEnvLease(**values, backend="e2b")
    assert lease.backend == "e2b" and lease.network_name is None
    with pytest.raises(ValueError, match="E2B sandbox"):
        SandboxEnvLease(
            **{
                **values,
                "services": (service.model_copy(update={"container_id": "d" * 64}),),
            },
            backend="e2b",
        )


def test_the_key_never_reaches_leases_plans_metadata_logs_or_capabilities(
    composed,  # noqa: F811 (the production fixture, imported above)
    caplog,
):
    from rsi_harness.models import RunRequest
    from tests.runtime.test_sandbox_production import ENV_TASK

    ports, plan, definition, store, mutate, transports = composed
    key_file = ports.data_root.parent / "e2b.key"
    key_file.write_text(KEY)
    definition = definition.model_copy(update={"sandbox": make_env_task(ENV_TASK)})
    ports.definition = definition
    ports.sandbox_policy = load_policy_text(
        ports.data_root.parent,
        e2b_policy_text(e2b=f'api_key_file = "{key_file}"\ndomain = "e2b.example"'),
    )
    fake = FakeE2B(ports.data_root.parent / "e2b")
    received = []

    def client(key, settings):
        received.append((key, settings.domain))
        return fake

    ports.e2b_client_factory = client
    caplog.set_level(logging.DEBUG)
    try:
        ports.bind_sandbox_lease(mutate)
        # Only the control transport: no env Docker client on this backend.
        assert transports == [{"timeout": 5}]
        final = ports.prepare_plan(
            definition,
            plan.images,
            plan.gpu_plan,
            RunRequest(task_dir=definition.source_dir),
            "run-1",
        )
        assert received == [(KEY, "e2b.example")]
        assert KEY in ports.agent_output_secrets  # redacted from agent output
        broker = ports._sandbox_broker
        assert broker.envs._runtime.backend_name == "e2b"
        token = ports.sandbox_lifecycle.prepare_work()
        answer = broker.capabilities(token.environment["RSI_SANDBOX_TOKEN"])
        lease = store.read("run-1")
        assert lease.sandbox_e2b.api_key_file == str(key_file)
        texts = {
            "lease": store.path_for("run-1").read_text(),
            "plan": final.model_dump_json(),
            "capabilities": json.dumps(answer),
            "logs": caplog.text,
        }
        for name, text in texts.items():
            assert KEY not in text, name
        assert answer["environments"]["backend"] == "e2b"
    finally:
        ports.sandbox_lifecycle.close()
        fake.shutdown()


# -- the pinned SDK (offline: its own messages, no network) -------------------------


def _events(pb, oneof, *items):
    return [
        pb.StartResponse(event=pb.ProcessEvent(event=oneof(field, value)))
        for field, value in items
    ]


def test_the_sdk_adapter_speaks_the_pinned_envd_process_rpc():
    pytest.importorskip("e2b")
    from types import SimpleNamespace

    from e2b.envd.process import process_pb as pb
    from packaging.version import Version
    from protobuf import Oneof

    events = _events(
        pb,
        Oneof,
        ("start", pb.ProcessEvent.StartEvent(pid=42)),
        ("data", pb.ProcessEvent.DataEvent(output=Oneof("stdout", b"\xffout"))),
        ("data", pb.ProcessEvent.DataEvent(output=Oneof("stderr", b"err"))),
        (
            "end",
            pb.ProcessEvent.EndEvent(
                exit_code=-1, exited=False, status="signal: terminated"
            ),
        ),
    )
    requests = []

    def start(request, *, headers, timeout_ms):
        requests.append((request, headers, timeout_ms))
        return iter(events)

    client = sandbox_e2b.SdkE2BClient(KEY, domain="e2b.example")
    commands = SimpleNamespace(
        _rpc=SimpleNamespace(start=start), _envd_version=Version("0.5.0")
    )
    client._boxes["sbx"] = SimpleNamespace(commands=commands)

    process = client.start(
        "sbx", ["setsid", "sleep", "1"], envs={"A": "1"}, cwd="/srv", user="agent"
    )

    assert process.pid == 42
    # Raw bytes, and a signal death as the Engine reports it.
    assert list(process) == [(1, b"\xffout"), (2, b"err")]
    assert process.exit_code == 143
    [(request, headers, timeout_ms)] = requests
    config = request.process
    assert (config.cmd, list(config.args), dict(config.envs), config.cwd) == (
        "setsid",
        ["sleep", "1"],
        {"A": "1"},
        "/srv",
    )
    assert timeout_ms is None and "Keepalive-Ping-Interval" in headers
    assert sandbox_e2b.shell_exit_code(3, "exit status 3") == 3
    assert sandbox_e2b.shell_exit_code(-1, "signal: killed") == 137


def test_the_sdk_adapter_creates_private_sandboxes_with_the_key_and_domain():
    pytest.importorskip("e2b")
    from types import SimpleNamespace

    created = []

    class Sandbox:
        @staticmethod
        def create(template, **options):
            created.append((template, options))
            return SimpleNamespace(sandbox_id="sbx-1")

    client = sandbox_e2b.SdkE2BClient(KEY, domain="e2b.example")
    client._e2b = SimpleNamespace(Sandbox=Sandbox)
    sandbox_id = client.create(
        "rsi-x", timeout=660, metadata={"rsi_run_id": "run-1"}, allow_internet=False
    )
    assert sandbox_id == "sbx-1"
    [(template, options)] = created
    assert template == "rsi-x"
    assert options["api_key"] == KEY and options["domain"] == "e2b.example"
    assert options["timeout"] == 660 and options["allow_internet_access"] is False
    assert options["network"] == {"allow_public_traffic": False}
    assert options["lifecycle"] == {"on_timeout": "kill"}
    assert options["metadata"] == {"rsi_run_id": "run-1"}
