"""The E2B env backend over an in-process fake of the SDK (no account).

The broker, journal, quotas, exec pump and stage spool are the real ones;
only the E2B SDK surface (tests/remote_fake.py) is faked. Its sandboxes are
directories and its commands real local processes, so execs, signals and
tar copies run for real. The tests marked ``every`` also run over the fake
as Modal's and Daytona's client, without E2B's pids, users and pause,
through sandbox_remote's emulation.
"""

import base64
import hashlib
import importlib.metadata
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
from rsi_harness.runtime.sandbox_archive import StageStore
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_contracts import (
    PROVIDER_CAPACITY,
    EnvE2BHost,
    ProviderCapacityError,
    SandboxError,
    SandboxOwner,
)
from rsi_harness.runtime.sandbox_e2b import (
    RECIPE,
    SDK_VERSION,
    check_e2b_host,
    parse_image_env,
    read_api_key,
    refuse_e2b_environ,
    template_shape,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    parse_env_spec,
)
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from rsi_harness.runtime.sandbox_remote import (
    FREEZE_TRIES,
    FROZEN_RECORD,
    RemoteTemplates,
    built_template_name,
    group_argv,
    read_registry_auth,
    refuse_unsupported,
    remote_env_runtime,
    sandbox_metadata,
    template_name,
)
from tests.remote_fake import FAKES, IMAGE_CONFIG, RemoteFake
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


MODAL_TOKENS = (
    'api_key_id_env = "RSI_TEST_MODAL_ID"\napi_key_env = "RSI_TEST_MODAL_SECRET"'
)
DAYTONA_KEY = 'api_key_env = "RSI_TEST_DAYTONA_KEY"\ntarget = "us"'


def e2b_policy_text(
    *,
    e2b='api_key_env = "RSI_TEST_E2B_KEY"',
    lifetime_sec=3600,
    backend="e2b",
    modal=MODAL_TOKENS,
    daytona=DAYTONA_KEY,
    **options,
):
    # Lifetimes fit in E2B's default one-hour max_sandbox_hours.
    text = env_policy_toml(**options).replace(
        "max_env_lifetime_sec = 14400", f"max_env_lifetime_sec = {lifetime_sec}"
    )
    text = text.replace(
        "no_new_privileges = true\n",
        f'no_new_privileges = true\nbackend = "{backend}"\n',
        1,
    )
    table = {"e2b": e2b, "modal": modal, "daytona": daytona}[backend]
    return text.replace(
        "\n[environments.judge]",
        f"\n[environments.host.{backend}]\n{table}\n\n[environments.judge]",
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
    fake: RemoteFake
    store: LeaseStore
    grant: object

    @property
    def envs(self):
        return self.broker.envs


def build_kit(tmp_path, mode="e2b", **options):
    fake = RemoteFake(tmp_path / mode, **FAKES[mode])
    grant = e2b_grant(tmp_path, backend=mode, **options)
    runtime = remote_env_runtime(
        fake, spool_root=tmp_path / "spool", host=grant.environments.host
    )
    store = LeaseStore(tmp_path / "leases")
    journal = SandboxJournal(authority(store))
    broker = SandboxBroker(
        grant, FakeSandboxBackend(), journal, time.monotonic, envs=runtime
    )
    return Kit(broker, fake, store, grant)


# The generic tests: over E2B's own pids, users and pause, and Modal's and
# Daytona's emulated ones.
every = pytest.mark.parametrize("kit", ["e2b", "modal", "daytona"], indirect=True)


@pytest.fixture
def kit(tmp_path, request):
    made = build_kit(tmp_path, getattr(request, "param", "e2b"))
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


def test_template_names_follow_the_digest_the_rounded_shape_and_the_recipe():
    digest = "sha256:" + "d" * 64
    assert template_shape(0.5, 1000) == (1, 1024)
    assert template_shape(2, 2048) == (2, 2048)
    assert template_shape(2.01, 2049) == (4, 2560)
    assert template_shape(3, 512) == (4, 512)
    assert template_shape(4.5, 512) == (6, 512)
    name = template_name("rsi", REFERENCE, (1, 1024), RECIPE)
    assert name == template_name("rsi", REFERENCE, (1, 1024), RECIPE)
    assert name.startswith("rsi-") and len(name) == 24
    # By digest: the same image under two names is one template.
    assert template_name(
        "rsi", f"docker.io/a/x@{digest}", (1, 1024), RECIPE
    ) == template_name("rsi", f"ghcr.io/b/y@{digest}", (1, 1024), RECIPE)
    others = {
        template_name("rsi", "docker.io/library/busybox:1.36", (1, 1024), RECIPE),
        template_name("rsi", REFERENCE, (2, 1024), RECIPE),
        template_name("rsi", REFERENCE, (1, 1536), RECIPE),
        template_name("team", REFERENCE, (1, 1024), RECIPE),
    }
    assert name not in others and len(others) == 4
    assert template_name("rsi", REFERENCE, (1, 1024), "r2") != name
    # E2B's names as before the remote seam: a new name rebuilds a template.
    assert sandbox_e2b.SdkE2BClient.recipe == RECIPE
    assert name == "rsi-0347d2be4658553c1f3c"
    steps, config = (("run", "x"),), {"User": "a"}
    built = built_template_name(
        "rsi", "d" * 64, REFERENCE, steps, config, (1, 1024), RECIPE
    )
    assert built == "rsi-ea576ab3e277903d5245"


def test_a_pull_builds_the_template_once_and_reuses_it(kit):
    work = open_work(kit)
    view = pull_view(kit, work)
    assert view["state"] == "succeeded"
    # The phase's per-container shape (4 cpus, 8192 MiB), the image as asked.
    [(name, image, cpus, memory)] = kit.fake.builds
    assert (image, cpus, memory) == (REFERENCE, 4, 8192)
    assert name == template_name("rsi", REFERENCE, (4, 8192), RECIPE)
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
    name = template_name("rsi", REFERENCE, (4, 8192), RECIPE)
    kit.fake.templates[name] = {"image": REFERENCE}
    work = open_work(kit)
    pull(kit, work)
    assert kit.fake.builds == []


def test_concurrent_ensures_of_one_name_build_once(tmp_path):
    fake = RemoteFake(tmp_path / "e2b")
    fake.build_gate = threading.Event()
    templates = RemoteTemplates(fake, "rsi")
    results = []

    def ensure():
        results.append(templates.ensure("rsi-x", REFERENCE, (1, 512)))

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


def test_a_forced_ensure_builds_again_after_the_running_build(tmp_path):
    fake = RemoteFake(tmp_path / "e2b")
    fake.build_gate = threading.Event()
    templates = RemoteTemplates(fake, "rsi")
    first = threading.Thread(
        target=templates.ensure, args=("rsi-x", REFERENCE, (1, 512))
    )
    first.start()
    until_true(lambda: len(fake.builds) == 1)
    results = []
    forced = threading.Thread(
        target=lambda: results.append(
            templates.ensure("rsi-x", REFERENCE, (1, 512), force=True)
        )
    )
    forced.start()
    time.sleep(0.1)
    fake.build_gate.set()
    for thread in (first, forced):
        thread.join(10)
    assert results == [True]
    assert [build[-1] for build in fake.dockerfile_builds] == [False, True]


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
            "Shell": ["/bin/bash", "-c"],
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
        "Shell": ["/bin/bash", "-c"],  # for builds on it
    }
    # The token never goes to the blob's CDN.
    assert ("https://cdn.example/b", None) in seen
    served = blob + b" "
    with pytest.raises(InfrastructureError, match="does not match its digest"):
        sandbox_e2b.registry_image_config("docker.io/alexgshaw/fix-git", "20251031")


@pytest.mark.parametrize("scheme", ["Bearer", "Basic"])
def test_registry_image_config_logs_in_to_a_private_registry(monkeypatch, scheme):
    httpx = pytest.importorskip("httpx")
    blob = json.dumps({"config": {"User": "app"}}).encode()
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    base = "https://ghcr.io/v2/acme/judge"
    basic = "Basic " + base64.b64encode(b"bot:s3cret").decode()
    granted = "Bearer t0k" if scheme == "Bearer" else basic
    seen = []

    def answer(request):
        url = str(request.url)
        seen.append((url, request.headers.get("authorization")))
        if url.startswith("https://ghcr.io/token"):
            if request.headers.get("authorization") != basic:
                return httpx.Response(401)
            return httpx.Response(200, json={"token": "t0k"})
        if url.startswith("https://cdn.example/"):
            return httpx.Response(200, content=blob)
        if request.headers.get("authorization") != granted:
            challenge = (
                'Bearer realm="https://ghcr.io/token",service="ghcr.io"'
                if scheme == "Bearer"
                else 'Basic realm="ghcr.io"'
            )
            return httpx.Response(401, headers={"www-authenticate": challenge})
        if url == f"{base}/manifests/1.0":
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
    login = ("bot", "s3cret")
    found = sandbox_e2b.registry_image_config("ghcr.io/acme/judge", "1.0", login=login)
    assert found == {"User": "app"}
    # Neither the login nor the token goes to the blob's CDN.
    assert ("https://cdn.example/b", None) in seen
    # Anonymously: refused by the realm, or no login to answer Basic.
    with pytest.raises((httpx.HTTPStatusError, InfrastructureError)):
        sandbox_e2b.registry_image_config("ghcr.io/acme/judge", "1.0")


# -- envs ---------------------------------------------------------------------------


@every
def test_env_lifecycle_runs_one_sandbox_with_metadata_network_and_timeout(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    public = ready(kit, work, handle)
    private = ready(kit, work, handle, "private", single(handle, network="none"))

    box = sandbox_of(kit, public)
    assert box.allow_internet and not sandbox_of(kit, private).allow_internet
    shape = {"e2b": (4, 8192), "modal": (4.0, 8192), "daytona": (4, 8, 10)}
    shape = shape[kit.fake.name]
    assert box.template == template_name("rsi", REFERENCE, shape, kit.fake.recipe)
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
    assert lease.backend == kit.fake.name and lease.network_name is None
    assert lease.services[0].sandbox_id == box.sandbox_id
    status = kit.broker.env_status(work.credential, public)
    assert status["services"]["main"]["state"] == "running"
    listing = kit.broker.env_list(work.credential)["envs"]
    assert {item["env_id"] for item in listing} == {public, private}
    capabilities = kit.broker.capabilities(work.credential)["environments"]
    assert capabilities["backend"] == kit.fake.name

    kit.broker.env_destroy(work.credential, public)
    assert box.state is None
    assert kit.broker.env_status(work.credential, public)["state"] == "removed"
    assert [env.env_id for env in kit.broker.journal.envs()] == [private]


@every
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
            lambda spec: spec.update(
                network="allowlist", allowlist=["pypi.org", "pypi.org:8443"]
            ),
            "spec.allowlist.1",
        ),
        (
            lambda spec: spec.update(
                network="allowlist", allowlist=["1.1.1.1", "1.0.0.1:443"]
            ),
            "spec.allowlist.1",
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
        refuse_unsupported(parse_env_spec(spec), "e2b")
    assert refused.value.code == "unsupported"
    assert refused.value.field == field
    assert "unsupported on the e2b backend" in str(refused.value)
    assert not any(entry in str(refused.value) for entry in spec.get("allowlist", ()))


@pytest.mark.parametrize(
    ("entry", "refused"),
    [
        ("example.com", False),
        ("example.com:443", False),
        ("example.com:80", True),
        ("1.1.1.1", False),
        ("8.8.8.0/24", False),
        ("1.1.1.1:443", True),
    ],
)
def test_modal_holds_hostnames_only_on_443_and_ips_without_ports(entry, refused):
    spec = single("i" + "a" * 32, network="allowlist")
    spec["allowlist"] = [entry]
    if not refused:
        refuse_unsupported(parse_env_spec(spec), "modal")
        return
    with pytest.raises(SandboxError, match="unsupported on the modal backend") as error:
        refuse_unsupported(parse_env_spec(spec), "modal")
    assert error.value.field == "spec.allowlist.0"


@pytest.mark.parametrize(
    ("entries", "field", "what"),
    [
        (["example.com", "pypi.org:443", "files.example:80"], None, None),
        (["1.1.1.1", "8.8.8.0/24"] + [f"10.0.0.{n}" for n in range(8)], None, None),
        (["example.com:8443"], "spec.allowlist.0", "an allowlist port is"),
        (["pypi.org", "1.1.1.1:443"], "spec.allowlist.1", "an allowlist port is"),
        (
            ["example.com", "1.1.1.1"],
            "spec.allowlist",
            "hostnames together with IPs or CIDRs are",
        ),
        (
            [f"10.0.0.{n}" for n in range(11)],
            "spec.allowlist",
            "more than 10 IPs and CIDRs are",
        ),
    ],
)
def test_daytona_holds_hostnames_or_ips_never_both(entries, field, what):
    spec = single("i" + "a" * 32, network="allowlist")
    spec["allowlist"] = entries
    if field is None:
        refuse_unsupported(parse_env_spec(spec), "daytona")
        return
    with pytest.raises(SandboxError) as refused:
        refuse_unsupported(parse_env_spec(spec), "daytona")
    assert (refused.value.code, refused.value.field) == ("unsupported", field)
    assert refused.value.message == f"{what} unsupported on the daytona backend"


def test_an_allowlist_env_hands_its_entries_to_e2b(tmp_path):
    task = TASK.replace(
        'network = ["public", "none"]', 'network = ["public", "none", "allowlist"]', 1
    )
    made = build_kit(tmp_path, task=task, allowlist={})
    try:
        work = open_work(made)
        handle = pull(made, work)
        granted = made.broker.capabilities(work.credential)["environments"]
        assert "allowlist" in granted["network"]
        spec = single(handle, network="allowlist")
        spec["allowlist"] = [
            "1.1.1.1/32",
            "8.8.8.0/24",
            "Example.COM:443",
            "example.com:80",
            "pypi.org",
        ]
        box = sandbox_of(made, ready(made, work, handle, "allow", spec))
        assert box.allow_out == ("1.1.1.1", "8.8.8.0/24", "example.com", "pypi.org")
        assert not box.allow_internet
        # A service off the env's network gets nothing of it.
        spec["services"]["main"]["network"] = "none"
        private = ready(made, work, handle, "private", spec)
        assert sandbox_of(made, private).allow_out == ()
        # Private ranges stay the operator's to approve, as on Docker.
        spec["allowlist"] = ["10.0.0.0/24"]
        with pytest.raises(SandboxError) as refused:
            made.broker.env_create(work.credential, spec, "refused")
        assert (refused.value.code, refused.value.field) == (
            "permission",
            "spec.allowlist.0",
        )
    finally:
        made.broker.close()
        made.fake.shutdown()


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


@every
def test_a_provider_capacity_refusal_is_a_quota_the_caller_may_wait_on(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    kit.fake.fail["create"] = ProviderCapacityError(
        "daytona refused the sandbox for capacity: Total CPU limit exceeded"
    )
    before = len(kit.fake.calls)
    with pytest.raises(SandboxError) as refused:
        kit.broker.env_create(work.credential, single(handle), "full")
    assert (refused.value.code, refused.value.field) == ("quota", PROVIDER_CAPACITY)
    assert "nothing remains" in refused.value.message
    # The client proved nothing landed: nothing is listed again under the
    # rate limit that refused it.
    assert [call for call in kit.fake.calls[before:] if call[0] == "list"] == []
    # Rolled back: no sandbox, no live charge, nothing journaled.
    assert kit.fake.live() == []
    usage = kit.broker.capabilities(work.credential)["environments"]["usage"]
    assert usage["envs_live"] == 0
    assert kit.broker.journal.envs() == ()
    # With room again, the caller's next request creates the env.
    env_id = ready(kit, work, handle, "again")
    assert sandbox_of(kit, env_id).state == "running"


# -- builds ---------------------------------------------------------------------------

BUILD_TASK = TASK.replace("pull = true", "pull = true\nbuild = true", 1)
DOCKERFILE = b"""ARG BASE=busybox:1.37.0
FROM $BASE
ARG GREETING=hi
ENV APP_HOME=/srv/app
WORKDIR $APP_HOME
COPY app/ ./app/
COPY --chown=agent:agent --chmod=755 data.txt /opt/
RUN echo "$GREETING" > greeting
USER agent
CMD ["sleep", "300"]
"""
CONTEXT = {"Dockerfile": DOCKERFILE, "app/run.sh": b"echo run\n", "data.txt": b"x"}


@pytest.fixture
def builds(tmp_path):
    made = build_kit(tmp_path, task=BUILD_TASK)
    try:
        yield made
    finally:
        try:
            made.broker.close()
        finally:
            made.fake.shutdown()


def start_build(kit, session, files=CONTEXT, request_id="build", **options):
    fields = dict(
        dockerfile=None,
        dockerfile_inline=None,
        target=None,
        build_args={},
        labels={},
        no_cache=False,
        network="public",
        timeout_sec=60,
    )
    fields.update(options)
    stage_id = stage(kit, session, tar_of(files), request_id + "-stage")
    return kit.broker.image_build(
        session.credential, stage_id=stage_id, request_id=request_id, **fields
    )["job_id"]


def build_view(kit, session, files=CONTEXT, request_id="build", wait=10, **options):
    job_id = start_build(kit, session, files, request_id, **options)
    job = kit.envs.images.jobs[job_id]
    if job.thread is not None:
        job.thread.join(wait)
    return kit.broker.job_wait(session.credential, job_id, 0, 0)


def built(kit, session, files=CONTEXT, request_id="build", **options):
    view = build_view(kit, session, files, request_id, **options)
    assert view["state"] == "succeeded", view
    return view["result"]["image"]


def test_a_dockerfile_build_makes_a_template_whose_envs_run_its_config(
    builds, tmp_path
):
    from rsi_harness.runtime.recovery import retained_sandbox_resources

    work = open_work(builds)
    assert builds.broker.capabilities(work.credential)["environments"]["build"]
    image = built(builds, work, build_args={"GREETING": "yo"})
    [(name, base, steps, files, force)] = builds.fake.dockerfile_builds
    assert (base, files, force) == (
        REFERENCE,
        ["Dockerfile", "app/run.sh", "data.txt"],
        False,
    )
    env = "PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 APP_HOME=/srv/app"
    # The base image's USER and WORKDIR open the steps; COPY is root's
    # without --chown; ARGs reach RUN only.
    assert steps == (
        ("user", "root"),
        ("workdir", "/"),
        ("env", {"APP_HOME": "/srv/app"}),
        ("workdir", "/srv/app"),
        ("copy", "app", "/srv/app/app/", "root", None),
        ("copy", "data.txt", "/opt/", "agent:agent", 0o755),
        (
            "run",
            f"export GREETING=yo {env} && "
            "exec /bin/sh -c 'echo \"$GREETING\" > greeting'",
        ),
        ("user", "agent"),
    )
    assert name.startswith("rsi-") and name != template_name(
        "rsi", REFERENCE, (4, 8192), RECIPE
    )
    assert (image["workdir"], image["user"], image["cmd"]) == (
        "/srv/app",
        "agent",
        ["sleep", "300"],
    )
    assert image["env_count"] == 3
    # A cache entry like a pulled template: nothing for recovery to remove.
    [record] = [
        item
        for item in builds.broker.journal.images()
        if item.handle == image["handle"]
    ]
    assert (record.kind, record.state) == ("pulled", "present")
    assert retained_sandbox_resources(builds.store.read("run-1")) == ()
    # The build input and its extracted context are gone.
    assert list((tmp_path / "spool" / "build").iterdir()) == []

    handle = image["handle"]
    spec = single(handle)
    del spec["services"]["main"]["command"]
    ready(builds, work, handle, "env", spec)
    service = next(item for item in builds.fake.started if item[1][0] == "setsid")
    assert service[1] == ("setsid", "sleep", "300")
    assert service[3:] == ("/srv/app", "agent")
    [listed] = builds.broker.image_list(work.credential)["images"]
    assert listed["kind"] == "built"


def test_modal_builds_send_the_dockerfiles_own_lines(tmp_path):
    made = build_kit(tmp_path, "modal", task=BUILD_TASK)
    try:
        work = open_work(made)
        # Modal's builder resolves users as Docker does: no E2B refusals.
        dockerfile = DOCKERFILE + (
            b"ARG BASE\nARG TARGETARCH\nUSER 1000\nRUN id -u > uid\n"
            b"COPY --chown=agent data.txt /opt/own/\n"
        )
        files = {**CONTEXT, "Dockerfile": dockerfile}
        image = built(made, work, files, build_args={"GREETING": "it's"})
        [(name, base, steps, context, force)] = made.fake.dockerfile_builds
        assert (base, context, force) == (
            REFERENCE,
            ["Dockerfile", "app/run.sh", "data.txt"],
            False,
        )
        # Build args and inherited ARGs become the stage ARGs' defaults; USER
        # is left out and a RUN under a non-root user switches with setpriv.
        *lines, run_as, copy = steps
        assert lines == [
            "ARG GREETING='it'\"'\"'s'",
            "ENV APP_HOME=/srv/app",
            "WORKDIR $APP_HOME",
            "COPY app /srv/app/app/",
            "COPY --chown=agent:agent --chmod=755 data.txt /opt/",
            'RUN echo "$GREETING" > greeting',
            'CMD ["sleep", "300"]',
            "ARG BASE=busybox:1.37.0",
            "ARG TARGETARCH=amd64",
        ]
        assert run_as.startswith("RUN u=1000; ") and "setpriv" in run_as, run_as
        assert copy == "COPY --chown=agent data.txt /opt/own/"
        assert name != template_name("rsi", REFERENCE, (4.0, 8192), made.fake.recipe)
        assert (image["workdir"], image["user"]) == ("/srv/app", "1000")
        env_id = ready(made, work, image["handle"])
        assert sandbox_of(made, env_id).template == name
    finally:
        made.broker.close()
        made.fake.shutdown()


def test_daytona_has_no_image_build(tmp_path):
    made = build_kit(tmp_path, "daytona", task=BUILD_TASK)
    try:
        work = open_work(made)
        with pytest.raises(SandboxError) as refused:
            start_build(made, work)
        assert (refused.value.code, refused.value.message) == (
            "unsupported",
            "image_build is unsupported on the daytona backend",
        )
        assert made.fake.dockerfile_builds == []
    finally:
        made.broker.close()
        made.fake.shutdown()


def test_identical_builds_reuse_the_handle_then_the_template(builds, tmp_path):
    work = open_work(builds)
    first = built(builds, work)
    # B9: the same build in the same session answers with its image.
    assert built(builds, work, request_id="again")["handle"] == first["handle"]
    [(name, *_)] = builds.fake.dockerfile_builds
    # Released, the handle is unbound and the template stays.
    builds.broker.image_release(work.credential, first["handle"])
    assert built(builds, work, request_id="third")["handle"] != first["handle"]
    assert len(builds.fake.builds) == 1
    # Another broker (a new run) finds the template on E2B and builds nothing.
    other = build_kit(tmp_path / "other", task=BUILD_TASK)
    try:
        other.fake.templates.update(builds.fake.templates)
        session = open_work(other)
        built(other, session)
        assert other.fake.builds == []
        assert ("template_exists", name) in other.fake.calls
        # no_cache rebuilds it, skipping E2B's layer cache.
        built(other, session, request_id="fresh", no_cache=True)
        [(again, _, _, _, force)] = other.fake.dockerfile_builds
        assert (again, force) == (name, True)
    finally:
        other.broker.close()
        other.fake.shutdown()


def _symlinked():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(b"FROM busybox:1.37.0\n")
        archive.addfile(info, io.BytesIO(b"FROM busybox:1.37.0\n"))
        link = tarfile.TarInfo("key")
        link.type, link.linkname = tarfile.SYMTYPE, "/home/op/e2b.key"
        archive.addfile(link)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "case, field",
    [
        ("multi-stage", "dockerfile"),
        ("symlink", "stage_id"),
        ("network none", "network"),
        ("archive", "dockerfile"),
    ],
)
def test_builds_e2b_cannot_make_as_docker_does_are_refused_at_admission(
    builds, case, field
):
    work = open_work(builds)
    files, network = dict(CONTEXT), "public"
    if case == "multi-stage":
        files["Dockerfile"] = b"FROM busybox:1.37.0 AS a\nFROM busybox:1.37.0\n"
    elif case == "network none":
        network = "none"
    elif case == "archive":
        files["Dockerfile"] = b"FROM busybox:1.37.0\nADD *.tar /opt/\n"
        files["data.tar"] = tar_of({"inner": b"x"})
    with pytest.raises(SandboxError) as refused:
        if case == "symlink":
            stage_id = builds.broker.stage_put(
                work.credential,
                None,
                0,
                True,
                hashlib.sha256(_symlinked()).hexdigest(),
                "stage",
                _symlinked(),
            )["stage_id"]
            builds.broker.image_build(
                work.credential,
                stage_id=stage_id,
                dockerfile=None,
                dockerfile_inline=None,
                target=None,
                build_args={},
                labels={},
                no_cache=False,
                network="public",
                timeout_sec=60,
                request_id="build",
            )
        else:
            start_build(builds, work, files, network=network)
    assert (refused.value.code, refused.value.field) == ("unsupported", field)
    assert "unsupported on the e2b backend" in refused.value.message
    assert builds.envs.images.jobs == {}
    assert builds.fake.builds == []


@pytest.mark.parametrize(
    ("config", "dockerfile", "what"),
    [
        # E2B runs a step as a user name only.
        ({"User": "1000"}, b"RUN true\n", "RUN as a numeric USER"),
        ({"OnBuild": ["RUN true"]}, b"", "a base image's ONBUILD"),
        ({"Env": ["PKG=data.tar"]}, b"ADD $PKG /opt/\n", "ADD of a local archive"),
    ],
)
def test_a_build_e2b_cannot_make_on_the_base_images_config_fails_unbuilt(
    builds, config, dockerfile, what
):
    # The base image's config is known only in the job.
    builds.fake.image_configs[REFERENCE] = config
    work = open_work(builds)
    files = {
        "Dockerfile": b"FROM busybox:1.37.0\n" + dockerfile,
        "data.tar": tar_of({"inner": b"x"}),
    }
    view = build_view(builds, work, files)
    assert (view["state"], view["error"]["kind"]) == ("failed", "refused"), view
    assert f"{what} unsupported" in view["error"]["message"]
    assert builds.fake.builds == []


def test_a_build_extracts_its_context_only_above_the_spool_floor(builds):
    real = builds.envs._spool_admit

    def admit(field_name):
        if threading.current_thread() is not threading.main_thread():  # the job
            raise SandboxError("quota", field_name, "spool disk below the floor")
        real(field_name)

    builds.envs._spool_admit = admit
    work = open_work(builds)
    view = build_view(builds, work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "quota"), view
    assert builds.fake.builds == []


def test_a_build_cancelled_while_e2b_builds_binds_nothing(builds):
    builds.fake.build_gate = threading.Event()
    work = open_work(builds)
    job_id = start_build(builds, work)
    until_true(lambda: builds.fake.builds)
    assert builds.broker.job_cancel(work.credential, job_id) == {"state": "running"}
    builds.fake.build_gate.set()
    builds.envs.images.jobs[job_id].thread.join(10)
    view = builds.broker.job_wait(work.credential, job_id, 0, 0)
    assert view["state"] == "canceled", view
    assert builds.broker.image_list(work.credential)["images"] == []
    assert builds.broker.journal.images() == ()


def test_e2b_builds_need_public_build_network_and_no_builder(tmp_path):
    # No builder image, and nothing reserved for one.
    grant = e2b_grant(tmp_path, task=BUILD_TASK)
    plain = e2b_grant(tmp_path)
    assert grant.reserved_disk_mb == plain.reserved_disk_mb
    assert grant.environments.work.build is not None
    text = e2b_policy_text().replace('builder_image = "', '# builder_image = "')
    resolve_env_grant(
        make_env_task(BUILD_TASK),
        load_policy_text(tmp_path, text),
        builder_images={},
        parent_cpus=1,
        parent_memory_mb=256,
    )
    offline = BUILD_TASK.replace(
        "build = true", 'build = true\nbuild_network = ["none"]', 1
    )
    with pytest.raises(SetupError, match="e2b template builds always reach the"):
        e2b_grant(tmp_path, task=offline)
    with pytest.raises(SetupError, match="modal template builds always reach"):
        e2b_grant(tmp_path, task=offline, backend="modal")
    # Daytona builds nothing: its image_build is refused when called.
    daytona = e2b_grant(tmp_path, task=offline, backend="daytona")
    assert daytona.environments.work.build.network == ("none",)


# -- execs --------------------------------------------------------------------------


@every
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


@every
def test_execs_get_the_image_env_spec_env_cwd_and_user(kit):
    # The registry's ENV, which a backend without envd's record applies.
    kit.fake.image_configs[REFERENCE] = {
        "Env": [*IMAGE_CONFIG["Env"], "IMAGE_ONLY=from-image"]
    }
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
    if kit.fake.native_user:
        # E2B starts processes as a user name only.
        with pytest.raises(SandboxError) as refused:
            start_exec(kit, work, env_id, ["true"], "nobody", user="4242")
        assert (refused.value.code, refused.value.field) == ("invalid", "user")
        return
    # setpriv runs a uid without a passwd entry as Docker does.
    view = run(kit, work, env_id, ["sh", "-c", 'echo "$HOME"'], "bare", user="4242")
    assert view["stdout"] == b"/\n"
    assert kit.fake.raw[-1] == tuple(
        group_argv(
            ["setpriv", "--reuid=4242", "--regid=0", "--clear-groups", "--"]
            + ["sh", "-c", 'echo "$HOME"']
        )
    )


@every
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


@every
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


@every
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


def _stage_files(tmp_path):
    return [path for path in (tmp_path / "spool").rglob("*.tar") if path.is_file()]


@every
def test_copy_out_streams_and_stops_at_max_bytes(kit, tmp_path):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    root = sandbox_of(kit, env_id).root
    (root / "data").mkdir()
    (root / "data" / "big").write_bytes(b"x" * (4 * MIB))

    with pytest.raises(SandboxError) as refused:
        kit.broker.copy_out(work.credential, env_id, "main", "/data/big", MIB, [])
    assert refused.value.code == "quota"
    # Refused from the file's header, long before the archive was read.
    assert kit.fake.streamed[-1] < 2 * MIB
    assert _stage_files(tmp_path) == []
    assert list((root / "tmp").iterdir()) == []

    out = kit.broker.copy_out(work.credential, env_id, "main", "/data/big", 8 * MIB, [])
    assert out["bytes"] == 4 * MIB
    assert kit.fake.streamed[-1] > 4 * MIB
    assert list((root / "tmp").iterdir()) == []


@every
def test_a_cut_copy_out_stream_is_an_unknown_outcome(kit, tmp_path, monkeypatch):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    root = sandbox_of(kit, env_id).root
    (root / "data").mkdir()
    (root / "data" / "big").write_bytes(b"x" * MIB)

    def cut(sandbox_id, path):
        def chunks():
            yield (root / path.lstrip("/")).read_bytes()[:4096]
            raise RuntimeError("the stream ended early")

        return sandbox_e2b.ChunkReader(chunks, lambda error: error)

    monkeypatch.setattr(kit.fake, "read_stream", cut)
    with pytest.raises(SandboxError) as cut_off:
        kit.broker.copy_out(work.credential, env_id, "main", "/data/big", 8 * MIB, [])
    assert cut_off.value.code == "unknown-outcome"
    assert "ended early" in str(cut_off.value)
    assert _stage_files(tmp_path) == []
    assert list((root / "tmp").iterdir()) == []


@every
def test_copy_out_opens_no_stream_when_its_stage_cannot_be_made(kit, monkeypatch):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))

    def full(self):
        raise OSError("spool full")

    monkeypatch.setattr(StageStore, "create_result", full)
    with pytest.raises(OSError, match="spool full"):
        kit.broker.copy_out(work.credential, env_id, "main", "/etc", 8 * MIB, [])
    assert kit.fake.streamed == []
    assert list((sandbox_of(kit, env_id).root / "tmp").iterdir()) == []


@pytest.mark.parametrize("mode", ["e2b", "modal", "daytona"])
def test_tool_install_uploads_the_operators_tmux_once(tmp_path, mode):
    tool = tmp_path / "tmux"
    tool.write_bytes(b"echo fake-tmux\n")
    sha = hashlib.sha256(tool.read_bytes()).hexdigest()
    made = build_kit(tmp_path, mode, tmux={"path": str(tool), "sha256": sha})
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


# -- emulation ----------------------------------------------------------------------

emulated = pytest.mark.parametrize("kit", ["modal", "daytona"], indirect=True)


@emulated
def test_without_native_pids_commands_report_their_group_through_the_wrapper(kit):
    work = open_work(kit)
    handle = pull(kit, work)
    env_id = ready(kit, work, handle, spec=single(handle, working_dir="/srv"))
    # Broker commands run as they are, as root; the service is wrapped.
    assert kit.fake.raw[:2] == [
        ("mkdir", "-p", "/srv"),
        tuple(group_argv(["sleep", "600"])),
    ]
    # The command is its group's leader and the pid the broker signals.
    script = "echo $$; cut -d' ' -f5 /proc/$$/stat"
    view = run(kit, work, env_id, ["sh", "-c", script])
    pid, group = view["stdout"].decode().split()
    assert pid == group
    assert kit.fake.raw[-1] == tuple(group_argv(["sh", "-c", script]))


@emulated
def test_without_native_users_execs_switch_through_setpriv_with_their_home(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    view = run(kit, work, env_id, ["sh", "-c", 'echo "$HOME"'], user="1000")
    assert view["stdout"] == b"/home/agent\n"
    assert kit.fake.raw[-1] == tuple(
        group_argv(
            ["setpriv", "--reuid=1000", "--regid=1000", "--init-groups", "--"]
            + ["sh", "-c", 'echo "$HOME"']
        )
    )
    assert kit.fake.started[-1][4] == "agent"
    # The environment's own HOME wins, as on the Engine; root needs no setpriv.
    view = run(
        kit,
        work,
        env_id,
        ["sh", "-c", 'echo "$HOME"'],
        "own",
        user="agent",
        env={"HOME": "/srv"},
    )
    assert view["stdout"] == b"/srv\n"
    run(kit, work, env_id, ["true"], "as-root", user="root")
    assert kit.fake.raw[-1] == tuple(group_argv(["true"]))
    with pytest.raises(SandboxError) as refused:
        start_exec(kit, work, env_id, ["true"], "nobody", user="nobody")
    assert (refused.value.code, refused.value.field) == ("invalid", "user")
    assert "no passwd entry" in refused.value.message


@pytest.mark.parametrize("kit", ["modal"], indirect=True)
def test_a_command_past_modals_argument_limit_is_refused_at_exec_create(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    # The limit counts what start sends: the command and the group wrapper.
    with pytest.raises(SandboxError) as refused:
        start_exec(kit, work, env_id, ["echo", "x" * (2**16 - 20)])
    assert (refused.value.code, refused.value.field) == ("invalid", "cmd")
    assert "exceed 65536 bytes (modal backend)" in refused.value.message
    fits = run(kit, work, env_id, ["echo", "x" * (2**16 - 200)], "fits")
    assert fits["exit_code"] == 0 and len(fits["stdout"]) == 2**16 - 199


@emulated
def test_a_signal_freeze_keeps_streams_and_refuses_work_while_frozen(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    exec_id = start_exec(
        kit, work, env_id, ["sh", "-c", "echo one; sleep 1; echo two; exit 5"]
    )
    wait_exec(kit, work, exec_id, until=lambda view: view["stdout"])
    kit.broker.freeze_work()
    box = sandbox_of(kit, env_id)
    assert box.frozen and box.state == "running"
    assert kit.fake.raw[-1][3:] == (
        "rsi-freeze",
        "/proc",
        FROZEN_RECORD,
        "kill",
        "",
        str(FREEZE_TRIES),
    )
    assert kit.broker.env_status(work.credential, env_id)["state"] == "paused"
    lease = kit.envs._envs[env_id].lease
    with pytest.raises(SandboxError, match="paused"):
        kit.envs._backend.archive_target(lease, "main").require()
    kit.broker.resume_work()
    assert not box.frozen
    view = wait_exec(kit, work, exec_id)
    assert (view["state"], view["exit_code"]) == ("exited", 5)
    assert view["stdout"] == b"one\ntwo\n"
    assert not any(call[0] in ("connect", "pause", "resume") for call in kit.fake.calls)


@emulated
def test_a_signal_freeze_that_fails_quarantines_the_env(kit, monkeypatch):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    box = sandbox_of(kit, env_id)
    monkeypatch.setattr(
        kit.fake,
        "_freeze_script",
        lambda box, script: ["sh", "-c", "echo not stopped: 7 >&2; exit 1"],
    )
    kit.broker.freeze_work()
    assert box.state is None
    assert kit.broker.env_status(work.credential, env_id)["state"] != "paused"


# -- phases -------------------------------------------------------------------------


@every
def test_freeze_pauses_work_sandboxes_and_resume_resumes_them(kit):
    work = open_work(kit)
    env_id = ready(kit, work, pull(kit, work))
    box = sandbox_of(kit, env_id)
    kit.broker.freeze_work()
    assert box.paused
    assert kit.broker.env_status(work.credential, env_id)["state"] == "paused"
    judge = open_round(kit)
    assert judge is not None
    kit.broker.close_judge()
    box.timeout = 0
    kit.broker.resume_work()
    assert box.state == "running" and not box.paused
    if kit.fake.freeze == "native":
        # E2B restarts its timer on a pause: the timeout is set again.
        assert 500 < box.timeout <= 661
    assert kit.broker.env_status(work.credential, env_id)["state"] == "ready"
    assert run(kit, work, env_id, ["echo", "back"])["stdout"] == b"back\n"


@pytest.mark.parametrize("hours", [None, 2])
def test_a_sandbox_timeout_stays_within_the_teams_maximum_length(tmp_path, hours):
    # E2B refuses a timeout above the team's limit: an env may live that
    # long, and a longer one is refused as over the grant.
    options = {}
    if hours is not None:
        options = {"e2b": f"{E2B_KEY}\nmax_sandbox_hours = {hours}"}
    lifetime = (hours or 1) * 3600
    kit = build_kit(tmp_path, lifetime_sec=lifetime, **options)
    kit.fake.max_hours = hours or 1
    try:
        work = kit.broker.open_session(
            SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
        )
        kit.broker.activate_work(time.monotonic() + 3 * 3600)
        handle = pull(kit, work)
        with pytest.raises(SandboxError) as refused:
            kit.broker.env_create(
                work.credential,
                {**single(handle), "lifetime_sec": lifetime + 1},
                "too-long",
            )
        assert refused.value.code == "quota"
        assert "exceeds max_env_lifetime_sec" in str(refused.value)
        spec = {**single(handle), "lifetime_sec": lifetime}
        env_id = ready(kit, work, handle, spec=spec)
        box = sandbox_of(kit, env_id)
        wanted = (3600,) if hours is None else range(lifetime - 100, lifetime + 61)
        assert box.timeout in wanted
        kit.broker.freeze_work()
        kit.broker.resume_work()
        assert box.state == "running" and box.timeout in wanted
    finally:
        try:
            kit.broker.close()
        finally:
            kit.fake.shutdown()


def test_env_lifetimes_must_fit_in_the_accounts_maximum_sandbox_length(tmp_path):
    with pytest.raises(SetupError, match="max_env_lifetime_sec exceeds"):
        load_policy_text(tmp_path, e2b_policy_text(lifetime_sec=7200))
    two_hours = e2b_policy_text(
        e2b=f"{E2B_KEY}\nmax_sandbox_hours = 2", lifetime_sec=7200
    )
    policy = load_policy_text(tmp_path, two_hours, "two-hours.toml")
    assert policy.environments.work.max_env_lifetime_sec == 7200
    # Daytona sets no maximum sandbox length.
    long = e2b_policy_text(backend="daytona", lifetime_sec=30 * 3600)
    policy = load_policy_text(tmp_path, long, "daytona.toml")
    assert policy.environments.work.max_env_lifetime_sec == 30 * 3600


@every
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
    assert sandbox_of(kit, work_env).paused
    assert [env.env_id for env in kit.broker.journal.envs()] == [work_env]
    assert not kit.broker.recovery_required
    kit.broker.resume_work()
    assert not sandbox_of(kit, work_env).paused


@every
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
        remote_client=client,
    )
    return manager, store, clients


def _crashed_run(tmp_path, mode="e2b"):
    """A broker that died with a Work env paused and a Judge env running;
    another run's sandbox lives beside them."""
    from tests.runtime.test_sandbox_env_recovery import resource_lease

    made = build_kit(tmp_path, mode)
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
    assert sandbox_of(made, work_env).paused
    durable = made.store.read("run-1")
    lease = resource_lease(
        *durable.sandbox_envs,
        **{f"sandbox_{made.fake.name}": made.grant.environments.host.remote},
    )
    return made, lease, other


@pytest.mark.parametrize("mode", ["e2b", "modal", "daytona"])
@pytest.mark.parametrize("action", ["recover", "cleanup"])
def test_recovery_kills_the_runs_sandboxes_by_metadata(tmp_path, action, mode):
    made, lease, other = _crashed_run(tmp_path, mode)
    try:
        assert len(lease.sandbox_envs) == 2
        manager, store, clients = _recovery(tmp_path, lease, made.fake)

        getattr(manager, action)("run-1")

        assert [box.sandbox_id for box in made.fake.live()] == [other]
        assert clients == [made.grant.environments.host.remote]
        assert made.fake.closed
        recovered = store.read("run-1")
        assert recovered.sandbox_envs == ()
        assert not recovered.recovery_required
        assert ("list", {"rsi_run_id": "run-1"}) in made.fake.calls
    finally:
        made.fake.shutdown()


def test_recovery_reads_the_ledger_beside_the_runs_lease(tmp_path):
    made, lease, other = _crashed_run(tmp_path, "daytona")
    try:
        made.fake.ledger = None  # a client that keeps one
        manager, store, _ = _recovery(tmp_path, lease, made.fake)
        manager.recover("run-1")
        assert made.fake.ledger.path == store.root / "run-1.sandboxes.jsonl"
        assert made.fake.ledger.run_id == "run-1"
        assert [box.sandbox_id for box in made.fake.live()] == [other]
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


def test_registry_credentials_are_read_from_the_auth_file_only(tmp_path):
    path = tmp_path / "auth.json"
    encoded = base64.b64encode(b"bot:pa:ss").decode()
    path.write_text(
        json.dumps(
            {
                "auths": {
                    "https://index.docker.io/v1/": {"auth": encoded},
                    "nvcr.io": {"username": "$oauthtoken", "password": "n9c"},
                    "registry-1.docker.io": {"username": "r1", "password": "r1"},
                    "https://Registry.Example:5000/v2/": {"auth": encoded},
                    "ghcr.io": {},
                },
                "credsStore": "desktop",
                "credHelpers": {"ghcr.io": "gh"},
            }
        )
    )
    assert read_registry_auth(str(path), "e2b") == {
        "docker.io": ("bot", "pa:ss"),
        "nvcr.io": ("$oauthtoken", "n9c"),
        # Only index.docker.io is renamed, as pull_reference does.
        "registry-1.docker.io": ("r1", "r1"),
        "registry.example:5000": ("bot", "pa:ss"),
    }
    for text in (
        "{",
        '{"auths": {"a.io": {"auth": "!!"}}}',
        '{"auths": {"a.io": {"username": "bot"}}}',
        '{"auths": {"a.io": {"auth": "' + base64.b64encode(b"bot").decode() + '"}}}',
        '{"auths": ["a.io"]}',
    ):
        path.write_text(text)
        with pytest.raises(SetupError, match=f"registry_auth_file {path} is") as bad:
            read_registry_auth(str(path), "e2b")
        assert "bot" not in str(bad.value)
    with pytest.raises(SetupError, match="unreadable"):
        read_registry_auth(str(tmp_path / "missing"), "e2b")
    with pytest.raises(ValueError, match="absolute"):
        EnvE2BHost(api_key_env="K", registry_auth_file="auth.json")


def test_the_sdk_adapter_logs_in_to_private_registries_on_each_pull(
    tmp_path, monkeypatch
):
    pytest.importorskip("e2b")
    from types import SimpleNamespace

    path = tmp_path / "auth.json"
    settings = EnvE2BHost(api_key_env="K", registry_auth_file=str(path))
    # Only a pull reads the file: recovery and cleanup need the key alone.
    client = sandbox_e2b.e2b_client(settings, KEY)
    with pytest.raises(SetupError, match=f"registry_auth_file {path} is unreadable"):
        client.image_config("ghcr.io/acme/judge", "1.0")
    path.write_text(
        json.dumps({"auths": {"ghcr.io": {"username": "bot", "password": "s3cret"}}})
    )
    built, logs = [], []

    class Template:
        def __init__(self, file_context_path=None):
            pass

        def from_image(self, image, username=None, password=None):
            built.append((image, username, password))
            return self

        def set_user(self, user):
            return self

        def set_workdir(self, path):
            return self

        def run_cmd(self, command, user=None):
            return self

        @staticmethod
        def build(template, name, on_build_logs, **options):
            on_build_logs(SimpleNamespace(message="pull with s3cret"))
            raise RuntimeError("denied for bot:s3cret")

    client._e2b = SimpleNamespace(Template=Template)
    errors = []
    for image in ("ghcr.io/acme/judge:1.0", "docker.io/library/busybox:1.37.0"):
        with pytest.raises(InfrastructureError) as failed:
            client.build_template("rsi-x", image, (1, 512), log=logs.append)
        errors.append(str(failed.value))
    assert built == [
        ("ghcr.io/acme/judge:1.0", "bot", "s3cret"),
        ("docker.io/library/busybox:1.37.0", None, None),
    ]
    # The password never reaches the build's logs or errors.
    assert logs[0] == "pull with ***\n" and "bot:***" in errors[0]

    logins = []

    def config(repository, tag, *, proxy, login):
        logins.append(login)
        return {"User": "app"}

    monkeypatch.setattr(sandbox_e2b, "registry_image_config", config)
    assert client.image_config("ghcr.io/acme/judge", "1.0") == {"User": "app"}
    client.image_config("docker.io/library/busybox", "1.37.0")
    assert logins == [("bot", "s3cret"), None]


def test_the_sdk_adapter_turns_build_steps_into_template_calls(tmp_path):
    pytest.importorskip("e2b")
    from types import SimpleNamespace

    calls = []

    class Template:
        def __init__(self, file_context_path=None):
            calls.append(("context", file_context_path))

        def __getattr__(self, method):
            def record(*args, **kwargs):
                calls.append((method, *args, *sorted(kwargs.items())))
                return self

            return record

        @staticmethod
        def build(template, name, on_build_logs, **options):
            calls.append(("build", name, sorted(options)))
            calls.append(("skip_cache", options["skip_cache"]))

    client = sandbox_e2b.SdkE2BClient(KEY)
    client._e2b = SimpleNamespace(Template=Template)
    client.build_template(
        "rsi-x",
        REFERENCE,
        (1, 512),
        log=lambda line: None,
        steps=(
            ("user", "agent"),
            ("workdir", "/srv"),
            ("env", {"A": "1"}),
            ("run", "exec /bin/sh -c true"),
            ("copy", "app", "/srv/", "root", 0o755),
            ("add", "x.txt", "/etc/x.txt", "agent:agent", None),
        ),
        context=str(tmp_path),
        force=True,
    )
    assert calls == [
        ("context", str(tmp_path)),
        ("from_image", REFERENCE, ("password", None), ("username", None)),
        ("set_user", "agent"),
        # As BuildKit: a WORKDIR the steps make is the USER's, with its group.
        (
            "run_cmd",
            "test -d /srv || { mkdir -p /srv && "
            'chown agent:"$(id -g agent 2>/dev/null || echo 0)" /srv; }',
            ("user", "root"),
        ),
        ("set_workdir", "/srv"),
        ("set_envs", {"A": "1"}),
        ("run_cmd", "exec /bin/sh -c true"),
        ("copy", "app", "/srv/", ("mode", 0o755), ("user", "root")),
        ("copy", "x.txt", "/etc/x.txt", ("mode", None), ("user", "agent:agent")),
        # The recipe's own last steps: root, /, the ENV recording.
        ("set_user", "root"),
        ("set_workdir", "/"),
        ("run_cmd", sandbox_e2b.CAPTURE_ENV, ("user", "root")),
        # Uploads get the SDK's timeout, not the client's 60 s.
        ("build", "rsi-x", ["api_key", "cpu_count", "memory_mb", "skip_cache"]),
        ("skip_cache", True),
    ]


@pytest.mark.parametrize(
    "code, tags, status, expected",
    [
        (404, [], None, False),
        # Another team's name, as Template.exists.
        (403, [], None, True),
        (200, ["v1"], None, False),
        # A failed build leaves the name and its default tag behind.
        (200, ["default"], "error", False),
        (200, ["v1", "default"], "ready", True),
        # A deployment without tags or build status: the name alone.
        (200, 404, None, True),
        (200, ["default"], 404, True),
    ],
)
def test_a_template_exists_only_with_a_ready_default_build(
    monkeypatch, code, tags, status, expected
):
    pytest.importorskip("e2b")
    from types import SimpleNamespace

    from e2b.api.client.api.tags import get_templates_template_id_tags
    from e2b.api.client.api.templates import (
        get_templates_aliases_alias,
        get_templates_template_id_builds_build_id_status,
    )

    def response(found, parsed):
        if found == 404:
            return SimpleNamespace(status_code=404, parsed=None)
        return SimpleNamespace(status_code=200, parsed=parsed)

    asked = []
    monkeypatch.setattr(
        get_templates_aliases_alias,
        "sync_detailed",
        lambda alias, client: SimpleNamespace(
            status_code=code, parsed=SimpleNamespace(template_id="t1")
        ),
    )
    monkeypatch.setattr(
        get_templates_template_id_tags,
        "sync_detailed",
        lambda template_id, client: response(
            tags,
            [
                SimpleNamespace(tag=tag, build_id=f"b-{tag}")
                for tag in ([] if tags == 404 else tags)
            ],
        ),
    )

    def build_status(template_id, build_id, client):
        asked.append((template_id, build_id))
        return response(status, SimpleNamespace(status=SimpleNamespace(value=status)))

    monkeypatch.setattr(
        get_templates_template_id_builds_build_id_status, "sync_detailed", build_status
    )
    client = sandbox_e2b.SdkE2BClient(KEY)
    assert client.template_exists("rsi-x") is expected
    assert asked == ([("t1", "b-default")] if status else [])


def sdk_installed(
    monkeypatch, version: str | None = SDK_VERSION, missing: tuple[str, ...] = ()
) -> None:
    """Make package metadata report e2b at ``version`` (None: missing) and
    the rest of its extra installed, but ``missing``."""
    real = importlib.metadata.version

    def fake(name: str) -> str:
        if name in missing or (name == "e2b" and version is None):
            raise importlib.metadata.PackageNotFoundError(name)
        if name == "e2b":
            return version
        return "1.0" if name in ("dockerfile-parse", "socksio") else real(name)

    monkeypatch.setattr(importlib.metadata, "version", fake)


@pytest.mark.parametrize(
    "name,value",
    [
        ("E2B_DOMAIN", "e2b.example"),
        ("E2B_API_URL", "https://api.e2b.example"),
        ("E2B_SANDBOX_URL", "https://gw.e2b.example"),
        ("E2B_DEBUG", "True"),
    ],
)
def test_the_sdks_endpoint_and_debug_variables_are_refused(monkeypatch, name, value):
    refuse_e2b_environ({"E2B_DEBUG": "false", "E2B_DOMAIN": ""})
    with pytest.raises(SetupError, match=f"{name}.*domain"):
        refuse_e2b_environ({name: value})
    # Before the SDK is imported: also without the extra.
    monkeypatch.setenv(name, value)
    with pytest.raises(SetupError, match=name):
        sandbox_e2b.SdkE2BClient(KEY)


def test_the_submit_host_check_needs_the_pinned_sdk_and_the_key(monkeypatch):
    settings = EnvE2BHost(api_key_env=KEY_ENV)
    monkeypatch.setenv(KEY_ENV, KEY)
    for version, found in ((None, "none"), ("2.51.0", "2.51.0")):
        sdk_installed(monkeypatch, version)
        with pytest.raises(SetupError, match=rf"e2b=={SDK_VERSION} \(found {found}"):
            check_e2b_host(settings)
    # A venv synced before builds or a SOCKS proxy needed more of the extra.
    sdk_installed(monkeypatch, missing=("dockerfile-parse", "socksio"))
    with pytest.raises(SetupError, match=r"needs dockerfile-parse; install"):
        check_e2b_host(settings)
    socks = EnvE2BHost(api_key_env=KEY_ENV, proxy="socks5h://proxy:1080")
    with pytest.raises(SetupError, match=r"needs dockerfile-parse, socksio; "):
        check_e2b_host(socks)
    sdk_installed(monkeypatch)
    check_e2b_host(settings)
    check_e2b_host(socks)
    monkeypatch.setenv("E2B_SANDBOX_URL", "https://gw.e2b.example")
    with pytest.raises(SetupError, match="E2B_SANDBOX_URL"):
        check_e2b_host(settings)
    monkeypatch.delenv("E2B_SANDBOX_URL")
    monkeypatch.setenv(KEY_ENV, " ")
    with pytest.raises(SetupError, match="holds no key"):
        check_e2b_host(settings)


def test_the_e2b_extra_pins_the_checked_sdk_version():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    extra = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    assert f"e2b=={SDK_VERSION}" in extra["e2b"]


def test_sdk_errors_and_build_logs_never_carry_the_key():
    pytest.importorskip("e2b")
    client = sandbox_e2b.SdkE2BClient(KEY)
    cleaned = client._clean(RuntimeError(f"401 for key {KEY}"))
    assert KEY not in str(cleaned) and "***" in str(cleaned)


def test_the_sdk_adapter_streams_files_lazily_and_closes_them():
    pytest.importorskip("e2b")
    from types import SimpleNamespace

    reads, closed = [], []

    class Stream:
        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def __iter__(self):
            return self

        def __next__(self):
            chunk = next(self._chunks)
            if isinstance(chunk, Exception):
                raise chunk
            return chunk

        def close(self):
            closed.append(self)

    def read(path, **options):
        reads.append((path, options))
        return Stream([b"ab", b"cd", RuntimeError(f"boom {KEY}")])

    client = sandbox_e2b.SdkE2BClient(KEY)
    client._boxes["sbx"] = SimpleNamespace(files=SimpleNamespace(read=read))

    client.read_stream("sbx", "/tmp/a.tar").close()
    assert reads == [] and closed == []

    with client.read_stream("sbx", "/tmp/a.tar") as source:
        assert (source.read(1), source.read(9), source.read(9)) == (b"a", b"b", b"cd")
        assert reads == [("/tmp/a.tar", {"format": "stream", "user": "root"})]
        with pytest.raises(OSError) as failed:
            source.read(9)
    assert KEY not in str(failed.value) and "***" in str(failed.value)
    assert len(closed) == 1


def test_the_key_never_reaches_leases_plans_metadata_logs_or_capabilities(
    composed,  # noqa: F811 (the production fixture, imported above)
    caplog,
    monkeypatch,
):
    from rsi_harness.models import RunRequest
    from tests.runtime.test_sandbox_production import ENV_TASK

    sdk_installed(monkeypatch)
    ports, plan, definition, store, mutate, transports = composed
    key_file = ports.data_root.parent / "e2b.key"
    key_file.write_text(KEY)
    definition = definition.model_copy(update={"sandbox": make_env_task(ENV_TASK)})
    ports.definition = definition
    ports.sandbox_policy = load_policy_text(
        ports.data_root.parent,
        e2b_policy_text(e2b=f'api_key_file = "{key_file}"\ndomain = "e2b.example"'),
    )
    fake = RemoteFake(ports.data_root.parent / "e2b")
    received = []

    def client(credentials, settings):
        received.append((credentials, settings.domain))
        return fake

    ports.remote_client_factory = client
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
        assert received == [((KEY,), "e2b.example")]
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


def test_a_local_run_checks_the_e2b_host_before_any_client(
    composed,  # noqa: F811 (the production fixture, imported above)
    monkeypatch,
):
    from tests.runtime.test_sandbox_production import ENV_TASK

    ports, _plan, definition, _store, mutate, _transports = composed
    ports.definition = definition.model_copy(
        update={"sandbox": make_env_task(ENV_TASK)}
    )
    ports.sandbox_policy = load_policy_text(ports.data_root.parent, e2b_policy_text())
    monkeypatch.setenv(KEY_ENV, KEY)
    made = []
    ports.remote_client_factory = lambda *args: made.append(args)
    sdk_installed(monkeypatch, None)
    with pytest.raises(SetupError, match=r"e2b==.*found none.*\[e2b\]"):
        ports.bind_sandbox_lease(mutate)
    assert made == [] and ports._sandbox_broker_factory is None


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


@pytest.mark.parametrize(
    ("network", "internet", "egress"),
    [
        ({"allow_internet": False}, False, {}),
        ({"allow_internet": True}, True, {}),
        (
            {"allow_internet": False, "allow_out": ("1.1.1.1", "8.8.8.0/24")},
            True,
            {"allow_out": ["1.1.1.1", "8.8.8.0/24"], "deny_out": ["0.0.0.0/0"]},
        ),
    ],
    ids=("none", "public", "allowlist"),
)
def test_the_sdk_adapter_creates_private_sandboxes_with_the_key_and_domain(
    network, internet, egress
):
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
        "rsi-x", timeout=660, metadata={"rsi_run_id": "run-1"}, **network
    )
    assert sandbox_id == "sbx-1"
    [(template, options)] = created
    assert template == "rsi-x"
    assert options["api_key"] == KEY and options["domain"] == "e2b.example"
    assert options["timeout"] == 660 and options["allow_internet_access"] is internet
    assert options["network"] == {"allow_public_traffic": False, **egress}
    assert options["lifecycle"] == {"on_timeout": "kill"}
    assert options["metadata"] == {"rsi_run_id": "run-1"}


@every
@pytest.mark.parametrize("source", ["service", "image", "exec"])
def test_remote_primary_group_is_refused_without_launching_a_wrong_identity(
    kit, source
):
    if source == "image":
        kit.fake.image_configs[REFERENCE] = {"User": "1000:2000"}
    work = open_work(kit)
    handle = pull(kit, work)
    if source == "exec":
        env_id = ready(kit, work, handle)
        before = len(kit.fake.started)
        with pytest.raises(SandboxError, match="primary group"):
            start_exec(kit, work, env_id, ["true"], "explicit-group", user="1000:2000")
        assert len(kit.fake.started) == before
    else:
        spec = single(handle, **({"user": "1000:2000"} if source == "service" else {}))
        before = len(kit.fake.started)
        with pytest.raises(SandboxError, match="primary group"):
            kit.broker.env_create(work.credential, spec, "explicit-group")
        assert len(kit.fake.started) == before
