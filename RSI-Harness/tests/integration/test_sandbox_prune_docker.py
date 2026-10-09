"""prune-images against the real daemon as a non-root docker-group user.

A real broker (the production runtime composition; only the iptables half is
faked) pulls a tiny image the host does not have and one it already has,
then the operator command, run as a subprocess on the test's data root,
removes exactly the first and only once nothing uses it.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from rsi_harness.integrations.sandbox_client import SandboxClient
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_envs import docker_env_runtime
from rsi_harness.runtime.sandbox_ledger import PullLedger, pull_ledger_root
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from tests.integration.sandbox_support import sandbox_firewall
from tests.integration.test_sandbox_env_docker import BUSYBOX, remove_labelled
from tests.integration.test_sandbox_envs_docker import TASK, env_spec, pulled, up
from tests.runtime.test_sandbox_budget import authority
from tests.sandbox_helpers import (
    FakeSandboxBackend,
    builder_inspect,
    env_policy_toml,
    load_policy_text,
    make_env_task,
)

pytestmark = pytest.mark.integration

# About 2 MiB; absent from the reference host (checked before every run).
FRESH = "public.ecr.aws/docker/library/busybox:1.36.1"
CLI = Path(sys.executable).with_name("rsi-harness")
REQUIRED = os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1"


def unavailable(message):
    if REQUIRED:
        pytest.fail(message)
    pytest.skip(message)


def image_id(client, reference):
    try:
        return client.images.get(reference).id
    except ImageNotFound:
        return None


@pytest.fixture
def world(tmp_path):
    try:
        client = docker.from_env(timeout=60)
        client.ping()
        client.images.get(BUSYBOX)
        docker_root = client.info()["DockerRootDir"]
    except (DockerException, OSError, KeyError) as error:
        unavailable(f"Docker/{BUSYBOX} capability unavailable: {error}")
    if image_id(client, FRESH) is not None:
        client.close()
        # A leftover of a crashed run must not silently drop this coverage.
        unavailable(f"{FRESH} is already on this host; the test needs an absent tag")
    run_id = f"prune-{uuid.uuid4().hex[:12]}"
    data_root = tmp_path / "data"
    policy = load_policy_text(
        tmp_path,
        env_policy_toml().replace(
            'registries = ["docker.io"]', 'registries = ["docker.io", "public.ecr.aws"]'
        ),
    )
    grant = resolve_env_grant(
        make_env_task(TASK),
        policy,
        builder_images={"work": builder_inspect(), "judge": builder_inspect()},
        parent_cpus=1,
        parent_memory_mb=256,
    )
    runtime = docker_env_runtime(
        client,
        sandbox_firewall(client),
        run_id=run_id,
        spool_root=tmp_path / "sb" / "spool",
        docker_root=docker_root,
        host=grant.environments.host,
        data_root=data_root,
    )
    journal = SandboxJournal(authority(LeaseStore(data_root / "leases"), run_id))
    broker = SandboxBroker(
        grant, FakeSandboxBackend(), journal, time.monotonic, envs=runtime
    )
    # A short root: the endpoint socket path must fit 107 bytes.
    root = Path(tempfile.mkdtemp(prefix="rsi-prune-"))
    lifecycle = SandboxLifecycle()
    lifecycle.configure(broker, root / "sb", run_id, "task")
    cached = image_id(client, BUSYBOX)
    try:
        yield client, lifecycle, run_id, data_root
    finally:
        try:
            lifecycle.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)
            filters = {"label": f"rsi-harness.run-id={run_id}"}
            errors = remove_labelled(client, filters)
            # Only the image this test brought to the host, if it is left.
            try:
                fresh = image_id(client, FRESH)
                if fresh is not None and fresh != cached:
                    client.images.remove(FRESH)
            except DockerException as error:
                errors.append(f"{FRESH}: {type(error).__name__}")
            client.close()
            assert errors == []


def prune(data_root, *options):
    result = subprocess.run(
        [str(CLI), "sandbox", "prune-images", "--data-root", str(data_root), *options],
        cwd=data_root.parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def pull_fresh(work):
    """public.ecr.aws rate-limits anonymous pulls, in the progress stream
    ("toomanyrequests") or as a refused request: FRESH exists, so retry any
    registry error with backoff."""
    for attempt in range(4):
        job = work.follow_job(work.image_pull(FRESH, "missing"))
        error = job.get("error") or {}
        if job["state"] == "succeeded" or error.get("kind") != "registry":
            assert job["state"] == "succeeded", error
            return job
        time.sleep(2 + 4 * attempt)
    unavailable(f"{FRESH}: the registry keeps refusing pulls: {error}")


def row_of(output, image):
    rows = [line for line in output.splitlines() if line.startswith(image[7:19])]
    assert len(rows) == 1, output
    return rows[0]


def test_prune_removes_only_what_the_sandbox_pulled_first(world):
    client, lifecycle, run_id, data_root = world
    cached = image_id(client, BUSYBOX)
    lifecycle_endpoint = lifecycle.prepare_work()
    lifecycle.activate_work(time.monotonic() + 900)
    work = SandboxClient(
        lifecycle_endpoint.directory / "s",
        lifecycle_endpoint.environment["RSI_SANDBOX_TOKEN"],
    )

    job = pull_fresh(work)
    fresh_handle, fresh = job["result"]["image"]["handle"], image_id(client, FRESH)
    cached_handle = pulled(work)
    [entry] = PullLedger(pull_ledger_root(data_root)).entries()
    assert (entry.image_id, entry.references, entry.run_id) == (
        fresh,
        (FRESH,),
        run_id,
    )

    env_id = up(work, env_spec(fresh_handle))
    output = prune(data_root, "--dry-run")
    assert "kept" in row_of(output, fresh) and "used by container" in output
    assert cached[7:19] not in output

    work.env_destroy(env_id)
    deadline = time.monotonic() + 60
    while client.containers.list(all=True, filters={"ancestor": fresh}):
        assert time.monotonic() < deadline
        time.sleep(0.5)
    output = prune(data_root, "--dry-run")
    assert f"held by run {run_id}" in row_of(output, fresh)

    assert work.image_release(fresh_handle) == {"ok": True}
    output = prune(data_root, "--dry-run")
    assert "would remove" in row_of(output, fresh)
    assert image_id(client, FRESH) == fresh

    output = prune(data_root, "--yes")
    assert "removed" in row_of(output, fresh)
    assert "Freed: " in output
    assert image_id(client, FRESH) is None
    with pytest.raises(ImageNotFound):
        client.images.get(fresh)
    # The image already on the host, pulled in the same run, is untouched.
    assert image_id(client, BUSYBOX) == cached
    assert work.image_release(cached_handle) == {"ok": True}
    assert PullLedger(pull_ledger_root(data_root)).entries() == ()
    assert prune(data_root, "--dry-run").startswith("No pulled images")
