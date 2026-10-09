"""The v2 Harbor plugin against an in-process FakeBroker on a temp socket.

The fake speaks the real wire (HTTP/1.1 over a Unix socket, framed
stage_put, the error envelope) through the real stdlib client, and keeps a
small in-memory file tree per env service. It records every request, so the
tests pin the exact broker calls each Harbor method makes.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import http.server
import io
import json
import posixpath
import socketserver
import struct
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult, ServiceOperationsUnsupportedError
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

from rsi_harness.integrations import sandbox_harbor_env as plugin
from rsi_harness.integrations.sandbox_client import SandboxClient
from rsi_harness.integrations.sandbox_harbor_env import (
    ManagedSandboxEnvironment,
    ManagedSandboxError,
)

LIMITS = {
    "max_envs_live": 4,
    "max_envs_created": 400,
    "max_services_per_env": 8,
    "max_containers_live": 16,
    "max_cpus_live": 16,
    "max_memory_mb_live": 32768,
    "max_disk_mb_live": 16384,
    "cpus_per_container": 4,
    "memory_mb_per_container": 8192,
    "pids_per_container": 8192,
    "disk_mb_per_container": 10240,
    "max_env_lifetime_sec": 14400,
    "max_wait_timeout_sec": 900,
    "max_execs_running": 64,
    "max_exec_output_bytes": 16777216,
    "max_jobs_running": 4,
    "max_pull_mb": 20480,
    "max_log_bytes": 2147483648,
    "max_upload_bytes": 17179869184,
    "max_download_bytes": 17179869184,
    "max_operations": 1000000,
}


class Failure(Exception):
    def __init__(self, code, field, message="refused"):
        super().__init__(message)
        self.code, self.field, self.message = code, field, message


class FakeBroker:
    """Scriptable v2 broker: ``fail[op]`` pops errors, ``on_exec`` plans an
    exec as {"output": bytes, "exit_code": int, "state": str, "hang": bool}."""

    def __init__(self, path: Path, *, network=("public", "none"), build=None):
        self.path = path
        self.network = list(network)
        self.build = build
        self.allowlist = None
        self.tools = []
        self.limits = dict(LIMITS)
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, list[Failure]] = {}
        self.envs: dict[str, dict] = {}
        self.stages: dict[str, bytes] = {}
        self.execs: dict[str, dict] = {}
        self.replies: dict[str, dict] = {}
        self.status = "ready"
        # An env's lifetime at create; env_start refuses a longer wait than
        # what is left of it, as the broker does.
        self.expires_in_sec = 600.0
        self.bash = True
        self.on_exec = lambda meta: {"output": b"", "exit_code": 0}
        self.lock = threading.RLock()
        broker = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def address_string(self):
                return "fake"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                operation = self.path.removeprefix("/v1/")
                payload = None
                if self.headers["Content-Type"] == "application/octet-stream":
                    (size,) = struct.unpack("!I", body[:4])
                    meta = json.loads(body[4 : 4 + size])
                    payload = body[4 + size :]
                else:
                    meta = json.loads(body)
                status, reply = broker.dispatch(operation, meta, payload)
                data = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.server = Server(str(path), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def ops(self, *names):
        return [op for op, _ in self.calls if not names or op in names]

    def meta(self, op):
        return [meta for name, meta in self.calls if name == op]

    def dispatch(self, operation, meta, payload):
        with self.lock:
            self.calls.append((operation, meta))
            queue = self.fail.get(operation)
            try:
                if queue:
                    raise queue.pop(0)
                return 200, getattr(self, "op_" + operation)(meta, payload)
            except Failure as failure:
                return 400, {
                    "error": {
                        "code": failure.code,
                        "field": failure.field,
                        "message": failure.message,
                    }
                }

    # -- ops ------------------------------------------------------------------------

    def op_capabilities(self, meta, payload):
        return {
            "version": 1,
            "versions": [1, 2],
            "grant": None,
            "environments": {
                "network": self.network,
                "pull": True,
                "registries": ["docker.io"],
                "build": self.build,
                "allowlist": self.allowlist,
                "tools": self.tools,
                "limits": self.limits,
                "usage": {},
                "session_remaining_sec": 1000.0,
            },
        }

    def replay(self, meta, make):
        key = meta["request_id"]
        if key not in self.replies:
            self.replies[key] = make()
        return self.replies[key]

    def op_image_pull(self, meta, payload):
        return self.replay(meta, lambda: {"job_id": "j" + uuid.uuid4().hex})

    def op_image_build(self, meta, payload):
        return self.replay(meta, lambda: {"job_id": "j" + uuid.uuid4().hex})

    def op_job_wait(self, meta, payload):
        pulls = self.meta("image_pull") + self.meta("image_build")
        index = [self.replies[item["request_id"]]["job_id"] for item in pulls].index(
            meta["job_id"]
        )
        return {
            "job_id": meta["job_id"],
            "kind": "pull",
            "state": "succeeded",
            "log": "",
            "next_offset": 0,
            "log_truncated": False,
            "result": {"image": {"handle": "i" + format(index, "032x")}},
            "error": None,
        }

    def op_job_cancel(self, meta, payload):
        return {"state": "canceled"}

    def op_image_release(self, meta, payload):
        return {"ok": True}

    def op_env_create(self, meta, payload):
        def make():
            env_id = "e" + uuid.uuid4().hex
            self.envs[env_id] = {
                "spec": meta["spec"],
                "state": "created",
                "expires_at": time.monotonic() + self.expires_in_sec,
                "files": {
                    name: {"/": ("dir", b"", 0o755)}
                    for name in meta["spec"]["services"]
                },
            }
            if self.bash and "main" in meta["spec"]["services"]:
                self.envs[env_id]["files"]["main"]["/bin/bash"] = ("file", b"", 0o755)
            return {
                "env_id": env_id,
                "state": "created",
                "network": None,
                "services": {},
                "notes": [],
                "expires_in_sec": self.expires_in_sec,
            }

        return self.replay(meta, make)

    def env(self, meta):
        env = self.envs.get(meta["env_id"])
        if env is None or env["state"] == "removed":
            raise Failure("invalid", "env_id", "unknown env")
        return env

    def op_env_start(self, meta, payload):
        env = self.env(meta)
        if meta["wait_timeout_sec"] > self.limits["max_wait_timeout_sec"]:
            raise Failure("quota", "wait_timeout_sec", "exceeds max_wait_timeout_sec")
        if meta["wait_timeout_sec"] > env["expires_at"] - time.monotonic():
            raise Failure(
                "quota", "wait_timeout_sec", "exceeds the env's remaining lifetime"
            )
        env["state"] = "starting"
        return {"state": "starting"}

    def op_env_status(self, meta, payload):
        env = self.env(meta)
        env["state"] = self.status
        return {
            "env_id": meta["env_id"],
            "state": self.status,
            "reason": None if self.status == "ready" else "unhealthy",
            "remaining_sec": 500.0,
            "services": {
                name: {
                    "state": "running",
                    "health": None if self.status == "ready" else "unhealthy",
                    "exit_code": None,
                    "oom_kills": 0,
                    "disk_mb_used": 0,
                    "started_at": None,
                    "diagnostics": None
                    if self.status == "ready"
                    else {"health_tail": "redis down", "log_tail": ""},
                }
                for name in env["spec"]["services"]
            },
        }

    def op_env_destroy(self, meta, payload):
        self.envs[meta["env_id"]]["state"] = "removed"
        return {"state": "removed"}

    def op_env_stop_service(self, meta, payload):
        self.env(meta)
        return {"state": "exited", "exit_code": 0}

    def op_stage_put(self, meta, payload):
        stage_id = meta["stage_id"] or "s" + uuid.uuid4().hex
        self.stages[stage_id] = self.stages.get(stage_id, b"") + payload
        return {
            "stage_id": stage_id,
            "bytes": len(self.stages[stage_id]),
            "entries": None,
        }

    def op_stage_get(self, meta, payload):
        data = self.stages[meta["stage_id"]]
        return data[meta["offset"] : meta["offset"] + meta["length"]]

    def tree(self, meta):
        return self.env(meta)["files"][meta["service"]]

    def op_copy_in(self, meta, payload):
        tree = self.tree(meta)
        dest = meta["dest_dir"]
        parts = [part for part in dest.split("/") if part]
        for index in range(1, len(parts) + 1):
            tree.setdefault("/" + "/".join(parts[:index]), ("dir", b"", 0o755))
        entries = 0
        with tarfile.open(fileobj=io.BytesIO(self.stages.pop(meta["stage_id"]))) as tar:
            for member in tar:
                path = posixpath.join(dest, member.name)
                entries += 1
                if member.isdir():
                    tree[path] = ("dir", b"", member.mode)
                elif member.issym():
                    tree[path] = ("symlink", member.linkname.encode(), 0o777)
                else:
                    tree[path] = ("file", tar.extractfile(member).read(), member.mode)
        return {"entries": entries, "bytes": 0}

    def op_copy_out(self, meta, payload):
        tree = self.tree(meta)
        root = meta["path"]
        if root not in tree:
            raise Failure("invalid", "path", f"{root} does not exist")
        base = posixpath.basename(root)
        buffer = io.BytesIO()
        skipped = 0
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for path in sorted(tree):
                if path != root and not path.startswith(root.rstrip("/") + "/"):
                    continue
                relative = path[len(root) :].lstrip("/")
                parts = relative.split("/")
                # GNU tar --exclude: any path prefix matching drops the subtree.
                prefixes = ["/".join(parts[: index + 1]) for index in range(len(parts))]
                if relative and any(
                    fnmatch.fnmatch(prefix, pattern)
                    or fnmatch.fnmatch(posixpath.basename(prefix), pattern)
                    for pattern in meta["exclude"]
                    for prefix in prefixes
                ):
                    skipped += 1
                    continue
                kind, data, mode = tree[path]
                entry = tarfile.TarInfo(posixpath.join(base, relative).rstrip("/"))
                entry.mode = mode
                if kind == "dir":
                    entry.type = tarfile.DIRTYPE
                    tar.addfile(entry)
                else:
                    entry.size = len(data)
                    tar.addfile(entry, io.BytesIO(data))
        stage_id = "s" + uuid.uuid4().hex
        self.stages[stage_id] = buffer.getvalue()
        return {"stage_id": stage_id, "bytes": 0, "entries": 0, "skipped": skipped}

    def op_path_stat(self, meta, payload):
        found = self.tree(meta).get(meta["path"])
        if found is None:
            return {
                "exists": False,
                "kind": None,
                "size": None,
                "mode": None,
                "mtime": None,
                "link_target": None,
            }
        return {
            "exists": True,
            "kind": found[0],
            "size": len(found[1]),
            "mode": found[2],
            "mtime": 0,
            "link_target": None,
        }

    def op_tool_install(self, meta, payload):
        tree = self.tree(meta)
        path = "/usr/local/bin/" + meta["tool"]
        installed = path not in tree
        if installed:
            tree[path] = ("file", b"static " + meta["tool"].encode(), 0o755)
        return {"tool": meta["tool"], "path": path, "installed": installed}

    def op_exec_start(self, meta, payload):
        def make():
            exec_id = "x" + uuid.uuid4().hex
            plan = {"state": "exited", "hang": False, **self.on_exec(meta)}
            self.execs[exec_id] = {"meta": meta, **plan}
            return {"exec_id": exec_id}

        self.env(meta)
        return self.replay(meta, make)

    def op_exec_wait(self, meta, payload):
        record = self.execs[meta["exec_id"]]
        if record["hang"]:
            deadline = time.monotonic() + min(meta["wait_sec"], 0.2)
            while record["hang"] and time.monotonic() < deadline:
                self.lock.release()
                time.sleep(0.01)
                self.lock.acquire()
        output = record["output"]
        chunk = output[meta["stdout_offset"] : meta["stdout_offset"] + 5]
        state = "running" if record["hang"] else record["state"]
        return {
            "state": state,
            "exit_code": None
            if state in ("running", "timed_out")
            else record["exit_code"],
            "signal": record.get("signal"),
            "stdout_b64": base64.b64encode(chunk).decode(),
            "stderr_b64": "",
            "stdout_offset": meta["stdout_offset"] + len(chunk),
            "stderr_offset": 0,
            "stdout_total": len(output),
            "stderr_total": 0,
            "truncated": False,
            "oom_kills": 0,
            "reason": "timeout" if state == "timed_out" else None,
        }

    def op_exec_kill(self, meta, payload):
        record = self.execs[meta["exec_id"]]
        if meta["signal"] == "KILL" or not record.get("ignore_term"):
            record.update(
                hang=False, state="killed", exit_code=None, signal=meta["signal"]
            )
        return {"delivered": True, "state": "running" if record["hang"] else "killed"}

    def op_image_list(self, meta, payload):
        return {"images": []}


@pytest.fixture
def broker():
    with tempfile.TemporaryDirectory(prefix="rsi-hb-") as root:
        fake = FakeBroker(Path(root) / "s")
        try:
            yield fake
        finally:
            fake.close()


def make_task(tmp_path, *, compose=None, dockerfile=False, files=None, **config):
    environment = tmp_path / "environment"
    environment.mkdir(exist_ok=True)
    if compose is not None:
        (environment / "docker-compose.yaml").write_text(compose)
    if dockerfile:
        (environment / "Dockerfile").write_text("FROM busybox\n")
    for name, text in (files or {}).items():
        (environment / name).parent.mkdir(parents=True, exist_ok=True)
        (environment / name).write_text(text)
    values = {"docker_image": "busybox:1.37.0", "cpus": 1, "memory_mb": 512}
    values.update(config)
    return environment, EnvironmentConfig(**values)


def make_env(
    broker,
    tmp_path,
    *,
    network=NetworkMode.PUBLIC,
    mounts=None,
    options=None,
    **task,
):
    """``options`` are the plugin's own keyword arguments (``--ek``)."""
    environment, config = make_task(tmp_path, **task)
    trial = TrialPaths(tmp_path / "trial")
    policy = NetworkPolicy(network_mode=network)
    return ManagedSandboxEnvironment(
        environment_dir=environment,
        environment_name="task",
        session_id="task__abc__env",
        trial_paths=trial,
        task_env_config=config,
        network_policy=policy,
        phase_network_policies=[policy],
        mounts=mounts
        if mounts is not None
        else [
            {"type": "bind", "source": str(tmp_path / name), "target": f"/logs/{name}"}
            for name in ("agent", "verifier", "artifacts")
        ],
        **{"client": SandboxClient(str(broker.path), "token"), **(options or {})},
    )


def run(coroutine):
    return asyncio.run(coroutine)


# -- construction and validation -----------------------------------------------------


def test_capabilities_follow_the_grant(broker, tmp_path):
    environment = make_env(broker, tmp_path)
    capabilities = environment.capabilities
    assert (capabilities.docker_compose, capabilities.mounted) == (True, False)
    assert capabilities.disable_internet
    assert not capabilities.gpus and not capabilities.dynamic_network_policy
    assert ManagedSandboxEnvironment.type() == "rsi-managed-sandbox"
    resources = ManagedSandboxEnvironment.resource_capabilities()
    assert resources.cpu_limit and resources.memory_limit
    broker.network = ["public"]
    assert not make_env(broker, tmp_path).capabilities.disable_internet


def sidecars(count, **fields):
    """A compose of ``count`` redis sidecars with the given service fields."""
    body = ", ".join(f"{key}: {value}" for key, value in fields.items())
    return (
        "services: {"
        + ", ".join(f"kv{index}: {{image: redis, {body}}}" for index in range(count))
        + "}"
    )


@pytest.mark.parametrize(
    ("changes", "limits", "message"),
    [
        ({"cpus": 8}, {}, "cpus=8 exceeds"),
        ({"memory_mb": 16384}, {}, "memory_mb=16384 exceeds"),
        ({"storage_mb": 20480}, {}, "storage_mb=20480 exceeds"),
        # Within disk_mb_per_container, above the live disk.
        (
            {"storage_mb": 9000},
            {"max_disk_mb_live": 8192},
            r"storage_mb=9000 exceeds the sandbox grant \(8192\)",
        ),
        (
            {"compose": "services: {main: {privileged: true}}"},
            {},
            "services.main.privileged",
        ),
        (
            {"compose": "services: {kv: {image: redis, cpus: 16}}"},
            {},
            "service kv cpus=16",
        ),
        # Whole-env sums against the live caps: waiting never admits them.
        ({"compose": sidecars(5, cpus=4)}, {}, r"cpus=21\.0; .* \(max_cpus_live\)"),
        (
            {"compose": sidecars(4, mem_limit="8g")},
            {},
            r"memory_mb=33280; .* \(max_memory_mb_live\)",
        ),
        ({"compose": sidecars(8)}, {}, "9 services; .* allows 8 per env"),
        (
            {"compose": sidecars(3)},
            {"max_containers_live": 3},
            r"containers=4; .* \(max_containers_live\)",
        ),
        ({"os": "windows"}, {}, "windows|Linux"),
        ({"gpus": 1}, {}, "gpus=1: managed sandbox environments have no GPUs"),
    ],
)
def test_definition_fails_before_start_and_never_clamps(
    broker, tmp_path, changes, limits, message
):
    broker.limits.update(limits)
    with pytest.raises((ValueError, RuntimeError), match=message):
        make_env(broker, tmp_path, **changes)
    assert broker.ops("env_create") == []


def test_plugin_options_reach_the_broker_and_are_validated(
    broker, tmp_path, monkeypatch
):
    # Without client=, the endpoint environment names the broker.
    monkeypatch.setenv("RSI_SANDBOX_SOCKET", str(broker.path))
    monkeypatch.setenv("RSI_SANDBOX_TOKEN", "token")
    environment = make_env(
        broker,
        tmp_path,
        options={"client": None, "pull_policy": "always", "lifetime_sec": 10},
    )

    async def scenario():
        await environment.start(force_build=False)
        await environment.stop(delete=True)

    run(scenario())
    assert broker.meta("image_pull")[0]["policy"] == "always"
    assert broker.meta("env_create")[0]["spec"]["lifetime_sec"] == 10
    for options, message in (
        ({"lifetime_sec": 14401}, "max_env_lifetime_sec"),
        ({"lifetime_sec": 0}, "max_env_lifetime_sec"),
        ({"on_cancel": "x"}, "on_cancel"),
        ({"pull_policy": "never"}, "pull_policy"),
    ):
        with pytest.raises(ValueError, match=message):
            make_env(broker, tmp_path, options=options)


def test_compose_interpolates_harbors_variables_but_no_endpoint_secret(
    broker, tmp_path, monkeypatch
):
    monkeypatch.setenv("RSI_SANDBOX_TOKEN", "secret")
    monkeypatch.setenv("FROM_PROCESS", "p")
    environment = make_env(
        broker,
        tmp_path,
        compose="""
services:
  main:
    environment:
      C: ${CPUS}
      M: ${MEMORY}
      I: ${PREBUILT_IMAGE_NAME}
      X: ${CONTEXT_DIR}
      V: ${ENV_VERIFIER_LOGS_PATH}
      E: ${TASK_VAR}
      P: ${FROM_PROCESS}
      T: ${RSI_SANDBOX_TOKEN:-none}
""",
        env={"TASK_VAR": "t"},
    )
    main = environment._translation.spec["services"]["main"]
    assert main["env"] == {
        "C": "1",
        "M": "512M",
        "I": "busybox:1.37.0",
        "X": str((tmp_path / "environment").resolve()),
        "V": "/logs/verifier",
        "E": "t",
        "P": "p",
        "T": "none",
        "TASK_VAR": "t",
    }


def test_network_policy_maps_or_is_refused(broker, tmp_path):
    make_env(broker, tmp_path, network=NetworkMode.NO_NETWORK)
    with pytest.raises(ValueError, match="allowlist"):
        environment, config = make_task(tmp_path)
        ManagedSandboxEnvironment(
            environment_dir=environment,
            environment_name="task",
            session_id="s",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=config,
            network_policy=NetworkPolicy(
                network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["example.com"]
            ),
            client=SandboxClient(str(broker.path), "token"),
        )
    broker.network = ["none"]
    with pytest.raises(ValueError, match="not include"):
        make_env(broker, tmp_path, network=NetworkMode.PUBLIC)


def allowlist_env(broker, tmp_path, hosts):
    environment, config = make_task(tmp_path)
    policy = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=hosts)
    return ManagedSandboxEnvironment(
        environment_dir=environment,
        environment_name="task",
        session_id="s",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=config,
        network_policy=policy,
        phase_network_policies=[policy],
        mounts=[],
        client=SandboxClient(str(broker.path), "token"),
    )


def test_an_allowlist_policy_becomes_an_allowlist_env(broker, tmp_path):
    broker.network = ["allowlist", "none"]
    broker.allowlist = {"max_entries": 3, "patterns": [], "private_cidrs": []}
    hosts = ["pypi.org", "93.184.215.14", "151.101.0.0/16"]
    environment = allowlist_env(broker, tmp_path, hosts)
    capabilities = environment.capabilities
    assert capabilities.network_allowlist and capabilities.disable_internet
    assert capabilities.network_allowlist_hostnames
    assert capabilities.network_allowlist_ipv4_addresses
    assert capabilities.network_allowlist_ipv4_cidrs
    assert not capabilities.network_allowlist_wildcard_hostnames
    assert not capabilities.network_allowlist_ipv6_addresses

    async def scenario():
        await environment.start(force_build=False)
        await environment.stop(delete=True)

    run(scenario())
    spec = broker.meta("env_create")[0]["spec"]
    assert (spec["network"], spec["allowlist"]) == ("allowlist", hosts)


@pytest.mark.parametrize(
    ("hosts", "match"),
    (
        (["*.pypi.org"], "wildcard hostnames is not supported"),
        (["2001:db8::1"], "IPv6 addresses is not supported"),
        (["a.example", "b.example", "c.example", "d.example"], "grant allows 3"),
    ),
)
def test_an_allowlist_the_broker_cannot_enforce_is_refused(
    broker, tmp_path, hosts, match
):
    broker.network = ["allowlist"]
    broker.allowlist = {"max_entries": 3, "patterns": [], "private_cidrs": []}
    with pytest.raises(ValueError, match=match):
        allowlist_env(broker, tmp_path, hosts)


def test_only_harbor_log_mounts_are_accepted(broker, tmp_path):
    with pytest.raises(ValueError, match="log mounts"):
        make_env(
            broker,
            tmp_path,
            mounts=[{"type": "bind", "source": "/host", "target": "/data"}],
        )


def test_version_guard_refuses_another_harbor(broker, tmp_path, monkeypatch):
    monkeypatch.setattr(plugin.harbor, "__version__", "0.22.0")
    with pytest.raises(RuntimeError, match="0.21"):
        make_env(broker, tmp_path)


def test_a_phase_without_environments_is_refused(broker, tmp_path, monkeypatch):
    monkeypatch.setattr(
        FakeBroker,
        "op_capabilities",
        lambda self, meta, payload: {
            "version": 1,
            "versions": [1, 2],
            "grant": {},
            "environments": None,
        },
    )
    with pytest.raises(ValueError, match="grants no brokered environments"):
        make_env(broker, tmp_path)


def test_preflight_needs_the_endpoint(monkeypatch):
    monkeypatch.delenv("RSI_SANDBOX_SOCKET", raising=False)
    monkeypatch.setenv("RSI_SANDBOX_TOKEN", "t")
    with pytest.raises(SystemExit, match="RSI_SANDBOX_SOCKET"):
        ManagedSandboxEnvironment.preflight()
    monkeypatch.setenv("RSI_SANDBOX_SOCKET", "/run/rsi-harness/sandbox/s")
    ManagedSandboxEnvironment.preflight()


# -- start and stop -------------------------------------------------------------------


def test_prebuilt_start_pulls_creates_starts_and_prepares_logs(broker, tmp_path):
    environment = make_env(
        broker, tmp_path, files={"data/input.txt": "input"}, workdir="/app"
    )

    async def scenario():
        await environment.start(force_build=False)
        await environment.stop(delete=True)

    run(scenario())
    assert broker.ops(
        "image_pull", "env_create", "env_start", "copy_in", "env_destroy"
    ) == [
        "image_pull",
        "env_create",
        "env_start",
        "copy_in",  # the /logs directories
        "copy_in",  # environment/ uploaded for a prebuilt image
        "env_destroy",
    ]
    assert broker.meta("image_pull")[0]["ref"] == "busybox:1.37.0"
    spec = broker.meta("env_create")[0]["spec"]
    assert spec["network"] == "public"
    main = spec["services"]["main"]
    assert main["command"] == ["sh", "-c", "sleep infinity"]
    assert (main["cpus"], main["memory_mb"]) == (1.0, 512)
    assert main["image"] == "i" + "0" * 32
    copies = broker.meta("copy_in")
    assert [item["dest_dir"] for item in copies] == ["/logs", "/app"]
    env = next(iter(broker.envs.values()))
    assert env["files"]["main"]["/logs/verifier"] == ("dir", b"", 0o777)
    assert env["files"]["main"]["/app/data/input.txt"][1] == b"input"
    assert env["state"] == "removed"


def test_compose_start_seeds_before_start_and_routes_sidecars(broker, tmp_path):
    environment = make_env(
        broker,
        tmp_path,
        compose="""
services:
  main:
    environment: {KV: kvstore}
    depends_on: {kv: {condition: service_healthy}}
    volumes: ["./seed/value.txt:/seed/value.txt:ro"]
  kv:
    image: redis:7-alpine
    healthcheck: {test: [CMD, redis-cli, ping], interval: 1s}
    networks: {default: {aliases: [kvstore]}}
""",
        files={"seed/value.txt": "42\n"},
        env={"TASK_VAR": "t"},
    )

    async def scenario():
        await environment.start(force_build=False)
        result = await environment.service_exec("redis-cli ping", service="kv")
        await environment.service_download_file(
            "/seed/value.txt", tmp_path / "out" / "value.txt", service="main"
        )
        # Files only kv has: every service-routed file op reaches kv.
        kv = next(iter(broker.envs.values()))["files"]["kv"]
        kv["/data"] = ("dir", b"", 0o755)
        kv["/data/dump.rdb"] = ("file", b"rdb", 0o644)
        kv["/data/tmp"] = ("dir", b"", 0o755)
        kv["/data/tmp/scratch"] = ("file", b"x", 0o644)
        dirs = (
            await environment.service_is_dir("/data", service="kv"),
            await environment.service_is_dir("/data"),  # main: no /data
        )
        await environment.service_download_dir(
            "/data", tmp_path / "kv-all", service="kv"
        )
        await environment.service_download_dir_with_exclusions(
            source_dir="/data",
            target_dir=tmp_path / "kv-some",
            exclude=["tmp"],
            service="kv",
        )
        await environment.stop_service("main")
        await environment.stop(delete=False)
        return result, dirs

    result, dirs = run(scenario())
    assert result.return_code == 0
    assert dirs == (True, False)
    stats = [(item["service"], item["path"]) for item in broker.meta("path_stat")]
    assert stats[-2:] == [("kv", "/data"), ("main", "/data")]
    assert [item["service"] for item in broker.meta("copy_out")] == [
        "main",
        "kv",
        "kv",
    ]
    assert broker.meta("copy_out")[-1]["exclude"] == ["tmp"]

    def files(root):
        return {
            str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
        }

    assert files(tmp_path / "kv-all") == {"dump.rdb", "tmp/scratch"}
    assert files(tmp_path / "kv-some") == {"dump.rdb"}
    order = broker.ops("image_pull", "env_create", "copy_in", "env_start")
    assert order == [
        "image_pull",
        "image_pull",
        "env_create",
        "copy_in",
        "env_start",
        "copy_in",
    ]
    spec = broker.meta("env_create")[0]["spec"]
    assert spec["services"]["kv"]["aliases"] == ["kvstore"]
    assert spec["services"]["kv"]["cpus"] == 1.0  # the sidecar default
    assert spec["services"]["main"]["env"] == {"KV": "kvstore", "TASK_VAR": "t"}
    seed = broker.meta("copy_in")[0]
    assert (seed["service"], seed["dest_dir"]) == ("main", "/seed")
    kv_exec = broker.meta("exec_start")[-1]
    assert (kv_exec["service"], kv_exec["argv"]) == (
        "kv",
        ["sh", "-c", "redis-cli ping"],
    )
    # Sidecars inherit no main defaults.
    assert (kv_exec["cwd"], kv_exec["env"], kv_exec["user"]) == (None, {}, None)
    assert (tmp_path / "out" / "value.txt").read_text() == "42\n"
    assert broker.meta("env_stop_service")[0]["service"] == "main"
    assert broker.ops("env_destroy") == ["env_destroy"]
    with pytest.raises(RuntimeError, match="not started|stopping"):
        run(environment.service_exec("true", service="kv"))


def test_unknown_sidecar_is_unsupported(broker, tmp_path):
    environment = make_env(broker, tmp_path)

    async def scenario():
        await environment.start(force_build=False)
        try:
            with pytest.raises(ServiceOperationsUnsupportedError):
                await environment.service_exec("true", service="db")
        finally:
            await environment.stop(delete=True)

    run(scenario())


def test_build_is_chosen_without_a_prebuilt_image_and_fails_cleanly(broker, tmp_path):
    environment = make_env(broker, tmp_path, dockerfile=True, docker_image=None)
    with pytest.raises(ManagedSandboxError, match="unsupported"):
        run(environment.start(force_build=False))
    assert broker.ops("image_build", "stage_put", "env_create") == []


def test_force_build_with_a_dockerfile_builds_through_the_broker(broker, tmp_path):
    broker.build = {"network": ["public"], "max_build_sec": 3600}
    environment = make_env(broker, tmp_path, dockerfile=True, build_timeout_sec=120.0)

    async def scenario():
        await environment.start(force_build=True)
        await environment.stop(delete=False)
        # delete=False destroys the env but keeps what the plugin built.
        assert broker.meta("image_release") == []
        await environment.stop(delete=True)

    run(scenario())
    assert broker.ops("image_pull") == []
    build = broker.meta("image_build")[0]
    assert (build["network"], build["timeout_sec"]) == ("public", 120.0)
    staged = broker.meta("stage_put")[0]
    assert staged["final"] and staged["sha256"]
    handle = broker.meta("env_create")[0]["spec"]["services"]["main"]["image"]
    # delete=True releases the images this plugin built, never pulled ones.
    assert broker.meta("image_release") == [{"image": handle}]
    assert broker.ops("env_destroy") == ["env_destroy"]


def test_stop_with_delete_after_a_failed_start_releases_built_images(broker, tmp_path):
    broker.build = {"network": ["public"], "max_build_sec": 3600}
    broker.status = "failed"
    environment = make_env(broker, tmp_path, dockerfile=True, docker_image=None)

    async def scenario():
        with pytest.raises(ManagedSandboxError, match="redis down"):
            await environment.start(force_build=False)
        # The failed start's cleanup destroyed the env and kept the image.
        assert broker.meta("image_release") == []
        await environment.stop(delete=True)

    run(scenario())
    handle = broker.meta("env_create")[0]["spec"]["services"]["main"]["image"]
    assert broker.meta("image_release") == [{"image": handle}]
    assert broker.ops("env_destroy") == ["env_destroy"]


def test_live_quota_is_retried_and_other_errors_are_not(broker, tmp_path, monkeypatch):
    monkeypatch.setattr(plugin, "_LIVE_RETRY_SEC", 0.01)
    environment = make_env(broker, tmp_path)
    broker.fail["env_create"] = [Failure("quota", "max_envs_live")] * 2

    async def scenario():
        await environment.start(force_build=False)
        await environment.stop(delete=True)

    run(scenario())
    ids = [item["request_id"] for item in broker.meta("env_create")]
    assert len(ids) == 3 and len(set(ids)) == 3
    broker.fail["env_create"] = [Failure("quota", "max_operations")]
    environment = make_env(broker, tmp_path)
    with pytest.raises(ManagedSandboxError, match="max_operations"):
        run(environment.start(force_build=False))


def test_start_waits_only_for_what_is_left_of_the_env_lifetime(broker, tmp_path):
    # Spec A8: a Judge with a 900 s verifier timeout (= max_wait_timeout_sec)
    # creates envs that live a little less than that. The create and the
    # seeding spend some of it, so their expires_in_sec is a stale wait.
    broker.expires_in_sec = 897.0
    original = broker.op_env_create

    def slow_create(meta, payload):
        created = original(meta, payload)
        time.sleep(0.2)
        return created

    broker.op_env_create = slow_create
    environment = make_env(broker, tmp_path)

    async def scenario():
        await environment.start(force_build=False)
        await environment.stop(delete=True)

    run(scenario())
    [start] = broker.meta("env_start")
    assert 890.0 < start["wait_timeout_sec"] < 897.0 - 0.2 - plugin._START_SLACK_SEC
    assert broker.ops("env_destroy") == ["env_destroy"]


def test_an_env_without_lifetime_left_is_destroyed_unstarted(broker, tmp_path):
    broker.expires_in_sec = plugin._START_SLACK_SEC / 2
    environment = make_env(broker, tmp_path)

    with pytest.raises(ManagedSandboxError, match="lifetime ended before"):
        run(environment.start(force_build=False))

    assert broker.ops("env_start") == []
    [env] = broker.envs.values()
    assert env["state"] == "removed"


def test_a_start_refused_for_lifetime_is_measured_again_and_resent_once(
    broker, tmp_path
):
    # Slot queueing can spend the 1 s slack before the broker sees the wait.
    lifetime = Failure(
        "quota", "wait_timeout_sec", "exceeds the env's remaining lifetime"
    )
    broker.expires_in_sec = 120.0
    broker.fail["env_start"] = [lifetime]
    environment = make_env(broker, tmp_path)

    async def scenario():
        await environment.start(force_build=False)
        await environment.stop(delete=True)

    run(scenario())
    first, second = broker.meta("env_start")
    assert second["wait_timeout_sec"] <= first["wait_timeout_sec"] < 120.0
    assert first["request_id"] != second["request_id"]

    broker.fail["env_start"] = [lifetime, lifetime]
    environment = make_env(broker, tmp_path)
    with pytest.raises(plugin.wire.ProtocolError, match="remaining lifetime"):
        run(environment.start(force_build=False))
    assert len(broker.meta("env_start")) == 4
    broker.fail["env_start"] = [Failure("quota", "max_operations")]
    environment = make_env(broker, tmp_path)
    with pytest.raises(plugin.wire.ProtocolError, match="refused"):
        run(environment.start(force_build=False))
    assert len(broker.meta("env_start")) == 5
    assert all(env["state"] == "removed" for env in broker.envs.values())


def test_failed_start_reports_diagnostics_and_destroys(broker, tmp_path):
    broker.status = "failed"
    environment = make_env(broker, tmp_path)

    async def scenario():
        with pytest.raises(ManagedSandboxError, match="redis down") as raised:
            await environment.start(force_build=False)
        # start() returns only after its cleanup task finished.
        assert environment._cleanup_task.done()
        return raised.value

    error = run(scenario())
    assert "environment failed (unhealthy)" in str(error)
    [env] = broker.envs.values()
    assert env["state"] == "removed"
    assert broker.ops("env_create", "env_destroy") == ["env_create", "env_destroy"]


def test_stop_during_start_fails_start_without_a_foreign_cancellation(broker, tmp_path):
    environment = make_env(broker, tmp_path)
    gate = threading.Event()
    original = broker.op_env_start

    def slow_start(meta, payload):
        broker.lock.release()
        try:
            gate.wait(5)
        finally:
            broker.lock.acquire()
        return original(meta, payload)

    broker.op_env_start = slow_start

    async def scenario():
        start = asyncio.create_task(environment.start(force_build=False))
        while not broker.meta("env_start"):
            await asyncio.sleep(0.01)
        stop = asyncio.create_task(environment.stop(delete=True))
        await asyncio.sleep(0.05)
        gate.set()
        # Harbor never asked to cancel start(): it sees a sandbox error.
        with pytest.raises(ManagedSandboxError, match="stopped during startup"):
            await start
        await stop

    run(scenario())
    [env] = broker.envs.values()
    assert env["state"] == "removed"
    assert broker.ops("env_destroy") == ["env_destroy"]


def test_busy_destroy_is_retried_within_the_lifecycle_bound(
    broker, tmp_path, monkeypatch
):
    environment = started(broker, tmp_path)
    broker.fail["env_destroy"] = [Failure("busy", "env_id")] * 3
    run(environment.stop(delete=True))
    assert broker.ops("env_destroy") == ["env_destroy"] * 4
    # Past the bound a busy broker ends the cleanup; stop() may retry it.
    environment = started(broker, tmp_path)
    monkeypatch.setattr(plugin, "_LIFECYCLE_WAIT_SEC", 0.2)
    broker.fail["env_destroy"] = [Failure("busy", "env_id")] * 10000

    async def scenario():
        with pytest.raises(ManagedSandboxError, match="cleanup"):
            await environment.stop(delete=True)
        cleanup = environment._cleanup_task
        await asyncio.wait_for(asyncio.gather(cleanup, return_exceptions=True), 5)
        assert cleanup.exception().code == "busy"
        broker.fail["env_destroy"] = []
        await environment.stop(delete=True)

    run(scenario())
    assert [env["state"] for env in broker.envs.values()] == ["removed"] * 2


def test_cancelled_start_replays_a_lost_create_and_destroys(broker, tmp_path):
    environment = make_env(broker, tmp_path)
    gate = threading.Event()
    original = broker.op_env_create

    def slow_create(meta, payload):
        broker.lock.release()
        try:
            gate.wait(5)
        finally:
            broker.lock.acquire()
        return original(meta, payload)

    broker.op_env_create = slow_create

    async def scenario():
        start = asyncio.create_task(environment.start(force_build=False))
        while not broker.meta("env_create"):
            await asyncio.sleep(0.01)
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start
        gate.set()
        await environment.stop(delete=True)

    run(scenario())
    creates = broker.meta("env_create")
    # The replay used the original request_id and learned the env it made.
    assert len({item["request_id"] for item in creates}) == 1
    assert len(broker.envs) == 1
    assert next(iter(broker.envs.values()))["state"] == "removed"


# -- exec ---------------------------------------------------------------------------


def started(broker, tmp_path, **task):
    environment = make_env(broker, tmp_path, **task)
    run(environment.start(force_build=False))
    return environment


def without_tmux(meta):
    """An image without tmux on PATH: ``command -v tmux`` fails."""
    return {
        "output": b"",
        "exit_code": 1 if meta["argv"][-1] == "command -v tmux" else 0,
    }


def test_an_offered_tmux_is_installed_into_main_before_agent_setup(broker, tmp_path):
    broker.tools = ["tmux"]
    broker.on_exec = without_tmux
    environment = started(
        broker, tmp_path, files={"data/input.txt": "input"}, workdir="/app"
    )

    assert broker.ops("copy_in", "exec_start", "tool_install") == [
        "copy_in",  # the /logs directories
        "exec_start",  # command -v tmux, as uid 0
        "tool_install",
        "exec_start",  # tmux -V
        "copy_in",  # environment/ uploaded for a prebuilt image
    ]
    probe, check = broker.meta("exec_start")
    assert (probe["argv"], probe["user"]) == (["bash", "-c", "command -v tmux"], "0")
    assert (check["argv"], check["user"]) == (["bash", "-c", "tmux -V"], "0")
    [install] = broker.meta("tool_install")
    assert install == {
        "env_id": environment._env_id,
        "service": "main",
        "tool": "tmux",
    }
    tree = next(iter(broker.envs.values()))["files"]["main"]
    assert tree["/usr/local/bin/tmux"] == ("file", b"static tmux", 0o755)
    run(environment.stop(delete=True))


def test_tmux_on_path_is_kept_and_nothing_is_installed(broker, tmp_path):
    broker.tools = ["tmux"]
    started(broker, tmp_path)
    assert [meta["argv"][-1] for meta in broker.meta("exec_start")] == [
        "command -v tmux"
    ]
    assert broker.ops("tool_install") == []


@pytest.mark.parametrize(
    ("tools", "options"), [([], {}), (["tmux"], {"inject_tmux": "off"})]
)
def test_no_tmux_is_injected_unless_offered_and_wanted(
    broker, tmp_path, tools, options
):
    broker.tools = tools
    broker.on_exec = without_tmux
    environment = make_env(broker, tmp_path, options=options)
    run(environment.start(force_build=False))
    assert broker.ops("exec_start", "tool_install") == []
    with pytest.raises(ValueError, match="inject_tmux"):
        make_env(broker, tmp_path, options={"inject_tmux": "always"})


def test_a_refused_tmux_install_fails_the_start_and_destroys_the_env(broker, tmp_path):
    broker.tools = ["tmux"]
    broker.on_exec = without_tmux
    broker.fail["tool_install"] = [
        Failure("infrastructure", "tool", "differs from its approved sha256")
    ]
    environment = make_env(broker, tmp_path)
    with pytest.raises(ManagedSandboxError, match="tool_install tmux infrastructure"):
        run(environment.start(force_build=False))
    assert next(iter(broker.envs.values()))["state"] == "removed"


def test_exec_uses_bash_workdir_env_user_and_merged_output(broker, tmp_path):
    environment = started(broker, tmp_path, workdir="/app", env={"PERSIST": "1"})
    broker.on_exec = lambda meta: {"output": "héllo wörld\n".encode(), "exit_code": 3}
    seen = []

    async def collect(text, stream):
        seen.append((stream, text))

    async def scenario():
        with environment.scoped_output_callback(collect):
            return await environment.exec("echo hi", env={"X": "y"}, user=1000)

    result = run(scenario())
    assert result == ExecResult(stdout="héllo wörld\n", stderr=None, return_code=3)
    assert "".join(text for _, text in seen) == "héllo wörld\n"
    meta = broker.meta("exec_start")[-1]
    assert meta["argv"] == ["bash", "-c", "echo hi"]
    assert (meta["cwd"], meta["user"], meta["merge_stderr"]) == ("/app", "1000", True)
    assert meta["env"] == {"PERSIST": "1", "X": "y"}
    assert meta["timeout_sec"] is None


def test_exec_without_bash_falls_back_to_sh(broker, tmp_path):
    broker.bash = False
    environment = started(broker, tmp_path)
    run(environment.exec("true"))
    assert broker.meta("exec_start")[-1]["argv"][0] == "sh"


def test_timeout_raises_harbors_runtime_error(broker, tmp_path):
    environment = started(broker, tmp_path)
    broker.on_exec = lambda meta: {"output": b"", "state": "timed_out"}
    with pytest.raises(RuntimeError, match="^Command timed out after 5 seconds$"):
        run(environment.exec("sleep 60", timeout_sec=5))
    assert broker.meta("exec_start")[-1]["timeout_sec"] == 5.0


def test_killed_exec_reports_the_signal_status(broker, tmp_path):
    environment = started(broker, tmp_path)
    broker.on_exec = lambda meta: {
        "output": b"",
        "state": "killed",
        "exit_code": None,
        "signal": "KILL",
    }
    assert run(environment.exec("sleep 60")).return_code == 137


@pytest.mark.parametrize("on_cancel", ["kill-group", "detach"])
def test_cancel_kills_the_exec_group_unless_detached(broker, tmp_path, on_cancel):
    environment = started(broker, tmp_path)
    environment._on_cancel = on_cancel
    broker.on_exec = lambda meta: {"output": b"", "hang": True, "ignore_term": True}

    async def scenario():
        task = asyncio.create_task(environment.exec("sleep 600"))
        while not broker.meta("exec_wait"):
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.gather(*environment._kills)

    run(scenario())
    kills = [(item["signal"], item["scope"]) for item in broker.meta("exec_kill")]
    if on_cancel == "detach":
        assert kills == []
    else:
        # TERM to the process group, then KILL when it outlives the grace.
        assert kills == [("TERM", "group"), ("KILL", "group")]


@pytest.mark.parametrize("on_cancel", ["kill-group", "detach"])
def test_cancel_during_exec_start_still_kills_the_started_group(
    broker, tmp_path, on_cancel
):
    environment = started(broker, tmp_path)
    environment._on_cancel = on_cancel
    broker.on_exec = lambda meta: {"output": b"", "hang": True, "ignore_term": True}
    gate = threading.Event()
    original = broker.op_exec_start

    def slow_start(meta, payload):
        broker.lock.release()
        try:
            gate.wait(5)
        finally:
            broker.lock.acquire()
        return original(meta, payload)

    broker.op_exec_start = slow_start

    async def scenario():
        task = asyncio.create_task(environment.exec("sleep 600"))
        while not broker.meta("exec_start"):
            await asyncio.sleep(0.01)
        # Harbor cancels while exec_start is in flight; the broker still
        # starts the process after the caller has gone.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        while environment._kills:
            await asyncio.gather(*environment._kills)

    run(scenario())
    [exec_id] = broker.execs
    kills = [(item["exec_id"], item["signal"]) for item in broker.meta("exec_kill")]
    if on_cancel == "detach":
        assert kills == []
    else:
        assert kills == [(exec_id, "TERM"), (exec_id, "KILL")]
    # The kill learned the exec from the one start, never a second one.
    assert len(broker.meta("exec_start")) == 1


def test_transport_loss_replays_exec_start_with_the_same_id(broker, tmp_path):
    environment = started(broker, tmp_path)
    broker.fail["exec_start"] = [Failure("unknown-outcome", "request")]
    assert run(environment.exec("true")).return_code == 0
    ids = [item["request_id"] for item in broker.meta("exec_start")[-2:]]
    assert ids[0] == ids[1]


# -- files --------------------------------------------------------------------------


def test_upload_file_follows_docker_cp_target_rules(broker, tmp_path):
    environment = started(broker, tmp_path)
    source = tmp_path / "local.txt"
    source.write_text("data")

    async def scenario():
        await environment.upload_file(source, "/opt/renamed.txt")
        await environment.upload_file(source, "/logs/agent")
        await environment.upload_file(source, "/new/dir/")

    run(scenario())
    tree = next(iter(broker.envs.values()))["files"]["main"]
    assert tree["/opt/renamed.txt"][1] == b"data"
    assert tree["/logs/agent/local.txt"][1] == b"data"
    assert tree["/new/dir/local.txt"][1] == b"data"


def test_upload_dir_download_dir_and_kind_checks(broker, tmp_path):
    environment = started(broker, tmp_path)
    local = tmp_path / "tests"
    (local / "sub").mkdir(parents=True)
    (local / "test.sh").write_text("#!/bin/sh\n")
    (local / "sub" / "data.json").write_text("{}")
    target = tmp_path / "down"

    async def scenario():
        await environment.upload_dir(local, "/tests")
        await environment.download_dir("/tests", target)
        return (
            await environment.is_dir("/tests/sub"),
            await environment.is_file("/tests/test.sh"),
            await environment.is_file("/tests/missing"),
        )

    assert run(scenario()) == (True, True, False)
    assert (target / "test.sh").read_text() == "#!/bin/sh\n"
    assert (target / "sub" / "data.json").read_text() == "{}"
    assert broker.meta("path_stat")[-1]["follow"] is True


def test_filtered_and_excluded_downloads_match_harbor(broker, tmp_path):
    environment = started(broker, tmp_path)
    tree = next(iter(broker.envs.values()))["files"]["main"]
    for path in (
        "/logs/verifier/reward.txt",
        "/logs/verifier/big/blob.bin",
        "/logs/verifier/keep.log",
        "/logs/verifier/trace.log",
    ):
        tree[path] = ("file", path.encode(), 0o644)
    tree["/logs/verifier/big"] = ("dir", b"", 0o755)

    async def scenario():
        await environment.download_dir_filtered(
            source_dir="/logs/verifier",
            target_dir=tmp_path / "filtered",
            include=["*.log"],
            exclude=["trace.*"],
            protect=["reward.txt"],
        )
        await environment.download_dir_with_exclusions(
            source_dir="/logs/verifier",
            target_dir=tmp_path / "excluded",
            exclude=["big"],
        )
        await environment.download_dir_filtered(
            source_dir="/logs/verifier",
            target_dir=tmp_path / "none",
            include=["*.nothing"],
        )

    run(scenario())
    filtered = {
        str(path.relative_to(tmp_path / "filtered"))
        for path in (tmp_path / "filtered").rglob("*")
        if path.is_file()
    }
    # Harbor: include narrows, exclude wins, protect is always kept.
    assert filtered == {"keep.log", "reward.txt"}
    excluded = {
        str(path.relative_to(tmp_path / "excluded"))
        for path in (tmp_path / "excluded").rglob("*")
    }
    assert excluded == {"keep.log", "reward.txt", "trace.log"}
    assert broker.meta("copy_out")[1]["exclude"] == ["big"]
    assert (tmp_path / "none").is_dir() and not any((tmp_path / "none").iterdir())


def test_download_file_refuses_a_directory(broker, tmp_path):
    environment = started(broker, tmp_path)
    with pytest.raises(ManagedSandboxError, match="not a regular file"):
        run(environment.download_file("/logs", tmp_path / "x"))


def test_operations_after_stop_are_refused(broker, tmp_path):
    environment = started(broker, tmp_path)
    run(environment.stop(delete=True))
    with pytest.raises(RuntimeError, match="stopping or stopped"):
        run(environment.exec("true"))
    # stop is idempotent.
    run(environment.stop(delete=True))
    assert broker.ops("env_destroy") == ["env_destroy"]


def test_a_lost_pull_reply_is_replayed_and_the_job_cancelled(broker, tmp_path):
    environment = make_env(broker, tmp_path)
    original = broker.op_image_pull

    def lost(meta, payload):
        # The broker started the job; its first reply never reached the
        # client. A replay of the same request_id returns the recorded job.
        first = meta["request_id"] not in broker.replies
        result = original(meta, payload)
        if first:
            raise Failure("unknown-outcome", "request")
        return result

    broker.op_image_pull = lost
    with pytest.raises(ManagedSandboxError, match="unknown-outcome"):
        run(environment.start(force_build=False))
    pulls = broker.meta("image_pull")
    assert len(pulls) == 2 and pulls[0]["request_id"] == pulls[1]["request_id"]
    [cancel] = broker.meta("job_cancel")
    assert cancel["job_id"] == broker.replies[pulls[0]["request_id"]]["job_id"]
    assert broker.ops("env_create") == []
