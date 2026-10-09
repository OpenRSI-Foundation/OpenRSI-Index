"""Caller-side compose front-end: golden translations, interpolation, merge
order and one refusal per rejected key (spec 6)."""

import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

from rsi_harness.integrations import sandbox_compose as compose
from rsi_harness.integrations.sandbox_compose import (
    ComposeError,
    ImageBuild,
    ImagePull,
    interpolate,
    load_project,
    translate,
    translation_json,
)
from rsi_harness.runtime.sandbox_env_contracts import parse_env_spec

FIXTURES = Path(__file__).parents[1] / "fixtures" / "compose"
GOLDEN = (
    "ancient-puzzle",
    "security-vulhub-minio",
    "home-server-https",
    "postgres-csv-clean",
    "simple-web-scraper",
    "interactive-maze-game",
)
# What the Terminal-Bench 1 harness exported to `docker compose`.
TB1_ENV = {
    "T_BENCH_TASK_LOGS_PATH": "/host/logs",
    "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
    "T_BENCH_TASK_AGENT_LOGS_PATH": "/host/agent-logs",
    "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
    "T_BENCH_TEST_DIR": "/tests",
    "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": "tb__client",
    "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": "tb-client",
    "T_BENCH_TASK_DOCKER_NAME_PREFIX": "tb",
}


def relative(node, root):
    if isinstance(node, str):
        if node == root:
            return "."
        return "./" + node[len(root) + 1 :] if node.startswith(root + "/") else node
    if isinstance(node, list):
        return [relative(item, root) for item in node]
    if isinstance(node, dict):
        return {key: relative(value, root) for key, value in node.items()}
    return node


def handles(translation):
    return {
        name: "i" + format(index, "032x")
        for index, name in enumerate(translation.services)
    }


def project(tmp_path, text, *, environ=None, name="docker-compose.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return load_project([path], environ=environ or {}, project_dir=tmp_path)


def translated(tmp_path, text, **options):
    environ = options.pop("environ", None)
    loaded = project(tmp_path, text, environ=environ)
    return translate(
        loaded,
        project_dir=tmp_path,
        network=options.pop("network", "public"),
        disk_mb=options.pop("disk_mb", 1024),
        **options,
    )


@pytest.mark.parametrize("name", GOLDEN)
def test_terminal_bench_1_compose_files_translate_to_the_golden_env_spec(name):
    root = FIXTURES / name
    loaded = load_project(
        [root / "docker-compose.yaml"], environ=TB1_ENV, project_dir=root
    )
    result = translate(
        loaded,
        project_dir=root,
        network="public",
        disk_mb=2048,
        drop_targets=("/logs", "/agent-logs"),
    )
    actual = relative(translation_json(result), str(root.resolve()))
    expected = json.loads((root / "expected.json").read_text())
    assert json.loads(json.dumps(actual, sort_keys=True)) == expected
    # Bound to image handles, every golden spec passes the broker's own check.
    spec = parse_env_spec(result.env_spec(handles(result)))
    assert set(spec.services) == set(result.services)


def test_golden_highlights_are_the_translated_semantics():
    root = FIXTURES / "postgres-csv-clean"
    loaded = load_project(
        [root / "docker-compose.yaml"], environ=TB1_ENV, project_dir=root
    )
    result = translate(
        loaded,
        project_dir=root,
        network="none",
        disk_mb=2048,
        drop_targets=("/logs", "/agent-logs"),
    )
    db = result.spec["services"]["db"]
    # container_name becomes a DNS alias; service_healthy keeps its check.
    assert db["aliases"] == ["postgres_db"]
    assert db["healthcheck"]["test"] == [
        "CMD-SHELL",
        "pg_isready -U postgres -d customers_db",
    ]
    assert result.spec["services"]["client"]["depends_on"] == {
        "db": {"condition": "healthy", "required": True}
    }
    # ./init becomes a seeded volume, filled once before start.
    seeded = [name for name, item in result.spec["volumes"].items() if item["seeded"]]
    assert len(seeded) == 2
    assert {seed.dest_dir for seed in result.seeds} == {
        "/docker-entrypoint-initdb.d",
        "/app/data",
    }
    assert isinstance(result.images["db"], ImagePull)
    assert isinstance(result.images["client"], ImageBuild)
    # The TB1 host log bind has no dropped target here: it is refused.
    with pytest.raises(ComposeError, match="volumes./logs: host path"):
        translate(loaded, project_dir=root, network="none", disk_mb=2048)


# -- interpolation -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$$HOME", "$HOME"),
        ("a$$$$b", "a$$b"),
        ("${SET}", "value"),
        ("$SET-x", "value-x"),
        ("${UNSET}", ""),
        ("${UNSET:-fallback}", "fallback"),
        ("${EMPTY:-fallback}", "fallback"),
        ("${EMPTY-fallback}", ""),
        ("${UNSET-fallback}", "fallback"),
        ("${SET:+alt}", "alt"),
        ("${EMPTY:+alt}", ""),
        ("${EMPTY+alt}", "alt"),
        ("${UNSET:-${SET}}", "value"),
        ("${UNSET:-${ALSO:-deep}}", "deep"),
        ("cost $5", "cost $5"),
    ],
)
def test_interpolation_follows_compose_syntax(text, expected):
    environ = {"SET": "value", "EMPTY": ""}
    assert interpolate(text, environ, "x") == expected


@pytest.mark.parametrize(
    "text", ["${UNSET:?needs a value}", "${UNSET?needs a value}", "${EMPTY:?}"]
)
def test_required_variables_fail_with_their_message(text):
    with pytest.raises(ComposeError):
        interpolate(text, {"EMPTY": ""}, "services.main.image")


def test_dotenv_sits_below_the_process_environment(tmp_path):
    (tmp_path / ".env").write_text(
        "# comment\n"
        "export TAG=from-dotenv\n"
        "ONLY_DOTENV='literal $NOT'\n"
        'QUOTED="two words"\n'
        "DERIVED=${TAG}-x # trailing comment\n"
    )
    result = translated(
        tmp_path,
        """
services:
  main:
    image: busybox:${TAG}
    environment:
      A: ${ONLY_DOTENV}
      B: ${QUOTED}
      C: ${DERIVED}
      D: $${KEEP}
""",
        environ={"TAG": "from-process"},
    )
    assert result.images["main"] == ImagePull("busybox:from-process", "missing")
    assert result.spec["services"]["main"]["env"] == {
        "A": "literal $NOT",
        "B": "two words",
        # .env values interpolate while the file is read (the process wins).
        "C": "from-process-x",
        "D": "${KEEP}",
    }


def test_env_file_is_below_environment_and_bare_keys_pass_through(tmp_path):
    (tmp_path / "service.env").write_text("A=file\nB=file\n")
    result = translated(
        tmp_path,
        """
services:
  main:
    image: busybox
    env_file: service.env
    environment:
      - B=inline
      - FROM_SHELL
      - MISSING
""",
        environ={"FROM_SHELL": "shell"},
    )
    assert result.spec["services"]["main"]["env"] == {
        "A": "file",
        "B": "inline",
        "FROM_SHELL": "shell",
    }


# -- merge order ----------------------------------------------------------------------


def test_layers_merge_in_order_with_compose_rules(tmp_path):
    (tmp_path / "base.yaml").write_text(
        """
services:
  main:
    image: busybox
    command: ["sh", "-c", "sleep infinity"]
    environment: {A: base, B: base}
    cap_drop: [NET_ADMIN]
    volumes: ["data:/data", "logs:/logs"]
    depends_on: [kv]
  kv:
    image: redis:7-alpine
volumes: {data: {}, logs: {}}
"""
    )
    override = {
        "services": {
            "main": {
                "command": "python -m server --port 1",
                "environment": {"B": "override"},
                "cap_drop": ["SYS_ADMIN"],
                "volumes": ["./cache:/data"],
                "depends_on": {"kv": {"condition": "service_healthy"}},
            },
            "kv": {"healthcheck": {"test": ["CMD", "redis-cli", "ping"]}},
        }
    }
    (tmp_path / "cache").mkdir()
    loaded = load_project(
        [tmp_path / "base.yaml", override], environ={}, project_dir=tmp_path
    )
    result = translate(loaded, project_dir=tmp_path, network="none", disk_mb=512)
    main = result.spec["services"]["main"]
    assert main["command"] == ["python", "-m", "server", "--port", "1"]
    assert main["env"] == {"A": "base", "B": "override"}
    assert main["cap_drop"] == ["CAP_NET_ADMIN", "CAP_SYS_ADMIN"]
    # Volumes merge by target: the bind replaced the named volume at /data.
    targets = {mount["target"]: mount["volume"] for mount in main["mounts"]}
    assert targets["/logs"] == "logs"
    assert targets["/data"].startswith("seed-")
    assert main["depends_on"] == {"kv": {"condition": "healthy", "required": True}}


# -- translation details ---------------------------------------------------------


def test_resources_healthcheck_and_network_details(tmp_path):
    result = translated(
        tmp_path,
        """
services:
  main:
    image: busybox
    cpus: 0.333
    mem_limit: 512m
    pids_limit: 128
    shm_size: 32m
    tmpfs: ["/scratch:size=16m", "/run/app"]
    ulimits: {nofile: {soft: 1024, hard: 4096}, nproc: 64}
    stop_signal: INT
    stop_grace_period: 1m30s
    healthcheck:
      test: curl -f http://localhost
      interval: 1m30s
      timeout: 500ms
      retries: 5
    networks:
      default:
        aliases: [web]
    links: ["db:database"]
  db:
    image: postgres:16
    deploy: {resources: {limits: {cpus: "2", memory: 1G}}}
    network_mode: none
""".replace("1m30s\n    healthcheck", "20s\n    healthcheck"),
    )
    main = result.spec["services"]["main"]
    assert (main["cpus"], main["memory_mb"], main["pids"], main["shm_mb"]) == (
        0.34,
        512,
        128,
        32,
    )
    assert main["tmpfs"] == {"/scratch": 16, "/run/app": 64}
    assert main["nofile"] == 4096
    assert (main["stop_signal"], main["stop_grace_sec"]) == ("SIGINT", 20)
    assert main["healthcheck"] == {
        "test": ["CMD-SHELL", "curl -f http://localhost"],
        "interval_sec": 90.0,
        "timeout_sec": 0.5,
        "retries": 5,
    }
    assert main["aliases"] == ["web"]
    assert main["depends_on"] == {"db": {"condition": "started", "required": True}}
    db = result.spec["services"]["db"]
    assert (db["cpus"], db["memory_mb"], db["network"]) == (2.0, 1024, "none")
    # A link alias names its target; a network-less target keeps no alias.
    assert "aliases" not in db
    notes = "\n".join(result.notes)
    assert "cpus: 0.333 rounds up to 0.34" in notes
    assert "nproc" in notes and "soft becomes the hard limit" in notes


def test_healthy_dependency_on_an_image_check_uses_the_image(tmp_path):
    result = translated(
        tmp_path,
        """
services:
  main:
    image: busybox
    depends_on: {db: {condition: service_healthy, required: false}}
  db:
    image: postgres:16
""",
    )
    assert result.spec["services"]["db"]["healthcheck"] == "image"
    assert result.spec["services"]["main"]["depends_on"]["db"]["required"] is False


def test_profiles_configs_and_secrets(tmp_path):
    (tmp_path / "app.conf").write_text("conf\n")
    result = translated(
        tmp_path,
        """
services:
  main:
    image: busybox
    configs: [app, {source: inline, target: /etc/inline.txt, mode: "0600"}]
    secrets: [token]
  debug:
    image: busybox
    profiles: [debug]
configs:
  app: {file: ./app.conf}
  inline: {content: "hello"}
secrets:
  token: {environment: TOKEN}
""",
        environ={"TOKEN": "s3cret"},
    )
    assert result.services == ("main",)
    seeds = {(seed.dest_dir, seed.name): seed for seed in result.seeds}
    assert seeds[("/", "app")].source == str((tmp_path / "app.conf").resolve())
    assert seeds[("/etc", "inline.txt")].content == b"hello"
    assert seeds[("/etc", "inline.txt")].mode == 0o600
    assert seeds[("/run/secrets", "token")].content == b"s3cret"
    enabled = translated(
        tmp_path,
        (tmp_path / "docker-compose.yaml").read_text(),
        environ={"TOKEN": "x", "COMPOSE_PROFILES": "debug"},
    )
    assert enabled.services == ("debug", "main")


def test_seed_archives_hold_regular_entries_only(tmp_path):
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "a.txt").write_text("a")
    (tree / "link").symlink_to("sub/a.txt")
    (tree / "hard").hardlink_to(tree / "sub" / "a.txt")
    buffer = io.BytesIO()
    compose.write_seed_archive(
        compose.Seed(service="main", dest_dir="/x", kind="dir", source=str(tree)),
        buffer,
    )
    buffer.seek(0)
    with tarfile.open(fileobj=buffer) as archive:
        members = {member.name: member for member in archive}
    assert set(members) == {"hard", "link", "sub", "sub/a.txt"}
    assert members["hard"].isreg() and members["link"].issym()
    buffer = io.BytesIO()
    compose.write_seed_archive(
        compose.Seed(
            service="main", dest_dir="/x", kind="file", name="f", content=b"data"
        ),
        buffer,
    )
    buffer.seek(0)
    with tarfile.open(fileobj=buffer) as archive:
        assert archive.extractfile("f").read() == b"data"


def test_json_compose_loads_without_pyyaml(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    path = tmp_path / "compose.json"
    path.write_text(json.dumps({"services": {"main": {"image": "busybox"}}}))
    loaded = load_project([path], environ={}, project_dir=tmp_path)
    assert loaded["services"]["main"]["image"] == "busybox"
    path.write_text("services: {main: {image: busybox}}")
    with pytest.raises(ComposeError, match="PyYAML is unavailable"):
        load_project([path], environ={}, project_dir=tmp_path)


# -- refusals --------------------------------------------------------------------------

REJECTED = [
    ("privileged: true", "services.main.privileged"),
    ("cap_add: [NET_ADMIN]", "services.main.cap_add"),
    ("devices: ['/dev/fuse:/dev/fuse']", "services.main.devices"),
    ("device_cgroup_rules: ['c 1:3 mr']", "services.main.device_cgroup_rules"),
    ("gpus: all", "services.main.gpus"),
    ("runtime: nvidia", "services.main.runtime"),
    ("isolation: process", "services.main.isolation"),
    ("security_opt: ['seccomp=unconfined']", "services.main.security_opt"),
    ("security_opt: ['apparmor=unconfined']", "services.main.security_opt"),
    ("sysctls: {net.ipv4.ip_forward: 1}", "services.main.sysctls"),
    ("oom_kill_disable: true", "services.main.oom_kill_disable"),
    ("storage_opt: {size: 1G}", "services.main.storage_opt"),
    ("blkio_config: {weight: 300}", "services.main.blkio_config"),
    ("cpuset: '0-1'", "services.main.cpuset"),
    ("cpu_quota: 50000", "services.main.cpu_quota"),
    ("cpu_period: 100000", "services.main.cpu_period"),
    ("cpu_rt_runtime: 400ms", "services.main.cpu_rt_runtime"),
    ("pid: host", "services.main.pid"),
    ("ipc: host", "services.main.ipc"),
    ("uts: host", "services.main.uts"),
    ("userns_mode: host", "services.main.userns_mode"),
    ("cgroup: host", "services.main.cgroup"),
    ("cgroup_parent: m-executor", "services.main.cgroup_parent"),
    ("network_mode: host", "services.main.network_mode"),
    ("network_mode: bridge", "services.main.network_mode"),
    ("network_mode: 'container:other'", "services.main.network_mode"),
    ("network_mode: 'service:other'", "services.main.network_mode"),
    ("volumes_from: [other]", "services.main.volumes_from"),
    ("external_links: [other]", "services.main.external_links"),
    ("volumes: ['/var/run/docker.sock:/var/run/docker.sock']", "host path"),
    ("volumes: ['/etc:/host-etc:ro']", "host path"),
    ("volumes: ['~/secrets:/secrets']", "host path"),
    ("volumes: ['../outside:/x']", "leaves the project"),
    (
        "volumes: [{type: npipe, source: pipe, target: /pipe}]",
        "npipe mounts are refused",
    ),
    (
        "volumes: [{type: volume, source: data, target: /d, volume: {subpath: a}}]",
        "subpath",
    ),
    ("dns: [8.8.8.8]", "services.main.dns"),
    ("dns_search: [example.com]", "services.main.dns_search"),
    ("dns_opt: [use-vc]", "services.main.dns_opt"),
    ("mac_address: 02:42:ac:11:65:43", "services.main.mac_address"),
    (
        "networks: {default: {ipv4_address: 172.16.238.10}}",
        "services.main.networks.default.ipv4_address",
    ),
    (
        "networks: {default: {link_local_ips: [169.254.8.8]}}",
        "services.main.networks.default.link_local_ips",
    ),
    ("extra_hosts: ['host.docker.internal:host-gateway']", "host-gateway"),
    ("extra_hosts: ['db:example.com']", "services.main.extra_hosts.db: expected an IP"),
    ("extra_hosts: ['db=fe80::1%eth0']", "services.main.extra_hosts.db: scoped"),
    ("domainname: example.com", "services.main.domainname"),
    ("scale: 2", "services.main.scale"),
    ("deploy: {replicas: 3}", "services.main.deploy.replicas"),
    (
        "deploy: {resources: {reservations: {devices: [{capabilities: [gpu]}]}}}",
        "reservations.devices",
    ),
    ("use_api_socket: true", "services.main.use_api_socket"),
    ("provider: {type: model}", "services.main.provider"),
    ("models: [llm]", "services.main.models"),
    ("post_start: [{command: id}]", "services.main.post_start"),
    ("pre_stop: [{command: id}]", "services.main.pre_stop"),
    ("extends: {service: base}", "services.main.extends"),
    ("ulimits: {memlock: -1}", "services.main.ulimits.memlock"),
    ("platform: linux/arm64", "services.main.platform"),
    ("stop_grace_period: 1m", "services.main.stop_grace_period"),
    ("environment: {NVIDIA_VISIBLE_DEVICES: all}", "reserved key"),
    ("build: {context: ., secrets: [token]}", "services.main.build.secrets"),
    ("build: {context: ., ssh: [default]}", "services.main.build.ssh"),
    ("build: {context: ., network: host}", "services.main.build.network"),
    (
        "build: {context: ., additional_contexts: {x: ../x}}",
        "services.main.build.additional_contexts",
    ),
    ("build: {context: ., cache_from: [x]}", "services.main.build.cache_from"),
    ("build: {context: ., cache_to: [x]}", "services.main.build.cache_to"),
    ("build: {context: ., entitlements: [network.host]}", "build.entitlements"),
    ("build: {context: ., privileged: true}", "services.main.build.privileged"),
    ("build: {context: 'https://github.com/x/y.git'}", "only local contexts"),
    ("tmpfs: ['/proc']", "services.main.tmpfs: reserved mount target /proc"),
    ("tmpfs: ['/dev/shm:size=8m']", "reserved mount target /dev/shm"),
    ("volumes: ['data:/sys/fs']", "reserved mount target /sys/fs"),
    ("volumes: ['./x:/run/rsi-harness/sandbox']", "reserved mount target"),
    ("volumes: [{type: tmpfs, target: /}]", "reserved mount target /"),
]


@pytest.mark.parametrize(("fragment", "message"), REJECTED)
def test_every_rejected_service_key_is_refused_with_its_path(
    tmp_path, fragment, message
):
    text = (
        "services:\n  main:\n    image: busybox\n    "
        + fragment
        + "\n  other:\n    image: busybox\nvolumes: {data: {}}\n"
    )
    with pytest.raises(ComposeError, match=message.replace(".", r"\.")):
        translated(tmp_path, text)


def test_reserved_mount_roots_are_the_broker_s():
    from rsi_harness.runtime import sandbox_env_contracts

    assert compose.RESERVED_MOUNT_ROOTS == sandbox_env_contracts._RESERVED_MOUNT_ROOTS


def test_extra_hosts_keep_ip_literals(tmp_path):
    result = translated(
        tmp_path,
        "services: {main: {image: busybox, extra_hosts: "
        "['db:10.0.0.5', 'v6=[2001:db8::1]']}}",
    )
    assert result.spec["services"]["main"]["extra_hosts"] == [
        ["db", "10.0.0.5"],
        ["v6", "2001:db8::1"],
    ]
    parse_env_spec(result.env_spec(handles(result)))


def test_normalizations_are_notes_never_silent(tmp_path):
    result = translated(
        tmp_path,
        """
services:
  main:
    image: busybox
    restart: always
    expose: ["8080"]
    mem_limit: 256m
    memswap_limit: 512m
    labels: {team: x}
    logging: {driver: none}
    stdin_open: true
    deploy: {labels: {team: x}, restart_policy: {condition: any}}
""",
    )
    notes = [note for note in result.notes if note.startswith("services.main.")]
    assert notes == [
        "services.main.labels: ignored",
        "services.main.logging: ignored",
        "services.main.stdin_open: ignored",
        "services.main.restart: 'always' becomes 'no'",
        "services.main.expose: dropped; services reach each other on container ports",
        "services.main.deploy.restart_policy: restart is always 'no'",
        "services.main.deploy.labels: ignored",
        "services.main.memswap_limit: swap is the operator's swap_ratio",
    ]
    main = result.spec["services"]["main"]
    assert "restart" not in main and main["memory_mb"] == 256


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("include: [other.yaml]\nservices: {main: {image: busybox}}", "include"),
        ("models: {llm: {model: x}}\nservices: {main: {image: busybox}}", "models"),
        (
            "services: {main: {image: busybox, volumes: [data:/d]}}\n"
            "volumes: {data: {driver: rexray}}",
            "volumes.data.driver",
        ),
        (
            "services: {main: {image: busybox, volumes: [data:/d]}}\n"
            "volumes: {data: {driver_opts: {type: none, o: bind, device: /}}}",
            "volumes.data.driver_opts",
        ),
        (
            "services: {main: {image: busybox, volumes: [data:/d]}}\n"
            "volumes: {data: {external: true}}",
            "volumes.data.external",
        ),
        (
            "services: {main: {image: busybox, networks: [net]}}\n"
            "networks: {net: {external: true}}",
            "networks.net.external",
        ),
        (
            "services: {main: {image: busybox, networks: [net]}}\n"
            "networks: {net: {internal: true}}",
            "networks.net.internal",
        ),
        (
            "services: {main: {image: busybox, networks: [net]}}\n"
            "networks: {net: {driver: macvlan}}",
            "networks.net.driver",
        ),
        (
            "services: {main: {image: busybox, networks: [net]}}\n"
            "networks: {net: {ipam: {config: [{subnet: 10.0.0.0/24}]}}}",
            "networks.net.ipam",
        ),
        (
            "services: {main: {image: busybox, networks: [a, b]}}\n"
            "networks: {a: {}, b: {}}",
            "more than one network",
        ),
        (
            "services:\n  main: {image: busybox, networks: [a]}\n"
            "  kv: {image: busybox, networks: [b]}\nnetworks: {a: {}, b: {}}",
            "different networks",
        ),
        (
            "services: {main: {image: busybox, secrets: [s]}}\n"
            "secrets: {s: {external: true}}",
            "external secrets",
        ),
        ("services: {main: {image: !reset null}}", "tag"),
        ("services: {main: {image: !override busybox}}", "tag"),
        ("services: {main: {image: !!python/object:os.system x}}", "tag"),
        (
            "services:\n  main: {image: busybox, depends_on: [kv]}\n"
            "  kv: {image: busybox, profiles: [off]}",
            "not enabled",
        ),
        (
            "services:\n  main: {image: busybox, container_name: kv}\n"
            "  kv: {image: busybox}",
            "already names",
        ),
        (
            "services:\n  main: {image: busybox, depends_on: "
            "{kv: {condition: service_healthy}}}\n"
            "  kv: {image: busybox, healthcheck: {disable: true}}",
            "depends on it being healthy",
        ),
        ("services: {main: {}}", "needs an image or a build"),
        (
            "services: {main: {image: busybox, volumes: "
            + str([f"/v{index}" for index in range(9)]).replace("'", "")
            + "}}",
            "at most 8 volumes",
        ),
        (
            "services: {main: {image: busybox, mem_limit: 64m, shm_size: 64m, "
            "tmpfs: ['/t:size=8m']}}",
            "exceed the memory limit",
        ),
    ],
)
def test_project_level_refusals(tmp_path, text, message):
    with pytest.raises(ComposeError, match=message):
        translated(tmp_path, text)


def test_log_mounts_the_caller_downloads_are_dropped_with_a_note(tmp_path):
    result = translated(
        tmp_path,
        """
services:
  main:
    image: busybox
    volumes:
      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
""",
        environ={
            "HOST_VERIFIER_LOGS_PATH": "/logs/verifier",
            "ENV_VERIFIER_LOGS_PATH": "/logs/verifier",
        },
        drop_targets=("/logs/verifier",),
    )
    assert "mounts" not in result.spec["services"]["main"]
    assert any("log mount dropped" in note for note in result.notes)


def test_an_allowlist_comes_from_the_caller_and_x_rsi_allowlist(tmp_path):
    text = "services: {main: {image: busybox}}"
    result = translated(
        tmp_path,
        "x-rsi-allowlist: [pypi.org:443, 8.8.8.8]\n" + text,
        network="allowlist",
        allowlist=["8.8.8.8", "example.com"],
    )

    assert (result.spec["network"], result.spec["allowlist"]) == (
        "allowlist",
        ["8.8.8.8", "example.com", "pypi.org:443"],
    )
    assert parse_env_spec(result.env_spec(handles(result))).allowlist == (
        "8.8.8.8",
        "example.com",
        "pypi.org:443",
    )
    assert "allowlist" not in translated(tmp_path, text).spec
    assert translated(tmp_path, text, network="allowlist").spec["allowlist"] == []
    with pytest.raises(ComposeError, match="need network allowlist, not public"):
        translated(tmp_path, "x-rsi-allowlist: [pypi.org]\n" + text)
    with pytest.raises(ComposeError, match="need network allowlist, not none"):
        translated(tmp_path, text, network="none", allowlist=["pypi.org"])
    with pytest.raises(ComposeError, match="x-rsi-allowlist: expected a list"):
        translated(tmp_path, "x-rsi-allowlist: pypi.org\n" + text)
    with pytest.raises(ComposeError, match="public, none or allowlist"):
        translated(tmp_path, text, network="host")


def test_cli_compose_runs_an_x_rsi_allowlist_project_as_an_allowlist_env(
    tmp_path, monkeypatch, capsys
):
    from rsi_harness.integrations import sandbox_client
    from tests.integrations.test_harbor_env_plugin import FakeBroker

    (tmp_path / "compose.yaml").write_text(
        "x-rsi-allowlist: [pypi.org, '8.8.8.8:53']\n"
        "services: {main: {image: busybox}}\n"
    )
    with tempfile.TemporaryDirectory(prefix="rsi-cli-") as root:
        broker = FakeBroker(Path(root) / "s", network=("allowlist", "public"))
        monkeypatch.setenv("RSI_SANDBOX_SOCKET", str(broker.path))
        monkeypatch.setenv("RSI_SANDBOX_TOKEN", "token")
        base = ["compose", "--project-directory", str(tmp_path)]
        try:
            assert sandbox_client.main([*base, "--network", "allowlist", "config"]) == 0
            spec = json.loads(capsys.readouterr().out)["spec"]
            assert (spec["network"], spec["allowlist"]) == (
                "allowlist",
                ["pypi.org", "8.8.8.8:53"],
            )
            # Never silently another network: the default (public) is refused.
            assert sandbox_client.main([*base, "config"]) == 2
            assert "need network allowlist" in capsys.readouterr().err
        finally:
            broker.close()


def test_front_end_imports_nothing_from_the_harness():
    source = Path(compose.__file__).read_text()
    assert "rsi_harness" not in source.replace("``rsi_harness``", "")


# -- rsi-sandbox compose --------------------------------------------------------------


def test_cli_compose_runs_a_project_as_one_env(tmp_path, monkeypatch, capsys):
    from rsi_harness.integrations import sandbox_client
    from tests.integrations.test_harbor_env_plugin import FakeBroker

    project_dir = tmp_path / "proj"
    (project_dir / "seed").mkdir(parents=True)
    (project_dir / "seed" / "value.txt").write_text("42\n")
    (project_dir / "compose.yaml").write_text(
        """
services:
  app:
    image: busybox
    command: sleep infinity
    volumes: ["./seed/value.txt:/seed/value.txt"]
    depends_on: {kv: {condition: service_healthy}}
  kv:
    image: redis:7-alpine
    healthcheck: {test: [CMD, redis-cli, ping]}
"""
    )
    monkeypatch.setattr(sandbox_client.tempfile, "gettempdir", lambda: str(tmp_path))
    with tempfile.TemporaryDirectory(prefix="rsi-cli-") as root:
        broker = FakeBroker(Path(root) / "s", network=("none",))
        monkeypatch.setenv("RSI_SANDBOX_SOCKET", str(broker.path))
        monkeypatch.setenv("RSI_SANDBOX_TOKEN", "token")
        base = ["compose", "--project-directory", str(project_dir)]
        try:
            assert sandbox_client.main(base + ["config"]) == 0
            config = json.loads(capsys.readouterr().out)
            assert config["spec"]["network"] == "none"
            assert sandbox_client.main(base + ["up"]) == 0
            assert json.loads(capsys.readouterr().out)["state"] == "ready"
            session = hashlib.sha256(b"token").hexdigest()[:12]
            state = tmp_path / f"rsi-sandbox-compose-{os.getuid()}-{session}-proj.json"
            assert state.stat().st_mode & 0o777 == 0o600
            assert sandbox_client.main(base + ["up"]) == 2  # already up
            # Work's project is not its Judge's: another session sees none.
            monkeypatch.setenv("RSI_SANDBOX_TOKEN", "judge-token")
            assert sandbox_client.main(base + ["ps"]) == 2
            assert "is not up" in capsys.readouterr().err
            monkeypatch.setenv("RSI_SANDBOX_TOKEN", "token")
            broker.on_exec = lambda meta: {"output": b"pong\n", "exit_code": 0}
            assert (
                sandbox_client.main(base + ["exec", "kv", "--", "redis-cli", "ping"])
                == 0
            )
            assert capsys.readouterr().out.endswith("pong\n")
            assert (
                sandbox_client.main(
                    base + ["cp", "app:/seed/value.txt", str(tmp_path / "out")]
                )
                == 0
            )
            assert (tmp_path / "out" / "value.txt").read_text() == "42\n"
            assert sandbox_client.main(base + ["stop", "app"]) == 0
            assert sandbox_client.main(base + ["ps"]) == 0
            assert sandbox_client.main(base + ["down"]) == 0
            assert not state.exists()
            # A planted link is never followed.
            state.symlink_to(tmp_path / "out" / "value.txt")
            assert sandbox_client.main(base + ["ps"]) == 2
            assert sandbox_client.main(base + ["up"]) == 2
            assert state.is_symlink()
        finally:
            broker.close()
    ops = broker.ops("image_pull", "env_create", "copy_in", "env_start", "env_destroy")
    assert ops == [
        "image_pull",
        "image_pull",
        "env_create",
        "copy_in",
        "env_start",
        "env_destroy",
    ]
    exec_meta = broker.meta("exec_start")[0]
    assert (exec_meta["service"], exec_meta["argv"]) == ("kv", ["redis-cli", "ping"])


@pytest.mark.parametrize("compose_up", [False, True])
def test_cli_up_waits_only_for_what_is_left_of_the_env_lifetime(
    tmp_path, monkeypatch, capsys, compose_up
):
    """Spec A8: a Judge's envs live less than the default 300 s wait. The
    default is capped at what is left, an explicit wait is honoured or
    refused before env_start (and the env destroyed), and a lifetime refusal
    of the default wait (slot queueing spent the slack) is resent once."""
    from rsi_harness.integrations import sandbox_client
    from tests.integrations.test_harbor_env_plugin import Failure, FakeBroker

    (tmp_path / "compose.yaml").write_text("services: {main: {image: busybox}}")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"services": {"main": {"image": "i" + "0" * 32}}}))
    monkeypatch.setattr(sandbox_client.tempfile, "gettempdir", lambda: str(tmp_path))

    def up(*options):
        if compose_up:
            base = ["compose", "--project-directory", str(tmp_path), *options]
            code = sandbox_client.main([*base, "up"])
            assert sandbox_client.main([*base, "down"]) == 0  # even a refused env
            return code
        return sandbox_client.main(["up", str(spec), *options])

    with tempfile.TemporaryDirectory(prefix="rsi-cli-") as root:
        broker = FakeBroker(Path(root) / "s", network=("none",))
        broker.expires_in_sec = 120.0
        monkeypatch.setenv("RSI_SANDBOX_SOCKET", str(broker.path))
        monkeypatch.setenv("RSI_SANDBOX_TOKEN", "token")
        try:
            assert up() == 0
            assert up("--wait-timeout", "60") == 0
            capsys.readouterr()
            assert up("--wait-timeout", "300") == 2
            error = json.loads(capsys.readouterr().err.splitlines()[-1])["error"]
            assert (error["code"], error["field"]) == ("quota", "wait_timeout_sec")
            assert "exceeds the env's remaining lifetime" in error["message"]
            lifetime = Failure(
                "quota", "wait_timeout_sec", "exceeds the env's remaining lifetime"
            )
            broker.fail["env_start"] = [lifetime]
            assert up() == 0
            broker.fail["env_start"] = [lifetime, lifetime]
            assert up() == 2
            assert "remaining lifetime" in capsys.readouterr().err
        finally:
            broker.close()
    waits = [meta["wait_timeout_sec"] for meta in broker.meta("env_start")]
    assert len(waits) == 6
    assert 115.0 < waits[0] < 120.0 - compose.START_SLACK_SEC and waits[1] == 60
    assert waits[3] <= waits[2] < 120.0 and waits[5] <= waits[4] < 120.0
    # The refused explicit wait never reached env_start; that env is gone.
    assert len(broker.meta("env_create")) == 5
    assert len(broker.meta("env_destroy")) == (6 if compose_up else 1)
    assert all(env["state"] == "removed" for env in broker.envs.values()) == compose_up
