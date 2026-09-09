from pathlib import PurePosixPath

import pytest

from rsi_harness.errors import UnsupportedTaskError
from rsi_harness.task.compose import ComposeMainServiceParser


def write_compose(tmp_path, text):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    path = environment_dir / "docker-compose.yaml"
    path.write_text(text)
    return path


def test_parse_extracts_supported_main_service_and_nvidia_requirement(tmp_path):
    """Dropping a supported main value would silently change its container."""
    path = write_compose(
        tmp_path,
        """
services:
  main:
    build: .
    working_dir: /workspace
    user: "1000:1000"
    environment:
      LITERAL: visible
      TOKEN: ${TOKEN}
      FROM_HOST:
    shm_size: 2g
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
""",
    )

    service = ComposeMainServiceParser().parse(path)

    assert service.build_context == path.parent.resolve()
    assert service.image is None
    assert service.workdir == PurePosixPath("/workspace")
    assert service.user == "1000:1000"
    assert service.environment == (
        ("FROM_HOST", "${FROM_HOST}"),
        ("LITERAL", "visible"),
        ("TOKEN", "${TOKEN}"),
    )
    assert service.shm_size == "2g"
    assert service.gpu_requirement is not None
    assert service.gpu_requirement.count == 2


@pytest.mark.parametrize(
    ("compose", "message"),
    (
        pytest.param(
            "services:\n  main:\n    build: .\n    image: task-image:latest\n",
            "build.*image",
            id="build-and-image",
        ),
        pytest.param(
            "services:\n  main:\n    build: ..\n",
            "build context.*environment",
            id="outside-build-context",
        ),
    ),
)
def test_parse_rejects_ambiguous_or_external_build_sources(tmp_path, compose, message):
    path = write_compose(tmp_path, compose)

    with pytest.raises(UnsupportedTaskError, match=message):
        ComposeMainServiceParser().parse(path)


@pytest.mark.parametrize("shm_size", ("0", "-1g", "1.5g", "enormous"))
def test_parse_rejects_invalid_shm_size(tmp_path, shm_size) -> None:
    path = write_compose(
        tmp_path,
        f"services:\n  main:\n    image: task-image\n    shm_size: {shm_size}\n",
    )

    with pytest.raises(UnsupportedTaskError, match="shm_size"):
        ComposeMainServiceParser().parse(path)


def test_parse_accepts_all_allocated_gpus_sentinel(tmp_path):
    """Converting 'all' to a host count would violate caller-owned allocation."""
    path = write_compose(
        tmp_path,
        """
services:
  main:
    image: task-image
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
""",
    )

    service = ComposeMainServiceParser().parse(path)

    assert service.gpu_requirement is not None
    assert service.gpu_requirement.count == "all"


@pytest.mark.parametrize("count", ("true", "false"))
def test_parse_rejects_boolean_gpu_counts(tmp_path, count):
    """YAML booleans must not be coerced into integer GPU requirements."""
    path = write_compose(
        tmp_path,
        f"""
services:
  main:
    image: task-image
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {count}
              capabilities: [gpu]
""",
    )

    with pytest.raises(UnsupportedTaskError, match="boolean"):
        ComposeMainServiceParser().parse(path)


def test_parse_accepts_docker_root_workdir(tmp_path):
    path = write_compose(
        tmp_path,
        "services:\n  main:\n    image: task-image\n    working_dir: /\n",
    )

    service = ComposeMainServiceParser().parse(path)

    assert service.workdir == PurePosixPath("/")


@pytest.mark.parametrize("workdir", ("workspace",))
def test_parse_rejects_invalid_workdir_with_typed_task_error(tmp_path, workdir):
    """Leaking model validation would break the compiler's setup-error contract."""
    path = write_compose(
        tmp_path,
        f"services:\n  main:\n    image: task-image\n    working_dir: {workdir}\n",
    )

    with pytest.raises(UnsupportedTaskError, match="WORKDIR"):
        ComposeMainServiceParser().parse(path)


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("volumes: ['./host:/data']", "host volume"),
        ("network_mode: host", "host network"),
        ("privileged: true", "privileged"),
        ("devices: ['/dev/sda:/dev/sda']", "device"),
        ("cap_add: [SYS_ADMIN]", "cap_add"),
        ("ipc: host", "ipc"),
        ("pid: host", "pid"),
        ("extra_hosts: ['host.docker.internal:host-gateway']", "extra_hosts"),
        ("links: ['db']", "links"),
        ("command: sleep infinity", "command"),
    ],
)
def test_parse_rejects_unsafe_or_silently_ignored_main_values(
    tmp_path, setting, message
):
    """Ignoring a declared Compose value would run different semantics."""
    path = write_compose(
        tmp_path,
        f"services:\n  main:\n    image: task-image\n    {setting}\n",
    )

    with pytest.raises(UnsupportedTaskError, match=message):
        ComposeMainServiceParser().parse(path)


@pytest.mark.parametrize(
    ("compose", "message"),
    (
        pytest.param(
            """
services:
  main:
    image: task-image
  database:
    image: postgres:17
""",
            "sidecar",
            id="sidecar",
        ),
        pytest.param(
            """
services:
  main:
    image: task-image
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['GPU-private']
              capabilities: [gpu]
""",
            "device_ids",
            id="task-selected-gpus",
        ),
        pytest.param(
            """
services:
  main:
    image: task-image
    deploy:
      resources:
        reservations:
          memory: 1g
""",
            "reservation",
            id="unknown-reservation",
        ),
    ),
)
def test_parse_rejects_unsupported_service_topology_or_reservations(
    tmp_path, compose, message
):
    path = write_compose(tmp_path, compose)

    with pytest.raises(UnsupportedTaskError, match=message):
        ComposeMainServiceParser().parse(path)


@pytest.mark.parametrize(
    ("context", "create_file", "message"),
    (
        ("missing-context", False, "existing local directory"),
        ("context-file", True, "existing local directory"),
        ("https://example.com/task.git", False, "remote build context"),
    ),
)
def test_parse_rejects_unbuildable_compose_contexts(
    tmp_path, context, create_file, message
):
    """A persisted build context must identify a buildable local directory."""
    path = write_compose(
        tmp_path,
        f"services:\n  main:\n    build: {context}\n",
    )
    if create_file:
        (path.parent / context).write_text("not a directory")

    with pytest.raises(UnsupportedTaskError, match=message):
        ComposeMainServiceParser().parse(path)
