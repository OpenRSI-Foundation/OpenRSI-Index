"""Real Docker: a Work container has no file watches.

The container is created as the Harness creates Work; every exec, as root
or not, and what it starts, finds inotify and fanotify unavailable (ENOSYS)
and dnotify and leases refused (EINVAL), also from 32-bit (i386) calls,
and the legacy socket families Docker's default denies refused, while
files, python and GNU tail -f (which then polls) work.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from rsi_harness.models import ContainerSpec
from rsi_harness.runtime.docker import DockerContainerRuntime
from tests.integration.test_judge_model_access_docker import PYTHON_IMAGE, _docker
from tests.judge_agent.test_watch_filter import COMPAT, IA32_PROBE, _run_probe

pytestmark = pytest.mark.integration

PROBE = r"""
import ctypes, errno, json, os, subprocess, tempfile, time
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
def call(*args):
    result = libc.syscall(*(ctypes.c_long(value) for value in args))
    return "ok" if result >= 0 else errno.errorcode[ctypes.get_errno()]
directory = tempfile.mkdtemp()
descriptor = os.open(directory, os.O_RDONLY)
found = {
    "inotify_init": call(253),
    "inotify_init1": call(294, 0),
    "inotify_add_watch": call(254, 0, 0, 0),
    "fanotify_init": call(300, 0x200, 0),
    "fanotify_mark": call(301, -1, 0, 0, 0, 0),
    "x32_inotify_init1": call(0x40000000 | 294, 0),
    "dnotify": call(72, descriptor, 1026, 1),
    "dnotify_high": call(72, descriptor, (1 << 32) | 1026, 1),
    "getfl": call(72, descriptor, 3, 0),
    "inet_socket": call(41, 2, 2, 0),
    # Families Docker's default denies since moby/profiles seccomp v0.3.0.
    "legacy_sockets": sorted({
        call(41, family, 2, 0)
        for family in (3, 4, 5, 6, 11, 12, 19, 20, 23, 25, 34, 37)
    }),
}
leased = os.path.join(directory, "leased")
open(leased, "w").close()
reader = os.open(leased, os.O_RDONLY)
found["lease"] = call(72, reader, 1024, 1)
found["lease_high"] = call(72, reader, (1 << 32) | 1024, 1)
found["getlease"] = call(72, reader, 1025, 0)
# F_SETDELEG on a bad descriptor: EBADF from a kernel without the profile.
found["deleg"] = call(72, -1, 1040, 0)
found["deleg_high"] = call(72, -1, (1 << 32) | 1040, 0)
path = os.path.join(directory, "log")
with open(path, "w") as stream:
    stream.write("a\n")
os.rename(path, path + ".1")
os.rename(path + ".1", path)
tail = subprocess.Popen(["tail", "-f", path], stdout=subprocess.PIPE, text=True)
time.sleep(0.5)
with open(path, "a") as stream:
    stream.write("b\n")
found["tail"] = [tail.stdout.readline(), tail.stdout.readline()]
tail.kill()
child = subprocess.run(
    ["python3", "-c", "import ctypes; l = ctypes.CDLL(None, use_errno=True); "
     "print(l.syscall(294, 0), ctypes.get_errno())"],
    capture_output=True, text=True,
)
found["child"] = child.stdout.split()
print(json.dumps(found))
"""


@pytest.mark.parametrize("user", [None, "root", "65534"])
def test_every_work_exec_finds_file_watches_unavailable(tmp_path: Path, user):
    client = _docker(PYTHON_IMAGE)
    engine = tmp_path / "engine"
    task = tmp_path / "task"
    engine.mkdir()
    task.mkdir()
    runtime = DockerContainerRuntime(
        client,
        run_id=f"rsi-no-watch-{uuid.uuid4().hex[:8]}",
        task_id="no-watch",
        role="work",
        task_source_dir=task,
        allowed_mount_roots=(engine,),
        omit_gpu_device_requests_for_tests=True,
    )
    work = runtime.create(
        ContainerSpec(image=PYTHON_IMAGE, command=("sleep", "infinity"))
    )
    try:
        # Without a network, there is no policy lease to start it through.
        client.containers.get(work.container_id).start()
        result = runtime.exec(
            work, ("python3", "-I", "-c", PROBE), user=user, timeout_seconds=60
        )
        assert result.exit_code == 0, result.output
        found = json.loads(result.output.strip().splitlines()[-1])
        ia32 = None
        if _ia32_here():
            watch = json.dumps(COMPAT["x86_64"]["watch"])
            result = runtime.exec(
                work,
                ("python3", "-I", "-c", IA32_PROBE, watch),
                user=user,
                timeout_seconds=60,
            )
            assert result.exit_code == 0, result.output
            ia32 = json.loads(result.output.strip().splitlines()[-1])
    finally:
        runtime.remove(work)

    assert found == {
        "inotify_init": "ENOSYS",
        "inotify_init1": "ENOSYS",
        "inotify_add_watch": "ENOSYS",
        "fanotify_init": "ENOSYS",
        "fanotify_mark": "ENOSYS",
        "x32_inotify_init1": "ENOSYS",
        "dnotify": "EINVAL",
        "dnotify_high": "EINVAL",
        "getfl": "ok",
        "inet_socket": "ok",
        "legacy_sockets": ["EPERM"],
        "lease": "EINVAL",
        "lease_high": "EINVAL",
        "getlease": "ok",
        "deleg": "EINVAL",
        "deleg_high": "EINVAL",
        "tail": ["a\n", "b\n"],
        "child": ["-1", "38"],
    }
    if ia32 is not None:
        assert ia32 == {
            "getpid": True,
            **dict.fromkeys(COMPAT["x86_64"]["watch"], "ENOSYS"),
            **{
                f"{name}_{check}": value
                for name in ("fcntl", "fcntl64")
                for check, value in (
                    ("lease", "EINVAL"),
                    ("dnotify", "EINVAL"),
                    ("getlease", 2),
                    ("deleg_bad_fd", "EINVAL"),
                    ("getfl", "ok"),
                )
            },
        }


def _ia32_here() -> bool:
    """This x86_64 kernel runs i386 calls (int 0x80)."""
    if os.uname().machine != "x86_64":
        return False
    process, stdout, _ = _run_probe(IA32_PROBE, "{}", filtered=False)
    return process.returncode == 0 and json.loads(stdout)["getpid"]
