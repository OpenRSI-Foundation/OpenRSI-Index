"""Work's Docker seccomp profile: Docker's default without file watches.

A Judge reads Work's paused files: a watch set before would queue every
open and read for Work to count after it resumes, and a lease or
delegation the open breaks signals Work with the file. Every process in the
Work container (each exec, and what it starts) gets Docker's default profile
(``docker_default_seccomp.json``, moby/profiles seccomp/v0.3.0, verbatim;
it replaces the daemon's own default, so keep it no older than the newest
Docker's) with inotify and fanotify returning ENOSYS, as on a kernel built
without them, so watchers fall back to polling, and ``fcntl`` with F_NOTIFY
(dnotify), F_SETLEASE or F_SETDELEG EINVAL. The cluster's rsi-box applies
the same rule (``judge_isolation.watch_filter``).

Not covered: page-cache residency (mincore, cachestat) or read timing can
still show Work which pages the Judge read of any file that shares an inode
with what Work can map. On Docker that is the image's lower layers in every
rootfs mode and the split WORKDIR volume; on a cluster, Work's workspace and
any other source shared between Work and Judge binds. This exists without
the watch rule too and is not changed by it.
"""

from __future__ import annotations

import errno
import functools
import json
from importlib import resources

WATCH_SYSCALLS = (
    "inotify_init",
    "inotify_init1",
    "inotify_add_watch",
    "fanotify_init",
    "fanotify_mark",
)
F_SETLEASE = 1024
F_NOTIFY = 1026
F_SETDELEG = 1040
_FCNTL = ("fcntl", "fcntl64")
# The refused commands differ from F_SETLEASE in bits 1 and 4 only (1042,
# unassigned, is refused too): the other bits decide.
_REFUSED_MASK = ~((F_SETLEASE ^ F_NOTIFY) | (F_SETLEASE ^ F_SETDELEG)) & 0xFFFFFFFF


def work_seccomp_option() -> str:
    """The Work container's ``security_opt`` entry."""
    return "seccomp=" + work_seccomp_profile()


@functools.cache
def work_seccomp_profile() -> str:
    """The profile as Docker's ``seccomp=`` security option takes it."""
    profile = json.loads(
        resources.files(__package__)
        .joinpath("docker_default_seccomp.json")
        .read_text(encoding="utf-8")
    )
    removed = set(WATCH_SYSCALLS) | set(_FCNTL)
    rules = []
    for rule in profile["syscalls"]:
        names = [name for name in rule["names"] if name not in removed]
        if names:
            rules.append({**rule, "names": names})
    rules.append(
        {
            "names": list(WATCH_SYSCALLS),
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": errno.ENOSYS,
        }
    )

    # The kernel reads fcntl's command as 32 bits. Rules that overlap with
    # different actions resolve by their order, so the allowed commands are
    # spelled out disjointly: a deciding low bit that differs from the
    # refused commands'.
    def fcntl(action: str, mask: int, value: int, **extra: int) -> dict:
        command = {
            "index": 1,
            "value": mask,
            "valueTwo": value,
            "op": "SCMP_CMP_MASKED_EQ",
        }
        return {"names": list(_FCNTL), "action": action, "args": [command], **extra}

    rules.append(
        fcntl("SCMP_ACT_ERRNO", _REFUSED_MASK, F_SETLEASE, errnoRet=errno.EINVAL)
    )
    rules += [
        fcntl("SCMP_ACT_ALLOW", 1 << bit, ~F_SETLEASE & (1 << bit))
        for bit in range(32)
        if _REFUSED_MASK >> bit & 1
    ]
    return json.dumps({**profile, "syscalls": rules}, separators=(",", ":"))
