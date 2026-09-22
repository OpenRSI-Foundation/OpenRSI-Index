"""Small Linux Landlock boundary for the candidate subprocess, not root attestation.

Uses the stable filesystem ABI. No mount, namespace or privileged Docker mode.
An unsupported/disabled kernel is an infrastructure failure, never a score.
"""
import ctypes
import os
from pathlib import Path
import platform


class Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]


def restrict_filesystem(readonly, readwrite):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('candidate isolation requires Linux x86_64/aarch64 with Landlock')
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(result, name):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, f'{name}: {os.strerror(code)}')
        return result

    abi = checked(libc.syscall(444, 0, 0, 1), 'Landlock ABI query')
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13  # REFER
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE
    config = Ruleset(handled)
    ruleset = checked(libc.syscall(444, ctypes.byref(config), ctypes.sizeof(config), 0), 'Landlock ruleset creation')
    try:
        for paths, writable in [(readonly, False), (readwrite, True)]:
            for raw in paths:
                path = Path(raw)
                if not path.exists():
                    continue
                allowed = handled if writable else (1 | (1 << 2) | (1 << 3))
                if not path.is_dir():
                    allowed &= 1 | (1 << 1) | (1 << 2) | (1 << 14)
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    rule = PathRule(allowed, fd)
                    checked(libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0), f'Landlock rule for {path}')
                finally:
                    os.close(fd)
        checked(libc.prctl(38, 1, 0, 0, 0), 'PR_SET_NO_NEW_PRIVS')
        checked(libc.syscall(446, ruleset, 0), 'Landlock restrict_self')
    finally:
        os.close(ruleset)


def drop_to_candidate():
    if os.getuid() != 0:
        raise RuntimeError('Judge launcher expects standard root phase identity before dropping candidate privileges')
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
