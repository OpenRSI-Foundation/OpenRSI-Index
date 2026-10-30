"""rsi-nonet: exec a command under a seccomp filter without IP sockets.

    python3 -I -S -B rsi_nonet.py FILTER -- COMMAND [ARG]...

FILTER is the packed program of
``rsi_harness.judge_agent.judge_isolation.network_filter()`` in hex. The
wrapper sets no_new_privs, installs the filter and execs COMMAND in the same
process, so COMMAND keeps its PID. RSI Harness wraps ``apptainer exec`` of a
phase without network in it where the site has no network namespaces
(``network_isolation = "seccomp"``). Exit status when COMMAND does not run:
125 usage or setup error, 126 cannot execute, 127 not found.

Standard library only, Python 3.6 or later (a host's own python3 may run
it), no RSI Harness imports when run.
"""

import ctypes
import os
import sys

PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def command(argv, python=None):
    """``argv`` wrapped: run under :func:`network_filter` of this machine."""
    from rsi_harness.judge_agent.judge_isolation import network_filter

    return [
        python or sys.executable,
        "-I",
        "-S",
        "-B",
        os.path.realpath(__file__),
        network_filter().hex(),
        "--",
    ] + list(argv)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 3 or arguments[1] != "--":
        sys.stderr.write("usage: rsi_nonet.py FILTER -- COMMAND [ARG]...\n")
        return 125
    try:
        program = bytes.fromhex(arguments[0])
    except ValueError:
        program = b""
    if not program or len(program) % 8:
        sys.stderr.write("rsi-nonet: FILTER is not a seccomp program\n")
        return 125
    command = arguments[2:]
    libc = ctypes.CDLL(None, use_errno=True)
    buffer = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(len(program) // 8, ctypes.addressof(buffer))
    zero = ctypes.c_ulong(0)
    for what, result in (
        ("set no_new_privs", lambda: libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)),
        (
            "install the filter",
            lambda: libc.prctl(
                ctypes.c_int(PR_SET_SECCOMP),
                ctypes.c_ulong(SECCOMP_MODE_FILTER),
                ctypes.c_ulong(ctypes.addressof(fprog)),
                zero,
                zero,
            ),
        ),
    ):
        if result() != 0:
            error = os.strerror(ctypes.get_errno())
            sys.stderr.write(f"rsi-nonet: cannot {what}: {error}\n")
            return 125
    try:
        os.execvp(command[0], command)
    except OSError as error:
        sys.stderr.write(f"rsi-nonet: cannot execute {command[0]}: {error.strerror}\n")
        return 127 if isinstance(error, FileNotFoundError) else 126


if __name__ == "__main__":
    sys.exit(main())
