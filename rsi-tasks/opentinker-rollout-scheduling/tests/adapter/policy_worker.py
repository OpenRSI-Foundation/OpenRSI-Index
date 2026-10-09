"""Unprivileged callback worker; its pipes carry only source and causal metadata."""
import collections
import functools
import heapq
import itertools
import json
import math
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path
import pyseccomp as seccomp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_diagnostics import exception_detail

LIMIT = 1048576


def main():
    # Load public helpers before closing the filesystem. No candidate imports occur here.
    source = sys.stdin.buffer.readline(8 * LIMIT + 1)
    if len(source) > 8 * LIMIT:
        raise ValueError("source_size")
    source = json.loads(source)["source"]
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (3, 3))
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    # Default-deny includes open/openat, sockets, clone/fork, exec, ptrace and process_vm_*.
    filt = seccomp.SyscallFilter(defaction=seccomp.KILL_PROCESS)
    for call in ("read", "write", "close", "fstat", "lseek", "brk", "mmap", "mremap",
                 "munmap", "mprotect", "madvise", "rt_sigaction", "rt_sigprocmask",
                 "rt_sigreturn", "sigaltstack", "futex", "clock_gettime", "gettimeofday",
                 "getrandom", "getpid", "gettid", "sched_yield", "exit", "exit_group"):
        filt.add_rule(seccomp.ALLOW, call)
    filt.load()
    # Preserve the protocol writer; user print is not a protocol response.
    writer = sys.stdout
    class Sink:
        def write(self, value):
            return len(value)
        def flush(self):
            pass
    sys.stdout = Sink()
    sys.stderr = Sink()
    scope = {"__name__": "candidate_policy"}
    phase = "compile"
    try:
        compiled = compile(source, "/workspace/candidate/policy.py", "exec")
        phase = "module"
        exec(compiled, scope)
        phase = "Policy.__init__"
        policy = scope["Policy"]()
    except BaseException as exc:
        writer.write(json.dumps({"error": "policy_import", "detail": exception_detail(exc, phase)}) + "\n")
        writer.flush()
        return
    writer.write('{"ok":true}\n'); writer.flush()
    for line in sys.stdin.buffer:
        message = json.loads(line)
        method = message["method"]
        phase = "Policy." + method
        try:
            if method == "reset":
                policy.reset(message["value"])
                result = None
            elif method == "schedule":
                result = policy.schedule(message["value"])
            else:
                raise ValueError("method")
        except BaseException as exc:
            encoded = json.dumps({"error": "policy_callback", "detail": exception_detail(exc, phase)})
        else:
            try:
                encoded = json.dumps({"result": result}, allow_nan=False, separators=(",", ":"))
                if len(encoded.encode()) > LIMIT:
                    encoded = '{"error":"policy_output_size"}'
            except BaseException as exc:
                encoded = json.dumps({"error": "policy_serialization",
                                      "detail": exception_detail(exc, "Policy.schedule.result")})
        writer.write(encoded + "\n"); writer.flush()


if __name__ == "__main__":
    main()
