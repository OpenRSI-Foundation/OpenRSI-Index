"""Explicit operator policy for native functional tests, never production defaults.

Probe creation without restricting the pytest process. Only the known ABI 6
EINVAL backport selects compatibility; unrelated failures remain failures.
Security policy tests must call the production defaults directly instead.
"""

import ctypes
import errno
from functools import cache

from rsi_harness.judge_agent import judge_isolation as isolation


@cache
def native_signal_compatibility() -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    abi, _ = isolation.landlock_abi(libc)
    if abi < 1:
        return False
    try:
        ruleset = isolation.Ruleset(libc, abi)
    except isolation.Failure:
        if abi != 6 or ctypes.get_errno() != errno.EINVAL:
            raise
        # Confirm that filesystem and abstract-socket confinement still works.
        ruleset = isolation.Ruleset(libc, abi, allow_missing_landlock_signal=True)
        try:
            assert ruleset.scoped == 1
        finally:
            ruleset.close()
        return True
    else:
        ruleset.close()
        return False
