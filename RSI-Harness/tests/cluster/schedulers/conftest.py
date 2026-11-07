from __future__ import annotations

import time

import pytest


def _until(condition, seconds=10.0):
    ends = time.monotonic() + seconds
    while time.monotonic() < ends:
        if condition():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def until():
    """Wait (up to ``seconds``) for ``condition()``; whether it came true."""
    return _until
