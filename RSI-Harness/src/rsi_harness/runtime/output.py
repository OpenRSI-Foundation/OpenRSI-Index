"""Bounded output capture shared by container runtimes."""

from __future__ import annotations

import threading


class BoundedBytes:
    """Retain bounded byte prefixes and suffixes while counting the full stream."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("output limit must be non-negative")
        self.limit = limit
        self.total = 0
        self._head_limit = (limit + 1) // 2
        self._tail_limit = limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self.total += len(chunk)
            needed = self._head_limit - len(self._head)
            if needed > 0:
                self._head.extend(chunk[:needed])
                chunk = chunk[needed:]
            if self._tail_limit and chunk:
                self._tail.extend(chunk)
                if len(self._tail) > self._tail_limit:
                    del self._tail[: len(self._tail) - self._tail_limit]

    @property
    def value(self) -> bytes:
        with self._lock:
            return bytes(self._head + self._tail)

    @property
    def truncated(self) -> bool:
        with self._lock:
            return self.total > self.limit
