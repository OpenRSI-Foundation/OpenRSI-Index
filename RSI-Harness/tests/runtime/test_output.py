from __future__ import annotations

import pytest

from rsi_harness.runtime.output import BoundedBytes


@pytest.mark.parametrize("limit", [0, 1, 2, 7, 8, 9, 20])
@pytest.mark.parametrize("chunk_size", [1, 3, 20])
def test_bounded_capture_retains_exact_head_and_tail(limit, chunk_size):
    capture = BoundedBytes(limit)
    data = b"0123456789abcdefghij"
    for offset in range(0, len(data), chunk_size):
        capture.append(data[offset : offset + chunk_size])
    head = min(len(data), (limit + 1) // 2)
    tail = min(len(data) - head, limit // 2)
    assert capture.value == data[:head] + (data[-tail:] if tail else b"")
    assert len(capture.value) <= limit
    assert capture.total == len(data)
    assert capture.truncated is (len(data) > limit)


def test_bounded_capture_marks_single_oversized_chunk_truncated():
    capture = BoundedBytes(5)
    capture.append(b"one oversized line\n")
    assert capture.value == b"onee\n"
    assert capture.truncated


def test_bounded_capture_preserves_untruncated_split_utf8():
    data = "中文🧪\n".encode()
    capture = BoundedBytes(len(data))
    for byte in data:
        capture.append(bytes([byte]))
    assert capture.value == data
    assert capture.value.decode() == "中文🧪\n"
    assert not capture.truncated


def test_bounded_capture_rejects_negative_limit():
    with pytest.raises(ValueError, match="non-negative"):
        BoundedBytes(-1)
