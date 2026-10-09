"""Operator tools: the broker copies only the bytes whose hash was approved."""

import hashlib
import io
import os
import tarfile

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.runtime.sandbox_contracts import (
    EnvHostPolicy,
    EnvToolFile,
    SandboxError,
)
from rsi_harness.runtime.sandbox_tools import (
    MAX_TOOL_BYTES,
    check_tools,
    read_tool,
    tool_file,
    tool_tar,
)

BINARY = b"\x7fELF static tmux"


def host(tmp_path, data=BINARY, sha256=None):
    path = tmp_path / "tmux"
    path.write_bytes(data)
    digest = sha256 or hashlib.sha256(data).hexdigest()
    return EnvHostPolicy(
        pool_disk_mb=1,
        disk_floor_mb=1,
        disk_hard_floor_mb=1,
        tmux=EnvToolFile(path=str(path), sha256=digest),
    )


def test_a_matching_file_is_read_and_tarred_as_one_root_owned_0755_file(tmp_path):
    source = tool_file(host(tmp_path), "tmux")
    with tarfile.open(fileobj=io.BytesIO(tool_tar("tmux", read_tool(source)))) as tar:
        [member] = tar.getmembers()
        assert (member.name, member.mode, member.uid, member.gid) == (
            "tmux",
            0o755,
            0,
            0,
        )
        assert member.isfile() and tar.extractfile(member).read() == BINARY
    check_tools(host(tmp_path))


def test_a_tool_needs_the_operator_key_and_a_known_name(tmp_path):
    bare = host(tmp_path).model_copy(update={"tmux": None})
    for policy in (bare, None):
        with pytest.raises(SandboxError) as caught:
            tool_file(policy, "tmux")
        assert (caught.value.code, caught.value.field) == ("permission", "tool")
    for name in ("vim", "../tmux", 1):
        with pytest.raises(SandboxError, match="invalid: tool"):
            tool_file(host(tmp_path), name)
    check_tools(bare)


@pytest.mark.parametrize(
    "spoil",
    [
        lambda path: path.write_bytes(BINARY + b"changed"),
        lambda path: path.unlink(),
        lambda path: (path.unlink(), os.mkfifo(path)),
        lambda path: (path.unlink(), path.mkdir()),
        lambda path: path.write_bytes(b"x" * (MAX_TOOL_BYTES + 1)),
    ],
)
def test_a_changed_missing_or_odd_file_is_refused_with_no_host_detail(tmp_path, spoil):
    policy = host(tmp_path)
    spoil(tmp_path / "tmux")
    with pytest.raises(SandboxError) as caught:
        read_tool(policy.tmux)
    assert (caught.value.code, caught.value.field) == ("infrastructure", "tool")
    assert str(tmp_path) not in caught.value.message
    with pytest.raises(SetupError, match=r"environments\.host\.tmux"):
        check_tools(policy)
