import os
import socket
from pathlib import PurePosixPath

import pytest

from rsi_harness.runtime.special_files import (
    SPECIAL_FILE_LIMIT,
    SpecialFileScanError,
    find_special_files,
    scan_failure_refusal,
    special_files_refusal,
)


def test_the_refusal_lists_the_paths_and_says_it_is_not_counted():
    found = [(PurePosixPath("src/pipe"), "FIFO, socket or device")]

    message = special_files_refusal(found, PurePosixPath("/testbed"))

    assert "/testbed/src/pipe (FIFO, socket or device)" in message
    assert "not counted" in message
    assert "maybe more" not in message


def test_a_refusal_at_the_limit_says_there_may_be_more():
    found = [
        (PurePosixPath(f"pipe-{index}"), "FIFO, socket or device")
        for index in range(SPECIAL_FILE_LIMIT)
    ]

    message = special_files_refusal(found, PurePosixPath("/testbed"))

    assert "/testbed/pipe-0" in message and "and maybe more" in message


def test_a_refusal_escapes_a_name_it_cannot_print():
    found = [(PurePosixPath("bad\nname"), "FIFO")]

    message = special_files_refusal(found, PurePosixPath("/testbed"))

    assert "\n" not in message and "'/testbed/bad\\nname' (FIFO)" in message


def test_finds_fifos_and_sockets_but_follows_no_symlink(tmp_path, monkeypatch):
    (tmp_path / "src" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "code.py").write_text("x = 1\n")
    os.mkfifo(tmp_path / "src" / "deep" / "pipe")
    monkeypatch.chdir(tmp_path / "src")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("control.sock")  # relative: AF_UNIX paths are short
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    os.mkfifo(outside / "pipe")
    (tmp_path / "linked-dir").symlink_to(outside)
    (tmp_path / "linked-pipe").symlink_to(outside / "pipe")
    try:
        found = find_special_files(tmp_path)
    finally:
        listener.close()

    assert sorted(found) == [
        (PurePosixPath("src/control.sock"), "socket"),
        (PurePosixPath("src/deep/pipe"), "FIFO"),
    ]


def test_an_ordinary_workspace_has_none(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "file").write_text("data")
    (tmp_path / "a" / "link").symlink_to("/dev/null")

    assert find_special_files(tmp_path) == []


def test_the_walk_stops_at_the_limit(tmp_path):
    for index in range(SPECIAL_FILE_LIMIT + 3):
        os.mkfifo(tmp_path / f"pipe-{index}")

    found = find_special_files(tmp_path)

    assert len(found) == SPECIAL_FILE_LIMIT
    message = special_files_refusal(found, PurePosixPath("/testbed"))
    assert "/testbed/pipe-" in message and "and maybe more" in message


def test_the_walk_is_bounded_and_fails_closed(tmp_path):
    for index in range(4):
        (tmp_path / f"file-{index}").write_text("data")

    with pytest.raises(SpecialFileScanError) as failed:
        find_special_files(tmp_path, entry_limit=3)

    message = scan_failure_refusal(failed.value, PurePosixPath("/testbed"))
    assert "not counted" in message and "more than 3 entries" in message


def test_a_symlinked_root_is_not_followed(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")

    with pytest.raises(SpecialFileScanError):
        find_special_files(tmp_path / "link")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")
def test_a_directory_it_cannot_read_fails_the_walk(tmp_path):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    os.mkfifo(hidden / "pipe")
    hidden.chmod(0)
    try:
        with pytest.raises(SpecialFileScanError) as failed:
            find_special_files(tmp_path)
    finally:
        hidden.chmod(0o700)

    assert failed.value.path == PurePosixPath("hidden")
    message = scan_failure_refusal(failed.value, PurePosixPath("/testbed"))
    assert "/testbed/hidden: Permission denied" in message


def test_the_walk_enters_a_directory_on_another_device(tmp_path, monkeypatch):
    # A btrfs subvolume Work creates has its own st_dev but is no mount, and
    # the Judge's bind shows it.
    (tmp_path / "sub").mkdir()
    os.mkfifo(tmp_path / "sub" / "pipe")
    real = os.fstat
    sub = os.stat(tmp_path / "sub").st_ino

    def fstat(descriptor):
        result = real(descriptor)
        if result.st_ino != sub:
            return result
        values = list(result)
        values[2] = result.st_dev + 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", fstat)

    assert find_special_files(tmp_path) == [(PurePosixPath("sub/pipe"), "FIFO")]


def test_the_walk_reaches_below_path_max(tmp_path):
    name = "d" * 200
    parts = [name] * 30  # about 6000 characters, past PATH_MAX
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts:
            os.mkdir(part, dir_fd=descriptor)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        os.mkfifo("pipe", dir_fd=descriptor)
    finally:
        os.close(descriptor)

    found = find_special_files(tmp_path)

    assert found == [(PurePosixPath(*parts, "pipe"), "FIFO")]
    message = special_files_refusal(found, PurePosixPath("/testbed"))
    assert len(message) < 600 and message.count("...") == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_an_unreadable_file_is_not_a_failure(tmp_path):
    # The Judge runs as the same user and cannot read it either.
    secret = tmp_path / "secret"
    secret.write_text("data")
    secret.chmod(0)

    assert find_special_files(tmp_path) == []
