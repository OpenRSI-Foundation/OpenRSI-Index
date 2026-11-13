from pathlib import PurePosixPath

from rsi_harness.runtime.special_files import SPECIAL_FILE_LIMIT, special_files_refusal


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
