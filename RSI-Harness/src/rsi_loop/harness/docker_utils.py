# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Docker container utilities, adapted from SWE-bench."""

from __future__ import annotations

import io
import os
import signal
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from docker.models.containers import Container

HEREDOC_DELIMITER = "EOF_RSI_LOOP_1399519320"


def copy_to_container(container: Container, src: Path, dst: Path) -> None:
    """Copy a local file into a Docker container."""
    if os.path.dirname(dst) == "":
        raise ValueError(f"Destination path parent directory cannot be empty: {dst}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(src, arcname=dst.name)
    buf.seek(0)

    container.exec_run(f"mkdir -p {dst.parent}")
    container.put_archive(os.path.dirname(dst), buf.read())


def write_to_container(container: Container, data: str, dst: Path) -> None:
    """Write a string to a file inside a Docker container."""
    command = f"cat <<'{HEREDOC_DELIMITER}' > {dst}\n{data}\n{HEREDOC_DELIMITER}"
    container.exec_run(command)


def exec_run_with_timeout(
    container: Container,
    cmd: str | list[str],
    timeout: int | None = 60,
    log_file: Path | None = None,
    user: str | None = None,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
    stream_to_stdout: bool = False,
    shutdown_event: threading.Event | None = None,
    log_append: bool = False,
    on_chunk: Callable[[bytes], None] | None = None,
) -> tuple[str, bool, float]:
    """
    Run a command in a container with a timeout.

    Returns:
        (output, timed_out, elapsed_seconds)
    """
    output, _exit_code, timed_out, elapsed = _exec_run(
        container, cmd, timeout, log_file, user, workdir, environment,
        stream_to_stdout=stream_to_stdout,
        shutdown_event=shutdown_event,
        log_append=log_append,
        on_chunk=on_chunk,
    )
    return output, timed_out, elapsed


def exec_run_with_exit_code(
    container: Container,
    cmd: str | list[str],
    timeout: int | None = 60,
    user: str | None = None,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
) -> tuple[str, int, bool, float]:
    """
    Run a command in a container with a timeout, returning the exit code.

    Returns:
        (output, exit_code, timed_out, elapsed_seconds)
        exit_code is -1 if timed out before completion.
    """
    return _exec_run(container, cmd, timeout, None, user, workdir, environment)


def _exec_run(
    container: Container,
    cmd: str | list[str],
    timeout: int | None = 60,
    log_file: Path | None = None,
    user: str | None = None,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
    stream_to_stdout: bool = False,
    shutdown_event: threading.Event | None = None,
    log_append: bool = False,
    on_chunk: Callable[[bytes], None] | None = None,
) -> tuple[str, int, bool, float]:
    """Shared implementation for exec_run_with_timeout / exec_run_with_exit_code."""
    chunks: list[bytes] = []
    exec_id = None
    exception = None
    timed_out = False
    log_fh = None

    try:
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_file, "ab" if log_append else "wb")

        def run_command():
            nonlocal exec_id, exception
            try:
                kwargs = {}
                if user:
                    kwargs["user"] = user
                if workdir:
                    kwargs["workdir"] = workdir
                if environment:
                    kwargs["environment"] = environment
                exec_id = container.client.api.exec_create(container.id, cmd, **kwargs)["Id"]
                exec_stream = container.client.api.exec_start(exec_id, stream=True)
                for chunk in exec_stream:
                    chunks.append(chunk)
                    if log_fh:
                        log_fh.write(chunk)
                        log_fh.flush()
                    if stream_to_stdout:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    if on_chunk is not None:
                        try:
                            on_chunk(chunk)
                        except Exception:
                            pass  # sync is best-effort; never disturb the run
            except Exception as e:
                exception = e

        thread = threading.Thread(target=run_command)
        start_time = time.time()
        thread.start()

        deadline = start_time + timeout if timeout else None
        while thread.is_alive():
            remaining = (deadline - time.time()) if deadline else 1.0
            if remaining <= 0:
                break
            thread.join(min(remaining, 1.0))
            if shutdown_event and shutdown_event.is_set():
                break
    finally:
        if log_fh:
            log_fh.close()

    if exception:
        raise exception

    if thread.is_alive():
        if exec_id is not None:
            exec_pid = container.client.api.exec_inspect(exec_id)["Pid"]
            container.exec_run(["/bin/sh", "-c", f"kill -TERM {exec_pid}"], detach=True)
        timed_out = True
        exit_code = -1
    else:
        exit_code = container.client.api.exec_inspect(exec_id)["ExitCode"] if exec_id else -1

    end_time = time.time()
    output = b"".join(chunks).decode(errors="replace")
    return output, exit_code, timed_out, end_time - start_time


def cleanup_container(client, container, logger=None) -> None:
    """Stop and remove a Docker container. SIGINT is masked during cleanup."""
    if not container:
        return

    # Mask SIGINT so a second Ctrl+C cannot interrupt cleanup.
    # signal.signal() only works in the main thread; silently skip in workers.
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        pass

    container_id = container.id
    log_info = logger.info if logger else print
    log_error = logger.error if logger else print

    # Stop
    try:
        log_info(f"Stopping container {container.name}...")
        container.stop(timeout=15)
    except Exception as e:
        log_error(f"Failed to stop container {container.name}: {e}")
        try:
            container_info = client.api.inspect_container(container_id)
            pid = container_info["State"].get("Pid", 0)
            if pid > 0:
                log_info(f"Forcefully killing container PID {pid}...")
                os.kill(pid, signal.SIGKILL)
        except Exception as e2:
            log_error(f"Failed to kill container {container.name}: {e2}")

    # Remove
    try:
        log_info(f"Removing container {container.name}...")
        container.remove(force=True)
        log_info(f"Container {container.name} removed.")
    except Exception as e:
        log_error(
            f"Failed to remove container {container.name}: {e}\n"
            f"{traceback.format_exc()}"
        )

    if old_handler is not None:
        try:
            signal.signal(signal.SIGINT, old_handler)
        except ValueError:
            pass


def remove_image(client, image_id: str, logger=None) -> None:
    """Remove a Docker image by ID/tag."""
    log_info = logger.info if logger else print
    log_error = logger.error if logger else print
    try:
        log_info(f"Removing image {image_id}...")
        client.images.remove(image_id, force=True)
        log_info(f"Image {image_id} removed.")
    except Exception as e:
        log_error(f"Failed to remove image {image_id}: {e}")
