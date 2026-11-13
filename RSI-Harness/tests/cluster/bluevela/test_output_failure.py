"""Storage failures must end the child, not silently lose a research log."""

import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rsi_harness.errors import InfrastructureError
from tests.cluster.bluevela.test_runtime import _codex_runtime


def process_runtime(tmp_path, monkeypatch, *, sleep=True):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    pidfile = tmp_path / "pid"
    script = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        "print('prefix', flush=True); print('private-secret', flush=True); "
        + ("time.sleep(120)" if sleep else "")
    )
    monkeypatch.setattr(
        runtime, "_base_command", lambda **_o: (sys.executable, "-c", script)
    )
    monkeypatch.setattr(runtime, "_network_command", lambda *_a, **_o: ())
    return runtime, pidfile


def run(runtime, output, **options):
    return runtime._run(
        (),
        devices=(),
        network_mode="public",
        timeout_seconds=120,
        output_path=output,
        output_redact_values=("private-secret",),
        **options,
    )


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT, errno.EIO])
@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("track_work", [False, True])
def test_storage_failure_ends_running_child_promptly(
    tmp_path, monkeypatch, error_number, operation, track_work
):
    runtime, pidfile = process_runtime(tmp_path, monkeypatch)
    output = tmp_path / "output.log"
    original = Path.open
    streams = []

    class BrokenSink:
        def __init__(self, stream):
            self.stream, self.calls = stream, 0

        def __getattr__(self, name):
            method = getattr(self.stream, name)
            if name != operation:
                return method

            def fail(*args):
                self.calls += 1
                if self.calls > 1:
                    raise OSError(error_number, "private-secret storage failure")
                return method(*args)

            return fail

    def open_sink(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if path == output and mode == "a":
            streams.append(stream)
            return BrokenSink(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_sink)
    started = time.monotonic()
    with pytest.raises(InfrastructureError, match="output capture failed") as caught:
        run(runtime, output, track_work=track_work)
    assert time.monotonic() - started < 15
    assert "private-secret" not in str(caught.value)
    assert output.read_text().startswith("prefix\n")
    assert "private-secret" not in output.read_text()
    assert all(stream.closed for stream in streams)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


@pytest.mark.parametrize("operation", ["write", "fsync", "close"])
@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT, errno.EIO])
def test_late_storage_failure_cannot_report_a_successful_capture(
    tmp_path, monkeypatch, operation, error_number
):
    runtime, pidfile = process_runtime(tmp_path, monkeypatch, sleep=False)
    output = tmp_path / "output.log"
    original = Path.open
    streams = []

    def fail(*_args):
        raise OSError(error_number, "late storage failure")

    class BrokenSink:
        def __getattr__(self, name):
            if name == operation:
                if name == "close":
                    self.stream.close()
                return fail
            return getattr(self.stream, name)

        def __init__(self, stream):
            self.stream = stream

    def open_sink(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if path == output and mode == "a":
            streams.append(stream)
            return BrokenSink(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_sink)
    if operation == "fsync":
        monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(InfrastructureError, match="output capture failed"):
        run(runtime, output)
    assert all(stream.closed for stream in streams)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_unwritable_output_never_starts_child(tmp_path, monkeypatch):
    runtime, _ = process_runtime(tmp_path, monkeypatch)

    def unexpected(*_args, **_kwargs):
        pytest.fail("child started before output was opened")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    with pytest.raises(IsADirectoryError):
        run(runtime, tmp_path)


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT, errno.EIO])
def test_output_open_storage_error_never_starts_child(
    tmp_path, monkeypatch, error_number
):
    runtime, _ = process_runtime(tmp_path, monkeypatch)
    output = tmp_path / "output.log"
    original = Path.open

    def open_sink(path, *args, **kwargs):
        if path == output:
            raise OSError(error_number, "cannot open output")
        return original(path, *args, **kwargs)

    def unexpected(*_args, **_kwargs):
        pytest.fail("child started without an output sink")

    monkeypatch.setattr(Path, "open", open_sink)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    with pytest.raises(OSError) as caught:
        run(runtime, output)
    assert caught.value.errno == error_number


@pytest.mark.parametrize("reason", ["running", "paused"])
def test_refused_second_work_closes_sink_and_keeps_first_work(
    tmp_path, monkeypatch, reason
):
    """A second Work process starts neither beside a running one nor while a
    Judge round holds Work (an Agent that exited before the round's pause,
    retried meanwhile, would otherwise run beside a Judge reading WORKDIR)."""
    runtime, _ = process_runtime(tmp_path, monkeypatch)
    output = tmp_path / "output.log"
    original_open = Path.open
    streams = []
    script = "" if reason == "paused" else "import time; time.sleep(60)"
    first = subprocess.Popen([sys.executable, "-c", script])
    if reason == "paused":
        first.wait(timeout=10)
        runtime._paused = True
    runtime._process = first

    def unexpected(*_args, **_kwargs):
        pytest.fail("a second Work process started")

    def open_sink(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path == output and mode == "a":
            streams.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_sink)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    match = "a Judge round holds Work" if reason == "paused" else "already running"
    try:
        with pytest.raises(InfrastructureError, match=match):
            run(runtime, output, track_work=True)
        assert runtime._process is first
        assert streams and all(stream.closed for stream in streams)
    finally:
        first.kill()
        first.wait()


def test_spawn_failure_closes_already_open_output(tmp_path, monkeypatch):
    runtime, _ = process_runtime(tmp_path, monkeypatch)
    output = tmp_path / "output.log"
    original = Path.open
    streams = []

    def open_sink(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if path == output and mode == "a":
            streams.append(stream)
        return stream

    def fail(*_args, **_kwargs):
        raise OSError(errno.EAGAIN, "cannot spawn")

    monkeypatch.setattr(Path, "open", open_sink)
    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError, match="cannot spawn"):
        run(runtime, output)
    assert streams and all(stream.closed for stream in streams)


def test_failed_output_callback_ends_paused_child_without_resuming(
    tmp_path, monkeypatch
):
    runtime, pidfile = process_runtime(tmp_path, monkeypatch)
    original = os.killpg
    signals = []

    def killpg(pid, sig):
        signals.append(sig)
        return original(pid, sig)

    def callback(_line):
        os.killpg(int(pidfile.read_text()), signal.SIGSTOP)
        raise OSError(errno.EDQUOT, "callback journal full")

    monkeypatch.setattr(os, "killpg", killpg)
    with pytest.raises(InfrastructureError, match="callback journal full"):
        run(runtime, tmp_path / "output.log", output_callback=callback, track_work=True)
    assert signal.SIGCONT not in signals
    assert runtime._process.poll() is not None


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT, errno.EIO])
def test_remote_controller_output_failure_stops_remote_and_local_owner(
    tmp_path, monkeypatch, error_number
):
    runtime, _ = process_runtime(tmp_path, monkeypatch)
    output = tmp_path / "controller.log"
    pidfile = tmp_path / "controller-pid"
    original = Path.open
    stops = []

    class Sink:
        def __init__(self, stream):
            self.stream = stream

        def write(self, _data):
            raise OSError(error_number, "storage failed")

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

    def open_sink(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return Sink(stream) if path == output and mode == "wb" else stream

    from rsi_harness.cluster.bluevela import runtime as module

    monkeypatch.setattr(Path, "open", open_sink)
    monkeypatch.setattr(module, "stop_command", lambda *a, **k: ("remote-stop",))

    def stop(command, **_kwargs):
        stops.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", stop)
    script = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        "print('ready',flush=True); time.sleep(120)"
    )
    started = time.monotonic()
    with pytest.raises(InfrastructureError, match="output capture failed"):
        runtime._run_host_controller(
            (sys.executable, "-c", script), timeout_seconds=120, output_path=output
        )
    assert time.monotonic() - started < 15
    assert stops == [("remote-stop",)]
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
