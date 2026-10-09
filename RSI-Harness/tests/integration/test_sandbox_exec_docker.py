"""Real brokered execs as a non-root user: the in-container group kill.

The root-only host path (pidfd signals to the members listed in the service
cgroup) belongs to the operator root check (spec 8, item 5); here it is shown
to be refused as non-root without touching the group. Run as root, the pump
defaults to that path and this file is the check.
"""

import base64
import os
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException

from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_exec import (
    ExecPump,
    ExecTarget,
    HostPidfdKiller,
    ProcessTable,
    parse_exec_request,
)
from tests.integration.test_sandbox_env_docker import (
    BUSYBOX,
    HANDLES,
    RealEnv,
    remove_labelled,
    single,
)

pytestmark = pytest.mark.integration

MIB = 1024**2
SESSION = "work-session"


@pytest.fixture
def exec_run(tmp_path):
    try:
        client = docker.from_env(timeout=60)
        client.ping()
        client.images.get(BUSYBOX)
    except (DockerException, OSError) as error:
        message = f"Docker/{BUSYBOX} capability unavailable: {error}"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    run_id = f"m4-exec-{uuid.uuid4().hex[:12]}"
    finished = []
    pump = ExecPump(client.api, tmp_path / "spool", on_finish=finished.append)
    try:
        yield client, run_id, pump, finished
    finally:
        pump.close()
        filters = {"label": f"rsi-harness.run-id={run_id}"}
        errors = remove_labelled(client, filters)
        leftovers = (
            client.containers.list(all=True, filters=filters),
            client.volumes.list(filters=filters),
            client.networks.list(filters=filters),
        )
        client.close()
        assert (errors, leftovers) == ([], ([], [], []))
        assert not (tmp_path / "spool" / "x").exists()


def ready_env(client, run_id, command="sleep 600", **values):
    box = RealEnv(
        client, run_id, single(BUSYBOX, command, **values), {BUSYBOX: HANDLES[BUSYBOX]}
    )
    return box.create().start()


def destroy(pump, box):
    pump.env_stopped(box.env_id, release=True)
    box.destroy()


def start(pump, box, argv, *, request_id=None, output_limit=MIB, **fields):
    request = parse_exec_request({"service": "main", "argv": argv, **fields})
    return pump.start(
        box.backend.exec_target(box.lease, "main"),
        request,
        session=SESSION,
        env_id=box.env_id,
        request_id=request_id or uuid.uuid4().hex,
        output_limit=output_limit,
    )


def run_to_end(pump, exec_id, *, timeout=60.0):
    """Read an exec to its end the way a client does: offsets and long-polls."""
    out, err = bytearray(), bytearray()
    offsets = (0, 0)
    end = time.monotonic() + timeout
    while True:
        view = pump.wait(
            exec_id,
            session=SESSION,
            stdout_offset=offsets[0],
            stderr_offset=offsets[1],
            wait_sec=min(5.0, max(0.0, end - time.monotonic())),
        )
        out += base64.b64decode(view["stdout_b64"])
        err += base64.b64decode(view["stderr_b64"])
        offsets = (view["stdout_offset"], view["stderr_offset"])
        totals = (view["stdout_total"], view["stderr_total"])
        if view["state"] != "running" and offsets == totals:
            return view, bytes(out), bytes(err)
        assert time.monotonic() < end, f"exec did not finish: {view['state']}"


def processes(client, box):
    """(host pid, state, argv) of every task in the service's cgroup."""
    table = ProcessTable()
    attrs = box.container("main").attrs
    scope = table.scope(attrs["Id"], attrs["State"]["Pid"])
    found = []
    for pid in table.members(scope):
        info = table.stat(pid)
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        except OSError:
            continue
        if info is not None:
            found.append((pid, info.state, [part.decode() for part in argv]))
    return found


def sleeping(found, seconds):
    return any(argv == ["sleep", seconds] for _, _, argv in found)


def await_processes(client, box, predicate, *, timeout=5.0):
    end = time.monotonic() + timeout
    while True:
        found = processes(client, box)
        if predicate(found):
            return found
        assert time.monotonic() < end, found
        time.sleep(0.05)


def test_real_timeout_kills_the_command_and_the_env_keeps_running(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id)
    try:
        began = time.monotonic()
        exec_id = start(pump, box, ["sleep", "300"], timeout_sec=2)

        view, out, err = run_to_end(pump, exec_id)

        elapsed = time.monotonic() - began
        assert (view["state"], view["reason"], view["signal"], view["exit_code"]) == (
            "timed_out",
            "timeout",
            "TERM",
            143,
        )
        assert 2.0 <= elapsed < 3.0, elapsed
        assert box.backend.status(box.lease)["main"].state == "running"
        assert not any(
            argv == ["sleep", "300"] for _, _, argv in processes(client, box)
        )
        view, out, _ = run_to_end(pump, start(pump, box, ["echo", "ok"]))
        assert (view["state"], view["exit_code"], out) == ("exited", 0, b"ok\n")
        assert [summary.state for summary in finished] == ["timed_out", "exited"]
    finally:
        destroy(pump, box)


def test_real_timeout_kills_a_group_that_ignores_term_two_seconds_later(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id)
    try:
        began = time.monotonic()
        # TERM is ignored by the shell and inherited as ignored by its child.
        exec_id = start(
            pump, box, ["sh", "-c", "trap '' TERM; sleep 100 & wait"], timeout_sec=1
        )

        view, _, _ = run_to_end(pump, exec_id)

        elapsed = time.monotonic() - began
        assert (view["state"], view["reason"], view["signal"], view["exit_code"]) == (
            "timed_out",
            "timeout",
            "KILL",
            137,
        )
        assert 3.0 <= elapsed < 4.0, elapsed
        # The whole group went with KILL (in-container `kill -s KILL -<pgid>`).
        assert not any(
            argv == ["sleep", "100"] for _, _, argv in processes(client, box)
        )
        assert box.backend.status(box.lease)["main"].state == "running"
    finally:
        destroy(pump, box)


def test_real_held_stream_never_delays_another_execs_timeout(exec_run):
    client, run_id, pump, finished = exec_run
    box, other = ready_env(client, run_id), ready_env(client, run_id)
    try:
        for held_in in (other, box):  # another session's env, then its own
            # Exits at 1.9 s leaving a child on its stdout: the Engine then
            # holds this exec's lock (and its exec_inspect) for up to 2 s.
            held = start(pump, held_in, ["sh", "-c", "sleep 1.9; sleep 30 & echo x"])
            began = time.monotonic()
            exec_id = start(pump, box, ["sleep", "300"], timeout_sec=2)
            await_processes(client, box, lambda found: sleeping(found, "300"))
            await_processes(
                client, box, lambda found: not sleeping(found, "300"), timeout=10
            )
            killed = time.monotonic() - began

            view, _, _ = run_to_end(pump, exec_id)

            ended = time.monotonic() - began
            assert (view["state"], view["reason"], view["signal"]) == (
                "timed_out",
                "timeout",
                "TERM",
            )
            assert 2.0 <= killed < 2.5, killed  # TERM on time in both cases
            # The Engine handles one container's exit events in order, so in
            # the same container this exit is reported only once the held
            # exec's 2 s stream wait ended (VERIFIED: gone at 2.1 s, reported
            # at 4.0 s); another container is never held up.
            assert ended < (3.0 if held_in is other else 5.0), ended
            view, out, _ = run_to_end(pump, held)
            assert (view["state"], view["exit_code"], out) == ("exited", 0, b"x\n")
    finally:
        destroy(pump, box)
        destroy(pump, other)


def test_real_output_under_load_is_never_cut_by_the_drain(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id, cpus=2.0, memory_mb=512, pids=1024)
    size = 12 * MIB
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(
                pool.map(
                    lambda _: start(
                        pump,
                        box,
                        ["sh", "-c", f"head -c {size} /dev/zero; exit 3"],
                        output_limit=16 * MIB,
                    ),
                    range(32),
                )
            )
            views = [
                view
                for view, _, _ in pool.map(lambda item: run_to_end(pump, item), ids)
            ]

        # Never a false success: a short stream is always marked truncated,
        # and the drain waits for output still in flight (none was short).
        assert [
            (view["state"], view["exit_code"], view["stdout_total"], view["truncated"])
            for view in views
        ] == [("exited", 3, size, False)] * 32
    finally:
        destroy(pump, box)


def test_real_refusals_are_caller_errors_not_broker_failures(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id)
    try:
        with pytest.raises(SandboxError) as caught:
            start(pump, box, ["true"], user="nosuchuser")  # the Engine's 400
        assert (caught.value.code, caught.value.field) == ("invalid", "user")
        assert "nosuchuser" not in str(caught.value)

        # The service ends between exec_start's inspect and its /proc reads.
        target = box.backend.exec_target(box.lease, "main")
        seen = [target.inspect()]
        box.container("main").kill()
        end = time.monotonic() + 10
        while box.container("main").attrs["State"]["Running"]:
            assert time.monotonic() < end
            time.sleep(0.05)
        stale = ExecTarget(
            target.container_id, lambda: seen.pop() if seen else target.inspect()
        )
        request = parse_exec_request({"service": "main", "argv": ["true"]})
        with pytest.raises(SandboxError) as caught:
            pump.start(
                stale,
                request,
                session=SESSION,
                env_id=box.env_id,
                request_id="late",
                output_limit=MIB,
            )
        assert (caught.value.code, caught.value.field) == ("invalid", "service")
        assert finished == []
    finally:
        destroy(pump, box)


def test_real_group_interrupt_kills_background_children_without_zombies(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id)
    try:
        exec_id = start(pump, box, ["sh", "-c", "sleep 1000 & sleep 1001 & wait"])
        group = await_processes(
            client,
            box,
            lambda found: (
                {"1000", "1001"}
                <= {argv[-1] for _, _, argv in found if argv[0] == "sleep"}
            ),
        )
        leader = next(pid for pid, _, argv in group if argv[0] == "sh")
        table = ProcessTable()
        attrs = box.container("main").attrs
        process = table.process(
            attrs["Id"], table.scope(attrs["Id"], attrs["State"]["Pid"]), leader
        )
        members = {pid for pid, _ in table.targets(process, group=True)}
        assert len(members) == 3
        if os.geteuid() != 0:
            # As non-root the host pidfd path is refused (EPERM), touching nothing.
            killer = HostPidfdKiller(table)
            assert killer.signal(process, signal.SIGTERM, group=True) is False
            assert {pid for pid, _ in table.targets(process, group=True)} == members

        result = pump.kill(exec_id, session=SESSION, signal="TERM", scope="group")

        # The group may already be gone when the call returns.
        assert result["delivered"] is True and result["state"] in ("running", "killed")
        view, _, _ = run_to_end(pump, exec_id, timeout=10)
        assert (view["state"], view["reason"], view["signal"], view["exit_code"]) == (
            "killed",
            "interrupt",
            "TERM",
            143,
        )
        # docker-init reaps the orphaned members: none is left, not even a zombie.
        left = await_processes(
            client,
            box,
            lambda found: (
                not members & {pid for pid, _, _ in found}
                and all(state != "Z" for _, state, _ in found)
            ),
        )
        assert left and table.targets(process, group=True) == ()
        assert box.backend.status(box.lease)["main"].state == "running"
    finally:
        destroy(pump, box)


def test_real_output_past_the_cap_is_truncated_with_the_real_exit_code(
    exec_run, tmp_path
):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id)
    try:
        exec_id = start(
            pump,
            box,
            ["sh", "-c", "head -c 52428800 /dev/zero; echo tail >&2; exit 7"],
            output_limit=16 * MIB,
        )

        view, out, err = run_to_end(pump, exec_id)

        assert (view["state"], view["exit_code"], view["truncated"]) == (
            "exited",
            7,
            True,
        )
        assert (view["stdout_total"], len(out), out.count(0)) == (16 * MIB,) * 3
        assert err == b"tail\n"
        spooled = tmp_path / "spool" / "x" / exec_id / "stdout"
        assert spooled.stat().st_size == 16 * MIB  # the rest was read and dropped
    finally:
        destroy(pump, box)


def test_real_oom_in_a_64_mib_service_counts_one_kill_and_the_env_keeps_running(
    exec_run,
):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id, memory_mb=64)
    try:
        # Beyond memory plus swap (the default swap_ratio 1: 64 + 64 MiB).
        exec_id = start(
            pump, box, ["dd", "if=/dev/zero", "of=/dev/null", "bs=160M", "count=1"]
        )

        view, _, _ = run_to_end(pump, exec_id)

        assert (view["state"], view["exit_code"], view["oom_kills"]) == (
            "exited",
            137,
            1,
        )
        assert box.backend.status(box.lease)["main"].state == "running"
        # Per exec, not sticky like State.OOMKilled.
        view, _, _ = run_to_end(pump, start(pump, box, ["true"]))
        assert (view["exit_code"], view["oom_kills"]) == (0, 0)
    finally:
        destroy(pump, box)


def test_real_background_process_holding_stdout_does_not_hang_completion(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id)
    try:
        began = time.monotonic()
        exec_id = start(pump, box, ["sh", "-c", "sleep 30 & echo started"])

        view, out, _ = run_to_end(pump, exec_id, timeout=10)

        assert (view["state"], view["exit_code"], out) == ("exited", 0, b"started\n")
        # dockerd itself waits up to 2 s for the streams; the pump adds ≤ 1 s.
        assert time.monotonic() - began < 4.0
        # The service the command left behind keeps running.
        assert any(argv == ["sleep", "30"] for _, _, argv in processes(client, box))
    finally:
        destroy(pump, box)


def test_real_64_concurrent_execs_stream_their_output(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id, cpus=2.0, memory_mb=512, pids=1024)
    script = (
        "while [ ! -e /tmp/go ]; do sleep 0.05; done; "
        'i=0; while [ $i -lt 400 ]; do echo "$TAG-$i"; i=$((i+1)); done; '
        'echo "$TAG" >&2; exit $CODE'
    )
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(
                pool.map(
                    lambda index: start(
                        pump,
                        box,
                        ["sh", "-c", script],
                        env={"TAG": f"t{index}", "CODE": str(index % 7)},
                    ),
                    range(64),
                )
            )
        assert pump.running(SESSION) == 64  # all attached, all waiting

        assert box.container("main").exec_run(["touch", "/tmp/go"]).exit_code == 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda exec_id: run_to_end(pump, exec_id), ids))

        for index, (view, out, err) in enumerate(results):
            expected = "".join(f"t{index}-{line}\n" for line in range(400)).encode()
            assert (view["state"], view["exit_code"], view["truncated"]) == (
                "exited",
                index % 7,
                False,
            )
            assert (out, err) == (expected, f"t{index}\n".encode())
        assert len(finished) == 64 and pump.running(SESSION) == 0
    finally:
        destroy(pump, box)


def test_real_request_id_replays_and_merged_output_keeps_its_order(exec_run):
    client, run_id, pump, finished = exec_run
    box = ready_env(client, run_id)
    try:
        argv = ["sh", "-c", "echo a; sleep 0.2; echo b >&2; sleep 0.2; echo c"]
        exec_id = start(pump, box, argv, request_id="same", merge_stderr=True)
        assert start(pump, box, argv, request_id="same", merge_stderr=True) == exec_id
        # The Engine lists running execs only: the replay started nothing.
        assert len(box.container("main").attrs.get("ExecIDs") or []) == 1

        view, out, err = run_to_end(pump, exec_id)

        assert (view["state"], out, err) == ("exited", b"a\nb\nc\n", b"")
    finally:
        destroy(pump, box)
