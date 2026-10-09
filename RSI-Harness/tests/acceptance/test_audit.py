"""The acceptance audits with fakes: A5 leftovers, A6 findings, the A7 kill
trigger and the verdict rows scripts/operator/sandbox_acceptance.sh prints."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.runtime.network import firewall_rule_chains, managed_bridge_interface
from tests.acceptance import audit
from tests.acceptance.audit import Observed

RUN = "0123456789abcdef0123456789abcdef"
ENV_NET = "rsi-sbnet-00112233445566aa"
BUILDER_NET = "rsi-sbbnet-77889900aabbccdd"
DIGEST = "sha256:" + "d" * 64


class Item(SimpleNamespace):
    def __init__(self, name, *, labels=None, status="running", attrs=None, tags=()):
        super().__init__(
            name=name,
            id=name + "0" * 64,
            labels=labels or {},
            status=status,
            attrs=attrs or {},
            tags=list(tags),
        )


class Listing:
    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.items)


class Images:
    def __init__(self, labelled=(), tagged=(), present=()):
        self.labelled, self.tagged, self.present = labelled, tagged, set(present)

    def list(self, **kwargs):
        return list(self.tagged if "name" in kwargs else self.labelled)

    def get(self, image_id):
        if image_id not in self.present:
            raise LookupError(image_id)
        return image_id


def client(*, containers=(), networks=(), volumes=(), images=None, top=None):
    return SimpleNamespace(
        containers=Listing(*containers),
        networks=Listing(*networks),
        volumes=Listing(*volumes),
        images=images or Images(),
        api=SimpleNamespace(top=top or (lambda _id: {"Processes": []})),
    )


def command(outputs):
    def run(argv):
        return outputs.get(argv[0], "")

    return run


def lease(data_root: Path, **records) -> None:
    (data_root / "leases").mkdir(parents=True, exist_ok=True)
    (data_root / "leases" / f"{RUN}.json").write_text(json.dumps(records))


def test_a_clean_run_has_no_leftovers_and_lists_pulled_images_apart(tmp_path):
    lease(
        tmp_path,
        sandbox_images=[
            {"kind": "pulled", "image_id": DIGEST, "pre_existing": True},
        ],
    )
    report = audit.leftovers(
        client(),
        RUN,
        tmp_path,
        Observed(networks=[ENV_NET]),
        command=command({"ip": "lo UNKNOWN\neth0 UP\n"}),
    )

    assert report["clean"] is True
    assert all(value == [] for value in report["findings"].values())
    assert report["pulled (cache, not leftovers)"] == [f"{DIGEST} pre_existing=True"]


def test_every_kind_of_leftover_of_the_run_is_found(tmp_path):
    (tmp_path / RUN / "sb").mkdir(parents=True)
    lease(tmp_path, sandbox_images=[{"kind": "built", "image_id": DIGEST}])
    rule = f"rsi-{RUN}-sbx-00112233445566aa"
    chain = firewall_rule_chains(rule)[0]
    bridge = managed_bridge_interface(ENV_NET)
    prefix = audit.built_tag_prefix(RUN)
    loop_file = f"{tmp_path.resolve()}/{RUN}/sb/build/77889900aabbccdd.img"
    fake = client(
        containers=[Item("rsi-sbx-0011223344556677-0")],
        networks=[Item(ENV_NET)],
        volumes=[Item("rsi-sbvol-0011223344556677-0")],
        images=Images(tagged=[Item("x", tags=[prefix + "1" * 32])], present=[DIGEST]),
    )

    report = audit.leftovers(
        fake,
        RUN,
        tmp_path,
        Observed(networks=[ENV_NET], loop_devices=["/dev/loop9"]),
        command=command(
            {
                "ip": f"{bridge}@if3 UP\nlo UNKNOWN\n",
                "iptables": f"-N {chain}\n-A DOCKER-USER --comment {rule}:x\n",
                "losetup": f"/dev/loop7: []: ({loop_file})\n/dev/loop9: []: (/x)\n",
            }
        ),
    )

    findings = report["findings"]
    assert report["clean"] is False
    assert findings["containers"] == ["rsi-sbx-0011223344556677-0"]
    assert findings["networks"] == [ENV_NET]
    assert findings["volumes"] == ["rsi-sbvol-0011223344556677-0"]
    assert findings["built images"] == [DIGEST, prefix + "1" * 32]
    assert findings["bridges"] == [bridge]
    assert len(findings["iptables"]) == 2
    # A freed loop device serving another file now is not the run's.
    assert [line.split(":")[0] for line in findings["loop devices"]] == ["/dev/loop7"]
    assert findings["sb"] == [str(tmp_path / RUN / "sb")]


def test_objects_of_other_runs_are_never_leftovers(tmp_path):
    other = "f" * 32
    fake = client(
        images=Images(
            tagged=[Item("x", tags=[audit.built_tag_prefix(other) + "1" * 32])]
        )
    )

    report = audit.leftovers(
        fake,
        RUN,
        tmp_path,
        Observed(networks=[ENV_NET]),
        command=command(
            {
                "ip": f"{managed_bridge_interface('rsi-sbnet-ffff')} UP\n",
                "iptables": f"-A DOCKER-USER --comment rsi-{other}-sbx-1:x\n",
                "losetup": "/dev/loop3: []: (/var/lib/other/sb/build/1.img)\n",
            }
        ),
    )

    assert report["clean"] is True


def test_rules_are_derived_from_env_and_builder_networks():
    assert audit.rule_ids(RUN, [ENV_NET, BUILDER_NET, "bridge"]) == [
        f"rsi-{RUN}-sbb-77889900aabbccdd",
        f"rsi-{RUN}-sbx-00112233445566aa",
    ]


@pytest.mark.parametrize(
    ("attrs", "found"),
    [
        ({"Mounts": [{"Source": "/run/rsi-harness/1/s", "Destination": "/x"}]}, []),
        (
            {"Mounts": [{"Source": "/var/run/docker.sock", "Destination": "/d"}]},
            ["mount /var/run/docker.sock -> /d"],
        ),
        (
            {"HostConfig": {"Binds": ["/run/containerd/containerd.sock:/c"]}},
            ["mount /run/containerd/containerd.sock:/c"],
        ),
        ({"Config": {"Env": ["A=1", "DOCKER_HOST=tcp://x"]}}, ["env DOCKER_HOST"]),
    ],
)
def test_a6_container_findings(attrs, found):
    assert audit.container_findings(attrs) == found


def builder(status="running"):
    return Item(
        "rsi-sbb-1", labels={"rsi-harness.role": "sandbox-builder"}, status=status
    )


def env_service(phase="work", status="running"):
    return Item(
        "rsi-sbx-1-0",
        labels={"rsi-harness.role": "sandbox-env", "rsi-harness.sandbox-phase": phase},
        status=status,
    )


def test_kill_moments():
    planned = {"sandbox_images": [{"kind": "built", "state": "planned"}]}
    loading = {
        "sandbox_images": [{"kind": "built", "state": "loading", "handle": "i1"}]
    }
    top = lambda _id: {"Processes": [["root", "buildctl build --frontend x"]]}  # noqa: E731
    idle = lambda _id: {"Processes": [["root", "buildkitd"]]}  # noqa: E731

    assert audit.kill_moment("env_create", [], {}, idle) is None
    assert audit.kill_moment("env_create", [env_service()], {}, idle)
    assert audit.kill_moment("build", [builder()], planned, idle) is None
    assert audit.kill_moment("build", [builder()], {}, top) is None
    assert audit.kill_moment("build", [builder()], planned, top)
    assert audit.kill_moment("load", [], planned, idle) is None
    assert audit.kill_moment("load", [], loading, idle) == "image i1 loading"
    assert (
        audit.kill_moment("paused", [env_service("judge", "paused")], {}, idle) is None
    )
    assert audit.kill_moment("paused", [env_service("work", "paused")], {}, idle)
    judge_paused = {"sandbox_envs": [journaled_env("paused", "judge")]}
    work_ready = {"sandbox_envs": [journaled_env("ready")]}
    work_paused = {"sandbox_envs": [journaled_env("ready"), journaled_env("paused")]}
    assert audit.kill_moment("paused", [], judge_paused, idle) is None
    assert audit.kill_moment("paused", [], work_ready, idle) is None
    assert audit.kill_moment("paused", [], work_paused, idle) == "Work env e1 paused"


def journaled_env(state, phase="work"):
    """A lease record of an env (SandboxEnvLease) as the journal writes it."""
    return {"env_id": "e1", "owner": {"phase": phase}, "state": state}


@pytest.fixture
def harness():
    """A stand-in for the harness process the watcher may SIGKILL."""
    process = subprocess.Popen(["sleep", "60"])
    yield process
    process.kill()
    process.wait()


def test_watch_kills_the_harness_at_the_moment(tmp_path, harness):
    lease(
        tmp_path, sandbox_images=[{"kind": "built", "state": "loading", "handle": "i"}]
    )

    observed = audit.watch(
        client(), tmp_path, harness.pid, tmp_path / "w.json", kill_at="load"
    )

    assert harness.wait(5) == -9
    assert (observed.run_id, observed.killed_at) == (RUN, "load")
    assert Observed.load(tmp_path / "w.json").kill_detail == "image i loading"


def test_watch_kills_at_a_journaled_pause_without_ever_probing(tmp_path, harness):
    # The a7-paused failure: a socket probe (docker exec) in a Work env that
    # the submit froze meanwhile blocked the watcher until the Judge ended.
    lease(tmp_path, sandbox_envs=[journaled_env("ready")])
    probes = []
    service = env_service("work")
    service.exec_run = lambda *args, **kwargs: probes.append(args)

    class Freezing(Listing):
        def list(self, **kwargs):
            found = super().list(**kwargs)
            lease(tmp_path, sandbox_envs=[journaled_env("paused")])
            return found

    fake = client()
    fake.containers = Freezing(service)

    observed = audit.watch(
        fake, tmp_path, harness.pid, tmp_path / "w.json", kill_at="paused", interval=0
    )

    assert harness.wait(5) == -9
    assert (observed.killed_at, observed.kill_detail) == (
        "paused",
        "Work env e1 paused",
    )
    assert fake.containers.calls  # it did look before the freeze
    assert probes == []
    assert observed.sockets == {}


def test_watch_gives_up_on_a_moment_that_never_comes(tmp_path, harness):
    lease(tmp_path)
    ticks = iter([0.0, 5.0, 11.0])

    observed = audit.watch(
        client(),
        tmp_path,
        harness.pid,
        tmp_path / "w.json",
        kill_at="paused",
        give_up_sec=10,
        interval=0,
        clock=lambda: next(ticks),
    )

    assert harness.wait(5) == -9
    assert observed.killed_at is None
    assert observed.kill_detail == "paused never reached in 10s"


def test_watch_records_containers_networks_and_sockets(tmp_path):
    lease(tmp_path)
    judge = Item("judge-1", labels={"rsi-harness.role": "judge"})
    judge.exec_run = lambda *a, **k: SimpleNamespace(
        output=b"/run/rsi-harness/sandbox/s\n"
    )
    child = env_service("judge")
    child.exec_run = lambda *a, **k: SimpleNamespace(output=b"")
    work = Item(
        "work-1",
        labels={"rsi-harness.role": "work"},
        attrs={"Config": {"Env": ["DOCKER_HOST=unix:///x"]}},
    )
    process = subprocess.Popen(["sleep", "1"])
    try:
        observed = audit.watch(
            client(containers=[judge, child, work], networks=[Item(ENV_NET)]),
            tmp_path,
            process.pid,
            tmp_path / "w.json",
            interval=0.05,
        )
    finally:
        process.wait()

    assert observed.networks == [ENV_NET]
    assert observed.sockets == {
        "judge-1": {"role": "judge", "sockets": ["/run/rsi-harness/sandbox/s"]},
        "rsi-sbx-1-0": {"role": "sandbox-env", "sockets": []},
    }
    [finding] = [item for item in observed.containers.values() if item["findings"]]
    assert finding == {
        "name": "work-1",
        "role": "work",
        "findings": ["env DOCKER_HOST"],
    }


CONSOLE = """Run completed in 812.3s
  Run ID:           {run}
  Status:           completed
  Rounds:           1

Judge rounds:
  round-1: completed score=1.0
"""


def scenario_dir(
    tmp_path, *, notes=(), summary=None, clean=True, observed=None, teardown=5.0
):
    directory = tmp_path / "scenario"
    logs = directory / "logs" / "runs" / RUN
    logs.mkdir(parents=True)
    (directory / "run.log").write_text(CONSOLE.format(run=RUN))
    if teardown is not None:
        trial_result(logs, "fix-git__1", teardown)
    (logs / "agent_output.txt").write_text(
        "".join(f"RSI-ACCEPTANCE {line}\n" for line in notes)
    )
    if summary is not None:
        (logs / "harbor-summary.json").write_text(json.dumps(summary))
    (directory / "leftovers.json").write_text(
        json.dumps({"clean": clean, "findings": {"containers": [] if clean else ["x"]}})
    )
    (observed or Observed()).save(directory / "watch.json")
    return directory


def trial_result(logs, name, teardown):
    """An oracle trial of the Judge's Harbor job: its env stopped
    ``teardown`` seconds after its verifier ended."""
    trial = logs / "verifier" / "harbor-jobs" / "tb2-oracle" / name
    trial.mkdir(parents=True)
    ended = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
    finished = ended + timedelta(seconds=teardown)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "trial_name": name,
                "verifier": {"finished_at": ended.isoformat()},
                "finished_at": finished.isoformat(),
            }
        )
    )


def suite(ok=True):
    report = {"ok": ok, "trials": 6, "expected": 6}
    return {"oracle": report, "nop": report}


def test_run_result_reads_the_console():
    result = audit.run_result(CONSOLE.format(run=RUN))

    assert result == {
        "status": "completed",
        "rounds": [{"round": "round-1", "status": "completed", "score": "1.0"}],
    }


ENGINE_PATHS = {"POST /v1/containers/json": 400, "GET /v1/containers/json": 400}


def test_a1_and_a6_pass_on_a_clean_scored_run(tmp_path):
    work = {
        "ok": True,
        "sockets": ["/run/rsi-harness/sandbox/s"],
        "engine_paths": ENGINE_PATHS,
    }
    observed = Observed(
        containers={"a": {"name": "judge", "role": "judge", "findings": []}},
        sockets={
            "judge": {"role": "judge", "sockets": ["/run/rsi-harness/sandbox/s"]},
            "child": {"role": "sandbox-env", "sockets": []},
        },
    )
    directory = scenario_dir(
        tmp_path,
        notes=[
            f"a6-work {json.dumps(work)}",
            "a6-run-step pass",
            "a6-child-sockets none",
        ],
        summary={"suites": {"tb2": suite()}},
        observed=observed,
    )

    rows = audit.verdict("a1", directory)

    assert [(name, ok) for name, ok, _ in rows] == [
        ("A1", True),
        ("A6", True),
        ("A5 a1", True),
    ]
    assert "tb2: oracle 6/6 ok, nop ok" in rows[0][2]
    assert "1 oracle envs destroyed within 5.0 s" in rows[0][2]


@pytest.mark.parametrize(
    ("break_it", "needle"),
    [
        ("slow teardown", "env teardown over 60 s"),
        ("no trial", "no oracle trial result"),
        ("quarantine", "quarantined"),
        ("unknown outcome", "unknown-outcome"),
    ],
)
def test_a1_needs_a_normal_run(tmp_path, break_it, needle):
    """Spec M7's A1 path: every oracle env destroyed within 60 s, and no
    quarantine, unknown outcome or fail-closed anywhere in the run."""
    directory = scenario_dir(
        tmp_path,
        summary={"suites": {"tb2": suite()}},
        teardown={"slow teardown": 75.0, "no trial": None}.get(break_it, 5.0),
    )
    if break_it == "quarantine":
        with (directory / "run.log").open("a") as log:
            log.write("sandbox env e1 quarantined: drift\n")
    if break_it == "unknown outcome":
        trial_log = next((directory / "logs").rglob("fix-git__1")) / "trial.log"
        trial_log.write_text("env_destroy: unknown-outcome, retrying\n")

    [a1, _, _] = audit.verdict("a1", directory)

    assert a1[1] is False
    assert needle in a1[2]


@pytest.mark.parametrize("answer", [400, 404, None])
def test_a6_requires_the_engine_path_to_be_unsupported(tmp_path, answer):
    """Spec A6: ``/v1/containers/json`` answers 400 unsupported; a 404 (no
    route) or a probe that never asked is no pass."""
    engine = dict(ENGINE_PATHS)
    if answer is None:
        del engine["GET /v1/containers/json"]
    else:
        engine["GET /v1/containers/json"] = answer
    work = {
        "ok": True,
        "sockets": ["/run/rsi-harness/sandbox/s"],
        "engine_paths": engine,
    }
    observed = Observed(
        sockets={
            "judge": {"role": "judge", "sockets": ["/run/rsi-harness/sandbox/s"]},
            "child": {"role": "sandbox-env", "sockets": []},
        }
    )
    directory = scenario_dir(
        tmp_path,
        notes=[
            f"a6-work {json.dumps(work)}",
            "a6-run-step pass",
            "a6-child-sockets none",
        ],
        observed=observed,
    )

    [_, a6, _] = audit.verdict("a1", directory)

    assert a6[1] is (answer == 400)
    assert "relaxed" not in a6[2]
    if answer != 400:
        assert f"GET /v1/containers/json answered {answer}, not 400" in a6[2]


@pytest.mark.parametrize("missing", ["judge", "sandbox-env"])
def test_a6_fails_without_a_socket_probe_of_each_role(tmp_path, missing):
    """A watcher that never probed a Judge or a child is no evidence."""
    sockets = {
        "judge": {"role": "judge", "sockets": ["/run/rsi-harness/sandbox/s"]},
        "child": {"role": "sandbox-env", "sockets": []},
    }
    observed = Observed(
        sockets={name: v for name, v in sockets.items() if v["role"] != missing}
    )
    work = {"ok": True, "sockets": ["/run/rsi-harness/sandbox/s"]}
    directory = scenario_dir(
        tmp_path,
        notes=[
            f"a6-work {json.dumps(work)}",
            "a6-run-step pass",
            "a6-child-sockets none",
        ],
        observed=observed,
    )

    [_, a6, _] = audit.verdict("a1", directory)

    assert a6[1] is False
    assert ("Judge sockets []" if missing == "judge" else "child sockets []") in a6[2]


def test_a6_fails_on_a_socket_or_a_missing_probe(tmp_path):
    observed = Observed(
        sockets={
            "judge": {
                "role": "judge",
                "sockets": ["/run/rsi-harness/sandbox/s", "/var/run/docker.sock"],
            }
        }
    )
    directory = scenario_dir(tmp_path, observed=observed)

    [_, a6, _] = audit.verdict("a1", directory)

    assert a6[1] is False
    assert "docker.sock" in a6[2] and "child sockets" in a6[2]


def test_the_swebench_scenario_needs_a_scored_normal_run(tmp_path):
    report = {"ok": True, "trials": 3, "expected": 3}
    summary = {"suites": {"swebench": {"oracle": report, "nop": report}}}
    directory = scenario_dir(tmp_path, summary=summary)

    rows = audit.verdict("swebench", directory)

    assert [(name, ok) for name, ok, _ in rows] == [
        ("SWE", True),
        ("A5 swebench", True),
    ]
    assert "swebench: oracle 3/3 ok, nop ok" in rows[0][2]

    with (directory / "run.log").open("a") as log:
        log.write("sandbox env e1 quarantined: drift\n")
    [swe, _] = audit.verdict("swebench", directory)
    assert swe[1] is False and "quarantined" in swe[2]


def test_a2_and_a3_read_the_work_suites_and_the_judge(tmp_path):
    directory = scenario_dir(
        tmp_path,
        notes=[
            "work-suite tb2 oracle=True nop=True",
            "work-suite compose oracle=True nop=False",
        ],
        summary={"suites": {"compose": suite()}},
    )

    rows = {name: ok for name, ok, _ in audit.verdict("a2", directory)}

    assert rows == {"A2": True, "A3": False, "A5 a2": True}


@pytest.mark.parametrize(
    ("killed_at", "recover", "clean", "ok"),
    [
        ("build", "0", True, True),
        (None, "0", True, False),
        ("build", "1", True, False),
        ("build", "0", False, False),
    ],
)
def test_a7_needs_the_kill_the_recovery_and_no_leftovers(
    tmp_path, killed_at, recover, clean, ok
):
    directory = scenario_dir(
        tmp_path,
        clean=clean,
        observed=Observed(killed_at=killed_at, kill_detail="buildctl running"),
    )
    (directory / "recover.rc").write_text(recover + "\n")

    [a7, a5] = audit.verdict("a7-build", directory)

    assert a7[:2] == ("A7 build", ok)
    assert a5[:2] == ("A5 a7-build", clean)


def test_host_diff_fails_on_the_checks_objects_and_lists_other_runs_apart():
    before = {"containers": {"a": "m9-root-1"}, "bridges": {"rsi000000000001": None}}
    after = {
        "containers": {"a": "m9-root-1", "b": "m8-build-2", "c": RUN, "d": "f" * 32},
        "bridges": {"rsi000000000001": None, "rsi000000000002": None},
        "iptables": {
            f"-A DOCKER-USER --comment rsi-{RUN}-sbx-1:forward -j RSI_F_1": (
                f"rule:rsi-{RUN}-sbx-1:forward"
            ),
            "-A INPUT --comment rsi-other-sbx-2:input -j RSI_I_2": (
                "rule:rsi-other-sbx-2:input"
            ),
        },
    }

    split = audit.host_diff(before, after, runs={RUN})

    assert split["ours"] == {
        "containers": ["b", "c"],
        # A bridge no network names is never excused.
        "bridges": ["rsi000000000002"],
        "iptables": [f"-A DOCKER-USER --comment rsi-{RUN}-sbx-1:forward -j RSI_F_1"],
    }
    assert split["other runs"]["containers"] == [f"d (owner {'f' * 32})"]
    assert len(split["other runs"]["iptables"]) == 1


def test_host_snapshot_keeps_only_rsi_objects_with_their_owners():
    network = Item(ENV_NET, attrs={"Labels": {"rsi-harness.run-id": RUN}})
    network.labels = None
    fake = client(
        containers=[Item("c", labels={"rsi-harness.run-id": "m9-root-1"})],
        networks=[network],
        images=Images(
            tagged=[
                Item("i", tags=["rsi-sbx-img:1-2"], labels={"rsi-harness.run-id": RUN})
            ]
        ),
    )
    bridge = managed_bridge_interface(ENV_NET)
    snapshot = audit.host_snapshot(
        fake,
        command=command(
            {
                "ip": f"{bridge}@if1 UP\nrsi0123456789ab@if1 UP\nrsi915h UP\neth0 UP\n",
                "iptables": (
                    "-N RSI_F_1\n-N RSI_F_9\n-A INPUT -j ACCEPT\n"
                    "-A DOCKER-USER -o x -m comment --comment rsi-r-sbx-1:forward"
                    " -j RSI_F_1\n"
                ),
                "losetup": "/dev/loop1: (/d/run7/sb/build/1.img)\n/dev/loop2: (/x)\n",
            }
        ),
    )

    assert snapshot["containers"] == {"c": "m9-root-1"}
    assert snapshot["networks"] == {ENV_NET: RUN}
    assert snapshot["bridges"] == {bridge: RUN, "rsi0123456789ab": None}
    assert snapshot["iptables"] == {
        "-N RSI_F_1": "rule:rsi-r-sbx-1:forward",
        "-N RSI_F_9": None,
        "-A DOCKER-USER -o x -m comment --comment rsi-r-sbx-1:forward -j RSI_F_1": (
            "rule:rsi-r-sbx-1:forward"
        ),
    }
    assert snapshot["loop devices"] == {"/dev/loop1: (/d/run7/sb/build/1.img)": "run7"}
    assert snapshot["built images"] == {"rsi-sbx-img:1-2": RUN}
    assert fake.containers.calls == [
        {"all": True, "filters": {"label": "rsi-harness.run-id"}}
    ]


def test_the_acceptance_runs_are_found_under_the_scratch(tmp_path):
    (tmp_path / "acc" / "a1" / "data" / "leases").mkdir(parents=True)
    (tmp_path / "acc" / "a1" / "data" / "leases" / f"{RUN}.json").write_text("{}")
    (tmp_path / "acc" / "a2" / "logs" / "runs" / ("e" * 32)).mkdir(parents=True)

    assert audit.runs_under(tmp_path) == {RUN, "e" * 32}


def test_observed_round_trips(tmp_path):
    observed = Observed(run_id=RUN, networks=[ENV_NET], images=[DIGEST])
    observed.save(tmp_path / "w.json")

    assert Observed.load(tmp_path / "w.json") == observed
    assert Observed.load(tmp_path / "missing.json") == Observed()


def test_the_workdir_volume_is_retained_until_cleanup_then_a_leftover(tmp_path):
    fake = client(volumes=[Item("rsi-harness-workdir-" + "a" * 64)])
    outputs = command({})

    before = audit.leftovers(fake, RUN, tmp_path, Observed(), command=outputs)
    after = audit.leftovers(
        fake, RUN, tmp_path, Observed(), strict=True, command=outputs
    )

    assert before["clean"] is True
    assert before["retained until cleanup"] == ["rsi-harness-workdir-" + "a" * 64]
    assert after["clean"] is False
    assert after["findings"]["volumes"] == ["rsi-harness-workdir-" + "a" * 64]


def test_a5_also_needs_nothing_after_cleanup(tmp_path):
    directory = scenario_dir(tmp_path, summary={"suites": {"build": suite()}})
    (directory / "leftovers-after-cleanup.json").write_text(
        json.dumps({"clean": False, "findings": {"volumes": ["v"]}})
    )

    [a4, a5] = audit.verdict("a4", directory)

    assert a4[:2] == ("A4", True)
    assert a5[1] is False and "after cleanup" in a5[2]


def test_a8_needs_the_judge_across_the_work_deadline(tmp_path):
    def a8(submit_lead, judge_from, judge_to):
        directory = scenario_dir(
            tmp_path / str(submit_lead),
            notes=[
                "work-deadline 1000",
                f"late-submit {1000 - submit_lead}",
            ],
        )
        report = next((directory / "logs").rglob(RUN)) / "submissions" / "round-1"
        report.mkdir(parents=True)
        (report / "report.json").write_text(
            json.dumps(
                {
                    "submitted_at": 1000 + judge_to,
                    "runtime_seconds": judge_to - judge_from,
                }
            )
        )
        [row, _] = audit.verdict("a8", directory)
        return row

    ok = a8(60, -55, 400)
    assert ok[:2] == ("A8", True)
    assert "submitted 60 s before the Work deadline" in ok[2]
    # The Judge ended before the deadline: A8 was not exercised.
    assert a8(61, -55, -5)[1] is False
    # The submit came far too early.
    assert a8(300, -290, 400)[1] is False


GPU = "GPU-7b7ea8bd-7f75-8bea-4001-d2a5cd66bade"
VLLM_NOTES = (
    "checkpoint-download 0 Qwen/Qwen2.5-1.5B-Instruct@989aa79 12s",
    "checkpoint-bytes 3098996406",
    "submit-exit 0",
)
PLUGIN = "rsi_sandbox_harbor:ManagedSandboxEnvironment"


def vllm_summary(**changes):
    """The Judge's vllm-demo-summary.json of a run that did everything."""
    summary = {
        "reward": 0.5,
        "vllm": {
            "healthy": True,
            "on_judge_gpu": True,
            "checkpoint": "/workspace/checkpoint",
            "served_checkpoint": True,
            "serving_gpus": [GPU],
            "expected_gpus": [GPU],
            "gpus": [{"uuid": GPU, "name": "NVIDIA H100 NVL", "vllm_mib": 48529}],
            "models": [{"id": "rsi-checkpoint", "root": "/workspace/checkpoint"}],
        },
        "completions": {
            "requests_served": 24,
            "requests_aborted": 0,
            "access_log_200": 24,
            "prompt_tokens": 46350.0,
            "generation_tokens": 2658.0,
        },
        "trials": [
            {
                "task": "rsi/hello-file",
                "environment": PLUGIN,
                "outcome": "verified",
                "reward": 1.0,
                "requested": True,
                "input_tokens": 9120,
                "completed": True,
            },
            {
                "task": "regex-log",
                "environment": PLUGIN,
                "outcome": "verified",
                "reward": 0.0,
                "requested": True,
                "input_tokens": 37230,
                "completed": True,
            },
        ],
        "infra_errors": [],
    }
    for key, value in changes.items():
        summary[key] = value
    return summary


def vllm_observed(envs=2, judge_sockets=None):
    """The watch record: the Judge (its endpoint and vLLM's own IPC socket)
    and ``envs`` broker env containers."""
    containers = {"j": {"name": "judge", "role": "judge", "findings": []}}
    for index in range(envs):
        containers[f"e{index}"] = {
            "name": f"rsi-sbenv-{index}",
            "role": "sandbox-env",
            "findings": [],
        }
    sockets = (
        [audit.ENDPOINT_SOCKET, "/tmp/0c1d2e3f-ipc"]
        if judge_sockets is None
        else judge_sockets
    )
    return Observed(
        containers=containers,
        sockets={"judge": {"role": "judge", "sockets": sockets}},
    )


def vllm_dir(tmp_path, summary, notes=VLLM_NOTES, console=None, observed=None):
    directory = scenario_dir(
        tmp_path,
        notes=notes,
        teardown=None,
        observed=vllm_observed() if observed is None else observed,
    )
    if console is not None:
        (directory / "run.log").write_text(console)
    verifier = next((directory / "logs").rglob(RUN)) / "verifier"
    verifier.mkdir(parents=True, exist_ok=True)
    (verifier / "vllm-demo-summary.json").write_text(json.dumps(summary))
    return directory


def test_the_vllm_demo_passes_whatever_the_score(tmp_path):
    directory = vllm_dir(tmp_path, vllm_summary())

    rows = audit.verdict("vllm", directory)

    assert [(name, ok) for name, ok, _ in rows] == [
        ("V0 Work", True),
        ("V1 vLLM", True),
        ("V2 requests", True),
        ("V3 trials", True),
        ("A5 vllm", True),
    ]
    assert f"serving on {GPU} NVIDIA H100 NVL +48529 MiB" in rows[1][2]
    assert "model roots ['/workspace/checkpoint']" in rows[1][2]
    assert "24 completions served (metrics; 0 aborted apart)" in rows[2][2]
    assert "per trial in rsi/hello-file 9120, regex-log 37230" in rows[2][2]
    assert (
        "rsi/hello-file verified reward 1.0; regex-log verified reward 0.0"
        in rows[3][2]
    )
    assert "2 broker env container(s)" in rows[3][2]
    assert "no infrastructure error" in rows[3][2]


BREAKS = {
    "download failed": ("V0 Work", "download 1 "),
    "no submit": ("V0 Work", "submit exit None, 0 submission report(s)"),
    "not on a Judge GPU": ("V1 vLLM", "serving on no GPU"),
    "no Judge GPU named": ("V1 vLLM", "Judge GPUs []"),
    "another GPU": ("V1 vLLM", "GPU-other"),
    "another checkpoint": ("V1 vLLM", "model roots ['/root/.cache/hub/qwen']"),
    "no request": ("V2 requests", "0 completions"),
    "all aborted": ("V2 requests", "0 completions served (metrics; 24 aborted"),
    "a silent trial": ("V2 requests", "no request from ['regex-log']"),
    "infrastructure error": ("V3 trials", "RuntimeError"),
    "incomplete trial": ("V3 trials", "regex-log unverified reward None"),
    "no trial": ("V3 trials", "no trial result"),
    "round failed": ("V3 trials", "Judge round"),
    "quarantine": ("V3 trials", "quarantined"),
    "outside the plugin": ("V3 trials", "not through rsi_sandbox_harbor"),
    "too few envs": ("V3 trials", "1 sandbox env container(s) for 2 trials"),
    "no Judge probe": ("V3 trials", "Judge sockets []"),
    "a control socket": ("V3 trials", "/var/run/docker.sock"),
    "an A6 finding": ("V3 trials", "judge: env DOCKER_HOST"),
}


@pytest.mark.parametrize("break_it", list(BREAKS))
def test_the_vllm_demo_fails_on_any_missing_evidence(tmp_path, break_it):
    row, needle = BREAKS[break_it]
    summary = vllm_summary()
    vllm = summary["vllm"]
    trials = summary["trials"]
    notes = list(VLLM_NOTES)
    console = None
    observed = vllm_observed()
    if break_it == "download failed":
        notes[0] = "checkpoint-download 1 Qwen/Qwen2.5-1.5B-Instruct@989aa79 3s"
    if break_it == "no submit":
        notes.remove("submit-exit 0")
    if break_it == "not on a Judge GPU":
        vllm.update(on_judge_gpu=False, serving_gpus=[])
    if break_it == "no Judge GPU named":
        # As demo_report would have once said for a Judge with no GPU named.
        vllm.update(expected_gpus=[])
    if break_it == "another GPU":
        vllm.update(serving_gpus=[GPU, "GPU-other"])
    if break_it == "another checkpoint":
        vllm["models"] = [{"id": "rsi-checkpoint", "root": "/root/.cache/hub/qwen"}]
    if break_it == "no request":
        summary["completions"]["requests_served"] = 0
    if break_it == "all aborted":
        summary["completions"].update(requests_served=0, requests_aborted=24)
    if break_it == "a silent trial":
        trials[1].update(requested=False, input_tokens=0)
    if break_it == "infrastructure error":
        summary["infra_errors"] = ["regex-log__x: RuntimeError: env_start failed"]
    if break_it == "incomplete trial":
        trials[1].update(outcome="unverified", reward=None, completed=False)
    if break_it == "no trial":
        summary["trials"] = []
    if break_it == "round failed":
        console = CONSOLE.format(run=RUN).replace(
            "round-1: completed score=1.0", "round-1: failed score=None"
        )
    if break_it == "quarantine":
        console = CONSOLE.format(run=RUN) + "sandbox env e1 quarantined: drift\n"
    if break_it == "outside the plugin":
        trials[1]["environment"] = "docker"
    if break_it == "too few envs":
        observed = vllm_observed(envs=1)
    if break_it == "no Judge probe":
        observed.sockets = {}
    if break_it == "a control socket":
        observed = vllm_observed(
            judge_sockets=[audit.ENDPOINT_SOCKET, "/var/run/docker.sock"]
        )
    if break_it == "an A6 finding":
        observed.containers["j"]["findings"] = ["env DOCKER_HOST"]
    directory = vllm_dir(tmp_path, summary, notes, console, observed)

    rows = {name: (ok, detail) for name, ok, detail in audit.verdict("vllm", directory)}

    assert rows[row][0] is False, rows[row]
    assert needle in rows[row][1]
    others = [name for name, (ok, _) in rows.items() if not ok and name != row]
    assert others == []


def test_the_vllm_demo_takes_the_harness_submission_report_for_the_submit(tmp_path):
    # The Work deadline passed during the round: agent.sh never printed
    # its post-submit note, but the Harness recorded the submission.
    notes = [note for note in VLLM_NOTES if not note.startswith("submit-exit")]
    directory = vllm_dir(tmp_path, vllm_summary(), notes)
    report = next((directory / "logs").rglob(RUN)) / "submissions" / "round-1"
    report.mkdir(parents=True)
    (report / "report.json").write_text(json.dumps({"submitted_at": 1.0}))

    [work, *_] = audit.verdict("vllm", directory)

    assert work[:2] == ("V0 Work", True)
    assert "submit exit None, 1 submission report(s)" in work[2]


def test_the_vllm_demo_fails_without_a_judge_summary(tmp_path):
    directory = scenario_dir(tmp_path, notes=VLLM_NOTES, teardown=None)

    rows = {name: ok for name, ok, _ in audit.verdict("vllm", directory)}

    assert rows == {
        "V0 Work": True,
        "V1 vLLM": False,
        "V2 requests": False,
        "V3 trials": False,
        "A5 vllm": True,
    }


class FakeDocker:
    def __init__(self, fake):
        self.fake = fake

    def from_env(self, **_kwargs):
        return self.fake


@pytest.mark.parametrize("strict", [False, True])
def test_the_leftovers_cli_with_and_without_strict(
    tmp_path, monkeypatch, capsys, strict
):
    """The exact command lines scripts/operator/sandbox_acceptance.sh runs."""
    import docker

    fake = client(volumes=[Item("rsi-harness-workdir-" + "a" * 64)])
    monkeypatch.setattr(docker, "from_env", FakeDocker(fake).from_env)
    monkeypatch.setattr(audit, "_command", command({}))
    Observed(run_id=RUN).save(tmp_path / "watch.json")
    argv = ["leftovers", "--data-root", str(tmp_path), "--observed"]
    argv += [str(tmp_path / "watch.json"), "--run-id", RUN]

    code = audit.main([*argv, "--strict"] if strict else argv)

    report = json.loads(capsys.readouterr().out)
    assert (code, report["clean"]) == ((1, False) if strict else (0, True))
    assert report["run_id"] == RUN


def test_the_host_diff_cli_fails_only_on_the_checks_objects(
    tmp_path, monkeypatch, capsys
):
    import docker

    fake = client(
        containers=[
            Item("mine", labels={"rsi-harness.run-id": "m9-root-1"}),
            Item("theirs", labels={"rsi-harness.run-id": "f" * 32}),
        ]
    )
    monkeypatch.setattr(docker, "from_env", FakeDocker(fake).from_env)
    monkeypatch.setattr(audit, "_command", command({}))
    before = tmp_path / "before.json"
    before.write_text(json.dumps({"containers": {}}))

    assert audit.main(["host-diff", str(before), "--runs-under", str(tmp_path)]) == 1
    split = json.loads(capsys.readouterr().out)
    assert split["ours"]["containers"] == ["mine"]
    assert split["other runs"]["containers"] == [f"theirs (owner {'f' * 32})"]

    fake.containers.items.pop(0)
    assert audit.main(["host-diff", str(before)]) == 0
