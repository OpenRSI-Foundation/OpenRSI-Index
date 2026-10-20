"""A reported ABI must not silently weaken the ruleset on broken backports."""

import ctypes
import errno
import json
import os
import struct

import pytest

from rsi_harness.judge_agent import judge_isolation as isolation
from rsi_harness.judge_agent import rsi_agent, rsi_isolate


@pytest.fixture(autouse=True)
def strict_policy(monkeypatch):
    monkeypatch.delenv(isolation.ALLOW_MISSING_SIGNAL_ENV, raising=False)


def kernel(monkeypatch, failures):
    calls = []
    descriptors = []

    def create(_libc, number, address, size, flags):
        assert number == isolation.LANDLOCK_CREATE_RULESET
        assert flags == 0
        attr = ctypes.string_at(address, size).ljust(24, b"\0")
        fs, net, scoped = struct.unpack("<QQQ", attr)
        calls.append((fs, net, scoped, size))
        error = failures.get(scoped)
        if error:
            ctypes.set_errno(error)
            return -1
        fd = os.open(os.devnull, os.O_RDONLY)
        descriptors.append(fd)
        return fd

    monkeypatch.setattr(isolation, "syscall", create)
    return calls, descriptors


def test_strict_default_fails_closed_on_incomplete_abi6(monkeypatch):
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL})
    with pytest.raises(
        isolation.Failure, match="landlock_create_ruleset failed"
    ) as exc:
        isolation.Ruleset(None, 6)
    assert "judge_allow_missing_landlock_signal" in exc.value.detail
    assert calls == [(65535, 0, 3, 24)]


def test_opt_in_keeps_all_fs_and_abstract_socket_rights(monkeypatch, capsys):
    calls, descriptors = kernel(monkeypatch, {3: errno.EINVAL})
    ruleset = isolation.Ruleset(None, 6, allow_missing_landlock_signal=True)
    assert ruleset.scoped == 1
    assert ruleset.handled == 65535
    assert calls == [(65535, 0, 3, 24), (65535, 0, 1, 24)]
    assert "signal scoping is unavailable" in capsys.readouterr().err
    ruleset.close()
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("error", [errno.EPERM, errno.ENOMEM, errno.EMFILE])
def test_unrelated_errors_never_trigger_retry(monkeypatch, error):
    calls, _ = kernel(monkeypatch, {3: error})
    with pytest.raises(isolation.Failure):
        isolation.Ruleset(None, 6, allow_missing_landlock_signal=True)
    assert len(calls) == 1


def test_rejected_abstract_scope_does_not_fall_back_to_zero(monkeypatch):
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL, 1: errno.EINVAL})
    with pytest.raises(isolation.Failure):
        isolation.Ruleset(None, 6, allow_missing_landlock_signal=True)
    assert [call[2] for call in calls] == [3, 1]


def test_newer_abis_do_not_use_abi6_workaround(monkeypatch):
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL})
    with pytest.raises(isolation.Failure):
        isolation.Ruleset(None, 7, allow_missing_landlock_signal=True)
    assert len(calls) == 1


def test_complete_abi6_uses_full_policy_even_with_opt_in(monkeypatch, capsys):
    calls, _ = kernel(monkeypatch, {})
    ruleset = isolation.Ruleset(None, 6, allow_missing_landlock_signal=True)
    assert ruleset.scoped == 3
    ruleset.close()
    assert len(calls) == 1
    assert not capsys.readouterr().err


def test_report_probes_creation_and_reports_actual_scope(monkeypatch, capsys):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    calls, descriptors = kernel(monkeypatch, {3: errno.EINVAL})
    monkeypatch.setattr(isolation, "load_libc", lambda: None)
    monkeypatch.setattr(
        rsi_isolate,
        "isolation_mode",
        lambda _: {"mode": "landlock", "landlock_abi": 6, "fallback": "none"},
    )
    assert rsi_isolate.report_isolation() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["landlock_scoped"] == 1
    assert report["landlock_signal_scope"] is False
    assert len(calls) == 2
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_report_cannot_claim_success_from_abi_alone(monkeypatch, capsys):
    kernel(monkeypatch, {3: errno.EINVAL})
    monkeypatch.setattr(isolation, "load_libc", lambda: None)
    monkeypatch.setattr(
        rsi_isolate,
        "isolation_mode",
        lambda _: {"mode": "landlock", "landlock_abi": 6, "fallback": "none"},
    )
    with pytest.raises(isolation.Failure):
        rsi_isolate.report_isolation()
    assert not capsys.readouterr().out


def test_operator_policy_cannot_be_forwarded_to_command(monkeypatch):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    assert isolation.ALLOW_MISSING_SIGNAL_ENV not in rsi_isolate._environment([])
    for suffix in ("", "=1"):
        with pytest.raises(isolation.Failure, match="never passed on"):
            rsi_isolate._environment([isolation.ALLOW_MISSING_SIGNAL_ENV + suffix])


@pytest.mark.parametrize("value", ["1", "true", "anything"])
def test_shared_ruleset_does_not_take_policy_from_environment(monkeypatch, value):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, value)
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL})
    with pytest.raises(isolation.Failure):
        isolation.Ruleset(None, 6)
    assert len(calls) == 1


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_isolate_parses_explicit_operator_policy(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, value)
    assert rsi_isolate._allow_missing_landlock_signal() is (value == "1")


@pytest.mark.parametrize("value", ["", "yes", "false", "2"])
def test_isolate_rejects_invalid_operator_policy(monkeypatch, value):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, value)
    with pytest.raises(isolation.Failure, match="must be 0 or 1"):
        rsi_isolate.report_isolation()


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("operation", ["probe", "run"])
def test_rsi_agent_uses_spec_policy_despite_opposite_environment(
    monkeypatch, allow, operation, capsys
):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "0" if allow else "1")
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL})
    spec = {"allow_missing_landlock_signal": allow}
    monkeypatch.setattr(rsi_agent, "_load_spec", lambda *_: spec)
    monkeypatch.setattr(isolation, "load_libc", lambda: None)
    monkeypatch.setattr(
        rsi_agent,
        "isolation_mode",
        lambda *_: {"mode": "landlock", "landlock_abi": 6, "fallback": "none"},
    )
    monkeypatch.setattr(isolation, "readable_paths", lambda _: [])
    monkeypatch.setattr(isolation, "allow_devices", lambda _: None)

    def invoke():
        if operation == "probe":
            rsi_agent.report_isolation()
        else:
            ruleset = rsi_agent.call_ruleset(None, 6, spec, "/state", [])
            ruleset.close()

    if allow:
        invoke()
        assert [call[2] for call in calls] == [3, 1]
        if operation == "probe":
            assert json.loads(capsys.readouterr().out)["landlock_signal_scope"] is False
    else:
        with pytest.raises(isolation.Failure):
            invoke()
        assert len(calls) == 1


def test_isolate_run_uses_the_same_policy_as_its_probe(monkeypatch):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL})
    monkeypatch.setattr(isolation.Ruleset, "allow", lambda *_: True)
    monkeypatch.setattr(isolation, "allow_devices", lambda _: None)
    ruleset = rsi_isolate._ruleset(None, 6, [], [], "/private", None)
    assert ruleset.scoped == 1
    assert [call[2] for call in calls] == [3, 1]
    ruleset.close()


@pytest.mark.parametrize(
    "abi, failures, expected",
    [(4, {}, False), (6, {}, False), (6, {3: errno.EINVAL}, True)],
)
def test_native_functional_policy_selects_only_missing_signal(
    monkeypatch, abi, failures, expected
):
    from tests.landlock_helpers import native_signal_compatibility

    calls, _ = kernel(monkeypatch, failures)
    monkeypatch.setattr(isolation, "landlock_abi", lambda _: (abi, ""))
    assert native_signal_compatibility.__wrapped__() is expected
    assert [call[2] for call in calls] == (
        [3, 3, 1] if expected else [3 if abi >= 6 else 0]
    )


@pytest.mark.parametrize(
    "abi, failures",
    [
        (6, {3: errno.EPERM}),
        (7, {3: errno.EINVAL}),
        (6, {3: errno.EINVAL, 1: errno.EINVAL}),
    ],
)
def test_native_functional_policy_does_not_hide_other_failures(
    monkeypatch, abi, failures
):
    from tests.landlock_helpers import native_signal_compatibility

    kernel(monkeypatch, failures)
    monkeypatch.setattr(isolation, "landlock_abi", lambda _: (abi, ""))
    with pytest.raises(isolation.Failure):
        native_signal_compatibility.__wrapped__()
