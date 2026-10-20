"""A reported ABI must not silently weaken the ruleset on broken backports."""

import ctypes
import errno
import json
import os
import struct

import pytest

from rsi_harness.judge_agent import judge_isolation as isolation
from rsi_harness.judge_agent import rsi_isolate


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
    assert isolation.ALLOW_MISSING_SIGNAL_ENV in exc.value.detail
    assert calls == [(65535, 0, 3, 24)]


def test_opt_in_keeps_all_fs_and_abstract_socket_rights(monkeypatch, capsys):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    calls, descriptors = kernel(monkeypatch, {3: errno.EINVAL})
    ruleset = isolation.Ruleset(None, 6)
    assert ruleset.scoped == 1
    assert ruleset.handled == 65535
    assert calls == [(65535, 0, 3, 24), (65535, 0, 1, 24)]
    assert "signal scoping is unavailable" in capsys.readouterr().err
    ruleset.close()
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("error", [errno.EPERM, errno.ENOMEM, errno.EMFILE])
def test_unrelated_errors_never_trigger_retry(monkeypatch, error):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    calls, _ = kernel(monkeypatch, {3: error})
    with pytest.raises(isolation.Failure):
        isolation.Ruleset(None, 6)
    assert len(calls) == 1


def test_rejected_abstract_scope_does_not_fall_back_to_zero(monkeypatch):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL, 1: errno.EINVAL})
    with pytest.raises(isolation.Failure):
        isolation.Ruleset(None, 6)
    assert [call[2] for call in calls] == [3, 1]


def test_newer_abis_do_not_use_abi6_workaround(monkeypatch):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    calls, _ = kernel(monkeypatch, {3: errno.EINVAL})
    with pytest.raises(isolation.Failure):
        isolation.Ruleset(None, 7)
    assert len(calls) == 1


def test_complete_abi6_uses_full_policy_even_with_opt_in(monkeypatch, capsys):
    monkeypatch.setenv(isolation.ALLOW_MISSING_SIGNAL_ENV, "1")
    calls, _ = kernel(monkeypatch, {})
    ruleset = isolation.Ruleset(None, 6)
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
