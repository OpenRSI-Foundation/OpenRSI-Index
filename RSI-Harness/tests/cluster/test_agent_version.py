from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsi_harness.cluster.agent_version import parse_agent_version
from rsi_harness.cluster.bluevela import adapter
from rsi_harness.errors import SetupError


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("codex-cli 0.149.0\n", "0.149.0"),
        ("2.1.246 (Claude Code)\n", "2.1.246"),
        ("Wrapper starting CLI\ncodex-cli v0.149.0\n", "0.149.0"),
        ("v0.149.0-beta.1+build.17\n", "0.149.0-beta.1+build.17"),
        ("version 0.149.0\nCLI 0.149.0\n", "0.149.0"),
    ],
)
def test_version_contract_allows_banners_and_identical_repetition(output, expected):
    assert parse_agent_version(output) == expected


@pytest.mark.parametrize(
    "output",
    ["", "CLI unknown", "1.2.3.4", "codex-cli 0.149.0\nwrapper 1.2.3\n"],
)
def test_version_contract_refuses_unknown_or_ambiguous_output(output):
    with pytest.raises(SetupError, match="missing or ambiguous"):
        parse_agent_version(output)


def test_submit_host_probe_parses_both_streams_and_refuses_ambiguity(monkeypatch):
    result = CompletedProcess(["wrapper"], 0, "Wrapper starting\n", "v0.149.0\n")
    monkeypatch.setattr(adapter.subprocess, "run", lambda *_a, **_kw: result)
    assert adapter._binary_version(Path("/shared/wrapper")) == "0.149.0"
    result.stdout = "Wrapper version 1.2.3\n"
    with pytest.raises(SetupError, match="/shared/wrapper version.*ambiguous"):
        adapter._binary_version(Path("/shared/wrapper"))


def test_submit_host_probe_reports_exit_without_leaking_wrapper_output(monkeypatch):
    result = CompletedProcess(["wrapper"], 127, "private-wrapper-value", "")
    monkeypatch.setattr(adapter.subprocess, "run", lambda *_a, **_kw: result)
    with pytest.raises(SetupError) as caught:
        adapter._binary_version(Path("/shared/wrapper"))
    assert "/shared/wrapper version: exited 127" in str(caught.value)
    assert "private-wrapper-value" not in str(caught.value)
