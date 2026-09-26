from __future__ import annotations

import json

import pytest

from rsi_harness.runtime.redaction import redact_exact_values, redact_text


@pytest.mark.parametrize(
    "text",
    [
        "token budget; token embeddings; token counts",
        "predict token 2001 given tokens 1–2000",
        "eos_token = 151643; pad_token = '<pad>'; token = tokenizer.next()",
        "using arrival_token = barrier.arrive();",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN=1 REQUIRE_API_KEY=1",
        "basic block; digest length; credential rotation; secret sharing",
        '{"description":"Analyze answers, token lengths","token":151643,"loss":0.25}',
    ],
)
def test_research_content_is_not_a_credential(text: str) -> None:
    assert redact_text(text) == text


def test_json_trajectory_keeps_structure_and_feedback_around_credentials() -> None:
    event = {
        "type": "assistant",
        "headers": {"Authorization": "Bearer dummy-header", "X-API-Key": "dummy-key"},
        "content": (
            'token budget; headers={"Authorization":"Basic dummy-basic"}; loss=0.25'
        ),
        "token": 151643,
        "feedback": "keep the entire feedback",
    }
    safe = redact_text(json.dumps(event) + "\n")
    decoded = json.loads(safe)
    assert decoded == {
        **event,
        "headers": {"Authorization": "[REDACTED]", "X-API-Key": "[REDACTED]"},
        "content": 'token budget; headers={"Authorization":"[REDACTED]"}; loss=0.25',
    }
    assert safe.endswith("\n")


def test_explicit_headers_assignments_and_url_credentials_are_redacted() -> None:
    text = (
        "access_token=dummy-access&safe=yes\n"
        "Authorization: Digest username=user, response=dummy-response\n"
        'headers={"X-API-Key":"dummy-header"}; HF_TOKEN=dummy-hub\n'
        "https://example.test/?token=dummy-query&loss=0.25\n"
        "credential=dummy-credential"
    )
    safe = redact_text(text)
    for secret in (
        "dummy-access",
        "dummy-response",
        "dummy-header",
        "dummy-hub",
        "dummy-query",
        "dummy-credential",
    ):
        assert secret not in safe
    assert "safe=yes" in safe
    assert "loss=0.25" in safe


def test_known_credentials_are_redacted_when_json_escaped() -> None:
    secret = 'dummy-quoted"credential\\tail'
    event = {"text": secret, "loss": 0.25}
    safe = redact_exact_values(json.dumps(event), {secret})
    assert json.loads(safe) == {"text": "[REDACTED]", "loss": 0.25}


def test_repeated_redaction_preserves_existing_markers() -> None:
    text = "Authorization: Bearer dummy-only\naccess_token=dummy-access"
    expected = "Authorization: [REDACTED]\naccess_token=[REDACTED]"
    assert redact_text(text) == expected
    assert redact_text(expected) == expected


@pytest.mark.parametrize("text", [
    "Engine bug: access_token=dummy-secret&safe=yes",
    "Error: cannot normalize paths: credential=dummy-secret",
    "https://example.test/?access_token=dummy-secret&safe=yes",
    'config="HF_TOKEN=dummy-secret"',
])
def test_diagnostic_prefix_does_not_hide_credential_assignment(text: str) -> None:
    assert redact_text(text) == text.replace("dummy-secret", "[REDACTED]")


@pytest.mark.parametrize("text", [
    r"export GH_TOKEN=dummy-secret\nwandb login <YOUR_WANDB_KEY>",
    r"Authorization: Bearer dummy-secret\nloss=0.25",
    r"https://example.test/?token=dummy-secret\nnext command",
])
def test_escaped_newline_keeps_following_commands_and_feedback(text: str) -> None:
    assert redact_text(text) == text.replace("dummy-secret", "[REDACTED]").replace(
        "Bearer [REDACTED]", "[REDACTED]"
    )


def test_streamed_json_matches_complete_output_redaction(tmp_path) -> None:
    from rsi_harness.cluster.bluevela.runtime import _safe_output
    from rsi_harness.runtime.docker import _RedactedOutputWriter

    raw = (
        json.dumps(
            {
                "description": "Analyze answers, token lengths",
                "text": "softmax(x); dummy-runtime-key",
                "headers": {"Authorization": "Bearer dummy-header"},
                "eos_token": 151643,
            }
        )
        + "\n"
    )
    path = tmp_path / "trajectory.jsonl"
    chunks = []
    writer = _RedactedOutputWriter(path, ("dummy-runtime-key",), chunks.append)
    encoded = raw.encode()
    for offset in range(0, len(encoded), 7):
        writer.append(encoded[offset : offset + 7])
    writer.finish()
    expected = {
        "description": "Analyze answers, token lengths",
        "text": "softmax(x); [REDACTED]",
        "headers": {"Authorization": "[REDACTED]"},
        "eos_token": 151643,
    }
    assert json.loads(path.read_text()) == expected
    assert "".join(chunks) == path.read_text()
    assert json.loads(_safe_output(raw, {"dummy-runtime-key"})) == expected
