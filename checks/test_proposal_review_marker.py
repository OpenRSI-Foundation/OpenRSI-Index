import pytest
from proposal_review_marker import (
    ReviewMarker,
    canonical_proposal_bytes,
    canonical_proposal_hash,
    parse_review_marker,
    render_review_marker,
)


def test_hash_is_stable_and_body_sensitive():
    assert canonical_proposal_hash("T", "B") == canonical_proposal_hash("T", "B")
    assert canonical_proposal_hash("T", "B") != canonical_proposal_hash("T", "B ")


def test_canonical_hash_uses_literal_unicode_json_and_fixed_digest():
    # This fixture's digest was independently calculated for the literal UTF-8
    # bytes of {"body":"café ☃","title":"提案"}.
    expected_bytes = '{"body":"café ☃","title":"提案"}'.encode()

    assert canonical_proposal_bytes("提案", "café ☃") == expected_bytes
    assert b"\\u" not in canonical_proposal_bytes("提案", "café ☃")
    assert canonical_proposal_hash("提案", "café ☃") == (
        "c1616bf2b924838fecfe134fed6ead5e8d34665e0c78746e5ca3784ad173d9b5"
    )


def test_pass_marker_round_trips_unicode_as_schema_2():
    marker = render_review_marker(
        decision="Pass",
        proposal_sha256="a" * 64,
        discussion_node_id="D_kw中文",
    )

    assert "D_kw中文" in marker
    assert "\\u" not in marker
    assert parse_review_marker(marker) == ReviewMarker(
        schema=2,
        decision="Pass",
        proposal_sha256="a" * 64,
        discussion_node_id="D_kw中文",
    )


def test_reject_decision_is_not_eligible():
    assert ReviewMarker(2, "Reject", "a" * 64, "D_1").eligible is False


@pytest.mark.parametrize(
    "decision", ["Accept", "Strong Accept", "Strong Reject", "require human review"]
)
def test_removed_decisions_cannot_be_rendered(decision):
    with pytest.raises(ValueError, match="canonical"):
        render_review_marker(decision, "a" * 64, "D_1")


@pytest.mark.parametrize("decision", ["Accept", "Strong Accept"])
def test_legacy_schema_1_acceptance_markers_remain_read_compatible_and_eligible(decision):
    marker = (
        '<!-- rsi-proposal-review:{"decision":"'
        + decision
        + '",'
        '"discussion_node_id":"D_1","proposal_sha256":"'
        + "a" * 64
        + '","schema":1} -->'
    )

    parsed = parse_review_marker(marker)
    assert parsed == ReviewMarker(1, decision, "a" * 64, "D_1")
    assert parsed.eligible is True


def test_schema_2_pass_is_eligible_but_legacy_pass_is_not_canonical():
    assert ReviewMarker(2, "Pass", "a" * 64, "D_1").eligible is True
    legacy_pass = (
        '<!-- rsi-proposal-review:{"decision":"Pass",'
        '"discussion_node_id":"D_1","proposal_sha256":"'
        + "a" * 64
        + '","schema":1} -->'
    )
    assert parse_review_marker(legacy_pass) is None


@pytest.mark.parametrize(
    "body",
    [
        "<!-- rsi-proposal-review:{not-json} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Accept\"} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Maybe\",\"discussion_node_id\":\"D_1\",\"proposal_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"schema\":1} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Accept\",\"discussion_node_id\":\"D_1\",\"proposal_sha256\":\"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\",\"schema\":1} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Accept\",\"discussion_node_id\":\"D_1\",\"proposal_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"schema\":2} -->",
    ],
)
def test_parse_rejects_malformed_or_noncanonical_marker(body):
    assert parse_review_marker(body) is None


def test_parse_rejects_duplicate_markers():
    marker = render_review_marker("Pass", "a" * 64, "D_1")
    assert parse_review_marker(f"{marker}\n{marker}") is None


def test_parse_rejects_expected_discussion_node_id_mismatch():
    marker = render_review_marker("Pass", "a" * 64, "D_1")
    assert parse_review_marker(marker, expected_discussion_node_id="D_2") is None


def test_parse_allows_expected_discussion_node_id_match():
    marker = render_review_marker("Pass", "a" * 64, "D_1")
    assert parse_review_marker(marker, expected_discussion_node_id="D_1") == ReviewMarker(
        2, "Pass", "a" * 64, "D_1"
    )
