from pathlib import Path

SERVICE = (
    Path(__file__).parents[1]
    / "tools"
    / "rubric-review-service"
    / "src"
    / "index.ts"
)
JUDGE_RESULT = SERVICE.with_name("judge-result.ts")


def test_hosted_rubric_review_service_uses_three_decisions():
    service_source = SERVICE.read_text(encoding="utf-8")
    validation_source = JUDGE_RESULT.read_text(encoding="utf-8")
    source = service_source + validation_source

    assert 'decision: "Reject" | "Accept" | "Strong Accept";' in source
    assert 'enum: ["Reject", "Accept", "Strong Accept"]' in source
    assert "isJudgeDecision(result.decision)" in source
    assert 'responseBody.status !== "completed"' in service_source
    assert "parseJudgeResult(JSON.parse(text))" in service_source
    assert "require human review" not in source.casefold()
    assert "strong reject" not in source.casefold()
