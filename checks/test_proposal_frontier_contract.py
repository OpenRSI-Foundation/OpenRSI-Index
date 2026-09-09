from pathlib import Path

ROOT = Path(__file__).parent.parent
WORKER = ROOT / "tools/rubric-review-service/src/index.ts"
REVIEW_DOC = ROOT / "docs/TASK_PROPOSAL_RUBRIC_REVIEW.md"
SERVICE_README = ROOT / "tools/rubric-review-service/README.md"


def test_worker_enables_high_context_web_search_and_supplies_review_date():
    worker = WORKER.read_text()

    assert 'type: "web_search"' in worker
    assert 'search_context_size: "high"' in worker
    assert 'tool_choice: "required"' in worker
    assert "review_date_utc" in worker


def test_review_docs_disclose_web_search_and_required_identity_field():
    for path in (REVIEW_DOC, SERVICE_README):
        documentation = path.read_text().lower()
        assert "web search" in documentation
        assert "contributor full name" in documentation


def test_review_docs_name_skills_as_the_private_source_of_truth():
    documentation = REVIEW_DOC.read_text(encoding="utf-8")

    assert "RSI-Index/RSI-Skills" in documentation
    assert "RUBRIC_REPO_TOKEN" not in documentation
    assert "Zhuofeng-Li/RSI-Index-Rubrics" not in documentation
