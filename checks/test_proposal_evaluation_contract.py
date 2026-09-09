from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_AGENT = ROOT / ".agents" / "skills" / "proposal-agent"
SKILL = PROPOSAL_AGENT / "SKILL.md"
TEMPLATE = PROPOSAL_AGENT / "references" / "proposal-template.md"
RUBRIC = PROPOSAL_AGENT / "references" / "task-proposal-rubric.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"


def rubric_copies() -> dict[str, Path]:
    copies = {"public proposal-agent reference": RUBRIC}
    workspace = ROOT.parents[2] if ROOT.parent.name == ".worktrees" else ROOT.parent
    skills_root = workspace / "RSI-Skills"
    matching_worktree = skills_root / ".worktrees" / ROOT.name
    if matching_worktree.is_dir():
        skills_root = matching_worktree
    authoritative = skills_root / "rubrics" / "task-proposal.md"
    private_reference = (
        skills_root
        / ".agents"
        / "skills"
        / "proposal-agent"
        / "references"
        / "task-proposal-rubric.md"
    )
    if authoritative.is_file() and private_reference.is_file():
        copies["private authoritative rubric"] = authoritative
        copies["private proposal-agent reference"] = private_reference
    return copies


def test_proposal_agent_allows_justified_fixed_protocol_retraining():
    skill = SKILL.read_text(encoding="utf-8").lower()
    template = TEMPLATE.read_text(encoding="utf-8").lower()
    rubric = RUBRIC.read_text(encoding="utf-8").lower()
    normalized_rubric = " ".join(rubric.split())

    for source in (skill, rubric):
        body = " ".join(source.split())
        assert "prefer direct evaluation" in body
        assert "evaluation-time retraining under a fixed protocol" in body
        assert "why direct evaluation cannot answer" in body
        assert "evaluator-owned source code" in body
        assert "run budget" in body
        assert "metric capture" in body
        assert "matched baseline/candidate execution" in body
        assert "retry" in body or "retries" in body
        assert "candidate-failure" in body
        assert "fabricated outputs" in body

    assert "configuration or manifest" in template
    assert "retraining justification" in template
    assert "fixed evaluation execution contract" in template
    assert "matched baseline/candidate execution under the same evaluator-owned protocol" in normalized_rubric
    assert "except the declared candidate-versus-baseline treatment" in normalized_rubric
    assert "must not rerun the candidate's training recipe" not in rubric


def test_round_six_routes_unresolved_task_contract_decisions_before_rendering():
    skill = " ".join(SKILL.read_text(encoding="utf-8").lower().split())

    assert "simulate handing the complete proposal directly" in skill
    assert "harbor-task-agent" in skill
    for contract_area in (
        "research question",
        "baseline",
        "evaluation or reward",
        "action space",
        "data boundary",
        "network boundary",
        "compute contract",
    ):
        assert contract_area in skill
    assert "route the gap back to its owning round" in skill
    assert "must not render or write the proposal" in skill
    assert "without a separate assumption confirmation" in skill
    assert "repository paths" in skill
    assert "implementation details remain agent-owned" in skill


def test_existing_template_rows_close_the_default_task_generation_contract():
    skill = " ".join(SKILL.read_text(encoding="utf-8").lower().split())
    template = " ".join(TEMPLATE.read_text(encoding="utf-8").lower().split())

    generation_defaults = (
        "candidate-producing compute runs in work",
        "judge reloads and evaluates the complete materialized candidate snapshot",
        "candidate-only evaluation",
        "solution materializes the traceable baseline without training or evaluation",
        "candidate-invalid artifacts are unscored",
        "shared base/work/judge snapshot model",
    )
    for required_default in generation_defaults:
        assert required_default in skill
        assert required_default in template

    for required_default in (
        "web search separately from each external service",
        "additional-data access",
        "concrete existing delivery/access interface",
    ):
        assert required_default in template


def test_reference_rubric_rejects_failed_task_generation_readiness():
    rubric = " ".join(RUBRIC.read_text(encoding="utf-8").lower().split())

    assert "evaluate all nine gates before deciding" in rubric
    assert "### 9. task-generation readiness" in rubric
    assert "task-generation readiness | pass / fail" in rubric
    assert "a task-generation readiness failure forces `reject`" in rubric
    assert "one or more of the nine gates fail" in rubric


def test_all_available_rubric_copies_use_only_the_nine_gate_pass_reject_rule():
    gate_rows = (
        "Contributor Expertise Alignment",
        "Source Repository",
        "Model-Development AutoResearch Scope",
        "Traceable Baseline",
        "Scientific Objective and Metric",
        "Research Action Space",
        "Evaluation Integrity",
        "Data and Network Boundaries",
        "Task-Generation Readiness",
    )

    assert len(rubric_copies()) in {1, 3}
    for label, path in rubric_copies().items():
        source = path.read_text(encoding="utf-8")
        normalized = " ".join(source.split())
        for gate in gate_rows:
            assert f"| {gate} | Pass / Fail |" in source, label
        assert "## Layer 2:" not in source, label
        assert "Reject: one or more of the nine gates fail." in normalized, label
        assert (
            "Pass: all nine gates pass; compute flags remain non-blocking unless "
            "another gate fails."
        ) in normalized, label
        assert "Decision: Reject | Pass" in source, label
        assert "**Strong Accept:**" not in source, label
        assert "Decision: Reject | Accept" not in source, label
        assert "Decision: Accept" not in source, label


def test_contributing_preserves_automatic_build_and_gpu_trajectory_contract():
    contributing = CONTRIBUTING.read_text(encoding="utf-8")
    normalized = " ".join(contributing.lower().split())

    assert "about 1 hour" in normalized
    for stage in (
        "submit or edit the discussion",
        "pass or reject initial check",
        "task building starts automatically",
        "`/task <answer or correction>` only if asked",
        "accepted",
        "private task repository",
    ):
        assert stage in normalized
    assert "send `/task` once" not in normalized
    assert "post `/task`" not in normalized
    assert "`/task confirm`" not in contributing
    assert "legacy" not in normalized

    private_repo = contributing.split("### 4. Use the private task repository", 1)[1]
    assert "must have access" in private_repo.lower()
    assert "required gpu resources" in private_repo.lower()
    assert "proposal trajectory" in private_repo.lower()
    assert "discussion record" in private_repo.lower()
    assert "clone" in private_repo.lower()
    assert "harbor-task-validator" in private_repo
    assert "rsi-task-runner" in private_repo
    assert "rsi-harness" in private_repo.lower()
    assert "experiment trajectory" in private_repo.lower()
    assert "without gpu" not in private_repo.lower()
    assert "with gpu" not in private_repo.lower()
    assert "no-gpu contribution path" not in private_repo.lower()


def test_readme_requires_hook_review_and_binding_verification_for_both_clients():
    normalized = " ".join(CONTRIBUTING.read_text(encoding="utf-8").lower().split())

    assert ".codex/hooks.json" in normalized
    assert ".claude/settings.json" in normalized
    assert "review" in normalized
    assert "trust" in normalized
    assert "declin" in normalized or "disabl" in normalized
    assert "prevents trajectory submission" in normalized
    assert "after the first response" in normalized
    assert "proposal_session.py status" in normalized
