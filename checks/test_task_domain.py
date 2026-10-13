"""Task Ideas Discussion domain labels: row parsing, the label plan, and sync."""

from __future__ import annotations

import json

import pytest

from checks import backfill_discussion_domain_labels as backfill
from checks import task_domain
from checks.discussion_recovery import GitHubClient
from checks.task_domain import (
    DOMAIN_LABELS,
    GENERAL,
    MANAGED_LABELS,
    SCIENCE,
    MissingLabelError,
    GENERAL_PIPELINE,
    SCIENCE_PIPELINE,
    domain_label,
    parse_domain,
    parse_training,
    pipeline,
    plan,
    sync,
)

TABLE = """# Title

| Section | Field | Proposal |
| --- | --- | --- |
| Task Type | Domain | {domain} |
| Contributor | Full name | Ada |
"""


@pytest.mark.parametrize(
    ("row", "label"),
    [
        ("| Task Type | Domain | AI4AI |", GENERAL),
        ("| Task Type | Domain | AI4Sci |", SCIENCE),
        ("|task type|domain|ai4sci|", SCIENCE),
        ("  |  TASK   TYPE |  DOMAIN  |   AI4SCI   |  ", SCIENCE),
        ("| Task Type | Domain | **AI4Sci** |", SCIENCE),
        ("| Task Type | Domain | AI4Sci |\r", SCIENCE),
        ("| Task Type | Domain | Robotics |", GENERAL),
        ("| Task Type | Domain |  |", GENERAL),
        ("| Task Type | Domain | AI4Sci (materials) |", GENERAL),
        ("Task Type | Domain | AI4Sci", GENERAL),
        ("| Task Type | Training | AI4Sci |", GENERAL),
    ],
)
def test_domain_row_decides_the_label(row, label):
    assert domain_label(f"intro\n{row}\nmore") == label


def test_missing_body_or_row_is_a_general_task():
    assert domain_label(None) == GENERAL
    assert domain_label("") == GENERAL
    assert domain_label("no table here") == GENERAL
    assert parse_domain(TABLE.format(domain="AI4Sci")) == "AI4Sci"
    assert parse_domain("| Task Type | Domain | |") is None


def test_row_does_not_span_lines_or_extra_cells():
    assert domain_label("| Task Type | Domain |\nAI4Sci |") == GENERAL
    assert domain_label("| Task Type | Domain | AI4Sci | extra |") == GENERAL


def test_first_domain_row_wins():
    body = TABLE.format(domain="AI4Sci") + "| Task Type | Domain | AI4AI |\n"
    assert domain_label(body) == SCIENCE


INTAKE_ROWS = """| Section | Field | Proposal |
| --- | --- | --- |
| Task Type | Domain | {domain} |
| Task Type | Involves model training | {training} |
| Contributor | Full name | Ada |
"""


@pytest.mark.parametrize(
    ("row", "answer"),
    [
        ("| Task Type | Involves model training | Yes |", "Yes"),
        ("| Task Type | Involves model training | No |", "No"),
        ("|task type|involves model training|no|", "no"),
        ("  | TASK  TYPE | Involves   Model  Training |  **No**  |", "No"),
        ("| Task Type | Involves model training |  |", None),
        ("| Task Type | Involves training | No |", None),
        ("| Task Type | Involves model training | No | extra |", None),
        ("| Task Type | Domain | No |", None),
    ],
)
def test_training_row_is_parsed_like_the_domain_row(row, answer):
    assert parse_training(f"x\n{row}\ny") == answer


@pytest.mark.parametrize(
    ("domain", "training", "expected"),
    [
        ("AI4AI", "Yes", GENERAL_PIPELINE),
        ("AI4AI", "No", GENERAL_PIPELINE),
        ("AI4Sci", "Yes", GENERAL_PIPELINE),
        ("AI4Sci", "No", SCIENCE_PIPELINE),
        ("ai4sci", "no", SCIENCE_PIPELINE),
        ("AI4Sci", "Maybe", GENERAL_PIPELINE),
        ("Robotics", "No", GENERAL_PIPELINE),
    ],
)
def test_training_answer_chooses_the_pipeline(domain, training, expected):
    body = INTAKE_ROWS.format(domain=domain, training=training)
    assert pipeline(body) == expected
    # The training answer never changes the label.
    assert domain_label(body) == domain_label(TABLE.format(domain=domain))


def test_missing_training_row_keeps_the_general_pipeline():
    assert pipeline(TABLE.format(domain="AI4Sci")) == GENERAL_PIPELINE
    assert pipeline("| Task Type | Involves model training | No |") == GENERAL_PIPELINE
    assert pipeline(None) == GENERAL_PIPELINE
    assert pipeline("") == GENERAL_PIPELINE


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (INTAKE_ROWS.format(domain="AI4Sci", training="No"), "science"),
        (INTAKE_ROWS.format(domain="AI4Sci", training="Yes"), "general"),
        (None, "general"),
        (7, "general"),
    ],
)
def test_pipeline_command_reads_the_event_and_writes_the_step_output(
    tmp_path, monkeypatch, capsys, body, expected
):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"discussion": {"body": body}}), encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert task_domain.main(["pipeline"]) == 0
    assert capsys.readouterr().out == f"{expected}\n"
    assert output.read_text(encoding="utf-8") == f"pipeline={expected}\n"


def test_mapping_table_and_managed_labels():
    assert DOMAIN_LABELS == {"ai4ai": "general task", "ai4sci": "science task"}
    assert set(MANAGED_LABELS) == set(DOMAIN_LABELS.values())
    assert task_domain.DEFAULT_LABEL == "general task"


def test_plan_adds_the_domain_label_and_removes_the_other_only():
    body = TABLE.format(domain="AI4Sci")
    assert plan(body, ["general task", "bug"]) == (SCIENCE, [SCIENCE], [GENERAL])
    assert plan(body, ["science task"]) == (SCIENCE, [], [])
    assert plan("", []) == (GENERAL, [GENERAL], [])


class FakeGraphQL(GitHubClient):
    def __init__(self, nodes):
        super().__init__(sleep=lambda _: None, clock=lambda: 0.0)
        self.nodes, self.mutations = nodes, []

    def graphql(self, query, **fields):
        if "addLabelsToLabelable" in query or "removeLabelsFromLabelable" in query:
            kind = "add" if "addLabelsToLabelable" in query else "remove"
            self.mutations.append((kind, fields))
            return {}
        if "label(name:" in query:
            return {
                "repository": {
                    f"l{i}": ({"id": f"L{i}", "name": n} if n in self.nodes["labels"] else None)
                    for i, n in enumerate(MANAGED_LABELS)
                }
            }
        return {"node": self.nodes["discussion"]}


def discussion(body, labels=(), category="Task Ideas"):
    return {
        "id": "D_1",
        "number": 9,
        "title": "T",
        "body": body,
        "category": {"name": category},
        "repository": {"owner": {"login": "o"}, "name": "r"},
        "labels": {"nodes": [{"id": f"X{i}", "name": n} for i, n in enumerate(labels)]},
    }


def test_apply_labels_the_discussion_and_removes_the_other_domain():
    client = FakeGraphQL(
        {
            "labels": MANAGED_LABELS,
            "discussion": discussion(TABLE.format(domain="AI4Sci"), ["general task", "bug"]),
        }
    )
    task_domain.apply_one(client, "D_1")
    assert client.mutations == [
        ("add", {"labelable": "D_1", "label": "L1"}),
        ("remove", {"labelable": "D_1", "label": "X0"}),
    ]


def test_apply_ignores_other_categories():
    client = FakeGraphQL(
        {"labels": MANAGED_LABELS, "discussion": discussion("", category="General")}
    )
    task_domain.apply_one(client, "D_1")
    assert client.mutations == []


def test_missing_repository_label_fails_clearly_without_changes():
    client = FakeGraphQL(
        {"labels": [GENERAL], "discussion": discussion(TABLE.format(domain="AI4Sci"), ["general task"])}
    )
    with pytest.raises(MissingLabelError, match="'science task' does not exist"):
        task_domain.apply_one(client, "D_1")
    assert client.mutations == []


def test_dry_run_sync_never_mutates():
    client = FakeGraphQL({"labels": (), "discussion": None})
    result = sync(client, discussion(""), {}, apply=False)
    assert result == (GENERAL, [GENERAL], [])
    assert client.mutations == []


def test_graphql_errors_fail_the_step(monkeypatch, capsys):
    def failing(args):
        return json.dumps({"errors": [{"type": "FORBIDDEN", "message": "no"}]})

    monkeypatch.setattr(task_domain, "GitHubClient", lambda: GitHubClient(call=failing))
    assert task_domain.main(["apply", "--discussion-node-id", "D_1"]) == 1
    assert "::error::" in capsys.readouterr().out


def test_missing_label_exits_nonzero(monkeypatch, capsys):
    client = FakeGraphQL({"labels": [], "discussion": discussion("")})
    monkeypatch.setattr(task_domain, "GitHubClient", lambda: client)
    assert task_domain.main(["apply", "--discussion-node-id", "D_1"]) == 1
    assert "'general task' does not exist" in capsys.readouterr().out


class FakeBackfill(GitHubClient):
    def __init__(self, labels, pages):
        super().__init__(sleep=lambda _: None, clock=lambda: 0.0)
        self.labels, self.pages, self.mutations = labels, pages, []

    def graphql(self, query, **fields):
        if "Labelable" in query:
            self.mutations.append(fields)
            return {}
        if "label(name:" in query:
            return {
                "repository": {
                    f"l{i}": ({"id": f"L{i}", "name": n} if n in self.labels else None)
                    for i, n in enumerate(MANAGED_LABELS)
                }
            }
        if "discussionCategories" in query:
            return {"repository": {"discussionCategories": {"nodes": [
                {"id": "C0", "name": "General"}, {"id": "C1", "name": "Task Ideas"},
            ]}}}
        assert fields["category"] == "C1"
        index = int(fields.get("cursor") or 0)
        return {"repository": {"discussions": {
            "nodes": self.pages[index],
            "pageInfo": {"hasNextPage": index + 1 < len(self.pages), "endCursor": str(index + 1)},
        }}}


def backfill_pages():
    return [
        [discussion("old proposal") | {"id": "D_1", "number": 1}],
        [
            discussion(TABLE.format(domain="AI4Sci"), ["general task"]) | {"id": "D_2", "number": 2},
            discussion("", ["general task"]) | {"id": "D_3", "number": 3},
        ],
    ]


def test_backfill_is_a_dry_run_by_default(monkeypatch, capsys):
    client = FakeBackfill(MANAGED_LABELS, backfill_pages())
    monkeypatch.setattr(backfill, "GitHubClient", lambda: client)
    assert backfill.main([]) == 0
    assert client.mutations == []
    out = capsys.readouterr().out
    assert "3 Task Ideas Discussions; would change 2." in out


def test_backfill_apply_labels_every_page(monkeypatch):
    client = FakeBackfill(MANAGED_LABELS, backfill_pages())
    monkeypatch.setattr(backfill, "GitHubClient", lambda: client)
    assert backfill.main(["--apply"]) == 0
    assert client.mutations == [
        {"labelable": "D_1", "label": "L0"},
        {"labelable": "D_2", "label": "L1"},
        {"labelable": "D_2", "label": "X0"},
    ]


def test_backfill_apply_refuses_before_labels_exist(monkeypatch):
    client = FakeBackfill([GENERAL], backfill_pages())
    monkeypatch.setattr(backfill, "GitHubClient", lambda: client)
    assert backfill.main(["--apply"]) == 1
    assert client.mutations == []

