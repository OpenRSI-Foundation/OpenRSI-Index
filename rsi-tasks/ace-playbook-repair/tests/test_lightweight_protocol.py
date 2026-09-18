from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


TASK = Path(__file__).resolve().parents[1]
ENVIRONMENT = TASK / "environment"
TESTS = TASK / "tests"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def environment_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ENVIRONMENT))
    return {
        "materializer": load_module("d32_materializer", ENVIRONMENT / "materialize_noop.py"),
        "selfcheck": load_module("d32_selfcheck", ENVIRONMENT / "candidate_selfcheck.py"),
        "runner": load_module("d32_runner", ENVIRONMENT / "local_ace_runner.py"),
    }


@pytest.fixture
def gate():
    return load_module("d32_gate", TESTS / "candidate_gate.py")


class Tokenizer:
    def encode(self, text):
        return text.split()


class Encoder:
    def encode(self, text):
        return text.split()


def write_complete_artifacts(root: Path, *, seed: int = 0):
    destination = root / "runs/repeat-1"
    destination.mkdir(parents=True)
    empty = "\n\n".join(f"## {heading}" for heading in (
        "STRATEGIES & INSIGHTS",
        "FORMULAS & CALCULATIONS",
        "CODE SNIPPETS & TEMPLATES",
        "COMMON MISTAKES TO AVOID",
        "PROBLEM-SOLVING HEURISTICS",
        "CONTEXT CLUES & INDICATORS",
        "OTHERS",
    )) + "\n"
    (destination / "best_playbook.txt").write_text(empty, encoding="utf-8")
    audit = {
        "fixed_checkpoints": [50, 100, 150, 200],
        "inspection_events": [],
        "playbook_token_budget": 80000,
        "schema_version": 1,
        "seed": seed,
    }
    (destination / "audit.json").write_text(json.dumps(audit) + "\n", encoding="utf-8")
    rows = [{
        "schema_version": 1,
        "record_type": "run",
        "seed": seed,
        "status": "complete",
        "examples_total": 200,
        "checkpoints": [50, 100, 150, 200],
    }]
    rows.extend({
        "schema_version": 1,
        "record_type": "example",
        "seed": seed,
        "example_index": index,
        "status": "processed",
    } for index in range(200))
    rows.extend({
        "schema_version": 1,
        "record_type": "checkpoint",
        "seed": seed,
        "checkpoint": checkpoint,
        "status": "complete",
    } for checkpoint in (50, 100, 150, 200))
    (destination / "adaptation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return destination


def judge_errors(gate, workspace: Path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", workspace)
    monkeypatch.setattr(gate, "expected_snapshot", lambda: {})
    monkeypatch.setattr(gate, "snapshot", lambda _path: iter(()))
    return gate.validate(Tokenizer())[0]


def public_errors(selfcheck, submission: Path):
    return selfcheck.validate(submission, Tokenizer(), Encoder())[0]


def test_materializer_creates_one_seed_zero_noop_accepted_by_both_gates(
    tmp_path, monkeypatch, environment_modules, gate
):
    submission = tmp_path / "workspace/submission"
    materializer = environment_modules["materializer"]
    selfcheck = environment_modules["selfcheck"]
    monkeypatch.setattr(materializer, "ROOT", submission)

    materializer.main()

    assert [path.name for path in (submission / "runs").iterdir()] == ["repeat-1"]
    assert public_errors(selfcheck, submission) == []
    assert judge_errors(gate, tmp_path / "workspace", monkeypatch) == []


def test_complete_205_row_artifacts_are_accepted_by_both_gates(
    tmp_path, monkeypatch, environment_modules, gate
):
    submission = tmp_path / "workspace/submission"
    write_complete_artifacts(submission)

    assert public_errors(environment_modules["selfcheck"], submission) == []
    assert judge_errors(gate, tmp_path / "workspace", monkeypatch) == []


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_both_gates_reject_missing_or_extra_repeat_directories(
    tmp_path, monkeypatch, environment_modules, gate, mutation
):
    submission = tmp_path / "workspace/submission"
    write_complete_artifacts(submission)
    if mutation == "missing":
        (submission / "runs/repeat-1").rename(submission / "runs/repeat-2")
    else:
        (submission / "runs/repeat-2").mkdir()

    assert public_errors(environment_modules["selfcheck"], submission)
    assert judge_errors(gate, tmp_path / "workspace", monkeypatch)


@pytest.mark.parametrize(
    "mutation",
    ["old_full_log", "wrong_seed", "duplicate_example", "missing_checkpoint", "duplicate_checkpoint", "event_outside_budget"],
)
def test_both_gates_reject_invalid_coverage_identity_and_checkpoint_events(
    tmp_path, monkeypatch, environment_modules, gate, mutation
):
    submission = tmp_path / "workspace/submission"
    destination = write_complete_artifacts(submission)
    log = destination / "adaptation.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if mutation == "old_full_log":
        rows = [{
            "schema_version": 1, "record_type": "run", "seed": 0,
            "status": "complete", "examples_total": 500,
            "checkpoints": [100, 200, 300, 400, 500],
        }]
        rows.extend({
            "schema_version": 1, "record_type": "example", "seed": 0,
            "example_index": index, "status": "processed",
        } for index in range(500))
        rows.extend({
            "schema_version": 1, "record_type": "checkpoint", "seed": 0,
            "checkpoint": checkpoint, "status": "complete",
        } for checkpoint in (100, 200, 300, 400, 500))
    elif mutation == "wrong_seed":
        rows[40]["seed"] = 1
    elif mutation == "duplicate_example":
        rows[100] = dict(rows[99])
    elif mutation == "missing_checkpoint":
        rows.pop()
    elif mutation == "duplicate_checkpoint":
        rows[-1] = dict(rows[-2])
    else:
        audit_path = destination / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["inspection_events"] = [{"checkpoint": 250, "findings": [], "repairs": []}]
        audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert public_errors(environment_modules["selfcheck"], submission)
    assert judge_errors(gate, tmp_path / "workspace", monkeypatch)


def install_fake_data_processor(monkeypatch, process):
    package_eval = types.ModuleType("eval")
    package_finance = types.ModuleType("eval.finance")
    module = types.ModuleType("eval.finance.data_processor")

    class DataProcessor:
        def __init__(self, task):
            assert task == "formula"

        def process_task_data(self, rows):
            return process(rows)

    module.DataProcessor = DataProcessor
    monkeypatch.setitem(sys.modules, "eval", package_eval)
    monkeypatch.setitem(sys.modules, "eval.finance", package_finance)
    monkeypatch.setitem(sys.modules, "eval.finance.data_processor", module)
    return DataProcessor


def test_public_data_selection_is_ordered_and_fails_closed_on_source_size(
    tmp_path, monkeypatch, environment_modules
):
    runner = environment_modules["runner"]
    data = tmp_path / "eval/finance/data"
    data.mkdir(parents=True)
    train = [{"ordinal": index} for index in range(500)]
    val = [{"ordinal": index} for index in range(300)]
    (data / "formula_train_subset_500.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train), encoding="utf-8"
    )
    (data / "formula_val_subset_300.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in val), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "WORKSPACE", tmp_path)
    install_fake_data_processor(monkeypatch, lambda rows: rows)

    assert [row["ordinal"] for row in runner.load_public("train")] == list(range(200))
    assert [row["ordinal"] for row in runner.load_public("val")] == list(range(100))

    (data / "formula_train_subset_500.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train[:-1]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="500"):
        runner.load_public("train")


def test_runner_passes_lightweight_budget_to_ace_and_logs_complete_coverage(
    tmp_path, monkeypatch, environment_modules
):
    runner = environment_modules["runner"]
    captured = {}
    monkeypatch.setattr(runner, "WORKSPACE", tmp_path)
    monkeypatch.setattr(runner, "load_public", lambda split: list(range(200 if split == "train" else 100)))

    class FakeACE:
        def __init__(self, *args, **kwargs):
            captured["init"] = (args, kwargs)

        def run(self, **kwargs):
            captured["run"] = kwargs
            target = Path(kwargs["config"]["save_dir"]) / "result/best_playbook.txt"
            target.parent.mkdir(parents=True)
            target.write_text("playbook", encoding="utf-8")

    fake_ace = types.ModuleType("ace")
    fake_ace.ACE = FakeACE
    fake_utils = types.ModuleType("utils")
    fake_utils.initialize_clients = None
    monkeypatch.setitem(sys.modules, "ace", fake_ace)
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    install_fake_data_processor(monkeypatch, lambda rows: rows)
    runner.openai = types.SimpleNamespace(OpenAI=lambda **_kwargs: object())
    destination = tmp_path / "submission/runs/repeat-1"
    destination.mkdir(parents=True)

    runner.run_repeat(0, destination)

    run = captured["run"]
    assert len(run["train_samples"]) == 200
    assert len(run["val_samples"]) == 100
    assert run["config"] == {
        "num_epochs": 1,
        "max_num_rounds": 3,
        "curator_frequency": 1,
        "eval_steps": 50,
        "save_steps": 50,
        "playbook_token_budget": 80000,
        "task_name": "formula",
        "mode": "offline",
        "json_mode": False,
        "no_ground_truth": False,
        "save_dir": str(destination / "ace-output"),
        "test_workers": 1,
        "use_bulletpoint_analyzer": False,
        "batch_size": 1,
    }
    audit = json.loads((destination / "audit.json").read_text(encoding="utf-8"))
    assert audit["seed"] == 0
    assert audit["fixed_checkpoints"] == [50, 100, 150, 200]
    assert audit["playbook_token_budget"] == 80000
    rows = (destination / "adaptation.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 205


def load_evaluator(monkeypatch):
    original = importlib.util.spec_from_file_location

    def remap(name, location, *args, **kwargs):
        path = Path(location)
        if path.parent == Path("/tests"):
            path = TESTS / path.name
        return original(name, path, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", remap)
    return load_module("d32_evaluate", TESTS / "evaluate.py")


def test_evaluator_scores_one_playbook_over_all_hidden_examples_with_null_std(monkeypatch):
    evaluator = load_evaluator(monkeypatch)
    playbook = "\n\n".join(f"## {heading}" for heading in evaluator.gate.SECTIONS) + "\n"
    examples = [{"context": f"Question {index}", "target": "1"} for index in range(200)]
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        if len(prompts) <= 150:
            return '{"final_answer":"1"}'
        if len(prompts) <= 190:
            return '{"final_answer":"2"}'
        return "{}"

    feedback = evaluator.evaluate_playbooks(
        [(1, playbook, 7, 9, [])], examples, fake_generate
    )

    assert len(prompts) == 200
    assert feedback["mean_accuracy"] == 0.75
    assert feedback["sample_standard_deviation"] is None
    assert len(feedback["repeats"]) == 1
    assert feedback["repeats"][0]["accuracy"] == 0.75
    assert feedback["total_unparsable_answers"] == 10
