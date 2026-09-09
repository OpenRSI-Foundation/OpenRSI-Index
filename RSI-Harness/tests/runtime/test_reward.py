import json
import math

import pytest

from rsi_harness.runtime.reward import read_reward


def write_reward_json(log_dir, rewards):
    (log_dir / "reward.json").write_text(json.dumps(rewards))


def test_json_reward_takes_precedence_over_text_compatibility(tmp_path):
    write_reward_json(tmp_path, {"reward": 0.75, "latency": 12})
    (tmp_path / "reward.txt").write_text("0.25\n")

    result = read_reward(tmp_path, None)

    assert dict(result.rewards) == {"reward": 0.75, "latency": 12.0}
    assert result.score == 0.75
    assert result.error is None


@pytest.mark.parametrize("text", ["0.5", " 12 \n", "-3.25e-2"])
def test_numeric_text_reward_is_harbor_compatible(tmp_path, text):
    (tmp_path / "reward.txt").write_text(text)

    result = read_reward(tmp_path, None)

    assert result.score == pytest.approx(float(text))
    assert dict(result.rewards) == {"reward": pytest.approx(float(text))}
    assert result.error is None


@pytest.mark.parametrize(
    ("rewards", "requested", "score"),
    [
        ({"reward": 0.75, "latency": 12}, None, 0.75),
        ({"accuracy": 0.8}, None, 0.8),
        ({"accuracy": 0.8, "latency": 12}, None, None),
        ({"accuracy": 0.8, "latency": 12}, "accuracy", 0.8),
        ({"reward": 0.75, "accuracy": 0.8}, "missing", None),
    ],
)
def test_primary_reward_selection(tmp_path, rewards, requested, score):
    write_reward_json(tmp_path, rewards)

    result = read_reward(tmp_path, requested)

    assert dict(result.rewards) == {key: float(value) for key, value in rewards.items()}
    assert result.score == score
    assert result.error is None


@pytest.mark.parametrize(
    ("filename", "content", "error_fragment"),
    [
        (None, None, "missing"),
        ("reward.txt", "", "empty"),
        ("reward.txt", "not-a-number", "numeric"),
        ("reward.json", "", "empty"),
        ("reward.json", "{", "JSON"),
        ("reward.json", "[]", "object"),
        ("reward.json", "{}", "empty"),
        ("reward.json", '{"reward": null}', "number"),
    ],
)
def test_missing_or_invalid_reward_returns_an_explicit_error(
    tmp_path, filename, content, error_fragment
):
    if filename is not None:
        (tmp_path / filename).write_text(content)

    result = read_reward(tmp_path, None)

    assert dict(result.rewards) == {}
    assert result.score is None
    assert error_fragment.lower() in result.error.lower()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("filename", ["reward.json", "reward.txt"])
def test_non_finite_reward_is_invalid(tmp_path, filename, value):
    content = json.dumps({"reward": value}) if filename.endswith("json") else str(value)
    (tmp_path / filename).write_text(content)

    result = read_reward(tmp_path, None)

    assert dict(result.rewards) == {}
    assert result.score is None
    assert "finite" in result.error.lower()


def test_malformed_json_does_not_fall_back_to_reward_text(tmp_path):
    (tmp_path / "reward.json").write_text("{")
    (tmp_path / "reward.txt").write_text("1")

    result = read_reward(tmp_path, None)

    assert dict(result.rewards) == {}
    assert result.score is None
    assert "json" in result.error.lower()


def test_arbitrary_size_integer_overflow_is_invalid_reward(tmp_path):
    write_reward_json(tmp_path, {"reward": 10**400})

    result = read_reward(tmp_path, None)

    assert dict(result.rewards) == {}
    assert result.score is None
    assert "finite" in result.error.lower()
