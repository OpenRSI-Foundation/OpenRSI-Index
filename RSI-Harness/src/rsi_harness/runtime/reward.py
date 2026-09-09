"""Read standard Harbor verifier rewards without inventing scores."""

from __future__ import annotations

import json
import math
from pathlib import Path

from harbor import VerifierResult
from pydantic import ValidationError

from rsi_harness.models import RewardResult


def read_reward(log_dir: Path, primary_key: str | None) -> RewardResult:
    """Read ``reward.json`` first, then Harbor's scalar text compatibility file."""
    root = Path(log_dir)
    json_path = root / "reward.json"
    text_path = root / "reward.txt"
    try:
        if json_path.exists():
            raw_text = json_path.read_text()
            if not raw_text.strip():
                return _invalid("reward.json is empty")
            try:
                raw_rewards = json.loads(raw_text)
            except json.JSONDecodeError as error:
                return _invalid(f"reward.json contains invalid JSON: {error.msg}")
            if not isinstance(raw_rewards, dict):
                return _invalid("reward.json must contain an object")
        elif text_path.exists():
            raw_text = text_path.read_text()
            if not raw_text.strip():
                return _invalid("reward.txt is empty")
            try:
                raw_rewards = {"reward": float(raw_text.strip())}
            except ValueError:
                return _invalid("reward.txt must contain one numeric value")
        else:
            return _invalid("Harbor reward file is missing")
    except OSError as error:
        return _invalid(f"Harbor reward file could not be read: {error}")

    if not raw_rewards:
        return _invalid("Harbor reward dictionary is empty")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_rewards.values()
    ):
        return _invalid("every Harbor reward must be a number")
    try:
        verified = VerifierResult.model_validate({"rewards": raw_rewards})
    except ValidationError as error:
        return _invalid(f"Harbor reward is invalid: {error.errors(include_url=False)}")
    if not verified.rewards:
        return _invalid("Harbor reward dictionary is empty")

    try:
        rewards = {
            name: float(value) for name, value in verified.rewards.items()
        }
    except (OverflowError, ValueError):
        return _invalid("every Harbor reward must be finite")
    if any(not math.isfinite(value) for value in rewards.values()):
        return _invalid("every Harbor reward must be finite")

    score: float | None = None
    if primary_key is not None:
        score = rewards.get(primary_key)
    elif "reward" in rewards:
        score = rewards["reward"]
    elif len(rewards) == 1:
        score = next(iter(rewards.values()))
    return RewardResult(rewards=rewards, score=score)


def _invalid(message: str) -> RewardResult:
    return RewardResult(error=message)


__all__ = ["read_reward"]
