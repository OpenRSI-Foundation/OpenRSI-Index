"""Aggregate task-quality contract; trajectories may differ across schedules."""

MAX_SUCCESS_RATE_DROP = 0.05


def success_count(traces):
    # Pinned ALFWorldGame emits +10 only on successful terminal transitions;
    # failures and ordinary/invalid steps all have non-positive rewards.
    return sum(any(turn["environment_done"] and turn["reward"] == 10.0
                   for turn in trace["turns"]) for trace in traces)


def quality_summary(rows):
    episodes = sum(row["baseline"]["episodes"] for row in rows)
    candidate_episodes = sum(row["candidate"]["episodes"] for row in rows)
    if not episodes or candidate_episodes != episodes:
        raise ValueError("quality_episode_coverage")
    baseline = sum(row["baseline"]["success_count"] for row in rows)
    candidate = sum(row["candidate"]["success_count"] for row in rows)
    # Integer comparison avoids floating-point ambiguity at the 5-point boundary.
    passed = 100 * (baseline - candidate) <= 5 * episodes
    return {"baseline_success_count": baseline, "candidate_success_count": candidate,
            "episodes_per_role": episodes, "baseline_success_rate": baseline / episodes,
            "candidate_success_rate": candidate / episodes,
            "success_rate_delta": (candidate - baseline) / episodes,
            "max_success_rate_drop": MAX_SUCCESS_RATE_DROP, "passed": passed,
            "interpretation": "observed aggregate quality threshold, not a statistical non-inferiority guarantee"}
