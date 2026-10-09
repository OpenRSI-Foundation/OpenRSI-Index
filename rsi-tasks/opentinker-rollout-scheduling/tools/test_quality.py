"""Boundary checks for the task-level quality acceptance rule."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from quality import quality_summary, success_count


class QualityTests(unittest.TestCase):
    def summary(self, baseline, candidate, episodes=384):
        return quality_summary([{"baseline": {"episodes": episodes, "success_count": baseline},
                                 "candidate": {"episodes": episodes, "success_count": candidate}}])

    def test_success_requires_terminal_success_reward(self):
        traces = [{"turns": [{"environment_done": done, "reward": reward}]}
                  for done, reward in [(True, 10.0), (False, 10.0), (True, -1), (False, -0.01)]]
        self.assertEqual(success_count(traces), 1)

    def test_full_protocol_boundary(self):
        self.assertTrue(self.summary(100, 81)["passed"])
        self.assertFalse(self.summary(100, 80)["passed"])
        self.assertTrue(self.summary(100, 120)["passed"])

    def test_exact_five_point_boundary(self):
        self.assertTrue(self.summary(80, 75, 100)["passed"])
        self.assertFalse(self.summary(80, 74, 100)["passed"])

    def test_all_rows_are_aggregated(self):
        rows = [{"baseline": {"episodes": 96, "success_count": 30},
                 "candidate": {"episodes": 96, "success_count": x}} for x in (30, 30, 30, 10)]
        self.assertFalse(quality_summary(rows)["passed"])

    def test_empty_or_unequal_work_is_rejected(self):
        with self.assertRaises(ValueError):
            quality_summary([])
        with self.assertRaises(ValueError):
            quality_summary([{"baseline": {"episodes": 96}, "candidate": {"episodes": 95}}])


if __name__ == "__main__":
    unittest.main()
