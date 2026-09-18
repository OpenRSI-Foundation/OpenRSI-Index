"""Real judge orchestration with expensive models/metrics replaced only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import evaluate
import parallel_encoder as parallel
import test_calibrate_baseline_multi_gpu as fixtures
from test_parallel_encoder import TEST_GPUS


class JudgeTests(unittest.TestCase):
    def test_candidate_failure_is_actionable_and_redacts_untrusted_values(self):
        # Break: removing task-owned validation codes/hints or leaking candidate fields.
        failure = evaluate.CandidateInvalid("manifest", Path("/workspace/submission/manifest.json"),
                    "manifest missing or too large", "regular JSON <=65536 bytes", "PRIVATE_CANDIDATE_CONTENT", "run the fixed selector or baseline Solution")
        with patch.object(evaluate.parallel, "discover_gpu_uuids", return_value=TEST_GPUS), \
             patch.object(evaluate, "parse_baseline", return_value=.1), \
             patch.object(evaluate, "judge", side_effect=failure), \
             patch.dict("os.environ", {}), contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised: evaluate.main()
        self.assertEqual(raised.exception.code, 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result.get("code"), "manifest")
        self.assertEqual(result.get("hint"), "run the fixed selector or baseline Solution")
        self.assertNotIn("PRIVATE_CANDIDATE_CONTENT", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_nonzero_candidate_is_scored_only_after_coverage_and_worker_cleanup(self):
        # Break: canonical-zero-only path in Judge, fifth model, premature reward.
        self.assertTrue(hasattr(evaluate, "judge"), "parallel candidate judge missing")
        import calibrate_baseline_multi_gpu as calibration
        with tempfile.TemporaryDirectory() as directory:
            e, zero, logs, events, pool = fixtures.MultiCalibrationTests().fixture(Path(directory), calibration)
            e.validate_adapter = lambda: (e.SUBMISSION, False, calibration.sha256(e.SUBMISSION / "adapter_model.safetensors"))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = evaluate.judge(e, logs, TEST_GPUS, .1, .4, pool)
            self.assertFalse(summary["artifact"]["baseline"])
            self.assertEqual(summary["reward"], .25)
            self.assertEqual(summary["total_metric_encodes"], 167188)
            self.assertEqual(len(summary["worker_stats"]), 4)
            self.assertEqual(json.loads((logs / "reward.json").read_text()), {"reward": .25})
            self.assertNotIn("load_base", events)
            self.assertLess(events.index("workspace"), events.index("workers_start"))
            self.assertEqual(events[-1], "workers_stop")

    def test_failed_or_partial_workers_never_publish_reward(self):
        # Break: publish scores despite missing subset, duplicate PID, worker errors.
        self.assertTrue(hasattr(evaluate, "judge"), "parallel candidate judge missing")
        import calibrate_baseline_multi_gpu as calibration
        for case in ("count", "keys", "pid", "worker_error", "binding"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                e, zero, logs, events, pool = fixtures.MultiCalibrationTests().fixture(Path(directory), calibration)
                if case == "count": pool.total_metric_encodes = 1
                elif case == "keys": e.evaluate_bright = lambda *args: {"biology": .2}
                elif case == "pid":
                    collect = pool.collect_stats
                    def duplicate(self):
                        stats = collect(self)
                        stats[1]["pid"] = stats[0]["pid"]
                        return stats
                    pool.collect_stats = duplicate
                elif case == "binding":
                    original = e.evaluate_beir
                    def change(*args):
                        (e.SUBMISSION / "manifest.json").write_text("changed")
                        return original(*args)
                    e.evaluate_beir = change
                else:
                    def fail(*args): raise RuntimeError("private model error")
                    e.evaluate_beir = fail
                with self.assertRaises(RuntimeError): evaluate.judge(e, logs, TEST_GPUS, .1, .4, pool)
                self.assertEqual(list(logs.iterdir()), [])


if __name__ == "__main__": unittest.main()
