"""CPU orchestration tests; only model/metric execution is replaced."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_calibrate_baseline as serial_fixtures
from test_parallel_encoder import TEST_GPUS
import evaluation_protocol as protocol


class MultiCalibrationTests(unittest.TestCase):
    def load(self):
        path = Path(__file__).with_name("calibrate_baseline_multi_gpu.py")
        self.assertTrue(path.is_file(), "multi-GPU coordinator is absent")
        spec = importlib.util.spec_from_file_location("multi_calibration_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, root, module):
        helper = serial_fixtures.CalibrationTests()
        evaluator, zero, logs, events = helper.fixture(root)
        evaluator.SUBMISSION = root / "adapter"
        (evaluator.SUBMISSION / "manifest.json").write_text("baseline fixture manifest")
        evaluator.SUBJECTS = ["biology", "pony", "theoremqa_theorems"]
        evaluator.BEIR_DATASETS = {"nfcorpus": "NFCorpus", "scifact": "SciFact", "fiqa": "FiQA-2018", "arguana": "ArguAna"}
        evaluator.sha256 = module.sha256
        evaluator.validate_adapter = lambda: (evaluator.SUBMISSION, True, module.sha256(evaluator.SUBMISSION / "adapter_model.safetensors"))
        class Pool:
            total_metric_encodes = 167188
            def __init__(self, gpus, binding, canonical=False):
                self.gpus = gpus
                self.stats = [{"gpu_uuid": gpu, "pid": 100+i, "peak_gpu_memory_mb": 1.0,
                               "metric_encodes": 167188 if i == 0 else 0, "load_seconds": 0.1}
                              for i, gpu in enumerate(gpus)]
            def __enter__(self):
                events.append("workers_start")
                return self
            def __exit__(self, *args):
                events.append("workers_stop")
            def consistency_check(self):
                events.append("consistency")
                return {"passed": True, "bitwise_equal": True, "worker_count": 4,
                        "text_count": 3, "protocol_count": 2, "reference_worker": 0}
            def collect_stats(self):
                return self.stats
        evaluator.evaluate_bright = lambda model, scratch: (events.append("bright") or {k: 0.25 for k in evaluator.SUBJECTS})
        evaluator.evaluate_beir = lambda model, scratch: (events.append("beir") or {k: 0.5 for k in evaluator.BEIR_DATASETS.values()})
        return evaluator, zero, logs, events, Pool

    def test_success_checks_assets_then_consistency_then_full_metrics_and_only_real_artifact(self):
        # Break caught: bypassing validators, fake inputs/reward, or incorrect aggregation.
        m = self.load()
        with tempfile.TemporaryDirectory() as directory:
            e, zero, logs, events, pool = self.fixture(Path(directory), m)
            with patch.object(m.parallel, "ZERO_ROOT", zero):
                result = m.measure(e, logs, TEST_GPUS, "sha256:" + "a"*64, pool)
            self.assertEqual((result["R"], result["G"]), ("0.25000", "0.50000"))
            self.assertEqual(set(result), {"R", "G", "provenance"})
            self.assertEqual([p.name for p in logs.iterdir()], ["baseline-calibration.json"])
            self.assertEqual(json.loads((logs / "baseline-calibration.json").read_text()), result)
            self.assertLess(events.index("workspace"), events.index("workers_start"))
            self.assertLess(events.index("consistency"), events.index("bright"))
            self.assertEqual(events[-3:], ["bright", "beir", "workers_stop"])
            self.assertEqual(result["provenance"]["total_metric_encodes"], 167188)
            self.assertEqual(result["provenance"]["protocol"], protocol.descriptor())

    def test_noncanonical_incomplete_nonfinite_inconsistent_or_failed_work_never_writes(self):
        # Break caught: promoting partial work or a different baseline to validated R/G.
        m = self.load()
        for case in ("nonbaseline", "bad_link", "missing_subject", "nan", "count", "consistency"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                e, zero, logs, events, pool = self.fixture(Path(directory), m)
                if case == "nonbaseline":
                    e.validate_adapter = lambda: (e.SUBMISSION, False, "digest")
                elif case == "bad_link":
                    path = e.SUBMISSION / "adapter_config.json"
                    path.unlink()
                    path.write_text("wrong")
                elif case == "missing_subject":
                    e.evaluate_bright = lambda *args: {"s0": 0.25}
                elif case == "nan":
                    e.evaluate_bright = lambda *args: {k: float("nan") for k in e.SUBJECTS}
                elif case == "count":
                    pool.total_metric_encodes = 1
                else:
                    pool.consistency_check = lambda self: {"passed": False}
                with patch.object(m.parallel, "ZERO_ROOT", zero), self.assertRaises(RuntimeError):
                    m.measure(e, logs, TEST_GPUS, "sha256:" + "a"*64, pool)
                self.assertEqual(list(logs.iterdir()), [])

    def test_existing_artifact_is_not_overwritten(self):
        # Break caught: destroying earlier evidence on retries.
        m = self.load()
        with tempfile.TemporaryDirectory() as directory:
            e, zero, logs, events, pool = self.fixture(Path(directory), m)
            (logs / "marker").write_text("retained")
            with self.assertRaises(RuntimeError):
                m.measure(e, logs, TEST_GPUS, "sha256:" + "a"*64, pool)
            self.assertEqual(events, [])

    def test_invalid_worker_resource_statistics_never_emit_artifact(self):
        # Break caught: untraceable PID or invalid counts/resources passing provenance.
        m = self.load()
        for field, value in (("pid", 0), ("pid", 1.5), ("metric_encodes", 167188.0),
                             ("peak_gpu_memory_mb", float("nan"))):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                e, zero, logs, events, pool = self.fixture(Path(directory), m)
                def collect(self):
                    self.stats[0][field] = value
                    return self.stats
                pool.collect_stats = collect
                with patch.object(m.parallel, "ZERO_ROOT", zero), self.assertRaises(RuntimeError):
                    m.measure(e, logs, TEST_GPUS, "sha256:" + "a"*64, pool)
                self.assertEqual(list(logs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
