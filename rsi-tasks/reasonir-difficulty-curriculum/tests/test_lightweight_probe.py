"""Probe the real batching/numerical orchestration with only CUDA/model replaced."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np


class ProbeTests(unittest.TestCase):
    def test_probe_artifact_requires_consistency_and_no_metric_encodes(self):
        # Break: publishing synthetic artifact after failed worker consistency.
        probe = importlib.import_module("probe_batching")
        import test_calibrate_baseline_multi_gpu as fixtures
        import calibrate_baseline_multi_gpu as calibration
        from test_parallel_encoder import TEST_GPUS
        from unittest.mock import patch
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                e, zero, logs, events, pool = fixtures.MultiCalibrationTests().fixture(Path(directory), calibration)
                pool.total_metric_encodes = 0
                original_stats = pool.collect_stats
                def collect(self):
                    stats = original_stats(self)
                    for stat in stats: stat["metric_encodes"] = 0
                    return stats
                pool.collect_stats = collect
                pool.probe = lambda self: {"synthetic_only": True, "passed": True}
                if not valid: pool.consistency_check = lambda self: {"passed": False}
                with patch.object(probe.parallel, "ZERO_ROOT", zero):
                    if valid:
                        result = probe.measure(e, logs, TEST_GPUS, pool)
                        self.assertTrue(result["synthetic_only"])
                        self.assertEqual([p.name for p in logs.iterdir()], ["batch-probe.json"])
                    else:
                        with self.assertRaises(RuntimeError): probe.measure(e, logs, TEST_GPUS, pool)
                        self.assertEqual(list(logs.iterdir()), [])

    def test_probe_measures_real_policy_calls_and_aggregates_only(self):
        # Break: comparing batch1 to itself, no GPU sync, hidden-output leak.
        self.assertTrue(Path(__file__).with_name("probe_batching.py").exists(), "batch probe missing")
        probe = importlib.import_module("probe_batching")
        seen = []
        sync = []
        cuda = SimpleNamespace(synchronize=lambda: sync.append(1), reset_peak_memory_stats=lambda: None,
                               max_memory_allocated=lambda: 1024**2)
        class Encoder:
            embed_eos = "<eos>"
            def tokenizer(self, text, **kwargs): return {"input_ids": [0] * min(len(text), kwargs["max_length"])}
            def encode(self, texts, **kwargs):
                seen.append(kwargs["batch_size"])
                return np.array([[1, len(text) / 100000, 0] for text in texts], dtype=np.float32)
        model = SimpleNamespace(base_model=SimpleNamespace(model=Encoder()))
        result = probe.measure_model(SimpleNamespace(torch=SimpleNamespace(cuda=cuda)), model)
        self.assertTrue(result["synthetic_only"])
        self.assertEqual(len(result["protocol_results"]), 2)
        self.assertTrue({1, 2, 4}.issubset(set(seen)))
        self.assertGreaterEqual(len(sync), 12)
        for item in result["protocol_results"]:
            self.assertEqual(set(item["policies"]), {"1", "2", "4"})
            self.assertGreater(item["policies"]["2"]["seconds"], 0)
            self.assertEqual(item["policies"]["2"]["peak_gpu_memory_mb"], 1)
        import json
        encoded = json.dumps(result)
        self.assertNotIn("embedding", encoded)
        self.assertNotIn("sentences", encoded)

    def test_superseded_serial_calibrator_fails_before_validation_or_model(self):
        # Break: silently measuring old single-GPU/full-BRIGHT baseline.
        import calibrate_baseline as legacy
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "superseded"):
            legacy.measure(object(), Path(directory))


if __name__ == "__main__": unittest.main()
