"""CPU behavioral checks: protocol coverage, device authorization, batch numerics."""
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


class LightweightProtocolTests(unittest.TestCase):
    def helper(self):
        self.assertTrue(Path(__file__).with_name("evaluation_protocol.py").exists(),
                        "lightweight coverage/batch protocol missing")
        return importlib.import_module("evaluation_protocol")

    def test_exact_coverage_rejects_missing_extra_and_wrong_counts(self):
        # Break: partial or full-BRIGHT output masquerades as the selected protocol.
        p = self.helper()
        bright = {"biology": .2, "pony": .3, "theoremqa_theorems": .4}
        beir = {"NFCorpus": .1, "SciFact": .2, "FiQA-2018": .3, "ArguAna": .4}
        self.assertEqual(p.validate_metrics(bright, beir, 167188), (.3, .25))
        for b, g, count in (({"biology": .2}, beir, 167188),
                (dict(bright, robotics=.2), beir, 167188), (bright, beir, 167187),
                (dict(bright, pony=float("nan")), beir, 167188)):
            with self.assertRaises(RuntimeError): p.validate_metrics(b, g, count)
        p.validate_dataset_counts("biology", 57359, 103)
        with self.assertRaises(RuntimeError): p.validate_dataset_counts("biology", 57359, 102)

    def test_length_batches_bound_padding_make_long_singletons_restore_order(self):
        # Break: padding budget overflow, long-text co-batching, or reordered outputs.
        p = self.helper()
        lengths = [9000, 2, 4, 3, 32768, 5]
        groups = p.batch_indices(lengths, 4)
        self.assertEqual(groups, [[1, 3, 2, 5], [0], [4]])
        self.assertEqual(p.batch_indices(lengths, 1), [[0], [1], [2], [3], [4], [5]])
        self.assertEqual(p.batch_indices([4097, 4098, 4099, 4100], 4), [[0, 1, 2], [3]])
        rows = p.restore_batches(6, groups, [np.array([[i, -i] for i in group], dtype=np.float32) for group in groups])
        self.assertEqual(rows.tolist(), [[0, 0], [1, -1], [2, -2], [3, -3], [4, -4], [5, -5]])
        with self.assertRaises(RuntimeError): p.batch_indices(lengths, 8)
        with self.assertRaises(RuntimeError): p.restore_batches(2, [[0], [0]], [np.ones((1, 2), dtype=np.float32)] * 2)

    def test_probe_numerics_reject_invalid_arrays_and_detect_retrieval_changes(self):
        # Break: numerical probe declares compatibility despite changed rankings.
        p = self.helper()
        reference = np.array([[1, 0], [0, 1], [.8, .2], [.2, .8]], dtype=np.float32)
        same = p.compare_embeddings(reference, reference.copy(), 2)
        self.assertTrue(same["passed"])
        self.assertTrue(same["ranking_equal"])
        self.assertIn("min_cosine_similarity", same)
        self.assertGreaterEqual(same["min_cosine_similarity"], .99999)
        near = reference.copy()
        near[0, 1] += .0005
        self.assertFalse(p.compare_embeddings(reference, near, 2)["passed"])
        actual = reference.copy()
        actual[2:] = actual[2:][::-1]
        self.assertFalse(p.compare_embeddings(reference, actual, 2)["passed"])
        for bad in (reference.astype(np.float64), np.full((4, 2), np.nan, dtype=np.float32), reference[:3]):
            with self.assertRaises(RuntimeError): p.compare_embeddings(reference, bad, 2)
        with self.assertRaises(RuntimeError): p.compare_embeddings(reference, np.zeros_like(reference), 2)

    def test_calibration_binding_rejects_old_or_different_policy(self):
        # Break: controller promotes old full-BRIGHT calibration to lightweight baseline.
        p = self.helper()
        artifact = {"R": "0.30000", "G": "0.25000", "provenance": {"protocol": p.descriptor()}}
        self.assertEqual(p.baseline_values(artifact), ("0.30000", "0.25000"))
        artifact["provenance"]["protocol"]["batch_size"] = 4
        with self.assertRaises(RuntimeError): p.baseline_values(artifact)

    def test_model_batching_preserves_fixed_preprocessing_and_rows(self):
        # Break: wrong instruction/EOS/limits, real encode always batch1, or row shuffle.
        p = self.helper()
        self.assertTrue(hasattr(p, "encode_batched"), "length-aware encode missing")
        from types import SimpleNamespace
        observed = []
        class Model:
            embed_eos = "<eos>"
            def tokenizer(self, text, **kwargs):
                if kwargs != {"padding": False, "truncation": True, "max_length": 2048, "add_special_tokens": True}:
                    raise AssertionError("tokenization protocol changed")
                if not text.startswith("prompt:") or not text.endswith("<eos>"):
                    raise AssertionError("preprocessing changed")
                return {"input_ids": [0] * len(text)}
            def encode(self, texts, **kwargs):
                observed.append((list(texts), kwargs))
                return np.array([[int(t), -int(t)] for t in texts], dtype=np.float32)
        model = SimpleNamespace(base_model=SimpleNamespace(model=Model()))
        result = p.encode_batched(model, ["100", "2", "300", "4"], "prompt:", 2048, 2)
        self.assertEqual(result.tolist(), [[100, -100], [2, -2], [300, -300], [4, -4]])
        self.assertEqual([texts for texts, kw in observed], [["2", "4"], ["100", "300"]])
        self.assertTrue(all(kw == {"instruction": "prompt:", "batch_size": 2, "max_length": 2048,
                                  "convert_to_tensor": False} for texts, kw in observed))

    def test_reference_batch_one_never_retokenizes_or_reorders(self):
        # Break: modifying the reference numerical path under the default policy.
        p = self.helper()
        self.assertTrue(hasattr(p, "encode_batched"), "length-aware encode missing")
        from types import SimpleNamespace
        def encode(texts, **kwargs):
            self.assertEqual(texts, ["3", "1"])
            self.assertEqual(kwargs["batch_size"], 1)
            return np.array([[3, 0], [1, 0]], dtype=np.float32)
        model = SimpleNamespace(base_model=SimpleNamespace(model=SimpleNamespace(encode=encode)))
        self.assertEqual(p.encode_batched(model, ["3", "1"], "", 32768, 1).tolist(), [[3, 0], [1, 0]])


if __name__ == "__main__": unittest.main()
