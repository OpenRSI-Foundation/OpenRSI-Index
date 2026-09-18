"""Task-owned diagnostic fields survive redaction and real worker transport."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import evaluate as e
import parallel_encoder as p
from test_parallel_encoder import TEST_GPUS, BINDING


class DiagnosticTests(unittest.TestCase):
    def invoke_main(self, failure):
        with patch.object(e.parallel, "discover_gpu_uuids", return_value=TEST_GPUS), \
             patch.object(e, "parse_baseline", return_value=.1), \
             patch.object(e, "judge", side_effect=failure), patch.dict("os.environ", {}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised: e.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertNotIn("PRIVATE", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())
        self.assertNotIn("reward", output.getvalue())
        return json.loads(output.getvalue())

    def test_rank_diagnostic_retains_fixed_path_expected_and_bounded_actual(self):
        # Break: every diagnostic reduced to generic path/code without expected/actual.
        failure = e.CandidateInvalid("adapter_config_field", Path("/workspace/submission/adapter/adapter_config.json"),
                    "adapter config field r differs", 16, 8, "use the fixed trainer")
        result = self.invoke_main(failure)
        self.assertEqual(result["path"], "/workspace/submission/adapter/adapter_config.json")
        self.assertEqual(result.get("expected"), 16)
        self.assertEqual(result.get("actual"), 8)
        self.assertIn("r", result["condition"])

    def test_count_shape_and_budget_expected_values_are_actionable(self):
        # Break: dropping safe numerical expectations or letting arbitrary dict keys through.
        key = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        fixed = {"updates": 1000, "exposures": 64000, "world_size": 4, "per_device_batch": 4, "gradient_accumulation": 4}
        for code, path, expected, actual, wanted in (
            ("tensor_count", "/workspace/submission/adapter/adapter_model.safetensors", 448, 447, 447),
            ("tensor", "/workspace/submission/adapter/adapter_model.safetensors", {"shape": [16, 4096], "dtype": "torch.float32", "finite": True},
                {"key": key, "shape": [8, 4096], "dtype": "torch.float16", "PRIVATE_KEY": "PRIVATE"},
                {"key": key, "shape": [8, 4096], "dtype": "torch.float16"}),
            ("training_budget", "/workspace/submission/manifest.json", fixed, dict(fixed, world_size=2, PRIVATE="PRIVATE"), dict(fixed, world_size=2)),
        ):
            with self.subTest(code=code):
                result = self.invoke_main(e.CandidateInvalid(code, Path(path), "task-owned condition", expected, actual, "use the fixed trainer"))
                self.assertEqual(result["path"], path)
                self.assertEqual(result.get("expected"), expected)
                self.assertEqual(result.get("actual"), wanted)

    def test_untrusted_strings_paths_shapes_and_oversized_metadata_are_redacted(self):
        # Break: unrestricted candidate strings survive via path, actual or JSON field lists.
        for code, expected, actual in (("adapter_config_field", 16, "PRIVATE" * 10000),
                ("tensor_count", 448, 10**10000),
                ("tensor", {"shape": [16, 4096], "dtype": "torch.float32", "finite": True}, {"key": "PRIVATE", "shape": ["PRIVATE"], "dtype": "PRIVATE"}),
                ("adapter_config_fields", sorted(e.EXPECTED_CONFIG_FIELDS), ["PRIVATE" * 10000]),
                ("adapter_load", "loadable fixed-schema adapter", "RuntimeError: PRIVATE")):
            with self.subTest(code=code):
                result = self.invoke_main(e.CandidateInvalid(code, Path("/workspace/submission/PRIVATE"),
                                "adapter config field r differs", expected, actual, "use fixed artifacts"))
                self.assertLess(len(json.dumps(result)), 4096)

    def test_ambiguous_numeric_device_diagnostic_is_actionable_without_raw_error(self):
        # Break: rejected numeric masks emit only opaque RuntimeError.
        inventory = [(str(i), gpu, str(i)) for i, gpu in enumerate(TEST_GPUS)]
        def discover(): return p.select_gpu_uuids(inventory, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        with patch.object(e.parallel, "discover_gpu_uuids", side_effect=discover), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised: e.main()
        self.assertEqual(raised.exception.code, 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result.get("code"), "gpu_visibility_order")
        self.assertIn("PCI_BUS_ID", result.get("hint", ""))
        self.assertIn("UUID", result["hint"])

    def test_worker_load_failure_is_sanitized_and_propagated_even_after_exit(self):
        # Break: worker CandidateInvalid becomes opaque RuntimeError, or private loader text leaks.
        self.assertTrue(hasattr(e, "safe_candidate_diagnostic"), "code-specific diagnostic sanitizer missing")
        messages = []
        connection = SimpleNamespace(send=messages.append, close=lambda: None)
        cuda = SimpleNamespace(device_count=lambda: 1, set_device=lambda i: None, reset_peak_memory_stats=lambda: None)
        def load_candidate(*args):
            raise e.CandidateInvalid("adapter_load", Path("/workspace/submission/adapter"),
                        "fixed PEFT loader rejected the candidate", "loadable fixed-schema adapter", "PRIVATE loader error", "rerun the fixed trainer and selector")
        evaluator = SimpleNamespace(torch=SimpleNamespace(cuda=cuda), load_base_model=lambda: object(),
                    load_candidate=load_candidate, CandidateInvalid=e.CandidateInvalid,
                    safe_candidate_diagnostic=e.safe_candidate_diagnostic)
        with patch.object(p, "load_evaluator", return_value=evaluator), \
             patch.object(p, "validate_candidate", return_value=(Path("."), BINDING)), patch.dict("os.environ", {}):
            with self.assertRaises(SystemExit): p.worker_main(connection, 0, TEST_GPUS[0], BINDING, False)
        self.assertEqual(len(messages), 1)
        self.assertNotIn("PRIVATE", json.dumps(messages))
        pool = p.ParallelEncoder(TEST_GPUS, BINDING)
        pool.processes = [SimpleNamespace(is_alive=lambda: False)]
        reply = SimpleNamespace(poll=lambda timeout: True, recv=lambda: messages[0])
        with self.assertRaises(p.WorkerCandidateInvalid) as raised: pool.receive(reply)
        result = self.invoke_main(raised.exception)
        self.assertEqual(result["code"], "adapter_load")
        self.assertEqual(result["path"], "/workspace/submission/adapter")
        self.assertEqual(result["expected"], "loadable fixed-schema adapter")
        self.assertIn("trainer", result["hint"])


if __name__ == "__main__": unittest.main()
