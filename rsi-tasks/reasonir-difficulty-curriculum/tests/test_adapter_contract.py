"""Focused regression fixtures for bounded candidate PEFT configuration errors."""

from pathlib import Path
import unittest

import evaluate


class AdapterConfigContractTests(unittest.TestCase):
    def test_non_object_root_is_candidate_invalid(self) -> None:
        with self.assertRaises(evaluate.CandidateInvalid) as raised:
            evaluate.validate_peft_config([], Path("/workspace/submission/adapter/adapter_config.json"))
        self.assertEqual(raised.exception.payload["code"], "adapter_config_type")

    def test_wrong_typed_numeric_is_candidate_invalid(self) -> None:
        document = dict(evaluate.EXPECTED_CONFIG_DEFAULTS)
        document["target_modules"] = list(evaluate.TARGETS)
        document["lora_dropout"] = []
        with self.assertRaises(evaluate.CandidateInvalid) as raised:
            evaluate.validate_peft_config(document, Path("/workspace/submission/adapter/adapter_config.json"))
        self.assertEqual(raised.exception.payload["code"], "adapter_config_field")
        self.assertIn("lora_dropout", raised.exception.payload["condition"])


if __name__ == "__main__":
    unittest.main()
