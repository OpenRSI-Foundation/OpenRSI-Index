"""Shared filesystem fixtures and fail-closed retired-calibrator regression."""
import hashlib
from pathlib import Path
import tempfile
import types
import unittest


class CalibrationTests(unittest.TestCase):
    def fixture(self, root):
        events = []
        tests = root / "tests"
        (tests / "assets").mkdir(parents=True)
        for name in ("assets/manifest.json", "model-lock.json", "workspace-tree.json", "evaluate.py"):
            (tests / name).write_text("immutable test fixture")
        zero = root / "zero"
        zero.mkdir()
        (zero / "adapter_model.safetensors").write_bytes(b"canonical-zero-fixture")
        (zero / "adapter_config.json").write_text("canonical-config-fixture")
        adapter = root / "adapter"
        adapter.mkdir()
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (adapter / name).symlink_to(zero / name)
        logs = root / "logs"
        logs.mkdir()
        evaluator = types.SimpleNamespace(
            TESTS=tests, BASE=root / "base", BASE_REVISION="fixed-revision", SUBMISSION=adapter,
            __file__=str(tests / "evaluate.py"),
            verify_hash_manifest=lambda *args: events.append(("hash", args)),
            verify_workspace=lambda: events.append("workspace"),
            validate_adapter=lambda: (adapter, True, hashlib.sha256(b"canonical-zero-fixture").hexdigest()),
            load_base_model=lambda: events.append("load_base"),
        )
        return evaluator, zero, logs, events

    def test_serial_protocol_rejects_before_any_validation(self):
        # Break: legacy calibration silently generates an incompatible baseline.
        import calibrate_baseline
        with tempfile.TemporaryDirectory() as temporary:
            evaluator, zero, logs, events = self.fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "superseded"):
                calibrate_baseline.measure(evaluator, logs)
            self.assertEqual(events, [])
            self.assertEqual(list(logs.iterdir()), [])


if __name__ == "__main__": unittest.main()
