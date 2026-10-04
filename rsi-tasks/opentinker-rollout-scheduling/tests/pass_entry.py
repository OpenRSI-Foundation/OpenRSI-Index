"""One disposable paired-pass child. Trace-bearing output stays in private scratch."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "adapter"))
from policy_diagnostics import sanitize_detail


def main():
    specification = json.loads(Path(sys.argv[1]).read_text())
    try:
        from runtime import run_pass
        value = run_pass(specification["records"], specification["warmup"], specification["source"],
                         (ROOT / "reference/policy.py").read_text(), specification["batch_size"],
                         specification["shards"], specification["scratch"])
        value = {"status": "complete", "value": value}
    except BaseException as exc:
        # Unwrap Ray's nested task errors without serializing tracebacks or hidden data.
        cause = exc
        for _ in range(12):
            child = getattr(cause, "cause", None)
            if child is None or child is cause:
                break
            cause = child
        code = getattr(cause, "code", None)
        allowed = {"policy_import", "policy_callback", "policy_serialization", "policy_output_size", "policy_protocol",
                   "policy_crash", "policy_timeout", "decision_shape", "decision_size", "admit_id",
                   "admit_state", "dispatch_shape", "dispatch_id", "dispatch_state", "replica", "schedule_stalled"}
        if code in allowed:
            value = {"status": "candidate_invalid", "code": code,
                     "detail": sanitize_detail(getattr(cause, "detail", None))}
        else:
            # Fixed strings classify infrastructure without leaking exception messages.
            message = str(cause)
            infrastructure = {"address_discovery", "gpu_contract", "prefix_cache_reset_failed", "backend_output", "warmup_failure",
                              "context_bound", "mask_length", "incomplete_timing", "incomplete_batch"}
            if message in infrastructure:
                code = message
            elif "out of memory" in message.lower():
                code = "runtime_oom"
            elif "illegal memory access" in message.lower():
                code = "runtime_cuda_fault"
            elif isinstance(cause, (ImportError, ModuleNotFoundError)):
                code = "runtime_import"
            else:
                code = "runtime_failure"
            value = {"status": "infrastructure", "code": code}
    # Private bounded transport only. The supervisor deletes it and owns reward.
    with Path(sys.argv[2]).open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
