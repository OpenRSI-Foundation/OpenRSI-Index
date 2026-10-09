"""Judge-only fixed paired evaluation and sole reward writer."""
import json
import ctypes
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gate import GateError, candidate, fixed
from adapter.policy_diagnostics import repair_detail
from quality import quality_summary, success_count

REPAIRS = {
    "policy_import": ("Policy", "Policy must import and construct inside the restricted worker"),
    "policy_callback": ("reset/schedule", "callbacks must return normally using permitted standard-library helpers"),
    "policy_serialization": ("schedule.result", "the returned decision must serialize as finite acyclic JSON"),
    "policy_output_size": ("schedule", "encoded decision must fit within 1 MiB"),
    "policy_protocol": ("schedule", "return a finite JSON-compatible decision without touching protocol pipes"),
    "policy_crash": ("policy", "avoid denied syscalls and stay below the 512 MiB memory bound"),
    "policy_timeout": ("reset/schedule", "each callback and initialization must finish in five seconds"),
    "decision_shape": ("schedule", "return exactly admit and dispatch lists"),
    "decision_size": ("schedule", "each decision list must fit the current batch"),
    "admit_id": ("admit", "use each pending trajectory ID at most once"),
    "admit_state": ("admit", "admit only pending trajectories"),
    "dispatch_shape": ("dispatch", "each entry must have exactly id and replica"),
    "dispatch_id": ("dispatch.id", "use each ready trajectory ID at most once"),
    "dispatch_state": ("dispatch.id", "dispatch only ready turns"),
    "replica": ("dispatch.replica", "use integer 0 or 1"),
    "schedule_stalled": ("schedule", "admit or dispatch when no work is in flight"),
}


def diagnostic(status, code, detail=None):
    if status == "candidate_invalid":
        field, condition = REPAIRS[code]
        return {"status": status, "errors": [{"code": code, "path": "/workspace/candidate/policy.py",
                                               "field": field, "condition": condition, "hint": condition,
                                               **repair_detail(detail)}]}
    hints = {"address_discovery": "require one usable local IPv4 before Ray startup",
             "gpu_contract": "provide exactly two compatible GPUs with at least 48 GB each",
             "runtime_import": "validate the pinned image dependency build",
             "dependency_missing": "rebuild the pinned dependency closure; a required distribution is missing",
             "baseline_policy_failure": "validate the fixed reference policy and restricted callback worker",
             "warmup_failure": "validate the fixed train warmup using the public development command",
             "runtime_oom": "stop Work GPU processes and validate fixed-runtime memory use",
             "runtime_cuda_fault": "qualify the pinned CUDA runtime and GPU health",
             "runtime_failure": "run public development and execution validation to inspect non-hidden runtime traces",
             "pass_timeout": "validate the full paired cadence within the 10800-second allowance"}
    return {"status": status, "code": code, "hint": hints.get(code, "validate fixed runtime and complete protocol before research")}


class Unscored(Exception):
    def __init__(self, envelope):
        self.envelope = envelope


def kill_tree(process):
    import psutil
    # main installs subreaper behavior: detached Ray children are adopted here
    # even when the pass driver crashes before its own finally block.
    children = [child for child in psutil.Process(os.getpid()).children(recursive=True)
                if child.pid != process.pid]
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=10)
    process.wait(timeout=10)


def child_pass(spec, root, deadline):
    root.mkdir(mode=0o700)
    request, result = root / "input.json", root / "output.json"
    request.write_text(json.dumps(spec))
    # Formal raw output contains prompts, observations and protected evaluator paths.
    # It is captured, never forwarded, and deleted with the private directory.
    environment = {"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": str(root), "TMPDIR": str(root),
                   "HF_HOME": str(root / "hf"), "XDG_CACHE_HOME": str(root / "cache"),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "RAY_USAGE_STATS_ENABLED": "0",
                   "WANDB_MODE": "disabled", "ALFWORLD_DATA": "/opt/data/alfworld"}
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        if name in os.environ:
            environment[name] = os.environ[name]
    with (root / "raw.log").open("xb") as log:
        process = subprocess.Popen(["/opt/venv/bin/python", "-B", "-I", str(ROOT / "pass_entry.py"),
                                    str(request), str(result)], cwd="/", env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
        try:
            while True:
                if os.fstat(log.fileno()).st_size > 512 * 1024**2:
                    raise Unscored(diagnostic("infrastructure", "pass_log_size"))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Unscored(diagnostic("infrastructure", "pass_timeout"))
                try:
                    process.wait(timeout=min(1, remaining))
                    if os.fstat(log.fileno()).st_size > 512 * 1024**2:
                        raise Unscored(diagnostic("infrastructure", "pass_log_size"))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            raise Unscored(diagnostic("infrastructure", "pass_timeout")) from None
        finally:
            kill_tree(process)
    if process.returncode != 0 or not result.is_file():
        raise Unscored(diagnostic("infrastructure", "pass_process_crash"))
    if result.stat().st_size > 256 * 1024**2:
        raise Unscored(diagnostic("infrastructure", "pass_result_size"))
    value = json.loads(result.read_text())
    if value["status"] != "complete":
        if spec["role"] == "baseline" and value["status"] == "candidate_invalid":
            raise Unscored(diagnostic("infrastructure", "baseline_policy_failure"))
        raise Unscored(diagnostic(value["status"], value["code"], value.get("detail")))
    return value["value"]


def equivalent(left, right, logprob=False):
    if logprob and isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-5)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k], k in {
            "log_probs", "response_logprobs", "rollout_log_probs"}) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(equivalent(a, b, logprob) for a, b in zip(left, right))
    return left == right


def validate_pass(value):
    if value["episodes"] != 96 or len(value["traces"]) != 96:
        raise Unscored(diagnostic("incomplete", "episode_coverage"))
    if (not 0 <= value["model_calls"] <= 1920 or not 0 <= value["generated_tokens"] <= 245760
            or not math.isfinite(value["seconds"]) or value["seconds"] <= 0):
        raise Unscored(diagnostic("incomplete", "counts_or_timing"))
    for prefix, count in (("model_wait", value["model_calls"]), ("environment_reset_wait", value["episodes"]),
                          ("environment_step_wait", value["model_calls"])):
        if value[prefix + "_count"] != count or any(
                not math.isfinite(value[prefix + suffix]) or value[prefix + suffix] < 0
                for suffix in ("_total_sec", "_mean_sec", "_max_sec")):
            raise Unscored(diagnostic("incomplete", "wait_diagnostics"))
    if (not math.isfinite(value["completed_trajectories_per_sec"])
            or value["completed_trajectories_per_sec"] <= 0):
        raise Unscored(diagnostic("incomplete", "throughput_diagnostics"))
    for trace in value["traces"]:
        if trace["termination"] not in {"context_limit", "step_limit", "turn_limit", "environment_done"}:
            raise Unscored(diagnostic("incomplete", "termination"))
        if len(trace["turns"]) > 20 or len(trace["response_ids"]) != len(trace["response_mask"]):
            raise Unscored(diagnostic("incomplete", "trace_shape"))
        if not all(math.isfinite(p) for p in trace["response_logprobs"]):
            raise Unscored(diagnostic("incomplete", "log_probability"))


def main():
    os.umask(0o077)
    # Linux PR_SET_CHILD_SUBREAPER, available under ordinary root without added caps.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("subreaper_setup")
    # Ray's fixed 8 GiB plasma store is a shared-memory file and inherits this
    # limit too. Keep a finite framework limit; monitor raw logs separately.
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**3, 16 * 1024**3))
    deadline = time.monotonic() + 10500
    fixed(ROOT)
    source = candidate(ROOT)
    formal = json.loads((ROOT / "formal.json").read_text())
    development = json.loads((ROOT / "dev_manifest.json").read_text())["episodes"]
    warmup = [development[i] for i in formal["warmup"]]
    baseline = (ROOT / "reference/policy.py").read_text()
    rows, ratios = [], []
    with tempfile.TemporaryDirectory(prefix="opentinker-judge-") as temporary:
        scratch = Path(temporary)
        for row, (batch_size, shards) in enumerate(((8, 2), (8, 8), (32, 2), (32, 8))):
            records = [formal["episodes"][i] for i in formal["orders"][row]]
            if len({x["sha256"] for x in records}) != 96:
                raise RuntimeError("formal_manifest_coverage")
            passes = {}
            order = ("baseline", "candidate") if row % 2 == 0 else ("candidate", "baseline")
            for kind in order:
                # Ray's Unix sockets need a short root. The supervisor owns cleanup
                # on both normal exit and driver/daemon failure.
                with tempfile.TemporaryDirectory(prefix="rt-", dir="/tmp") as ray_scratch:
                    value = child_pass({"records": records, "warmup": warmup, "role": kind,
                                        "source": baseline if kind == "baseline" else source,
                                        "batch_size": batch_size, "shards": shards, "scratch": ray_scratch},
                                       scratch / f"row-{row}-{kind}", deadline)
                validate_pass(value)
                passes[kind] = value
            left, right = passes["baseline"], passes["candidate"]
            for value in passes.values():
                value["success_count"] = success_count(value["traces"])
                value["success_rate"] = value["success_count"] / value["episodes"]
            matching = sum(equivalent(a, b) for a, b in zip(left["traces"], right["traces"]))
            ratio = left["seconds"] / right["seconds"]
            if not math.isfinite(ratio) or ratio <= 0:
                raise RuntimeError("timing_ratio")
            ratios.append(ratio)
            fields = ("seconds", "episodes", "success_count", "success_rate", "model_calls", "generated_tokens", "batch_mean_sec",
                      "batch_tail_sec", "ready_wait_mean_sec", "ready_wait_max_sec",
                      "completed_trajectories_per_sec", "model_wait_count", "model_wait_total_sec",
                      "model_wait_mean_sec", "model_wait_max_sec", "environment_reset_wait_count",
                      "environment_reset_wait_total_sec", "environment_reset_wait_mean_sec",
                      "environment_reset_wait_max_sec", "environment_step_wait_count",
                      "environment_step_wait_total_sec", "environment_step_wait_mean_sec",
                      "environment_step_wait_max_sec")
            rows.append({"batch_size": batch_size, "environment_shards": shards, "ratio": ratio,
                         "matching_trajectories": matching,
                         "packed_outputs_match": equivalent(left["packed"], right["packed"]),
                         **{kind: {key: passes[kind][key] for key in fields} for kind in passes}})
    quality = quality_summary(rows)
    if not quality["passed"]:
        raise Unscored({"status": "candidate_invalid", "errors": [{"code": "quality_regression",
                        "path": "/workspace/candidate/policy.py", "field": "schedule",
                        "condition": "aggregate success rate may decrease by at most 5 percentage points",
                        "hint": "inspect public task outcomes; faster execution must retain task success"}],
                        "quality": quality, "rows": rows})
    reward = 100 * math.exp(sum(math.log(ratio) for ratio in ratios) / 4)
    if not math.isfinite(reward) or reward <= 0:
        raise RuntimeError("reward_finite_positive")
    print(json.dumps({"status": "complete", "rows": rows, "quality": quality, "reward": reward}, allow_nan=False), flush=True)
    # All children, comparisons, cleanup and diagnostics succeeded before the only reward write.
    with open("/logs/verifier/reward.json", "x", encoding="utf-8") as stream:
        json.dump({"reward": reward}, stream, allow_nan=False)


if __name__ == "__main__":
    try:
        main()
    except GateError as exc:
        print(json.dumps({"status": "candidate_invalid", "errors": [exc.detail]}), flush=True)
        sys.exit(2)
    except Unscored as exc:
        print(json.dumps(exc.envelope), flush=True)
        sys.exit(3)
    except BaseException as exc:
        code = str(exc)
        if code not in {"fixed_asset_integrity", "fixed_source_inventory", "model_integrity", "model_inventory",
                        "data_integrity", "dependency_missing", "dependency_version", "subreaper_setup",
                        "formal_manifest_coverage", "timing_ratio", "reward_finite_positive"}:
            code = "evaluator_failure"
        print(json.dumps(diagnostic("infrastructure", code)), flush=True)
        sys.exit(4)
