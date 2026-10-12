"""Extract measured performance without promoting it to an official reward."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SLUG = "sglang-omni-tts-runtime"
SEGMENTS = (
    ("d154-astra-xhigh-650175-20261001T230314", "Initial research"),
    ("d154-rolling-astra-xhigh-655820-20261002T120036", "Rolling-baseline research"),
    ("omni-resume3-669485-20261004T141956Z-resume3", "Continuation"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_feedback(path):
    found = []
    with path.open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            text = line.lstrip("\0").strip()
            if not text.startswith("{"):
                continue
            try:
                record = json.loads(text)
            except ValueError:
                continue
            if isinstance(record, dict) and {"means", "guards", "runs"} <= record.keys():
                found.append((number, record))
    if len(found) > 1:
        raise ValueError(f"Ambiguous complete feedback in {path}")
    return found[0] if found else (None, None)


def extract(index):
    root = Path(index) / "rsi-logs" / SLUG
    first_start = None
    records = []
    for run_id, label in SEGMENTS:
        directory = Path("codex-gpt-6-astra/runs") / run_id / SLUG
        source = root / directory
        start = float((source / "started_at").read_text().splitlines()[-1])
        first_start = start if first_start is None else first_start
        state = json.loads((source / "evolve_state.json").read_text())
        for entry in state["submissions"]:
            path = source / "feedback" / f'{entry["round"]}.log'
            line, feedback = complete_feedback(path)
            item = {"run_id": run_id, "segment": label, "round": entry["round"],
                    "submission_index": len(records) + 1, "segment_start_unix": start,
                    "result_recorded_unix": entry["at"],
                    "elapsed_hours": (entry["at"] - start) / 3600,
                    "research_elapsed_hours": (entry["at"] - first_start) / 3600,
                    "official_reward": entry["score"], "complete": feedback is not None,
                    "throughput_qps": None,
                    "source_feedback": {"path": (directory / "feedback" / path.name).as_posix(),
                                        "sha256": sha256(path), "json_line": line}}
            if feedback is not None:
                runs, means, guards = feedback["runs"], feedback["means"], feedback["guards"]
                assert runs.keys() == means.keys() == guards.keys()
                for model, reps in runs.items():
                    assert len(reps) == 3
                    for rep in reps:
                        coverage = rep["coverage"]
                        assert all(coverage[k] == 1088 for k in ("tts", "wavs", "asr"))
                        assert coverage["failed"] == coverage["skipped"] == 0
                    for metric, mean in means[model].items():
                        computed = sum(rep["metrics"][metric] for rep in reps) / 3
                        assert math.isclose(mean, computed, rel_tol=1e-12, abs_tol=1e-14)
                passed = all(value for model in guards.values() for value in model.values())
                assert (entry["score"] is not None) == passed
                if passed:
                    assert feedback["reward"] == entry["score"]
                baseline = feedback.get("baseline", {}).get("initial_means")
                qps = means["cosy"]["throughput_qps"]
                item.update({"throughput_qps": qps, "means": means,
                             "protocol_sha256": feedback["identity"]["evaluator_sha256"]["protocol.json"],
                             "candidate_source_sha256": feedback.get("source_sha256"),
                             "feedback_created_unix": feedback["created_unix"],
                             "original_status": feedback["status"], "all_guards_passed": passed,
                             "guards": guards, "initial_means": baseline,
                             "measured_throughput_ratio_not_reward": qps / baseline["cosy"]["throughput_qps"] if baseline else None,
                             "repetitions": {model: [{key: rep[key] for key in
                                 ("repetition", "seed", "metrics", "coverage")} for rep in reps]
                                 for model, reps in runs.items()}})
            else:
                assert entry["score"] is None
            records.append(item)
    assert len(records) == 14
    assert sum(item["complete"] for item in records) == 11
    assert sum(item["official_reward"] is not None for item in records) == 1
    return {"schema": "sglang-recorded-measurements-v1",
            "metric": "CosyVoice3 throughput, requests/s; arithmetic mean of three complete repetitions",
            "limitations": "Historical protocols differ across the three segments. A measured throughput ratio is not an official reward. Incomplete evaluations remain null. No result is relabeled as protocol v5.",
            "submissions": records}


def local_studies(transcripts):
    """Read exact controlled study output from the complete sanitized transcript."""
    relative = Path("codex-gpt-6-astra/runs") / SEGMENTS[-1][0] / SLUG / "agent_output.txt"
    path = Path(transcripts) / relative
    requested = (("H19 reference-feature transport", "h19-ab2-", 162789, 162878),
                 ("H20 typed-byte transport", "h20-split-", 419070, 419167))
    buffers = [[] for _ in requested]
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            for i, (_, _, first, last) in enumerate(requested):
                if first <= number <= last:
                    buffers[i].append(line)
            if number >= requested[-1][-1]:
                break
    studies = []
    digest = sha256(path)
    for (name, prefix, first, last), lines in zip(requested, buffers):
        text = "".join(lines)
        payload = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
        rows = payload["completed"]
        assert len(rows) == 6
        for row in rows:
            assert row["run"].startswith(prefix) and row["exit_code"] == 0
            assert row["tts"]["completed_requests"] == row["diagnostic_asr"]["evaluated"] == 1088
            assert row["tts"]["failed_requests"] == row["diagnostic_asr"]["skipped"] == 0
        means = {kind: sum(row["tts"]["throughput_qps"] for row in rows if f"-{kind}-" in row["run"]) / 3
                 for kind in ("control", "candidate")}
        studies.append({"name": name, "runs": rows, "mean_throughput_qps": means,
                        "relative_change": means["candidate"] / means["control"] - 1,
                        "source_transcript": {"restored_path": relative.as_posix(), "sha256": digest,
                                              "first_line": first, "last_line": last},
                        "limitations": "Local CosyVoice-only paired study with diagnostic ASR, not a full Judge result or official reward. H20 separates generation and transcription. Controls differ, so gains must not be summed."})
    return studies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = extract(args.index)
    if args.transcripts:
        evidence["local_controlled_studies"] = local_studies(args.transcripts)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print("Verified 11 complete measurements, 14 submissions, one unchanged official reward")
