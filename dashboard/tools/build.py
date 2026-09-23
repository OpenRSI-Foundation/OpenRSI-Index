#!/usr/bin/env python3
"""Build a static, paginated dashboard from a committed OpenRSI-Index checkout."""
import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parsers import atif, event, transcript

DASHBOARD = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/OpenRSI-Foundation/OpenRSI-Index"
DOMAINS = {
    "ace-playbook-repair": "Agents", "gemm-h100-refined": "MLSys",
    "liger-tied-ce": "MLSys", "minference-sparse-prefill": "MLSys",
    "learnability-cot": "Post-training", "kev-decision-architecture": "Post-training",
    "reasonir-difficulty-curriculum": "Post-training", "molmo2-pointing-refined": "Vision",
    "molmoweb-interaction-context": "Vision", "isaaclab-peginsert-reward-search": "Robotics",
    "gpic-10m-autoresearch": "Vision", "pre-training-optimizer-update-geometry": "Pre-training",
    "post-training-qwen-122B-rl": "Post-training",
}
TITLES = {"gpic-10m-autoresearch": "GPIC · Text-to-image",
          "pre-training-optimizer-update-geometry": "Marin · Optimizer scaling ladder",
          "post-training-qwen-122B-rl": "Qwen-122B-RL · Research report"}


def finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def natural(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def publish_events(events, target):
    """Bound network/DOM work; oversized events continue without losing characters."""
    pages, page, size, count, tools = [], [], 0, 0, 0

    def flush():
        nonlocal page, size
        if page:
            name = f"{len(pages):04d}.json.gz"
            write(target / name, page)
            pages.append({"file": name, "count": len(page), "first": page[0]["id"]})
            page, size = [], 0

    for item in events:
        if not item.get("text", "").strip():
            continue
        tools += item["kind"] == "tool"
        body = item["text"]
        for offset in range(0, len(body), 24000):
            row = {**item, "text": body[offset:offset + 24000], "id": count + 1}
            if offset:
                row["title"] += " · continued"
            row_size = len(json.dumps(row, ensure_ascii=False).encode())
            if page and (len(page) >= 40 or size + row_size > 240000):
                flush()
            page.append(row)
            size += row_size
            count += 1
    flush()
    return {"pages": pages, "events": count, "tools": tools}


def build(repo, output):
    repo = repo.resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "rsi-logs"], text=True):
        raise ValueError("Commit changes to rsi-logs before exporting a traceable public snapshot.")
    tracked = set(subprocess.check_output(["git", "-C", str(repo), "ls-files", "rsi-logs"], text=True).splitlines())
    paths = {repo / p for p in tracked}
    roots = {p.parent for p in paths if p.name in {"evolve_state.json", "final_result.json", "experiments.jsonl"}}
    roots |= {p.parent.parent for p in paths if p.name == "trajectory.json"}
    roots |= {p.parent for p in paths if p.name == "formal-v8-research-visualization.html"}
    runs = []

    def url(path, mode="blob"):
        return f"{REPOSITORY}/{mode}/{commit}/{quote(path.relative_to(repo).as_posix())}"

    for root in sorted(roots):
        relative = root.relative_to(repo).as_posix()
        parts = root.relative_to(repo / "rsi-logs").parts
        signature = parts[0] == "signature-tasks"
        task = parts[1] if signature else parts[0]
        ident = hashlib.sha256(relative.encode()).hexdigest()[:16]
        final, state, plan = (read(root / name) if root / name in paths else {}
                              for name in ("final_result.json", "evolve_state.json", "run-plan.json"))
        config = plan.get("task", {})
        agent_config = config.get("agent", {})
        agent = final.get("agent") or agent_config.get("name") or ("codex" if "codex" in root.name else "claude-code" if "claude" in root.name else "report")
        model = final.get("model") or agent_config.get("model") or root.name.replace("claude-code-", "").replace("codex-", "")
        submissions = [{"round": str(s.get("round", i + 1)), "score": finite(s.get("score")),
                        "status": str(s.get("status", "unknown")), "at": finite(s.get("at"))}
                       for i, s in enumerate(state.get("submissions", [])) if isinstance(s, dict)]
        status = "timed_out" if final.get("timed_out") else str(final.get("status") or "archived")
        run = {"id": ident, "task": task, "title": TITLES.get(task, task),
               "track": "Signature" if signature else "Public", "domain": DOMAINS.get(task, "Other"),
               "agent": agent, "model": model, "effort": agent_config.get("reasoning_effort"),
               "status": status, "best_score": finite(final.get("best_score", state.get("best_score"))),
               "best_round": final.get("best_round", state.get("best_round")),
               "runtime_seconds": finite(final.get("runtime_seconds")), "submissions": submissions,
               "direction": config.get("score_direction", "maximize"), "path": relative,
               "source_url": url(root, "tree"), "files": [], "notes": [], "format": "Transcript"}
        if not final:
            run["notes"].append("This archive has no final run result; its current execution status is unknown.")
        if run["status"] == "timed_out":
            run["notes"].append("The run reached its time limit. Recorded submission scores remain available.")
        log_files = sorted((p for p in paths if p.parent == root and re.fullmatch(r"agent_output(?:\.resume-\d+)?\.(?:txt|jsonl|log)(?:\.gz)?", p.name)), key=natural)
        # The primary session precedes numbered resume sessions.
        log_files.sort(key=lambda p: (".resume-" in p.name, natural(p)))
        atif_path, experiments = root / "trajectory/trajectory.json", root / "experiments.jsonl"
        if atif_path in paths:
            data = read(atif_path)
            run["agent"] = data.get("agent", {}).get("name", agent)
            run["model"] = data.get("agent", {}).get("model_name", model)
            run["format"] = "ATIF trajectory"
            events = atif(data)
            log_files = [atif_path]
        elif experiments in paths:
            run["format"] = "Experiment journal"
            records = [json.loads(line) for line in experiments.read_text().splitlines() if line.strip()]
            # GPIC mixes screening protocols; do not combine FD values into a score curve.
            run["notes"].append("Experiment metrics use different screening protocols; no aggregate score is inferred.")
            events = (event("experiment", row, row.get("attempt_id", "Experiment")) for row in records)
            log_files = [experiments]
        elif log_files:
            def sessions(files=tuple(log_files), name=agent):
                for path in files:
                    yield event("system", path.name, "Session file")
                    yield from transcript(path, name)
            events = sessions()
        elif root / "formal-v8-research-visualization.html" in paths:
            run.update(format="Research report", agent="report", model="Historical report", status="report")
            run["notes"] = ["Historical research report; not a reproduced Harness run or a controlled data-selection comparison."]
            report = root / "README.md"
            events = [event("assistant", report.read_text(), "Research report")]
            log_files = [report, root / "formal-v8-research-visualization.html"]
            run["report_url"] = url(root / "formal-v8-research-visualization.html")
        else:
            events = []
            run["notes"].append("No agent transcript is included in this archive; recorded evaluations and source files remain available.")
        for path in log_files:
            run["files"].append({"name": path.name, "url": url(path), "bytes": path.stat().st_size})
        for name in ["evolve_state.json", "final_result.json", "RESEARCH_NOTES.md"]:
            if root / name in paths:
                run["files"].append({"name": name, "url": url(root / name), "bytes": (root / name).stat().st_size})
        trace = publish_events(events, output / ident)
        run.update(events=trace["events"], tools=trace["tools"])
        write(output / ident / "index.json", {**run, **trace})
        runs.append(run)
        print(f"{relative}: {trace['events']} events, {len(trace['pages'])} pages", flush=True)
    if not runs:
        raise ValueError("No public RSI logs found")
    manifest = {"version": 1, "commit": commit, "repository": REPOSITORY,
                "generated_at": datetime.now(timezone.utc).isoformat(), "runs": runs}
    write(output / "index.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DASHBOARD.parent)
    parser.add_argument("--output", type=Path, default=DASHBOARD / "assets/logs-data")
    args = parser.parse_args()
    build(args.repo, args.output)
