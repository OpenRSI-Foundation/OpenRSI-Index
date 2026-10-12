"""Report a Logs-repository publication check.

Writes a Markdown report to the job summary and, for a Hugging Face pull
request, posts the same report as one comment on that PR. The report carries
the checker's output, which names files, rules and line numbers only; the
checker never prints matched text. The Hub token is read from the HF_TOKEN
environment variable, never from the command line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Hub comments are not meant for whole logs; the end of the output (where the
# result line is) is kept and the run link gives the rest.
MAX_OUTPUT = 20_000
STATUS = {
    "pass": "PASS",
    "fail": "FAIL",
    "stale": "SKIPPED (stale)",
    "error": "ERROR (the check did not complete; see the workflow run)",
}
COMMAND = re.compile(r"^::(warning|error|notice)::", re.MULTILINE)


def fence(text: str) -> str:
    """A backtick fence longer than any backtick run in the text."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def build_report(
    status: str,
    output: str,
    *,
    repo: str,
    ref: str,
    sha: str,
    mode: str,
    run_url: str = "",
) -> str:
    # Workflow-command prefixes mean nothing on the Hub; keep the level as text.
    output = COMMAND.sub(lambda m: m.group(1).upper() + ": ", output).rstrip()
    if len(output) > MAX_OUTPUT:
        output = (
            "[earlier output omitted; see the workflow run]\n" + output[-MAX_OUTPUT:]
        )
    lines = [
        f"**OpenRSI-Index publication check: {STATUS[status]}**",
        "",
        f"- Repository: `{repo}`",
        f"- Ref: `{ref}`",
        f"- Revision: `{sha}`",
        f"- Mode: {mode}",
    ]
    if run_url:
        lines.append(f"- Workflow run: {run_url}")
    if status == "stale":
        lines += [
            "",
            "The ref moved before this run fetched it; the run for the newer "
            "revision reports instead.",
        ]
    if output:
        marker = fence(output)
        lines += ["", marker + "text", output, marker]
    lines += [
        "",
        "Suspected credentials are advisory warnings; maintainers merge only after "
        "a passing check and their own review.",
    ]
    return "\n".join(lines) + "\n"


def post_comment(repo: str, pr: int, body: str) -> None:
    from huggingface_hub import HfApi

    HfApi().comment_discussion(
        repo, pr, body, repo_type="dataset", token=os.environ["HF_TOKEN"]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", choices=sorted(STATUS), required=True)
    parser.add_argument("--output", type=Path, help="Checker output file")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--summary", type=Path, help="Append the report here")
    parser.add_argument(
        "--comment-pr", type=int, help="Post the report on this Hub PR number"
    )
    args = parser.parse_args(argv)
    output = ""
    if args.output and args.output.is_file():
        output = args.output.read_text(encoding="utf-8", errors="replace")
    body = build_report(
        args.status,
        output,
        repo=args.repo,
        ref=args.ref,
        sha=args.sha,
        mode=args.mode,
        run_url=args.run_url,
    )
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(body)
    if args.comment_pr is not None:
        if not os.environ.get("HF_TOKEN"):
            print("::notice::HF_TOKEN is not configured; no PR comment posted.")
            return 0
        try:
            post_comment(args.repo, args.comment_pr, body)
        except Exception as exc:  # noqa: BLE001 -- never echo a response that may quote a token
            print(f"::error::Posting the Hub PR comment failed ({type(exc).__name__}).")
            return 1
        print(f"Posted the check result on Hub PR #{args.comment_pr}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
