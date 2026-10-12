# Task and log publication checks

`Task and log publication (CPU only)` runs on every pull request. Unrelated PRs
explicitly pass with “not applicable”; only affected `rsi-tasks/` packages and
`rsi-logs/` archives are checked. Nested `signature-tasks/` paths are supported.
Task-only, log-only and combined task/log PRs are all supported.

This is a **publication check, not a scientific evaluation**. It never builds
task images, runs `solution/solve.sh`, runs `tests/test.sh`, calls a model,
allocates GPUs, installs task dependencies, or contacts an evaluation service.
The workflow receives no provider/cloud credentials and only a read-only GitHub
token. Dependencies for CPU configuration compilation come from the PR's **base**
Harness revision and frozen lockfile, not a contributor's environment.

## Task checks

- Parse TOML; require non-empty instructions, name and description.
- Accept any positive finite declared timeout. **There is no five-hour cap.**
- Check integer non-negative Work/Judge GPU declarations as metadata only.
- For runnable packages, require schema 1.4, a verifier entry and an environment
  image/build declaration; compile with the current RSI-Harness without running.
- Syntax-parse changed task-owned Python without importing/executing it.
- Reject publication symlinks and stored files of 100 MiB or more.
- Allow complete task removal, but reject partial removal of required files.

Existing GPIC and Marin signature archives have documented, narrowly named
compatibility exceptions in `publication-policy.toml`: GPIC retains its historical
schema/runtime; Marin has no standard `tests/test.sh`. These exceptions apply
only if that package already exists at the PR merge base. They never allow a new
task to claim a historical exception. These archives still undergo format,
integrity and secret checks; passing CI does not claim current-Harness runnability.

No fixed slug length, global pytest version, internet policy, oracle/nop score,
or binary `reward >= 1` threshold is imposed on a research task. Network and
reward semantics remain the task's declared scientific contract.

## Log checks

- Require a corresponding public task; GPIC's historical log name is explicitly
  mapped in the policy. A newly added archive needs a README describing its
  source, recorded protocol and limitations.
- Parse JSON/JSONL, including compressed structured records; reject duplicate
  JSON keys and non-finite JSON numbers. Raw unsuccessful outputs that are not
  JSON should be retained as `.txt`/`.log`, not rewritten to look successful.
- Validate changed evolution indexes: unique recorded rounds, finite numeric or
  null scores, and agreement between a declared best score and its recorded round.
  Failed, rejected, cancelled, negative, low and unscored records are legitimate.
  Round-number gaps and a single model are allowed. CI does not invent missing
  submissions, select score direction, demand a final result or closing summary.
- Verify existing publication/release file manifests against tracked files,
  SHA-256 and declared byte counts. Verify gzip restoration hash/size/line counts
  and ordered chunk checksums/whole-stream hashes where provided. Changing or
  removing a manifested file requires updating its manifest.
- Reassemble chunk streams for secret scanning, so a token split across a chunk
  boundary is not missed. Both `parts[].path/bytes` plus `sha256/bytes` and the
  runner's `parts[].file/size` plus `source_sha256/source_size` format are supported.
  Checksum-verified compressed chunks are decoded only after complete
  reassembly, never mistaken for independently complete gzip/archive files.

Manifests are currently optional for plain log files; an absent manifest is not
proof of trajectory completeness. CI can verify the bytes and records published,
but cannot prove that every original private event was uploaded or validate the
truth of a self-reported measurement. Operator review and source provenance remain
necessary. Original publication receipts should be named `source-publication.json`
so old private-path/hash receipts are not mistaken for the new public manifest.

## Advisory credential scanning and archive safety

Suspected credential matches are **warnings, not a merge gate**. Hardcoded
values and recorded code can produce false positives; each warning gives only
the file, rule and line for human review. A successful check does **not** certify
that a publication contains no secrets. Confirmed credentials still need to be
redacted before publication; this policy does not disable GitHub secret scanning
or repository push protection. Scanner installation/runtime failures remain
errors, so a skipped or incomplete scan cannot masquerade as a clean scan.

Gitleaks 8.30.1 is installed from a SHA-256-pinned release archive. Its provider
rules remain enabled; the policy includes only narrowly reviewed code/digest
false positives. There is no broad transcript/path exemption. Inline
`gitleaks:allow`, contributor `.gitleaksignore`, and contributor scanner configs
cannot suppress findings.

All files in an affected package are scanned, not just changed lines. Gzip,
tar (including gzip/bzip2/xz), zip, xz and bzip2 are read with bounded traversal;
compression signatures are recognized even after renaming. Archive members are
copied as inert regular files; log archive links, devices, traversal paths,
encrypted ZIPs and unsupported rar/7z/zstd wrappers fail. Task-source tar archives
may contain safe relative symbolic-link declarations naming a regular file in
the same archive (for example, upstream `CLAUDE.md` → `AGENTS.md`). These are
scanned as inert link-target metadata, never created or followed; the target
file is scanned independently. Absolute/traversal, missing, chained and hard
links remain errors. No executable modes are restored.
These archive-safety and integrity failures remain blocking even though suspected
credentials are advisory.

Bounds: 100 MiB per stored file, 1 GiB per expanded member/reconstructed stream,
8 GiB aggregate scan bytes (including wrappers and nested expansions), 100,000
members and four archive levels. Exceeding a bound **fails**, never silently
truncates/skips a scan. Larger publications need an explicitly reviewed budget
adjustment. The CPU CI job has a 20-minute safety timeout independent of research
budgets. Scanner diagnostics expose only file/rule/line identifiers; matched
text and scanner reports are not printed or uploaded as public artifacts.

After bootstrap, enforcement uses the checker and policies from the PR's base
revision. Proposed checker changes are unit-tested separately; they cannot waive
their own task/log failures. The first PR introducing these files uses its own
implementation solely because no base implementation exists yet.

## Local use

From the repository root, with the published base Harness environment installed:

```bash
RSI-Harness/.venv/bin/python -I checks/check_publication.py \
  --repo . --base-ref origin/main --compile \
  --gitleaks /absolute/path/to/gitleaks \
  --secret-config checks/publication-gitleaks.toml

python3 -I checks/test_publication.py
```

The checkout must match the checked HEAD. Local checks without `--gitleaks`
explicitly report that secret scanning was not performed; PR CI always supplies
it. `--full` requests an explicit whole-repository audit rather than an incremental
PR check. The legacy shell scripts and planted fixtures in `checks/` are retained
for historical reference, **not** used as the current publication policy.
`check-task-timeout.sh` has been updated to positive/finite validation with no cap.

The old PR `/run`, `/cheat`, `/validate`, `/review`, `/fortify`, automatic trial
dispatch/status and missing-tool chart/rubric workflows are retired. Their source
remains recoverable in Git history. Discussion proposal review/build dispatch,
Harness CPU tests and labels remain independent. The `gpu` label only describes
declared Work/Judge resources; it does not trigger computation.

This PR does not change branch-protection settings. If maintainers later want a
merge gate, require the stable `Task and log publication (CPU only)` check only
after this workflow has landed successfully.
