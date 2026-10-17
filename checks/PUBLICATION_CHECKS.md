# Task and log publication checks

`Task and log publication (CPU only)` runs on pull requests and pushes to
`main` that touch `rsi-tasks/`, the publication checker, its tests and
policies, or the workflow itself. A Harness change instead runs
`harness-publication-checks.yml`, which compiles every published task and runs
the checker's tests with the proposed Harness. Run logs are not stored here;
they are in the Hugging Face dataset (see
[Logs on Hugging Face](#logs-on-hugging-face)), where the log rules
below apply with paths reported as `rsi-logs/<path>`. Only affected task
packages are checked; a run with no package change passes with “not applicable”. A manual run
(`workflow_dispatch`) audits every published package (`--full`), and so does a
forced push to `main` (for example after a history rewrite). A newer push to
a PR cancels that PR's running check; runs for pushes to `main` and manual runs
are never cancelled, so each push's changes are audited. Nested `signature-tasks/` paths are supported.

Files directly under `rsi-tasks/`, `rsi-tasks/signature-tasks/`, `rsi-logs/` or
`rsi-logs/signature-tasks/` belong to no package. Only Markdown documentation
(`.md`) may sit there, and it is secret-scanned like package files; any other file
there fails, as does a file under `rsi-tasks/` with no enclosing `task.toml`.
Removing such a file always passes. Failures name the task or log root and the
file concerned.

This is a **publication check, not a scientific evaluation**. It never builds
task images, runs `solution/solve.sh`, runs `tests/test.sh`, calls a model,
allocates GPUs, installs task dependencies, or contacts an evaluation service.
The workflow receives no provider/cloud credentials and only a read-only GitHub
token. Dependencies for CPU configuration compilation come from the PR's **base**
Harness revision and frozen lockfile, not a contributor's environment.

## Task checks

- Parse TOML; require `task.toml`, non-empty instructions, name and description.
  Deleting `task.toml` while other task files remain fails with “missing
  task.toml”.
- Accept any positive finite declared timeout. **There is no five-hour cap.**
- Check integer non-negative Work/Judge GPU declarations as metadata only.
- For runnable packages, require schema 1.4, a verifier entry and an environment
  image/build declaration; compile with the current RSI-Harness without running.
- Syntax-parse changed task-owned Python without importing/executing it. Code
  the Harness runs (`tests/` and `solution/`) must parse; elsewhere, for example
  `environment/` starter code that is deliberately broken or targets the task
  image's Python, a parse failure is a warning only.
- Reject publication symlinks and stored files of 100 MiB or more. A larger
  task data file can be split losslessly with the task runner's
  `split_run_log.py`: inside a task, a `manifest.json` in exactly the runner's
  chunk format (`version`, `source_name`, `source_size`, `source_sha256`,
  `chunk_size`, `parts`) is verified and its stream reassembled for scanning as
  for logs below. Other task files named `manifest.json` are task data.
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
- Verify existing publication/release file manifests (`*manifest.json` and
  `publication.json` under `rsi-logs/`) against tracked files, SHA-256 and
  declared byte counts. Verify gzip restoration hash/size/line counts and ordered
  chunk checksums/whole-stream hashes where provided. Changing or removing a
  manifested file requires updating its manifest. A malformed manifest fails
  naming the manifest. Task files with such names are task data (for example
  verifier asset inventories), not log manifests; only a task `manifest.json` in
  the runner's exact chunk format is verified as one.
- Reassemble chunk streams for validation and secret scanning, so a token split
  across a chunk boundary is not missed. Parts named by a checksum-verified chunk
  manifest are scanned as raw bytes but never decompressed or parsed
  individually: they are decoded only after complete reassembly, never mistaken
  for independently complete gzip/archive files, so a split `.jsonl.gz` or
  `.tar.gz` is validated as one file. Parts without a manifest are checked as
  standalone files, where a fragment of a compressed stream fails. Both
  `parts[].path/bytes` plus `sha256/bytes` and the runner's `parts[].file/size`
  plus `source_sha256/source_size` format are supported.
- Verify version 2 manifests (`<name>.manifest.json`, e.g.
  `agent_output.manifest.json`): the raw file's size, SHA-256 and line count
  must match. They are written by `publish_logs.py` when it restores a gzip or
  chunked file, and record what the file was restored from.

Manifests are currently optional for plain log files; an absent manifest is not
proof of trajectory completeness. CI can verify the bytes and records published,
but cannot prove that every original private event was uploaded or validate the
truth of a self-reported measurement. Operator review and source provenance remain
necessary. Original publication receipts should be named `source-publication.json`
so old private-path/hash receipts are not mistaken for the new public manifest.

## Advisory credential scanning and archive safety

Suspected credential matches are **warnings, not a merge gate**. Hardcoded
values and recorded code can produce false positives; each warning gives only
the file, rule and line for human review. The file is the published path, with
`!member` appended for archive members or decompressed content
(`bundle.zip!cfg/run.yaml`, `log.txt.gz!log.txt`) and for the stream a chunk
manifest reassembles (`run.log.parts/manifest.json!run.log`). Every distinct
warning is printed; there is no cap. A successful check does **not** certify
that a publication contains no secrets. Confirmed credentials still need to be
redacted before publication; this policy does not disable GitHub secret scanning
or repository push protection. Scanner installation/runtime failures remain
errors, so a skipped or incomplete scan cannot masquerade as a clean scan.

Gitleaks 8.30.1 is installed from a SHA-256-pinned release archive. Its provider
rules remain enabled; the policy includes only narrowly reviewed code/digest
false positives. There is no broad transcript/path exemption: the scanner's
default configuration skips paths such as images, PDFs, `*.bin`, lockfiles,
`node_modules/` and virtualenvs, so staged copies get neutral file names that
keep only the suffixes path-specific rules need (YAML, Terraform/HCL, PHP,
PKCS#12, NuGet config). Inline `gitleaks:allow`, contributor `.gitleaksignore`,
and contributor scanner configs cannot suppress findings.

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
8 GiB aggregate scan bytes (including wrappers, nested expansions and tar header
blocks), 1 MiB of tar header records (PAX/long-name) per member, 100,000 members
(archive directory entries included; a ZIP's declared entry count is checked
before its directory is read) and four archive levels. Exceeding a bound **fails**, never silently truncates/skips a scan.
Larger publications need an explicitly reviewed budget adjustment. The CPU CI
job has a 20-minute safety timeout independent of research budgets. Scanner
diagnostics expose only file/rule/line identifiers; matched text and scanner
reports are not printed or uploaded as public artifacts.

Enforcement runs in its own job and executes only the checker, policies and
Harness from the PR's base revision; the proposed revision is read as data. If
the base lacks the checker or its policies, the job fails rather than falling
back to the proposed ones. The proposed checker's tests run in a separate job on
a separate runner, so test code cannot alter the enforcement job's files.

This is **not a security boundary**. With a `pull_request` trigger GitHub runs
the workflow file from the PR itself, so a PR can still edit this workflow, or
the checks it runs, to change its own result. The check informs reviewers, who
should look at any change to `.github/workflows/` or `checks/`. It becomes a
gate only if maintainers make it a required status check and protect those paths,
for example with code-owner review.

## Logs on Hugging Face

Run logs are published in the Hugging Face dataset
[OpenRSI-Foundation/OpenRSI-Index-Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs).
Its layout is the same as the former `rsi-logs/` folder, and files are stored raw: no gzip,
no chunk parts.

Upload logs with `checks/publish_logs.py`:

```bash
python3 -I checks/publish_logs.py <run directory> --task <task> --gitleaks <gitleaks>
```

It copies the run, restores any gzip or chunked files to raw files, runs the
log checks above plus the secret scan, and uploads only if they pass. Add
`--dry-run` to check without uploading, or `--create-pr` to upload as a Hub
pull request. A new log folder needs a `README.md`.

Hub pull requests are not checked automatically. Before merging one, run
`publish_logs.py --check-pr <N> --gitleaks <gitleaks>`.

`hf-logs-publication-checks.yml` audits the whole dataset whenever the checker,
its policies, `publish_logs.py` or `rsi-tasks/` change, and can be run manually.
A change only to `rsi-tasks/` that adds, removes or renames no task skips it.
It uses the checker's Logs mode (`--mount rsi-logs --task-repo <Index checkout>
--task-ref <ref>`), which differs from the checks here in three ways:
compressed files and chunk parts are rejected, a file left as an undownloaded
LFS pointer fails, and `.gitattributes` may only hold LFS rules at the root.

## History guard

`history-guard.yml` runs on every pull request. It fails a PR that does not
share history with its base branch, or that adds files under a path listed in
`[history_guard] forbidden_prefixes` of `checks/publication-policy.toml`:
`rsi-logs/` and the BRIGHT assets, which were removed from history in the
[2026-10 rewrite](../assets/docs/history-rewrite-2026-10.md).

## Local use

From the repository root, with the published base Harness environment installed:

```bash
RSI-Harness/.venv/bin/python -I checks/check_publication.py \
  --repo . --base-ref origin/main --compile \
  --gitleaks /absolute/path/to/gitleaks \
  --secret-config checks/publication-gitleaks.toml

python3 -I checks/test_publication.py
python3 -I checks/test_publish_logs.py
```

The checkout must match the checked HEAD. Local checks without `--gitleaks`
explicitly report that secret scanning was not performed; PR CI always supplies
it. `--full` requests a whole-repository audit rather than an incremental PR
check; a manual workflow run uses it. The scanner, compiler and Git LFS
integration tests skip without `PUBLICATION_GITLEAKS_BIN`, the Harness or
`git-lfs` unless `PUBLICATION_REQUIRE_INTEGRATION=1`, which CI sets. The legacy shell scripts and
planted fixtures in `checks/` are retained for historical reference, **not** used
as the current publication policy.
`check-task-timeout.sh` has been updated to positive/finite validation with no cap.

The old PR `/run`, `/cheat`, `/validate`, `/review`, `/fortify`, automatic trial
dispatch/status and missing-tool chart/rubric workflows are retired. Their source
remains recoverable in Git history. Discussion proposal review/build dispatch,
Harness CPU tests and labels remain independent; labels only describe and never
trigger computation.

PR area labels are kept in sync on every push and base-branch change by
`auto-label-pr.yml` from the
path table in `checks/auto_label.py`: `tasks` (`rsi-tasks/`),
`harness` (`RSI-Harness/`), `CI` (workflows, `checks/`,
`.github/WORKFLOW_API.md`, top-level dotfiles such as `.gitignore`), `docs` (README, CONTRIBUTING, LICENSE, PR template,
`assets/`) and `skills` (`.agents/`, `.claude/`, `.codex/`). A label is added
while the PR (or a renamed file's old path) touches its paths and removed once it
no longer does; other labels are left alone, and a missing area label is
created. A manual run with `open` relabels every open PR. Each Task Ideas
Discussion gets exactly one domain label from its `| Task Type | Domain | … |`
row (`checks/task_domain.py`): `AI4Sci` → `science task`; `AI4AI`, a missing
row or an unknown value → `general task`. The two Discussion labels are created
by a maintainer; the review workflow fails visibly if one is missing.
`checks/backfill_discussion_domain_labels.py` (dry run unless `--apply`) labels
existing Discussions.

Branch protection is unchanged. Before requiring `Task and log publication (CPU
only)` as a merge gate, remove its path filter (an unrelated PR then passes with
“not applicable”): a required check that never starts leaves the PR waiting.
