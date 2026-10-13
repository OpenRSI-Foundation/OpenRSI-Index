# Task and log publication checks

`Task and log publication (CPU only)` runs on pull requests and pushes to
`main` that touch `rsi-tasks/`, `rsi-logs/`, the publication checker, its tests
and policies, the RSI-Harness sources and lockfile it compiles tasks with, the
workflow itself, or the Logs upload tool, Logs audit workflow and history guard
whose tests it runs. Only affected `rsi-tasks/` packages and `rsi-logs/`
archives are checked; a run with no package change passes with “not applicable”. A manual run
(`workflow_dispatch`) audits every published package (`--full`), and so does a
push to `main` that was forced (for example after a history rewrite) or whose
previous revision the checkout cannot find: with no usable base there is no diff
to audit, so the head is audited against itself. A newer push to
a PR cancels that PR's running check; runs for pushes to `main` and manual runs
are never cancelled, so each push's changes are audited. Nested `signature-tasks/` paths are supported. Task-only, log-only and
combined task/log PRs are all supported.

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
- Verify version 2 manifests, which describe one raw file, optionally restored
  from an earlier compressed or chunked publication. One sits beside its file
  as `<name up to the first dot>.manifest.json` (`agent_output.manifest.json`
  for `agent_output.txt`):

  ```json
  {"version": 2, "artifact": "agent_output.txt", "size": 330542050,
   "sha256": "<hex>", "line_feeds": 5880480,
   "converted_from": {"artifact": "agent_output.txt.gz", "compression": "gzip",
     "compressed_size": 51574548, "compressed_sha256": "<hex>",
     "parts": [{"path": "part-0001.bin", "bytes": 1, "sha256": "<hex>"}],
     "original_manifest": {}},
   "note": "..."}
  ```

  The raw file's size, SHA-256 and line-feed count must match; `converted_from`
  (optional; `parts` only for a former chunked file, `original_manifest` the
  replaced manifest verbatim) is checked for shape only, because the former
  publication's bytes are no longer published.

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

## Logs repository checks (Hugging Face)

Run logs can also be published in a separate Hugging Face dataset repository,
`OpenRSI-Foundation/OpenRSI-Index-Logs` (`LOGS_REPO` in `checks/publish_logs.py`),
whose root has the layout of
`rsi-logs/` (`<task>/…`, `signature-tasks/<task>/…`). The checker's **Logs
mode** checks such a repository with the same rules as `rsi-logs/` here:

```bash
python3 -I checks/check_publication.py \
  --repo <Logs checkout with LFS content> --mount rsi-logs \
  --task-repo <OpenRSI-Index checkout> --task-ref <ref> \
  (--base-ref <base> --head-ref <head> | --full) \
  --policy checks/publication-policy.toml \
  --secret-config checks/publication-gitleaks.toml --gitleaks <gitleaks>
```

- Every Logs path `p` is checked, and reported, as `rsi-logs/<p>`.
- Tasks, aliases and historical exceptions come from `--task-ref` of
  `--task-repo` (`git ls-tree … rsi-tasks`), never from the Logs repository: a
  log needs its task merged into OpenRSI-Index first.
- `README.md` (the dataset card), `LICENSE` and `.gitattributes` at the root are
  repository infrastructure and are secret-scanned only. `.md` files at the root
  or under `signature-tasks/` are collection documentation; any other file
  outside a log root fails.
- `.gitattributes` is allowed only at the root, and only with
  `<pattern> filter=lfs diff=lfs merge=lfs -text` rules. Other attributes
  (`text`, `eol`, `ident`, `working-tree-encoding`, ...) would make a checkout
  differ from the blobs the Hub serves, so the checked bytes (and a re-encoded
  credential) would not be what is published.
- A new or modified `*.manifest.json` must be a version 2 manifest; any
  manifest with `"version": 2` named `*.manifest.json` or carrying `artifact`
  is verified as one, so a misspelled field fails instead of being skipped.
- The repository stores **raw files only**; large and binary files go through
  LFS/Xet, so there is no stored-size limit (the scan bounds below still apply).
  A new or modified compressed or archived file (`.gz`, `.xz`, `.bz2`, `.zst`,
  `.zip`, `.tar*`, `.7z`, `.rar`, or such content under another name) or a
  lossless chunk (a file under `*.parts/`, or a chunk manifest) fails with a
  hint to upload the raw file instead. ZIP containers with their own suffix
  (for example `.npz`) are raw files. A `--full` audit applies this to every
  file.
- A file that is still a Git LFS pointer fails: a skipped LFS download must
  never pass as a clean, tiny file. `--lfs-include` prints the LFS include
  patterns an incremental check reads (affected log roots and changed root
  files) and exits.
- `--base-ref` may be omitted with `--full`. A base that is missing or shares no
  history with the head fails with a request to audit with `--full`.

### Publishing logs: `checks/publish_logs.py`

Logs reach the Logs repository through `checks/publish_logs.py`, which checks
them before anything is uploaded:

```bash
pip install huggingface_hub==2.0.0      # the pinned client the tool requires
hf auth login                           # or set HF_TOKEN; the token is never printed
python3 -I checks/publish_logs.py <run directory or log tree> \
  --task <task path under rsi-tasks/, e.g. signature-tasks/x> \
  [--dest <path in the dataset; default: the task's log root>] \
  --gitleaks <gitleaks 8.30.1> [--create-pr] [--dry-run] [--staging <empty dir>]
python3 -I checks/publish_logs.py --check-pr <N> --gitleaks <gitleaks 8.30.1>
```

1. **Stage.** The source is copied (never modified) into a staging directory at
   `--dest`, which must lie inside the task's log root (the reviewed alias in
   `[log_task_aliases]` when there is one). Symbolic links and Git metadata are
   refused.
2. **Normalize** with the same rules as the conversion of the existing
   `rsi-logs/` history. A chunk directory (a `manifest.json` with ordered
   `parts`, in either chunk format above) is reassembled after every part and
   the whole stream are verified; `<x>.gz` (also a reassembled one) is
   decompressed to `<x>` and checked against any original manifest. Each
   converted file gets a version 2 manifest, keeping the replaced manifest as
   `original_manifest` (a chunked file that was not compressed records
   `"compression": "none"`). Listing manifests (`publication.json`,
   `*manifest.json` with `files`) list the raw file in place of the entries it
   replaces (a chunk directory's entries become one), with the same fields
   holding the raw file's values, followed by its version 2 manifest
   (`path`, `bytes`, `sha256`), so an entry for the manifest it replaced is
   dropped where it stood. If other entries record `uncompressed_*` digests,
   the raw file's entry gets them too (`"compression": null`), and a top-level
   `compression` description mentioning gzip is replaced with one for raw
   files. Rewritten listings use a two-space indent and keep the original's
   ASCII escaping when it was already in that format. Normalizing a log root
   as OpenRSI-Index published it reproduces the Logs repository's converted
   files byte for byte, except the version 2 manifests' `note` (and so their
   listed digests), which names an upload instead of the history import. Any
   other compressed or archived file (`.xz`, `.bz2`, `.zst`, `.zip`, `.tar`, gzip
   content under another name, …) is refused: unpack it yourself. Raw input
   passes through unchanged, and normalizing a normalized tree changes nothing.
3. **Check.** The staged tree is committed to a temporary Git repository and
   checked in Logs mode with `--full`, tasks from `--task-ref` (default `HEAD`)
   of the OpenRSI-Index checkout the tool lives in (`--index` to choose another).
   This covers the uploaded files only; a new log root also needs a `README.md`,
   staged or already on the Hub. Nothing scans a Hub upload or merge
   automatically, so an upload and `--check-pr` refuse to run without
   `--gitleaks` (or `PUBLICATION_GITLEAKS_BIN`); only a `--dry-run` upload
   check may skip the secret scan.
4. **Upload**, only if all of that passed: `huggingface_hub`'s `upload_folder`
   to `main` (maintainers) or, with `--create-pr`, as a Hub pull request.
   Large and binary files go through LFS/Xet automatically. `--dry-run` stops
   before the upload; `--staging` keeps the staged tree for inspection.

The upload does not delete files on the Hub. Because the pre-upload check sees
only the staged files, maintainers check a Hub pull request with
`publish_logs.py --check-pr N` before merging it: it fetches `refs/pr/N` and
`main` anonymously, downloads the LFS content of the affected log roots and runs
the checker incrementally from their merge base, exactly as for a PR here. The
checkout ignores conversion attributes in the submitted tree (`.git/info/attributes`:
`* -text -eol -ident -working-tree-encoding`, `core.autocrlf=false`) and pins the
LFS endpoint to the Hub repository, so the checked files are the published blobs. Hub
pull requests have no required checks; that local result and the maintainers'
review are the gate.

### Audit: `hf-logs-publication-checks.yml`

`Logs repository audit (CPU only)` checks the Hub's `main` (`--full`) against
this repository's tasks. It runs on pushes to `main` and pull requests that
change the checker, its policies, `publish_logs.py`, `rsi-tasks/` (a removed or
renamed task leaves orphaned logs) or the workflow, and manually with a
`revision` (`main`, `refs/pr/<N>` or a commit id). It has a read-only token and
no secrets.

- The checker and policies come from the trusted revision (the pushed revision,
  or a PR's base); the task list comes from the evaluated revision (the push, or
  the PR head), read with `git ls-tree` as data.
- The Logs repository id is read from `publish_logs.py` (`LOGS_REPO`), the only
  place it is defined. It is fetched anonymously over HTTPS and all of its LFS
  content is downloaded; the Logs revision is data and is never executed. The
  checkout neutralizes conversion attributes and pins the LFS endpoint exactly
  as `--check-pr` does.
- While the Logs repository is not published (the Hub answers 401/404), or when
  the trusted revision predates these checks, the run passes with a notice;
  any other Hub failure fails the run.
- Every run writes a job summary with the checker's output (paths, rules and
  line numbers only).

## History guard

`.github/workflows/history-guard.yml` (job `history-guard`) runs on every pull
request, with no path filter, so it can be made a required status check. It
fails a PR whose head shares no merge base with its base branch (for example a
branch made from a copy of the repository from before a history rewrite), and a
PR any of whose commits (merges included, compared with each parent) adds a file
under a prefix listed in the **base** revision's `checks/publication-policy.toml`:

```toml
[history_guard]
forbidden_prefixes = ["rsi-tasks/reasonir-difficulty-curriculum/tests/assets/bright/"]
```

Deleting such a file in a later commit does not help, because the blobs would
stay in history; the branch must be recreated without them. A PR cannot shorten
the list it is checked against, since the list is read from its base.

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
it. Logs mode is described above. `--full` requests a whole-repository audit
rather than an incremental PR check; a manual workflow run uses it. The checker
tests also cover Logs mode and the Logs audit and history-guard workflows
(running their shell steps against local fixture repositories);
`test_publish_logs.py` covers the upload tool's normalization (including a
byte-for-byte reproduction of the converted history from miniatures of the log
roots published compressed here), checks and `--check-pr` with the upload
mocked and no network. The scanner, compiler and
Git LFS integration tests skip without `PUBLICATION_GITLEAKS_BIN`, the Harness
or `git-lfs` unless `PUBLICATION_REQUIRE_INTEGRATION=1`, which CI sets. The legacy shell scripts and
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
path table in `checks/auto_label.py`: `tasks` (`rsi-tasks/`), `logs`
(`rsi-logs/`), `harness` (`RSI-Harness/`), `CI` (workflows, `checks/`,
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
