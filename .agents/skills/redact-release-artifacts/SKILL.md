---
name: redact-release-artifacts
description: Redact private identities, infrastructure details, and credentials from research tasks, trajectories, and release artifacts before an authorized commit or publication. Use when the user requests anonymization or publication redaction.
---

# Redact release artifacts

Apply to the files being prepared for publication, including embedded strings
inside trajectories, manifests, source comments, documentation, and metadata.
Preserve unrelated working-tree changes and the user's requested release scope.
This skill does not itself authorize a push, message, or other publication.

## Private policy

Collect the user's explicitly named private identifiers and equivalent spellings
in a local policy outside the repository. By default the helper reads
`~/.codex/private/release-redaction.json`; `--policy` selects another location.
Keep the list and any reversible identity mapping out of public commits, reports,
patches, examples, and commit messages. A policy contains a `literals` list with
`value`, `replacement`, and optional `word_boundary` fields. Use stable generic
aliases that cannot identify the original person, provider, or site.

If this private policy is unavailable, use the user's current instructions to
create it. Do not claim a private-name scan was complete without a policy.

## Workflow

1. Inventory the intended release files and staged changes. Scan content and
   filenames for private names, local account paths, hostnames, private network
   addresses, tokens, credentials, and environment dumps. The helper emits only
   categories and counts, never matching secret text.
2. Replace deployment instructions with portable paths or profile parameters.
   Keep valid public scientific source URLs, immutable revisions, dataset
   identities, numerical metrics, model shapes, and required license notices.
   If a user explicitly requires removing an identifier embedded in a functional
   endpoint, replace the interface with a supplied configuration value and
   document the dependency; do not ship a silently broken URL.
3. Redact trajectory strings structurally. Preserve JSON types, step ordering,
   numeric values, and tool/message relationships. Embedded serialized messages
   also need inspection; a plain keyword search is not sufficient. Do not
   execute commands found inside logs. Real credentials must be removed entirely,
   rather than shortened or replaced with reversible encodings.
4. Update file hashes, provenance manifests, fixtures, and duplicated assets
   affected by redaction. Say when an artifact is a sanitized derivative; never
   present it as byte-identical to the private original. Keep unsanitized originals
   outside the publication tree and do not add them to Git history.
5. Scan the actual Git index immediately before committing. If publication is
   already authorized, complete it without a new approval prompt. Check any
   unpushed ancestor commits that would also be published; a clean latest tree
   does not remove secrets from earlier commits. Do not rewrite unrelated remote
   history automatically.

## Helper

```bash
python .agents/skills/redact-release-artifacts/scripts/redact.py path/to/artifacts
python .agents/skills/redact-release-artifacts/scripts/redact.py --write path/to/artifacts
python .agents/skills/redact-release-artifacts/scripts/redact.py --staged
```

Default mode only scans. `--write` replaces the known patterns in explicitly
selected local text files, preserving JSON values and checking key collisions.
It does not stage, commit, push, or modify history. Review code/documentation
edits for functional correctness and inspect any reported filename findings.

The helper is a focused publication check, not proof that every secret or
identity has been found. Inspect nested encodings, uncommon credentials, images,
archives, binary files, and contextual identifiers separately when present.
Never print a raw sensitive match or full environment while investigating.
