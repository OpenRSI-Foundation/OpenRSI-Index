# Published log sanitization

These logs are sanitized copies of recorded runs. Host paths and user identities,
site and node names, host directory listings, and infrastructure environment or
mount details have been replaced with anonymous values. Anonymous paths describe
relationships within a run; they are not deployment instructions.

Task content, recorded scores, timings, model identities, public artifact digests,
and W&B experiment references are retained. Original event ordering is retained;
redacted environment values and directory listings are explicitly marked.

The ACE playbook repair, MolmoWeb interaction context, and ReasonIR difficulty
curriculum archives retain the complete sanitized agent and runner output,
persisted submissions, final summaries, and verifier evidence. Where a shutdown
summary differs from the persisted submission count, the task-level README
explains the discrepancy without rewriting either original measurement.

ACE's Codex transcript exceeds GitHub's uncompressed per-file limit and is
published as lossless gzip, with restoration instructions and published-byte
checksums in its run directory. Compression does not omit events. Sanitized
transcripts naturally have different byte hashes from their private originals;
recorded candidate/model/data digests within them are retained.
