# Published log sanitization

These logs are sanitized copies of recorded runs. Host paths and user identities,
site and node names, host directory listings, and infrastructure environment or
mount details have been replaced with anonymous values. Anonymous paths describe
relationships within a run; they are not deployment instructions.

Task content, recorded scores, timings, model identities, public artifact digests,
and W&B experiment references are retained. Original event ordering is retained;
redacted environment values and directory listings are explicitly marked.
