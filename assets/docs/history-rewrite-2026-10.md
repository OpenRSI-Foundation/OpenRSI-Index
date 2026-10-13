# History rewrite of October 2026

In October 2026 the history of `main` was rewritten to remove two paths from
every commit, so all commit IDs changed. Old `main` ended at
`2c2ab9f47dabf73eef19f31d704c7eb148503745`; its rewritten copy is
`b414e56bab7bdead9e6b31bb0cff8a48fc557329`.

| Removed path | Where it is now |
| --- | --- |
| `rsi-logs/` | The [OpenRSI-Index-Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs) dataset, with its full history |
| `rsi-tasks/reasonir-difficulty-curriculum/tests/assets/bright/` | Downloaded by the task's `prepare_eval_assets.py` (since #241) |

A full clone went from about 600 MiB to about 65 MiB. Everything else is
unchanged: files, authors, dates and commit messages. The only difference is
that commits which changed nothing but `rsi-logs/` are gone:
- 12 commits changed only `rsi-logs/` (184 commits on `main`
  became 172); their content is in the Logs dataset.

To find the new ID of an old commit, look it up in
[`history-rewrite-2026-10-commit-map.tsv`](history-rewrite-2026-10-commit-map.tsv)
(`old_commit`, `new_commit` or `removed`, matching Logs dataset commit, subject).

## Updating your clone

If `git log --oneline -1 main -- rsi-logs` prints anything, your clone still
has the old history. Commit or stash your work, then:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

Do this instead of cloning again: a proposal-agent session in progress keeps
its Discussion binding inside `.git`. To keep unmerged work, create a branch
from the new `main` and cherry-pick your own commits onto it. Don't merge or
rebase old history into it; the `history-guard` check rejects pull requests
that bring it back.
