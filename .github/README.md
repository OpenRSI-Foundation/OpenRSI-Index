# Discussion rubric review configuration

The Discussion rubric reviewer uses `gpt-6-luna` with `high` reasoning through
an OpenAI-compatible proxy, not the official OpenAI endpoint. It runs on a
GitHub-hosted runner independently of the private task-worker slots.

In this repository's **Settings → Secrets and variables → Actions**, configure
these before deploying the workflow:

| Type | Name | Value |
| --- | --- | --- |
| Repository secret | `RUBRIC_API_KEY` | The proxy API key. |
| Repository variable or secret | `RUBRIC_BASE_URL` | The proxy API base URL, including `/v1` (for example, `https://proxy.example/v1`). |

If both a variable and a secret define `RUBRIC_BASE_URL`, the variable takes
precedence. Only one is needed.

The workflow maps these to the SDK's `OPENAI_API_KEY` and `OPENAI_BASE_URL` only
for the rubric-review step. Both are required for a new review; a missing URL
must not silently send the proxy key to the official endpoint. Do not include
`/responses` in the base URL: the SDK appends it.

The reviewer no longer needs the shared `OPENAI_API_KEY` secret. The separate
`/run` and `/cheat` experiment workflows still reference it when using OpenAI;
keep it only if those calls are needed. No RSI-Skills secret, W2 login,
worker-slot provider configuration, or Codex `config.toml` change is needed for
this reviewer.

The proxy must be reachable from GitHub-hosted runners and support Responses,
web search, strict JSON output, and image inputs when a proposal contains images.
Existing API exponential backoff, rubric content, and review output format are
unchanged. Never commit API keys.
