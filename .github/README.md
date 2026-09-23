# Workflow API configuration

The Discussion rubric reviewer uses `gpt-6-luna` with `high` reasoning through
an OpenAI-compatible proxy, not the official OpenAI endpoint. It runs on a
GitHub-hosted runner independently of the private task-worker slots.
OpenAI calls in the PR `/run` and `/cheat` workflows use the same proxy settings.

In this repository's **Settings → Secrets and variables → Actions**, configure
these before deploying the workflow:

| Type | Name | Value |
| --- | --- | --- |
| Repository secret | `RUBRIC_API_KEY` | The proxy API key. |
| Repository variable or secret | `RUBRIC_BASE_URL` | The proxy API base URL, including `/v1` (for example, `https://proxy.example/v1`). |

If both a variable and a secret define `RUBRIC_BASE_URL`, the variable takes
precedence. Only one is needed.

The workflows map these to `OPENAI_API_KEY` and `OPENAI_BASE_URL`, plus
`OPENAI_API_BASE` for the trial clients that use LiteLLM. Both settings are
required for a new rubric review; a missing URL
must not silently send the proxy key to the official endpoint. Do not include
`/responses` in the base URL: the SDK appends it.

The old `OPENAI_API_KEY` repository secret is no longer used by these workflows
and can be removed once this change is deployed. The `RUBRIC_` setting names are
retained so existing configuration works without renaming secrets.

`/run` runs coding agents on PR task changes; `/cheat` runs adversarial trials
to check for reward hacking. Both can analyze the resulting trajectories and
report back to the PR. This change updates their OpenAI connection settings,
not their model matrix or separate Anthropic/Gemini credentials.
Harbor 0.14.0's analyzer uses the Claude Agent SDK (currently `sonnet`), so that
analysis still needs Anthropic credentials; the OpenAI proxy key does not
replace them.

No RSI-Skills secret, W2 login, worker-slot provider configuration, or Codex
`config.toml` change is needed.

The proxy must be reachable from GitHub-hosted runners and support Responses,
web search, strict JSON output, and image inputs when a proposal contains images.
Existing API exponential backoff, rubric content, and review output format are
unchanged. Never commit API keys.
