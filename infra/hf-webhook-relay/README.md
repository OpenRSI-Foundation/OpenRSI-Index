# Hugging Face webhook relay

A Cloudflare Worker that turns Hugging Face webhook deliveries for the Logs
dataset repository into GitHub `repository_dispatch` events, so that
`.github/workflows/hf-logs-publication-checks.yml` in OpenRSI-Index checks every
update to the Logs repository's `main` branch and to each of its pull requests.

```
Hub webhook (scope repo.content, X-Webhook-Secret)
  -> this Worker (checks the secret, the repository and the ref)
  -> POST /repos/<GITHUB_REPO>/dispatches  event_type "hf-logs"
  -> Logs repository publication checks (comments on the Hub PR)
```

For each entry of the delivery's `updatedRefs` whose `ref` is `refs/heads/main`
or `refs/pr/<N>` and whose `newSha` is set (deleted refs are skipped), the
Worker sends one dispatch with

```json
{
  "event_type": "hf-logs",
  "client_payload": {
    "repo": "<HF_REPO>",
    "ref": "refs/pr/<N> | refs/heads/main",
    "sha": "<newSha>",
    "old_sha": "<oldSha or null>",
    "pr": "<N as a number, or null for main>"
  }
}
```

It accepts only `POST` requests whose `X-Webhook-Secret` header equals
`WEBHOOK_SECRET` (compared in constant time), and relays only `repo.content`
events of the dataset named by `HF_REPO`. Everything else gets a `2xx` and is
ignored, so the Hub neither retries nor suspends the webhook. The response is
sent before GitHub is called (`ctx.waitUntil`); a dispatch that fails is logged
by HTTP status only and is covered by the workflow's daily full audit. The
Worker never logs the secret, the token or the payload. The workflow validates
the payload again before it uses it.

## Configuration

| Name | Kind | Value |
|---|---|---|
| `HF_REPO` | variable (`wrangler.toml`) | The Logs dataset repository, `owner/name` |
| `GITHUB_REPO` | variable (`wrangler.toml`) | The repository that runs the checks |
| `WEBHOOK_SECRET` | secret | A long random string; the same value is entered in the Hub webhook |
| `GITHUB_TOKEN` | secret | A token that may create repository dispatch events on `GITHUB_REPO` |

`wrangler.toml` holds no secrets. For `GITHUB_TOKEN`, use a fine-grained personal
access token (or a GitHub App installation token) limited to the
`GITHUB_REPO` repository only, with the single repository permission
**Contents: Read and write**, the minimum the repository dispatch endpoint
accepts. Give it an expiry and rotate it. Contents write also allows pushing,
so keep the token only in the Worker's secret store.

If `HF_REPO` changes, change it in `wrangler.toml` and in the `LOGS_REPO`
value of `.github/workflows/hf-logs-publication-checks.yml`; the contract tests
in `checks/test_publication.py` require both to agree.

## Deploy

Requires Node.js and a Cloudflare account; `wrangler` runs through `npx`. The
Worker itself has no npm dependencies.

```bash
cd infra/hf-webhook-relay
node --test worker.test.mjs                 # offline tests (fetch is mocked)
npx wrangler login                          # or set CLOUDFLARE_API_TOKEN
openssl rand -hex 32                        # generate the webhook secret
npx wrangler secret put WEBHOOK_SECRET      # paste the generated value
npx wrangler secret put GITHUB_TOKEN        # paste the fine-grained token
npx wrangler deploy                         # prints https://hf-webhook-relay.<subdomain>.workers.dev
```

Check it: `curl -i -X POST <worker URL>` answers `401` without the secret.

## Create the Hub webhook

Hub webhooks belong to a user account, not to an organization. Use an account of
a maintainer who can read the Logs repository:

1. Open <https://huggingface.co/settings/webhooks> and create a webhook.
2. **Target repositories**: add the dataset `OpenRSI-Foundation/OpenRSI-Index-Logs`
   (type *dataset*).
3. **Webhook URL**: the Worker URL printed by `wrangler deploy`.
4. **Secret**: the `WEBHOOK_SECRET` value.
5. **Triggers**: *Repo update* only. That delivers the `repo.content` events
   (new commits on `main` and on PR refs, including a newly opened PR); the
   Worker ignores the other scopes, so community events are unnecessary.
6. Save, push a commit to a test PR, and confirm in the webhook's *Activity* tab
   that the delivery got `202`, and in OpenRSI-Index's Actions tab that
   *Logs repository publication checks* ran. Deliveries can be replayed there.

The workflow posts its result to the Hub PR with the GitHub secret `HF_TOKEN`
(a Hub token with write access to the Logs repository's discussions). Without
it the workflow still runs and writes its job summary but posts no comment.
