// Relays Hugging Face webhooks for the Logs dataset repository to GitHub.
//
// For each updated main or PR ref of env.HF_REPO, sends one repository_dispatch
// (event_type "hf-logs") to env.GITHUB_REPO, whose workflow checks that revision.
// The Hub needs a quick 2xx, so dispatches run after the response
// (ctx.waitUntil); the daily full audit covers a dispatch that failed.
// Nothing secret is logged: not the webhook secret, the token or the payload.

export const EVENT_TYPE = "hf-logs";
export const REF_PATTERN = /^refs\/(pr\/[0-9]+|heads\/main)$/;
const SHA_PATTERN = /^[0-9a-f]{40}$/;
const NULL_SHA = "0".repeat(40);
const GITHUB_API = "https://api.github.com";

const encoder = new TextEncoder();

// Compares SHA-256 digests in constant time, so neither the secret's content
// nor its length is revealed by timing.
export async function secretsEqual(given, expected) {
  const [a, b] = await Promise.all(
    [given, expected].map(async (value) => {
      const digest = await crypto.subtle.digest("SHA-256", encoder.encode(value));
      return new Uint8Array(digest);
    }),
  );
  let difference = 0;
  for (let i = 0; i < a.length; i++) difference |= a[i] ^ b[i];
  return difference === 0;
}

// The dispatches a webhook payload asks for; empty when it is not relayed.
export function dispatchesFor(body, env) {
  if (
    !body ||
    typeof body !== "object" ||
    body.event?.scope !== "repo.content" ||
    body.repo?.type !== "dataset" ||
    body.repo?.name !== env.HF_REPO ||
    !Array.isArray(body.updatedRefs)
  ) {
    return [];
  }
  const dispatches = [];
  for (const update of body.updatedRefs) {
    const ref = update?.ref;
    const newSha = update?.newSha;
    // Deleted refs have no newSha; other branches and tags are not checked.
    if (typeof ref !== "string" || !REF_PATTERN.test(ref)) continue;
    if (typeof newSha !== "string" || !SHA_PATTERN.test(newSha)) continue;
    const oldSha =
      typeof update.oldSha === "string" &&
      SHA_PATTERN.test(update.oldSha) &&
      update.oldSha !== NULL_SHA
        ? update.oldSha
        : null;
    const pr = ref.startsWith("refs/pr/")
      ? Number(ref.slice("refs/pr/".length))
      : null;
    dispatches.push({
      event_type: EVENT_TYPE,
      client_payload: { repo: env.HF_REPO, ref, sha: newSha, old_sha: oldSha, pr },
    });
  }
  return dispatches;
}

async function dispatch(payload, env) {
  const ref = payload.client_payload.ref;
  try {
    const response = await fetch(`${GITHUB_API}/repos/${env.GITHUB_REPO}/dispatches`, {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        "Content-Type": "application/json",
        "User-Agent": "hf-webhook-relay",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify(payload),
    });
    // Status only: GitHub error bodies are not needed to diagnose a relay.
    console.log(`dispatch ${ref} ${payload.client_payload.sha}: HTTP ${response.status}`);
    return response.ok;
  } catch (error) {
    console.log(`dispatch ${ref}: request failed (${error?.name ?? "Error"})`);
    return false;
  }
}

function reply(status, text) {
  return new Response(text + "\n", {
    status,
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}

export default {
  async fetch(request, env, ctx) {
    if (request.method !== "POST") {
      return new Response("method not allowed\n", {
        status: 405,
        headers: { Allow: "POST", "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    if (!env.WEBHOOK_SECRET || !env.GITHUB_TOKEN || !env.HF_REPO || !env.GITHUB_REPO) {
      console.log("relay is not configured: set HF_REPO, GITHUB_REPO and both secrets");
      return reply(500, "relay not configured");
    }
    const secret = request.headers.get("X-Webhook-Secret") ?? "";
    if (!(await secretsEqual(secret, env.WEBHOOK_SECRET))) {
      return reply(401, "unauthorized");
    }
    let body;
    try {
      body = await request.json();
    } catch {
      return reply(400, "invalid JSON");
    }
    const dispatches = dispatchesFor(body, env);
    if (dispatches.length === 0) {
      // Other repositories, scopes and refs are acknowledged and ignored, so
      // the Hub does not retry or suspend the webhook.
      return reply(200, "ignored");
    }
    ctx.waitUntil(Promise.all(dispatches.map((payload) => dispatch(payload, env))));
    return reply(202, `dispatching ${dispatches.length}`);
  },
};
