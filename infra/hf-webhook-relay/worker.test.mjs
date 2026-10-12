// Run with: node --test infra/hf-webhook-relay/worker.test.mjs
// No network: global fetch is replaced by a recorder.
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import worker, { dispatchesFor, secretsEqual } from "./worker.js";

const ENV = {
  HF_REPO: "OpenRSI-Foundation/OpenRSI-Index-Logs",
  GITHUB_REPO: "OpenRSI-Foundation/OpenRSI-Index",
  WEBHOOK_SECRET: "test-webhook-secret",
  GITHUB_TOKEN: "test-dispatch-token",
};
const A = "a".repeat(40);
const B = "b".repeat(40);

function payload(updatedRefs, overrides = {}) {
  return {
    event: { action: "update", scope: "repo.content" },
    repo: { type: "dataset", name: ENV.HF_REPO, headSha: B },
    updatedRefs,
    webhook: { id: "x", version: 3 },
    ...overrides,
  };
}

let calls;
let pending;
let logs;
const realFetch = globalThis.fetch;
const realLog = console.log;

beforeEach(() => {
  calls = [];
  pending = [];
  logs = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init, body: JSON.parse(init.body) });
    return new Response(null, { status: 204 });
  };
  console.log = (...args) => logs.push(args.join(" "));
});

afterEach(() => {
  globalThis.fetch = realFetch;
  console.log = realLog;
});

const ctx = { waitUntil: (promise) => pending.push(promise) };

async function deliver(body, { secret = ENV.WEBHOOK_SECRET, method = "POST", env = ENV } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (secret !== null) headers["X-Webhook-Secret"] = secret;
  const request = new Request("https://relay.example/", {
    method,
    headers,
    body: method === "POST" ? (typeof body === "string" ? body : JSON.stringify(body)) : undefined,
  });
  const response = await worker.fetch(request, env, ctx);
  await Promise.all(pending);
  return response;
}

test("secret comparison", async () => {
  assert.equal(await secretsEqual("same", "same"), true);
  assert.equal(await secretsEqual("same", "Same"), false);
  assert.equal(await secretsEqual("", "same"), false);
  assert.equal(await secretsEqual("same-but-longer", "same"), false);
});

test("PR update is relayed as one repository_dispatch", async () => {
  const response = await deliver(payload([{ ref: "refs/pr/7", oldSha: A, newSha: B }]));
  assert.equal(response.status, 202);
  assert.equal(calls.length, 1);
  const [{ url, init, body }] = calls;
  assert.equal(url, "https://api.github.com/repos/OpenRSI-Foundation/OpenRSI-Index/dispatches");
  assert.equal(init.method, "POST");
  assert.equal(init.headers.Authorization, "Bearer test-dispatch-token");
  assert.ok(init.headers["User-Agent"]);
  assert.deepEqual(body, {
    event_type: "hf-logs",
    client_payload: {
      repo: "OpenRSI-Foundation/OpenRSI-Index-Logs",
      ref: "refs/pr/7",
      sha: B,
      old_sha: A,
      pr: 7,
    },
  });
});

test("new main and new PR refs carry a null old_sha", async () => {
  await deliver(
    payload([
      { ref: "refs/heads/main", oldSha: null, newSha: A },
      { ref: "refs/pr/1", oldSha: "0".repeat(40), newSha: B },
    ]),
  );
  assert.deepEqual(
    calls.map((c) => c.body.client_payload),
    [
      { repo: ENV.HF_REPO, ref: "refs/heads/main", sha: A, old_sha: null, pr: null },
      { repo: ENV.HF_REPO, ref: "refs/pr/1", sha: B, old_sha: null, pr: 1 },
    ],
  );
});

test("deleted refs, other branches, tags and conversion refs are ignored", async () => {
  const response = await deliver(
    payload([
      { ref: "refs/pr/3", oldSha: A, newSha: null },
      { ref: "refs/heads/dev", oldSha: A, newSha: B },
      { ref: "refs/tags/v1", oldSha: null, newSha: B },
      { ref: "refs/convert/parquet", oldSha: A, newSha: B },
      { ref: "refs/heads/main\nx", oldSha: A, newSha: B },
      { ref: "refs/pr/4", oldSha: A, newSha: "not-a-sha" },
    ]),
  );
  assert.equal(response.status, 200);
  assert.equal(calls.length, 0);
});

test("only the configured dataset and the repo.content scope are relayed", async () => {
  const update = [{ ref: "refs/heads/main", oldSha: A, newSha: B }];
  for (const body of [
    payload(update, { repo: { type: "model", name: ENV.HF_REPO } }),
    payload(update, { repo: { type: "dataset", name: "someone/else" } }),
    payload(update, { event: { action: "create", scope: "discussion" } }),
    payload(update, { event: { action: "update", scope: "repo.config" } }),
    payload(undefined),
  ]) {
    const response = await deliver(body);
    assert.equal(response.status, 200);
  }
  assert.equal(calls.length, 0);
});

test("wrong or missing secret is rejected before anything is sent", async () => {
  const body = payload([{ ref: "refs/heads/main", oldSha: A, newSha: B }]);
  assert.equal((await deliver(body, { secret: "wrong" })).status, 401);
  assert.equal((await deliver(body, { secret: null })).status, 401);
  assert.equal(calls.length, 0);
});

test("only POST is accepted", async () => {
  const response = await deliver(null, { method: "GET" });
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("Allow"), "POST");
  assert.equal(calls.length, 0);
});

test("invalid JSON is a client error", async () => {
  assert.equal((await deliver("{not json")).status, 400);
  assert.equal(calls.length, 0);
});

test("a relay without its configuration fails closed", async () => {
  const body = payload([{ ref: "refs/heads/main", oldSha: A, newSha: B }]);
  const response = await deliver(body, { env: { ...ENV, WEBHOOK_SECRET: "" } });
  assert.equal(response.status, 500);
  assert.equal(calls.length, 0);
});

test("the response does not wait for GitHub, and failures stay quiet about secrets", async () => {
  let release;
  globalThis.fetch = (url, init) =>
    new Promise((resolve) => {
      calls.push({ url, init, body: JSON.parse(init.body) });
      release = () => resolve(new Response("{\"message\":\"Bad credentials\"}", { status: 401 }));
    });
  const request = new Request("https://relay.example/", {
    method: "POST",
    headers: { "X-Webhook-Secret": ENV.WEBHOOK_SECRET },
    body: JSON.stringify(payload([{ ref: "refs/heads/main", oldSha: A, newSha: B }])),
  });
  const response = await worker.fetch(request, ENV, ctx);
  assert.equal(response.status, 202);
  assert.equal(pending.length, 1);
  release();
  await Promise.all(pending);
  const output = logs.join("\n");
  assert.match(output, /HTTP 401/);
  for (const secret of [ENV.WEBHOOK_SECRET, ENV.GITHUB_TOKEN, "Bad credentials"]) {
    assert.ok(!output.includes(secret));
  }
});

test("a network failure is logged by error type only", async () => {
  globalThis.fetch = async () => {
    throw new TypeError(`failed with ${ENV.GITHUB_TOKEN}`);
  };
  await deliver(payload([{ ref: "refs/heads/main", oldSha: A, newSha: B }]));
  const output = logs.join("\n");
  assert.match(output, /request failed \(TypeError\)/);
  assert.ok(!output.includes(ENV.GITHUB_TOKEN));
});

test("dispatchesFor tolerates malformed updatedRefs entries", () => {
  assert.deepEqual(dispatchesFor(payload([null, 3, {}, { ref: 5 }]), ENV), []);
  assert.deepEqual(dispatchesFor(payload("refs/heads/main"), ENV), []);
});
