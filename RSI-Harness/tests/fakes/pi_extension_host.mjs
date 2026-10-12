// Loads pi's rsi-never-stop extension the way pi does and feeds it a scripted
// sequence of pi events, for unit tests without pi. No dependencies.
//
// usage: node --experimental-strip-types pi_extension_host.mjs EXTENSION SCENARIO.json
//
// The extension's own stdout lines are the transcript under test. Everything
// the host observes goes to stderr as JSON lines {host: {kind, ...}}:
//   {kind: "result", event, value, ms}
//                                    a handler's return value and how long
//                                    it took to return, in milliseconds
//   {kind: "send", text, options}    a pi.sendUserMessage call
//
// Scenario keys:
//   takeOverStdout  true: send ordinary stdout writes to stderr first, as
//                   pi's print mode does
//   queueFollowUps  false: sendUserMessage never queues anything
//   ctxThrows       true: every ctx method throws
//   events          [{type, ...event fields}] dispatched in order. After an
//                   agent_before_settle, the host delivers the queued
//                   follow-ups as user message_end events, as pi does.
//   exitCode        exit with this code at the end (default 0)

import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [extensionPath, scenarioPath] = process.argv.slice(2);
const scenario = JSON.parse(readFileSync(scenarioPath, 'utf8'));
const report = (obj) => process.stderr.write(`${JSON.stringify({ host: obj })}\n`);

if (scenario.takeOverStdout) {
  process.stdout.write = (chunk, ...rest) => process.stderr.write(chunk, ...rest);
}

const handlers = new Map();
const pending = [];
const pi = {
  on(name, handler) {
    if (!handlers.has(name)) handlers.set(name, []);
    handlers.get(name).push(handler);
  },
  getActiveTools: () => ['read', 'bash', 'edit', 'write'],
  sendUserMessage(text, options) {
    report({ kind: 'send', text, options });
    // pi queues asynchronously, after its input handlers ran.
    if (scenario.queueFollowUps !== false) setImmediate(() => pending.push(text));
  },
};
const fail = () => {
  throw new Error('ctx is unavailable');
};
const ctx = scenario.ctxThrows
  ? { hasPendingMessages: fail, sessionManager: { getSessionId: fail } }
  : {
      cwd: '/workspace',
      model: { id: 'fake-model' },
      thinkingLevel: 'off',
      sessionManager: { getSessionId: () => 'sess-pi-1' },
      hasPendingMessages: () => pending.length > 0,
    };

const extension = await import(pathToFileURL(extensionPath).href);
extension.default(pi);

async function dispatch(event) {
  for (const handler of handlers.get(event.type) ?? []) {
    const started = performance.now();
    const value = await handler(event, ctx);
    const ms = Math.round(performance.now() - started);
    report({ kind: 'result', event: event.type, value: value ?? null, ms });
  }
}

for (const event of scenario.events ?? []) {
  await dispatch(event);
  if (event.type === 'agent_before_settle') {
    while (pending.length > 0) {
      const text = pending.shift();
      await dispatch({ type: 'message_end', message: { role: 'user', content: [{ type: 'text', text }] } });
    }
  }
}
process.exit(scenario.exitCode ?? 0);
