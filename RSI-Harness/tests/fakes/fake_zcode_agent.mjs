#!/usr/bin/env node
// Scripted ZCode Protocol server (`zcode app-server` stand-in) for
// rsi-agent-driver tests. NDJSON envelopes {id,method,params} /
// {id,result|error} / {method,params}. No dependencies.
//
// usage: node fake_zcode_agent.mjs SCENARIO.json
//
// Scenario keys:
//   log            JSONL path (required)
//   createError    {code,message}: fail session/create
//   rejectSends    answer this many session/send calls with -32010 first
//   sendErrors     [{code,message}]: answer the next session/send calls with
//                  these errors, in order, before accepting any
//   ignoreStdinEnd true: keep running when stdin closes, so only the
//                  driver's signals can end it
//   replayFile     JSONL of {turn, frame}: notifications a real ZCode sent,
//                  with "{session}" for its session id
//   turns          list of turn specs; the last one repeats. A turn spec:
//     text         text_delta streamed in the turn
//     thought      reasoning_delta streamed in the turn
//     tool         {name,input,output,success}
//     hostRequests [methods] sent to the client during the turn
//     fail         {type,message}: the turn fails (turn.failed)
//     delayMs      turn duration before it ends
//     lockMs       how long the prompt lock is held after the turn ends
//     noIdle       true: never emit the idle state.updated (watchdog test)
//     idleDelayMs  wait this long between releasing the lock and the idle signal
//     empty        true: end at once with no events but the idle signal
//     exit         exit with this code instead of running the turn
//     replay       emit the replayFile frames of this turn name instead

import { appendFileSync, readFileSync } from 'node:fs';

const scenario = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const turns = scenario.turns ?? [{}];
const sessionId = 'sess_fake-zcode-1';
let rejectSends = scenario.rejectSends ?? 0;
const sendErrors = [...(scenario.sendErrors ?? [])];
let locked = false;
let sendCount = 0;
let nextId = 1;
let seq = 0;
const pending = new Map();

const log = (event, data = {}) =>
  appendFileSync(scenario.log, `${JSON.stringify({ t: Date.now(), pid: process.pid, event, ...data })}\n`);
const send = (obj) => process.stdout.write(`${JSON.stringify(obj)}\n`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const event = (type, payload, session = sessionId) => {
  seq += 1;
  send({
    method: 'session/event',
    params: { sessionId: session, type, seq, payload, deliveryKind: 'desktop-continuous' },
  });
};
const state = (reason) => {
  log('state', { reason });
  send({ method: 'state.updated', params: { sessionId, reason, scope: 'session', type: 'state.updated' } });
};
const request = (method, params) =>
  new Promise((resolve) => {
    const id = `server-${nextId++}`;
    pending.set(id, resolve);
    send({ id, method, params });
  });

log('start', { argv: process.argv.slice(2) });
process.on('SIGTERM', () => {
  log('sigterm');
  process.exit(143);
});
// Noise from other sessions must be ignored by the driver.
send({ method: 'state.updated', params: { sessionId: 'sess_other', reason: 'prompt_completed' } });

const IDLE_REASONS = new Set(['prompt_completed', 'prompt_failed']);

function replayTurn(name) {
  const frames = readFileSync(scenario.replayFile, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((entry) => entry.turn === name);
  for (const { frame } of frames) {
    const msg = JSON.parse(JSON.stringify(frame).replaceAll('{session}', sessionId));
    if (msg.method === 'state.updated' && IDLE_REASONS.has(msg.params.reason)) {
      // Real ZCode releases the prompt lock before the idle signal too.
      locked = false;
      log('state', { reason: msg.params.reason });
    }
    send(msg);
  }
}

async function runTurn(spec, n) {
  if (spec.exit !== undefined) {
    log('exit', { code: spec.exit });
    process.exit(spec.exit);
  }
  if (spec.replay) {
    replayTurn(spec.replay);
    return;
  }
  state('prompt_started');
  if (spec.empty) {
    locked = false;
    state('prompt_completed');
    return;
  }
  const turnId = `turn_${n}`;
  event('turn.started', { turnNumber: n, input: '' });
  // Another session ending its turn mid-turn must not end ours.
  event('turn.completed', { resultType: 'success', turnId: 'other' }, 'sess_other');
  send({ method: 'state.updated', params: { sessionId: 'sess_other', reason: 'prompt_completed' } });
  for (const method of spec.hostRequests ?? []) {
    const reply = await request(method, { sessionId });
    log('host_reply', { method, reply });
  }
  if (spec.thought) event('model.streaming', { kind: 'reasoning_delta', delta: spec.thought, done: false });
  if (spec.text) event('model.streaming', { kind: 'text_delta', delta: spec.text, done: false });
  if (spec.tool) {
    const id = `call_${n}`;
    event('model.streaming', { kind: 'tool_call', toolCallId: id, toolName: spec.tool.name, input: spec.tool.input });
    event('tool.updated', { kind: 'started', toolCallId: id, toolName: spec.tool.name });
    event('tool.updated', {
      kind: 'result',
      toolCallId: id,
      result: { success: spec.tool.success ?? true, content: spec.tool.output ?? '' },
    });
  }
  if (spec.delayMs) await sleep(spec.delayMs);
  if (spec.fail) event('turn.failed', { error: spec.fail, turnId });
  else event('turn.completed', { usage: { inputTokens: 100, outputTokens: 10 }, resultType: 'success', turnId });
  if (spec.lockMs) await sleep(spec.lockMs);
  locked = false;
  // ZCode releases the lock before it broadcasts the idle signal.
  if (spec.idleDelayMs) await sleep(spec.idleDelayMs);
  if (!spec.noIdle) state(spec.fail ? 'prompt_failed' : 'prompt_completed');
}

function onMessage(msg) {
  if (msg.method === undefined) {
    const resolve = pending.get(msg.id);
    if (resolve) {
      pending.delete(msg.id);
      resolve(msg.error ? { error: msg.error } : { result: msg.result });
    }
    return;
  }
  log('recv', { method: msg.method, params: msg.params });
  switch (msg.method) {
    case 'session/create':
      if (scenario.createError) send({ id: msg.id, error: scenario.createError });
      else send({ id: msg.id, result: { session: { sessionId, status: 'idle' } } });
      break;
    case 'session/subscribe':
      send({ id: msg.id, result: { subscribed: true } });
      break;
    case 'session/send': {
      if (sendErrors.length) {
        const error = sendErrors.shift();
        log('send_error', { error });
        send({ id: msg.id, error });
        break;
      }
      if (rejectSends > 0 || locked) {
        rejectSends = Math.max(0, rejectSends - 1);
        log('send_rejected', { locked });
        send({ id: msg.id, error: { code: -32010, message: 'A prompt is already running' } });
        break;
      }
      sendCount += 1;
      locked = true;
      const spec = turns[Math.min(sendCount - 1, turns.length - 1)];
      log('prompt', { n: sendCount, text: msg.params?.content, sessionId: msg.params?.sessionId });
      send({ id: msg.id, result: { accepted: true, stateRevision: sendCount } });
      runTurn(spec, sendCount);
      break;
    }
    case 'session/compact':
      send({ id: msg.id, result: { compact: { state: 'accepted' } } });
      setTimeout(() => state('session_compacted'), 20);
      break;
    case 'session/stop':
      send({ id: msg.id, result: {} });
      break;
    default:
      send({ id: msg.id, error: { code: -32601, message: 'unknown' } });
  }
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let i;
  while ((i = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, i);
    buffer = buffer.slice(i + 1);
    if (line.trim()) onMessage(JSON.parse(line));
  }
});
process.stdin.on('end', () => {
  log('stdin_end');
  if (scenario.ignoreStdinEnd) setInterval(() => {}, 1 << 30);
  else process.exit(0);
});
