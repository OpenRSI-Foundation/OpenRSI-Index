#!/usr/bin/env node
// Scripted ACP v1 agent for rsi-agent-driver tests. No dependencies.
//
// usage: node fake_acp_agent.mjs SCENARIO.json
//
// Every message it receives or sends of interest is appended to
// scenario.log as one JSON line, so tests assert on what the driver did.
//
// Scenario keys:
//   log           JSONL path (required)
//   initError     {code,message}: fail initialize
//   sessionError  {code,message}: fail session/new
//   ignoreSigterm true: survive SIGTERM (tests the driver's SIGKILL)
//   ignoreStdinEnd true: keep running when stdin closes, like a CLI stuck in
//                 a tool call, so only the driver's signals can end it
//   stderrAtStart [lines] written to stderr before initialize answers
//   model         {url, model}: answer prompts by calling an OpenAI
//                 chat-completions endpoint (tests/fakes/fake_model_server.py)
//   turns         list of turn specs; the last one repeats. A turn spec:
//     stopReason  default "end_turn"
//     error       {code,message}: answer session/prompt with this error
//     thought     agent_thought_chunk text
//     text        agent_message_chunk text (sent in two chunks)
//     tool        {kind,title,rawInput,output,status,lateInput}
//     permission  {options:[...]}: ask session/request_permission first
//     clientRequest  method name: send it to the client and log the reply
//     stderr      [lines] to write to stderr
//     delayMs     wait before answering (a slow turn)
//     hold        true: never answer (until session/cancel)
//     exit        exit with this code instead of answering
//     selfTurnMs  after answering, run an agent-initiated turn for this long;
//                 prompts arriving meanwhile are queued and answered at once
//                 with end_turn and no updates (Kimi's behaviour)
//   wire          path of a Kimi-style wire.jsonl to write agent.turn.* to

import { appendFileSync, readFileSync } from 'node:fs';

const scenario = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const turns = scenario.turns ?? [{}];
const sessionId = 'fake-acp-session-1';
let promptCount = 0;
let nextId = 1;
const pending = new Map();
let activePrompt = null;
let selfTurnUntil = 0;
let queued = 0;
const history = [];

const log = (event, data = {}) =>
  appendFileSync(scenario.log, `${JSON.stringify({ t: Date.now(), pid: process.pid, event, ...data })}\n`);
const send = (obj) => process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', ...obj })}\n`);
const update = (u) => send({ method: 'session/update', params: { sessionId, update: u } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const wire = (type) => {
  if (scenario.wire) appendFileSync(scenario.wire, `${JSON.stringify({ type, agentId: 'main', time: Date.now() })}\n`);
};
const request = (method, params) =>
  new Promise((resolve) => {
    const id = `agent-${nextId++}`;
    pending.set(id, resolve);
    send({ id, method, params });
  });

log('start', { argv: process.argv.slice(2), cwd: process.cwd() });
process.on('SIGTERM', () => {
  log('sigterm');
  if (!scenario.ignoreSigterm) process.exit(143);
});
for (const line of scenario.stderrAtStart ?? []) process.stderr.write(`${line}\n`);

async function runSelfTurn(ms) {
  selfTurnUntil = Date.now() + ms;
  wire('agent.turn.started');
  await sleep(ms);
  selfTurnUntil = 0;
  wire('agent.turn.ended');
  log('self_turn_end', { queued });
  queued = 0;
}

async function callModel(text) {
  history.push({ role: 'user', content: text });
  for (let step = 0; step < 8; step += 1) {
    const response = await fetch(`${scenario.model.url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${process.env.FAKE_API_KEY ?? ''}` },
      body: JSON.stringify({
        model: scenario.model.model ?? 'fake-model',
        stream: true,
        messages: history,
        tools: [{ type: 'function', function: { name: 'bash', parameters: { type: 'object' } } }],
      }),
    });
    if (!response.ok) {
      return { error: { code: -32603, message: `model HTTP ${response.status}` } };
    }
    const body = await response.text();
    let content = '';
    let finish = null;
    const calls = [];
    for (const line of body.split('\n')) {
      if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
      const chunk = JSON.parse(line.slice(6));
      const choice = chunk.choices?.[0] ?? {};
      const delta = choice.delta ?? {};
      if (delta.content) {
        content += delta.content;
        update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: delta.content } });
      }
      if (delta.reasoning_content) {
        update({ sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: delta.reasoning_content } });
      }
      for (const call of delta.tool_calls ?? []) calls.push(call);
      if (choice.finish_reason) finish = choice.finish_reason;
    }
    history.push({ role: 'assistant', content, ...(calls.length ? { tool_calls: calls } : {}) });
    if (!calls.length) return { stopReason: finish === 'length' ? 'max_tokens' : 'end_turn' };
    for (const call of calls) {
      const args = JSON.parse(call.function.arguments || '{}');
      update({ sessionUpdate: 'tool_call', toolCallId: call.id, title: 'bash', kind: 'execute', status: 'pending' });
      update({ sessionUpdate: 'tool_call_update', toolCallId: call.id, status: 'in_progress', rawInput: args });
      const output = `ran: ${args.command}\n`;
      update({
        sessionUpdate: 'tool_call_update',
        toolCallId: call.id,
        status: 'completed',
        content: [{ type: 'content', content: { type: 'text', text: output } }],
      });
      history.push({ role: 'tool', tool_call_id: call.id, content: output });
    }
  }
  return { stopReason: 'max_turn_requests' };
}

async function runTurn(spec, text) {
  for (const line of spec.stderr ?? []) process.stderr.write(`${line}\n`);
  if (spec.exit !== undefined) {
    log('exit', { code: spec.exit });
    process.exit(spec.exit);
  }
  if (spec.permission) {
    const reply = await request('session/request_permission', {
      sessionId,
      toolCall: { toolCallId: 'perm-1', title: 'Permission', kind: 'execute', status: 'pending' },
      options: spec.permission.options,
    });
    log('permission_reply', { reply });
  }
  if (spec.clientRequest) {
    const reply = await request(spec.clientRequest, { sessionId });
    log('client_request_reply', { method: spec.clientRequest, reply });
  }
  if (scenario.model) return callModel(text);
  if (spec.thought) update({ sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: spec.thought } });
  if (spec.text) {
    const half = Math.ceil(spec.text.length / 2);
    update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: spec.text.slice(0, half) } });
    update({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: spec.text.slice(half) } });
  }
  if (spec.tool) {
    const tool = spec.tool;
    const id = `tool-${promptCount}`;
    update({
      sessionUpdate: 'tool_call',
      toolCallId: id,
      title: tool.title ?? 'tool',
      kind: tool.kind ?? 'other',
      status: 'pending',
      rawInput: tool.lateInput ? {} : (tool.rawInput ?? {}),
    });
    update({
      sessionUpdate: 'tool_call_update',
      toolCallId: id,
      status: 'in_progress',
      ...(tool.lateInput ? { rawInput: tool.rawInput } : {}),
    });
    let output = tool.output ?? '';
    if (tool.outputEnv) output += process.env[tool.outputEnv] ?? '';
    if (tool.outputRepeat) output = output.repeat(tool.outputRepeat);
    update({
      sessionUpdate: 'tool_call_update',
      toolCallId: id,
      status: tool.status ?? 'completed',
      content: [{ type: 'content', content: { type: 'text', text: output } }],
    });
  }
  if (spec.hold) return new Promise(() => {});
  if (spec.delayMs) await sleep(spec.delayMs);
  if (spec.error) return { error: spec.error };
  return { stopReason: spec.stopReason ?? 'end_turn' };
}

async function onPrompt(msg) {
  const text = msg.params?.prompt?.[0]?.text ?? '';
  if (selfTurnUntil > Date.now()) {
    queued += 1;
    log('prompt', { n: promptCount, text, sessionId: msg.params?.sessionId, queued: true, queueLength: queued });
    send({ id: msg.id, result: { stopReason: 'end_turn' } });
    return;
  }
  promptCount += 1;
  if (activePrompt !== null) log('concurrent_prompt', { n: promptCount });
  activePrompt = msg.id;
  const spec = turns[Math.min(promptCount - 1, turns.length - 1)];
  log('prompt', { n: promptCount, text, sessionId: msg.params?.sessionId });
  const outcome = await runTurn(spec, text);
  if (activePrompt !== msg.id) return;
  activePrompt = null;
  if (outcome.error) send({ id: msg.id, error: outcome.error });
  else send({ id: msg.id, result: { stopReason: outcome.stopReason } });
  if (spec.selfTurnMs) runSelfTurn(spec.selfTurnMs);
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
    case 'initialize':
      if (scenario.initError) send({ id: msg.id, error: scenario.initError });
      else
        send({
          id: msg.id,
          result: { protocolVersion: 1, agentInfo: { name: 'fake-acp', version: '0.0.1' }, agentCapabilities: {} },
        });
      break;
    case 'session/new':
      if (scenario.sessionError) send({ id: msg.id, error: scenario.sessionError });
      else send({ id: msg.id, result: { sessionId } });
      break;
    case 'session/set_mode':
      send({ id: msg.id, result: {} });
      break;
    case 'session/prompt':
      onPrompt(msg);
      break;
    case 'session/cancel':
      if (activePrompt !== null) {
        send({ id: activePrompt, result: { stopReason: 'cancelled' } });
        activePrompt = null;
      }
      break;
    default:
      if (msg.id !== undefined) send({ id: msg.id, error: { code: -32601, message: 'unknown' } });
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
