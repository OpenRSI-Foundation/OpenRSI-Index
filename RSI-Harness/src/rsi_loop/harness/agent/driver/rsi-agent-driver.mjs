#!/usr/bin/env node
// rsi-agent-driver: keep one coding-agent CLI process working in one session.
//
// The driver starts the CLI as a child speaking ACP v1 or the ZCode Protocol
// over stdio, sends the task prompt, and at every turn end sends the
// continuation reason into the SAME session. It never relaunches the CLI: when
// the child exits, the driver exits with the child's code.
//
// stdout carries only normalized, Claude-Code-stream-json-compatible JSON
// lines, because the harness stores it as agent_output.txt and the visualizer,
// redaction and Judge trajectory already read that format. The harness merges
// stderr into the same stream, so the driver never writes to stderr itself.
//
// Single file, Node >= 22, no dependencies: it is baked into the Work image.

import { spawn } from 'node:child_process';
import { closeSync, globSync, openSync, readFileSync, readSync, statSync } from 'node:fs';
import { constants as osConstants } from 'node:os';

const DRIVER_VERSION = '1.0.0';
const EXIT_SETUP = 64;
// sysexits EX_SOFTWARE: the CLI ended although nothing asked it to.
const EXIT_AGENT_ENDED = 70;
const TOOL_RESULT_CAP = 64 * 1024;
const STDERR_LINE_CAP = 4 * 1024;
const MAX_FRAME = 64 * 1024 * 1024;
const STOP_HOOK_PREFIX = 'Stop hook feedback:\n';
const ZCODE_IDLE_REASONS = new Set([
  'prompt_completed',
  'prompt_failed',
  'session_compacted',
  'session_compact_failed',
  'session_compact_cancelled',
]);
// Tools that need a desktop host service a headless run does not provide.
// --zcode-tool-deny adds the Agent's own.
const ZCODE_TOOL_DENYLIST = ['CronCreate', 'CronDelete', 'OffPeakCreate'];
const ZCODE_HOST_REPLIES = {
  'interaction/requestPermission': { decision: 'allow' },
  'interaction/requestUserInput': {
    action: 'decline',
    reason: 'Unattended run: no user is available. Decide yourself and continue.',
  },
  'session/requestRuntimePreferences': {
    nativeSearchEnhancementsEnabled: false,
    memoryEnabled: false,
    askUserQuestionAutoResolutionEnabled: true,
  },
  'subagents/readRuntimeConfig': { kind: 'built-in-fallback' },
  'interaction/requestProviderRuntimeHeaders': { headersApplied: false },
};
const ACP_TOOL_NAMES = {
  execute: 'Bash',
  read: 'Read',
  edit: 'Edit',
  search: 'Grep',
  fetch: 'WebFetch',
};

// Production timings. Tests shorten them through RSI_AGENT_DRIVER_TIMINGS
// (a JSON object with the same keys); nothing else reads that variable.
const TIMINGS = {
  backoffBaseMs: 5_000,
  backoffMaxMs: 300_000,
  // A turn this short with no content is an error for backoff purposes: some
  // CLIs map a failed model call to a normal end_turn.
  fastTurnMs: 2_000,
  // An empty reply this fast means the prompt was queued behind a turn the
  // agent started itself (queued-prompt guard).
  queuedTurnMs: 1_000,
  queuedWaitMs: 30_000,
  idleQuietMs: 3_000,
  idlePollMs: 1_000,
  setupTimeoutMs: 600_000,
  zcodeWatchdogMs: 60_000,
  zcodeSendTimeoutMs: 60_000,
  zcodeRetryMs: 1_000,
  // After a signal: how long the CLI gets to wind the turn down before it is
  // sent SIGTERM. With killGraceMs this stays inside the harness's 10 s.
  cancelGraceMs: 1_000,
  killGraceMs: 5_000,
  stderrBurst: 50,
  stderrWindowMs: 10_000,
};

const now = () => Date.now();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

// ---------------------------------------------------------------------------
// Output: normalized transcript on stdout
// ---------------------------------------------------------------------------

process.stdout.on('error', () => {});

function writeLine(obj) {
  try {
    process.stdout.write(`${JSON.stringify(obj)}\n`);
  } catch {
    // A closed stdout must not take the session down with it.
  }
}

function capText(text, cap) {
  if (text.length <= cap) return text;
  const half = Math.floor(cap / 2);
  const omitted = text.length - 2 * half;
  return `${text.slice(0, half)}\n[... ${omitted} characters truncated ...]\n${text.slice(-half)}`;
}

class Transcript {
  constructor({ agentName, agentVersion, model, protocol }) {
    this.agentName = agentName;
    this.agentVersion = agentVersion;
    this.model = model;
    this.protocol = protocol;
    this.sessionId = null;
    this.blocks = [];
    this.closed = false;
    this.initWritten = false;
    // stderr seen before init is held back so that the init line stays near
    // the top of the file, where the run scanner looks for the model.
    this.earlyStderr = [];
    this.stderrWindowStart = now();
    this.stderrInWindow = 0;
    this.stderrSuppressed = 0;
  }

  emit(obj) {
    if (!this.closed) writeLine(obj);
  }

  init(extra) {
    this.emit({
      type: 'system',
      subtype: 'init',
      agent: this.agentName,
      agent_version: this.agentVersion,
      model: this.model,
      session_id: this.sessionId,
      cwd: process.cwd(),
      permissionMode: 'bypassPermissions',
      rsi_driver: this.protocol,
      rsi_driver_version: DRIVER_VERSION,
      ...extra,
    });
    this.initWritten = true;
    this.releaseEarlyStderr();
  }

  user(text, continuation) {
    this.flushAssistant();
    const event = { type: 'user', timestamp: new Date().toISOString(), session_id: this.sessionId };
    if (continuation) event.rsi_continuation = continuation;
    // Marked like a Claude Stop-hook message so the visualizer shows it as a
    // continuation marker; the model itself receives the bare reason.
    const shown = continuation ? `${STOP_HOOK_PREFIX}${text}` : text;
    event.message = { role: 'user', content: [{ type: 'text', text: shown }] };
    this.emit(event);
  }

  text(delta) {
    this.appendBlock('text', 'text', delta);
  }

  thinking(delta) {
    this.appendBlock('thinking', 'thinking', delta);
  }

  appendBlock(type, field, delta) {
    if (typeof delta !== 'string' || delta === '') return;
    const last = this.blocks.at(-1);
    if (last && last.type === type) last[field] += delta;
    else this.blocks.push({ type, [field]: delta });
  }

  flushAssistant(extra = []) {
    const content = [...this.blocks, ...extra];
    this.blocks = [];
    if (content.length === 0) return;
    this.emit({
      type: 'assistant',
      timestamp: new Date().toISOString(),
      session_id: this.sessionId,
      message: { role: 'assistant', model: this.model, content },
    });
  }

  toolUse(id, name, input) {
    this.flushAssistant([{ type: 'tool_use', id, name, input }]);
  }

  toolResult(id, content, isError) {
    this.flushAssistant();
    this.emit({
      type: 'user',
      timestamp: new Date().toISOString(),
      session_id: this.sessionId,
      message: {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: id,
            content: capText(String(content ?? ''), TOOL_RESULT_CAP),
            is_error: Boolean(isError),
          },
        ],
      },
    });
  }

  result(subtype, fields) {
    this.flushAssistant();
    this.emit({ type: 'result', subtype, session_id: this.sessionId, ...fields });
  }

  notice(text) {
    this.emit({ type: 'system', subtype: 'driver_notice', text });
  }

  stderr(line) {
    if (this.closed) return;
    const text = capText(line, STDERR_LINE_CAP);
    if (!this.initWritten) {
      if (this.earlyStderr.length < TIMINGS.stderrBurst) this.earlyStderr.push(text);
      else this.stderrSuppressed += 1;
      return;
    }
    const t = now();
    if (t - this.stderrWindowStart >= TIMINGS.stderrWindowMs) {
      this.reportSuppressed();
      this.stderrWindowStart = t;
      this.stderrInWindow = 0;
    }
    if (this.stderrInWindow >= TIMINGS.stderrBurst) {
      this.stderrSuppressed += 1;
      return;
    }
    this.stderrInWindow += 1;
    this.emit({ type: 'system', subtype: 'agent_stderr', text });
  }

  reportSuppressed() {
    if (this.stderrSuppressed === 0) return;
    this.emit({ type: 'system', subtype: 'agent_stderr', suppressed: this.stderrSuppressed });
    this.stderrSuppressed = 0;
  }

  releaseEarlyStderr() {
    const lines = this.earlyStderr;
    this.earlyStderr = [];
    for (const text of lines) this.emit({ type: 'system', subtype: 'agent_stderr', text });
    this.reportSuppressed();
  }
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

class UsageError extends Error {}

const VALUE_FLAGS = new Set([
  'protocol',
  'prompt-file',
  'continuation-reason',
  'acp-mode',
  'permission-prefer',
  'idle-probe',
  'zcode-tool-deny',
  'agent-name',
  'agent-version',
  'model',
]);
const BOOL_FLAGS = new Set(['version']);

function parseArgs(argv) {
  const sep = argv.indexOf('--');
  const own = sep >= 0 ? argv.slice(0, sep) : argv;
  const cli = sep >= 0 ? argv.slice(sep + 1) : [];
  const opts = {};
  for (let i = 0; i < own.length; i += 1) {
    const arg = own[i];
    if (!arg.startsWith('--')) throw new UsageError(`unexpected argument ${JSON.stringify(arg)}`);
    let name = arg.slice(2);
    let value;
    const eq = name.indexOf('=');
    if (eq >= 0) {
      value = name.slice(eq + 1);
      name = name.slice(0, eq);
    }
    if (BOOL_FLAGS.has(name)) {
      if (value !== undefined) throw new UsageError(`--${name} takes no value`);
      opts[name] = true;
    } else if (VALUE_FLAGS.has(name)) {
      if (value === undefined) {
        if (i + 1 >= own.length) throw new UsageError(`--${name} needs a value`);
        i += 1;
        value = own[i];
      }
      opts[name] = value;
    } else {
      throw new UsageError(`unknown option --${name}`);
    }
  }
  return { opts, cli };
}

function loadTimings() {
  const raw = process.env.RSI_AGENT_DRIVER_TIMINGS;
  if (!raw) return;
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new UsageError('RSI_AGENT_DRIVER_TIMINGS is not JSON');
  }
  for (const [key, value] of Object.entries(parsed)) {
    if (!(key in TIMINGS) || typeof value !== 'number' || value < 0) {
      throw new UsageError(`invalid RSI_AGENT_DRIVER_TIMINGS entry ${key}`);
    }
    TIMINGS[key] = value;
  }
}

// ---------------------------------------------------------------------------
// Child process and JSON-RPC framing
// ---------------------------------------------------------------------------

class Rpc {
  constructor(child, { jsonrpc, onRequest, onNotification }) {
    this.child = child;
    this.jsonrpc = jsonrpc;
    this.onRequest = onRequest;
    this.onNotification = onNotification;
    this.nextId = 1;
    this.pending = new Map();
    this.closed = false;
    this.buffer = '';
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk) => this.feed(chunk));
    child.stdout.on('end', () => this.close());
    child.stdin.on('error', () => {});
  }

  feed(chunk) {
    this.buffer += chunk;
    let newline;
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (line) this.dispatch(line);
    }
    // A frame that never ends must not grow memory without bound.
    if (this.buffer.length > MAX_FRAME) this.buffer = '';
  }

  dispatch(line) {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    if (!isObject(msg)) return;
    const hasId = msg.id !== undefined && msg.id !== null;
    if (typeof msg.method === 'string') {
      if (hasId) this.onRequest(msg);
      else this.onNotification(msg);
      return;
    }
    if (hasId && this.pending.has(msg.id)) {
      const entry = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      clearTimeout(entry.timer);
      // Runs before any later frame is dispatched, so it sees the stream
      // exactly as it stood when the response arrived.
      entry.onResponse?.(msg);
      entry.resolve(msg.error !== undefined ? { error: msg.error } : { result: msg.result });
    }
  }

  close() {
    this.closed = true;
    for (const [id, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.resolve({ error: { code: 'closed', message: 'agent closed its output' } });
      this.pending.delete(id);
    }
  }

  send(obj) {
    if (this.child.stdin.destroyed || !this.child.stdin.writable) return;
    const frame = this.jsonrpc ? { jsonrpc: '2.0', ...obj } : obj;
    this.child.stdin.write(`${JSON.stringify(frame)}\n`);
  }

  request(method, params, timeoutMs, onResponse = null) {
    if (this.closed) {
      return Promise.resolve({ error: { code: 'closed', message: 'agent closed its output' } });
    }
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve) => {
      const entry = { resolve, timer: null, onResponse };
      if (timeoutMs) {
        entry.timer = setTimeout(() => {
          this.pending.delete(id);
          resolve({ error: { code: 'timeout', message: `${method} timed out` } });
        }, timeoutMs);
      }
      this.pending.set(id, entry);
      this.send({ id, method, params });
    });
  }

  notify(method, params) {
    this.send({ method, params });
  }

  reply(id, result) {
    this.send({ id, result });
  }

  replyError(id, code, message) {
    this.send({ id, error: { code, message } });
  }
}

// An async queue with an optional timeout on take(); null means it timed out.
class Channel {
  constructor() {
    this.items = [];
    this.waiters = [];
  }

  push(item) {
    const waiter = this.waiters.shift();
    if (waiter) waiter(item);
    else this.items.push(item);
  }

  take(timeoutMs) {
    if (this.items.length) return Promise.resolve(this.items.shift());
    return new Promise((resolve) => {
      let timer = null;
      const waiter = (item) => {
        clearTimeout(timer);
        resolve(item);
      };
      if (timeoutMs !== null && timeoutMs !== undefined) {
        timer = setTimeout(() => {
          const index = this.waiters.indexOf(waiter);
          if (index >= 0) this.waiters.splice(index, 1);
          resolve(null);
        }, timeoutMs);
      }
      this.waiters.push(waiter);
    });
  }
}

// ---------------------------------------------------------------------------
// Driver lifecycle
// ---------------------------------------------------------------------------

class Driver {
  constructor(opts, cli) {
    this.opts = opts;
    this.cli = cli;
    this.protocol = opts.protocol;
    this.continuation = opts['continuation-reason'] ?? null;
    const denied = (opts['zcode-tool-deny'] ?? '').split(',').map((name) => name.trim());
    this.zcodeToolDenylist = [...new Set([...ZCODE_TOOL_DENYLIST, ...denied.filter(Boolean)])];
    this.transcript = new Transcript({
      agentName: opts['agent-name'] ?? null,
      agentVersion: opts['agent-version'] ?? null,
      model: opts.model ?? null,
      protocol: this.protocol,
    });
    this.child = null;
    this.rpc = null;
    this.sessionId = null;
    this.sessionReady = false;
    this.stopping = false;
    this.finishing = false;
    this.expectingExit = false;
    this.childExit = null;
    this.childExited = new Promise((resolve) => {
      this.resolveChildExited = resolve;
    });
    this.turn = 0;
    this.continuations = 0;
    this.errorStreak = 0;
    this.contentSeen = 0;
  }

  // -- process ------------------------------------------------------------

  start() {
    // Same process group, not detached: a group-wide pause or kill reaches
    // the CLI together with the driver.
    this.child = spawn(this.cli[0], this.cli.slice(1), {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: process.env,
    });
    this.child.on('error', (error) => {
      if (this.child.pid !== undefined) return;
      this.childExit = { code: null, signal: null };
      this.resolveChildExited(this.childExit);
      this.fail(EXIT_SETUP, 'setup', { error: `cannot start agent: ${error.message}` });
    });
    let stderrBuffer = '';
    let discarding = false;
    this.child.stderr.setEncoding('utf8');
    this.child.stderr.on('data', (chunk) => {
      stderrBuffer += chunk;
      let newline;
      while ((newline = stderrBuffer.indexOf('\n')) >= 0) {
        const line = stderrBuffer.slice(0, newline).replace(/\r$/, '');
        stderrBuffer = stderrBuffer.slice(newline + 1);
        if (discarding) discarding = false;
        else if (line) this.transcript.stderr(line);
      }
      // Keep only the head of a line that has not ended yet.
      if (!discarding && stderrBuffer.length > STDERR_LINE_CAP) {
        this.transcript.stderr(`${stderrBuffer.slice(0, STDERR_LINE_CAP)}\n[... line truncated ...]`);
        discarding = true;
      }
      if (discarding) stderrBuffer = '';
    });
    this.child.on('exit', (code, signal) => {
      this.childExit = { code, signal };
      this.resolveChildExited(this.childExit);
      // A harness stop signals the driver and the CLI together; that exit
      // belongs to the signal path.
      if (this.finishing || this.expectingExit || this.stopping) return;
      // Let the last frames on stdout be read before the record is closed.
      const settle = new Promise((resolve) => {
        if (this.child.stdout.readableEnded) resolve();
        else this.child.stdout.once('end', resolve);
      });
      Promise.race([settle, sleep(500)]).then(() => {
        if (this.stopping) return;
        const details = { agent_exit_code: code, agent_signal: signal };
        if (!this.sessionReady) {
          this.fail(EXIT_SETUP, 'setup', { ...details, error: 'agent exited during setup' });
          return;
        }
        this.fail(code ? code : EXIT_AGENT_ENDED, 'agent_exited', details);
      });
    });
    this.rpc = new Rpc(this.child, {
      jsonrpc: this.protocol === 'acp',
      onRequest: (msg) => this.onRequest(msg),
      onNotification: (msg) => this.onNotification(msg),
    });
  }

  async stopChild() {
    if (!this.child || this.childExit) return;
    try {
      this.child.kill('SIGTERM');
    } catch {
      // Already gone.
    }
    const exited = await Promise.race([this.childExited, sleep(TIMINGS.killGraceMs).then(() => null)]);
    if (exited) return;
    try {
      this.child.kill('SIGKILL');
    } catch {
      // Already gone.
    }
    await Promise.race([this.childExited, sleep(1_000)]);
  }

  // Single exit point: write driver_exit, stop the child, exit.
  async fail(exitCode, reason, details = {}) {
    if (this.finishing) return;
    this.finishing = true;
    this.stopping = true;
    if (!this.transcript.initWritten) this.transcript.releaseEarlyStderr();
    this.transcript.reportSuppressed();
    this.transcript.result('driver_exit', {
      is_error: exitCode !== 0,
      reason,
      exit_code: exitCode,
      turns: this.turn,
      continuations: this.continuations,
      ...details,
    });
    this.transcript.closed = true;
    await this.stopChild();
    process.exit(exitCode);
  }

  async onSignal(name) {
    if (this.finishing) {
      // A second signal while shutting down: stop waiting.
      try {
        this.child?.kill('SIGKILL');
      } catch {
        // Already gone.
      }
      process.exit(128 + (osConstants.signals[name] ?? 15));
    }
    this.stopping = true;
    if (this.sessionReady && !this.childExit) {
      // Best effort: let the CLI end the turn and save its session first.
      let cancelled;
      if (this.protocol === 'acp') {
        this.rpc.notify('session/cancel', { sessionId: this.sessionId });
        cancelled = this.inflight ?? Promise.resolve();
      } else {
        cancelled = this.rpc.request('session/stop', { sessionId: this.sessionId }, TIMINGS.cancelGraceMs);
      }
      await Promise.race([cancelled, this.childExited, sleep(TIMINGS.cancelGraceMs)]);
    }
    await this.fail(128 + (osConstants.signals[name] ?? 15), 'signal', { signal: name });
  }

  // Never returns: used once the child is gone and finishing is underway.
  halt() {
    return new Promise(() => {});
  }

  // A request failed because the child closed its output: the exit handler
  // owns what happens next.
  agentGone(response) {
    return this.stopping || this.childExit !== null || response?.error?.code === 'closed';
  }

  backoffMs() {
    const exponent = Math.min(this.errorStreak, 30);
    return Math.min(TIMINGS.backoffBaseMs * 2 ** exponent, TIMINGS.backoffMaxMs);
  }

  async backoff() {
    const delay = this.backoffMs();
    this.errorStreak += 1;
    await sleep(delay);
  }

  // Close stdin and let the CLI end by itself (stop hook disabled).
  async endSingleTurn(exitCode) {
    this.expectingExit = true;
    this.child.stdin.end();
    const exited = await Promise.race([this.childExited, sleep(TIMINGS.killGraceMs).then(() => null)]);
    if (!exited) await this.stopChild();
    await this.fail(exitCode, 'completed', {});
  }

  readPrompt() {
    let text;
    try {
      text = readFileSync(this.opts['prompt-file'], 'utf8');
    } catch (error) {
      throw new UsageError(`cannot read prompt file: ${error.message}`);
    }
    // Some CLIs treat a leading "/" as a slash command.
    return text.trimStart().startsWith('/') ? `Task:\n${text}` : text;
  }

  // -- incoming messages --------------------------------------------------

  onRequest(msg) {
    if (this.protocol === 'acp') this.onAcpRequest(msg);
    else this.onZcodeRequest(msg);
  }

  onNotification(msg) {
    const params = isObject(msg.params) ? msg.params : {};
    if (!this.sessionId || params.sessionId !== this.sessionId) return;
    if (this.protocol === 'acp') {
      if (msg.method === 'session/update' && isObject(params.update)) this.onAcpUpdate(params.update);
    } else {
      this.onZcodeNotification(msg.method, params);
    }
  }

  // -- ACP ----------------------------------------------------------------

  onAcpRequest(msg) {
    if (msg.method !== 'session/request_permission') {
      this.rpc.replyError(msg.id, -32601, `method not supported by rsi-agent-driver: ${msg.method}`);
      return;
    }
    const options = Array.isArray(msg.params?.options) ? msg.params.options : [];
    const prefer = (this.opts['permission-prefer'] ?? 'allow_always,allow_once')
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean);
    let picked = null;
    for (const wanted of prefer) {
      picked = options.find((o) => o?.optionId === wanted) ?? options.find((o) => o?.kind === wanted);
      if (picked) break;
    }
    this.rpc.reply(
      msg.id,
      picked
        ? { outcome: { outcome: 'selected', optionId: picked.optionId } }
        : { outcome: { outcome: 'cancelled' } },
    );
  }

  onAcpUpdate(update) {
    const kind = update.sessionUpdate;
    if (kind === 'agent_message_chunk' || kind === 'agent_thought_chunk') {
      const text = update.content?.type === 'text' ? update.content.text : '';
      if (!text) return;
      this.contentSeen += 1;
      if (kind === 'agent_message_chunk') this.transcript.text(text);
      else this.transcript.thinking(text);
    } else if (kind === 'tool_call' || kind === 'tool_call_update') {
      this.contentSeen += 1;
      this.onAcpTool(update);
    } else if (kind === 'plan') {
      this.contentSeen += 1;
    } else if (kind === 'usage_update') {
      this.usage = { used: update.used, size: update.size };
    }
  }

  onAcpTool(update) {
    const id = update.toolCallId;
    if (typeof id !== 'string' && typeof id !== 'number') return;
    let tool = this.tools.get(id);
    if (!tool) {
      tool = { id: String(id), title: '', kind: '', status: 'pending', emitted: false, done: false };
      this.tools.set(id, tool);
    }
    if (typeof update.title === 'string' && update.title) tool.title = update.title;
    if (typeof update.kind === 'string' && update.kind) tool.kind = update.kind;
    if (typeof update.status === 'string' && update.status) tool.status = update.status;
    if (Array.isArray(update.locations) && update.locations.length) tool.locations = update.locations;
    if (Array.isArray(update.content)) tool.content = update.content;
    if ('rawOutput' in update) tool.rawOutput = update.rawOutput;
    if (isObject(update.rawInput) && Object.keys(update.rawInput).length) {
      tool.rawInput = { ...(tool.rawInput ?? {}), ...update.rawInput };
    } else if (update.rawInput !== undefined && !isObject(update.rawInput)) {
      tool.rawInput = update.rawInput;
    }
    if (tool.status !== 'pending' && !tool.emitted) this.emitAcpToolUse(tool);
    if ((tool.status === 'completed' || tool.status === 'failed') && !tool.done) {
      tool.done = true;
      this.transcript.toolResult(tool.id, acpToolOutput(tool), tool.status === 'failed');
      this.tools.delete(id);
    }
  }

  emitAcpToolUse(tool) {
    tool.emitted = true;
    const name = ACP_TOOL_NAMES[tool.kind] ?? (firstLine(tool.title) || tool.kind || 'tool');
    this.transcript.toolUse(tool.id, name, acpToolInput(tool, name));
  }

  async runAcp() {
    const init = await this.rpc.request(
      'initialize',
      {
        protocolVersion: 1,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: 'rsi-agent-driver', version: DRIVER_VERSION },
      },
      TIMINGS.setupTimeoutMs,
    );
    if (init.error) return this.setupFailed('initialize', init.error);
    const agentInfo = isObject(init.result?.agentInfo) ? init.result.agentInfo : {};
    this.transcript.agentVersion ??= agentInfo.version ?? null;
    this.transcript.agentName ??= agentInfo.name ?? null;
    const created = await this.rpc.request(
      'session/new',
      { cwd: process.cwd(), mcpServers: [] },
      TIMINGS.setupTimeoutMs,
    );
    const sessionId = created.result?.sessionId;
    if (created.error || typeof sessionId !== 'string') {
      return this.setupFailed('session/new', created.error ?? { message: 'no sessionId' });
    }
    this.sessionId = sessionId;
    this.transcript.sessionId = sessionId;
    this.tools = new Map();
    let modeNotice = null;
    if (this.opts['acp-mode']) {
      const mode = await this.rpc.request(
        'session/set_mode',
        { sessionId, modeId: this.opts['acp-mode'] },
        TIMINGS.setupTimeoutMs,
      );
      if (mode.error) modeNotice = `session/set_mode ${this.opts['acp-mode']} failed: ${errorText(mode.error)}`;
    }
    this.sessionReady = true;
    this.transcript.init({});
    if (modeNotice) this.transcript.notice(modeNotice);

    let text = this.prompt;
    for (;;) {
      if (this.stopping) return this.halt();
      this.turn += 1;
      this.transcript.user(text, this.turn > 1 ? this.continuations : 0);
      this.contentSeen = 0;
      this.usage = undefined;
      const started = now();
      this.inflight = this.rpc.request('session/prompt', {
        sessionId,
        prompt: [{ type: 'text', text }],
      });
      const response = await this.inflight;
      this.inflight = null;
      if (this.agentGone(response)) return this.halt();
      this.flushPendingTools();
      const duration = now() - started;
      const content = this.contentSeen > 0;
      let failed = false;
      let queued = false;
      if (response.error) {
        failed = true;
        this.transcript.result('turn_error', {
          is_error: true,
          turn: this.turn,
          duration_ms: duration,
          error: response.error,
        });
      } else {
        const stopReason = response.result?.stopReason ?? null;
        const usage = response.result?.usage ?? this.usage;
        queued = this.queuedGuard() && !content && duration < TIMINGS.queuedTurnMs;
        if (!queued && !content && duration < TIMINGS.fastTurnMs) {
          failed = true;
          this.transcript.result('turn_error', {
            is_error: true,
            turn: this.turn,
            stop_reason: stopReason,
            duration_ms: duration,
            error: { code: 'empty_turn', message: `turn ended after ${duration} ms without output` },
          });
        } else {
          this.transcript.result('turn_end', {
            is_error: false,
            turn: this.turn,
            stop_reason: stopReason,
            duration_ms: duration,
            ...(queued ? { rsi_queued: true } : {}),
            ...(usage !== undefined ? { usage } : {}),
          });
        }
      }
      if (this.continuation === null) return this.endSingleTurn(failed ? 1 : 0);
      if (queued) await this.waitAgentIdle();
      else if (failed) await this.backoff();
      else if (content) this.errorStreak = 0;
      this.continuations += 1;
      text = this.continuation;
    }
  }

  flushPendingTools() {
    for (const tool of this.tools.values()) {
      if (!tool.emitted) this.emitAcpToolUse(tool);
    }
    this.transcript.flushAssistant();
  }

  queuedGuard() {
    return Boolean(this.opts['idle-probe']);
  }

  // The prompt was queued behind a turn the agent started itself. Sending
  // more would only grow that queue, so wait until the agent is idle.
  async waitAgentIdle() {
    const probe = this.opts['idle-probe'];
    for (;;) {
      if (this.stopping) return;
      const state = kimiWireState(probe, this.sessionId);
      if (state === null) {
        // No readable wire file: fall back to a fixed wait.
        await sleep(TIMINGS.queuedWaitMs);
        return;
      }
      if (!state.busy && state.quietMs >= TIMINGS.idleQuietMs) return;
      await sleep(TIMINGS.idlePollMs);
    }
  }

  // -- ZCode Protocol -----------------------------------------------------

  onZcodeRequest(msg) {
    const reply = ZCODE_HOST_REPLIES[msg.method];
    if (reply) this.rpc.reply(msg.id, reply);
    else this.rpc.replyError(msg.id, -32601, `${msg.method} is not supported by a headless host`);
  }

  onZcodeNotification(method, params) {
    if (method === 'state.updated') {
      if (ZCODE_IDLE_REASONS.has(params.reason)) this.zEvents.push({ kind: 'idle', reason: params.reason });
      return;
    }
    if (method !== 'session/event') return;
    const payload = isObject(params.payload) ? params.payload : {};
    switch (params.type) {
      case 'turn.started':
        this.zEvents.push({ kind: 'started' });
        break;
      case 'turn.completed':
        this.zUsage = payload.usage;
        this.zEvents.push({ kind: 'completed' });
        break;
      case 'turn.failed':
        this.zEvents.push({ kind: 'failed', error: isObject(payload.error) ? payload.error : {} });
        break;
      case 'model.streaming':
        this.onZcodeStreaming(payload);
        break;
      case 'tool.updated':
        this.onZcodeTool(payload);
        break;
      default:
        break;
    }
  }

  onZcodeStreaming(payload) {
    const kind = String(payload.kind ?? '');
    if (kind === 'text_delta') {
      if (payload.delta) this.contentSeen += 1;
      this.transcript.text(payload.delta);
    } else if (/reason|think/.test(kind)) {
      if (payload.delta) this.contentSeen += 1;
      this.transcript.thinking(payload.delta);
    } else if (kind === 'tool_call') {
      this.contentSeen += 1;
      const input = isObject(payload.input) ? payload.input : { value: payload.input };
      this.transcript.toolUse(String(payload.toolCallId ?? ''), String(payload.toolName ?? 'tool'), input);
    }
  }

  onZcodeTool(payload) {
    const kind = payload.kind;
    if (!['result', 'failed', 'error', 'cancelled'].includes(kind)) return;
    this.contentSeen += 1;
    const result = isObject(payload.result) ? payload.result : {};
    let content = result.content ?? result.error ?? payload.error ?? '';
    if (typeof content !== 'string') content = JSON.stringify(content);
    const isError = kind !== 'result' || result.success === false;
    this.transcript.toolResult(String(payload.toolCallId ?? ''), content, isError);
  }

  // Events still queued belong to an earlier turn whose idle signal came
  // after the watchdog (or after a send that timed out but was accepted).
  // Dropping them keeps a later turn from ending on its predecessor's idle
  // signal.
  zcodeDropStaleEvents() {
    this.zEvents.items.length = 0;
  }

  async zcodeSend(text) {
    for (;;) {
      if (this.stopping || this.childExit) return this.halt();
      this.zcodeDropStaleEvents();
      const response = await this.rpc.request(
        'session/send',
        { sessionId: this.sessionId, content: text, toolDenylist: this.zcodeToolDenylist },
        TIMINGS.zcodeSendTimeoutMs,
        // Whatever arrived before the acceptance belongs to earlier turns;
        // this turn's events can only follow it.
        (msg) => {
          if (msg.result?.accepted) this.zcodeDropStaleEvents();
        },
      );
      if (!response.error && response.result?.accepted) return true;
      if (this.agentGone(response)) return this.halt();
      const code = response.error?.code;
      // -32010: the previous prompt still holds the session lock; timeouts
      // are retried the same way.
      if (code === -32010 || code === 'timeout' || (!response.error && !response.result?.accepted)) {
        await sleep(TIMINGS.zcodeRetryMs);
        continue;
      }
      this.transcript.result('turn_error', {
        is_error: true,
        turn: this.turn,
        error: response.error,
      });
      await this.backoff();
    }
  }

  // Wait for the idle state.updated that follows the turn. If the turn ended
  // but no idle signal arrives, send anyway after the watchdog: a premature
  // send is answered -32010 and retried.
  //
  // ZCode releases the session lock before it broadcasts the idle signal, so
  // a send can be accepted in between and the previous turn's idle signal
  // then arrives during this turn. For a prompt turn an idle signal
  // therefore only counts after this turn's own turn.completed/turn.failed.
  // One that comes before turn.started is held: if no turn.started follows
  // within the watchdog, the prompt ended without starting a turn and that
  // signal was its own. A compaction emits no turn events, so any idle signal ends
  // it.
  async zcodeWaitIdle({ turn = true } = {}) {
    const outcome = { failed: false, error: null, endedAt: null, watchdog: false, reason: null, idleAt: null };
    let turnEnded = !turn;
    let turnStarted = false;
    let heldIdle = null;
    for (;;) {
      const since = outcome.endedAt ?? heldIdle?.at ?? null;
      const timeout = since === null ? null : Math.max(0, since + TIMINGS.zcodeWatchdogMs - now());
      const event = await this.zEvents.take(timeout);
      if (event === null) {
        if (heldIdle && outcome.endedAt === null) {
          outcome.reason = heldIdle.reason;
          // The turn ended when its signal came, not when it was confirmed.
          outcome.idleAt = heldIdle.at;
          if (heldIdle.reason === 'prompt_failed') outcome.failed = true;
          return outcome;
        }
        outcome.watchdog = true;
        return outcome;
      }
      if (event.kind === 'started') {
        outcome.endedAt = null;
        turnStarted = true;
        heldIdle = null;
      } else if (event.kind === 'completed') {
        outcome.endedAt = now();
        turnEnded = true;
      } else if (event.kind === 'failed') {
        outcome.failed = true;
        outcome.error = event.error;
        outcome.endedAt = now();
        turnEnded = true;
      } else if (event.kind === 'idle') {
        if (!turnEnded) {
          // Once this turn has started, an idle signal before its end can
          // only be the previous turn's.
          if (!turnStarted) heldIdle ??= { reason: event.reason, at: now() };
          continue;
        }
        outcome.reason = event.reason;
        if (event.reason === 'prompt_failed') outcome.failed = true;
        return outcome;
      }
    }
  }

  async runZcode() {
    const cwd = process.cwd();
    const created = await this.rpc.request(
      'session/create',
      // A generated title is one more model request after the first prompt.
      { workspace: { workspacePath: cwd, workspaceKey: cwd }, mode: 'yolo', titleGenerationEnabled: false },
      TIMINGS.setupTimeoutMs,
    );
    const sessionId = created.result?.session?.sessionId;
    if (created.error || typeof sessionId !== 'string') {
      return this.setupFailed('session/create', created.error ?? { message: 'no sessionId' });
    }
    this.sessionId = sessionId;
    this.transcript.sessionId = sessionId;
    this.zEvents = new Channel();
    const subscribed = await this.rpc.request(
      'session/subscribe',
      { sessionId, deliveryKind: 'desktop-continuous' },
      TIMINGS.setupTimeoutMs,
    );
    if (subscribed.error) return this.setupFailed('session/subscribe', subscribed.error);
    this.sessionReady = true;
    this.transcript.init({});

    let text = this.prompt;
    let failedInRow = 0;
    for (;;) {
      if (this.stopping) return this.halt();
      this.turn += 1;
      this.transcript.user(text, this.turn > 1 ? this.continuations : 0);
      this.contentSeen = 0;
      this.zUsage = undefined;
      const started = now();
      await this.zcodeSend(text);
      const outcome = await this.zcodeWaitIdle();
      if (this.stopping || this.childExit) return this.halt();
      this.transcript.flushAssistant();
      const duration = (outcome.idleAt ?? now()) - started;
      const content = this.contentSeen > 0;
      let failed = outcome.failed;
      if (failed) {
        this.transcript.result('turn_error', {
          is_error: true,
          turn: this.turn,
          duration_ms: duration,
          error: outcome.error ?? { type: outcome.reason ?? 'prompt_failed' },
        });
      } else if (!content && duration < TIMINGS.fastTurnMs) {
        failed = true;
        this.transcript.result('turn_error', {
          is_error: true,
          turn: this.turn,
          duration_ms: duration,
          error: { code: 'empty_turn', message: `turn ended after ${duration} ms without output` },
        });
      } else {
        this.transcript.result('turn_end', {
          is_error: false,
          turn: this.turn,
          stop_reason: outcome.reason ?? 'watchdog',
          duration_ms: duration,
          ...(this.zUsage !== undefined ? { usage: this.zUsage } : {}),
        });
      }
      if (this.continuation === null) return this.endSingleTurn(failed ? 1 : 0);
      failedInRow = outcome.failed ? failedInRow + 1 : 0;
      if (failed) await this.backoff();
      else if (content) this.errorStreak = 0;
      // ZCode's session-lifetime auto-compact breaker can leave a session
      // failing every turn; an explicit compaction resets it.
      const overflow = outcome.error?.type === 'model_context_exceeded';
      if (overflow || (failedInRow > 0 && failedInRow % 3 === 0)) await this.zcodeCompact();
      this.continuations += 1;
      text = this.continuation;
    }
  }

  async zcodeCompact() {
    if (this.stopping || this.childExit) return;
    this.zcodeDropStaleEvents();
    const response = await this.rpc.request(
      'session/compact',
      { sessionId: this.sessionId },
      TIMINGS.zcodeSendTimeoutMs,
    );
    const state = response.result?.compact?.state ?? null;
    this.transcript.emit({
      type: 'system',
      subtype: 'compaction',
      session_id: this.sessionId,
      state: response.error ? 'error' : state,
      ...(response.error ? { error: response.error } : {}),
    });
    if (!response.error && state === 'accepted') await this.zcodeWaitIdle({ turn: false });
  }

  // -- common -------------------------------------------------------------

  setupFailed(step, error) {
    return this.fail(EXIT_SETUP, 'setup', { error: `${step} failed: ${errorText(error)}` });
  }

  async run() {
    for (const name of ['SIGTERM', 'SIGINT', 'SIGHUP']) {
      process.on(name, () => {
        this.onSignal(name);
      });
    }
    this.start();
    if (this.protocol === 'acp') await this.runAcp();
    else await this.runZcode();
  }
}

// ---------------------------------------------------------------------------
// Normalization helpers
// ---------------------------------------------------------------------------

function errorText(error) {
  if (!isObject(error)) return String(error);
  const message = typeof error.message === 'string' ? error.message : JSON.stringify(error);
  return error.code !== undefined ? `${error.code} ${message}` : message;
}

function firstLine(text) {
  return String(text ?? '').split('\n')[0].trim().slice(0, 80);
}

function acpToolInput(tool, name) {
  let input = {};
  if (isObject(tool.rawInput)) input = { ...tool.rawInput };
  else if (tool.rawInput !== undefined) input = { value: tool.rawInput };
  else {
    // Some agents only carry the arguments as JSON text in the content.
    const text = acpContentText(tool.content);
    try {
      const parsed = JSON.parse(text);
      if (isObject(parsed)) input = parsed;
    } catch {
      // Not JSON: no structured input.
    }
  }
  const location = tool.locations?.[0]?.path;
  if (name === 'Bash' && typeof input.command !== 'string') input.command = tool.title;
  if ((name === 'Read' || name === 'Edit') && !input.file_path) {
    const path = input.path ?? input.filePath ?? input.file ?? location;
    if (path) input.file_path = path;
  }
  if (name === 'Grep' && !input.pattern) {
    const pattern = input.query ?? input.regex ?? firstLine(tool.title);
    if (pattern) input.pattern = pattern;
  }
  if (name === 'WebFetch' && !input.url && typeof input.uri === 'string') input.url = input.uri;
  return input;
}

function acpContentText(content) {
  if (!Array.isArray(content)) return '';
  const parts = [];
  for (const item of content) {
    if (!isObject(item)) continue;
    if (item.type === 'content') {
      const block = item.content;
      if (block?.type === 'text') parts.push(String(block.text ?? ''));
      else if (block?.type) parts.push(`[${block.type}]`);
    } else if (item.type === 'diff') {
      parts.push(`diff ${item.path ?? ''}\n${item.newText ?? ''}`);
    } else if (item.type === 'terminal') {
      parts.push(`[terminal ${item.terminalId ?? ''}]`);
    }
  }
  return parts.join('\n');
}

function acpToolOutput(tool) {
  const text = acpContentText(tool.content);
  if (text) return text;
  const raw = tool.rawOutput;
  if (raw === undefined || raw === null) return '';
  if (typeof raw === 'string') return raw;
  if (isObject(raw)) {
    for (const key of ['output', 'text', 'stdout']) {
      if (typeof raw[key] === 'string' && raw[key]) return raw[key];
    }
  }
  return JSON.stringify(raw);
}

// ---------------------------------------------------------------------------
// Kimi idle probe: the newest agent.turn.* event of the main agent in the
// session's wire.jsonl, and how long the file has been quiet.
// ---------------------------------------------------------------------------

function expandEnv(pattern) {
  return pattern
    .replace(/^~(?=\/|$)/, process.env.HOME ?? '~')
    .replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)/g, (_, a, b) => process.env[a ?? b] ?? '');
}

function globFiles(pattern) {
  return globSync(pattern).filter((path) => {
    try {
      return statSync(path).isFile();
    } catch {
      return false;
    }
  });
}

function kimiWireState(probe, sessionId) {
  if (!probe.startsWith('kimi-wire:')) return null;
  const pattern = expandEnv(probe.slice('kimi-wire:'.length)).replaceAll('{session}', sessionId);
  const paths = globFiles(pattern);
  if (!paths.length) return null;
  const path = paths[0];
  let fd;
  try {
    const stat = statSync(path);
    const length = Math.min(stat.size, 4 * 1024 * 1024);
    const buffer = Buffer.alloc(length);
    fd = openSync(path, 'r');
    readSync(fd, buffer, 0, length, stat.size - length);
    let last = null;
    for (const line of buffer.toString('utf8').split('\n')) {
      if (!line.includes('"agent.turn.')) continue;
      try {
        const event = JSON.parse(line);
        if ((event.agentId ?? 'main') !== 'main') continue;
        if (event.type === 'agent.turn.started' || event.type === 'agent.turn.ended') last = event.type;
      } catch {
        // A partial line at the read boundary.
      }
    }
    return { busy: last === 'agent.turn.started', quietMs: now() - stat.mtimeMs };
  } catch {
    return null;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

function setupExit(message) {
  writeLine({
    type: 'result',
    subtype: 'driver_exit',
    is_error: true,
    reason: 'setup',
    exit_code: EXIT_SETUP,
    error: message,
  });
  process.exit(EXIT_SETUP);
}

function printVersion(cli) {
  process.stdout.write(`rsi-agent-driver ${DRIVER_VERSION}\n`);
  if (!cli.length) process.exit(0);
  const child = spawn(cli[0], [...cli.slice(1), '--version'], { stdio: ['ignore', 'inherit', 'inherit'] });
  child.on('error', (error) => {
    process.stdout.write(`cannot run ${cli[0]}: ${error.message}\n`);
    process.exit(EXIT_SETUP);
  });
  child.on('exit', (code, signal) => process.exit(code ?? (signal ? 128 + (osConstants.signals[signal] ?? 0) : 1)));
}

function main() {
  let parsed;
  try {
    parsed = parseArgs(process.argv.slice(2));
    loadTimings();
  } catch (error) {
    return setupExit(error.message);
  }
  const { opts, cli } = parsed;
  if (opts.version) return printVersion(cli);
  if (opts.protocol !== 'acp' && opts.protocol !== 'zcode') return setupExit('--protocol must be acp or zcode');
  if (!opts['prompt-file']) return setupExit('--prompt-file is required');
  if (!cli.length) return setupExit('missing agent command after --');
  if (opts['zcode-tool-deny'] !== undefined && opts.protocol !== 'zcode') {
    return setupExit('--zcode-tool-deny needs --protocol zcode');
  }
  if (opts['idle-probe'] && !opts['idle-probe'].startsWith('kimi-wire:')) {
    return setupExit('--idle-probe must be kimi-wire:GLOB');
  }
  const driver = new Driver(opts, cli);
  try {
    driver.prompt = driver.readPrompt();
  } catch (error) {
    return setupExit(error.message);
  }
  process.on('uncaughtException', (error) => {
    driver.fail(EXIT_AGENT_ENDED, 'internal_error', { error: String(error?.stack ?? error).slice(0, 4096) });
  });
  process.on('unhandledRejection', (error) => {
    driver.fail(EXIT_AGENT_ENDED, 'internal_error', { error: String(error?.stack ?? error).slice(0, 4096) });
  });
  driver.run();
}

main();
