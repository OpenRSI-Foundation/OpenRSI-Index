// rsi-never-stop: the pi Agent's stop hook and transcript, as one pi extension.
//
// Baked into the pi Agent's Work image and loaded with `pi -e`. pi (1.1.0) has
// an uncapped native stop point, the `agent_before_settle` boundary: when
// RSI_CONTINUATION_REASON is set (only while the run's stop hook is enabled),
// every settle queues that reason as a real follow-up user message, so the
// same pi process keeps working in the same session. A user message, unlike a
// custom_message entry, also resets pi's one-shot context-overflow recovery;
// without that, one failed overflow compaction left every later request
// overflowing for the rest of the run.
//
// stdout carries only normalized, Claude-Code-stream-json-compatible JSON
// lines, the same format rsi-agent-driver writes, because the harness stores
// it as agent_output.txt and the visualizer, redaction and Judge trajectory
// read that format. pi's print mode sends ordinary stdout writes to stderr,
// and an asynchronous write still pending when pi exits is lost, so each line
// is written to file descriptor 1 synchronously.
//
// A handler that throws ends pi's run, so every handler catches everything.
// Only Node built-ins are imported: pi loads this file as is.

import { writeSync } from "node:fs";

const AGENT_NAME = "pi";
const STOP_HOOK_PREFIX = "Stop hook feedback:\n";
const TOOL_RESULT_CAP = 64 * 1024;
// pi's built-in tools under the names the visualizer knows.
const TOOL_NAMES: Record<string, string> = {
	bash: "Bash",
	read: "Read",
	edit: "Edit",
	write: "Write",
	grep: "Grep",
	find: "Glob",
	ls: "LS",
};
const FILE_TOOLS = new Set(["Read", "Edit", "Write"]);

// Production timings. Tests shorten them through RSI_AGENT_DRIVER_TIMINGS, the
// knob rsi-agent-driver reads too; nothing else sets it.
const TIMINGS = {
	backoffBaseMs: 5_000,
	backoffMaxMs: 300_000,
	// A turn this short with no content counts as failed for the backoff, so
	// an endpoint that keeps answering with nothing is not called in a loop.
	fastTurnMs: 2_000,
};
try {
	const overrides = JSON.parse(process.env.RSI_AGENT_DRIVER_TIMINGS || "{}");
	for (const key of Object.keys(TIMINGS) as (keyof typeof TIMINGS)[]) {
		if (Number.isFinite(overrides?.[key]) && overrides[key] >= 0) TIMINGS[key] = overrides[key];
	}
} catch {
	// Keep the production timings.
}

const REASON = process.env.RSI_CONTINUATION_REASON || null;
const AGENT_VERSION = process.env.RSI_AGENT_VERSION || null;

const STDOUT_FD = 1;
const pause = new Int32Array(new SharedArrayBuffer(4));

function emit(event: Record<string, unknown>): void {
	try {
		const data = Buffer.from(`${JSON.stringify(event)}\n`);
		let offset = 0;
		while (offset < data.length) {
			try {
				offset += writeSync(STDOUT_FD, data, offset);
			} catch (error: any) {
				// A non-blocking pipe that is full: wait for the reader.
				if (error?.code !== "EAGAIN") throw error;
				Atomics.wait(pause, 0, 0, 1);
			}
		}
	} catch {
		// A closed stdout must not take the session down with it.
	}
}

function capText(text: string): string {
	if (text.length <= TOOL_RESULT_CAP) return text;
	const half = Math.floor(TOOL_RESULT_CAP / 2);
	const omitted = text.length - 2 * half;
	return `${text.slice(0, half)}\n[... ${omitted} characters truncated ...]\n${text.slice(-half)}`;
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((part: any) => (part?.type === "text" ? String(part.text ?? "") : part?.type ? `[${part.type}]` : ""))
		.join("");
}

function toolInput(name: string, args: unknown): Record<string, unknown> {
	const input: Record<string, unknown> =
		args !== null && typeof args === "object" && !Array.isArray(args) ? { ...(args as object) } : { value: args };
	if (FILE_TOOLS.has(name) && input.file_path === undefined && typeof input.path === "string") {
		input.file_path = input.path;
	}
	return input;
}

function usageOf(usage: any): Record<string, number> | undefined {
	if (!usage || typeof usage !== "object") return undefined;
	return {
		input_tokens: usage.input ?? 0,
		output_tokens: usage.output ?? 0,
		cache_read_input_tokens: usage.cacheRead ?? 0,
		cache_creation_input_tokens: usage.cacheWrite ?? 0,
	};
}

const now = () => Date.now();
const timestamp = () => new Date().toISOString();
const tick = () => new Promise((resolve) => setImmediate(resolve));
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function (pi: any) {
	let sessionId: string | null = null;
	let model: string | null = null;
	let turn = 0;
	let continuations = 0;
	// Reasons queued but not yet sent, to tell them from a user message.
	let queued = 0;
	let turnStarted = now();
	let turnContent = false;
	let lastAssistant: any = null;
	let errorStreak = 0;
	let exitWritten = false;

	const base = () => ({ session_id: sessionId });

	const onExit = (code: number) => {
		if (exitWritten) return;
		exitWritten = true;
		// pi exits 143 or 129 after SIGTERM or SIGHUP, 0 when the run settled.
		const reason = code === 143 || code === 129 ? "signal" : code === 0 ? "completed" : "agent_error";
		emit({
			type: "result",
			subtype: "agent_exit",
			...base(),
			is_error: code !== 0,
			reason,
			exit_code: code,
			turns: turn,
			continuations,
		});
	};

	pi.on("session_start", (event: any, ctx: any) => {
		// Only once a session runs: pi also loads extensions for commands that
		// start none.
		if (event?.reason === "startup") process.on("exit", onExit);
		try {
			sessionId = ctx.sessionManager?.getSessionId?.() ?? null;
			model = ctx.model?.id ?? null;
			if (event?.reason !== "startup") return;
			let tools: string[] = [];
			try {
				tools = (pi.getActiveTools?.() ?? []).map((name: string) => TOOL_NAMES[name] ?? name);
			} catch {
				// Leave the tool list out.
			}
			emit({
				type: "system",
				subtype: "init",
				agent: AGENT_NAME,
				agent_version: AGENT_VERSION,
				model,
				...base(),
				cwd: ctx.cwd ?? process.cwd(),
				tools,
				thinking_level: ctx.thinkingLevel ?? null,
				permissionMode: "bypassPermissions",
				rsi_extension: "rsi-never-stop",
				rsi_stop_hook: REASON !== null,
			});
		} catch {
			// Transcript only.
		}
	});

	pi.on("message_end", (event: any) => {
		try {
			const message = event?.message;
			if (!message) return;
			if (message.role === "user") {
				const text = textOf(message.content);
				// A user message starts the next model request: the turn starts here.
				turnStarted = now();
				turnContent = false;
				const record: Record<string, unknown> = { type: "user", timestamp: timestamp(), ...base() };
				if (queued > 0 && text === REASON) {
					queued -= 1;
					continuations += 1;
					record.rsi_continuation = continuations;
					// Marked like a Claude Stop-hook message so the visualizer shows it
					// as a continuation marker; the model itself receives the bare reason.
					record.message = { role: "user", content: [{ type: "text", text: `${STOP_HOOK_PREFIX}${text}` }] };
				} else {
					record.message = { role: "user", content: [{ type: "text", text }] };
				}
				emit(record);
				return;
			}
			if (message.role === "assistant") {
				lastAssistant = message;
				const content: any[] = [];
				for (const block of Array.isArray(message.content) ? message.content : []) {
					if (block?.type === "text" && block.text) content.push({ type: "text", text: block.text });
					else if (block?.type === "thinking" && block.thinking) {
						content.push({ type: "thinking", thinking: block.thinking });
					} else if (block?.type === "toolCall") {
						const name = TOOL_NAMES[block.name] ?? String(block.name ?? "tool");
						content.push({ type: "tool_use", id: String(block.id ?? ""), name, input: toolInput(name, block.arguments) });
					}
				}
				if (content.length > 0) {
					turnContent = true;
					emit({
						type: "assistant",
						timestamp: timestamp(),
						...base(),
						message: {
							role: "assistant",
							model: message.model ?? model,
							content,
							stop_reason: message.stopReason ?? null,
							usage: usageOf(message.usage),
						},
					});
				}
				if (message.stopReason === "error") {
					emit({ type: "system", subtype: "model_error", ...base(), text: String(message.errorMessage ?? "") });
				}
				return;
			}
			if (message.role === "toolResult") {
				emit({
					type: "user",
					timestamp: timestamp(),
					...base(),
					message: {
						role: "user",
						content: [
							{
								type: "tool_result",
								tool_use_id: String(message.toolCallId ?? ""),
								content: capText(textOf(message.content)),
								is_error: Boolean(message.isError),
							},
						],
					},
				});
			}
		} catch {
			// Transcript only.
		}
	});

	for (const name of ["session_compact", "session_compact_failed"]) {
		pi.on(name, (event: any) => {
			try {
				emit({
					type: "system",
					subtype: "compaction",
					...base(),
					state: name === "session_compact" ? "completed" : event?.aborted ? "aborted" : "failed",
					reason: event?.reason ?? null,
					...(event?.errorMessage ? { error: String(event.errorMessage) } : {}),
				});
			} catch {
				// Transcript only.
			}
		});
	}

	pi.on("agent_before_settle", async (event: any, ctx: any) => {
		let failed = true;
		try {
			turn += 1;
			const outcome = event?.outcome ?? "completed";
			const duration = now() - turnStarted;
			const stopReason = lastAssistant?.stopReason ?? null;
			failed = outcome !== "completed" || (!turnContent && duration < TIMINGS.fastTurnMs);
			if (outcome === "error" || failed) {
				emit({
					type: "result",
					subtype: "turn_error",
					...base(),
					is_error: true,
					turn,
					outcome,
					stop_reason: stopReason,
					duration_ms: duration,
					error: {
						code: outcome === "completed" ? "empty_turn" : outcome,
						message: String(lastAssistant?.errorMessage ?? `turn ended after ${duration} ms without output`),
					},
				});
			} else {
				emit({
					type: "result",
					subtype: "turn_end",
					...base(),
					is_error: false,
					turn,
					stop_reason: stopReason,
					duration_ms: duration,
					usage: usageOf(lastAssistant?.usage),
				});
			}
		} catch {
			// Transcript only; the continuation below must still happen.
		}
		if (REASON === null) return undefined;
		try {
			errorStreak = failed ? errorStreak + 1 : 0;
			if (errorStreak > 0 && TIMINGS.backoffBaseMs > 0) {
				const exponent = Math.min(errorStreak - 1, 16);
				await sleep(Math.min(TIMINGS.backoffBaseMs * 2 ** exponent, TIMINGS.backoffMaxMs));
			}
		} catch {
			// No backoff this time.
		}
		lastAssistant = null;
		try {
			if (!ctx.hasPendingMessages()) {
				queued += 1;
				pi.sendUserMessage(REASON, { deliverAs: "followUp" });
				// sendUserMessage queues asynchronously; give it a few ticks.
				for (let i = 0; i < 100 && !ctx.hasPendingMessages(); i++) await tick();
			}
			if (ctx.hasPendingMessages()) return { continue: true };
		} catch {
			// Fall back to an entry below.
		}
		// Queueing did not happen. A custom_message entry still reaches the model
		// as a user message, only without resetting the overflow recovery.
		queued = Math.max(0, queued - 1);
		continuations += 1;
		emit({
			type: "user",
			timestamp: timestamp(),
			...base(),
			rsi_continuation: continuations,
			rsi_fallback: "custom_message",
			message: { role: "user", content: [{ type: "text", text: `${STOP_HOOK_PREFIX}${REASON}` }] },
		});
		return {
			entries: [{ type: "custom_message", customType: "rsi-never-stop", content: REASON, display: false }],
			continue: true,
		};
	});
}
