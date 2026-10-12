#!/usr/bin/env python3
"""Fake model server for Agent tests: three model APIs (stdlib only).

usage: fake_model_server.py --log REQUESTS.jsonl [--scenario S.json]
                            [--host 127.0.0.1] [--port 0] [--port-file PATH]

Endpoints, matched by the end of the path, so a base URL with or without
``/v1`` reaches them (the log records the path, for a test to insist on one):
  GET  /v1/models                  one model, ``fake-model``
  POST /v1/chat/completions        OpenAI chat completions (driven Agents)
  POST /v1/responses               OpenAI Responses (Codex)
  POST /v1/messages                Anthropic Messages (Claude Code)
  POST /v1/messages/count_tokens   a fixed count; not logged
Each answers streamed (SSE) or plain, as the request asks, with tool calls.

Every request is appended to the log as one JSON line, the same for all three
APIs: the request number, the API, the request's path (without its query),
the action taken, the message count (chat messages, Responses input items or
Anthropic messages), the last message role (``tool`` for a tool result: a
``tool`` message, a ``function_call_output`` item, or a user message of
``tool_result`` blocks; Anthropic mid-conversation ``system`` messages are
skipped), the last user text (of its text blocks) and tool result, the model,
the tools offered, the tool call made, whether an API key was sent, and the
reasoning effort and output limit asked for, and which ``watch`` strings the
request contains anywhere (system prompt included). Tests assert on that log.

Scenario keys (all optional):
  cycle        actions for successive model requests (default ["text"])
  at           {"<request number>": action} overrides
  after_tool   action when the last message is a tool result (default "text")
  delay        seconds to wait before answering every request
  shell_tools  shell tool names to call, first one the request offers wins
               (default: bash, run_shell_command, shell, Bash, exec_command)
  shell_arguments
               further arguments for every shell tool call
  commands     shell commands for successive tool calls; afterwards each call
               runs ``default_command``. An entry may instead be
               {"tool": NAME, "arguments": {...}} to call another tool.
  default_command
               shell command once ``commands`` is used up
               (default ``echo tool-run-<request number>``)
  replies      {"<request number>": text} the text of that request's reply
               (for a text or mixed action)
  watch        strings to look for in every request, e.g. a marker that
               must not reach the model

Actions: text, tool, mixed, empty, reasoning, length, e400, e401, e429,
e500, overflow, hang (never answers; the client has to time out).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

MODEL_ID = "fake-model"
SHELL_TOOLS = ("bash", "run_shell_command", "shell", "Bash", "exec_command")
_TEXT_CAP = 2000
_OVERFLOW = (
    "This model's maximum context length is 16000 tokens. "
    "However, your messages resulted in 20000 tokens."
)
_ERRORS = {
    "e400": (400, "invalid_request_error", "fake bad request"),
    "e401": (401, "authentication_error", "fake invalid api key"),
    "e429": (429, "rate_limit_error", "fake rate limit"),
    "e500": (500, "server_error", "fake internal server error"),
    "overflow": (400, "invalid_request_error", _OVERFLOW),
}
# The Anthropic API's names for the same errors, and the overflow text
# Claude Code recognizes.
_ANTHROPIC_ERRORS = {
    "server_error": "api_error",
    _OVERFLOW: "prompt is too long: 20000 tokens > 16000 maximum",
}
_USAGE = {"input": 100, "output": 10}


def _text_of(content: Any) -> str:
    """The text of a message content: a string or a list of parts.

    Only text parts count (``text``, ``input_text`` or ``output_text``), so
    an Anthropic user message carrying tool results has none.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
            and part.get("type", "text") in ("text", "input_text", "output_text")
        )
    return ""


def _result_text(content: Any) -> str:
    """The text of a tool result, which may itself be a list of parts."""
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


# NamedTuples, not dataclasses: tests load this file by path, outside
# sys.modules, where a dataclass cannot be defined.
class _Turn(NamedTuple):
    """A model request reduced to what the log and the scenario use."""

    n_messages: int
    last_role: str | None
    last_user: str | None
    last_tool: str | None
    # Tool name -> its JSON schema.
    tools: dict[str, dict[str, Any]]
    reasoning_effort: Any = None
    max_tokens: Any = None


def _chat_turn(body: dict[str, Any]) -> _Turn:
    messages = body.get("messages") or []
    last = messages[-1] if messages else {}
    users = [_text_of(m.get("content")) for m in messages if m.get("role") == "user"]
    return _Turn(
        n_messages=len(messages),
        last_role=last.get("role"),
        last_user=users[-1] if users else None,
        last_tool=(
            _text_of(last.get("content")) if last.get("role") == "tool" else None
        ),
        tools={
            str(tool.get("function", {}).get("name")): tool.get("function", {}).get(
                "parameters"
            )
            or {}
            for tool in body.get("tools") or []
            if isinstance(tool, dict)
        },
        reasoning_effort=body.get("reasoning_effort"),
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
    )


def _responses_turn(body: dict[str, Any]) -> _Turn:
    items = body.get("input") or []
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    last = items[-1] if items else {}
    users = [
        _text_of(item.get("content"))
        for item in items
        if item.get("role") == "user" and item.get("type", "message") == "message"
    ]
    last_tool = None
    if last.get("type") in ("function_call_output", "custom_tool_call_output"):
        last_tool = _result_text(last.get("output"))
    return _Turn(
        n_messages=len(items),
        last_role="tool" if last_tool is not None else last.get("role"),
        last_user=users[-1] if users else None,
        last_tool=last_tool,
        tools={
            str(tool.get("name")): tool.get("parameters") or {}
            for tool in body.get("tools") or []
            if isinstance(tool, dict) and tool.get("name")
        },
        reasoning_effort=(body.get("reasoning") or {}).get("effort"),
        max_tokens=body.get("max_output_tokens"),
    )


def _messages_turn(body: dict[str, Any]) -> _Turn:
    messages = body.get("messages") or []
    turns = [m for m in messages if m.get("role") != "system"]
    last = turns[-1] if turns else {}
    users = [
        text
        for m in turns
        if m.get("role") == "user" and (text := _text_of(m.get("content")))
    ]
    content = last.get("content")
    results = (
        [
            _result_text(block.get("content"))
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        if isinstance(content, list)
        else []
    )
    output = body.get("output_config") or {}
    return _Turn(
        n_messages=len(messages),
        last_role="tool" if results else last.get("role"),
        last_user=users[-1] if users else None,
        last_tool="\n".join(results) if results else None,
        tools={
            str(tool.get("name")): tool.get("input_schema") or {}
            for tool in body.get("tools") or []
            if isinstance(tool, dict) and tool.get("name")
        },
        reasoning_effort=output.get("effort"),
        max_tokens=body.get("max_tokens"),
    )


_APIS = {
    "/chat/completions": ("chat", _chat_turn),
    "/responses": ("responses", _responses_turn),
    "/messages": ("messages", _messages_turn),
}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client killed mid-request (an Agent at its timeout) is expected.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


class FakeModelServer:
    """A scripted model server; usable in-process or as a script."""

    def __init__(
        self,
        log_path: Path,
        scenario: dict[str, Any] | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.log_path = Path(log_path)
        self.scenario = scenario or {}
        self._counter = itertools.count(1)
        self._commands = iter(self.scenario.get("commands") or ())
        self._lock = threading.Lock()
        self._hangs: list[threading.Event] = []
        handler = type("Handler", (_Handler,), {"server_state": self})
        self.httpd = _Server((host, port), handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> FakeModelServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        for event in self._hangs:
            event.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> FakeModelServer:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def requests(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def next_action(self, last_role: str | None) -> tuple[int, str]:
        with self._lock:
            number = next(self._counter)
        at = self.scenario.get("at", {})
        if str(number) in at:
            return number, at[str(number)]
        if last_role == "tool":
            return number, self.scenario.get("after_tool", "text")
        cycle = self.scenario.get("cycle") or ["text"]
        return number, cycle[(number - 1) % len(cycle)]

    def next_command(self, number: int) -> str | dict[str, Any]:
        default = self.scenario.get("default_command") or f"echo tool-run-{number}"
        with self._lock:
            return next(self._commands, default)

    def record(self, entry: dict[str, Any]) -> None:
        with self._lock, self.log_path.open("a") as log:
            log.write(json.dumps(entry) + "\n")

    def hang(self) -> None:
        event = threading.Event()
        self._hangs.append(event)
        event.wait()


class _Reply(NamedTuple):
    number: int
    model: str
    text: str | None
    reasoning: str | None
    # "stop", "length" or "tool_calls".
    finish: str
    # The tool call, in chat-completions form.
    tool: dict[str, Any] | None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_state: FakeModelServer

    def log_message(self, *_: object) -> None:
        pass

    def do_GET(self) -> None:
        if urlsplit(self.path).path.rstrip("/").endswith("/models"):
            self._json(
                200,
                {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]},
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length)
        body = json.loads(raw or b"{}")
        path = urlsplit(self.path).path.rstrip("/")
        if path.endswith("/messages/count_tokens"):
            self._json(200, {"input_tokens": _USAGE["input"]})
            return
        api, parse = next(
            (found for suffix, found in _APIS.items() if path.endswith(suffix)),
            (None, None),
        )
        if api is None or parse is None:
            self._json(404, {"error": {"message": "not found"}})
            return
        state = self.server_state
        turn = parse(body)
        number, action = state.next_action(turn.last_role)
        call = None
        if action in ("tool", "mixed"):
            call = self._tool_call(turn.tools, state.next_command(number), number)
        state.record(
            {
                "n": number,
                "t": round(time.time(), 3),
                "api": api,
                # As requested, so a test sees the URL a real endpoint would.
                "path": path,
                "action": action,
                "model": body.get("model"),
                "stream": bool(body.get("stream")),
                "n_messages": turn.n_messages,
                "last_role": turn.last_role,
                "last_user": turn.last_user,
                "last_tool": (
                    turn.last_tool[:_TEXT_CAP] if turn.last_tool is not None else None
                ),
                "tools": len(turn.tools),
                "tool_names": sorted(turn.tools),
                "call": call["function"] if call else None,
                "authorization": bool(
                    self.headers.get("authorization") or self.headers.get("x-api-key")
                ),
                "reasoning_effort": turn.reasoning_effort,
                "max_tokens": turn.max_tokens,
                "watched": [
                    text
                    for text in state.scenario.get("watch") or ()
                    if text in raw.decode("utf-8", errors="replace")
                ],
            }
        )
        if state.scenario.get("delay"):
            time.sleep(float(state.scenario["delay"]))
        if action == "hang":
            state.hang()
            return
        if action in _ERRORS:
            status, kind, message = _ERRORS[action]
            if api == "messages":
                error = {
                    "type": _ANTHROPIC_ERRORS.get(kind, kind),
                    "message": _ANTHROPIC_ERRORS.get(message, message),
                }
                self._json(status, {"type": "error", "error": error})
            else:
                self._json(status, {"error": {"message": message, "type": kind}})
            return
        reply = self._reply(body, action, number, call)
        text = (state.scenario.get("replies") or {}).get(str(number))
        if text is not None and reply.text is not None:
            reply = reply._replace(text=text)
        {
            "chat": self._chat,
            "responses": self._responses,
            "messages": self._messages,
        }[api](reply, stream=bool(body.get("stream")))

    def _tool_call(
        self,
        tools: dict[str, dict[str, Any]],
        command: str | dict[str, Any],
        number: int,
    ) -> dict[str, Any]:
        scenario = self.server_state.scenario
        if isinstance(command, dict):
            name = str(command["tool"])
            arguments: dict[str, Any] = dict(command.get("arguments") or {})
        else:
            candidates = scenario.get("shell_tools") or SHELL_TOOLS
            name = next((n for n in candidates if n in tools), candidates[0])
            properties = (tools.get(name) or {}).get("properties") or {}
            # Codex's shell takes ``cmd``; the others ``command``.
            key = "command"
            if "cmd" in properties and "command" not in properties:
                key = "cmd"
            arguments = {key: command, **(scenario.get("shell_arguments") or {})}
        # Fill the other required parameters the CLI declares, so its schema
        # validation accepts the call.
        schema = tools.get(name) or {}
        properties = schema.get("properties") or {}
        for key in schema.get("required") or ():
            if key in arguments:
                continue
            kind = (properties.get(key) or {}).get("type")
            arguments[key] = {"boolean": False, "integer": 0, "number": 0}.get(
                kind, "fake model command"
            )
        return {
            "id": f"call_{number}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    @staticmethod
    def _reply(
        body: dict[str, Any], action: str, number: int, tool: dict[str, Any] | None
    ) -> _Reply:
        text: str | None = f"Reply {number}: I think I am finished."
        reasoning: str | None = None
        finish = "stop"
        if action == "empty":
            text = None
        elif action == "reasoning":
            text, reasoning = None, f"thinking {number}"
        elif action == "length":
            text, finish = f"Reply {number}: truncated outp", "length"
        elif action in ("tool", "mixed"):
            text = None if action == "tool" else f"Let me run a command ({number})."
            finish = "tool_calls"
        return _Reply(
            number=number,
            model=body.get("model") or MODEL_ID,
            text=text,
            reasoning=reasoning,
            finish=finish,
            tool=tool,
        )

    # -- OpenAI chat completions ------------------------------------------

    def _chat(self, reply: _Reply, *, stream: bool) -> None:
        usage = {
            "prompt_tokens": _USAGE["input"],
            "completion_tokens": _USAGE["output"],
            "total_tokens": _USAGE["input"] + _USAGE["output"],
        }
        base = {
            "id": f"chatcmpl-{reply.number}",
            "created": int(time.time()),
            "model": reply.model,
        }
        if not stream:
            message: dict[str, Any] = {"role": "assistant", "content": reply.text}
            if reply.reasoning:
                message["reasoning_content"] = reply.reasoning
            if reply.tool:
                message["tool_calls"] = [reply.tool]
            choice = {"index": 0, "message": message, "finish_reason": reply.finish}
            self._json(
                200,
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [choice],
                    "usage": usage,
                },
            )
            return

        def chunk(delta: dict[str, Any], finish_reason: str | None = None, **extra):
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            payload = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [choice],
                **extra,
            }
            self._event(None, payload)

        self._start_stream()
        chunk({"role": "assistant", "content": ""})
        if reply.reasoning:
            chunk({"reasoning_content": reply.reasoning})
        for part in _halves(reply.text):
            chunk({"content": part})
        if reply.tool:
            chunk({"tool_calls": [{"index": 0, **reply.tool}]})
        chunk({}, reply.finish, usage=usage)
        self.wfile.write(b"data: [DONE]\n\n")
        self._end_stream()

    # -- OpenAI Responses --------------------------------------------------

    def _responses(self, reply: _Reply, *, stream: bool) -> None:
        number = reply.number
        items: list[dict[str, Any]] = []
        if reply.reasoning:
            summary = [{"type": "summary_text", "text": reply.reasoning}]
            items.append(
                {"id": f"rs_{number}", "type": "reasoning", "summary": summary}
            )
        if reply.text:
            items.append(
                {
                    "id": f"msg_{number}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": reply.text, "annotations": []}
                    ],
                }
            )
        if reply.tool:
            items.append(
                {
                    "id": f"fc_{number}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": reply.tool["id"],
                    "name": reply.tool["function"]["name"],
                    "arguments": reply.tool["function"]["arguments"],
                }
            )
        incomplete = reply.finish == "length"
        response: dict[str, Any] = {
            "id": f"resp_{number}",
            "object": "response",
            "created_at": int(time.time()),
            "model": reply.model,
            "status": "incomplete" if incomplete else "completed",
            "output": items,
            "usage": {
                "input_tokens": _USAGE["input"],
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": _USAGE["output"],
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": _USAGE["input"] + _USAGE["output"],
            },
        }
        if incomplete:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        if not stream:
            self._json(200, response)
            return
        self._start_stream()
        self._typed(
            "response.created",
            response={**response, "status": "in_progress", "output": [], "usage": None},
        )
        for index, item in enumerate(items):
            started = {**item, "status": "in_progress"}
            if item["type"] == "message":
                started["content"] = []
            elif item["type"] == "function_call":
                started["arguments"] = ""
            elif item["type"] == "reasoning":
                started["summary"] = []
            self._typed("response.output_item.added", output_index=index, item=started)
            where = {"item_id": item["id"], "output_index": index}
            if item["type"] == "message":
                for part in _halves(reply.text):
                    self._typed(
                        "response.output_text.delta",
                        **where,
                        content_index=0,
                        delta=part,
                    )
            elif item["type"] == "function_call":
                self._typed(
                    "response.function_call_arguments.delta",
                    **where,
                    delta=item["arguments"],
                )
            elif item["type"] == "reasoning":
                self._typed(
                    "response.reasoning_summary_text.delta",
                    **where,
                    summary_index=0,
                    delta=reply.reasoning,
                )
            self._typed("response.output_item.done", output_index=index, item=item)
        self._typed(
            "response.incomplete" if incomplete else "response.completed",
            response=response,
        )
        self._end_stream()

    # -- Anthropic Messages ------------------------------------------------

    def _messages(self, reply: _Reply, *, stream: bool) -> None:
        blocks: list[dict[str, Any]] = []
        if reply.reasoning:
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": reply.reasoning,
                    "signature": "fake-signature",
                }
            )
        if reply.text:
            blocks.append({"type": "text", "text": reply.text})
        if reply.tool:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{reply.number}",
                    "name": reply.tool["function"]["name"],
                    "input": json.loads(reply.tool["function"]["arguments"]),
                }
            )
        stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(
            reply.finish, "tool_use"
        )
        message: dict[str, Any] = {
            "id": f"msg_{reply.number}",
            "type": "message",
            "role": "assistant",
            "model": reply.model,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": _USAGE["input"],
                "output_tokens": _USAGE["output"],
            },
        }
        if not stream:
            self._json(200, message)
            return
        self._start_stream()
        self._typed(
            "message_start",
            message={
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": _USAGE["input"], "output_tokens": 1},
            },
        )
        for index, block in enumerate(blocks):
            kind = block["type"]
            empty = {
                "text": {"type": "text", "text": ""},
                "thinking": {"type": "thinking", "thinking": "", "signature": ""},
                "tool_use": {**block, "input": {}},
            }[kind]
            self._typed("content_block_start", index=index, content_block=empty)
            if kind == "text":
                deltas = [
                    {"type": "text_delta", "text": part}
                    for part in _halves(block["text"])
                ]
            elif kind == "thinking":
                deltas = [
                    {"type": "thinking_delta", "thinking": block["thinking"]},
                    {"type": "signature_delta", "signature": block["signature"]},
                ]
            else:
                deltas = [
                    {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    }
                ]
            for delta in deltas:
                self._typed("content_block_delta", index=index, delta=delta)
            self._typed("content_block_stop", index=index)
        self._typed(
            "message_delta",
            delta={"stop_reason": stop_reason, "stop_sequence": None},
            usage={"output_tokens": _USAGE["output"]},
        )
        self._typed("message_stop")
        self._end_stream()

    # -- transport ---------------------------------------------------------

    def _start_stream(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

    def _event(self, name: str | None, payload: dict[str, Any]) -> None:
        head = f"event: {name}\n" if name else ""
        self.wfile.write(f"{head}data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _typed(self, kind: str, **fields: Any) -> None:
        # Responses and Anthropic events name their type twice: as the SSE
        # event and in the payload.
        self._event(kind, {"type": kind, **fields})

    def _end_stream(self) -> None:
        self.wfile.flush()
        self.close_connection = True

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _halves(text: str | None) -> list[str]:
    """Stream a text in two deltas, so clients have to join them."""
    if not text:
        return []
    half = len(text) // 2
    return [text[:half], text[half:]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text()) if args.scenario else {}
    server = FakeModelServer(args.log, scenario, host=args.host, port=args.port)
    if args.port_file:
        args.port_file.write_text(str(server.httpd.server_address[1]))
    print(server.url, flush=True)
    server.httpd.serve_forever()


if __name__ == "__main__":
    main()
