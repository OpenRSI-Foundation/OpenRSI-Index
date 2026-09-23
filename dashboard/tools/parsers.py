"""Normalize public transcripts into ordered, inert text events (stdlib only)."""
import gzip
import json
import re


def text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text(item) for item in value)
    if isinstance(value, dict) and value.get("type") in {"text", "input_text", "output_text"}:
        return text(value.get("text"))
    return json.dumps(value, ensure_ascii=False, indent=2)


def event(kind, body, title=None, **extra):
    return {"kind": kind, "title": title or kind.title(), "text": text(body), **extra}


def codex_lines(lines):
    """Terminal sections, including fences and the preamble; no text is discarded."""
    markers = {"user": "user", "codex": "assistant", "thinking": "thinking",
               "exec": "tool", "apply patch": "tool", "file update": "tool", "file update:": "tool"}
    kind, title, buffer, fence = "system", "Session", [], None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line in markers and fence is None:
            if buffer:
                yield event(kind, "\n".join(buffer), title)
            kind, title, buffer = markers[line], line.title(), []
            continue
        if kind in {"user", "assistant", "thinking"}:
            match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
            if match:
                mark = match[1]
                if fence and mark[0] == fence[0] and len(mark) >= len(fence):
                    fence = None
                elif fence is None:
                    fence = mark
        buffer.append(line)
    if buffer:
        yield event(kind, "\n".join(buffer), title)


def jsonl_lines(lines):
    """Claude stream-json and Codex JSONL, preserving tool results and subagents."""
    seen = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            yield event("system", line.rstrip(), "Terminal output")
            continue
        if not isinstance(row, dict):
            yield event("system", row, "Record")
            continue
        uuid = row.get("uuid")
        if uuid and uuid in seen:
            continue
        if uuid:
            seen.add(uuid)
        kind = row.get("type")
        extra = {"at": row.get("timestamp"), "parent": row.get("parent_tool_use_id")}
        if kind in {"assistant", "user"}:
            message = row.get("message") or {}
            blocks = message.get("content", []) if isinstance(message, dict) else message
            if isinstance(blocks, str):
                blocks = [{"type": "text", "text": blocks}]
            if not isinstance(blocks, list):
                blocks = [blocks]
            for block in blocks:
                if not isinstance(block, dict):
                    yield event(kind, block, **extra)
                    continue
                typ = block.get("type")
                if typ == "text":
                    yield event(kind, block.get("text"), **extra)
                elif typ in {"thinking", "redacted_thinking"}:
                    yield event("thinking", block.get("thinking", "[redacted]"), **extra)
                elif typ == "tool_use":
                    yield event("tool", block.get("input"), block.get("name", "Tool call"),
                                call_id=block.get("id"), **extra)
                elif typ == "tool_result":
                    yield event("result", block.get("content"), "Tool result",
                                call_id=block.get("tool_use_id"), error=bool(block.get("is_error")), **extra)
                else:
                    yield event(kind, block, **extra)
        elif kind == "response_item":
            payload = row.get("payload") or {}
            typ = payload.get("type", "")
            if typ == "message":
                yield event(payload.get("role", "assistant"), payload.get("content"), **extra)
            elif typ in {"function_call", "custom_tool_call"}:
                yield event("tool", payload.get("arguments", payload.get("input")), payload.get("name"), **extra)
            elif typ.endswith("_call_output"):
                yield event("result", payload.get("output"), **extra)
            elif typ == "reasoning":
                yield event("thinking", payload.get("summary"), **extra)
            else:
                yield event("system", payload, **extra)
        elif kind in {"item.completed", "item.started", "item.updated"}:
            item = row.get("item") or {}
            typ = item.get("type")
            if typ in {"agent_message", "reasoning"}:
                yield event("thinking" if typ == "reasoning" else "assistant", item.get("text"), **extra)
            else:
                yield event("tool", item, str(typ or "Tool"), **extra)
        elif kind == "result":
            yield event("system" if row.get("is_error") else "assistant",
                        row.get("result") or row.get("errors") or row, "Session result", **extra)
        else:
            yield event("system", row, str(kind or "Record"), **extra)


def transcript(path, agent):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        head = stream.read(8192)
        stream.seek(0)
        # Container launch diagnostics can precede the first JSON event.
        is_json = any(line.lstrip().startswith('{"') for line in head.splitlines())
        parser = jsonl_lines if is_json and "OpenAI Codex" not in head else codex_lines
        yield from parser(stream)


def atif(data):
    for step in data.get("steps", []):
        role = "assistant" if step.get("source") == "agent" else step.get("source", "system")
        extra = {"at": step.get("timestamp"), "step": step.get("step_id")}
        if step.get("message"):
            yield event(role, step["message"], **extra)
        if step.get("reasoning_content"):
            yield event("thinking", step["reasoning_content"], **extra)
        for call in step.get("tool_calls", []):
            yield event("tool", call.get("arguments"), call.get("function_name", "Tool"),
                        call_id=call.get("tool_call_id"), **extra)
        for result in (step.get("observation") or {}).get("results", []):
            yield event("result", result.get("content"), "Tool result",
                        call_id=result.get("source_call_id"), **extra)
