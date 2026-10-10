"""Meter an OpenAI-compatible model server per trial (the sample's own logic,
not the Harness's: a reference to copy into a task that needs it).

    metering_proxy.py --upstream http://127.0.0.1:8000 --port 8001
        --log /logs/verifier/usage.jsonl [--host 127.0.0.1] [--timeout SEC]

It listens on loopback and forwards ``/v1/chat/completions``,
``/v1/completions`` and ``/v1/models`` to the upstream, either as they are or
under a per-trial base path ``/t/<trial>/v1/...``: an agent given
``api_base=http://127.0.0.1:8001/t/<trial>/v1`` has its requests attributed
to ``<trial>``. Every other path is 404, except its own health,
``/metering/health`` (a path of its own: no upstream answers it for it).

Each forwarded request appends one JSON line to the log: ``trial`` (null
without a base path), ``endpoint``, ``model`` (the request's, else the
response's), ``stream``, ``status`` (the upstream's, 502 when it cannot be
reached), ``prompt_tokens``, ``completion_tokens`` (from the response's
``usage``, null without one), ``latency_ms`` (until the last byte) and
``error`` (null, or why the exchange broke off).

A streamed request is passed through event by event. Its usage comes from
the stream itself: the proxy asks for it (``stream_options.include_usage``)
and, when the client did not, strips it again (the usage-only final chunk,
or ``usage`` on a chunk with choices), so the client sees the stream it
asked for. Standard library only: it runs next to Harbor with the system
Python.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROUTE = re.compile(
    r"^(?:/t/(?P<trial>[A-Za-z0-9._-]+))?"
    r"/v1/(?P<endpoint>chat/completions|completions|models)/?$"
)
# What an upstream that breaks off raises (http.client's IncompleteRead and
# BadStatusLine are no OSError).
BROKEN = (OSError, http.client.HTTPException)
# Request headers the upstream gets: hop-by-hop ones and Accept-Encoding
# (the proxy reads the body, so it must come unencoded) stay behind.
FORWARDED = frozenset({"authorization", "content-type", "accept", "user-agent"})


class Usage:
    """The append-only usage log, one JSON line per request."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        with self.lock, self.path.open("a") as log:
            log.write(line)


def take_usage(record: dict, body: object) -> None:
    usage = body.get("usage") if isinstance(body, dict) else None
    if isinstance(usage, dict):
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["completion_tokens"] = usage.get("completion_tokens")


def stream_event(lines: list[bytes], record: dict, strip: bool) -> bytes | None:
    """One server-sent event of a completion stream (its lines), its usage
    recorded (the last one seen wins: continuous usage is cumulative); with
    ``strip``, None for a usage-only chunk and a chunk with choices without
    its usage."""
    for index, line in enumerate(lines):
        if not line.startswith(b"data:"):
            continue
        try:
            chunk = json.loads(line[5:])
        except ValueError:
            continue
        if not isinstance(chunk, dict) or chunk.get("usage") is None:
            continue
        take_usage(record, chunk)
        if not record["model"]:
            record["model"] = chunk.get("model")
        if strip:
            if not chunk.get("choices"):
                return None
            del chunk["usage"]
            lines[index] = b"data: " + json.dumps(chunk).encode()
    return b"\n".join(lines) + b"\n\n"


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MeteringServer

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # The client went away (its request is logged).

    def do_GET(self) -> None:
        self.forward("GET")

    def do_POST(self) -> None:
        self.forward("POST")

    def reply(self, status: int, body: bytes, kind="application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status: int, message: str) -> None:
        error = {"error": {"message": message, "type": "metering_proxy"}}
        self.reply(status, json.dumps(error).encode())

    def forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        url = urlsplit(self.path)
        route = ROUTE.match(url.path)
        if method == "GET" and url.path == "/metering/health":
            self.reply(200, b"", "text/plain")
            return
        if not route or (method == "GET") != (route["endpoint"] == "models"):
            self.fail(404, f"{method} {url.path}: not forwarded")
            return
        try:
            request = json.loads(body) if body else None
        except ValueError:
            request = None
        record = {
            "trial": route["trial"],
            "endpoint": f"/v1/{route['endpoint']}",
            "model": request.get("model") if isinstance(request, dict) else None,
            "stream": False,
            "status": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_ms": None,
            "error": None,
        }
        strip = False
        if isinstance(request, dict) and request.get("stream") is True:
            record["stream"] = True
            options = request.get("stream_options")
            options = options if isinstance(options, dict) else {}
            if not options.get("include_usage"):
                strip = True
                request["stream_options"] = {**options, "include_usage": True}
                body = json.dumps(request).encode()
        self.started, self.logged = time.monotonic(), False
        try:
            self.exchange(method, url, body, record, strip)
        finally:
            self.log(record)

    def log(self, record: dict) -> None:
        """Append the request's line once, before the response's last bytes
        reach the client: whoever has the whole answer finds it logged."""
        if self.logged:
            return
        self.logged = True
        record["latency_ms"] = round((time.monotonic() - self.started) * 1000)
        self.server.usage.append(record)

    def exchange(self, method, url, body: bytes, record: dict, strip: bool) -> None:
        target = f"{self.server.upstream}/v1/{record['endpoint'][4:]}"
        if url.query:
            target += f"?{url.query}"
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() in FORWARDED
        }
        upstream = urllib.request.Request(
            target, data=body if method == "POST" else None, method=method
        )
        for name, value in headers.items():
            upstream.add_header(name, value)
        try:
            response = urllib.request.urlopen(
                upstream, timeout=self.server.upstream_timeout
            )
        except urllib.error.HTTPError as error:
            response = error
        except BROKEN as error:
            record["status"], record["error"] = 502, f"upstream: {error}"
            self.log(record)
            self.fail(502, f"upstream unreachable: {error}")
            return
        with response:
            record["status"] = response.status
            kind = response.headers.get("Content-Type") or "application/json"
            if kind.startswith("text/event-stream"):
                self.relay(response, kind, record, strip)
                return
            try:
                data = response.read()
            except BROKEN as error:
                record["status"], record["error"] = 502, f"upstream: {error}"
                self.log(record)
                self.fail(502, f"upstream broke off: {error}")
                return
            try:
                answer = json.loads(data)
            except ValueError:
                answer = None
            take_usage(record, answer)
            if not record["model"] and isinstance(answer, dict):
                record["model"] = answer.get("model")
            self.log(record)
            self.reply(response.status, data, kind)

    def relay(self, response, kind: str, record: dict, strip: bool) -> None:
        """Pass a server-sent event stream through, event by event, as a
        chunked response. A stream that ends before its ``data: [DONE]``
        (http.client reads a cut chunked body as its end) broke off."""
        self.send_response(response.status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        event: list[bytes] = []
        done = False
        try:
            while True:
                try:
                    line = response.readline()
                except BROKEN as error:
                    record["error"] = f"upstream: {error}"
                    line = b""
                if line.strip():
                    event.append(line.rstrip(b"\r\n"))
                    continue
                # A blank line ends an event, as does the stream's end.
                if event:
                    done = done or b"data: [DONE]" in event
                    passed = stream_event(event, record, strip)
                    if passed is not None:
                        self.chunk(passed)
                    event = []
                if not line:
                    break
            if not done and not record["error"]:
                record["error"] = "upstream: the stream ended before [DONE]"
            self.log(record)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError as error:
            # The client went away: stop reading the upstream too.
            record["error"] = f"client: {error}"
            self.close_connection = True

    def chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()


class MeteringServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, upstream: str, usage: Usage, timeout: float):
        super().__init__(address, Proxy)
        self.upstream = upstream.rstrip("/")
        self.usage = usage
        self.upstream_timeout = timeout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    server = MeteringServer(
        (args.host, args.port), args.upstream, Usage(args.log), args.timeout
    )
    print(f"metering {args.upstream} on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
