"""The vllm-in-judge sample's metering proxy (tests/metering_proxy.py, the
task's own reference, not the Harness's) against a fake OpenAI-compatible
upstream: it forwards completions and the model list, passes a stream
through, meters every request's tokens and latency for the trial whose base
path it came under, and passes upstream errors on."""

from __future__ import annotations

import http.client
import importlib.util
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SAMPLE = Path(__file__).parents[2] / "sample_tasks" / "vllm-in-judge"
USAGE = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}


def metering_proxy():
    """The sample's proxy, loaded without leaving bytecode in its /tests."""
    spec = importlib.util.spec_from_file_location(
        "metering_proxy", SAMPLE / "tests" / "metering_proxy.py"
    )
    module = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    return module


class Upstream(BaseHTTPRequestHandler):
    """A vLLM-like server: completions answer with usage, a stream sends two
    content chunks, the usage-only chunk when asked for (``usage_on_last``:
    on the last content chunk instead), then [DONE]. The model ``fail-400``
    is a bad request, ``break`` breaks off mid-stream and ``slow`` streams
    a chunk every 50 ms for 10 s."""

    protocol_version = "HTTP/1.1"
    received: list[tuple[str, str, dict | None, dict]]

    def log_message(self, format, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except BrokenPipeError:
            pass  # The proxy stopped reading a stream its client left.

    def reply(self, status, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.received.append(("GET", self.path, None, dict(self.headers)))
        self.reply(200, {"object": "list", "data": [{"id": "m", "root": "/c"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.received.append(("POST", self.path, body, dict(self.headers)))
        model = body.get("model")
        if model == "fail-400":
            self.reply(400, {"error": {"message": "bad", "code": 400}})
            return
        if not body.get("stream"):
            message = {"role": "assistant", "content": "ab"}
            choice = {"index": 0, "message": message, "finish_reason": "stop"}
            self.reply(
                200,
                {
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [choice],
                    "usage": USAGE,
                },
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        options = body.get("stream_options") or {}
        last = {"model": model, "choices": [{"delta": {"content": "b"}}]}
        chunks = [{"model": model, "choices": [{"delta": {"content": "a"}}]}, last]
        if options.get("usage_on_last"):
            last["usage"] = USAGE
        elif options.get("include_usage"):
            chunks.append({"model": model, "choices": [], "usage": USAGE})
        if model == "slow":
            chunks = chunks[:1] * 200
        for index, chunk in enumerate(chunks):
            self.event(f"data: {json.dumps(chunk)}\n\n".encode())
            if model == "break" and index == 0:
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if model == "slow":
                time.sleep(0.05)
        self.event(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")

    def event(self, data: bytes):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()


def serve(server):
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return server


@pytest.fixture
def proxy(tmp_path):
    module = metering_proxy()
    received: list = []
    handler = type("Handler", (Upstream,), {"received": received})
    upstream = serve(ThreadingHTTPServer(("127.0.0.1", 0), handler))
    log = tmp_path / "logs" / "usage.jsonl"
    servers = [upstream]

    def start(target: str | None = None):
        address = target or f"http://127.0.0.1:{upstream.server_address[1]}"
        server = module.MeteringServer(
            ("127.0.0.1", 0), address, module.Usage(log), 30.0
        )
        servers.append(serve(server))
        return server.server_address[1]

    def records():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    start.received = received
    start.records = records
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def request(port, method, path, body=None, headers=None):
    """(status, content type, body bytes) from the proxy."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        data = json.dumps(body).encode() if body is not None else None
        connection.request(
            method,
            path,
            body=data,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


def events(body: bytes) -> list:
    """The data of a server-sent event stream: JSON chunks and "[DONE]"."""
    found = []
    for block in body.decode().split("\n\n"):
        if block.startswith("data: "):
            data = block[6:]
            found.append(data if data == "[DONE]" else json.loads(data))
    return found


def test_a_completion_is_forwarded_unchanged_and_metered_for_its_trial(proxy):
    port = proxy()
    body = {"model": "rsi-checkpoint", "messages": [{"role": "user", "content": "x"}]}

    status, kind, data = request(
        port,
        "POST",
        "/t/rsi-hello-file/v1/chat/completions",
        body,
        {"Authorization": "Bearer unused", "Accept-Encoding": "gzip"},
    )

    assert (status, kind) == (200, "application/json")
    assert json.loads(data)["usage"] == USAGE
    [(method, path, sent, headers)] = proxy.received
    assert (method, path, sent) == ("POST", "/v1/chat/completions", body)
    assert headers["Authorization"] == "Bearer unused"
    # The proxy reads the body: it never asks for an encoded one.
    assert headers.get("Accept-Encoding") in (None, "identity")
    [record] = proxy.records()
    latency = record.pop("latency_ms")
    assert isinstance(latency, int) and 0 <= latency < 30000
    assert record == {
        "trial": "rsi-hello-file",
        "endpoint": "/v1/chat/completions",
        "model": "rsi-checkpoint",
        "stream": False,
        "status": 200,
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "error": None,
    }


def test_completions_and_models_outside_a_base_path_have_no_trial(proxy):
    port = proxy()

    status, _, _ = request(port, "POST", "/v1/completions", {"model": "m"})
    assert status == 200
    status, _, data = request(port, "GET", "/t/a.b_c-1/v1/models?x=1")
    assert status == 200 and json.loads(data)["data"][0]["id"] == "m"

    assert [item[:2] for item in proxy.received] == [
        ("POST", "/v1/completions"),
        ("GET", "/v1/models?x=1"),
    ]
    first, second = proxy.records()
    assert (first["trial"], first["endpoint"], first["completion_tokens"]) == (
        None,
        "/v1/completions",
        3,
    )
    assert (second["trial"], second["endpoint"], second["model"]) == (
        "a.b_c-1",
        "/v1/models",
        None,
    )
    assert second["prompt_tokens"] is None and second["status"] == 200


def test_a_stream_gets_its_usage_metered_and_stripped_when_not_asked_for(proxy):
    port = proxy()
    body = {"model": "m", "stream": True, "stream_options": {"x": 1}}

    status, kind, data = request(port, "POST", "/t/t1/v1/chat/completions", body)

    assert status == 200 and kind.startswith("text/event-stream")
    # The upstream was asked for usage, the client's options kept.
    [(_, _, sent, _)] = proxy.received
    assert sent["stream_options"] == {"x": 1, "include_usage": True}
    # The client sees the stream it asked for: no usage chunk.
    found = events(data)
    assert [item["choices"][0]["delta"]["content"] for item in found[:-1]] == [
        "a",
        "b",
    ]
    assert found[-1] == "[DONE]"
    assert not any("usage" in item for item in found[:-1])
    [record] = proxy.records()
    assert record["stream"] is True and record["trial"] == "t1"
    assert (record["prompt_tokens"], record["completion_tokens"]) == (11, 3)
    assert record["error"] is None


def test_a_stream_that_asked_for_usage_keeps_it(proxy):
    port = proxy()
    body = {"model": "m", "stream": True, "stream_options": {"include_usage": True}}

    _, _, data = request(port, "POST", "/t/t1/v1/completions", body)

    found = events(data)
    assert found[-2] == {"model": "m", "choices": [], "usage": USAGE}
    assert proxy.received[0][2]["stream_options"] == {"include_usage": True}
    assert proxy.records()[0]["completion_tokens"] == 3


def test_usage_on_a_chunk_with_choices_is_removed_from_that_chunk_only(proxy):
    port = proxy()
    body = {"model": "m", "stream": True, "stream_options": {"usage_on_last": True}}

    _, _, data = request(port, "POST", "/t/t1/v1/chat/completions", body)

    found = events(data)
    assert found[1] == {"model": "m", "choices": [{"delta": {"content": "b"}}]}
    assert found[-1] == "[DONE]" and len(found) == 3
    assert proxy.records()[0]["prompt_tokens"] == 11


def test_an_upstream_error_is_passed_on_and_logged_with_its_status(proxy):
    port = proxy()

    status, _, data = request(
        port, "POST", "/t/t1/v1/chat/completions", {"model": "fail-400"}
    )

    assert status == 400 and json.loads(data)["error"]["message"] == "bad"
    [record] = proxy.records()
    assert record["status"] == 400 and record["model"] == "fail-400"
    assert record["prompt_tokens"] is None and record["error"] is None


def test_an_unreachable_upstream_is_a_502_and_logged(proxy):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]
    port = proxy(f"http://127.0.0.1:{closed}")

    status, _, data = request(port, "POST", "/t/t1/v1/chat/completions", {"model": "m"})

    assert status == 502 and "upstream unreachable" in data.decode()
    [record] = proxy.records()
    assert record["status"] == 502 and record["error"].startswith("upstream: ")
    assert record["trial"] == "t1" and record["model"] == "m"


def test_a_stream_the_upstream_breaks_off_ends_and_is_logged(proxy):
    port = proxy()
    body = {"model": "break", "stream": True}

    status, _, data = request(port, "POST", "/t/t1/v1/chat/completions", body)

    # What came before the break reaches the client; the stream ends.
    assert status == 200
    assert events(data) == [
        {"model": "break", "choices": [{"delta": {"content": "a"}}]}
    ]
    [record] = proxy.records()
    assert record["status"] == 200 and record["error"].startswith("upstream: ")
    assert record["completion_tokens"] is None


def test_a_client_that_goes_away_mid_stream_is_logged(proxy):
    port = proxy()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request(
        "POST",
        "/t/t1/v1/chat/completions",
        body=json.dumps({"model": "slow", "stream": True}),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    assert response.status == 200
    response.read1(64)
    connection.sock.shutdown(socket.SHUT_RDWR)
    connection.close()

    deadline = time.monotonic() + 15
    while not proxy.records() and time.monotonic() < deadline:
        time.sleep(0.1)
    [record] = proxy.records()
    assert record["error"].startswith("client: ")
    # Long before the upstream's 10 s stream would have ended.
    assert record["latency_ms"] < 8000


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/chat/completions"),
        ("POST", "/v1/models"),
        ("POST", "/v1/embeddings"),
        ("POST", "/t/a/b/v1/chat/completions"),
        ("GET", "/metrics"),
    ],
)
def test_anything_else_is_not_forwarded_nor_logged(proxy, method, path):
    port = proxy()

    status, _, _ = request(port, method, path, {} if method == "POST" else None)

    assert status == 404
    assert proxy.received == [] and proxy.records() == []


def test_its_own_health_answers_without_the_upstream(proxy):
    port = proxy("http://127.0.0.1:9")

    assert request(port, "GET", "/metering/health")[0] == 200
    # /health is the upstream's: never the proxy's own answer.
    assert request(port, "GET", "/health")[0] == 404
    assert proxy.records() == []


def test_concurrent_trials_are_each_attributed_their_own_requests(proxy):
    port = proxy()

    def ask(trial, stream):
        for _ in range(5):
            body = {"model": trial, "stream": stream}
            status, _, _ = request(
                port, "POST", f"/t/{trial}/v1/chat/completions", body
            )
            assert status == 200

    threads = [
        threading.Thread(target=ask, args=(f"trial-{index}", index % 2 == 0))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = proxy.records()
    assert len(records) == 20
    # Each request under its own trial: the model each trial asked for.
    assert all(item["trial"] == item["model"] for item in records)
    assert sorted({item["trial"] for item in records}) == [
        f"trial-{index}" for index in range(4)
    ]
    assert all(item["completion_tokens"] == 3 for item in records)


def test_the_openai_client_streams_through_the_proxy(proxy):
    openai = pytest.importorskip("openai")
    port = proxy()
    client = openai.OpenAI(
        base_url=f"http://127.0.0.1:{port}/t/t9/v1", api_key="unused", max_retries=0
    )

    stream = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], stream=True
    )
    text = "".join(chunk.choices[0].delta.content for chunk in stream if chunk.choices)
    answer = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}]
    )

    assert text == "ab" and answer.usage.completion_tokens == 3
    assert [(item["trial"], item["stream"]) for item in proxy.records()] == [
        ("t9", True),
        ("t9", False),
    ]


def test_litellm_hosted_vllm_asks_under_its_trial_base_path(proxy, monkeypatch):
    # terminus-2's own path to vLLM: LiteLLM's hosted_vllm provider.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    port = proxy()
    base = f"http://127.0.0.1:{port}/t/rsi-hello-file/v1"
    messages = [{"role": "user", "content": "x"}]

    answer = litellm.completion(
        model="hosted_vllm/rsi-checkpoint",
        api_base=base,
        api_key="unused",
        messages=messages,
    )
    stream = litellm.completion(
        model="hosted_vllm/rsi-checkpoint",
        api_base=base,
        api_key="unused",
        messages=messages,
        stream=True,
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    assert answer.usage.prompt_tokens == 11 and text == "ab"
    assert [item[1] for item in proxy.received] == ["/v1/chat/completions"] * 2
    assert [
        (item["trial"], item["model"], item["stream"], item["completion_tokens"])
        for item in proxy.records()
    ] == [
        ("rsi-hello-file", "rsi-checkpoint", False, 3),
        ("rsi-hello-file", "rsi-checkpoint", True, 3),
    ]
