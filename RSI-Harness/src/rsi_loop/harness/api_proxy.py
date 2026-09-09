# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local reverse proxy for API access through an upstream HTTP(S) proxy.

When network isolation is enabled (internet=false) but the API endpoint
requires a corporate proxy, we cannot set proxy env vars inside the
container — iptables blocks all traffic except whitelisted IPs.

Solution: run a lightweight reverse proxy on the host that:
  1. Listens on <host>:<port>
  2. Forwards every request to the real API via the configured proxy
  3. The container reaches it via host.docker.internal:<port>

This avoids the container needing any proxy awareness or internet access.

Usage (separate terminal):
  python -m rsi_loop proxy --target https://api.anthropic.com --port 9090
"""

from __future__ import annotations

import logging
import signal
import socket
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import requests as req_lib

logger = logging.getLogger(__name__)


def _find_free_port(host: str = "0.0.0.0") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _make_handler(target_url: str, session: req_lib.Session):
    """Create a request handler class that proxies to target_url via session."""
    target_base = target_url.rstrip("/")

    class ProxyHandler(BaseHTTPRequestHandler):
        # HTTP/1.0: connection close signals end-of-body, no need for
        # Content-Length or Transfer-Encoding: chunked.
        protocol_version = "HTTP/1.0"

        def do_request(self):
            upstream_url = f"{target_base}{self.path}"

            body = None
            content_length = self.headers.get("Content-Length")
            if content_length:
                body = self.rfile.read(int(content_length))

            fwd_headers = {}
            for key, val in self.headers.items():
                if key.lower() in ("host", "transfer-encoding", "accept-encoding"):
                    continue
                fwd_headers[key] = val
            parsed = urlparse(target_base)
            fwd_headers["Host"] = parsed.hostname or parsed.netloc

            try:
                resp = session.request(
                    method=self.command,
                    url=upstream_url,
                    headers=fwd_headers,
                    data=body,
                    timeout=300,
                    stream=True,
                )
                self.send_response(resp.status_code)
                for key, val in resp.headers.items():
                    if key.lower() in (
                        "transfer-encoding",
                        "content-encoding",
                        "content-length",
                        "connection",
                        "keep-alive",
                    ):
                        continue
                    self.send_header(key, val)
                self.end_headers()
                for chunk in resp.iter_content(chunk_size=None):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception as exc:
                self.send_error(502, str(exc))

        do_GET = do_request
        do_POST = do_request
        do_PUT = do_request
        do_PATCH = do_request
        do_DELETE = do_request
        do_HEAD = do_request
        do_OPTIONS = do_request

        def log_message(self, format, *args):
            logger.debug("api-proxy: %s", format % args)

    return ProxyHandler


class APIProxy:
    """A local reverse proxy that forwards to an API via an upstream proxy."""

    def __init__(
        self,
        target_url: str,
        http_proxy: str | None = None,
        https_proxy: str | None = None,
        host: str = "0.0.0.0",
        port: int | None = None,
    ) -> None:
        self.target_url = target_url.rstrip("/")
        self.host = host
        self.port = port or _find_free_port(host)

        self._session = req_lib.Session()
        proxies: dict[str, str] = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        self._session.proxies = proxies

        handler = _make_handler(self.target_url, self._session)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread: threading.Thread | None = None

    @property
    def local_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        """Start the proxy in a background thread. Returns the local URL."""
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="rsi-loop-api-proxy",
        )
        self._thread.start()
        logger.info(
            f"API proxy started: {self.local_url} -> {self.target_url} "
            f"(via proxy)"
        )
        return self.local_url

    def run_forever(self) -> None:
        """Run the proxy in the foreground (blocks until SIGINT/SIGTERM)."""
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown(signum, frame):
            print("\nShutting down API proxy...")
            threading.Thread(
                target=self._server.shutdown, daemon=True
            ).start()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        try:
            self._server.serve_forever()
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def stop(self) -> None:
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)
        self._session.close()
        logger.info("API proxy stopped")
