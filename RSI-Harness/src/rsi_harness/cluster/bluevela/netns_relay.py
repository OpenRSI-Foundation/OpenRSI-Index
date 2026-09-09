#!/usr/bin/env python3
"""Expose two private Unix sockets on loopback inside an isolated netns."""

from __future__ import annotations

import argparse
import os
import select
import socket
import subprocess
import sys
import threading
from pathlib import Path


class _TcpToUnixRelay:
    def __init__(self, unix_socket: Path, tcp_port: int) -> None:
        self._unix_socket = unix_socket
        self._requested_port = tcp_port
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def port(self) -> int:
        if self._listener is None:
            raise RuntimeError("loopback relay is not started")
        return int(self._listener.getsockname()[1])

    def start(self) -> None:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self._requested_port))
        listener.listen()
        listener.settimeout(0.2)
        self._listener = listener
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _accept(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stopping.is_set():
            try:
                client, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stopping.is_set():
                    return
                continue
            threading.Thread(
                target=self._serve,
                args=(client,),
                daemon=True,
            ).start()

    def _serve(self, client: socket.socket) -> None:
        try:
            upstream = socket.socket(socket.AF_UNIX)
            upstream.connect(str(self._unix_socket))
            with upstream:
                sockets = (client, upstream)
                while not self._stopping.is_set():
                    readable, _writable, _errors = select.select(
                        sockets, (), (), 0.2
                    )
                    for source in readable:
                        target = upstream if source is client else client
                        content = source.recv(64 * 1024)
                        if not content:
                            return
                        target.sendall(content)
        except OSError:
            return
        finally:
            client.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-socket", required=True, type=Path)
    parser.add_argument("--submit-socket", required=True, type=Path)
    parser.add_argument("--submit-port", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("isolated relay requires a child command")
    if not 1 <= args.submit_port <= 65535:
        raise SystemExit("isolated relay requires a valid submit port")
    provider = _TcpToUnixRelay(args.provider_socket, 0)
    submit = _TcpToUnixRelay(args.submit_socket, args.submit_port)
    provider.start()
    try:
        submit.start()
    except BaseException:
        provider.stop()
        raise
    environment = os.environ.copy()
    proxy_url = f"http://127.0.0.1:{provider.port}"
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment[name] = proxy_url
    for name in ("NO_PROXY", "no_proxy"):
        environment[name] = "127.0.0.1,localhost"
    try:
        process = subprocess.Popen(command, env=environment)
        return process.wait()
    finally:
        submit.stop()
        provider.stop()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
