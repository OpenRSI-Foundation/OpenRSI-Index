"""A6 in Work: the endpoint socket is the only socket, and it is no Engine API.

Engine paths get no 2xx: ``POST /v1/containers`` is an unknown operation
and every other path under ``/v1/`` (``/v1/containers/json``, either method)
is unsupported too, 400 ``unsupported``; paths outside the API prefixes have
no route. Prints one ``RSI-ACCEPTANCE a6-work {json}`` line; ``ok`` is the
verdict.
"""

import http.client
import json
import os
import socket
import subprocess


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=30)
        self.unix_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect(self.unix_path)


def ask(method, path, body=None):
    connection = UnixConnection(os.environ["RSI_SANDBOX_SOCKET"])
    headers = {"Authorization": "Bearer " + os.environ["RSI_SANDBOX_TOKEN"]}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


found = subprocess.run(
    ["find", "/", "(", "-path", "/proc", "-o", "-path", "/sys", ")", "-prune"]
    + ["-o", "-type", "s", "-print"],
    capture_output=True,
    text=True,
).stdout.split()
status, body = ask("POST", "/v1/containers", b"{}")
try:
    code = json.loads(body)["error"]["code"]
except (ValueError, KeyError, TypeError):
    code = None
engine = {
    f"{method} {path}": ask(method, path, b"{}" if method == "POST" else None)[0]
    for method, path in (
        ("POST", "/v1/containers/json"),
        ("GET", "/v1/containers/json"),
        ("GET", "/_ping"),
        ("GET", "/version"),
        ("GET", "/v1.53/containers/json"),
        ("GET", "/containers/json"),
    )
}
report = {
    "sockets": sorted(found),
    "v1_containers": [status, code],
    "engine_paths": engine,
    "docker_host": "DOCKER_HOST" in os.environ,
}
report["ok"] = (
    report["sockets"] == [os.environ["RSI_SANDBOX_SOCKET"]]
    and (status, code) == (400, "unsupported")
    and all(value >= 400 for value in engine.values())
    and all(engine[key] == 400 for key in engine if " /v1/" in key)
    and not report["docker_host"]
)
print("RSI-ACCEPTANCE a6-work " + json.dumps(report, sort_keys=True))
