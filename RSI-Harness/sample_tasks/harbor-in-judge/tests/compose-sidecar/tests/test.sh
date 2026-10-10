#!/bin/bash
python3 - <<'PY'
import socket


def get(key):
    request = f"*2\r\n$3\r\nGET\r\n${len(key)}\r\n{key}\r\n"
    with socket.create_connection(("kvstore", 6379), timeout=10) as connection:
        connection.sendall(request.encode())
        reply = connection.makefile("rb")
        header = reply.readline()
        if header.startswith(b"$-1"):
            return None
        return reply.read(int(header[1:])).decode()


expected = open("/seed/value.txt").read().strip()
try:
    written = open("/app/answer.txt").read().strip()
except FileNotFoundError:
    written = None
ok = get("answer") == expected and written == expected
with open("/logs/verifier/reward.txt", "w") as output:
    output.write("1\n" if ok else "0\n")
PY
