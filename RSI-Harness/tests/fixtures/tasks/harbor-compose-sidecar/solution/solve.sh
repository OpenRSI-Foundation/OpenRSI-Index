#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import os
import socket

value = open("/seed/value.txt").read().strip()
items = ("SET", "answer", value)
request = f"*{len(items)}\r\n" + "".join(
    f"${len(item.encode())}\r\n{item}\r\n" for item in items
)
with socket.create_connection(("kvstore", 6379), timeout=10) as connection:
    connection.sendall(request.encode())
    assert connection.recv(64).startswith(b"+OK")
os.makedirs("/app", exist_ok=True)
with open("/app/answer.txt", "w") as output:
    output.write(value + "\n")
PY
