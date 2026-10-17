#!/usr/bin/env python3
"""A stand-in for ``codex app-server`` (stdio JSON lines), stdlib only.

It speaks the subset of the app-server protocol ``rsi-agent llm`` uses, as the
pinned Codex release does: ``initialize`` (with ``experimentalApi``), the
``initialized`` notification, ``thread/start``, ``turn/start``, then
notifications up to ``turn/completed``. Along the way it sends one server
request (an approval) that the client must refuse.

usage: fake_codex_app_server.py [-c KEY=VALUE ...] app-server

Every message it receives, its argv, environment, uid and working directory,
and the Codex login it found in ``$CODEX_HOME/auth.json`` are appended as one
JSON line to ``$FAKE_RECORD`` when set. ``$FAKE_MODE`` picks a behaviour:
``ok`` (default), ``plain`` (no final_answer phase), ``fail_turn``,
``rpc_error``, ``exit`` (exits after initialize), ``leak`` (prints the login
on stderr), ``refresh`` (rewrites the login, as a token refresh does),
``hang``, ``identity`` (answers with its uid, HOME and login file mode; also
for a prompt that starts with ``IDENTITY``), ``after_final`` (another agent
message after the final answer). A prompt that starts with ``LOGIN`` is
answered with the login's ``generation`` claim and whether its refresh token
was withheld. The model catalog named by its
``model_catalog_json`` override is recorded too.
"""

from __future__ import annotations

import json
import os
import sys
import time

MODE = os.environ.get("FAKE_MODE", "ok")
RECORD = os.environ.get("FAKE_RECORD")
received: list = []


def record(**extra) -> None:
    if not RECORD:
        return
    home = os.environ.get("CODEX_HOME", "")
    login = os.path.join(home, "auth.json")
    try:
        login_text = open(login).read()
        login_mode = oct(os.stat(login).st_mode & 0o777)
    except OSError:
        login_text = None
        login_mode = None
    catalog = None
    for item in sys.argv[1:]:
        if item.startswith("model_catalog_json="):
            try:
                with open(json.loads(item.split("=", 1)[1])) as stream:
                    catalog = json.load(stream)
            except (OSError, ValueError):
                catalog = "unreadable"
    entry = {
        "catalog": catalog,
        "argv": sys.argv[1:],
        "env": dict(os.environ),
        "uid": os.getuid(),
        "cwd": os.getcwd(),
        "login": login_text,
        "login_mode": login_mode,
        "received": received,
        **extra,
    }
    with open(RECORD, "a") as stream:
        stream.write(json.dumps(entry) + "\n")


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def receive() -> dict:
    line = sys.stdin.readline()
    if not line:
        record(eof=True)
        sys.exit(0)
    message = json.loads(line)
    received.append(message)
    return message


def notify(method: str, **params) -> None:
    send({"method": method, "params": params})


def main() -> int:
    if sys.argv[-1] != "app-server":
        sys.stderr.write("fake codex: only app-server is supported\n")
        return 2
    initialize = receive()
    assert initialize["method"] == "initialize", initialize
    send(
        {
            "id": initialize["id"],
            "result": {
                "userAgent": "fake/0",
                "codexHome": os.environ.get("CODEX_HOME"),
                "platformFamily": "unix",
                "platformOs": "linux",
            },
        }
    )
    if MODE == "exit":
        record()
        return 3
    initialized = receive()
    assert initialized == {"method": "initialized"}, initialized
    start = receive()
    assert start["method"] == "thread/start", start
    if MODE == "rpc_error":
        send({"id": start["id"], "error": {"code": -32600, "message": "bad model"}})
        receive()
        return 0
    params = start["params"]
    thread = {"id": "thr_1", "ephemeral": params.get("ephemeral")}
    send({"id": start["id"], "result": {"thread": thread, "model": params["model"]}})
    notify("thread/started", thread=thread)
    turn = receive()
    assert turn["method"] == "turn/start", turn
    send(
        {
            "id": turn["id"],
            "result": {"turn": {"id": "turn_1", "items": [], "status": "inProgress"}},
        }
    )
    notify("turn/started", threadId="thr_1", turn={"id": "turn_1"})
    if MODE == "hang":
        time.sleep(600)
    if MODE == "leak":
        sys.stderr.write(
            "login: " + open(os.environ["CODEX_HOME"] + "/auth.json").read()
        )
        sys.stderr.flush()
    if MODE == "refresh":
        with open(os.path.join(os.environ["CODEX_HOME"], "auth.json"), "w") as stream:
            stream.write('{"tokens": {"refresh_token": "REFRESHED"}}')
    # A server request the client must refuse: nothing may act or ask.
    send(
        {
            "id": 900,
            "method": "item/commandExecution/requestApproval",
            "params": {"threadId": "thr_1", "turnId": "turn_1", "itemId": "x"},
        }
    )
    refusal = receive()
    # Before the answer: the client ends this process once it has one.
    record(refusal=refusal)
    notify(
        "item/completed",
        threadId="thr_1",
        turnId="turn_1",
        item={"type": "userMessage", "id": "u1", "content": []},
    )
    notify(
        "item/completed",
        threadId="thr_1",
        turnId="turn_1",
        item={
            "type": "agentMessage",
            "id": "m1",
            "text": "thinking",
            "phase": "commentary",
        },
    )
    if MODE == "fail_turn":
        notify(
            "error",
            threadId="thr_1",
            turnId="turn_1",
            willRetry=False,
            error={"message": "usage limit reached"},
        )
        notify(
            "turn/completed",
            threadId="thr_1",
            turn={
                "id": "turn_1",
                "status": "failed",
                "items": [],
                "error": {"message": "usage limit reached"},
            },
        )
        receive()
        return 0
    schema = turn["params"].get("outputSchema")
    prompt = turn["params"]["input"][0]["text"]
    if MODE == "identity" or prompt.startswith("IDENTITY"):
        login = os.path.join(os.environ["CODEX_HOME"], "auth.json")
        answer = json.dumps(
            {
                "uid": os.getuid(),
                "home": os.environ.get("HOME"),
                "codex_home": os.environ.get("CODEX_HOME"),
                "login_mode": oct(os.stat(login).st_mode & 0o777),
                "login_owner": os.stat(login).st_uid,
                "cwd": os.getcwd(),
            }
        )
    elif prompt.startswith("LOGIN"):
        import base64

        with open(os.path.join(os.environ["CODEX_HOME"], "auth.json")) as stream:
            tokens = json.load(stream)["tokens"]
        payload = tokens["access_token"].split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        answer = json.dumps(
            {
                "generation": claims["generation"],
                "withheld": tokens["refresh_token"] == "withheld-by-rsi-harness",
            }
        )
    elif schema is not None:
        answer = json.dumps({"verdict": "clear"})
    else:
        answer = "plain text answer"
    notify(
        "item/completed",
        threadId="thr_1",
        turnId="turn_1",
        item={
            "type": "agentMessage",
            "id": "m2",
            "text": answer,
            "phase": None if MODE == "plain" else "final_answer",
        },
    )
    if MODE == "after_final":
        notify(
            "item/completed",
            threadId="thr_1",
            turnId="turn_1",
            item={
                "type": "agentMessage",
                "id": "m3",
                "text": "a note after the answer",
                "phase": "commentary",
            },
        )
    notify(
        "turn/completed",
        threadId="thr_1",
        turn={"id": "turn_1", "status": "completed", "items": [], "error": None},
    )
    # The client closes stdin when it has its answer.
    while sys.stdin.readline():
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
