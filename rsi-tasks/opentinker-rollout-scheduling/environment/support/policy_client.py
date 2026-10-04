"""Bounded asynchronous framing; callbacks cannot obtain trajectory contents."""
import asyncio
import json
import os
from pathlib import Path
from policy_diagnostics import repair_detail, sanitize_detail


class ContractError(Exception):
    def __init__(self, code, field="policy", condition="policy contract violated", detail=None):
        self.code, self.field, self.condition = code, field, condition
        self.detail = sanitize_detail(detail)
        super().__init__(code)

    def __str__(self):
        # Ray/dev tracebacks retain the same bounded repair detail as formal IPC.
        return json.dumps({"code": self.code, "path": "/workspace/candidate/policy.py",
                           "field": self.field, "condition": self.condition,
                           **repair_detail(self.detail)})


class PolicyClient:
    def __init__(self, source):
        self.source = source
        self.process = None

    async def start(self):
        worker = str(Path(__file__).with_name("policy_worker.py"))
        self.process = await asyncio.create_subprocess_exec(
            "/opt/venv/bin/python", "-B", "-I", worker,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, close_fds=True,
            env={"PATH": "/opt/venv/bin:/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            cwd="/", limit=1048577)
        await self._send({"source": self.source})
        response = await self._read()
        if response != {"ok": True}:
            code = "policy_import" if response.get("error") == "policy_import" else "policy_protocol"
            raise ContractError(code, detail=response.get("detail"))

    async def _send(self, value):
        self.process.stdin.write(json.dumps(value, allow_nan=False).encode() + b"\n")
        try:
            await asyncio.wait_for(self.process.stdin.drain(), 5)
        except (BrokenPipeError, ConnectionResetError):
            raise ContractError("policy_crash") from None

    async def _read(self):
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), 5)
            if not line:
                raise ContractError("policy_crash")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except asyncio.TimeoutError:
            raise ContractError("policy_timeout") from None
        except (ValueError, RecursionError):
            raise ContractError("policy_protocol") from None

    async def call(self, method, value):
        await self._send({"method": method, "value": value})
        result = await self._read()
        if "error" in result:
            # No arbitrary candidate-authored string becomes feedback.
            code = result["error"]
            if not isinstance(code, str) or code not in {"policy_import", "policy_callback", "policy_serialization", "policy_output_size"}:
                code = "policy_protocol"
            raise ContractError(code, detail=result.get("detail"))
        if set(result) != {"result"}:
            raise ContractError("policy_protocol")
        return result["result"]

    async def close(self):
        if self.process is not None:
            if self.process.returncode is None:
                self.process.kill()
            await self.process.wait()
