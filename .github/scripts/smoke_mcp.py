#!/usr/bin/env python3
"""Smoke test for an installed IHMT server: start it and talk MCP to it over stdio.

    python .github/scripts/smoke_mcp.py ihmt-memory
    python .github/scripts/smoke_mcp.py uvx --from dist/ihmt_memory-1.2.0-py3-none-any.whl ihmt-memory

Runs the command twice in throwaway folders, never against a real memory:
once with IHMT_HOME set, and once without it, to check the ``~/.ihmt`` default.
Each run lists the tools, saves a note and finds it again. Standard library only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

TIMEOUT = 120  # seconds; uvx may download the MCP SDK on the first run


class Server:
    def __init__(self, command: List[str], env: Dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        )
        self._watchdog = threading.Timer(TIMEOUT, self.proc.kill)
        self._watchdog.start()
        self._next_id = 0

    def send(self, message: Dict[str, Any]) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._next_id += 1
        self.send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        while True:  # skip notifications until our response arrives
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"FAIL: the server exited during {method} (code {self.proc.wait()})")
            reply = json.loads(line)
            if reply.get("id") == self._next_id:
                if "error" in reply:
                    raise SystemExit(f"FAIL: {method} -> {reply['error']}")
                return reply["result"]

    def call(self, tool: str, **arguments: Any) -> str:
        result = self.request("tools/call", {"name": tool, "arguments": arguments})
        text = "".join(part.get("text", "") for part in result.get("content", []))
        if result.get("isError"):
            raise SystemExit(f"FAIL: {tool} -> {text}")
        return text

    def close(self) -> None:
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._watchdog.cancel()


def check(command: List[str], env: Dict[str, str], store: Path) -> None:
    server = Server(command, env)
    try:
        server.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "ihmt-smoke-test", "version": "1"},
        })
        server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools = {tool["name"] for tool in server.request("tools/list", {})["tools"]}
        missing = {"search_memory", "save_memory", "mark_outdated"} - tools
        if missing:
            raise SystemExit(f"FAIL: tools missing: {sorted(missing)}")

        server.call("save_memory", content="2026-01-01: the smoke test keeps a note about the zebra lighthouse.")
        found = server.call("search_memory", query="smoke test zebra lighthouse note")
        if "zebra lighthouse" not in found:
            raise SystemExit(f"FAIL: search_memory did not find the saved note:\n{found}")
    finally:
        server.close()

    if not (store / "ihmt_memory" / "root.json").exists():
        raise SystemExit(f"FAIL: no memory store at {store}")
    print(f"ok: {len(tools)} tools, saved and found a note in {store}")


def main(command: List[str]) -> int:
    if not command:
        raise SystemExit(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        explicit = Path(tmp) / "explicit"
        check(command, {**os.environ, "IHMT_HOME": str(explicit)}, explicit)

        home = Path(tmp) / "home"
        home.mkdir()
        env = {key: value for key, value in os.environ.items() if key != "IHMT_HOME"}
        env.update(HOME=str(home), USERPROFILE=str(home))  # Path.home() reads USERPROFILE on Windows
        check(command, env, home / ".ihmt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
