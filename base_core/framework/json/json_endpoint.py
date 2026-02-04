# jsonl_endpoint.py
from __future__ import annotations

import json
import subprocess
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Iterator, Optional

# uses your TaskRunner interface (run/stream) :contentReference[oaicite:0]{index=0}


@dataclass
class _Pending:
    event: threading.Event
    response: Optional[dict] = None


class JsonlSubprocessEndpoint:
    """
    Generic, bidirectional JSONL endpoint for a subprocess.

    Transport:
      - Main -> Child: write JSON per line to child's stdin
      - Child -> Main: read JSON per line from child's stdout

    Request/Response convention (minimal & generic):
      - Request includes: {"id": "<uuid>"}
      - Response includes: {"reply_to": "<uuid>"}  (and anything else)
    """

    def __init__(
        self,
        argv: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        merge_stderr_to_stdout: bool = True,
        request_id_field: str = "id",
        reply_to_field: str = "reply_to",
    ) -> None:
        self._argv = argv
        self._env = env
        self._merge_stderr = merge_stderr_to_stdout

        self._request_id_field = request_id_field
        self._reply_to_field = reply_to_field

        self._proc: Optional[subprocess.Popen[str]] = None

        self._write_lock = threading.Lock()

        self._pending_lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("Subprocess already running.")

        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(subprocess.STDOUT if self._merge_stderr else subprocess.PIPE),
            text=True,
            bufsize=1,  # host-side line buffering
            env=self._env,
        )

        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Failed to open subprocess pipes.")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None

        # Unblock all waiting requests
        with self._pending_lock:
            for p in self._pending.values():
                p.response = {"ok": False, "error": "endpoint stopped"}
                p.event.set()
            self._pending.clear()

        if proc is None:
            return

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---------- sending ----------

    def send(self, message: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("Subprocess not running. Call start() first.")

        line = json.dumps(message, separators=(",", ":")) + "\n"
        with self._write_lock:
            proc.stdin.write(line)
            proc.stdin.flush()

    def request(self, message: Dict[str, Any], *, timeout_s: float = 2.0) -> dict:
        """
        Sends a message with an id and waits for a response with reply_to == id.

        IMPORTANT:
        - This requires that `produce(stop_event)` is currently being consumed
          (e.g., by TaskRunner.stream), otherwise nobody reads stdout.
        - Do NOT call request() from inside the same worker thread that is running `produce`.
        """
        req_id = uuid.uuid4().hex

        msg = dict(message)
        msg[self._request_id_field] = req_id

        pending = _Pending(event=threading.Event())
        with self._pending_lock:
            self._pending[req_id] = pending

        try:
            self.send(msg)
            if not pending.event.wait(timeout=timeout_s):
                raise TimeoutError(f"Timed out waiting for reply_to={req_id}")
            assert pending.response is not None
            return pending.response
        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)

    # ---------- receiving (to be run in TaskRunner.stream) ----------

    def produce(self, stop: threading.Event, *, forward_responses: bool = False) -> Iterator[dict]:
        """
        Blocking generator that reads stdout and yields dict messages.

        - Response messages are routed to pending requests and (by default) NOT yielded.
        - Non-response messages are yielded normally.

        If forward_responses=True, response messages are also yielded after routing.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("Subprocess not running. Call start() first.")

        for line in proc.stdout:
            if stop.is_set():
                break

            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(msg, dict):
                continue

            reply_to = msg.get(self._reply_to_field)
            if isinstance(reply_to, str):
                with self._pending_lock:
                    pending = self._pending.get(reply_to)
                if pending is not None:
                    pending.response = msg
                    pending.event.set()
                    if not forward_responses:
                        continue

            yield msg

        # EOF or stop: wake pending requests so nothing hangs forever
        with self._pending_lock:
            for p in self._pending.values():
                p.response = {"ok": False, "error": "stdout closed"}
                p.event.set()
            self._pending.clear()
