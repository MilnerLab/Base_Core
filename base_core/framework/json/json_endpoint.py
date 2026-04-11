from __future__ import annotations

import json
import subprocess
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional


@dataclass
class _Pending:
    event: threading.Event
    response: Optional[dict] = None


class JsonlSubprocessEndpoint:
    """
    Generic, bidirectional JSONL endpoint for a subprocess.

    Protocol:
      - command: {"kind": "command", "name": str, "id"?: str, "payload"?: {...}}
      - reply:   {"kind": "reply", "reply_to": str, "ok": bool, "payload"?: {...}}
      - event:   {"kind": "event", "name": str, "source"?: str, "payload"?: {...}}

    This transport should only carry small messages and handles / metadata.
    Large arrays belong in shared memory or another data plane.
    """

    _VALID_KINDS = {"command", "reply", "event"}

    def __init__(
        self,
        argv: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        merge_stderr_to_stdout: bool = False,
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
            bufsize=1,
            env=self._env,
        )

        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Failed to open subprocess pipes.")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None

        with self._pending_lock:
            for p in self._pending.values():
                p.response = {"kind": "reply", "ok": False, "error": "endpoint stopped"}
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
        self._validate_outgoing(message)

        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("Subprocess not running. Call start() first.")

        line = json.dumps(message, separators=(",", ":")) + "\n"
        with self._write_lock:
            proc.stdin.write(line)
            proc.stdin.flush()

    def send_command(
        self,
        name: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        target: Optional[str] = None,
    ) -> None:
        msg: Dict[str, Any] = {
            "kind": "command",
            "name": name,
            "payload": dict(payload or {}),
        }
        if target is not None:
            msg["target"] = target
        self.send(msg)

    def request(self, message: Dict[str, Any], *, timeout_s: float = 2.0) -> dict:
        """
        Sends a message with an id and waits for a response with reply_to == id.

        IMPORTANT:
        - This requires that `produce(stop_event)` is currently being consumed,
          otherwise nobody reads stdout.
        - Do NOT call request() from inside the same worker thread that is running `produce`.
        """
        msg = dict(message)
        msg.setdefault("kind", "command")
        if "name" not in msg:
            raise ValueError("request message must include 'name'")

        req_id = uuid.uuid4().hex
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

    def request_command(
        self,
        name: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        target: Optional[str] = None,
        timeout_s: float = 2.0,
    ) -> dict:
        msg: Dict[str, Any] = {
            "kind": "command",
            "name": name,
            "payload": dict(payload or {}),
        }
        if target is not None:
            msg["target"] = target
        return self.request(msg, timeout_s=timeout_s)

    # ---------- receiving ----------

    def produce(self, stop: threading.Event, *, forward_responses: bool = False) -> Iterator[dict]:
        """
        Blocking generator that reads stdout and yields validated dict messages.

        - Reply messages are routed to pending requests and by default NOT yielded.
        - Event messages are yielded normally.
        - Any malformed / non-protocol messages are ignored.
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

            if not self._is_valid_incoming(msg):
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

        with self._pending_lock:
            for p in self._pending.values():
                p.response = {"kind": "reply", "ok": False, "error": "stdout closed"}
                p.event.set()
            self._pending.clear()

    # ---------- validation ----------

    def _validate_outgoing(self, message: Mapping[str, Any]) -> None:
        kind = message.get("kind")
        if kind not in self._VALID_KINDS:
            raise ValueError("message must include kind in {'command', 'reply', 'event'}")

        if kind in {"command", "event"} and not isinstance(message.get("name"), str):
            raise ValueError("command/event message must include string field 'name'")

        payload = message.get("payload")
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("payload must be a dict when present")

        if kind == "reply" and not isinstance(message.get(self._reply_to_field), str):
            # Allow replies without reply_to only in the internal stop path, not on send().
            raise ValueError("reply message must include string field 'reply_to'")

    def _is_valid_incoming(self, message: object) -> bool:
        if not isinstance(message, dict):
            return False

        kind = message.get("kind")
        if kind not in self._VALID_KINDS:
            return False

        if kind in {"command", "event"} and not isinstance(message.get("name"), str):
            return False

        payload = message.get("payload")
        if payload is not None and not isinstance(payload, dict):
            return False

        if kind == "reply" and not isinstance(message.get(self._reply_to_field), str):
            return False

        return True
