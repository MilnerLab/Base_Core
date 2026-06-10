from __future__ import annotations

import json
from mailbox import Message
import subprocess
import threading
import uuid
from dataclasses import dataclass
from typing import Iterator, Optional

from base_core.framework.subprocess.messages import Kind, MessageRegistry



@dataclass
class _Pending:
    event: threading.Event
    response: Optional[dict] = None


class JsonlSubprocessEndpoint:
    """
    Bidirectional JSONL endpoint for a subprocess (main-process side).

    Public surface speaks typed Message objects; a MessageRegistry handles
    (de)serialization. The byte-level transport -- process lifecycle, the
    write lock, request/reply correlation, and stdout line reading -- is kept
    as private internals so message typing and transport stay decoupled.

    Wire envelope (added/stripped here, not part of a Message's payload):
      - command: {"kind":"command","name":str,"id"?:str,"payload":{...}}
      - reply:   {"kind":"reply","reply_to":str,"ok":bool,"payload"?:{...}}
      - event:   {"kind":"event","name":str,"source"?:str,"payload":{...}}

    Only small control messages travel here; bulk data goes through shared
    memory or another data plane.
    """

    def __init__(
        self,
        argv: list[str],
        registry: MessageRegistry,
        *,
        env: Optional[dict[str, str]] = None,
        merge_stderr_to_stdout: bool = False,
    ) -> None:
        self._argv = argv
        self._registry = registry
        self._env = env
        self._merge_stderr = merge_stderr_to_stdout

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

        with self._pending_lock:
            for p in self._pending.values():
                p.response = {"kind": "reply", "ok": False, "error": "endpoint stopped"}
                p.event.set()
            self._pending.clear()

        self._proc = None

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

    def send(self, message: Message) -> None:
        """Fire-and-forget send of a command or event message."""
        self._send_envelope(self._registry.envelope_for(message))

    def request(self, message: Message, *, timeout_s: float = 2.0) -> Optional[Message]:
        """
        Send a command and wait for its reply, returning the decoded reply
        Message -- or None if the reply has no registered message type (e.g.
        a bare ok/error reply). Use raw_request() when you need ok/error.

        Requires that produce() is being consumed on another thread, and must
        not be called from within that same thread.
        """
        if message.KIND != Kind.COMMAND:
            raise ValueError("request() requires a command message.")
        envelope = self._registry.envelope_for(message)
        return self._registry.decode(self._request_envelope(envelope, timeout_s))

    def raw_request(self, message: Message, *, timeout_s: float = 2.0) -> dict:
        """Like request() but returns the raw reply envelope (ok/error/payload)."""
        if message.KIND != Kind.COMMAND:
            raise ValueError("raw_request() requires a command message.")
        envelope = self._registry.envelope_for(message)
        return self._request_envelope(envelope, timeout_s)

    # ---------- receiving ----------

    def produce(self, stop: threading.Event) -> Iterator[Message]:
        """
        Blocking generator yielding decoded Message objects from the subprocess.

        Replies are routed to pending requests and not yielded. Unknown or
        malformed lines are skipped.
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
                envelope = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(envelope, dict):
                continue

            # Route replies to waiting requests; do not yield them.
            reply_to = envelope.get("reply_to")
            if isinstance(reply_to, str):
                with self._pending_lock:
                    pending = self._pending.get(reply_to)
                if pending is not None:
                    pending.response = envelope
                    pending.event.set()
                    continue

            msg = self._registry.decode(envelope)
            if msg is not None:
                yield msg

        # stdout closed: fail any outstanding requests.
        with self._pending_lock:
            for p in self._pending.values():
                p.response = {"kind": "reply", "ok": False, "error": "stdout closed"}
                p.event.set()
            self._pending.clear()

    # ---------- internals: raw transport ----------

    def _send_envelope(self, envelope: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("Subprocess not running. Call start() first.")
        line = json.dumps(envelope, separators=(",", ":")) + "\n"
        with self._write_lock:
            proc.stdin.write(line)
            proc.stdin.flush()

    def _request_envelope(self, envelope: dict, timeout_s: float) -> dict:
        req_id = uuid.uuid4().hex
        envelope = dict(envelope)
        envelope["id"] = req_id

        pending = _Pending(event=threading.Event())
        with self._pending_lock:
            self._pending[req_id] = pending

        try:
            self._send_envelope(envelope)
            if not pending.event.wait(timeout=timeout_s):
                raise TimeoutError(f"Timed out waiting for reply_to={req_id}")
            assert pending.response is not None
            return pending.response
        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)