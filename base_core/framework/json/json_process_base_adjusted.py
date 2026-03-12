from __future__ import annotations

import json
import sys
import threading
from typing import Any, Dict, Mapping, Optional


class JsonlStdioAppBase:
    """
    Generic JSONL stdio app base for subprocesses.

    Protocol:
      - command: {"kind": "command", "name": str, "id"?: str, "payload"?: {...}}
      - reply:   {"kind": "reply", "reply_to": str, "ok": bool, "payload"?: {...}}
      - event:   {"kind": "event", "name": str, "source"?: str, "payload"?: {...}}

    Notes:
      - Commands are read from stdin in a background thread.
      - Events / replies are written to stdout as JSONL.
      - Large payloads should NOT be sent here; only send handles / metadata.
    """

    def __init__(
        self,
        *,
        source: Optional[str] = None,
        request_id_field: str = "id",
        reply_to_field: str = "reply_to",
    ) -> None:
        self._stop = threading.Event()
        self._stdin_thread: Optional[threading.Thread] = None
        self._request_id_field = request_id_field
        self._reply_to_field = reply_to_field
        self._source = source
        self._write_lock = threading.Lock()

    # ----- output helpers -----

    def emit(self, message: Dict[str, Any]) -> None:
        with self._write_lock:
            sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    def emit_event(
        self,
        name: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        source: Optional[str] = None,
    ) -> None:
        msg: Dict[str, Any] = {
            "kind": "event",
            "name": name,
            "payload": dict(payload or {}),
        }
        event_source = source if source is not None else self._source
        if event_source is not None:
            msg["source"] = event_source
        self.emit(msg)

    def reply(self, request_msg: Mapping[str, Any], message: Dict[str, Any]) -> None:
        req_id = request_msg.get(self._request_id_field)
        out = dict(message)
        out.setdefault("kind", "reply")
        if isinstance(req_id, str):
            out[self._reply_to_field] = req_id
        self.emit(out)

    def reply_ok(
        self,
        request_msg: Mapping[str, Any],
        payload: Optional[Mapping[str, Any]] = None,
        *,
        source: Optional[str] = None,
    ) -> None:
        msg: Dict[str, Any] = {
            "kind": "reply",
            "ok": True,
            "payload": dict(payload or {}),
        }
        reply_source = source if source is not None else self._source
        if reply_source is not None:
            msg["source"] = reply_source
        self.reply(request_msg, msg)

    def reply_error(
        self,
        request_msg: Mapping[str, Any],
        error: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        source: Optional[str] = None,
    ) -> None:
        msg: Dict[str, Any] = {
            "kind": "reply",
            "ok": False,
            "error": error,
            "payload": dict(payload or {}),
        }
        reply_source = source if source is not None else self._source
        if reply_source is not None:
            msg["source"] = reply_source
        self.reply(request_msg, msg)

    # ----- input handling -----

    def on_command(
        self,
        name: str,
        payload: Dict[str, Any],
        message: Dict[str, Any],
    ) -> None:
        """
        Override in subclasses.

        `message` is the full command dict.
        Use `reply_ok(...)` / `reply_error(...)` for request-response commands.
        """
        raise NotImplementedError

    def on_message(self, message: Dict[str, Any]) -> None:
        """
        Default dispatcher for incoming JSON messages.
        Override only if you really want custom routing.
        """
        kind = message.get("kind")
        name = message.get("name")
        payload = message.get("payload", {})

        if kind != "command" or not isinstance(name, str):
            return
        if not isinstance(payload, dict):
            self.reply_error(message, "payload must be an object")
            return

        try:
            self.on_command(name, payload, message)
        except Exception as exc:  # noqa: BLE001
            self.reply_error(message, str(exc))

    # ----- main work -----

    def main(self, stop_event: threading.Event) -> None:
        """
        Override in subclasses. Do your device loop here.
        Check stop_event.is_set() periodically.
        """
        raise NotImplementedError

    # ----- lifecycle -----

    def run(self) -> None:
        self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._stdin_thread.start()

        try:
            self.main(self._stop)
        finally:
            self._stop.set()

    def stop(self) -> None:
        self._stop.set()

    # ----- internals -----

    def _stdin_loop(self) -> None:
        for line in sys.stdin:
            if self._stop.is_set():
                break

            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(msg, dict):
                self.on_message(msg)

        # Parent closed stdin -> stop
        self._stop.set()
