from __future__ import annotations

import json
import sys
import threading
from typing import Any, Callable, Optional, Type

from messages import Message, MessageRegistry, Kind


# A handler receives the decoded command message plus the raw envelope (so it
# can read the request id for replies). It returns an optional reply payload.
CommandHandler = Callable[["Message", dict], None]


class JsonlStdioAppBase:
    """
    JSONL stdio app base for subprocesses (the child side).

    Mirror of JsonlSubprocessEndpoint:
      - reads commands from stdin on a background thread
      - decodes them into typed Message objects via a MessageRegistry
      - dispatches each to a handler registered by message class
      - emits typed events / replies to stdout

    Subclasses register handlers in __init__ via self.on(MessageClass, handler)
    and implement main() for their device loop. No string command names, no
    PROTOCOL constants.
    """

    def __init__(
        self,
        registry: MessageRegistry,
        *,
        source: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._source = source
        self._stop = threading.Event()
        self._stdin_thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._handlers: dict[Type[Message], CommandHandler] = {}

    # ----- handler registration -----

    def on(self, message_cls: Type[Message], handler: CommandHandler) -> None:
        """Route incoming commands of `message_cls` to `handler`."""
        self._handlers[message_cls] = handler

    # ----- output -----

    def emit(self, message: Message) -> None:
        """Emit a typed event message to stdout."""
        envelope = self._registry.envelope_for(message)
        if self._source is not None and "source" not in envelope:
            envelope["source"] = self._source
        self._write(envelope)

    def reply_ok(self, request_envelope: dict, payload: Optional[dict] = None) -> None:
        self._reply(request_envelope, {"ok": True, "payload": dict(payload or {})})

    def reply_error(self, request_envelope: dict, error: str) -> None:
        self._reply(request_envelope, {"ok": False, "error": error})

    # ----- main work (override) -----

    def main(self, stop_event: threading.Event) -> None:
        """Override: device loop. Check stop_event.is_set() periodically."""
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

    def _write(self, envelope: dict) -> None:
        line = json.dumps(envelope, separators=(",", ":")) + "\n"
        with self._write_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def _reply(self, request_envelope: dict, body: dict) -> None:
        out = {"kind": Kind.REPLY, **body}
        req_id = request_envelope.get("id")
        if isinstance(req_id, str):
            out["reply_to"] = req_id
        if self._source is not None:
            out["source"] = self._source
        self._write(out)

    def _stdin_loop(self) -> None:
        for line in sys.stdin:
            if self._stop.is_set():
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
            self._dispatch(envelope)
        # Parent closed stdin -> stop.
        self._stop.set()

    def _dispatch(self, envelope: dict) -> None:
        if envelope.get("kind") != Kind.COMMAND:
            return
        message = self._registry.decode(envelope)
        if message is None:
            # Unknown/unregistered command. Reply with an error if it had an id.
            if isinstance(envelope.get("id"), str):
                self.reply_error(envelope, f"Unknown command: {envelope.get('name')}")
            return
        handler = self._handlers.get(type(message))
        if handler is None:
            self.reply_error(envelope, f"No handler for {type(message).__name__}")
            return
        try:
            handler(message, envelope)
        except Exception as exc:  # noqa: BLE001
            self.reply_error(envelope, str(exc))