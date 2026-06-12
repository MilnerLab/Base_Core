from __future__ import annotations

import json
import sys
import threading
from typing import Any, Callable, Optional, Type

from base_core.framework.subprocess.messages import (
    ErrorMessage,
    Kind,
    Message,
    MessageRegistry,
    OKMessage,
)


CommandHandler = Callable[["Message"], None]


class JsonlStdioAppBase:
    """
    JSONL stdio app base for subprocesses (the child side).

    Reads commands from stdin on a background thread, decodes them into typed
    Message objects via a MessageRegistry, dispatches each to a registered
    handler, and emits typed events / replies to stdout.

    Replies are named envelopes so endpoint.request() can decode them:
      {"kind":"reply","name":"ok","reply_to":"<id>","payload":{}}
      {"kind":"reply","name":"error","reply_to":"<id>","payload":{"error":"…"}}

    Subclasses register handlers via self.on(MessageClass, handler) and
    implement main() for their device loop.
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
        if self._source is not None:
            envelope["source"] = self._source
        self._write(envelope)

    def reply(self, request_id: Optional[str], reply_msg: Message) -> None:
        """Send any typed reply message correlated to request_id."""
        envelope = self._registry.envelope_for(reply_msg)
        envelope["kind"] = Kind.REPLY
        if isinstance(request_id, str):
            envelope["reply_to"] = request_id
        if self._source is not None:
            envelope["source"] = self._source
        self._write(envelope)

    def reply_ok(self, request_id: Optional[str]) -> None:
        """Send a typed OKMessage reply."""
        self.reply(request_id, OKMessage())

    def reply_error(self, request_id: Optional[str], error: str) -> None:
        """Send a typed ErrorMessage reply."""
        self.reply(request_id, ErrorMessage(error=error))

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
        self._stop.set()

    def _dispatch(self, envelope: dict) -> None:
        if envelope.get("kind") != Kind.COMMAND:
            return
        raw_id = envelope.get("id")
        request_id: Optional[str] = raw_id if isinstance(raw_id, str) else None
        message = self._registry.decode(envelope)
        if message is None:
            if request_id is not None:
                self.reply_error(request_id, f"Unknown command: {envelope.get('name')}")
            return
        self._dispatch_decoded(message)

    def _dispatch_decoded(self, msg: Message) -> None:
        """Dispatch an already-decoded message to the registered handler."""
        handler = self._handlers.get(type(msg))
        if handler is None:
            if msg.request_id is not None:
                self.reply_error(msg.request_id, f"No handler for {type(msg).__name__}")
            return
        try:
            handler(msg)
        except Exception as exc:
            self.reply_error(msg.request_id, str(exc))
