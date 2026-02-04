# jsonl_stdio_app_base.py
from __future__ import annotations

import json
import sys
import threading
from typing import Any, Dict, Optional


class JsonlStdioAppBase:
    """
    Generic JSONL stdio app base for subprocesses.

    - Reads JSONL from stdin in a background thread -> on_message(msg)
    - Writes JSONL to stdout via emit(msg)

    Request/Response convention (matches the endpoint default):
      - Request may include: {"id": "..."}
      - Response should include: {"reply_to": "..."} (use reply())
    """

    def __init__(self, *, request_id_field: str = "id", reply_to_field: str = "reply_to") -> None:
        self._stop = threading.Event()
        self._stdin_thread: Optional[threading.Thread] = None
        self._request_id_field = request_id_field
        self._reply_to_field = reply_to_field

    # ----- output -----

    def emit(self, message: Dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()

    def reply(self, request_msg: Dict[str, Any], message: Dict[str, Any]) -> None:
        req_id = request_msg.get(self._request_id_field)
        if isinstance(req_id, str):
            out = dict(message)
            out[self._reply_to_field] = req_id
            self.emit(out)
        else:
            self.emit(message)

    # ----- input handling -----

    def on_message(self, message: Dict[str, Any]) -> None:
        """
        Override in subclasses.
        """
        pass

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
